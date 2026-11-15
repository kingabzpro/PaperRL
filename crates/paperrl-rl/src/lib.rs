//! Trading environment for RL agents.
//!
//! The environment shares the batch simulator's fill accounting (through
//! `paperrl_sim::Portfolio`): an action is a target equity weight applied at
//! the current close and filled at the next open with slippage and fees. The
//! reward follows the PaperRL research spec:
//! `log(V_t+1 / V_t) - lambda_d * drawdown_increase - lambda_u * turnover`.

use paperrl_core::Bar;
use paperrl_sim::Portfolio;

mod ppo;
pub use ppo::{
    ACTION_WEIGHTS, ModelMeta, PpoConfig, PpoNet, PpoStrategy, TrainedPpo, compute_gae, train_ppo,
};

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct EnvConfig {
    /// Close-to-close returns carried in the observation.
    pub window: usize,
    /// Penalty per unit of drawdown increase (lambda_d).
    pub dd_penalty: f64,
    /// Penalty per unit of traded notional over equity (lambda_u).
    pub turnover_penalty: f64,
    pub initial_cash: f64,
    pub fee_bps: f64,
    pub slippage_bps: f64,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            window: 10,
            dd_penalty: 5.0,
            turnover_penalty: 1.0,
            initial_cash: 100_000.0,
            fee_bps: 1.0,
            slippage_bps: 5.0,
        }
    }
}

/// One environment step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Step {
    pub reward: f32,
    pub done: bool,
}

/// Builds the observation vector shared by the training environment and live
/// strategies, so both always see the same layout: `returns` oldest first,
/// then position weight, then drawdown.
pub fn obs_vector(returns: &[f64], position_weight: f64, drawdown: f64) -> Vec<f32> {
    let mut obs = Vec::with_capacity(returns.len() + 2);
    obs.extend(returns.iter().map(|r| *r as f32));
    obs.push(position_weight as f32);
    obs.push(drawdown as f32);
    obs
}

/// Single-symbol trading environment over a stored bar slice.
pub struct TradingEnv<'a> {
    bars: &'a [Bar],
    cfg: EnvConfig,
    pf: Portfolio,
    /// Index of the current close bar.
    i: usize,
    peak: f64,
    prev_dd: f64,
    prev_equity: f64,
    prev_turnover: f64,
}

impl<'a> TradingEnv<'a> {
    pub fn new(bars: &'a [Bar], cfg: EnvConfig) -> Self {
        assert!(
            bars.len() > cfg.window + 1,
            "need at least one steppable bar beyond the observation window"
        );
        Self {
            bars,
            cfg,
            pf: Portfolio::new(cfg.initial_cash, cfg.fee_bps, cfg.slippage_bps),
            i: cfg.window,
            peak: cfg.initial_cash,
            prev_dd: 0.0,
            prev_equity: cfg.initial_cash,
            prev_turnover: 0.0,
        }
    }

    /// Length of the observation vector: `window` returns + weight + drawdown.
    pub fn obs_len(&self) -> usize {
        self.cfg.window + 2
    }

    /// Features as of the current close: `window` close-to-close returns, then
    /// the position weight and the current drawdown (0 when at a new peak).
    pub fn obs(&self) -> Vec<f32> {
        let i = self.i;
        let returns: Vec<f64> = ((i - self.cfg.window + 1)..=i)
            .map(|k| self.bars[k].close / self.bars[k - 1].close - 1.0)
            .collect();
        let eq = self.pf.equity(self.bars[i].close);
        obs_vector(
            &returns,
            self.pf.position_weight(self.bars[i].close),
            (1.0 - eq / self.peak).max(0.0),
        )
    }

    /// Applies `action` (target weight in [0, 1]) at the current close,
    /// advances one bar, fills at the next open, marks at its close, and
    /// returns the reward. `done` is true on the last bar.
    pub fn step(&mut self, action: f64) -> Step {
        self.pf.on_close(action);
        self.i += 1;
        let bar = &self.bars[self.i];
        self.pf.fill_at_open(bar);
        let equity = self.pf.equity(bar.close);
        let dd = (1.0 - equity / self.peak).max(0.0);
        let dd_increase = (dd - self.prev_dd).max(0.0);
        let turnover = (self.pf.turnover - self.prev_turnover) / self.prev_equity.max(1e-12);
        let reward = (equity / self.prev_equity).ln()
            - self.cfg.dd_penalty * dd_increase
            - self.cfg.turnover_penalty * turnover;
        self.peak = self.peak.max(equity);
        self.prev_dd = dd;
        self.prev_equity = equity;
        self.prev_turnover = self.pf.turnover;
        Step {
            reward: reward as f32,
            done: self.i == self.bars.len() - 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn bars(opens: &[f64], closes: &[f64]) -> Vec<Bar> {
        opens
            .iter()
            .zip(closes)
            .enumerate()
            .map(|(i, (&o, &c))| Bar {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1 + i as u32, 0, 0, 0).unwrap(),
                open: o,
                high: o.max(c),
                low: o.min(c),
                close: c,
                volume: 1000.0,
            })
            .collect()
    }

    fn cfg(window: usize, dd: f64, turnover: f64) -> EnvConfig {
        EnvConfig {
            window,
            dd_penalty: dd,
            turnover_penalty: turnover,
            initial_cash: 10_000.0,
            fee_bps: 0.0,
            slippage_bps: 0.0,
        }
    }

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "left {a} vs right {b}");
    }

    #[test]
    fn obs_shape_and_initial_values() {
        let bs = bars(&[10.0, 10.0, 10.0], &[10.0, 12.0, 14.0]);
        let env = TradingEnv::new(&bs, cfg(1, 0.0, 0.0));
        assert_eq!(env.obs_len(), 3);
        // One return (12/10 - 1), flat position, zero drawdown.
        let obs = env.obs();
        approx(obs[0], 0.2);
        approx(obs[1], 0.0);
        approx(obs[2], 0.0);
    }

    #[test]
    fn flat_action_gives_zero_reward() {
        let bs = bars(&[10.0, 10.0, 10.0], &[10.0, 12.0, 14.0]);
        let mut env = TradingEnv::new(&bs, cfg(1, 5.0, 1.0));
        let s = env.step(0.0);
        approx(s.reward, 0.0);
        assert!(s.done);
    }

    #[test]
    fn buy_reward_is_log_growth() {
        let bs = bars(&[10.0, 10.0, 10.0], &[10.0, 12.0, 14.0]);
        let mut env = TradingEnv::new(&bs, cfg(1, 0.0, 0.0));
        // Fill at bar 2 open (10.0): 1000 shares, marked at close 14.0.
        let s = env.step(1.0);
        approx(s.reward, 1.4f32.ln());
        assert!(s.done);
    }

    #[test]
    fn turnover_penalty_applies_once_per_fill() {
        let bs = bars(&[10.0, 10.0, 10.0], &[10.0, 12.0, 14.0]);
        let mut env = TradingEnv::new(&bs, cfg(1, 0.0, 1.0));
        // Traded notional 10000 over prior equity 10000: penalty 1.0.
        let s = env.step(1.0);
        approx(s.reward, 1.4f32.ln() - 1.0);
    }

    #[test]
    fn drawdown_penalty_applies_on_increase() {
        let bs = bars(&[10.0, 10.0, 10.0], &[10.0, 12.0, 6.0]);
        let mut env = TradingEnv::new(&bs, cfg(1, 1.0, 0.0));
        // Equity falls to 6000: dd 0.4, reward ln(0.6) - 0.4.
        let s = env.step(1.0);
        approx(s.reward, 0.6f32.ln() - 0.4);
    }

    #[test]
    fn obs_reflects_position_after_fill() {
        let bs = bars(&[10.0, 10.0, 10.0], &[10.0, 10.0, 14.0]);
        let mut env = TradingEnv::new(&bs, cfg(1, 0.0, 0.0));
        env.step(1.0);
        let obs = env.obs();
        approx(obs[0], 0.4);
        approx(obs[1], 1.0);
        approx(obs[2], 0.0);
    }

    #[test]
    fn done_only_on_last_bar() {
        let bs = bars(&[10.0, 10.0, 10.0, 10.0], &[10.0, 11.0, 12.0, 13.0]);
        let mut env = TradingEnv::new(&bs, cfg(1, 0.0, 0.0));
        assert!(!env.step(1.0).done);
        assert!(env.step(1.0).done);
    }

    #[test]
    fn env_matches_batch_simulator_accounting() {
        use paperrl_sim::{SimConfig, run};

        // Enters at the close of bar `wait`, like the env's first action.
        struct Delayed {
            wait: usize,
            seen: usize,
        }
        impl paperrl_sim::Strategy for Delayed {
            fn name(&self) -> &str {
                "delayed"
            }
            fn on_close(&mut self, _bar: &Bar, _position_weight: f64, _drawdown: f64) -> f64 {
                self.seen += 1;
                if self.seen > self.wait {
                    1.0
                } else {
                    0.0
                }
            }
        }

        let bs = bars(
            &[10.0; 6],
            &[10.0, 11.0, 12.0, 11.0, 13.0, 14.0],
        );
        let window = 2;
        let mut env = TradingEnv::new(&bs, cfg(window, 0.0, 0.0));
        let mut rewards = Vec::new();
        loop {
            let s = env.step(1.0);
            rewards.push(s.reward);
            if s.done {
                break;
            }
        }

        let sim_cfg = SimConfig {
            initial_cash: 10_000.0,
            fee_bps: 0.0,
            slippage_bps: 0.0,
            periods_per_year: 252.0,
        };
        let mut strat = Delayed { wait: window, seen: 0 };
        let res = run(&bs, &mut strat, &sim_cfg);
        // The env's k-th reward must equal the simulator's log equity ratio
        // over the same bars: they share Portfolio, so this must hold exactly.
        for (k, r) in rewards.iter().enumerate() {
            let prev = res.equity[window + k];
            let cur = res.equity[window + 1 + k];
            approx(*r, (cur / prev).ln() as f32);
        }
    }

    #[test]
    #[should_panic(expected = "steppable")]
    fn env_rejects_zero_step_episodes() {
        // window 1 + 2 bars: i starts on the last bar, nothing to step into.
        let bs = bars(&[10.0, 10.0], &[10.0, 11.0]);
        let _ = TradingEnv::new(&bs, cfg(1, 0.0, 0.0));
    }

    #[test]
    fn same_actions_same_rewards() {
        // Window 2 + 6 bars: exactly 3 steps, done on the last.
        let bs = bars(
            &[10.0, 11.0, 10.5, 11.5, 11.0, 12.0],
            &[10.5, 11.0, 11.2, 11.8, 11.4, 12.5],
        );
        let mut a = TradingEnv::new(&bs, cfg(2, 5.0, 1.0));
        let mut b = TradingEnv::new(&bs, cfg(2, 5.0, 1.0));
        let mut last = Step { reward: 0.0, done: false };
        for action in [1.0, 0.5, 0.0] {
            let sa = a.step(action);
            assert_eq!(sa, b.step(action));
            last = sa;
        }
        assert!(last.done);
    }
}
