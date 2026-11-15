//! Trading environment for RL agents.
//!
//! The environment shares the batch simulator's fill accounting (through
//! `paperrl_sim::Portfolio`): an action is a target equity weight applied at
//! the current close and filled at the next open with slippage and fees. The
//! reward follows the PaperRL research spec:
//! `log(V_t+1 / V_t) - lambda_d * drawdown_increase - lambda_u * turnover`.

use paperrl_core::Bar;
use paperrl_sim::Portfolio;

mod features;
mod ppo;
pub use features::{
    Ema, ObsFeatures, RollingObs, Rsi, obs_vector, volatility, volume_ratio,
};
pub use ppo::{
    ACTION_WEIGHTS, ModelMeta, PpoConfig, PpoNet, PpoStrategy, TrainedPpo, compute_gae, train_ppo,
};

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct EnvConfig {
    /// Close-to-close returns carried in the observation.
    pub window: usize,
    /// Penalty per unit of drawdown increase beyond `dd_threshold` (lambda_d).
    pub dd_penalty: f64,
    /// Drawdown level where the penalty starts (ordinary fluctuations are free).
    #[serde(default = "default_dd_threshold")]
    pub dd_threshold: f64,
    /// Penalty per unit of position weight times rolling volatility (lambda_v).
    #[serde(default = "default_vol_penalty")]
    pub vol_penalty: f64,
    /// Penalty per unit of traded notional over equity (lambda_u).
    pub turnover_penalty: f64,
    pub initial_cash: f64,
    pub fee_bps: f64,
    pub slippage_bps: f64,
}

fn default_dd_threshold() -> f64 {
    0.10
}

fn default_vol_penalty() -> f64 {
    0.01
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            window: 10,
            // Shaping scales: daily log growth is ~1e-2, a drawdown increase on
            // a down day ~1e-2, so lambda_d = 1 roughly doubles the pain of
            // down days. lambda_u = 5e-3 makes one full entry (turnover 1.0)
            // cost about as much as eight lots of 6 bps fees; anything near
            // 1.0 makes "never invest" the optimal policy.
            dd_penalty: 0.5,
            dd_threshold: 0.15,
            vol_penalty: 0.01,
            turnover_penalty: 0.005,
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

/// Single-symbol trading environment over a stored bar slice.
pub struct TradingEnv<'a> {
    bars: &'a [Bar],
    cfg: EnvConfig,
    pf: Portfolio,
    /// Index of the current close bar.
    i: usize,
    peak: f64,
    prev_excess_dd: f64,
    prev_equity: f64,
    prev_turnover: f64,
    rolling: RollingObs,
    features: ObsFeatures,
}

impl<'a> TradingEnv<'a> {
    pub fn new(bars: &'a [Bar], cfg: EnvConfig) -> Self {
        assert!(
            bars.len() > cfg.window + 1,
            "need at least one steppable bar beyond the observation window"
        );
        let mut rolling = RollingObs::new(cfg.window);
        let mut features = None;
        for bar in bars.iter().take(cfg.window + 1) {
            features = rolling.push(bar).or(features);
        }
        Self {
            bars,
            cfg,
            pf: Portfolio::new(cfg.initial_cash, cfg.fee_bps, cfg.slippage_bps),
            i: cfg.window,
            peak: cfg.initial_cash,
            prev_excess_dd: 0.0,
            prev_equity: cfg.initial_cash,
            prev_turnover: 0.0,
            rolling,
            features: features.expect("window warmed during construction"),
        }
    }

    /// Length of the observation vector: `window` returns + 6 features
    /// (weight, drawdown, volatility, volume ratio, MACD histogram, RSI).
    pub fn obs_len(&self) -> usize {
        self.cfg.window + 6
    }

    /// Features as of the current close: `window` returns, weight, drawdown,
    /// volatility, volume ratio, MACD histogram, and RSI.
    pub fn obs(&self) -> Vec<f32> {
        let i = self.i;
        let eq = self.pf.equity(self.bars[i].close);
        let features = [
            self.pf.position_weight(self.bars[i].close),
            (1.0 - eq / self.peak).max(0.0),
            self.features.volatility,
            self.features.volume_ratio,
            self.features.macd_norm,
            self.features.rsi_norm,
        ];
        obs_vector(&self.features.returns, &features)
    }

    /// Applies `action` (target weight in [0, 1]) at the current close,
    /// advances one bar, fills at the next open, marks at its close, and
    /// returns the reward. `done` is true on the last bar.
    pub fn step(&mut self, action: f64) -> Step {
        self.pf.on_close(action);
        self.i += 1;
        let bar = &self.bars[self.i];
        if let Some(features) = self.rolling.push(bar) {
            self.features = features;
        }
        self.pf.fill_at_open(bar);
        let equity = self.pf.equity(bar.close);
        let dd = (1.0 - equity / self.peak).max(0.0);
        let excess_dd = (dd - self.cfg.dd_threshold).max(0.0);
        let dd_increase = (excess_dd - self.prev_excess_dd).max(0.0);
        let turnover = (self.pf.turnover - self.prev_turnover) / self.prev_equity.max(1e-12);
        let exposure = self.pf.position_weight(bar.close);
        let reward = (equity / self.prev_equity).ln()
            - self.cfg.dd_penalty * dd_increase
            - self.cfg.turnover_penalty * turnover
            - self.cfg.vol_penalty * exposure * self.features.volatility;
        self.peak = self.peak.max(equity);
        self.prev_excess_dd = excess_dd;
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

    fn ts(y: i32, m: u32, d: u32) -> chrono::DateTime<chrono::Utc> {
        use chrono::TimeZone;
        chrono::Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).unwrap()
    }

    fn bar_at(t: chrono::DateTime<chrono::Utc>, open: f64, close: f64) -> Bar {
        Bar {
            timestamp: t,
            open,
            high: open.max(close),
            low: open.min(close),
            close,
            volume: 1000.0,
        }
    }

    fn cfg(window: usize, dd: f64, turnover: f64) -> EnvConfig {
        EnvConfig {
            window,
            dd_penalty: dd,
            dd_threshold: 0.0,
            vol_penalty: 0.0,
            turnover_penalty: turnover,
            initial_cash: 10_000.0,
            fee_bps: 0.0,
            slippage_bps: 0.0,
        }
    }

    fn cfg_thr(window: usize, dd: f64, threshold: f64) -> EnvConfig {
        EnvConfig {
            window,
            dd_penalty: dd,
            dd_threshold: threshold,
            vol_penalty: 0.0,
            turnover_penalty: 0.0,
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
        // window returns + weight, drawdown, vol, volume ratio, MACD, RSI.
        assert_eq!(env.obs_len(), 7);
        let obs = env.obs();
        approx(obs[0], 0.2);
        approx(obs[1], 0.0);
        approx(obs[2], 0.0);
        // Single return window: population stdev is 0.
        approx(obs[3], 0.0);
        // Volume ratio vs the previous single bar: equal volumes -> 1.0.
        approx(obs[4], 1.0);
        // MACD signal seeds on the first histogram value; RSI still neutral.
        approx(obs[5], 0.0);
        approx(obs[6], 0.0);
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
    fn volatility_and_volume_ratio_hand_values() {
        // Population stdev of [0.2, -0.1]: mean 0.05, sd 0.15.
        approx(volatility(&[0.2, -0.1]) as f32, 0.15);
        assert_eq!(volatility(&[0.2]), 0.0);
        // 150 against a prior mean of 200.
        approx(volume_ratio(&[100.0, 300.0], 150.0) as f32, 0.75);
        assert_eq!(volume_ratio(&[], 5.0), 1.0);
    }

    #[test]
    fn env_and_rolling_obs_agree() {
        // Distinct volumes so the volume ratio carries signal.
        let bs: Vec<Bar> = (0..6)
            .map(|i| Bar {
                timestamp: chrono::DateTime::from_timestamp(86_400 * i as i64, 0).unwrap(),
                open: 10.0 + i as f64,
                high: 10.5 + i as f64,
                low: 9.5 + i as f64,
                close: 10.2 + i as f64,
                volume: 1000.0 * (i as f64 + 1.0),
            })
            .collect();
        let env = TradingEnv::new(&bs, cfg(3, 0.0, 0.0));
        let mut rolling = RollingObs::new(3);
        for (i, bar) in bs.iter().enumerate() {
            let rolled = rolling.push(bar);
            if i < 3 {
                assert!(rolled.is_none(), "warmup at bar {i}");
            }
            if i == 3 {
                // The env sits flat at i == window here: weight 0, drawdown 0.
                // Later bars would need the env to step, so stop here.
                let f = rolled.unwrap();
                assert_eq!(
                    env.obs(),
                    obs_vector(
                        &f.returns,
                        &[0.0, 0.0, f.volatility, f.volume_ratio, f.macd_norm, f.rsi_norm]
                    )
                );
                break;
            }
        }
    }

    #[test]
    fn drawdown_penalty_starts_at_threshold() {
        // Equity falls 40% below the peak; with threshold 0.10 only the
        // excess 0.30 is penalized.
        let bs = bars(&[10.0, 10.0, 10.0], &[10.0, 12.0, 6.0]);
        let mut env = TradingEnv::new(&bs, cfg_thr(1, 1.0, 0.10));
        let s = env.step(1.0);
        approx(s.reward, 0.6f32.ln() - 0.3);
    }

    #[test]
    fn vol_penalty_scales_with_exposure_and_volatility() {
        // Window 2 over closes [10, 12, 10.8, 14]: at the final step the two
        // returns are [-0.1, 0.2963], population stdev 0.19815. Full exposure
        // with lambda_v = 1.0 costs 0.19815 on top of the log growth ln(1.4).
        let bs = bars(&[10.0, 10.0, 10.0, 10.0], &[10.0, 12.0, 10.8, 14.0]);
        let mut c = cfg(2, 0.0, 0.0);
        c.vol_penalty = 1.0;
        let mut env = TradingEnv::new(&bs, c);
        let s = env.step(1.0);
        approx(s.reward, 1.4f32.ln() - 0.198148);
        // Flat exposure in a volatile market pays no volatility penalty.
        let mut flat = TradingEnv::new(&bs, c);
        approx(flat.step(0.0).reward, 0.0);
    }

    #[test]
    fn regime_features_follow_trend() {
        // Flat, then a sustained rise: RSI above 50 and a positive MACD
        // histogram.
        let mut up = RollingObs::new(3);
        let mut last = None;
        for i in 0..30 {
            let close = if i < 15 { 100.0 } else { 100.0 + (i - 14) as f64 };
            last = up
                .push(&bar_at(ts(2024, 1, 1 + i as u32 % 28), close, close))
                .or(last);
        }
        let f = last.unwrap();
        assert!(f.rsi_norm > 0.0, "rsi_norm {}", f.rsi_norm);
        assert!(f.macd_norm > 0.0, "macd_norm {}", f.macd_norm);

        let mut down = RollingObs::new(3);
        let mut last = None;
        for i in 0..30 {
            let close = if i < 15 { 100.0 } else { 100.0 - (i - 14) as f64 };
            last = down
                .push(&bar_at(ts(2024, 1, 1 + i as u32 % 28), close, close))
                .or(last);
        }
        let f = last.unwrap();
        assert!(f.rsi_norm < 0.0, "rsi_norm {}", f.rsi_norm);
        assert!(f.macd_norm < 0.0, "macd_norm {}", f.macd_norm);
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
