//! PPO-clip agent on Burn over the [`TradingEnv`].
//!
//! Actions are discrete target equity weights ([`ACTION_WEIGHTS`], the
//! 0%/25%/50%/100% design from the PaperRL spec). The policy and value
//! function share a two-layer backbone. Training is plain single-threaded
//! PPO-clip with GAE advantages on the pure-Rust ndarray backend.

use std::path::Path;

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::activation;
use burn::tensor::{Device, Tensor, TensorData};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};

use paperrl_core::Bar;

use crate::{EnvConfig, TradingEnv, obs_vector};

/// Device used for training: flex CPU backend with autodiff enabled.
fn train_device() -> Device {
    Device::flex().autodiff()
}

/// Target equity weight for each discrete action index (0% exit / 25% small /
/// 50% medium / 100% full, from the PaperRL action design).
pub const ACTION_WEIGHTS: [f64; 4] = [0.0, 0.25, 0.5, 1.0];

/// PPO hyperparameters; doubles as reproducibility metadata.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PpoConfig {
    pub hidden: usize,
    pub clip_eps: f64,
    pub gamma: f64,
    pub gae_lambda: f64,
    pub lr: f64,
    pub epochs: usize,
    pub rollout_steps: usize,
    pub iters: usize,
    pub value_coef: f64,
    pub ent_coef: f64,
    pub seed: u64,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            hidden: 32,
            clip_eps: 0.2,
            gamma: 0.99,
            gae_lambda: 0.95,
            lr: 3e-4,
            epochs: 4,
            rollout_steps: 512,
            iters: 50,
            value_coef: 0.5,
            ent_coef: 0.01,
            seed: 42,
        }
    }
}

/// Sidecar metadata saved next to every model checkpoint.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelMeta {
    /// Observation lookback used at training time (obs_len = window + 2).
    pub window: usize,
    pub symbol: String,
    pub timeframe: String,
    /// Environment shaping config used at training time.
    pub env: EnvConfig,
    pub config: PpoConfig,
    /// Per-iteration training curve.
    #[serde(default)]
    pub stats: Vec<TrainStats>,
}

/// Policy + value network: shared two-layer backbone, actor and critic heads.
#[derive(Module, Debug)]
pub struct PpoNet {
    input: Linear,
    hidden: Linear,
    actor: Linear,
    critic: Linear,
}

impl PpoNet {
    pub fn new(obs_len: usize, n_actions: usize, hidden: usize, device: &Device) -> Self {
        let lin = |i: usize, o: usize| LinearConfig::new(i, o).init(device);
        Self {
            input: lin(obs_len, hidden),
            hidden: lin(hidden, hidden),
            actor: lin(hidden, n_actions),
            critic: lin(hidden, 1),
        }
    }

    /// Actor logits `[batch, actions]` and value estimates `[batch, 1]`.
    pub fn forward(&self, obs: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
        let x = activation::relu(self.input.forward(obs));
        let x = activation::relu(self.hidden.forward(x));
        (self.actor.forward(x.clone()), self.critic.forward(x))
    }
}

/// Per-iteration training statistics.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct TrainStats {
    pub iter: usize,
    pub mean_reward: f32,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
}

/// A trained agent: network, metadata, and per-iteration statistics.
pub struct TrainedPpo {
    pub net: PpoNet,
    pub meta: ModelMeta,
    pub stats: Vec<TrainStats>,
}

impl TrainedPpo {
    /// Saves the network as burnpack plus a `<stem>.json` metadata sidecar.
    pub fn save(&self, model_path: &Path) -> paperrl_core::Result<()> {
        self.net
            .clone()
            .save_file(model_path)
            .map_err(|e| paperrl_core::Error::Other(e.to_string()))?;
        let meta_path = model_path.with_extension("json");
        let json = serde_json::to_vec_pretty(&self.meta)
            .map_err(|e| paperrl_core::Error::Other(e.to_string()))?;
        std::fs::write(meta_path, json)?;
        Ok(())
    }

    /// Loads a network plus metadata previously written by [`TrainedPpo::save`].
    pub fn load(model_path: &Path) -> paperrl_core::Result<Self> {
        let meta_path = model_path.with_extension("json");
        let meta: ModelMeta = serde_json::from_str(
            &std::fs::read_to_string(&meta_path)
                .map_err(|e| paperrl_core::Error::Other(format!("{}: {e}", meta_path.display())))?,
        )
        .map_err(|e| paperrl_core::Error::Other(format!("{}: {e}", meta_path.display())))?;
        let device = Device::flex();
        let net = PpoNet::new(
            meta.window + 2,
            ACTION_WEIGHTS.len(),
            meta.config.hidden,
            &device,
        )
        .try_load_file(model_path)
        .map_err(|e| paperrl_core::Error::Other(format!("{}: {e}", model_path.display())))?;
        Ok(Self { net, meta, stats: Vec::new() })
    }
}

/// Generalized advantage estimation over one rollout.
/// Returns `(advantages, returns)`.
pub fn compute_gae(
    rewards: &[f32],
    values: &[f32],
    dones: &[bool],
    last_value: f32,
    gamma: f64,
    gae_lambda: f64,
) -> (Vec<f32>, Vec<f32>) {
    let n = rewards.len();
    let mut advantages = vec![0.0f32; n];
    let mut gae = 0.0f32;
    for t in (0..n).rev() {
        let next_value = if dones[t] {
            0.0
        } else if t + 1 == n {
            last_value
        } else {
            values[t + 1]
        };
        let delta = rewards[t] + gamma as f32 * next_value - values[t];
        gae = delta + (gamma * gae_lambda) as f32 * if dones[t] { 0.0 } else { gae };
        advantages[t] = gae;
    }
    let returns = advantages
        .iter()
        .zip(values)
        .map(|(a, v)| a + v)
        .collect();
    (advantages, returns)
}

/// Value estimate of a single observation (no RNG draw).
fn value_of(net: &PpoNet, device: &Device, obs: &[f32]) -> f32 {
    let n = obs.len();
    let t = Tensor::<2>::from_data(TensorData::new(obs.to_vec(), vec![1, n]), device);
    let (_, value) = net.forward(t);
    value.into_data().as_slice::<f32>().expect("f32 value")[0]
}

/// One policy+value forward pass on a single observation: sampled action,
/// its log-probability, and the value estimate.
fn forward_action(
    net: &PpoNet,
    device: &Device,
    obs: &[f32],
    rng: &mut StdRng,
) -> (usize, f32, f32) {
    let n = obs.len();
    let t = Tensor::<2>::from_data(TensorData::new(obs.to_vec(), vec![1, n]), device);
    let (logits, value) = net.forward(t);
    let probs = activation::softmax(logits, 1)
        .into_data()
        .as_slice::<f32>()
        .expect("f32 probs")
        .to_vec();
    let v = value.into_data().as_slice::<f32>().expect("f32 value")[0];
    // Sample from the categorical distribution.
    let u: f32 = rng.random();
    let mut cum = 0.0;
    let mut action = probs.len() - 1;
    for (i, p) in probs.iter().enumerate() {
        cum += p;
        if u <= cum {
            action = i;
            break;
        }
    }
    let logprob = probs[action].max(1e-8).ln();
    (action, logprob, v)
}

fn normalize(v: &[f32]) -> Vec<f32> {
    if v.len() < 2 {
        return v.to_vec();
    }
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let var =
        v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (v.len() - 1) as f32;
    let sd = var.sqrt() + 1e-8;
    v.iter().map(|x| (x - mean) / sd).collect()
}

/// Trains PPO on one symbol's bars. Each iteration collects
/// `config.rollout_steps` steps and applies `config.epochs` update epochs.
pub fn train_ppo(
    bars: &[Bar],
    env_cfg: EnvConfig,
    config: PpoConfig,
    symbol: &str,
    timeframe: &str,
) -> TrainedPpo {
    let device = train_device();
    // Seeds network initialization (flex stores a global RNG for inits).
    device.seed(config.seed);
    let obs_len = env_cfg.window + 2;
    let n_actions = ACTION_WEIGHTS.len();
    let mut net = PpoNet::new(obs_len, n_actions, config.hidden, &device);
    let mut opt = AdamConfig::new().init();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut env = TradingEnv::new(bars, env_cfg);
    let mut stats = Vec::new();
    // Minibatch size capped by the rollout length.
    let minibatch = config.rollout_steps.min(64);

    for iter in 0..config.iters {
        // Collect one rollout with the current policy.
        let mut obs_batch: Vec<Vec<f32>> = Vec::with_capacity(config.rollout_steps);
        let mut actions = Vec::with_capacity(config.rollout_steps);
        let mut logprobs = Vec::with_capacity(config.rollout_steps);
        let mut rewards = Vec::with_capacity(config.rollout_steps);
        let mut dones = Vec::with_capacity(config.rollout_steps);
        let mut values = Vec::with_capacity(config.rollout_steps);
        let mut last_done = false;
        let mut sum_reward = 0.0f32;
        for _ in 0..config.rollout_steps {
            let obs = env.obs();
            let (action, logprob, value) = forward_action(&net, &device, &obs, &mut rng);
            let step = env.step(ACTION_WEIGHTS[action]);
            last_done = step.done;
            sum_reward += step.reward;
            obs_batch.push(obs);
            actions.push(action);
            logprobs.push(logprob);
            rewards.push(step.reward);
            dones.push(step.done);
            values.push(value);
            if step.done {
                env = TradingEnv::new(bars, env_cfg);
            }
        }
        // Bootstrap value for the state after the last transition: zero when
        // the rollout ended on a terminal, otherwise the critic's estimate of
        // the current (next) state.
        let bootstrap = if last_done { 0.0 } else { value_of(&net, &device, &env.obs()) };
        let (advantages, returns) =
            compute_gae(&rewards, &values, &dones, bootstrap, config.gamma, config.gae_lambda);
        let adv_norm = normalize(&advantages);
        let n = obs_batch.len();
        let onehot = |i: usize| {
            let mut v = vec![0.0f32; n_actions];
            v[actions[i]] = 1.0;
            v
        };

        // PPO-clip update epochs.
        let mut policy_loss_sum = 0.0f32;
        let mut value_loss_sum = 0.0f32;
        let mut entropy_sum = 0.0f32;
        let mut updates = 0usize;
        for _ in 0..config.epochs {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.shuffle(&mut rng);
            for chunk in idx.chunks(minibatch) {
                let m = chunk.len();
                let obs_rows: Vec<f32> =
                    chunk.iter().flat_map(|&i| obs_batch[i].clone()).collect();
                let onehot_rows: Vec<f32> = chunk.iter().flat_map(|&i| onehot(i)).collect();
                let old_lp: Vec<f32> = chunk.iter().map(|&i| logprobs[i]).collect();
                let a: Vec<f32> = chunk.iter().map(|&i| adv_norm[i]).collect();
                let r: Vec<f32> = chunk.iter().map(|&i| returns[i]).collect();

                let obs_t =
                    Tensor::<2>::from_data(TensorData::new(obs_rows, vec![m, obs_len]), &device);
                let (logits, v) = net.forward(obs_t);
                let logp_all = activation::log_softmax(logits, 1);
                let oh_t = Tensor::<2>::from_data(
                    TensorData::new(onehot_rows, vec![m, n_actions]),
                    &device,
                );
                // Log-prob of the taken actions via one-hot masking.
                let logp = (logp_all.clone() * oh_t).sum_dim(1);
                let old_lp_t =
                    Tensor::<2>::from_data(TensorData::new(old_lp, vec![m, 1]), &device);
                let ratio = (logp - old_lp_t).exp();
                let adv_t = Tensor::<2>::from_data(TensorData::new(a, vec![m, 1]), &device);
                let surr1 = ratio.clone() * adv_t.clone();
                let surr2 = ratio.clamp(1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv_t;
                let policy_loss = -surr1.min_pair(surr2).mean();
                let ret_t = Tensor::<2>::from_data(TensorData::new(r, vec![m, 1]), &device);
                let value_loss = (v - ret_t).powf_scalar(2.0).mean();
                let entropy = -(logp_all.clone().exp() * logp_all).sum_dim(1).mean();
                let loss = policy_loss.clone()
                    + value_loss.clone() * config.value_coef
                    - entropy.clone() * config.ent_coef;
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &net);
                net = opt.step(config.lr, net, grads);

                policy_loss_sum += -policy_loss.into_data().as_slice::<f32>().expect("f32")[0];
                value_loss_sum += value_loss.into_data().as_slice::<f32>().expect("f32")[0];
                entropy_sum += entropy.into_data().as_slice::<f32>().expect("f32")[0];
                updates += 1;
            }
        }
        let u = updates.max(1) as f32;
        stats.push(TrainStats {
            iter,
            mean_reward: sum_reward / n.max(1) as f32,
            policy_loss: policy_loss_sum / u,
            value_loss: value_loss_sum / u,
            entropy: entropy_sum / u,
        });
        tracing::info!(
            "ppo iter {iter}: mean_reward {:.4}, policy_loss {:.4}, value_loss {:.4}, entropy {:.4}",
            stats[stats.len() - 1].mean_reward,
            stats[stats.len() - 1].policy_loss,
            stats[stats.len() - 1].value_loss,
            stats[stats.len() - 1].entropy
        );
    }
    TrainedPpo {
        net,
        meta: ModelMeta {
            window: env_cfg.window,
            symbol: symbol.to_string(),
            timeframe: timeframe.to_string(),
            env: env_cfg,
            config,
            stats: stats.clone(),
        },
        stats,
    }
}

/// Runs a trained policy as a `paperrl_sim::Strategy` for backtesting.
#[derive(Clone)]
pub struct PpoStrategy {
    net: PpoNet,
    device: Device,
    meta: ModelMeta,
    prev_close: Option<f64>,
    returns: Vec<f64>,
}

impl PpoStrategy {
    pub fn load(model_path: &Path) -> paperrl_core::Result<Self> {
        let trained = TrainedPpo::load(model_path)?;
        Ok(Self {
            net: trained.net,
            device: Device::flex(),
            meta: trained.meta,
            prev_close: None,
            returns: Vec::new(),
        })
    }

    /// Metadata of the loaded checkpoint.
    pub fn meta(&self) -> &ModelMeta {
        &self.meta
    }
}

impl paperrl_sim::Strategy for PpoStrategy {
    fn name(&self) -> &str {
        "ppo"
    }

    fn on_close(&mut self, bar: &Bar, position_weight: f64, drawdown: f64) -> f64 {
        if let Some(prev) = self.prev_close {
            self.returns.push(bar.close / prev - 1.0);
        }
        self.prev_close = Some(bar.close);
        if self.returns.len() > self.meta.window {
            let drop = self.returns.len() - self.meta.window;
            self.returns.drain(0..drop);
        }
        if self.returns.len() < self.meta.window {
            // Warmup: not enough history for the observation yet.
            return 0.0;
        }
        let obs = obs_vector(&self.returns, position_weight, drawdown);
        let (probs, _) = {
            let k = obs.len();
            let t =
                Tensor::<2>::from_data(TensorData::new(obs, vec![1, k]), &self.device);
            let (logits, value) = self.net.forward(t);
            let p = activation::softmax(logits, 1)
                .into_data()
                .as_slice::<f32>()
                .expect("f32 probs")
                .to_vec();
            let v = value.into_data().as_slice::<f32>().expect("f32 value")[0];
            (p, v)
        };
        let (best, _) = probs.iter().enumerate().fold(
            (0usize, f32::NEG_INFINITY),
            |(bi, bp), (i, &p)| {
                if p > bp {
                    (i, p)
                } else {
                    (bi, bp)
                }
            },
        );
        ACTION_WEIGHTS[best]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// burn-flex stores the init seed in a process-global, so tests that call
    /// `train_ppo` must not run concurrently with each other.
    static TRAIN_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "left {a} vs right {b}");
    }

    #[test]
    fn action_weights_match_the_spec() {
        assert_eq!(ACTION_WEIGHTS, [0.0, 0.25, 0.5, 1.0]);
    }

    #[test]
    fn gae_hand_case_with_bootstrap_and_done() {
        let (adv, rets) = compute_gae(
            &[1.0, 1.0],
            &[0.5, 0.5],
            &[false, true],
            100.0,
            0.9,
            0.8,
        );
        // t=1: done -> no bootstrap: delta = 1 - 0.5, adv = 0.5, ret = 1.0.
        // t=0: delta = 1 + 0.9*0.5 - 0.5 = 0.95, adv = 0.95 + 0.9*0.8*0.5 = 1.31,
        //      ret = 1.31 + 0.5 = 1.81.
        approx(adv[0], 1.31);
        approx(adv[1], 0.5);
        approx(rets[0], 1.81);
        approx(rets[1], 1.0);
    }

    fn bars_rising(n: usize) -> Vec<Bar> {
        (0..n)
            .map(|i| Bar {
                timestamp: chrono::DateTime::from_timestamp(86_400 * i as i64, 0).unwrap(),
                open: 10.0 + i as f64,
                high: 10.5 + i as f64,
                low: 9.5 + i as f64,
                close: 10.2 + i as f64,
                volume: 1000.0,
            })
            .collect()
    }

    fn tiny_config(iters: usize, rollout: usize) -> PpoConfig {
        PpoConfig {
            hidden: 8,
            clip_eps: 0.2,
            gamma: 0.99,
            gae_lambda: 0.95,
            lr: 3e-4,
            epochs: 1,
            rollout_steps: rollout,
            iters,
            value_coef: 0.5,
            ent_coef: 0.01,
            seed: 42,
        }
    }

    #[test]
    fn train_runs_and_produces_finite_stats() {
        let bars = bars_rising(60);
        let env_cfg = EnvConfig { window: 4, ..EnvConfig::default() };
        let _train_guard = TRAIN_LOCK.lock().unwrap();
        let trained = train_ppo(&bars, env_cfg, tiny_config(2, 16), "TEST", "1d");
        assert_eq!(trained.stats.len(), 2);
        for s in &trained.stats {
            assert!(s.mean_reward.is_finite(), "{s:?}");
            assert!(s.policy_loss.is_finite(), "{s:?}");
            assert!(s.value_loss.is_finite(), "{s:?}");
            assert!(s.entropy.is_finite(), "{s:?}");
        }
        assert_eq!(trained.meta.window, 4);
        assert_eq!(trained.meta.symbol, "TEST");
    }

    #[test]
    fn save_then_load_roundtrips() {
        let bars = bars_rising(60);
        let env_cfg = EnvConfig { window: 4, ..EnvConfig::default() };
        let _train_guard = TRAIN_LOCK.lock().unwrap();
        let trained = train_ppo(&bars, env_cfg, tiny_config(1, 16), "TEST", "1d");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ppo-test.bpk");
        trained.save(&path).unwrap();
        assert!(path.exists());
        assert!(dir.path().join("ppo-test.json").exists());

        // The loaded policy must emit valid action weights.
        let mut strat = PpoStrategy::load(&path).unwrap();
        for bar in bars.iter().take(8) {
            let w = paperrl_sim::Strategy::on_close(&mut strat, bar, 0.0, 0.0);
            assert!(ACTION_WEIGHTS.contains(&w), "got {w}");
        }
    }

    #[test]
    fn same_seed_same_training_curve() {
        let bars = bars_rising(60);
        let env_cfg = EnvConfig { window: 4, ..EnvConfig::default() };
        let _train_guard = TRAIN_LOCK.lock().unwrap();
        let a = train_ppo(&bars, env_cfg, tiny_config(2, 16), "TEST", "1d");
        let b = train_ppo(&bars, env_cfg, tiny_config(2, 16), "TEST", "1d");
        assert_eq!(a.stats.len(), b.stats.len());
        for (x, y) in a.stats.iter().zip(&b.stats) {
            assert_eq!(x.mean_reward.to_bits(), y.mean_reward.to_bits(), "{x:?} vs {y:?}");
            assert_eq!(x.policy_loss.to_bits(), y.policy_loss.to_bits());
            assert_eq!(x.value_loss.to_bits(), y.value_loss.to_bits());
        }
    }

    #[test]
    fn save_writes_metadata_sidecar() {
        let bars = bars_rising(60);
        let env_cfg = EnvConfig { window: 4, ..EnvConfig::default() };
        let _train_guard = TRAIN_LOCK.lock().unwrap();
        let trained = train_ppo(&bars, env_cfg, tiny_config(1, 16), "TEST", "1d");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.bpk");
        trained.save(&path).unwrap();
        let meta: ModelMeta =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("m.json")).unwrap())
                .unwrap();
        assert_eq!(meta.window, 4);
        assert_eq!(meta.config.seed, 42);
        assert_eq!(meta.symbol, "TEST");
    }
}
