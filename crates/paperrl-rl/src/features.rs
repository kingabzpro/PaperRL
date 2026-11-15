//! Shared observation feature engineering: rolling returns, volatility,
//! volume ratio, MACD histogram, and RSI. The training environment and live
//! strategies both consume this module, so their observations cannot drift.

use paperrl_core::Bar;

/// Builds the observation vector: `returns` oldest first, then the feature
/// vector (position weight, drawdown, volatility, volume ratio, MACD, RSI).
pub fn obs_vector(returns: &[f64], features: &[f64]) -> Vec<f32> {
    let mut obs = Vec::with_capacity(returns.len() + features.len());
    obs.extend(returns.iter().map(|r| *r as f32));
    obs.extend(features.iter().map(|f| *f as f32));
    obs
}

/// Population standard deviation of the returns (0 for fewer than 2 values).
pub fn volatility(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    (returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64).sqrt()
}

/// Current volume over the mean of the prior `history` volumes.
pub fn volume_ratio(history: &[f64], current: f64) -> f64 {
    if history.is_empty() {
        return 1.0;
    }
    let mean = history.iter().sum::<f64>() / history.len() as f64;
    current / mean.max(1e-12)
}

/// Streaming exponential moving average seeded with its first input.
#[derive(Clone)]
pub struct Ema {
    k: f64,
    value: Option<f64>,
}

impl Ema {
    pub fn new(period: usize) -> Self {
        Self { k: 2.0 / (period as f64 + 1.0), value: None }
    }

    pub fn push(&mut self, x: f64) {
        self.value = Some(match self.value {
            Some(v) => x * self.k + v * (1.0 - self.k),
            None => x,
        });
    }

    pub fn value_or(&self, default: f64) -> f64 {
        self.value.unwrap_or(default)
    }
}

/// Streaming Wilder-smoothed RSI; neutral (50) until seeded.
#[derive(Clone)]
pub struct Rsi {
    period: usize,
    gains: Vec<f64>,
    losses: Vec<f64>,
    avg_gain: Option<f64>,
    avg_loss: Option<f64>,
    prev_close: Option<f64>,
}

impl Rsi {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            gains: Vec::new(),
            losses: Vec::new(),
            avg_gain: None,
            avg_loss: None,
            prev_close: None,
        }
    }

    pub fn push(&mut self, close: f64) {
        if let Some(prev) = self.prev_close {
            let change = close - prev;
            let gain = change.max(0.0);
            let loss = (-change).max(0.0);
            match (self.avg_gain, self.avg_loss) {
                (Some(g), Some(l)) => {
                    let n = self.period as f64;
                    self.avg_gain = Some((g * (n - 1.0) + gain) / n);
                    self.avg_loss = Some((l * (n - 1.0) + loss) / n);
                }
                _ => {
                    self.gains.push(gain);
                    self.losses.push(loss);
                    if self.gains.len() == self.period {
                        self.avg_gain = Some(self.gains.iter().sum::<f64>() / self.period as f64);
                        self.avg_loss = Some(self.losses.iter().sum::<f64>() / self.period as f64);
                    }
                }
            }
        }
        self.prev_close = Some(close);
    }

    /// RSI in [0, 100]; neutral 50 while warming up.
    pub fn value(&self) -> f64 {
        match (self.avg_gain, self.avg_loss) {
            (Some(g), Some(l)) => {
                if l < 1e-12 {
                    100.0
                } else {
                    100.0 - 100.0 / (1.0 + g / l)
                }
            }
            _ => 50.0,
        }
    }
}

/// Regime features produced for one bar.
#[derive(Debug, Clone, PartialEq)]
pub struct ObsFeatures {
    pub returns: Vec<f64>,
    pub volatility: f64,
    pub volume_ratio: f64,
    /// MACD histogram normalized by close.
    pub macd_norm: f64,
    /// RSI centered to [-0.5, 0.5].
    pub rsi_norm: f64,
}

/// Rolling per-bar observation inputs shared by the training environment and
/// live strategies, so both see identical features by construction.
#[derive(Clone)]
pub struct RollingObs {
    window: usize,
    prev_close: Option<f64>,
    returns: Vec<f64>,
    volumes: Vec<f64>,
    ema_fast: Ema,
    ema_slow: Ema,
    signal: Ema,
    rsi: Rsi,
}

impl RollingObs {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            prev_close: None,
            returns: Vec::new(),
            volumes: Vec::new(),
            ema_fast: Ema::new(12),
            ema_slow: Ema::new(26),
            signal: Ema::new(9),
            rsi: Rsi::new(14),
        }
    }

    /// Pushes one bar; once `window` returns and `window` prior volumes exist,
    /// returns the feature set for this bar.
    pub fn push(&mut self, bar: &Bar) -> Option<ObsFeatures> {
        let volume_ratio = if self.volumes.len() == self.window {
            Some(volume_ratio(&self.volumes, bar.volume))
        } else {
            None
        };
        if let Some(prev) = self.prev_close {
            self.returns.push(bar.close / prev - 1.0);
        }
        self.prev_close = Some(bar.close);
        self.volumes.push(bar.volume);
        if self.volumes.len() > self.window {
            self.volumes.remove(0);
        }
        if self.returns.len() > self.window {
            self.returns.remove(0);
        }
        if self.returns.len() < self.window {
            return None;
        }
        let vr = volume_ratio?;

        // MACD(12, 26, 9): histogram normalized by close.
        self.ema_fast.push(bar.close);
        self.ema_slow.push(bar.close);
        let macd = self.ema_fast.value_or(bar.close) - self.ema_slow.value_or(bar.close);
        self.signal.push(macd);
        let macd_norm = (macd - self.signal.value_or(macd)) / bar.close.max(1e-12);

        // RSI(14) centered to [-0.5, 0.5].
        self.rsi.push(bar.close);
        let rsi_norm = self.rsi.value() / 100.0 - 0.5;

        Some(ObsFeatures {
            returns: self.returns.clone(),
            volatility: volatility(&self.returns),
            volume_ratio: vr,
            macd_norm,
            rsi_norm,
        })
    }
}
