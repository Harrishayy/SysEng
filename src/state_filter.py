"""Measurement noise and state filtering.

Pipeline matches LQRPendulum.ino (at hardware rate, dt=0.002 s):
  Stage 1 — Moving average MA(5) on raw x and theta
  Stage 2 — Numerical derivative of MA output
  Stage 3 — Exponential LP on derivative
             THETA_DOT_ALPHA=0.20, X_DOT_ALPHA=0.10  @ 500 Hz

Auto-scaling to any sample rate
────────────────────────────────
Both the MA window and LP time constant are scaled so the filter produces
equivalent phase lag at any dt, not just the hardware rate.

  MA size  : max(1, round(5 * dt_hw / dt))
             → MA=5 @ 500 Hz,  MA=1 @ 50 Hz  (avoids 40 ms lag at 50 Hz)

  LP alpha  : computed from  tau_eff = max(tau_hw, dt)
              alpha = 1 - exp(-dt / tau_eff)
             → at hardware rate: original alpha (0.20 / 0.10)
             → when dt > tau_hw: alpha ≈ 0.632  (tau = dt, ~7° lag at 5.87 rad/s)
"""
import numpy as np


class MeasurementNoise:
    """Adds Gaussian noise to position measurements."""

    def __init__(self, position_std: float = 0.005, angle_std: float = 0.01, seed: int = None):
        self.position_std = position_std  # m
        self.angle_std = angle_std        # rad
        self.rng = np.random.default_rng(seed)

    def add_noise(self, state: np.ndarray) -> np.ndarray:
        """Add noise to x and theta only (velocities are derived)."""
        noisy_state = state.copy()
        noisy_state[0] += self.rng.normal(0, self.position_std)
        noisy_state[2] += self.rng.normal(0, self.angle_std)
        return noisy_state


class StateFilter:
    """Three-stage filter matching LQRPendulum.ino, auto-scaled to any dt."""

    # Hardware constants
    _HW_DT = 0.002
    _HW_MA_SIZE = 5
    _HW_THETA_DOT_ALPHA = 0.20
    _HW_X_DOT_ALPHA = 0.10

    def __init__(self, ma_size: int = None, dt: float = 0.002,
                 theta_dot_alpha: float = None, x_dot_alpha: float = None):
        self.dt = dt

        # ── MA size: keep same time window as hardware, minimum 2 ───────────
        if ma_size is None:
            ma_size = max(2, round(self._HW_MA_SIZE * self._HW_DT / dt))
        self.ma_size = ma_size

        # ── LP alphas: clamp tau to max(tau_hw, 2.5·dt) ─────────────────────
        # 2.5·dt gives τ=50 ms at 50 Hz → alpha≈0.33 (much heavier smoothing).
        # At hardware rate dt=0.002 s: 2.5·dt=5 ms < tau_hw, so original
        # hardware alphas (0.20/0.10) are preserved exactly.
        if theta_dot_alpha is None:
            tau_hw = -self._HW_DT / np.log(1 - self._HW_THETA_DOT_ALPHA)
            theta_dot_alpha = 1 - np.exp(-dt / max(tau_hw, 2.5 * dt))
        if x_dot_alpha is None:
            tau_hw = -self._HW_DT / np.log(1 - self._HW_X_DOT_ALPHA)
            x_dot_alpha = 1 - np.exp(-dt / max(tau_hw, 2.5 * dt))

        self.theta_dot_alpha = theta_dot_alpha
        self.x_dot_alpha = x_dot_alpha

        # MA circular buffers
        self.theta_buf = np.zeros(ma_size)
        self.x_buf = np.zeros(ma_size)
        self.ma_idx = 0
        self.ma_count = 0

        # Previous MA outputs for derivative
        self.prev_theta_ma = 0.0
        self.prev_x_ma = 0.0

        # LP state for derivatives
        self.theta_dot_filt = 0.0
        self.x_dot_filt = 0.0

        self.initialized = False

    def reset(self):
        """Reset filter state."""
        self.theta_buf = np.zeros(self.ma_size)
        self.x_buf = np.zeros(self.ma_size)
        self.ma_idx = 0
        self.ma_count = 0
        self.prev_theta_ma = 0.0
        self.prev_x_ma = 0.0
        self.theta_dot_filt = 0.0
        self.x_dot_filt = 0.0
        self.initialized = False

    def filter(self, noisy_state: np.ndarray) -> np.ndarray:
        """Apply MA → derivative → LP filter pipeline."""
        x_noisy = noisy_state[0]
        theta_noisy = noisy_state[2]

        # Stage 1: Update circular MA buffer
        self.theta_buf[self.ma_idx] = theta_noisy
        self.x_buf[self.ma_idx] = x_noisy
        self.ma_idx = (self.ma_idx + 1) % self.ma_size
        self.ma_count = min(self.ma_count + 1, self.ma_size)

        n = self.ma_count
        if n < self.ma_size:
            theta_ma = np.mean(self.theta_buf[:n])
            x_ma = np.mean(self.x_buf[:n])
        else:
            theta_ma = np.mean(self.theta_buf)
            x_ma = np.mean(self.x_buf)

        if not self.initialized:
            self.prev_theta_ma = theta_ma
            self.prev_x_ma = x_ma
            self.theta_dot_filt = 0.0
            self.x_dot_filt = 0.0
            self.initialized = True
            return np.array([x_ma, 0.0, theta_ma, 0.0])

        # Stage 2: Numerical derivative
        theta_dot_raw = (theta_ma - self.prev_theta_ma) / self.dt
        x_dot_raw = (x_ma - self.prev_x_ma) / self.dt

        # Stage 3: Exponential LP on derivative
        self.theta_dot_filt = (self.theta_dot_alpha * theta_dot_raw +
                               (1 - self.theta_dot_alpha) * self.theta_dot_filt)
        self.x_dot_filt = (self.x_dot_alpha * x_dot_raw +
                           (1 - self.x_dot_alpha) * self.x_dot_filt)

        self.prev_theta_ma = theta_ma
        self.prev_x_ma = x_ma

        return np.array([x_ma, self.x_dot_filt, theta_ma, self.theta_dot_filt])

    def get_parameters(self) -> dict:
        """Return filter parameters."""
        return {
            'ma_size': self.ma_size,
            'dt': self.dt,
            'theta_dot_alpha': self.theta_dot_alpha,
            'x_dot_alpha': self.x_dot_alpha
        }


class NoisyStateProcessor:
    """Combined noise injection and filtering."""

    def __init__(
        self,
        position_noise_std: float = 0.005,
        angle_noise_std: float = 0.01,
        ma_size: int = None,
        dt: float = 0.002,
        seed: int = None
    ):
        self.noise = MeasurementNoise(position_noise_std, angle_noise_std, seed)
        self.filter = StateFilter(ma_size=ma_size, dt=dt)

    def reset(self):
        """Reset filter state."""
        self.filter.reset()

    def process(self, true_state: np.ndarray) -> tuple:
        """Add noise and filter. Returns (noisy_state, filtered_state)."""
        noisy_state = self.noise.add_noise(true_state)
        filtered_state = self.filter.filter(noisy_state)
        return noisy_state, filtered_state
