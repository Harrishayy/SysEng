"""Measurement noise and state filtering."""
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
    """
    Two-stage state filter.
    
    Stage 1: Low-pass filter  y[k] = α*y[k-1] + (1-α)*x[k]
    Stage 2: Dirty derivative ẏ[k] = (y[k] - y[k-1]) / Ts
    """
    
    def __init__(self, tau_position: float = 0.05, tau_angle: float = 0.02, dt: float = 0.02):
        self.tau_position = tau_position
        self.tau_angle = tau_angle
        self.dt = dt
        
        # Smoothing factors: α = τ / (τ + Ts)
        self.alpha_position = tau_position / (tau_position + dt)
        self.alpha_angle = tau_angle / (tau_angle + dt)
        
        self.y_x_prev = None
        self.y_theta_prev = None
        self.initialized = False
    
    def reset(self):
        """Reset filter state."""
        self.y_x_prev = None
        self.y_theta_prev = None
        self.initialized = False
    
    def filter(self, noisy_state: np.ndarray) -> np.ndarray:
        """Apply filtering and estimate velocities."""
        x_noisy = noisy_state[0]
        theta_noisy = noisy_state[2]
        
        if not self.initialized:
            self.y_x_prev = x_noisy
            self.y_theta_prev = theta_noisy
            self.initialized = True
            return np.array([x_noisy, 0.0, theta_noisy, 0.0])
        
        # Low-pass filter
        y_x = self.alpha_position * self.y_x_prev + (1 - self.alpha_position) * x_noisy
        y_theta = self.alpha_angle * self.y_theta_prev + (1 - self.alpha_angle) * theta_noisy
        
        # Dirty derivative
        x_dot_est = (y_x - self.y_x_prev) / self.dt
        theta_dot_est = (y_theta - self.y_theta_prev) / self.dt
        
        self.y_x_prev = y_x
        self.y_theta_prev = y_theta
        
        return np.array([y_x, x_dot_est, y_theta, theta_dot_est])
    
    def get_parameters(self) -> dict:
        """Return filter parameters."""
        return {
            'tau_position': self.tau_position,
            'tau_angle': self.tau_angle,
            'dt': self.dt,
            'alpha_position': self.alpha_position,
            'alpha_angle': self.alpha_angle
        }


class NoisyStateProcessor:
    """Combined noise injection and filtering."""
    
    def __init__(
        self,
        position_noise_std: float = 0.005,
        angle_noise_std: float = 0.01,
        tau_position: float = 0.05,
        tau_angle: float = 0.02,
        dt: float = 0.02,
        seed: int = None
    ):
        self.noise = MeasurementNoise(position_noise_std, angle_noise_std, seed)
        self.filter = StateFilter(tau_position, tau_angle, dt)
    
    def reset(self):
        """Reset filter state."""
        self.filter.reset()
    
    def process(self, true_state: np.ndarray) -> tuple:
        """Add noise and filter. Returns (noisy_state, filtered_state)."""
        noisy_state = self.noise.add_noise(true_state)
        filtered_state = self.filter.filter(noisy_state)
        return noisy_state, filtered_state
