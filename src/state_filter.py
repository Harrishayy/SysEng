"""
State filtering and noise simulation for cart-pole system.

Implements:
- Measurement noise injection
- Two-stage filtering:
  Stage 1: Low-pass filtering of position
  Stage 2: Velocity estimation via dirty derivative
"""
import numpy as np


class MeasurementNoise:
    """
    Adds realistic measurement noise to state observations.
    """
    
    def __init__(
        self,
        position_std: float = 0.005,    # Standard deviation for cart position (m)
        angle_std: float = 0.01,        # Standard deviation for angle (rad)
        seed: int = None
    ):
        """
        Initialize measurement noise generator.
        
        Args:
            position_std: Standard deviation of position noise (m)
            angle_std: Standard deviation of angle noise (rad)
            seed: Random seed for reproducibility
        """
        self.position_std = position_std
        self.angle_std = angle_std
        self.rng = np.random.default_rng(seed)
    
    def add_noise(self, state: np.ndarray) -> np.ndarray:
        """
        Add measurement noise to state.
        
        In practice, encoders measure position directly, so we add noise
        only to x and theta (not velocities, which are derived).
        
        Args:
            state: True state [x, x_dot, theta, theta_dot]
            
        Returns:
            Noisy measurement [x_noisy, x_dot, theta_noisy, theta_dot]
        """
        noisy_state = state.copy()
        noisy_state[0] += self.rng.normal(0, self.position_std)  # x
        noisy_state[2] += self.rng.normal(0, self.angle_std)     # theta
        return noisy_state


class StateFilter:
    """
    Two-stage state filter for cart-pole measurements.
    
    Stage 1: Low-pass filter for position measurements
        τ*ẏ(t) + y(t) = x(t)  (continuous)
        y[k] = α*y[k-1] + (1-α)*x[k]  (discrete)
        where α = τ / (τ + Ts)
    
    Stage 2: Dirty derivative for velocity estimation
        ẏ[k] = (y[k] - y[k-1]) / Ts
    """
    
    def __init__(
        self,
        tau_position: float = 0.05,     # Time constant for cart position filter (s)
        tau_angle: float = 0.02,        # Time constant for angle filter (s)
        dt: float = 0.02                # Sampling period (s)
    ):
        """
        Initialize the state filter.
        
        Args:
            tau_position: Filter time constant for cart position (s)
            tau_angle: Filter time constant for pendulum angle (s)
            dt: Sampling period (s)
        """
        self.tau_position = tau_position
        self.tau_angle = tau_angle
        self.dt = dt
        
        # Compute smoothing factors: α = τ / (τ + Ts)
        self.alpha_position = tau_position / (tau_position + dt)
        self.alpha_angle = tau_angle / (tau_angle + dt)
        
        # Filter state (previous filtered values)
        self.y_x_prev = None        # Previous filtered cart position
        self.y_theta_prev = None    # Previous filtered angle
        
        self.initialized = False
    
    def reset(self):
        """Reset the filter state."""
        self.y_x_prev = None
        self.y_theta_prev = None
        self.initialized = False
    
    def filter(self, noisy_state: np.ndarray) -> np.ndarray:
        """
        Apply two-stage filtering to noisy measurements.
        
        Stage 1: Low-pass filter positions
        Stage 2: Estimate velocities via dirty derivative
        
        Args:
            noisy_state: Noisy measurement [x_noisy, x_dot_noisy, theta_noisy, theta_dot_noisy]
                        Note: x_dot and theta_dot in input are ignored (we estimate them)
        
        Returns:
            Filtered state [x_filtered, x_dot_estimated, theta_filtered, theta_dot_estimated]
        """
        x_noisy = noisy_state[0]
        theta_noisy = noisy_state[2]
        
        if not self.initialized:
            # Initialize filter with first measurement
            self.y_x_prev = x_noisy
            self.y_theta_prev = theta_noisy
            self.initialized = True
            
            # For first sample, velocity estimate is zero
            return np.array([x_noisy, 0.0, theta_noisy, 0.0])
        
        # Stage 1: Low-pass filtering of positions
        # y[k] = α*y[k-1] + (1-α)*x[k]
        y_x = self.alpha_position * self.y_x_prev + (1 - self.alpha_position) * x_noisy
        y_theta = self.alpha_angle * self.y_theta_prev + (1 - self.alpha_angle) * theta_noisy
        
        # Stage 2: Velocity estimation (dirty derivative)
        # ẏ[k] = (y[k] - y[k-1]) / Ts
        x_dot_est = (y_x - self.y_x_prev) / self.dt
        theta_dot_est = (y_theta - self.y_theta_prev) / self.dt
        
        # Update previous values for next iteration
        self.y_x_prev = y_x
        self.y_theta_prev = y_theta
        
        return np.array([y_x, x_dot_est, y_theta, theta_dot_est])
    
    def get_parameters(self) -> dict:
        """
        Get filter parameters.
        
        Returns:
            Dictionary with filter parameters
        """
        return {
            'tau_position': self.tau_position,
            'tau_angle': self.tau_angle,
            'dt': self.dt,
            'alpha_position': self.alpha_position,
            'alpha_angle': self.alpha_angle
        }


class NoisyStateProcessor:
    """
    Combined noise injection and filtering for simulation.
    
    Wraps MeasurementNoise and StateFilter for easy use.
    """
    
    def __init__(
        self,
        position_noise_std: float = 0.005,
        angle_noise_std: float = 0.01,
        tau_position: float = 0.05,
        tau_angle: float = 0.02,
        dt: float = 0.02,
        seed: int = None
    ):
        """
        Initialize the noisy state processor.
        
        Args:
            position_noise_std: Standard deviation of position noise (m)
            angle_noise_std: Standard deviation of angle noise (rad)
            tau_position: Filter time constant for cart position (s)
            tau_angle: Filter time constant for angle (s)
            dt: Sampling period (s)
            seed: Random seed for reproducibility
        """
        self.noise = MeasurementNoise(
            position_std=position_noise_std,
            angle_std=angle_noise_std,
            seed=seed
        )
        self.filter = StateFilter(
            tau_position=tau_position,
            tau_angle=tau_angle,
            dt=dt
        )
    
    def reset(self):
        """Reset the filter state."""
        self.filter.reset()
    
    def process(self, true_state: np.ndarray) -> tuple:
        """
        Add noise and filter the state.
        
        Args:
            true_state: True state [x, x_dot, theta, theta_dot]
            
        Returns:
            Tuple of (noisy_state, filtered_state)
        """
        noisy_state = self.noise.add_noise(true_state)
        filtered_state = self.filter.filter(noisy_state)
        return noisy_state, filtered_state
