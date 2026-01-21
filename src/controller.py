"""Controllers for cart-pole stabilization."""
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.signal import place_poles


class PIDController:
    """PID controller for angle stabilization."""
    
    def __init__(self, kp: float = 100.0, ki: float = 0.5, kd: float = 20.0, setpoint: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.previous_time = None
    
    def compute(self, theta: float, theta_dot: float, t: float) -> float:
        """Compute control force from angle error."""
        error = theta - self.setpoint
        
        # Time step
        if self.previous_time is None:
            self.previous_time = t
            dt = 0.01
        else:
            dt = max(t - self.previous_time, 0.01)
        
        # PID terms
        P = self.kp * error
        self.integral = np.clip(self.integral + error * dt, -100.0, 100.0)
        I = self.ki * self.integral
        D = self.kd * theta_dot
        
        self.previous_time = t
        force = np.clip(P + I + D, -100.0, 100.0)
        return force
    
    def reset(self):
        """Reset internal state."""
        self.integral = 0.0
        self.previous_time = None


class LQRController:
    """LQR controller using optimal state feedback."""
    
    def __init__(
        self,
        cart_mass: float,
        pendulum_mass: float,
        rod_length: float,
        cart_friction: float,
        rotational_damping: float,
        gravity: float,
        Q: np.ndarray = None,
        R: np.ndarray = None,
        setpoint: np.ndarray = None
    ):
        self.M = cart_mass
        self.m = pendulum_mass
        self.L = rod_length
        self.b = cart_friction
        self.c = rotational_damping
        self.g = gravity
        
        # Cost matrices
        self.Q = Q if Q is not None else np.diag([1.0, 1.0, 100.0, 10.0])
        self.R = R if R is not None else np.array([[0.1]])
        self.setpoint = np.array(setpoint if setpoint is not None else [0.0, 0.0, 0.0, 0.0])
        
        # Compute gains
        self.A, self.B = self._linearize_system()
        self.K = self._compute_lqr_gain()
    
    def _linearize_system(self) -> tuple:
        """Linearize around upright equilibrium."""
        M, m, L, b, c, g = self.M, self.m, self.L, self.b, self.c, self.g
        M_total = M + m
        
        A = np.array([
            [0, 1, 0, 0],
            [0, -b/M_total, -m*g/M_total, 0],
            [0, 0, 0, 1],
            [0, b/(L*M_total), g*(M+m)/(L*M_total), -c/(m*L**2)]
        ])
        
        B = np.array([[0], [1/M_total], [0], [-1/(L*M_total)]])
        return A, B
    
    def _compute_lqr_gain(self) -> np.ndarray:
        """Solve CARE for optimal gain K."""
        P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        return np.linalg.inv(self.R) @ self.B.T @ P
    
    def compute(self, state: np.ndarray, t: float) -> float:
        """Compute control force: u = -K(x - x_setpoint)."""
        error = state - self.setpoint
        force = -float(self.K @ error.reshape(-1, 1))
        return np.clip(force, -100.0, 100.0)
    
    def get_gains(self) -> dict:
        """Return gain values."""
        return {
            'k_x': self.K[0, 0],
            'k_x_dot': self.K[0, 1],
            'k_theta': self.K[0, 2],
            'k_theta_dot': self.K[0, 3]
        }


class PolePlacementController:
    """State feedback controller with specified pole locations."""
    
    def __init__(
        self,
        cart_mass: float,
        pendulum_mass: float,
        rod_length: float,
        cart_friction: float,
        rotational_damping: float,
        gravity: float,
        poles: np.ndarray = None,
        setpoint: np.ndarray = None
    ):
        self.M = cart_mass
        self.m = pendulum_mass
        self.L = rod_length
        self.b = cart_friction
        self.c = rotational_damping
        self.g = gravity
        
        self.desired_poles = np.array(poles if poles is not None else [-2.0, -3.0, -4.0, -5.0])
        self.setpoint = np.array(setpoint if setpoint is not None else [0.0, 0.0, 0.0, 0.0])
        
        self.A, self.B = self._linearize_system()
        self.K = self._compute_pole_placement_gain()
        self.achieved_poles = np.linalg.eigvals(self.A - self.B @ self.K)
    
    def _linearize_system(self) -> tuple:
        """Linearize around upright equilibrium."""
        M, m, L, b, c, g = self.M, self.m, self.L, self.b, self.c, self.g
        M_total = M + m
        
        A = np.array([
            [0, 1, 0, 0],
            [0, -b/M_total, -m*g/M_total, 0],
            [0, 0, 0, 1],
            [0, b/(L*M_total), g*(M+m)/(L*M_total), -c/(m*L**2)]
        ])
        
        B = np.array([[0], [1/M_total], [0], [-1/(L*M_total)]])
        return A, B
    
    def _compute_pole_placement_gain(self) -> np.ndarray:
        """Compute K such that eig(A-BK) = desired_poles."""
        result = place_poles(self.A, self.B, self.desired_poles)
        return result.gain_matrix
    
    def compute(self, state: np.ndarray, t: float) -> float:
        """Compute control force: u = -K(x - x_setpoint)."""
        error = state - self.setpoint
        force = -float(self.K @ error.reshape(-1, 1))
        return np.clip(force, -100.0, 100.0)
    
    def get_gains(self) -> dict:
        """Return gain values."""
        return {
            'k_x': self.K[0, 0],
            'k_x_dot': self.K[0, 1],
            'k_theta': self.K[0, 2],
            'k_theta_dot': self.K[0, 3]
        }
    
    def get_poles(self) -> dict:
        """Return desired and achieved poles."""
        return {'desired': self.desired_poles, 'achieved': self.achieved_poles}
