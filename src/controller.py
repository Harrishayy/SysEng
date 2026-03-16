"""Controllers for cart-pole stabilization."""
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.signal import place_poles


class PIDController:
    """Cascaded PID: position error → angle setpoint → force."""

    def __init__(
        self,
        kp: float = 55.0,
        ki: float = 2.0,
        kd: float = 26.0,
        kp_pos: float = 0.05,
        ki_pos: float = 0.004,
        kd_pos: float = 0.11,
        x_target: float = 2.0,
        max_angle_setpoint: float = 0.09
    ):
        # Angle PID gains
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Position control gains (outer loop)
        self.kp_pos = kp_pos
        self.ki_pos = ki_pos
        self.kd_pos = kd_pos
        self.x_target = x_target
        self.max_angle_setpoint = max_angle_setpoint

        self.integral = 0.0
        self.pos_integral = 0.0
        self.previous_time = None

    def compute(self, state: np.ndarray, t: float) -> float:
        """Compute control force using cascaded position → angle control."""
        x, x_dot, theta, theta_dot = state

        # Time step
        if self.previous_time is None:
            self.previous_time = t
            dt = 0.02
        else:
            dt = max(t - self.previous_time, 0.02)

        # Outer loop: position PID → desired angle
        pos_error = self.x_target - x
        self.pos_integral = np.clip(self.pos_integral + pos_error * dt, -10.0, 10.0)
        angle_setpoint = (self.kp_pos * pos_error +
                         self.ki_pos * self.pos_integral -
                         self.kd_pos * x_dot)
        angle_setpoint = np.clip(angle_setpoint, -self.max_angle_setpoint, self.max_angle_setpoint)

        # Inner loop: angle PID
        angle_error = theta - angle_setpoint

        # PID terms
        P = self.kp * angle_error
        self.integral = np.clip(self.integral + angle_error * dt, -100.0, 100.0)
        I = self.ki * self.integral
        D = self.kd * theta_dot

        self.previous_time = t
        force = np.clip(P + I + D, -100.0, 100.0)
        return force

    def reset(self):
        """Reset internal state."""
        self.integral = 0.0
        self.pos_integral = 0.0
        self.previous_time = None


class LQRController:
    """LQR controller using optimal state feedback.

    Linearisation matches LQRPendulum.ino exactly:
        A[1,2] = -m*g/M    (NOT -(m*g)/(M+m))
        B[1]   = 1/M
        B[3]   = -1/(M*L)
    """

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

        # Cost matrices matching LQRPendulum.ino Python header
        self.Q = Q if Q is not None else np.diag([5.0, 1.0, 100.0, 10.0])
        self.R = R if R is not None else np.array([[0.01]])
        self.setpoint = np.array(setpoint if setpoint is not None else [0.0, 0.0, 0.0, 0.0])

        # Compute gains
        self.A, self.B = self._linearize_system()
        self.K = self._compute_lqr_gain()

    def _linearize_system(self) -> tuple:
        """Linearize around upright equilibrium (matches .ino A/B matrices)."""
        M, m, L, b, c, g = self.M, self.m, self.L, self.b, self.c, self.g

        A = np.array([
            [0,       1,               0,                          0],
            [0,    -b/M,         -m*g/M,               c/(M*L)      ],
            [0,       0,               0,                          1],
            [0, b/(M*L), (M+m)*g/(M*L), -(M+m)*c/(M*m*L**2)         ]
        ])

        B = np.array([[0], [1/M], [0], [-1/(M*L)]])
        return A, B

    def _compute_lqr_gain(self) -> np.ndarray:
        """Solve CARE for optimal gain K. Returns 1-D array of shape (4,)."""
        P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        return (np.linalg.inv(self.R) @ self.B.T @ P).flatten()

    def compute(self, state: np.ndarray, t: float) -> float:
        """Compute control force: u = -K(x - x_setpoint)."""
        error = state - self.setpoint
        force = -float(self.K @ error)
        return np.clip(force, -100.0, 100.0)

    def get_gains(self) -> dict:
        """Return gain values."""
        return {
            'k_x': self.K[0],
            'k_x_dot': self.K[1],
            'k_theta': self.K[2],
            'k_theta_dot': self.K[3]
        }


class PolePlacementController:
    """State feedback controller with specified pole locations.

    If poles=None, derives default poles from LQR closed-loop eigenvalues
    (with Q=diag([5,1,100,10]), R=0.01) so both controllers give identical K.
    """

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

        self.setpoint = np.array(setpoint if setpoint is not None else [0.0, 0.0, 0.0, 0.0])

        self.A, self.B = self._linearize_system()

        if poles is None:
            # Derive poles from LQR closed-loop eigenvalues for consistency
            Q_lqr = np.diag([5.0, 1.0, 100.0, 10.0])
            R_lqr = np.array([[0.01]])
            P = solve_continuous_are(self.A, self.B, Q_lqr, R_lqr)
            K_lqr = (np.linalg.inv(R_lqr) @ self.B.T @ P).flatten()
            self.desired_poles = np.linalg.eigvals(self.A - self.B @ K_lqr.reshape(1, -1))
        else:
            self.desired_poles = np.array(poles)

        self.K = self._compute_pole_placement_gain()
        self.achieved_poles = np.linalg.eigvals(self.A - self.B @ self.K.reshape(1, -1))

    def _linearize_system(self) -> tuple:
        """Linearize around upright equilibrium (matches .ino A/B matrices)."""
        M, m, L, b, c, g = self.M, self.m, self.L, self.b, self.c, self.g

        A = np.array([
            [0,       1,               0,                          0],
            [0,    -b/M,         -m*g/M,               c/(M*L)      ],
            [0,       0,               0,                          1],
            [0, b/(M*L), (M+m)*g/(M*L), -(M+m)*c/(M*m*L**2)         ]
        ])

        B = np.array([[0], [1/M], [0], [-1/(M*L)]])
        return A, B

    def _compute_pole_placement_gain(self) -> np.ndarray:
        """Compute K such that eig(A-BK) = desired_poles. Returns 1-D array."""
        result = place_poles(self.A, self.B, self.desired_poles)
        return result.gain_matrix.flatten()

    def compute(self, state: np.ndarray, t: float) -> float:
        """Compute control force: u = -K(x - x_setpoint)."""
        error = state - self.setpoint
        force = -float(self.K @ error)
        return np.clip(force, -100.0, 100.0)

    def get_gains(self) -> dict:
        """Return gain values."""
        return {
            'k_x': self.K[0],
            'k_x_dot': self.K[1],
            'k_theta': self.K[2],
            'k_theta_dot': self.K[3]
        }

    def get_poles(self) -> dict:
        """Return desired and achieved poles."""
        return {'desired': self.desired_poles, 'achieved': self.achieved_poles}
