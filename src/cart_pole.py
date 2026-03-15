"""Cart-pole system dynamics."""
import numpy as np


class CartPole:
    """
    Cart-pole (inverted pendulum) system.
    
    State: [x, x_dot, theta, theta_dot]
    """
    
    def __init__(
        self,
        cart_mass: float = 1.0,
        pendulum_mass: float = 0.05,
        rod_length: float = 0.8,
        cart_friction: float = 0.1,
        rotational_damping: float = 0.01,
        gravity: float = 9.81,
        rod_full_length: float = None
    ):
        """Initialize with physical parameters.

        rod_length     — pivot-to-CoM distance (used in dynamics), e.g. 0.5 m
        rod_full_length — physical rod tip distance for visualization only;
                          defaults to rod_length if not provided, e.g. 0.6 m
        """
        self.M = cart_mass          # kg
        self.m = pendulum_mass      # kg
        self.L = rod_length         # m  (pivot to CoM — used in equations of motion)
        self.L_full = rod_full_length if rod_full_length is not None else rod_length  # m (visual rod tip)
        self.b = cart_friction      # N/(m/s)
        self.c = rotational_damping # N*m/(rad/s)
        self.g = gravity            # m/s^2
        
        # State indices
        self.X = 0
        self.X_DOT = 1
        self.THETA = 2
        self.THETA_DOT = 3
    
    def dynamics(self, t: float, state: np.ndarray, force: float = 0.0, pendulum_torque: float = 0.0) -> np.ndarray:
        """
        Compute state derivatives. Returns [x_dot, x_ddot, theta_dot, theta_ddot].
        
        Args:
            t: Current time
            state: Current state [x, x_dot, theta, theta_dot]
            force: Force applied to cart (N)
            pendulum_torque: Torque/impulse applied to pendulum mass (N*m)
        """
        x, x_dot, theta, theta_dot = state
        
        # Mass matrix
        A = np.array([
            [self.M + self.m, self.m * self.L * np.cos(theta)],
            [self.m * self.L * np.cos(theta), self.m * self.L**2]
        ])
        
        # Force vector (pendulum_torque added to rotational equation)
        B = np.array([
            force - self.b * x_dot + self.m * self.L * theta_dot**2 * np.sin(theta),
            -self.c * theta_dot + self.m * self.g * self.L * np.sin(theta) + pendulum_torque
        ])
        
        # Solve for accelerations
        x_ddot, theta_ddot = np.linalg.solve(A, B)
        
        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])
    
    def apply_constraints(self, state: np.ndarray) -> np.ndarray:
        """Apply physical hard stop: pendulum cannot rotate past bottom (θ = ±π).

        In hardware the rod cannot pass through the cart rail, so spinning is
        impossible.  Model this as an inelastic stop at ±π.
        """
        if abs(state[self.THETA]) >= np.pi / 2:
            state = state.copy()
            state[self.THETA] = np.sign(state[self.THETA]) * np.pi / 2
            state[self.THETA_DOT] = 0.0
        return state

    def get_pendulum_position(self, state: np.ndarray) -> tuple:
        """Get pendulum CoM position (x, y) relative to cart pivot."""
        x = state[self.X]
        theta = state[self.THETA]

        pendulum_x = x + self.L * np.sin(theta)
        pendulum_y = self.L * np.cos(theta)

        return pendulum_x, pendulum_y

    def get_rod_tip_position(self, state: np.ndarray) -> tuple:
        """Get physical rod tip position (x, y) relative to cart pivot."""
        x = state[self.X]
        theta = state[self.THETA]

        tip_x = x + self.L_full * np.sin(theta)
        tip_y = self.L_full * np.cos(theta)

        return tip_x, tip_y
    
    def get_energy(self, state: np.ndarray) -> dict:
        """Compute kinetic, potential, and total energy."""
        x, x_dot, theta, theta_dot = state
        
        # Cart KE
        KE_cart = 0.5 * self.M * x_dot**2
        
        # Pendulum KE
        v_pend_x = x_dot + self.L * theta_dot * np.cos(theta)
        v_pend_y = -self.L * theta_dot * np.sin(theta)
        KE_pend = 0.5 * self.m * (v_pend_x**2 + v_pend_y**2)
        
        # Potential energy
        PE = self.m * self.g * self.L * np.cos(theta)
        
        return {
            'kinetic': KE_cart + KE_pend,
            'potential': PE,
            'total': KE_cart + KE_pend + PE
        }
