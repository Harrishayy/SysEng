"""
CartPole dynamics class for inverted pendulum simulation.
"""
import numpy as np


class CartPole:
    """
    Represents the physical cart-pole (inverted pendulum) system.
    
    State vector: [x, x_dot, theta, theta_dot]
        - x: cart position (m)
        - x_dot: cart velocity (m/s)
        - theta: pendulum angle from vertical (rad)
        - theta_dot: angular velocity (rad/s)
    """
    
    def __init__(
        self,
        cart_mass: float = 1.0,
        pendulum_mass: float = 0.05,
        rod_length: float = 0.8,
        cart_friction: float = 0.1,
        rotational_damping: float = 0.01,
        gravity: float = 9.81
    ):
        """
        Initialize the cart-pole system with physical parameters.
        
        Args:
            cart_mass: Mass of the cart (kg)
            pendulum_mass: Mass of the pendulum bob (kg)
            rod_length: Length of the pendulum rod (m)
            cart_friction: Cart friction coefficient (N/m/s)
            rotational_damping: Rotational damping coefficient (N*m/rad/s)
            gravity: Gravitational acceleration (m/s^2)
        """
        self.M = cart_mass
        self.m = pendulum_mass
        self.L = rod_length
        self.b = cart_friction
        self.c = rotational_damping
        self.g = gravity
        
        # State indices for clarity
        self.X = 0
        self.X_DOT = 1
        self.THETA = 2
        self.THETA_DOT = 3
    
    def dynamics(self, t: float, state: np.ndarray, force: float = 0.0) -> np.ndarray:
        """
        Compute the state derivatives for the cart-pole system.
        
        Args:
            t: Current time (s)
            state: State vector [x, x_dot, theta, theta_dot]
            force: External force applied to the cart (N)
            
        Returns:
            State derivatives [x_dot, x_ddot, theta_dot, theta_ddot]
        """
        x, x_dot, theta, theta_dot = state
        
        # Mass matrix A
        A = np.array([
            [self.M + self.m, self.m * self.L * np.cos(theta)],
            [self.m * self.L * np.cos(theta), self.m * self.L**2]
        ])
        
        # Forcing vector B
        B = np.array([
            force - self.b * x_dot + self.m * self.L * theta_dot**2 * np.sin(theta),
            -self.c * theta_dot + self.m * self.g * self.L * np.sin(theta)
        ])
        
        # Solve for accelerations
        accelerations = np.linalg.solve(A, B)
        x_ddot, theta_ddot = accelerations
        
        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])
    
    def get_pendulum_position(self, state: np.ndarray) -> tuple:
        """
        Calculate the pendulum bob position relative to cart.
        
        Args:
            state: State vector [x, x_dot, theta, theta_dot]
            
        Returns:
            Tuple of (pendulum_x, pendulum_y) in world coordinates
        """
        x = state[self.X]
        theta = state[self.THETA]
        
        # Pendulum bob position (theta=0 is straight up)
        pendulum_x = x + self.L * np.sin(theta)
        pendulum_y = self.L * np.cos(theta)
        
        return pendulum_x, pendulum_y
    
    def get_energy(self, state: np.ndarray) -> dict:
        """
        Calculate the system's kinetic and potential energy.
        
        Args:
            state: State vector [x, x_dot, theta, theta_dot]
            
        Returns:
            Dictionary with 'kinetic', 'potential', and 'total' energy
        """
        x, x_dot, theta, theta_dot = state
        
        # Cart kinetic energy
        KE_cart = 0.5 * self.M * x_dot**2
        
        # Pendulum kinetic energy (translational + rotational)
        v_pend_x = x_dot + self.L * theta_dot * np.cos(theta)
        v_pend_y = -self.L * theta_dot * np.sin(theta)
        KE_pend = 0.5 * self.m * (v_pend_x**2 + v_pend_y**2)
        
        # Potential energy (reference: cart level)
        PE = self.m * self.g * self.L * np.cos(theta)
        
        kinetic = KE_cart + KE_pend
        potential = PE
        
        return {
            'kinetic': kinetic,
            'potential': potential,
            'total': kinetic + potential
        }
