"""
Controller classes for cart-pole stabilization.
"""
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.signal import place_poles


class PIDController:
    """
    PID controller for cart-pole stabilization.
    
    The controller computes the force to apply to the cart to keep
    the pendulum upright (theta = 0).
    """
    
    def __init__(
        self,
        kp: float = 100.0,
        ki: float = 0.5,
        kd: float = 20.0,
        setpoint: float = 0.0
    ):
        """
        Initialize the PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            setpoint: Target angle (rad), typically 0 for upright
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        
        # Internal state
        self.integral = 0.0
        self.previous_time = None
    
    def compute(self, theta: float, theta_dot: float, t: float) -> float:
        """
        Compute the control force based on current angle state.
        
        Args:
            theta: Current pendulum angle (rad)
            theta_dot: Current angular velocity (rad/s)
            t: Current time (s)
            
        Returns:
            Control force to apply to cart (N)
        """
        # Compute error (positive error means pendulum leaning to positive angle)
        error = theta - self.setpoint
        
        # Initialize time tracking on first call
        if self.previous_time is None:
            self.previous_time = t
            dt = 0.01  # Small default dt
        else:
            dt = t - self.previous_time
            if dt <= 0:
                dt = 0.01
        
        # Proportional term
        P = self.kp * error
        
        # Integral term (with anti-windup)
        self.integral += error * dt
        max_integral = 100.0
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        I = self.ki * self.integral
        
        # Derivative term (using angular velocity directly for stability)
        D = self.kd * theta_dot
        
        # Control force
        force = P + I + D
        
        # Update for next iteration
        self.previous_time = t
        
        # Apply force limits
        max_force = 100.0
        force = np.clip(force, -max_force, max_force)
        
        return force
    
    def reset(self):
        """Reset the controller's internal state."""
        self.integral = 0.0
        self.previous_time = None


class LQRController:
    """
    Linear Quadratic Regulator (LQR) controller for cart-pole stabilization.
    
    Uses full state feedback with optimal gains computed by solving
    the continuous-time algebraic Riccati equation (CARE).
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
        """
        Initialize the LQR controller.
        
        Args:
            cart_mass: Mass of the cart (kg)
            pendulum_mass: Mass of the pendulum bob (kg)
            rod_length: Length of the pendulum rod (m)
            cart_friction: Cart friction coefficient (N/m/s)
            rotational_damping: Rotational damping coefficient (N*m/rad/s)
            gravity: Gravitational acceleration (m/s^2)
            Q: State cost matrix (4x4), default: diagonal [1, 1, 10, 1]
            R: Control cost matrix (1x1), default: [[0.1]]
            setpoint: Desired state [x_desired, 0, 0, 0], default: [0, 0, 0, 0]
        """
        self.M = cart_mass
        self.m = pendulum_mass
        self.L = rod_length
        self.b = cart_friction
        self.c = rotational_damping
        self.g = gravity
        
        # Default cost matrices if not provided
        if Q is None:
            # Penalize position, velocity, angle, and angular velocity
            # Higher weight on angle to prioritize keeping pendulum upright
            Q = np.diag([1.0, 1.0, 100.0, 10.0])
        if R is None:
            # Control effort cost
            R = np.array([[0.1]])
        
        self.Q = Q
        self.R = R
        # Desired state setpoint: [x_desired, 0, 0, 0]
        if setpoint is None:
            setpoint = np.array([0.0, 0.0, 0.0, 0.0])
        self.setpoint = np.array(setpoint, dtype=float)
        
        # Compute linearized system matrices
        self.A, self.B = self._linearize_system()
        
        # Solve Riccati equation for optimal gains
        self.K = self._compute_lqr_gain()
    
    def _linearize_system(self) -> tuple:
        """
        Linearize the cart-pole dynamics around the upright equilibrium.
        
        The system is linearized around [x=0, x_dot=0, theta=0, theta_dot=0].
        
        Returns:
            Tuple of (A, B) matrices for state-space representation
        """
        M, m, L, b, c, g = self.M, self.m, self.L, self.b, self.c, self.g
        
        # Total mass
        M_total = M + m
        
        # Denominator for the linearized equations
        denom = M_total * L - m * L
        
        # State matrix A (4x4)
        # State: [x, x_dot, theta, theta_dot]
        A = np.array([
            [0, 1, 0, 0],
            [0, -b/M_total, -m*g/M_total, 0],
            [0, 0, 0, 1],
            [0, b/(L*M_total), g*(M+m)/(L*M_total), -c/(m*L**2)]
        ])
        
        # Input matrix B (4x1)
        B = np.array([
            [0],
            [1/M_total],
            [0],
            [-1/(L*M_total)]
        ])
        
        return A, B
    
    def _compute_lqr_gain(self) -> np.ndarray:
        """
        Compute the LQR gain matrix K by solving the CARE.
        
        Returns:
            Gain matrix K (1x4)
        """
        # Solve continuous-time algebraic Riccati equation
        P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        
        # Compute optimal gain: K = R^-1 * B^T * P
        K = np.linalg.inv(self.R) @ self.B.T @ P
        
        return K
    
    def compute(self, state: np.ndarray, t: float) -> float:
        """
        Compute the control force using LQR full state feedback with setpoint tracking.
        
        Args:
            state: Current state vector [x, x_dot, theta, theta_dot]
            t: Current time (s) - not used but kept for interface consistency
            
        Returns:
            Control force (N)
        """
        # LQR control law: u = -K * (x - x_setpoint)
        # Compute error from setpoint
        error = state - self.setpoint
        
        # Apply LQR feedback on the error
        force = -float(self.K @ error.reshape(-1, 1))
        
        # Apply force limits
        max_force = 100.0
        force = np.clip(force, -max_force, max_force)
        
        return force
    
    def get_gains(self) -> dict:
        """
        Get the individual state feedback gains.
        
        Returns:
            Dictionary with gains for each state variable
        """
        return {
            'k_x': self.K[0, 0],
            'k_x_dot': self.K[0, 1],
            'k_theta': self.K[0, 2],
            'k_theta_dot': self.K[0, 3]
        }


class PolePlacementController:
    """
    Pole Placement Controller for cart-pole stabilization.
    
    Uses full state feedback with gains computed by placing the 
    closed-loop poles at desired locations. This allows direct 
    specification of system response characteristics.
    
    The closed-loop system: ẋ = (A - BK)x
    has eigenvalues at the specified pole locations.
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
        """
        Initialize the Pole Placement controller.
        
        Args:
            cart_mass: Mass of the cart (kg)
            pendulum_mass: Mass of the pendulum bob (kg)
            rod_length: Length of the pendulum rod (m)
            cart_friction: Cart friction coefficient (N/m/s)
            rotational_damping: Rotational damping coefficient (N*m/rad/s)
            gravity: Gravitational acceleration (m/s^2)
            poles: Desired closed-loop poles (4 complex values). 
                   Default: [-2, -3, -4, -5] (all real, stable)
            setpoint: Desired state [x_desired, 0, 0, 0], default: [0, 0, 0, 0]
        """
        self.M = cart_mass
        self.m = pendulum_mass
        self.L = rod_length
        self.b = cart_friction
        self.c = rotational_damping
        self.g = gravity
        
        # Default poles if not provided
        # All poles in left half-plane for stability
        # More negative = faster response but more control effort
        if poles is None:
            poles = np.array([-2.0, -3.0, -4.0, -5.0])
        self.desired_poles = np.array(poles)
        
        # Desired state setpoint
        if setpoint is None:
            setpoint = np.array([0.0, 0.0, 0.0, 0.0])
        self.setpoint = np.array(setpoint, dtype=float)
        
        # Compute linearized system matrices
        self.A, self.B = self._linearize_system()
        
        # Compute gain matrix using pole placement
        self.K = self._compute_pole_placement_gain()
        
        # Store achieved poles for verification
        self.achieved_poles = np.linalg.eigvals(self.A - self.B @ self.K)
    
    def _linearize_system(self) -> tuple:
        """
        Linearize the cart-pole dynamics around the upright equilibrium.
        
        The system is linearized around [x=0, x_dot=0, theta=0, theta_dot=0].
        
        Returns:
            Tuple of (A, B) matrices for state-space representation
        """
        M, m, L, b, c, g = self.M, self.m, self.L, self.b, self.c, self.g
        
        # Total mass
        M_total = M + m
        
        # State matrix A (4x4)
        # State: [x, x_dot, theta, theta_dot]
        A = np.array([
            [0, 1, 0, 0],
            [0, -b/M_total, -m*g/M_total, 0],
            [0, 0, 0, 1],
            [0, b/(L*M_total), g*(M+m)/(L*M_total), -c/(m*L**2)]
        ])
        
        # Input matrix B (4x1)
        B = np.array([
            [0],
            [1/M_total],
            [0],
            [-1/(L*M_total)]
        ])
        
        return A, B
    
    def _compute_pole_placement_gain(self) -> np.ndarray:
        """
        Compute the gain matrix K using pole placement.
        
        Uses scipy.signal.place_poles to find K such that
        eig(A - BK) = desired_poles
        
        Returns:
            Gain matrix K (1x4)
        """
        # Use scipy's pole placement algorithm
        result = place_poles(self.A, self.B, self.desired_poles)
        K = result.gain_matrix
        
        return K
    
    def compute(self, state: np.ndarray, t: float) -> float:
        """
        Compute the control force using pole placement state feedback.
        
        Args:
            state: Current state vector [x, x_dot, theta, theta_dot]
            t: Current time (s) - not used but kept for interface consistency
            
        Returns:
            Control force (N)
        """
        # Control law: u = -K * (x - x_setpoint)
        error = state - self.setpoint
        
        # Apply state feedback
        force = -float(self.K @ error.reshape(-1, 1))
        
        # Apply force limits
        max_force = 100.0
        force = np.clip(force, -max_force, max_force)
        
        return force
    
    def get_gains(self) -> dict:
        """
        Get the individual state feedback gains.
        
        Returns:
            Dictionary with gains for each state variable
        """
        return {
            'k_x': self.K[0, 0],
            'k_x_dot': self.K[0, 1],
            'k_theta': self.K[0, 2],
            'k_theta_dot': self.K[0, 3]
        }
    
    def get_poles(self) -> dict:
        """
        Get the desired and achieved pole locations.
        
        Returns:
            Dictionary with desired and achieved poles
        """
        return {
            'desired': self.desired_poles,
            'achieved': self.achieved_poles
        }