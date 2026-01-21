"""
Controller classes for cart-pole stabilization.
"""
import numpy as np
from scipy.linalg import solve_continuous_are


class PIDController:
    """
    Cascaded PID controller for cart-pole stabilization with position control.
    
    The controller computes the force to apply to the cart to:
    1. Keep the pendulum upright (theta = 0) - primary objective
    2. Move cart to desired position (x = x_setpoint) - secondary objective
    
    Uses a cascaded approach where position control is weighted less than angle control.
    """
    
    def __init__(
        self,
        kp_theta: float = 100.0,
        ki_theta: float = 0.5,
        kd_theta: float = 20.0,
        kp_x: float = 10.0,
        ki_x: float = 0.1,
        kd_x: float = 15.0,
        theta_setpoint: float = 0.0,
        x_setpoint: float = 0.0
    ):
        """
        Initialize the cascaded PID controller.
        
        Args:
            kp_theta: Proportional gain for angle
            ki_theta: Integral gain for angle
            kd_theta: Derivative gain for angle
            kp_x: Proportional gain for position
            ki_x: Integral gain for position
            kd_x: Derivative gain for position (velocity damping)
            theta_setpoint: Target angle (rad), typically 0 for upright
            x_setpoint: Target cart position (m)
        """
        # Angle PID gains
        self.kp_theta = kp_theta
        self.ki_theta = ki_theta
        self.kd_theta = kd_theta
        self.theta_setpoint = theta_setpoint
        
        # Position PID gains
        self.kp_x = kp_x
        self.ki_x = ki_x
        self.kd_x = kd_x
        self.x_setpoint = x_setpoint
        
        # Internal state
        self.integral_theta = 0.0
        self.integral_x = 0.0
        self.previous_time = None
    
    def compute(self, state: np.ndarray, t: float) -> float:
        """
        Compute the control force based on current state.
        
        Args:
            state: Current state vector [x, x_dot, theta, theta_dot]
            t: Current time (s)
            
        Returns:
            Control force to apply to cart (N)
        """
        x, x_dot, theta, theta_dot = state
        
        # Initialize time tracking on first call
        if self.previous_time is None:
            self.previous_time = t
            dt = 0.01  # Small default dt
        else:
            dt = t - self.previous_time
            if dt <= 0:
                dt = 0.01
        
        # --- Angle Control (Primary) ---
        error_theta = theta - self.theta_setpoint
        
        # Proportional term
        P_theta = self.kp_theta * error_theta
        
        # Integral term (with anti-windup)
        self.integral_theta += error_theta * dt
        max_integral = 100.0
        self.integral_theta = np.clip(self.integral_theta, -max_integral, max_integral)
        I_theta = self.ki_theta * self.integral_theta
        
        # Derivative term (using angular velocity)
        D_theta = self.kd_theta * theta_dot
        
        # Angle control force
        force_theta = P_theta + I_theta + D_theta
        
        # --- Position Control (Secondary) ---
        error_x = x - self.x_setpoint
        
        # Proportional term
        P_x = self.kp_x * error_x
        
        # Integral term (with anti-windup)
        self.integral_x += error_x * dt
        self.integral_x = np.clip(self.integral_x, -max_integral, max_integral)
        I_x = self.ki_x * self.integral_x
        
        # Derivative term (velocity damping)
        D_x = self.kd_x * x_dot
        
        # Position control force
        force_x = P_x + I_x + D_x
        
        # --- Combined Control ---
        # Combine both controllers (angle control is primary, position is secondary)
        # Negative sign because positive force moves cart right
        force = force_theta + force_x
        
        # Update for next iteration
        self.previous_time = t
        
        # Apply force limits
        max_force = 100.0
        force = np.clip(force, -max_force, max_force)
        
        return force
    
    def reset(self):
        """Reset the controller's internal state."""
        self.integral_theta = 0.0
        self.integral_x = 0.0
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


class SMCController:
    """
    Sliding Mode Controller (SMC) for cart-pole stabilization.
    
    A robust nonlinear controller that drives the system state to a 
    sliding surface and keeps it there despite disturbances.
    
    Sliding surface: s = lambda * theta + theta_dot
    When s = 0, the system follows theta_dot = -lambda * theta,
    which is exponentially stable.
    """
    
    def __init__(
        self,
        cart_mass: float,
        pendulum_mass: float,
        rod_length: float,
        cart_friction: float,
        rotational_damping: float,
        gravity: float,
        lambda_: float = 10.0,
        eta: float = 20.0,
        phi: float = 0.1,
        use_saturation: bool = True
    ):
        """
        Initialize the SMC controller.
        
        Args:
            cart_mass: Mass of the cart (kg)
            pendulum_mass: Mass of the pendulum bob (kg)
            rod_length: Length of the pendulum rod (m)
            cart_friction: Cart friction coefficient (N/m/s)
            rotational_damping: Rotational damping coefficient (N*m/rad/s)
            gravity: Gravitational acceleration (m/s^2)
            lambda_: Sliding surface slope (determines convergence rate on surface)
            eta: Switching gain (determines reaching speed to surface)
            phi: Boundary layer thickness (for saturation smoothing)
            use_saturation: If True, use sat() instead of sign() for smoother control
        """
        self.M = cart_mass
        self.m = pendulum_mass
        self.L = rod_length
        self.b = cart_friction
        self.c = rotational_damping
        self.g = gravity
        
        self.lambda_ = lambda_
        self.eta = eta
        self.phi = phi
        self.use_saturation = use_saturation
    
    def _compute_sliding_surface(self, theta: float, theta_dot: float) -> float:
        """
        Compute the sliding surface value.
        
        s = lambda * theta + theta_dot
        
        Args:
            theta: Pendulum angle (rad)
            theta_dot: Angular velocity (rad/s)
            
        Returns:
            Sliding surface value s
        """
        return self.lambda_ * theta + theta_dot
    
    def _switching_function(self, s: float) -> float:
        """
        Compute the switching function (sign or saturation).
        
        Args:
            s: Sliding surface value
            
        Returns:
            Switching value in [-1, 1]
        """
        if self.use_saturation:
            # Saturation function for smooth control (boundary layer approach)
            if abs(s) <= self.phi:
                return s / self.phi
            else:
                return np.sign(s)
        else:
            # Pure sign function (may cause chattering)
            return np.sign(s)
    
    def compute(self, state: np.ndarray, t: float) -> float:
        """
        Compute the control force using sliding mode control.
        
        The control law consists of:
        1. Equivalent control (u_eq): keeps system on sliding surface
        2. Switching control (u_sw): drives system to sliding surface
        
        Args:
            state: State vector [x, x_dot, theta, theta_dot]
            t: Current time (s)
            
        Returns:
            Control force (N)
        """
        x, x_dot, theta, theta_dot = state
        
        # System parameters
        M, m, L, b, c, g = self.M, self.m, self.L, self.b, self.c, self.g
        
        # Compute sliding surface
        s = self._compute_sliding_surface(theta, theta_dot)
        
        # Precompute trigonometric terms
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Mass matrix elements (from dynamics)
        M11 = M + m
        M12 = m * L * cos_theta
        M22 = m * L**2
        
        # Determinant of mass matrix
        det_M = M11 * M22 - M12 * M12
        
        # For the sliding surface s = lambda*theta + theta_dot,
        # we need s_dot = 0 on the surface.
        # s_dot = lambda*theta_dot + theta_ddot
        
        # From the equations of motion, theta_ddot depends on F.
        # We can derive the equivalent control by setting s_dot = 0.
        
        # The RHS of the dynamics (without F):
        # For theta_ddot: comes from solving the coupled equations
        
        # Simplified equivalent control derivation:
        # theta_ddot = f(state) + g(state)*F
        # For s_dot = 0: lambda*theta_dot + f + g*F = 0
        # F_eq = -(lambda*theta_dot + f) / g
        
        # Compute f and g coefficients for theta_ddot
        # From the dynamics: A * [x_ddot, theta_ddot]^T = B
        # where B1 = F - b*x_dot + m*L*theta_dot^2*sin(theta)
        #       B2 = -c*theta_dot + m*g*L*sin(theta)
        
        # Solving for theta_ddot:
        # theta_ddot = (M11 * B2 - M12 * B1) / det_M
        
        # B2 term (independent of F)
        B2_no_F = -c * theta_dot + m * g * L * sin_theta
        
        # B1 terms split into F-dependent and F-independent parts
        B1_no_F = -b * x_dot + m * L * theta_dot**2 * sin_theta
        
        # theta_ddot = (M11 * B2_no_F - M12 * B1_no_F) / det_M - M12 / det_M * F
        # So: f = (M11 * B2_no_F - M12 * B1_no_F) / det_M
        #     g = -M12 / det_M
        
        f = (M11 * B2_no_F - M12 * B1_no_F) / det_M
        g_coef = -M12 / det_M
        
        # Equivalent control: sets s_dot = lambda*theta_dot + theta_ddot = 0
        # lambda*theta_dot + f + g*F_eq = 0
        # F_eq = -(lambda*theta_dot + f) / g
        
        if abs(g_coef) > 1e-10:
            u_eq = -(self.lambda_ * theta_dot + f) / g_coef
        else:
            u_eq = 0.0
        
        # Switching control: drives system toward sliding surface
        # u_sw = -eta * sign(s) / g  (or with saturation)
        if abs(g_coef) > 1e-10:
            u_sw = -self.eta * self._switching_function(s) / g_coef
        else:
            u_sw = -self.eta * self._switching_function(s)
        
        # Total control
        force = u_eq + u_sw
        
        # Apply force limits
        max_force = 100.0
        force = np.clip(force, -max_force, max_force)
        
        return force
    
    def get_sliding_surface(self, state: np.ndarray) -> float:
        """
        Get the current sliding surface value (useful for plotting).
        
        Args:
            state: State vector [x, x_dot, theta, theta_dot]
            
        Returns:
            Sliding surface value s
        """
        return self._compute_sliding_surface(state[2], state[3])
    
    def get_parameters(self) -> dict:
        """
        Get the controller parameters.
        
        Returns:
            Dictionary with controller parameters
        """
        return {
            'lambda': self.lambda_,
            'eta': self.eta,
            'phi': self.phi,
            'use_saturation': self.use_saturation
        }