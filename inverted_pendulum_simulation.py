"""
Mobile Inverted Pendulum Simulation
====================================
University Engineering Project - Week 2 Phase A

A complete simulation of a cart-pole system with:
- Lagrangian-derived nonlinear dynamics
- Linear and rotational friction
- Kalman Filter for state estimation
- PID and LQR controllers
- Interactive Matplotlib visualization

Author: Systems Engineering Coursework
"""

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from collections import deque
from abc import ABC, abstractmethod


# =============================================================================
# PHYSICAL SYSTEM - CartPoleSystem
# =============================================================================

class CartPoleSystem:
    """
    Cart-Pole (Mobile Inverted Pendulum) system modeled using Lagrangian mechanics.
    
    State vector: [x, x_dot, theta, theta_dot]
    - x: cart position (m)
    - x_dot: cart velocity (m/s)
    - theta: pendulum angle from vertical (rad), theta=0 is upright
    - theta_dot: angular velocity (rad/s)
    
    The equations of motion are derived from the Lagrangian:
    L = T - V where T is kinetic energy and V is potential energy.
    """
    
    def __init__(self, M=1.0, m=0.05, L=0.8, b=0.1, c=0.005, g=9.81):
        """
        Initialize the cart-pole system parameters.
        
        Parameters:
        -----------
        M : float - Cart mass (kg)
        m : float - Pendulum tip mass (kg)
        L : float - Rod length (m)
        b : float - Linear drag coefficient on cart (N·s/m)
        c : float - Rotational damping at pivot (N·m·s/rad)
        g : float - Gravitational acceleration (m/s²)
        """
        self.M = M
        self.m = m
        self.L = L
        self.b = b
        self.c = c
        self.g = g
        
    def dynamics(self, t, state, F):
        """
        Compute state derivatives using Lagrangian-derived equations of motion.
        
        The Lagrangian derivation:
        --------------------------
        Position of cart: (x, 0)
        Position of pendulum tip: (x + L*sin(θ), L*cos(θ))
        
        Kinetic Energy:
        T = 0.5*M*ẋ² + 0.5*m*[(ẋ + L*θ̇*cos(θ))² + (L*θ̇*sin(θ))²]
        T = 0.5*M*ẋ² + 0.5*m*[ẋ² + 2*L*ẋ*θ̇*cos(θ) + L²*θ̇²]
        T = 0.5*(M+m)*ẋ² + m*L*ẋ*θ̇*cos(θ) + 0.5*m*L²*θ̇²
        
        Potential Energy (θ=0 is upright, so max potential):
        V = m*g*L*cos(θ)
        
        Lagrangian: L = T - V
        
        Euler-Lagrange equations with dissipation:
        d/dt(∂L/∂q̇) - ∂L/∂q = Q (generalized forces including friction)
        
        This gives us:
        (M+m)ẍ + m*L*θ̈*cos(θ) - m*L*θ̇²*sin(θ) = F - b*ẋ
        m*L²*θ̈ + m*L*ẍ*cos(θ) - m*g*L*sin(θ) = -c*θ̇
        """
        x, x_dot, theta, theta_dot = state
        
        # Trigonometric terms
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Mass matrix terms
        # From the coupled equations:
        # [M+m,    m*L*cos(θ)] [ẍ ]   [F - b*ẋ + m*L*θ̇²*sin(θ)]
        # [m*L*cos(θ), m*L²  ] [θ̈] = [-c*θ̇ + m*g*L*sin(θ)    ]
        
        # Determinant of mass matrix
        det = (self.M + self.m) * self.m * self.L**2 - (self.m * self.L * cos_theta)**2
        
        # Right-hand side terms
        rhs1 = F - self.b * x_dot + self.m * self.L * theta_dot**2 * sin_theta
        rhs2 = -self.c * theta_dot + self.m * self.g * self.L * sin_theta
        
        # Solve for accelerations using Cramer's rule
        x_ddot = (self.m * self.L**2 * rhs1 - self.m * self.L * cos_theta * rhs2) / det
        theta_ddot = ((self.M + self.m) * rhs2 - self.m * self.L * cos_theta * rhs1) / det
        
        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])
    
    def step(self, state, F, dt):
        """
        Advance the system by one time step using RK45 integration.
        
        Parameters:
        -----------
        state : ndarray - Current state [x, x_dot, theta, theta_dot]
        F : float - Applied force to cart (N)
        dt : float - Time step (s)
        
        Returns:
        --------
        ndarray - New state after time step
        """
        sol = solve_ivp(
            lambda t, s: self.dynamics(t, s, F),
            [0, dt],
            state,
            method='RK45',
            t_eval=[dt]
        )
        return sol.y[:, -1]
    
    def linearize(self):
        """
        Return the linearized state-space matrices (A, B) around the upright equilibrium.
        
        Linearization around θ=0, θ̇=0, x=0, ẋ=0:
        Using small angle approximations: sin(θ) ≈ θ, cos(θ) ≈ 1
        
        The linearized equations become:
        (M+m)ẍ + m*L*θ̈ = F - b*ẋ
        m*L²*θ̈ + m*L*ẍ = m*g*L*θ - c*θ̇
        
        Solving via Cramer's rule with mass matrix determinant:
        det = (M+m)*m*L² - (m*L)² = m*L²*M
        
        Results:
        ẍ = [m*L²*(F - b*ẋ) - m*L*(m*g*L*θ - c*θ̇)] / det
        θ̈ = [(M+m)*(m*g*L*θ - c*θ̇) - m*L*(F - b*ẋ)] / det
        
        State-space form: ẋ = A*x + B*u
        where state = [x, ẋ, θ, θ̇], input u = F
        """
        M, m, L, b, c, g = self.M, self.m, self.L, self.b, self.c, self.g
        
        # Mass matrix determinant at θ=0: (M+m)*m*L² - (m*L)² = m*L²*M
        det_M = m * L**2 * M
        
        # A matrix coefficients from linearized EoM
        # Derived from Cramer's rule solution:
        # ẍ: coeff of ẋ = -b*m*L²/det, coeff of θ = -m²*g*L²/det, coeff of θ̇ = m*L*c/det
        # θ̈: coeff of ẋ = m*L*b/det, coeff of θ = (M+m)*m*g*L/det (POSITIVE - unstable!), coeff of θ̇ = -(M+m)*c/det
        A = np.array([
            [0, 1, 0, 0],
            [0, -b*m*L**2/det_M, -m**2*g*L**2/det_M, m*L*c/det_M],
            [0, 0, 0, 1],
            [0, m*L*b/det_M, (M+m)*m*g*L/det_M, -(M+m)*c/det_M]
        ])
        
        # B matrix: effect of force F
        # ẍ: coeff of F = m*L²/det (positive - force accelerates cart)
        # θ̈: coeff of F = -m*L/det (negative - cart accel causes opposite pendulum accel)
        B = np.array([
            [0],
            [m*L**2/det_M],
            [0],
            [-m*L/det_M]
        ])
        
        return A, B
    
    def get_measured_state(self, true_state, pos_noise_std=0.01, angle_noise_std=0.02):
        """
        Return noisy measurements of position and angle (simulating real sensors).
        
        Parameters:
        -----------
        true_state : ndarray - True state [x, x_dot, theta, theta_dot]
        pos_noise_std : float - Standard deviation of position noise (m)
        angle_noise_std : float - Standard deviation of angle noise (rad)
        
        Returns:
        --------
        ndarray - Measured state with noise on x and theta only
        """
        measured = true_state.copy()
        measured[0] += np.random.normal(0, pos_noise_std)  # x
        measured[2] += np.random.normal(0, angle_noise_std)  # theta
        return measured


# =============================================================================
# KALMAN FILTER - State Estimation
# =============================================================================

class KalmanFilter:
    """
    Discrete Kalman Filter for state estimation from noisy measurements.
    
    Estimates the full state [x, x_dot, theta, theta_dot] from noisy 
    measurements of only x and theta.
    """
    
    def __init__(self, A, B, dt, Q_process=None, R_measurement=None):
        """
        Initialize the Kalman Filter.
        
        Parameters:
        -----------
        A : ndarray - Continuous-time state matrix (4x4)
        B : ndarray - Continuous-time input matrix (4x1)
        dt : float - Time step for discretization
        Q_process : ndarray - Process noise covariance (4x4)
        R_measurement : ndarray - Measurement noise covariance (2x2)
        """
        self.n_states = 4
        self.n_measurements = 2
        
        # Discretize the continuous system: A_d ≈ I + A*dt, B_d ≈ B*dt
        self.A_d = np.eye(4) + A * dt
        self.B_d = B * dt
        
        # Measurement matrix: we measure x and theta
        self.H = np.array([
            [1, 0, 0, 0],  # x measurement
            [0, 0, 1, 0]   # theta measurement
        ])
        
        # Process noise covariance
        if Q_process is None:
            self.Q = np.diag([0.001, 0.01, 0.001, 0.01])
        else:
            self.Q = Q_process
            
        # Measurement noise covariance
        if R_measurement is None:
            self.R = np.diag([0.01**2, 0.02**2])  # Match sensor noise
        else:
            self.R = R_measurement
            
        # Initial state estimate and covariance
        self.x_est = np.zeros(4)
        self.P = np.eye(4) * 0.1
        
    def reset(self, initial_state=None):
        """Reset the filter state."""
        if initial_state is not None:
            self.x_est = initial_state.copy()
        else:
            self.x_est = np.zeros(4)
        self.P = np.eye(4) * 0.1
        
    def predict(self, u):
        """
        Prediction step: propagate state estimate forward.
        
        Parameters:
        -----------
        u : float - Control input (force)
        """
        # State prediction: x_k|k-1 = A*x_k-1|k-1 + B*u_k-1
        self.x_est = self.A_d @ self.x_est + self.B_d.flatten() * u
        
        # Covariance prediction: P_k|k-1 = A*P_k-1|k-1*A^T + Q
        self.P = self.A_d @ self.P @ self.A_d.T + self.Q
        
    def update(self, z):
        """
        Update step: incorporate measurement.
        
        Parameters:
        -----------
        z : ndarray - Measurement vector [x_meas, theta_meas]
        """
        # Innovation: y = z - H*x_k|k-1
        y = z - self.H @ self.x_est
        
        # Innovation covariance: S = H*P*H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: K = P*H^T*S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update: x_k|k = x_k|k-1 + K*y
        self.x_est = self.x_est + K @ y
        
        # Covariance update: P_k|k = (I - K*H)*P_k|k-1
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
    def step(self, u, z):
        """
        Perform one complete filter cycle.
        
        Parameters:
        -----------
        u : float - Control input
        z : ndarray - Measurement [x_meas, theta_meas]
        
        Returns:
        --------
        ndarray - Estimated state
        """
        self.predict(u)
        self.update(z)
        return self.x_est.copy()


# =============================================================================
# CONTROLLERS - PID and LQR
# =============================================================================

class Controller(ABC):
    """Abstract base class for controllers."""
    
    @abstractmethod
    def compute_control(self, state, setpoint):
        """Compute control signal given current state and setpoint."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset controller internal state."""
        pass


class PIDController(Controller):
    """
    PID Controller for the inverted pendulum.
    
    Uses angle error as primary feedback with optional position control.
    """
    
    def __init__(self, Kp=150.0, Ki=0.5, Kd=20.0, Kp_pos=5.0, dt=0.02):
        """
        Initialize PID controller.
        
        Parameters:
        -----------
        Kp : float - Proportional gain for angle
        Ki : float - Integral gain for angle
        Kd : float - Derivative gain for angle
        Kp_pos : float - Proportional gain for position
        dt : float - Time step
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Kp_pos = Kp_pos
        self.dt = dt
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.integral_limit = 50.0  # Anti-windup
        
    def compute_control(self, state, setpoint):
        """
        Compute control force.
        
        Parameters:
        -----------
        state : ndarray - [x, x_dot, theta, theta_dot]
        setpoint : dict - {'x': target_x, 'theta': target_theta}
        
        Returns:
        --------
        float - Control force (N)
        
        Control Logic:
        - When theta > 0 (tilted right), cart must accelerate RIGHT to catch it
        - So F should be proportional to +theta (not -theta)
        """
        x, x_dot, theta, theta_dot = state
        target_theta = setpoint.get('theta', 0.0)
        target_x = setpoint.get('x', 0.0)
        
        # Angle error: positive theta means tilted right, need positive force
        # So we use theta directly (not setpoint - theta)
        angle_term = theta - target_theta
        
        # Position error: if cart is too far right, tilt pendulum left to bring it back
        x_error = x - target_x
        
        # PID on angle (to stabilize pendulum)
        self.integral += angle_term * self.dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        
        # Control signal: push in direction of tilt to catch the pendulum
        F = (self.Kp * angle_term + 
             self.Ki * self.integral + 
             self.Kd * theta_dot)  # Damping on angular velocity
        
        # Position control: add force to return cart to target position
        # Small gain to avoid fighting the balance controller
        F += self.Kp_pos * x_error + 0.5 * self.Kp_pos * x_dot
        
        self.prev_error = angle_term
        
        return np.clip(F, -100, 100)  # Force limits
    
    def reset(self):
        """Reset integral and derivative terms."""
        self.integral = 0.0
        self.prev_error = 0.0
        
    def set_gains(self, Kp=None, Ki=None, Kd=None):
        """Update PID gains."""
        if Kp is not None:
            self.Kp = Kp
        if Ki is not None:
            self.Ki = Ki
        if Kd is not None:
            self.Kd = Kd


class LQRController(Controller):
    """
    Linear Quadratic Regulator (LQR) controller.
    
    Uses the linearized system to compute optimal state feedback gains.
    """
    
    def __init__(self, A, B, Q=None, R=None):
        """
        Initialize LQR controller.
        
        Parameters:
        -----------
        A : ndarray - State matrix (4x4)
        B : ndarray - Input matrix (4x1)
        Q : ndarray - State cost matrix (4x4)
        R : ndarray - Control cost matrix (1x1)
        """
        self.A = A
        self.B = B
        
        # Default Q and R matrices
        if Q is None:
            # Penalize position, velocity, angle, and angular velocity
            self.Q = np.diag([10, 1, 100, 10])
        else:
            self.Q = Q
            
        if R is None:
            self.R = np.array([[0.1]])
        else:
            self.R = R
            
        # Compute LQR gain
        self.K = self._compute_gain()
        
    def _compute_gain(self):
        """
        Compute the optimal LQR gain matrix K.
        
        Solves the continuous-time algebraic Riccati equation (CARE):
        A^T*P + P*A - P*B*R^{-1}*B^T*P + Q = 0
        
        Then: K = R^{-1}*B^T*P
        """
        try:
            P = solve_continuous_are(self.A, self.B, self.Q, self.R)
            K = np.linalg.inv(self.R) @ self.B.T @ P
            return K.flatten()
        except Exception as e:
            print(f"LQR computation failed: {e}")
            # Fallback gains
            return np.array([5, 10, 100, 20])
    
    def compute_control(self, state, setpoint):
        """
        Compute control force using state feedback.
        
        Parameters:
        -----------
        state : ndarray - [x, x_dot, theta, theta_dot]
        setpoint : dict - {'x': target_x, 'theta': target_theta}
        
        Returns:
        --------
        float - Control force (N)
        """
        # Error state (deviation from setpoint)
        target = np.array([
            setpoint.get('x', 0.0),
            0.0,  # target velocity = 0
            setpoint.get('theta', 0.0),
            0.0   # target angular velocity = 0
        ])
        
        error = state - target
        
        # LQR control: u = -K*error
        F = -self.K @ error
        
        return np.clip(F, -100, 100)
    
    def reset(self):
        """Reset controller (LQR is stateless)."""
        pass
    
    def update_weights(self, Q=None, R=None):
        """Update Q and R matrices and recompute gains."""
        if Q is not None:
            self.Q = Q
        if R is not None:
            self.R = R
        self.K = self._compute_gain()


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

class Simulation:
    """
    Main simulation class that coordinates the system, filter, and controller.
    """
    
    def __init__(self, dt=0.02):
        """
        Initialize the simulation.
        
        Parameters:
        -----------
        dt : float - Time step (s), default 50Hz
        """
        self.dt = dt
        
        # Create the physical system
        self.system = CartPoleSystem()
        
        # Get linearized matrices for filter and LQR
        self.A, self.B = self.system.linearize()
        
        # Create Kalman filter
        self.kalman = KalmanFilter(self.A, self.B, dt)
        
        # Create controllers
        self.pid_controller = PIDController(dt=dt)
        self.lqr_controller = LQRController(self.A, self.B)
        
        # Active controller
        self.active_controller = 'LQR'
        
        # State
        self.true_state = np.array([0.0, 0.0, 0.1, 0.0])  # Start slightly tilted
        self.estimated_state = self.true_state.copy()
        self.measured_state = self.true_state.copy()
        
        # Setpoint
        self.setpoint = {'x': 0.0, 'theta': 0.0}
        
        # External disturbance
        self.disturbance_force = 0.0
        
        # Time
        self.time = 0.0
        
        # History for plotting
        self.history_length = 500
        self.time_history = deque(maxlen=self.history_length)
        self.theta_true_history = deque(maxlen=self.history_length)
        self.theta_meas_history = deque(maxlen=self.history_length)
        self.theta_est_history = deque(maxlen=self.history_length)
        self.x_true_history = deque(maxlen=self.history_length)
        self.x_meas_history = deque(maxlen=self.history_length)
        self.x_est_history = deque(maxlen=self.history_length)
        self.control_history = deque(maxlen=self.history_length)
        
        # Last control signal
        self.last_control = 0.0
        
    def reset(self, initial_state=None):
        """Reset the simulation."""
        if initial_state is None:
            initial_state = np.array([0.0, 0.0, 0.1, 0.0])
        
        self.true_state = initial_state.copy()
        self.estimated_state = initial_state.copy()
        self.measured_state = initial_state.copy()
        self.time = 0.0
        self.disturbance_force = 0.0
        self.last_control = 0.0
        
        # Reset filter and controllers
        self.kalman.reset(initial_state)
        self.pid_controller.reset()
        self.lqr_controller.reset()
        
        # Clear history
        self.time_history.clear()
        self.theta_true_history.clear()
        self.theta_meas_history.clear()
        self.theta_est_history.clear()
        self.x_true_history.clear()
        self.x_meas_history.clear()
        self.x_est_history.clear()
        self.control_history.clear()
        
    def set_controller(self, controller_type):
        """Switch between PID and LQR controllers."""
        self.active_controller = controller_type
        
    def apply_disturbance(self, force):
        """Apply an external disturbance force."""
        self.disturbance_force = force
        
    def step(self):
        """
        Advance simulation by one time step.
        
        Returns:
        --------
        tuple - (true_state, estimated_state, control_signal)
        """
        # Get noisy measurement
        self.measured_state = self.system.get_measured_state(self.true_state)
        measurement = np.array([self.measured_state[0], self.measured_state[2]])
        
        # Update Kalman filter
        self.estimated_state = self.kalman.step(self.last_control, measurement)
        
        # Compute control using estimated state
        if self.active_controller == 'PID':
            control = self.pid_controller.compute_control(self.estimated_state, self.setpoint)
        else:
            control = self.lqr_controller.compute_control(self.estimated_state, self.setpoint)
        
        # Add disturbance
        total_force = control + self.disturbance_force
        self.disturbance_force *= 0.9  # Decay disturbance
        
        # Update true state
        self.true_state = self.system.step(self.true_state, total_force, self.dt)
        
        # Wrap angle to [-π, π]
        self.true_state[2] = np.arctan2(np.sin(self.true_state[2]), 
                                         np.cos(self.true_state[2]))
        
        # Store history
        self.time_history.append(self.time)
        self.theta_true_history.append(self.true_state[2])
        self.theta_meas_history.append(self.measured_state[2])
        self.theta_est_history.append(self.estimated_state[2])
        self.x_true_history.append(self.true_state[0])
        self.x_meas_history.append(self.measured_state[0])
        self.x_est_history.append(self.estimated_state[0])
        self.control_history.append(control)
        
        self.time += self.dt
        self.last_control = control
        
        return self.true_state, self.estimated_state, control


# =============================================================================
# VISUALIZATION
# =============================================================================

class PendulumVisualizer:
    """
    Interactive Matplotlib visualization for the inverted pendulum.
    """
    
    def __init__(self, simulation):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        simulation : Simulation - The simulation instance to visualize
        """
        self.sim = simulation
        self.paused = False
        
        # Setup figure and layout
        self._setup_figure()
        self._setup_animation_plot()
        self._setup_data_plots()
        self._setup_widgets()
        self._setup_keyboard()
        
    def _setup_figure(self):
        """Create the figure and grid layout."""
        # Use a dark theme for modern look - with error handling
        try:
            plt.style.use('dark_background')
        except Exception:
            pass  # Fall back to default style
        
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.patch.set_facecolor('#1a1a2e')
        
        # Create grid: 3 rows, 2 columns
        # Top row: animation (spans both columns)
        # Middle row: theta plot, x plot
        # Bottom row: sliders and controls
        self.gs = GridSpec(3, 2, height_ratios=[2, 1, 0.5], 
                          width_ratios=[1, 1],
                          hspace=0.3, wspace=0.25)
        
        self.fig.suptitle('Mobile Inverted Pendulum Simulation', 
                         fontsize=16, fontweight='bold', color='#e94560')
        
    def _setup_animation_plot(self):
        """Setup the main animation subplot."""
        self.ax_anim = self.fig.add_subplot(self.gs[0, :])
        self.ax_anim.set_xlim(-2.5, 2.5)
        self.ax_anim.set_ylim(-0.5, 1.5)
        self.ax_anim.set_aspect('equal')
        self.ax_anim.set_facecolor('#16213e')
        self.ax_anim.grid(True, alpha=0.2, color='#0f3460')
        self.ax_anim.set_xlabel('Position (m)', color='#eee')
        self.ax_anim.set_ylabel('Height (m)', color='#eee')
        self.ax_anim.set_title('Cart-Pole System', fontsize=12, color='#eee')
        
        # Draw track
        self.track = self.ax_anim.axhline(y=0, color='#4a5568', linewidth=3)
        self.ax_anim.fill_between([-3, 3], [-0.1, -0.1], [0, 0], 
                                   color='#2d3748', alpha=0.5)
        
        # Cart (rectangle)
        self.cart_width, self.cart_height = 0.4, 0.2
        self.cart = Rectangle(
            (-self.cart_width/2, -self.cart_height/2), 
            self.cart_width, self.cart_height,
            facecolor='#e94560',
            edgecolor='#ff6b6b',
            linewidth=2
        )
        self.ax_anim.add_patch(self.cart)
        
        # Wheels
        wheel_radius = 0.05
        self.wheel1 = Circle((-0.12, -0.1), wheel_radius, 
                            facecolor='#333', edgecolor='#555', linewidth=2)
        self.wheel2 = Circle((0.12, -0.1), wheel_radius,
                            facecolor='#333', edgecolor='#555', linewidth=2)
        self.ax_anim.add_patch(self.wheel1)
        self.ax_anim.add_patch(self.wheel2)
        
        # Pendulum rod
        self.rod, = self.ax_anim.plot([0, 0], [0, 0.8], 
                                       color='#00d9ff', linewidth=4,
                                       solid_capstyle='round')
        
        # Pendulum tip mass
        self.tip = Circle((0, 0.8), 0.06, facecolor='#ffd700', 
                         edgecolor='#ffed4a', linewidth=2)
        self.ax_anim.add_patch(self.tip)
        
        # Force arrow
        self.force_arrow = self.ax_anim.annotate(
            '', xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=2)
        )
        
        # Info text
        self.info_text = self.ax_anim.text(
            0.02, 0.98, '', transform=self.ax_anim.transAxes,
            fontsize=10, verticalalignment='top',
            fontfamily='monospace', color='#eee',
            bbox=dict(boxstyle='round', facecolor='#16213e', alpha=0.8)
        )
        
        # Controller indicator
        self.controller_text = self.ax_anim.text(
            0.98, 0.98, '', transform=self.ax_anim.transAxes,
            fontsize=11, fontweight='bold',
            verticalalignment='top', horizontalalignment='right',
            color='#00ff88',
            bbox=dict(boxstyle='round', facecolor='#16213e', alpha=0.8, edgecolor='#00ff88')
        )
        
    def _setup_data_plots(self):
        """Setup the rolling data plots."""
        # Theta plot
        self.ax_theta = self.fig.add_subplot(self.gs[1, 0])
        self.ax_theta.set_facecolor('#16213e')
        self.ax_theta.set_xlim(0, 10)
        self.ax_theta.set_ylim(-0.5, 0.5)
        self.ax_theta.set_xlabel('Time (s)', color='#eee')
        self.ax_theta.set_ylabel('Angle (rad)', color='#eee')
        self.ax_theta.set_title('Pendulum Angle θ', fontsize=11, color='#eee')
        self.ax_theta.grid(True, alpha=0.2)
        self.ax_theta.axhline(y=0, color='#4a5568', linestyle='--', alpha=0.5)
        
        self.line_theta_meas, = self.ax_theta.plot([], [], 'o', markersize=1,
                                                    color='#ff6b6b', alpha=0.4,
                                                    label='Measured')
        self.line_theta_est, = self.ax_theta.plot([], [], '-', linewidth=2,
                                                   color='#00ff88',
                                                   label='Estimated')
        self.ax_theta.legend(loc='upper right', fontsize=8)
        
        # Position plot
        self.ax_x = self.fig.add_subplot(self.gs[1, 1])
        self.ax_x.set_facecolor('#16213e')
        self.ax_x.set_xlim(0, 10)
        self.ax_x.set_ylim(-1.5, 1.5)
        self.ax_x.set_xlabel('Time (s)', color='#eee')
        self.ax_x.set_ylabel('Position (m)', color='#eee')
        self.ax_x.set_title('Cart Position x', fontsize=11, color='#eee')
        self.ax_x.grid(True, alpha=0.2)
        self.ax_x.axhline(y=0, color='#4a5568', linestyle='--', alpha=0.5)
        
        self.line_x_meas, = self.ax_x.plot([], [], 'o', markersize=1,
                                            color='#ff6b6b', alpha=0.4,
                                            label='Measured')
        self.line_x_est, = self.ax_x.plot([], [], '-', linewidth=2,
                                           color='#00ff88',
                                           label='Estimated')
        self.ax_x.legend(loc='upper right', fontsize=8)
        
    def _setup_widgets(self):
        """Setup interactive widgets (sliders, buttons)."""
        # Slider axes
        ax_kp = self.fig.add_axes([0.15, 0.08, 0.2, 0.02])
        ax_ki = self.fig.add_axes([0.15, 0.05, 0.2, 0.02])
        ax_kd = self.fig.add_axes([0.15, 0.02, 0.2, 0.02])
        
        # PID Sliders
        self.slider_kp = Slider(ax_kp, 'Kp', 0, 300, valinit=150, 
                                color='#e94560', valfmt='%.0f')
        self.slider_ki = Slider(ax_ki, 'Ki', 0, 5, valinit=0.5,
                                color='#e94560', valfmt='%.2f')
        self.slider_kd = Slider(ax_kd, 'Kd', 0, 50, valinit=20,
                                color='#e94560', valfmt='%.0f')
        
        # Slider callbacks
        self.slider_kp.on_changed(self._update_pid_gains)
        self.slider_ki.on_changed(self._update_pid_gains)
        self.slider_kd.on_changed(self._update_pid_gains)
        
        # Controller selection radio buttons
        ax_radio = self.fig.add_axes([0.55, 0.02, 0.1, 0.08])
        ax_radio.set_facecolor('#16213e')
        self.radio = RadioButtons(ax_radio, ('LQR', 'PID'), active=0)
        self.radio.on_clicked(self._on_controller_change)
        
        # Style radio buttons (compatible with different matplotlib versions)
        try:
            # For older matplotlib versions
            for circle in self.radio.circles:
                circle.set_facecolor('#00ff88')
        except AttributeError:
            # For newer matplotlib versions (3.7+)
            pass
        for label in self.radio.labels:
            label.set_color('#eee')
            label.set_fontsize(10)
        
        # Reset button
        ax_reset = self.fig.add_axes([0.75, 0.02, 0.08, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset', color='#16213e', hovercolor='#e94560')
        self.btn_reset.label.set_color('#eee')
        self.btn_reset.on_clicked(self._on_reset)
        
        # Pause button
        ax_pause = self.fig.add_axes([0.85, 0.02, 0.08, 0.04])
        self.btn_pause = Button(ax_pause, 'Pause', color='#16213e', hovercolor='#e94560')
        self.btn_pause.label.set_color('#eee')
        self.btn_pause.on_clicked(self._on_pause)
        
        # Instructions text
        self.fig.text(0.02, 0.12, 
                     '← → : Apply disturbance force | Space: Pause | R: Reset',
                     fontsize=9, color='#888', fontfamily='monospace')
        
    def _setup_keyboard(self):
        """Setup keyboard event handling."""
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
    def _update_pid_gains(self, val):
        """Update PID controller gains from sliders."""
        self.sim.pid_controller.set_gains(
            Kp=self.slider_kp.val,
            Ki=self.slider_ki.val,
            Kd=self.slider_kd.val
        )
        
    def _on_controller_change(self, label):
        """Switch controller type."""
        self.sim.set_controller(label)
        
    def _on_reset(self, event):
        """Reset the simulation."""
        self.sim.reset(np.array([0.0, 0.0, 0.15, 0.0]))
        
    def _on_pause(self, event):
        """Toggle pause state."""
        self.paused = not self.paused
        self.btn_pause.label.set_text('Resume' if self.paused else 'Pause')
        
    def _on_key_press(self, event):
        """Handle keyboard input."""
        if event.key == 'left':
            self.sim.apply_disturbance(-15.0)
        elif event.key == 'right':
            self.sim.apply_disturbance(15.0)
        elif event.key == 'up':
            self.sim.apply_disturbance(-25.0)
        elif event.key == 'down':
            self.sim.apply_disturbance(25.0)
        elif event.key == ' ':
            self._on_pause(None)
        elif event.key == 'r':
            self._on_reset(None)
            
    def init_animation(self):
        """Initialize animation artists."""
        self.line_theta_meas.set_data([], [])
        self.line_theta_est.set_data([], [])
        self.line_x_meas.set_data([], [])
        self.line_x_est.set_data([], [])
        return (self.cart, self.wheel1, self.wheel2, self.rod, self.tip,
                self.line_theta_meas, self.line_theta_est,
                self.line_x_meas, self.line_x_est,
                self.info_text, self.controller_text)
    
    def update(self, frame):
        """Update animation frame."""
        if not self.paused:
            # Step simulation
            true_state, est_state, control = self.sim.step()
            
            x, x_dot, theta, theta_dot = true_state
            
            # Update cart position
            self.cart.set_xy((x - self.cart_width/2, -self.cart_height/2))
            self.wheel1.center = (x - 0.12, -0.1)
            self.wheel2.center = (x + 0.12, -0.1)
            
            # Update pendulum (theta=0 is upright)
            rod_x = x + self.sim.system.L * np.sin(theta)
            rod_y = self.sim.system.L * np.cos(theta)
            self.rod.set_data([x, rod_x], [0, rod_y])
            self.tip.center = (rod_x, rod_y)
            
            # Update info text
            info = (f"t = {self.sim.time:.2f}s\n"
                   f"x = {x:.3f} m\n"
                   f"θ = {np.degrees(theta):.1f}°\n"
                   f"F = {control:.1f} N")
            self.info_text.set_text(info)
            
            # Update controller text
            self.controller_text.set_text(f"Controller: {self.sim.active_controller}")
            
            # Update data plots
            times = list(self.sim.time_history)
            if len(times) > 1:
                # Theta plot
                self.line_theta_meas.set_data(times, list(self.sim.theta_meas_history))
                self.line_theta_est.set_data(times, list(self.sim.theta_est_history))
                
                # Position plot
                self.line_x_meas.set_data(times, list(self.sim.x_meas_history))
                self.line_x_est.set_data(times, list(self.sim.x_est_history))
                
                # Adjust x-axis limits for rolling plot
                t_max = max(times)
                t_min = max(0, t_max - 10)
                self.ax_theta.set_xlim(t_min, t_max + 0.5)
                self.ax_x.set_xlim(t_min, t_max + 0.5)
                
                # Adjust y-axis for position if needed
                x_vals = list(self.sim.x_est_history)
                if x_vals:
                    x_range = max(abs(min(x_vals)), abs(max(x_vals)), 0.5)
                    self.ax_x.set_ylim(-x_range * 1.2, x_range * 1.2)
        
        return (self.cart, self.wheel1, self.wheel2, self.rod, self.tip,
                self.line_theta_meas, self.line_theta_est,
                self.line_x_meas, self.line_x_est,
                self.info_text, self.controller_text)
    
    def run(self):
        """Start the animation."""
        self.anim = FuncAnimation(
            self.fig, self.update,
            init_func=self.init_animation,
            frames=None,
            interval=20,  # 50 Hz
            blit=False,  # Disabled for better compatibility with widgets
            cache_frame_data=False
        )
        plt.show()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Run the inverted pendulum simulation."""
    print("=" * 60)
    print("Mobile Inverted Pendulum Simulation")
    print("=" * 60)
    print("\nControls:")
    print("  Arrow Keys : Apply disturbance force to cart")
    print("  Space      : Pause/Resume simulation")
    print("  R          : Reset simulation")
    print("\nSliders adjust PID gains when PID controller is active.")
    print("Use radio buttons to switch between LQR and PID controllers.")
    print("-" * 60)
    
    try:
        # Create simulation with 50Hz update rate
        sim = Simulation(dt=0.02)
        
        # Initialize with slight angle offset to make it interesting
        sim.reset(np.array([0.0, 0.0, 0.15, 0.0]))
        
        # Create and run visualization
        viz = PendulumVisualizer(sim)
        viz.run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
