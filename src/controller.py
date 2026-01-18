"""
Controller classes for cart-pole stabilization.
"""
import numpy as np


class PIDController:
    """
    PID controller for cart-pole stabilization.
    
    The controller computes the force to apply to the cart to keep
    the pendulum upright (theta = 0).
    """
    
    def __init__(
        self,
        kp: float = 50.0,
        ki: float = 0.0,
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
        self.previous_error = 0.0
        self.previous_time = None
    
    def compute(self, theta: float, theta_dot: float, t: float) -> float:
        """
        Compute the control force based on current state.
        
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
            self.previous_error = error
            dt = 0.01  # Small default dt
        else:
            dt = t - self.previous_time
            if dt <= 0:
                dt = 0.01
        
        # Proportional term
        P = self.kp * error
        
        # Integral term (with anti-windup)
        self.integral += error * dt
        # Clamp integral to prevent windup
        max_integral = 100.0
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        I = self.ki * self.integral
        
        # Derivative term (using angular velocity is more stable than numerical derivative)
        D = self.kd * theta_dot
        
        # Control force (negative because we want to oppose the lean)
        # If theta > 0 (leaning right), we need force < 0 (push left) to catch it
        force = P + I + D
        
        # Update for next iteration
        self.previous_error = error
        self.previous_time = t
        
        # Optional: Add force limits to be realistic
        max_force = 100.0  # N
        force = np.clip(force, -max_force, max_force)
        
        return force
    
    def reset(self):
        """Reset the controller's internal state."""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None