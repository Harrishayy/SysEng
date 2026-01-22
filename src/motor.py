"""DC Motor model for cart-pole system."""
import numpy as np


class MotorModel:
    """
    DC Motor model converting desired force to actual motor output.
    
    4 motors drive the cart wheels. Each motor:
    - Voltage range: 3V-9V (below 3V = stall)
    - Speed: 90 RPM at 4.5V (after 1:48 gear reduction)
    - Wheel radius: 0.03m
    """
    
    def __init__(
        self,
        num_motors: int = 4,
        wheel_radius: float = 0.03,
        voltage_min: float = 3.0,
        voltage_max: float = 9.0,
        rpm_at_nominal: float = 90.0,
        voltage_nominal: float = 4.5,
        motor_resistance: float = 2.0,  # Ohms (typical for small DC motor)
        torque_constant: float = 0.01   # Nm/A (estimated)
    ):
        self.num_motors = num_motors
        self.wheel_radius = wheel_radius
        self.voltage_min = voltage_min
        self.voltage_max = voltage_max
        
        # Motor characteristics
        self.rpm_per_volt = rpm_at_nominal / voltage_nominal  # 20 RPM/V
        self.R = motor_resistance
        self.Kt = torque_constant  # Torque constant
        self.Ke = torque_constant  # Back-EMF constant (equal to Kt in SI units)
        
        # Max speed at max voltage
        self.max_rpm = self.rpm_per_volt * voltage_max  # 180 RPM
        self.max_omega = self.max_rpm * 2 * np.pi / 60   # rad/s
        self.max_speed = self.max_omega * wheel_radius   # m/s
        
        # Stall torque per motor (at max voltage, zero speed)
        # τ_stall = Kt * V_max / R
        self.stall_torque = self.Kt * voltage_max / self.R
        
        # Max force (all motors at stall)
        self.max_force = self.num_motors * self.stall_torque / wheel_radius
    
    def compute_motor_output(self, desired_force: float, cart_velocity: float) -> dict:
        """
        Convert desired force to motor voltage and compute actual force.
        
        DC Motor model:
        - Torque = Kt * I = Kt * (V - Ke*ω) / R
        - Force = num_motors * Torque / wheel_radius
        
        Args:
            desired_force: Force requested by controller (N)
            cart_velocity: Current cart velocity (m/s)
            
        Returns:
            dict with voltage, actual_force, saturated flag
        """
        # Preserve sign of desired force
        force_sign = np.sign(desired_force) if desired_force != 0 else 1.0
        
        # Wheel angular velocity from cart velocity
        omega_wheel = cart_velocity / self.wheel_radius  # rad/s
        
        # Back-EMF voltage (opposes motion)
        V_back_emf = self.Ke * abs(omega_wheel)
        
        # Required torque per motor
        desired_torque_total = abs(desired_force) * self.wheel_radius
        desired_torque_per_motor = desired_torque_total / self.num_motors
        
        # Required current per motor: I = τ / Kt
        desired_current = desired_torque_per_motor / self.Kt
        
        # Required voltage: V = I*R + V_back_emf
        # Sign of back-EMF depends on whether we're accelerating or braking
        if force_sign * cart_velocity >= 0:
            # Accelerating in direction of motion or starting from rest
            required_voltage = desired_current * self.R + V_back_emf
        else:
            # Braking (force opposes velocity)
            required_voltage = desired_current * self.R - V_back_emf
        
        # Apply voltage limits
        saturated = False
        if required_voltage > self.voltage_max:
            applied_voltage = self.voltage_max
            saturated = True
        elif required_voltage < self.voltage_min:
            # Below minimum voltage - motor stalls
            if abs(desired_force) < 0.1:  # Near zero force requested
                applied_voltage = 0.0
            else:
                applied_voltage = self.voltage_min
                saturated = True
        else:
            applied_voltage = required_voltage
        
        # Compute actual current and torque at applied voltage
        if applied_voltage < self.voltage_min:
            actual_force = 0.0
        else:
            # Actual current depends on back-EMF direction
            if force_sign * cart_velocity >= 0:
                actual_current = (applied_voltage - V_back_emf) / self.R
            else:
                actual_current = (applied_voltage + V_back_emf) / self.R
            
            actual_current = max(0, actual_current)  # Current can't be negative
            actual_torque_per_motor = self.Kt * actual_current
            actual_force = self.num_motors * actual_torque_per_motor / self.wheel_radius
        
        # Apply original force sign
        actual_force = force_sign * actual_force
        
        return {
            'voltage': applied_voltage,
            'actual_force': actual_force,
            'desired_force': desired_force,
            'saturated': saturated,
            'back_emf': V_back_emf
        }
    
    def get_specs(self) -> dict:
        """Return motor specifications."""
        return {
            'num_motors': self.num_motors,
            'wheel_radius': self.wheel_radius,
            'voltage_range': (self.voltage_min, self.voltage_max),
            'max_speed': self.max_speed,
            'max_force': self.max_force,
            'stall_torque_per_motor': self.stall_torque
        }
