"""DC Motor model for cart-pole system."""
import numpy as np


class MotorModel:
    """
    DC Motor model converting desired force to actual motor output.

    Matches hardware in LQRPendulum.ino:
      4x Pololu #4862 (9.68:1 gearbox, 12 V nominal)
      Supply: 10.6 V (user measured)
      No-load speed @ 12 V: 800 RPM (gearbox output)
      Stall torque @ 12 V: 0.16671 N·m (gearbox output)
      Stall torque @ 10.6 V: 0.14727 N·m per motor
      Wheel radius: 0.04 m
      F_max = 4 x 0.14727 / 0.04 = 14.73 N
    """

    def __init__(
        self,
        num_motors: int = 4,
        wheel_radius: float = 0.04,
        voltage_min: float = 0.0,
        voltage_max: float = 10.6,
        rpm_at_nominal: float = 800.0,
        voltage_nominal: float = 12.0,
        motor_resistance: float = 10.31,   # Ω  (12 V / ~1.16 A stall)
        torque_constant: float = 0.14323   # N·m/A (gearbox output torque / stall current)
    ):
        self.num_motors = num_motors
        self.wheel_radius = wheel_radius
        self.voltage_min = voltage_min
        self.voltage_max = voltage_max

        # Motor characteristics
        self.rpm_per_volt = rpm_at_nominal / voltage_nominal
        self.R = motor_resistance
        self.Kt = torque_constant
        self.Ke = torque_constant  # Back-EMF constant (SI: equal to Kt)

        # Max speed at max voltage
        self.max_rpm = self.rpm_per_volt * voltage_max
        self.max_omega = self.max_rpm * 2 * np.pi / 60
        self.max_speed = self.max_omega * wheel_radius

        # Stall torque per motor at max voltage
        self.stall_torque = self.Kt * voltage_max / self.R

        # Max force (all motors at stall)
        self.max_force = self.num_motors * self.stall_torque / wheel_radius

    def compute_motor_output(self, desired_force: float, cart_velocity: float) -> dict:
        """
        Convert desired force to motor voltage and compute actual force.

        DC Motor model:
          Torque = Kt * I = Kt * (V - Ke*ω) / R
          Force  = num_motors * Torque / wheel_radius

        Args:
            desired_force: Force requested by controller (N)
            cart_velocity: Current cart velocity (m/s)

        Returns:
            dict with voltage, actual_force, saturated flag
        """
        force_sign = np.sign(desired_force) if desired_force != 0 else 1.0

        omega_wheel = cart_velocity / self.wheel_radius
        V_back_emf = self.Ke * abs(omega_wheel)

        desired_torque_total = abs(desired_force) * self.wheel_radius
        desired_torque_per_motor = desired_torque_total / self.num_motors

        desired_current = desired_torque_per_motor / self.Kt

        if force_sign * cart_velocity >= 0:
            required_voltage = desired_current * self.R + V_back_emf
        else:
            required_voltage = desired_current * self.R - V_back_emf

        saturated = False
        if required_voltage > self.voltage_max:
            applied_voltage = self.voltage_max
            saturated = True
        elif required_voltage < self.voltage_min:
            if abs(desired_force) < 0.1:
                applied_voltage = 0.0
            else:
                applied_voltage = self.voltage_min
                saturated = (self.voltage_min > 0.0)
        else:
            applied_voltage = required_voltage

        if applied_voltage <= self.voltage_min and self.voltage_min == 0.0:
            if abs(desired_force) < 0.1:
                actual_force = 0.0
            else:
                if force_sign * cart_velocity >= 0:
                    actual_current = (applied_voltage - V_back_emf) / self.R
                else:
                    actual_current = (applied_voltage + V_back_emf) / self.R
                actual_current = max(0, actual_current)
                actual_torque_per_motor = self.Kt * actual_current
                actual_force = self.num_motors * actual_torque_per_motor / self.wheel_radius
        else:
            if force_sign * cart_velocity >= 0:
                actual_current = (applied_voltage - V_back_emf) / self.R
            else:
                actual_current = (applied_voltage + V_back_emf) / self.R
            actual_current = max(0, actual_current)
            actual_torque_per_motor = self.Kt * actual_current
            actual_force = self.num_motors * actual_torque_per_motor / self.wheel_radius

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
