"""Test controller robustness: max initial angle and max disturbance."""
import numpy as np
from dataclasses import dataclass
from cart_pole import CartPole
from motor import MotorModel
from state_filter import NoisyStateProcessor
from controller import PIDController, LQRController, PolePlacementController


@dataclass
class RobustnessResult:
    controller_name: str
    max_initial_angle_deg: float
    max_disturbance_N: float


def test_stability(
    cart_pole: CartPole,
    controller,
    motor: MotorModel,
    state_processor: NoisyStateProcessor,
    initial_angle_deg: float,
    disturbance_N: float = 0.0,
    disturbance_time: float = 2.0,
    dt: float = 0.02,
    duration: float = 10.0
) -> bool:
    """
    Test if controller can stabilize from given initial angle with disturbance.
    Returns True if pendulum stays upright (|theta| < 90 deg throughout).
    """
    n_steps = int(duration / dt)
    state = np.array([0.0, 0.0, np.deg2rad(initial_angle_deg), 0.0])
    filtered_state = state.copy()
    
    state_processor.reset()
    if hasattr(controller, 'reset'):
        controller.reset()
    
    disturbance_applied = False
    disturbance_step = int(disturbance_time / dt)
    
    t = 0.0
    for i in range(n_steps):
        # Check if pendulum has fallen
        if abs(state[2]) > np.pi / 2:  # > 90 degrees
            return False
        
        cart_velocity = state[1]
        
        # Compute control
        if controller is None:
            desired_force = 0.0
        else:
            desired_force = controller.compute(filtered_state, t)
        
        # Motor model
        motor_out = motor.compute_motor_output(desired_force, cart_velocity)
        actual_force = motor_out['actual_force']
        
        # Apply disturbance impulse at specified time
        if i == disturbance_step and disturbance_N != 0.0:
            actual_force += disturbance_N
            disturbance_applied = True
        
        # Integrate
        state_dot = cart_pole.dynamics(t, state, actual_force)
        state = state + state_dot * dt
        
        # Filter
        _, filtered_state = state_processor.process(state)
        
        t += dt
    
    # Check final stability: angle should be small and not growing
    return abs(state[2]) < np.deg2rad(30)  # Final angle < 30 deg


def find_max_initial_angle(
    cart_pole: CartPole,
    controller,
    motor: MotorModel,
    state_processor: NoisyStateProcessor,
    min_angle: float = 5.0,
    max_angle: float = 90.0,
    tolerance: float = 1.0
) -> float:
    """Binary search to find maximum initial angle controller can handle."""
    low, high = min_angle, max_angle
    
    while high - low > tolerance:
        mid = (low + high) / 2
        
        # Reset for each test
        state_processor.reset()
        if hasattr(controller, 'reset'):
            controller.reset()
        
        if test_stability(cart_pole, controller, motor, state_processor, mid):
            low = mid  # Can handle this angle, try higher
        else:
            high = mid  # Failed, try lower
    
    return low


def find_max_disturbance(
    cart_pole: CartPole,
    controller,
    motor: MotorModel,
    state_processor: NoisyStateProcessor,
    initial_angle_deg: float = 5.0,
    min_dist: float = 0.0,
    max_dist: float = 50.0,
    tolerance: float = 0.5
) -> float:
    """Binary search to find maximum disturbance controller can handle."""
    low, high = min_dist, max_dist
    
    while high - low > tolerance:
        mid = (low + high) / 2
        
        # Reset for each test
        state_processor.reset()
        if hasattr(controller, 'reset'):
            controller.reset()
        
        if test_stability(cart_pole, controller, motor, state_processor, 
                         initial_angle_deg, disturbance_N=mid):
            low = mid  # Can handle this disturbance, try higher
        else:
            high = mid  # Failed, try lower
    
    return low


def run_robustness_tests():
    """Run robustness tests for all controllers."""
    print("=" * 60)
    print("Controller Robustness Testing")
    print("=" * 60)
    
    # Setup
    cart_pole = CartPole(
        cart_mass=1.0, pendulum_mass=0.05, rod_length=0.8,
        cart_friction=0.1, rotational_damping=0.01, gravity=9.81
    )
    
    motor = MotorModel(
        num_motors=4, wheel_radius=0.03,
        voltage_min=3.0, voltage_max=9.0,
        rpm_at_nominal=90.0, voltage_nominal=4.5
    )
    
    state_processor = NoisyStateProcessor(
        position_noise_std=0.005, angle_noise_std=0.01,
        tau_position=0.1, tau_angle=0.08, dt=0.02
    )
    
    # Controllers (same as generate_plots.py)
    pid = PIDController(kp=35.0, ki=0.5, kd=12.0, kp_pos=1.0, ki_pos=0.02, 
                        kd_pos=2.5, x_target=2.0, max_angle_setpoint=0.12)
    lqr = LQRController(
        cart_mass=cart_pole.M, pendulum_mass=cart_pole.m,
        rod_length=cart_pole.L, cart_friction=cart_pole.b,
        rotational_damping=cart_pole.c, gravity=cart_pole.g,
        Q=np.diag([8.0, 3.0, 50.0, 5.0]),
        R=np.array([[0.3]])
    )
    pole_placement = PolePlacementController(
        cart_mass=cart_pole.M, pendulum_mass=cart_pole.m,
        rod_length=cart_pole.L, cart_friction=cart_pole.b,
        rotational_damping=cart_pole.c, gravity=cart_pole.g,
        poles=np.array([-2.0, -2.5, -3.0, -3.5])
    )
    
    controllers = [
        (pid, 'PID'),
        (lqr, 'LQR'),
        (pole_placement, 'Pole Placement')
    ]
    
    results = []
    
    for controller, name in controllers:
        print(f"\nTesting {name}...")
        
        # Test max initial angle
        print("  Finding max initial angle...", end=" ", flush=True)
        max_angle = find_max_initial_angle(
            cart_pole, controller, motor, state_processor,
            min_angle=5.0, max_angle=60.0, tolerance=1.0
        )
        print(f"{max_angle:.1f}°")
        
        # Test max disturbance (from stable position with 5° initial angle)
        print("  Finding max disturbance...", end=" ", flush=True)
        max_dist = find_max_disturbance(
            cart_pole, controller, motor, state_processor,
            initial_angle_deg=5.0, min_dist=0.0, max_dist=100.0, tolerance=0.5
        )
        print(f"{max_dist:.1f} N")
        
        results.append(RobustnessResult(name, max_angle, max_dist))
    
    # Print summary
    print("\n" + "=" * 60)
    print("ROBUSTNESS TEST RESULTS")
    print("=" * 60)
    print(f"{'Controller':<20} {'Max Initial Angle':<20} {'Max Disturbance':<20}")
    print("-" * 60)
    for r in results:
        print(f"{r.controller_name:<20} {r.max_initial_angle_deg:>15.1f}° {r.max_disturbance_N:>17.1f} N")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = run_robustness_tests()
