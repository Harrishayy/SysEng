"""Output controller performance metrics."""
import numpy as np
from cart_pole import CartPole
from controller import PIDController, LQRController, PolePlacementController
from motor import MotorModel
from state_filter import NoisyStateProcessor

cart_pole = CartPole(
    cart_mass=1.2, pendulum_mass=0.91, rod_length=0.5,
    cart_friction=0.1, rotational_damping=0.01, gravity=9.81,
    rod_full_length=0.6
)
motor = MotorModel(
    num_motors=4, wheel_radius=0.04,
    voltage_min=0.0, voltage_max=10.6,
    rpm_at_nominal=800.0, voltage_nominal=12.0
)

def analyze(controller):
    sp = NoisyStateProcessor(position_noise_std=0.005, angle_noise_std=0.01,
                             dt=0.02, seed=42)
    state = np.array([0.0, 0.0, np.deg2rad(5.7), 0.0])
    times, positions = [], []
    t = 0.0
    for _ in range(500):
        times.append(t)
        positions.append(state[0])
        _, fs = sp.process(state)
        f = controller.compute(fs, t)
        mo = motor.compute_motor_output(f, state[1])
        state = cart_pole.apply_constraints(state + cart_pole.dynamics(t, state, mo['actual_force']) * 0.02)
        t += 0.02
    
    times, positions = np.array(times), np.array(positions)
    target = 2.0
    
    # Rise time (to 90% of target)
    rise_idx = np.where(positions >= 0.9 * target)[0]
    rise_time = times[rise_idx[0]] if len(rise_idx) > 0 else float('inf')
    
    # Max position and overshoot
    max_pos = np.max(positions)
    overshoot = max(0, max_pos - target)
    overshoot_pct = overshoot / target * 100
    
    # Settling time (within 5% of target)
    settled = np.abs(positions - target) < 0.05 * target
    settling_time = float('inf')
    for i in range(len(settled)):
        if np.all(settled[i:]):
            settling_time = times[i]
            break
    
    # Oscillation amplitude (std dev in last 2 seconds)
    last_2s = positions[-100:]  # last 100 samples at 0.02s = 2s
    oscillation = np.max(last_2s) - np.min(last_2s)
    
    return {
        'rise_time': rise_time,
        'max_pos': max_pos,
        'overshoot': overshoot,
        'overshoot_pct': overshoot_pct,
        'final_pos': positions[-1],
        'oscillation': oscillation,
        'settling_time': settling_time
    }

def print_results(name, r):
    print(f"{name}:")
    print(f"  Rise time (90%):   {r['rise_time']:.2f}s")
    print(f"  Max position:      {r['max_pos']:.3f}m")
    print(f"  Overshoot:         {r['overshoot']:.3f}m ({r['overshoot_pct']:.1f}%)")
    print(f"  Final position:    {r['final_pos']:.3f}m")
    print(f"  Oscillation (2s):  {r['oscillation']:.3f}m")
    if r['settling_time'] < 100:
        print(f"  Settling time (5%): {r['settling_time']:.2f}s")
    else:
        print(f"  Settling time (5%): N/A")
    print()

# Create controllers
pid = PIDController(x_target=2.0)
lqr = LQRController(
    cart_mass=cart_pole.M, pendulum_mass=cart_pole.m,
    rod_length=cart_pole.L, cart_friction=0.0,
    rotational_damping=0.0, gravity=cart_pole.g,
    setpoint=[2.0, 0.0, 0.0, 0.0]
)
pp = PolePlacementController(
    cart_mass=cart_pole.M, pendulum_mass=cart_pole.m,
    rod_length=cart_pole.L, cart_friction=0.0,
    rotational_damping=0.0, gravity=cart_pole.g,
    setpoint=[2.0, 0.0, 0.0, 0.0]
)

# Analyze and print
print()
print_results("PID", analyze(pid))
print_results("LQR", analyze(lqr))
print_results("Pole Placement", analyze(pp))
