"""Generate comparison plots for all control modes."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

from cart_pole import CartPole
from controller import PIDController, LQRController, PolePlacementController
from state_filter import NoisyStateProcessor
from motor import MotorModel


@dataclass
class SimulationResult:
    """Container for simulation results."""
    name: str
    time: np.ndarray
    states: np.ndarray           # (n_steps, 4)
    filtered_states: np.ndarray  # (n_steps, 4)
    desired_forces: np.ndarray
    actual_forces: np.ndarray
    voltages: np.ndarray


def run_simulation(
    cart_pole: CartPole,
    controller,
    controller_name: str,
    motor: MotorModel,
    state_processor: NoisyStateProcessor,
    initial_state: np.ndarray,
    dt: float = 0.02,
    duration: float = 10.0,
    use_motor: bool = True,
    use_noise: bool = True
) -> SimulationResult:
    """Run a single simulation with given controller."""
    
    n_steps = int(duration / dt)
    time = np.zeros(n_steps)
    states = np.zeros((n_steps, 4))
    filtered_states = np.zeros((n_steps, 4))
    desired_forces = np.zeros(n_steps)
    actual_forces = np.zeros(n_steps)
    voltages = np.zeros(n_steps)
    
    state = initial_state.copy()
    filtered_state = state.copy()
    state_processor.reset()
    
    # Reset PID if applicable
    if hasattr(controller, 'reset'):
        controller.reset()
    
    t = 0.0
    for i in range(n_steps):
        cart_velocity = state[1]
        
        # Compute desired force from controller
        if controller is None:
            desired_force = 0.0
        elif controller_name == 'PID':
            desired_force = controller.compute(filtered_state[2], filtered_state[3], t)
        else:
            desired_force = controller.compute(filtered_state, t)
        
        # Convert through motor model
        if use_motor:
            motor_out = motor.compute_motor_output(desired_force, cart_velocity)
            actual_force = motor_out['actual_force']
            voltage = motor_out['voltage']
        else:
            actual_force = desired_force
            voltage = 0.0
        
        # Integrate dynamics
        state_dot = cart_pole.dynamics(t, state, actual_force)
        state = state + state_dot * dt
        
        # Add noise and filter
        if use_noise:
            _, filtered_state = state_processor.process(state)
        else:
            filtered_state = state.copy()
        
        # Increment time and record (after step, same as main_interactive)
        t += dt
        time[i] = t
        states[i] = state.copy()
        filtered_states[i] = filtered_state.copy()
        desired_forces[i] = desired_force
        actual_forces[i] = actual_force
        voltages[i] = voltage
    
    return SimulationResult(
        name=controller_name,
        time=time,
        states=states,
        filtered_states=filtered_states,
        desired_forces=desired_forces,
        actual_forces=actual_forces,
        voltages=voltages
    )


def generate_comparison_plots(
    initial_angle_deg: float = 10.0,
    duration: float = 10.0,
    use_motor: bool = True,
    use_noise: bool = True,
    save_plots: bool = True
):
    """Generate comparison plots for all controllers."""
    
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
    
    # Controllers
    pid = PIDController(kp=100, ki=0.5, kd=20.0, setpoint=0.0)
    lqr = LQRController(
        cart_mass=cart_pole.M, pendulum_mass=cart_pole.m,
        rod_length=cart_pole.L, cart_friction=cart_pole.b,
        rotational_damping=cart_pole.c, gravity=cart_pole.g
    )
    pole_placement = PolePlacementController(
        cart_mass=cart_pole.M, pendulum_mass=cart_pole.m,
        rod_length=cart_pole.L, cart_friction=cart_pole.b,
        rotational_damping=cart_pole.c, gravity=cart_pole.g
    )
    
    controllers = [
        (None, 'Uncontrolled'),
        (pid, 'PID'),
        (lqr, 'LQR'),
        (pole_placement, 'Pole Placement')
    ]
    
    # Initial state
    initial_state = np.array([0.0, 0.0, np.deg2rad(initial_angle_deg), 0.0])
    
    # Run all simulations
    print(f"Running simulations (init angle: {initial_angle_deg}°, duration: {duration}s)")
    print(f"Motor model: {'ON' if use_motor else 'OFF'}, Noise: {'ON' if use_noise else 'OFF'}")
    print("-" * 50)
    
    results = []
    for ctrl, name in controllers:
        print(f"  Running {name}...")
        result = run_simulation(
            cart_pole, ctrl, name, motor, state_processor,
            initial_state, dt=0.02, duration=duration,
            use_motor=use_motor, use_noise=use_noise
        )
        results.append(result)
    
    print("-" * 50)
    
    # Colors for each controller
    colors = {
        'Uncontrolled': 'gray',
        'PID': 'blue',
        'LQR': 'green',
        'Pole Placement': 'red'
    }
    
    # Create plots directory
    plots_dir = Path(__file__).parent.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Plot 1: State comparison (2x2 grid)
    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle('State Comparison: All Controllers', fontsize=14)
    
    state_labels = ['Position (m)', 'Velocity (m/s)', 'Angle (deg)', 'Angular Velocity (deg/s)']
    angle_convert = [1, 1, np.rad2deg(1), np.rad2deg(1)]
    
    for idx, (ax, label, conv) in enumerate(zip(axes.flat, state_labels, angle_convert)):
        for result in results:
            ax.plot(result.time, result.states[:, idx] * conv,
                   color=colors[result.name], label=result.name, linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        if idx == 2:  # Angle plot
            ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        fig1.savefig(plots_dir / 'comparison_states.png', dpi=150)
        print(f"Saved: {plots_dir / 'comparison_states.png'}")
    
    # Plot 2: Control effort comparison
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig2.suptitle('Control Effort Comparison', fontsize=14)
    
    # Desired force
    ax_des = axes2[0]
    for result in results:
        if result.name != 'Uncontrolled':
            ax_des.plot(result.time, result.desired_forces,
                       color=colors[result.name], label=result.name, linewidth=1.5)
    ax_des.set_ylabel('Desired Force (N)')
    ax_des.grid(True, alpha=0.3)
    ax_des.legend(loc='best')
    ax_des.axhline(0, color='k', linestyle='--', alpha=0.3)
    
    # Actual force (motor output)
    ax_act = axes2[1]
    for result in results:
        if result.name != 'Uncontrolled':
            ax_act.plot(result.time, result.actual_forces,
                       color=colors[result.name], label=result.name, linewidth=1.5)
    ax_act.set_ylabel('Motor Force (N)')
    ax_act.grid(True, alpha=0.3)
    ax_act.legend(loc='best')
    ax_act.axhline(0, color='k', linestyle='--', alpha=0.3)
    
    # Voltage
    ax_volt = axes2[2]
    for result in results:
        if result.name != 'Uncontrolled':
            ax_volt.plot(result.time, result.voltages,
                        color=colors[result.name], label=result.name, linewidth=1.5)
    ax_volt.set_ylabel('Motor Voltage (V)')
    ax_volt.set_xlabel('Time (s)')
    ax_volt.grid(True, alpha=0.3)
    ax_volt.legend(loc='best')
    ax_volt.axhline(3.0, color='k', linestyle=':', alpha=0.5, label='V_min')
    ax_volt.axhline(9.0, color='k', linestyle=':', alpha=0.5, label='V_max')
    ax_volt.set_ylim(-0.5, 10)
    
    plt.tight_layout()
    if save_plots:
        fig2.savefig(plots_dir / 'comparison_control.png', dpi=150)
        print(f"Saved: {plots_dir / 'comparison_control.png'}")
    
    # Plot 3: Angle-only comparison (cleaner view)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.set_title('Pendulum Angle: Controller Comparison', fontsize=14)
    
    for result in results:
        ax3.plot(result.time, np.rad2deg(result.states[:, 2]),
                color=colors[result.name], label=result.name, linewidth=2)
    
    ax3.axhline(0, color='k', linestyle='--', alpha=0.3, label='Setpoint')
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Angle (degrees)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=11)
    ax3.set_xlim(0, duration)
    
    plt.tight_layout()
    if save_plots:
        fig3.savefig(plots_dir / 'comparison_angle.png', dpi=150)
        print(f"Saved: {plots_dir / 'comparison_angle.png'}")
    
    # Plot 4: Phase portrait (theta vs theta_dot)
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    ax4.set_title('Phase Portrait: Angle vs Angular Velocity', fontsize=14)
    
    for result in results:
        theta = np.rad2deg(result.states[:, 2])
        theta_dot = np.rad2deg(result.states[:, 3])
        ax4.plot(theta, theta_dot, color=colors[result.name], label=result.name, linewidth=1.5)
        # Mark start point
        ax4.plot(theta[0], theta_dot[0], 'o', color=colors[result.name], markersize=10)
    
    ax4.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax4.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Angle (degrees)', fontsize=12)
    ax4.set_ylabel('Angular Velocity (deg/s)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    if save_plots:
        fig4.savefig(plots_dir / 'comparison_phase.png', dpi=150)
        print(f"Saved: {plots_dir / 'comparison_phase.png'}")
    
    # Plot 5: Performance metrics
    fig5, axes5 = plt.subplots(1, 3, figsize=(15, 5))
    fig5.suptitle('Performance Metrics', fontsize=14)
    
    # Calculate metrics for controlled systems only
    controlled_results = [r for r in results if r.name != 'Uncontrolled']
    names = [r.name for r in controlled_results]
    
    # Settling time (time to stay within 2 degrees)
    settling_times = []
    for r in controlled_results:
        angles = np.abs(np.rad2deg(r.states[:, 2]))
        settled_idx = np.where(angles < 2)[0]
        if len(settled_idx) > 0:
            # Check if it stays settled
            first_settled = settled_idx[0]
            if np.all(angles[first_settled:] < 2):
                settling_times.append(r.time[first_settled])
            else:
                settling_times.append(r.time[-1])
        else:
            settling_times.append(r.time[-1])
    
    ax5a = axes5[0]
    bars1 = ax5a.bar(names, settling_times, color=[colors[n] for n in names])
    ax5a.set_ylabel('Settling Time (s)')
    ax5a.set_title('Settling Time (< 2°)')
    ax5a.grid(True, alpha=0.3, axis='y')
    
    # Max overshoot
    overshoots = []
    for r in controlled_results:
        angles = np.rad2deg(r.states[:, 2])
        # After first zero crossing
        zero_crossings = np.where(np.diff(np.sign(angles)))[0]
        if len(zero_crossings) > 0:
            after_cross = angles[zero_crossings[0]:]
            overshoot = np.max(np.abs(after_cross))
        else:
            overshoot = np.max(np.abs(angles))
        overshoots.append(overshoot)
    
    ax5b = axes5[1]
    bars2 = ax5b.bar(names, overshoots, color=[colors[n] for n in names])
    ax5b.set_ylabel('Max Overshoot (deg)')
    ax5b.set_title('Maximum Overshoot')
    ax5b.grid(True, alpha=0.3, axis='y')
    
    # Total control effort (integral of |force|)
    efforts = []
    for r in controlled_results:
        effort = np.trapezoid(np.abs(r.actual_forces), r.time)
        efforts.append(effort)
    
    ax5c = axes5[2]
    bars3 = ax5c.bar(names, efforts, color=[colors[n] for n in names])
    ax5c.set_ylabel('Control Effort (N·s)')
    ax5c.set_title('Total Control Effort')
    ax5c.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_plots:
        fig5.savefig(plots_dir / 'comparison_metrics.png', dpi=150)
        print(f"Saved: {plots_dir / 'comparison_metrics.png'}")
    
    print("-" * 50)
    print("All plots generated successfully!")
    
    return results


def main():
    """Main entry point."""
    print("=" * 60)
    print("Cart-Pole Control Comparison Plot Generator")
    print("=" * 60)
    print()
    
    # Generate plots with default settings
    results = generate_comparison_plots(
        initial_angle_deg=10.0,
        duration=10.0,
        use_motor=True,
        use_noise=True,
        save_plots=True
    )
    
    # Show all plots
    plt.show()


if __name__ == "__main__":
    main()
