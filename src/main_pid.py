"""
Main entry point for controlled cart-pole simulation with PID controller.

Run this file to see the cart-pole stabilized at the upright position.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from cart_pole import CartPole
from simulator import Simulator
from visualizer import Visualizer
from controller import PIDController
from state_filter import NoisyStateProcessor


def main():
    # Create the cart-pole system with physical parameters
    cart_pole = CartPole(
        cart_mass=1.0,          # kg
        pendulum_mass=0.05,     # kg
        rod_length=0.8,         # m
        cart_friction=0.1,      # N/m/s
        rotational_damping=0.01, # N*m/rad/s
        gravity=9.81            # m/s^2
    )
    
    # Simulation parameters
    dt = 0.02  # 50 Hz sampling
    
    # Create simulator
    simulator = Simulator(cart_pole)
    
    # Create state processor (noise + filtering)
    state_processor = NoisyStateProcessor(
        position_noise_std=0.005,   # 5mm position noise
        angle_noise_std=0.01,       # ~0.6 deg angle noise
        tau_position=0.05,          # Position filter time constant (s)
        tau_angle=0.02,             # Angle filter time constant (s)
        dt=dt,
        seed=42                     # For reproducibility
    )
    
    # Create PID controller
    # Tuned gains for this system
    controller = PIDController(
        kp=100,    # Proportional gain
        ki=0.5,     # Integral gain (small to prevent windup)
        kd=20.0,    # Derivative gain
        setpoint=0.0  # Target: upright position
    )
    
    # Print filter parameters
    params = state_processor.filter.get_parameters()
    print("State Filter Parameters:")
    print(f"  Position filter: τ = {params['tau_position']:.3f}s, α = {params['alpha_position']:.4f}")
    print(f"  Angle filter:    τ = {params['tau_angle']:.3f}s, α = {params['alpha_angle']:.4f}")
    print()
    
    # Define initial conditions
    # Starting with a moderate angle offset to test controller
    initial_state = np.array([
        0.0,    # x: cart position (m)
        0.0,    # x_dot: cart velocity (m/s)
        np.deg2rad(15.0),    # theta: pendulum angle (rad) - about 15 degrees
        0.0     # theta_dot: angular velocity (rad/s)
    ])
    
    print(f"Initial angle: {np.rad2deg(initial_state[2]):.1f} degrees")
    print(f"Target angle: {np.rad2deg(controller.setpoint):.1f} degrees")
    
    # Run controlled simulation with noise
    print("Running PID-controlled simulation with measurement noise...")
    result = simulator.run_with_noise(
        initial_state=initial_state,
        duration=10.0,
        dt=dt,
        controller=controller,
        state_processor=state_processor
    )
    print(f"Simulation complete. {len(result.time)} timesteps.")
    
    # Create plots directory
    plots_dir = Path(__file__).parent.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Create visualizer
    visualizer = Visualizer(cart_pole)
    
    # Plot state trajectories with noise comparison
    print("Generating state plots...")
    fig = visualizer.plot_states_with_noise(
        result,
        save_path=plots_dir / "pid_states.png"
    )
    fig.suptitle('PID Controlled Cart-Pole (with noise)', fontsize=14, y=1.0)
    
    # Calculate and plot control force (using filtered states as controller sees them)
    print("Generating control force plot...")
    fig2, ax = plt.subplots(figsize=(10, 3))
    forces = []
    controller.reset()  # Reset controller state
    for i in range(len(result.time)):
        t = result.time[i]
        # Use filtered states (what controller actually received)
        theta = result.filtered_states[2, i]
        theta_dot = result.filtered_states[3, i]
        force = controller.compute(theta, theta_dot, t)
        forces.append(force)
    
    ax.plot(result.time, forces, 'g-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control Force (N)')
    ax.set_title('PID Controller Output')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save control force plot
    force_plot_path = plots_dir / "pid_control_force.png"
    fig2.savefig(force_plot_path, dpi=300, bbox_inches='tight')
    print(f"Control force plot saved to {force_plot_path}")
    
    # Create and display animation
    print("Creating animation...")
    anim = visualizer.animate(result, interval=20)
    
    # Show all plots and animation
    visualizer.show()


if __name__ == "__main__":
    main()