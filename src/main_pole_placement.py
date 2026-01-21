"""
Main entry point for Pole Placement-controlled cart-pole simulation.

Run this file to see the cart-pole stabilized using pole placement control.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from cart_pole import CartPole
from simulator import Simulator
from visualizer import Visualizer
from controller import PolePlacementController
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
    
    # Define desired closed-loop poles
    # All poles must be in left half-plane (negative real part) for stability
    # More negative = faster response but requires more control effort
    # Complex conjugate pairs give oscillatory response
    desired_poles = np.array([-3.0, -4.0, -5.0, -6.0])  # All real, stable
    
    # Alternative: complex conjugate pairs for slightly underdamped response
    # desired_poles = np.array([-2+1j, -2-1j, -4+1j, -4-1j])
    
    # Define desired setpoint
    setpoint = np.array([0.0, 0.0, 0.0, 0.0])  # Cart at origin, pendulum upright
    
    # Create Pole Placement controller
    controller = PolePlacementController(
        cart_mass=cart_pole.M,
        pendulum_mass=cart_pole.m,
        rod_length=cart_pole.L,
        cart_friction=cart_pole.b,
        rotational_damping=cart_pole.c,
        gravity=cart_pole.g,
        poles=desired_poles,
        setpoint=setpoint
    )
    
    # Print pole placement info
    gains = controller.get_gains()
    poles_info = controller.get_poles()
    
    print("Pole Placement Controller")
    print("=" * 40)
    print("\nDesired Poles:")
    for i, p in enumerate(poles_info['desired']):
        print(f"  p{i+1} = {p:.4f}")
    
    print("\nAchieved Poles:")
    for i, p in enumerate(poles_info['achieved']):
        if np.iscomplex(p) and np.imag(p) != 0:
            print(f"  p{i+1} = {np.real(p):.4f} + {np.imag(p):.4f}j")
        else:
            print(f"  p{i+1} = {np.real(p):.4f}")
    
    print("\nState Feedback Gains K:")
    print(f"  K_x         = {gains['k_x']:.4f}")
    print(f"  K_x_dot     = {gains['k_x_dot']:.4f}")
    print(f"  K_theta     = {gains['k_theta']:.4f}")
    print(f"  K_theta_dot = {gains['k_theta_dot']:.4f}")
    
    # Print filter parameters
    filter_params = state_processor.filter.get_parameters()
    print("\nState Filter Parameters:")
    print(f"  Position filter: τ = {filter_params['tau_position']:.3f}s, α = {filter_params['alpha_position']:.4f}")
    print(f"  Angle filter:    τ = {filter_params['tau_angle']:.3f}s, α = {filter_params['alpha_angle']:.4f}")
    print()
    
    # Define initial conditions
    initial_state = np.array([
        0.0,    # x: cart position (m)
        0.0,    # x_dot: cart velocity (m/s)
        0.3,    # theta: pendulum angle (rad) - about 17 degrees
        0.0     # theta_dot: angular velocity (rad/s)
    ])
    
    print(f"Initial angle: {np.rad2deg(initial_state[2]):.1f} degrees")
    print(f"Initial position: {initial_state[0]:.2f} meters")
    
    # Run controlled simulation with noise
    print("Running Pole Placement-controlled simulation with measurement noise...")
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
        save_path=plots_dir / "pole_placement_states.png"
    )
    fig.suptitle('Pole Placement Controlled Cart-Pole (with noise)', fontsize=14, y=1.0)
    
    # Calculate and plot control force
    print("Generating control force plot...")
    fig2, ax = plt.subplots(figsize=(10, 3))
    forces = []
    for i in range(len(result.time)):
        t = result.time[i]
        state = result.filtered_states[:, i]
        force = controller.compute(state, t)
        forces.append(force)
    
    ax.plot(result.time, forces, 'g-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control Force (N)')
    ax.set_title('Pole Placement Controller Output')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save control force plot
    force_plot_path = plots_dir / "pole_placement_control_force.png"
    fig2.savefig(force_plot_path, dpi=300, bbox_inches='tight')
    print(f"Control force plot saved to {force_plot_path}")
    
    # Create and display animation
    print("Creating animation...")
    anim = visualizer.animate(result, interval=20)
    
    # Show all plots and animation
    visualizer.show()


if __name__ == "__main__":
    main()
