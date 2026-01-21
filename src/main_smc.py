"""
Main entry point for SMC-controlled cart-pole simulation.

Run this file to see the cart-pole stabilized using Sliding Mode Control.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from cart_pole import CartPole
from simulator import Simulator
from visualizer import Visualizer
from controller import SMCController
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
    
    # Create SMC controller
    controller = SMCController(
        cart_mass=cart_pole.M,
        pendulum_mass=cart_pole.m,
        rod_length=cart_pole.L,
        cart_friction=cart_pole.b,
        rotational_damping=cart_pole.c,
        gravity=cart_pole.g,
        lambda_=10.0,        # Sliding surface slope
        eta=20.0,            # Switching gain
        phi=0.1,             # Boundary layer thickness
        use_saturation=True  # Use smooth saturation instead of sign()
    )
    
    # Print SMC parameters
    params = controller.get_parameters()
    print("SMC Parameters:")
    print(f"  lambda (surface slope) = {params['lambda']}")
    print(f"  eta (switching gain)   = {params['eta']}")
    print(f"  phi (boundary layer)   = {params['phi']}")
    print(f"  use_saturation         = {params['use_saturation']}")
    
    # Print filter parameters
    filter_params = state_processor.filter.get_parameters()
    print("\nState Filter Parameters:")
    print(f"  Position filter: τ = {filter_params['tau_position']:.3f}s, α = {filter_params['alpha_position']:.4f}")
    print(f"  Angle filter:    τ = {filter_params['tau_angle']:.3f}s, α = {filter_params['alpha_angle']:.4f}")
    print()
    
    # Define initial conditions
    # Starting with a moderate angle offset to test controller
    initial_state = np.array([
        0.0,    # x: cart position (m)
        0.0,    # x_dot: cart velocity (m/s)
        0.3,    # theta: pendulum angle (rad) - about 17 degrees
        0.0     # theta_dot: angular velocity (rad/s)
    ])
    
    print(f"Initial angle: {np.rad2deg(initial_state[2]):.1f} degrees")
    
    # Run controlled simulation with noise
    print("Running SMC-controlled simulation with measurement noise...")
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
        save_path=plots_dir / "smc_states.png"
    )
    fig.suptitle('SMC Controlled Cart-Pole (with noise)', fontsize=14, y=1.0)
    
    # Calculate control force and sliding surface (using filtered states)
    print("Generating control force and sliding surface plots...")
    forces = []
    sliding_surface = []
    for i in range(len(result.time)):
        t = result.time[i]
        # Use filtered states (what controller actually received)
        state = result.filtered_states[:, i]
        force = controller.compute(state, t)
        forces.append(force)
        sliding_surface.append(controller.get_sliding_surface(state))
    
    # Create figure with two subplots
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Control force plot
    ax1.plot(result.time, forces, 'g-', linewidth=1.5)
    ax1.set_ylabel('Control Force (N)')
    ax1.set_title('SMC Controller Output')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    
    # Sliding surface plot
    ax2.plot(result.time, sliding_surface, 'm-', linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Sliding Surface s')
    ax2.set_title('Sliding Surface (s = λθ + θ̇)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    # Show boundary layer
    ax2.axhline(controller.phi, color='r', linestyle=':', alpha=0.5, label=f'Boundary layer (±{controller.phi})')
    ax2.axhline(-controller.phi, color='r', linestyle=':', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plots
    force_plot_path = plots_dir / "smc_control_force.png"
    fig2.savefig(force_plot_path, dpi=300, bbox_inches='tight')
    print(f"Control force plot saved to {force_plot_path}")
    
    # Create and display animation
    print("Creating animation...")
    anim = visualizer.animate(result, interval=20)
    
    # Show all plots and animation
    visualizer.show()


if __name__ == "__main__":
    main()
