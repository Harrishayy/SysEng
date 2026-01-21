"""
Main entry point for the cart-pole simulation.

Run this file to see the uncontrolled cart-pole simulation.
"""
import numpy as np
from pathlib import Path

from cart_pole import CartPole
from simulator import Simulator
from visualizer import Visualizer
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
    
    # Print filter parameters
    params = state_processor.filter.get_parameters()
    print("State Filter Parameters:")
    print(f"  Position filter: τ = {params['tau_position']:.3f}s, α = {params['alpha_position']:.4f}")
    print(f"  Angle filter:    τ = {params['tau_angle']:.3f}s, α = {params['alpha_angle']:.4f}")
    print()
    
    # Define initial conditions
    # Starting with a small angle offset to observe natural dynamics
    initial_state = np.array([
        0.0,    # x: cart position (m)
        0.0,    # x_dot: cart velocity (m/s)
        0.2,    # theta: pendulum angle (rad) - about 11.5 degrees
        0.0     # theta_dot: angular velocity (rad/s)
    ])
    
    # Run simulation with noise (no control force)
    print("Running simulation with measurement noise...")
    result = simulator.run_with_noise(
        initial_state=initial_state,
        duration=10.0,
        dt=dt,
        controller=None,
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
    visualizer.plot_states_with_noise(result, save_path=plots_dir / "uncontrolled_states.png")
    
    # Create and display animation
    print("Creating animation...")
    anim = visualizer.animate(result, interval=20)
    
    # Show all plots and animation
    visualizer.show()


if __name__ == "__main__":
    main()
