"""
Main entry point for LQR-controlled cart-pole simulation.

Run this file to see the cart-pole stabilized using optimal LQR control.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from cart_pole import CartPole
from simulator import Simulator
from visualizer import Visualizer
from controller import LQRController
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
    
    # Create LQR controller with custom cost matrices
    # Q matrix: penalize deviations in [x, x_dot, theta, theta_dot]
    # Higher weights = stronger control on that state
    Q = np.diag([3.0, 0.5, 100.0, 10.0])  # Penalize cart position, angle most critical
    R = np.array([[0.1]])  # Control effort cost
    
    # Define desired setpoint: cart at x=2.0m, pendulum balanced
    setpoint = np.array([2.0, 0.0, 0.0, 0.0])  # [x_desired, x_dot_desired, theta_desired, theta_dot_desired]
    
    controller = LQRController(
        cart_mass=cart_pole.M,
        pendulum_mass=cart_pole.m,
        rod_length=cart_pole.L,
        cart_friction=cart_pole.b,
        rotational_damping=cart_pole.c,
        gravity=cart_pole.g,
        Q=Q,
        R=R,
        setpoint=setpoint
    )
    
    # Print LQR gains and setpoint
    gains = controller.get_gains()
    print("LQR Gains:")
    print(f"  K_x         = {gains['k_x']:.4f}")
    print(f"  K_x_dot     = {gains['k_x_dot']:.4f}")
    print(f"  K_theta     = {gains['k_theta']:.4f}")
    print(f"  K_theta_dot = {gains['k_theta_dot']:.4f}")
    print()
    print(f"Setpoint: {setpoint}")
    
    # Print filter parameters
    params = state_processor.filter.get_parameters()
    print("\nState Filter Parameters:")
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
    print(f"Initial position: {initial_state[0]:.2f} meters")
    
    # Define disturbance function (impulse forces at specific times)
    def disturbance_func(t):
        """
        Apply impulse disturbances at specific times.
        - t=3s: Push right (+20 N for 0.2s)
        - t=6s: Push left (-15 N for 0.2s)
        """
        if 3.0 <= t < 3.2:
            return 20.0  # Push right
        elif 6.0 <= t < 6.2:
            return -15.0  # Push left
        return 0.0
    
    print("\nDisturbances:")
    print("  t=3.0s: +20 N impulse (0.2s duration)")
    print("  t=6.0s: -15 N impulse (0.2s duration)")
    
    # Run controlled simulation with noise and disturbances
    print("\nRunning LQR-controlled simulation with noise and disturbances...")
    result = simulator.run_with_noise(
        initial_state=initial_state,
        duration=10.0,
        dt=dt,
        controller=controller,
        state_processor=state_processor,
        disturbance_func=disturbance_func
    )
    print(f"Simulation complete. {len(result.time)} timesteps.")
    
    # Create plots directory
    plots_dir = Path(__file__).parent.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Create visualizer
    visualizer = Visualizer(cart_pole)
    
    # Plot comprehensive view (states, forces, disturbances)
    print("Generating comprehensive plots...")
    fig = visualizer.plot_comprehensive(
        result,
        save_path=plots_dir / "lqr_states.png"
    )
    fig.suptitle('LQR Controlled Cart-Pole (with noise and disturbances)', fontsize=14, y=1.0)
    
    # Also save force plot separately
    fig2 = visualizer.plot_forces(
        result,
        save_path=plots_dir / "lqr_control_force.png"
    )
    
    # Create and display animation
    print("Creating animation...")
    anim = visualizer.animate(result, interval=20)
    
    # Show all plots and animation
    visualizer.show()


if __name__ == "__main__":
    main()
