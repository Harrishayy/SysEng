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
    
    # Create simulator
    simulator = Simulator(cart_pole)
    
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
    
    # Run controlled simulation
    print("Running LQR-controlled simulation...")
    result = simulator.run(
        initial_state=initial_state,
        duration=10.0,
        dt=0.02,
        controller=controller
    )
    print(f"Simulation complete. {len(result.time)} timesteps.")
    
    # Create plots directory
    plots_dir = Path(__file__).parent.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Create visualizer
    visualizer = Visualizer(cart_pole)
    
    # Plot state trajectories
    print("Generating state plots...")
    fig = visualizer.plot_states(
        result,
        save_path=plots_dir / "lqr_states.png"
    )
    fig.suptitle('LQR Controlled Cart-Pole', fontsize=14, y=1.0)
    
    # Calculate and plot control force
    print("Generating control force plot...")
    fig2, ax = plt.subplots(figsize=(10, 3))
    forces = []
    for i in range(len(result.time)):
        t = result.time[i]
        state = result.states[:, i]
        force = controller.compute(state, t)
        forces.append(force)
    
    ax.plot(result.time, forces, 'g-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control Force (N)')
    ax.set_title('LQR Controller Output')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save control force plot
    force_plot_path = plots_dir / "lqr_control_force.png"
    fig2.savefig(force_plot_path, dpi=300, bbox_inches='tight')
    print(f"Control force plot saved to {force_plot_path}")
    
    # Create and display animation
    print("Creating animation...")
    anim = visualizer.animate(result, interval=20)
    
    # Show all plots and animation
    visualizer.show()


if __name__ == "__main__":
    main()
