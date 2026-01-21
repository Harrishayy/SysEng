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
    
    # Create PID controller with position setpoint
    # Tuned gains for this system
    controller = PIDController(
        # Angle control gains (primary)
        kp_theta=100,    # Proportional gain for angle
        ki_theta=0.5,    # Integral gain for angle
        kd_theta=20.0,   # Derivative gain for angle
        # Position control gains (secondary)
        kp_x=7.0,       # Proportional gain for position
        ki_x=0.0,        # Integral gain for position
        kd_x=12.0,       # Derivative gain for position velocity
        # Setpoints
        theta_setpoint=0.0,  # Target: upright position
        x_setpoint=2.0       # Target: cart at 2.0m
    )
    
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
    print(f"Target position: {controller.x_setpoint:.2f} meters")
    
    # Run controlled simulation
    print("Running controlled simulation...")
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
        save_path=plots_dir / "pid_states.png"
    )
    fig.suptitle('PID Controlled Cart-Pole', fontsize=14, y=1.0)
    
    # Calculate and plot control force
    print("Generating control force plot...")
    fig2, ax = plt.subplots(figsize=(10, 3))
    forces = []
    controller.reset()  # Reset controller state
    for i in range(len(result.time)):
        t = result.time[i]
        state = result.states[:, i]
        force = controller.compute(state, t)
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