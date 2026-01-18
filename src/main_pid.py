"""
Main entry point for controlled cart-pole simulation with PID controller.

Run this file to see the cart-pole stabilized at the upright position.
"""
import numpy as np
import matplotlib.pyplot as plt

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
    
    # Create PID controller
    # Tuned gains for this system
    controller = PIDController(
        kp=100,    # Proportional gain
        ki=0.5,     # Integral gain (small to prevent windup)
        kd=20.0,    # Derivative gain
        setpoint=0.0  # Target: upright position
    )
    
    # Define initial conditions
    # Starting with a moderate angle offset to test controller
    initial_state = np.array([
        0.0,    # x: cart position (m)
        0.0,    # x_dot: cart velocity (m/s)
        0.3,    # theta: pendulum angle (rad) - about 17 degrees
        0.0     # theta_dot: angular velocity (rad/s)
    ])
    
    print(f"Initial angle: {np.rad2deg(initial_state[2]):.1f} degrees")
    
    # Run controlled simulation
    print("Running controlled simulation...")
    result = simulator.run(
        initial_state=initial_state,
        duration=10.0,
        dt=0.02,
        controller=controller
    )
    print(f"Simulation complete. {len(result.time)} timesteps.")
    
    # Create visualizer
    visualizer = Visualizer(cart_pole)
    
    # Plot state trajectories
    print("Generating state plots...")
    fig = visualizer.plot_states(result)
    fig.suptitle('PID Controlled Cart-Pole', fontsize=14, y=1.0)
    
    # Calculate and plot control force
    fig2, ax = plt.subplots(figsize=(10, 3))
    forces = []
    controller.reset()  # Reset controller state
    for i in range(len(result.time)):
        t = result.time[i]
        theta = result.theta[i]
        theta_dot = result.theta_dot[i]
        force = controller.compute(theta, theta_dot, t)
        forces.append(force)
    
    ax.plot(result.time, forces, 'g-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control Force (N)')
    ax.set_title('PID Controller Output')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Create and display animation
    print("Creating animation...")
    anim = visualizer.animate(result, interval=20)
    
    # Show all plots and animation
    visualizer.show()


if __name__ == "__main__":
    main()
