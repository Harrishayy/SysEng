"""PID-controlled cart-pole simulation."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from cart_pole import CartPole
from simulator import Simulator
from visualizer import Visualizer
from controller import PIDController
from state_filter import NoisyStateProcessor


def main():
    # System parameters
    cart_pole = CartPole(
        cart_mass=1.0, pendulum_mass=0.05, rod_length=0.8,
        cart_friction=0.1, rotational_damping=0.01, gravity=9.81
    )
    
    dt = 0.02  # 50 Hz
    simulator = Simulator(cart_pole)
    
    # Noise and filtering
    state_processor = NoisyStateProcessor(
        position_noise_std=0.005, angle_noise_std=0.01,
        tau_position=0.05, tau_angle=0.02, dt=dt, seed=42
    )
    
    # PID controller
    controller = PIDController(kp=100, ki=0.5, kd=20.0, setpoint=0.0)
    
    # Initial state: 15 deg offset
    initial_state = np.array([0.0, 0.0, np.deg2rad(15.0), 0.0])
    
    print(f"Initial angle: {np.rad2deg(initial_state[2]):.1f}°")
    print("Running PID-controlled simulation...")
    
    result = simulator.run_with_noise(
        initial_state=initial_state,
        duration=10.0, dt=dt,
        controller=controller,
        state_processor=state_processor
    )
    print(f"Complete. {len(result.time)} timesteps.")
    
    # Save plots
    plots_dir = Path(__file__).parent.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    visualizer = Visualizer(cart_pole)
    
    fig = visualizer.plot_states_with_noise(result, save_path=plots_dir / "pid_states.png")
    fig.suptitle('PID Controlled Cart-Pole', fontsize=14, y=1.0)
    
    # Control force plot
    fig2, ax = plt.subplots(figsize=(10, 3))
    controller.reset()
    forces = [controller.compute(result.filtered_states[2, i], result.filtered_states[3, i], result.time[i])
              for i in range(len(result.time))]
    
    ax.plot(result.time, forces, 'g-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control Force (N)')
    ax.set_title('PID Controller Output')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    fig2.savefig(plots_dir / "pid_control_force.png", dpi=300, bbox_inches='tight')
    
    print("Creating animation...")
    visualizer.animate(result, interval=20)
    visualizer.show()


if __name__ == "__main__":
    main()
