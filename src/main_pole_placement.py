"""Pole Placement-controlled cart-pole simulation."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from cart_pole import CartPole
from simulator import Simulator
from visualizer import Visualizer
from controller import PolePlacementController
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
    
    # Pole placement controller
    desired_poles = np.array([-3.0, -4.0, -5.0, -6.0])  # All real, stable
    setpoint = np.array([0.0, 0.0, 0.0, 0.0])
    
    controller = PolePlacementController(
        cart_mass=cart_pole.M, pendulum_mass=cart_pole.m,
        rod_length=cart_pole.L, cart_friction=cart_pole.b,
        rotational_damping=cart_pole.c, gravity=cart_pole.g,
        poles=desired_poles, setpoint=setpoint
    )
    
    # Print controller info
    gains = controller.get_gains()
    poles_info = controller.get_poles()
    
    print("Pole Placement Controller")
    print("=" * 40)
    print("\nDesired Poles:", [f"{p:.2f}" for p in poles_info['desired']])
    print("Achieved Poles:", [f"{np.real(p):.2f}" for p in poles_info['achieved']])
    print("\nGains:", {k: f"{v:.4f}" for k, v in gains.items()})
    
    # Initial state: 17 deg offset
    initial_state = np.array([0.0, 0.0, 0.3, 0.0])
    
    print(f"\nInitial angle: {np.rad2deg(initial_state[2]):.1f}°")
    print("Running Pole Placement simulation...")
    
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
    
    fig = visualizer.plot_states_with_noise(result, save_path=plots_dir / "pole_placement_states.png")
    fig.suptitle('Pole Placement Controlled Cart-Pole', fontsize=14, y=1.0)
    
    # Control force plot
    fig2, ax = plt.subplots(figsize=(10, 3))
    forces = [controller.compute(result.filtered_states[:, i], result.time[i])
              for i in range(len(result.time))]
    
    ax.plot(result.time, forces, 'g-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control Force (N)')
    ax.set_title('Pole Placement Controller Output')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    fig2.savefig(plots_dir / "pole_placement_control_force.png", dpi=300, bbox_inches='tight')
    
    print("Creating animation...")
    visualizer.animate(result, interval=20)
    visualizer.show()


if __name__ == "__main__":
    main()
