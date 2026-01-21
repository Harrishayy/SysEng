"""Uncontrolled cart-pole simulation."""
import numpy as np
from pathlib import Path

from cart_pole import CartPole
from simulator import Simulator
from visualizer import Visualizer
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
    
    # Initial state: small angle offset
    initial_state = np.array([0.0, 0.0, 0.2, 0.0])  # ~11.5 deg
    
    print("Running uncontrolled simulation...")
    result = simulator.run_with_noise(
        initial_state=initial_state,
        duration=10.0, dt=dt,
        controller=None,
        state_processor=state_processor
    )
    print(f"Complete. {len(result.time)} timesteps.")
    
    # Save plots
    plots_dir = Path(__file__).parent.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    visualizer = Visualizer(cart_pole)
    visualizer.plot_states_with_noise(result, save_path=plots_dir / "uncontrolled_states.png")
    
    print("Creating animation...")
    visualizer.animate(result, interval=20)
    visualizer.show()


if __name__ == "__main__":
    main()
