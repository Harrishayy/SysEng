"""LQR-controlled cart-pole simulation with disturbances."""
import numpy as np
from pathlib import Path

from cart_pole import CartPole
from simulator import Simulator
from visualizer import Visualizer
from controller import LQRController
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
    
    # LQR controller
    Q = np.diag([3.0, 0.5, 100.0, 10.0])  # State cost
    R = np.array([[0.1]])                  # Control cost
    setpoint = np.array([2.0, 0.0, 0.0, 0.0])  # Move to x=2m
    
    controller = LQRController(
        cart_mass=cart_pole.M, pendulum_mass=cart_pole.m,
        rod_length=cart_pole.L, cart_friction=cart_pole.b,
        rotational_damping=cart_pole.c, gravity=cart_pole.g,
        Q=Q, R=R, setpoint=setpoint
    )
    
    gains = controller.get_gains()
    print("LQR Gains:", {k: f"{v:.4f}" for k, v in gains.items()})
    print(f"Setpoint: x={setpoint[0]}m")
    
    # Initial state: 15 deg offset
    initial_state = np.array([0.0, 0.0, np.deg2rad(15.0), 0.0])
    
    # Disturbance: impulses at t=3s and t=6s
    def disturbance_func(t):
        if 3.0 <= t < 3.2: return 20.0   # Push right
        if 6.0 <= t < 6.2: return -15.0  # Push left
        return 0.0
    
    print("\nDisturbances: +20N at t=3s, -15N at t=6s")
    print("Running LQR simulation...")
    
    result = simulator.run_with_noise(
        initial_state=initial_state,
        duration=10.0, dt=dt,
        controller=controller,
        state_processor=state_processor,
        disturbance_func=disturbance_func
    )
    print(f"Complete. {len(result.time)} timesteps.")
    
    # Save plots
    plots_dir = Path(__file__).parent.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    visualizer = Visualizer(cart_pole)
    
    fig = visualizer.plot_comprehensive(result, save_path=plots_dir / "lqr_states.png")
    fig.suptitle('LQR Controlled Cart-Pole (with disturbances)', fontsize=14, y=1.0)
    
    visualizer.plot_forces(result, save_path=plots_dir / "lqr_control_force.png")
    
    print("Creating animation...")
    visualizer.animate(result, interval=20)
    visualizer.show()


if __name__ == "__main__":
    main()
