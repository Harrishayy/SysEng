# Inverted Pendulum (Cart-Pole) Simulation

Modular simulation of an inverted pendulum on a cart with PID, LQR, and Pole Placement controllers, realistic DC motor model, measurement noise, state filtering, and interactive visualization.

## System

**State:** `[x, x_dot, theta, theta_dot]` (position, velocity, angle from vertical, angular velocity)

**Physical params:** cart 1kg, pendulum 0.05kg, rod 0.8m, cart friction 0.1, rotational damping 0.01

## Project Structure

```
src/
├── cart_pole.py        # Dynamics
├── controller.py       # PID, LQR, Pole Placement
├── state_filter.py     # Noise + filtering
├── motor.py            # DC motor model
├── main_interactive.py # Interactive GUI
├── generate_plots.py   # Comparison plots
├── robustness_test.py  # Max angle/disturbance tests
└── metrics.py          # Performance output

plots/                  # Generated plots
```

## Usage

```bash
cd src
python main_interactive.py   # Interactive simulation
python generate_plots.py     # Generate comparison plots
python robustness_test.py    # Run robustness tests
python metrics.py            # Print performance metrics
```

## Control Algorithms

All controllers have dual objectives:
1. Keep pendulum upright (θ = 0)
2. Move cart to target (x = 2m)

| Controller | Parameters |
|------------|------------|
| **PID** (cascaded) | Kp=55, Ki=2, Kd=26, Kp_pos=0.05, Ki_pos=0.004, Kd_pos=0.11 |
| **LQR** | Q=diag([1,1,100,10]), R=0.5 |
| **Pole Placement** | poles=[-1.5, -1.7, -1.9, -2.1] |

## Motor Model (LP 12V)

| Spec | Value |
|------|-------|
| Motors | 4x |
| Voltage | 1-12V |
| No-load speed | 170 RPM @ 12V |
| Stall torque | 3.7 kg·cm (0.9A) |
| Wheel radius | 30mm |
| Max force | ~4.8N |

Includes back-EMF and voltage saturation.

## Noise & Filtering

- Gaussian noise: σ_pos=5mm, σ_angle=0.01rad
- Low-pass + dirty derivative filter (τ_pos=0.1s, τ_angle=0.08s, 50Hz)

## Dependencies

numpy, scipy, matplotlib
