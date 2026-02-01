# Inverted Pendulum (Cart-Pole) Simulation

A modular simulation of an inverted pendulum on a cart with multiple control algorithms (PID, LQR, Pole Placement), realistic DC motor model, measurement noise, state filtering, and interactive real-time visualization.

## System Description

The cart-pole system consists of:
- A cart moving horizontally along a track
- A rigid pendulum attached via a pivot
- Friction on the cart and rotational damping at the pivot

**State Vector:** `[x, x_dot, theta, theta_dot]`
- `x`: cart position (m)
- `x_dot`: cart velocity (m/s)
- `theta`: pendulum angle from vertical (rad, 0 = upright)
- `theta_dot`: angular velocity (rad/s)

## Project Structure

```
src/
├── cart_pole.py        # Physical parameters and nonlinear dynamics
├── controller.py       # PID, LQR, and Pole Placement controllers
├── state_filter.py     # Measurement noise and two-stage filtering
├── motor.py            # DC motor model (voltage limits, back-EMF)
├── main_interactive.py # Interactive GUI with real-time control
└── generate_plots.py   # Batch comparison plots for all controllers

plots/                  # Auto-generated comparison plots
```

## Running the Simulations

**Interactive Simulation:**
```bash
cd src
python main_interactive.py
```

Features:
- Switch between controllers (None/PID/LQR/Pole Placement)
- Adjust PID gains with sliders
- Apply disturbance impulses
- Toggle noise and motor model on/off
- Real-time plots of angle, force, and voltage

**Generate Comparison Plots:**
```bash
cd src
python generate_plots.py
```

Generates 5 comparison plots in `/plots`:
- State trajectories for all controllers
- Control effort (desired vs actual force, voltage)
- Angle comparison
- Phase portrait
- Performance metrics (settling time, overshoot, effort)

## Control Algorithms

All controllers have dual objectives:
1. Keep pendulum upright (θ = 0)
2. Move cart to target position (x = 2m)

### PID Controller
Cascaded control: position error → angle setpoint → force.
- Outer loop: position PD generates desired tilt angle
- Inner loop: angle PID stabilizes to that setpoint
- Default gains: Kp=78, Ki=1.5, Kd=32, Kp_pos=0.11, Ki_pos=0.003, Kd_pos=0.50

### LQR Controller
Optimal full-state feedback minimizing J = ∫(x'Qx + u'Ru)dt.
- Default: Q = diag([1, 1, 100, 10]), R = 0.5
- Setpoint: [2, 0, 0, 0] (x=2m, stationary, upright)

### Pole Placement Controller
Full-state feedback with directly specified closed-loop poles.
- Default poles: [-1.5, -1.7, -1.9, -2.1]
- Setpoint: [2, 0, 0, 0] (x=2m, stationary, upright)

## DC Motor Model

Realistic motor model for physical implementation (LP 12V):
- **4 motors**, voltage range 1V-12V
- **170 RPM at 12V** (no-load speed)
- **Stall torque: 3.7 kg·cm**, Stall current: 0.9A
- **30mm wheel radius**
- **Max force: ~4.8N**

Converts controller force to motor voltage with:
- Back-EMF at high speeds
- Voltage saturation
- Stall below minimum voltage

## Measurement Noise & Filtering

**Noise:** Gaussian on position (σ=5mm) and angle (σ=0.01 rad)

**Two-stage filter:**
1. Low-pass: y[k] = α·y[k-1] + (1-α)·x[k]
2. Dirty derivative: ẏ[k] = (y[k] - y[k-1]) / Ts

Default: τ_pos=0.1s, τ_angle=0.08s, Ts=0.02s (50Hz)

## Dependencies

- numpy
- scipy
- matplotlib
