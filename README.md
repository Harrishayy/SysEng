# Inverted Pendulum (Cart-Pole) Simulation

A modular simulation of an inverted pendulum on a cart using numerical integration, with optional PID control for stabilization.

## System Description

The cart-pole system consists of:
- A cart that moves horizontally along a track
- A rigid pendulum attached to the cart via a pivot
- Friction on the cart and rotational damping on the pivot

The simulation solves the coupled nonlinear equations of motion for the cart position and pendulum angle using Runge-Kutta integration (RK45).

**State Vector:** `[x, x_dot, theta, theta_dot]`
- `x`: cart position (m)
- `x_dot`: cart velocity (m/s)
- `theta`: pendulum angle from vertical (rad, positive = clockwise)
- `theta_dot`: angular velocity (rad/s)

## Project Structure

```
src/
├── cart_pole.py         # CartPole class - physical parameters and system dynamics
├── simulator.py         # Simulator class - numerical integration using scipy
├── visualizer.py        # Visualizer class - state plots and animated visualization
├── controller.py        # PIDController class for stabilization
├── main_uncontrolled.py # Uncontrolled/passive simulation
└── main_pid.py          # PID-controlled simulation
```

## Running the Simulations

**Uncontrolled (passive system):**
```bash
cd src
python main_uncontrolled.py
```

This runs an uncontrolled simulation starting with the pendulum at 0.2 rad (≈11.5°), showing the natural dynamics as it falls and oscillates.

**PID Controlled (stabilization):**
```bash
cd src
python main_pid.py
```

This demonstrates PID control that stabilizes the pendulum at the upright position from an initial angle of 0.3 rad (≈17°). Shows state trajectories, control force, and animation.

## PID Controller

The PID controller computes a force to apply to the cart based on the pendulum angle error:

- **Proportional (Kp):** Reacts to current angle error
- **Integral (Ki):** Eliminates steady-state error
- **Derivative (Kd):** Provides damping using angular velocity

**Tuned gains:**
- Kp = 100 N/rad
- Ki = 0.5 N/(rad·s)
- Kd = 20 N/(rad/s)

**Safety features:**
- Force saturation: ±100 N
- Integral anti-windup

## Dependencies

- numpy
- scipy
- matplotlib
