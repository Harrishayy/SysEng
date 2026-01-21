# Inverted Pendulum (Cart-Pole) Simulation

A modular simulation of an inverted pendulum on a cart using numerical integration, with multiple control algorithms (PID, LQR, Pole Placement) and realistic measurement noise with state filtering.

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
├── controller.py        # PIDController, LQRController, and SMCController classes
├── state_filter.py      # Measurement noise and two-stage state filtering
├── main_uncontrolled.py # Uncontrolled/passive simulation
├── main_pid.py          # PID-controlled simulation
├── main_lqr.py          # LQR-controlled simulation
└── main_smc.py          # SMC-controlled simulation

plots/                    # Auto-generated plots from simulations
├── uncontrolled_states.png
├── pid_states.png
├── pid_control_force.png
├── lqr_states.png
├── lqr_control_force.png
├── smc_states.png
└── smc_control_force.png
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

**LQR Controlled (optimal control):**
```bash
cd src
python main_lqr.py
```

This demonstrates optimal LQR control using full state feedback. The controller gains are computed by solving the continuous-time algebraic Riccati equation (CARE) with tunable Q and R cost matrices.

**Pole Placement Controlled (direct pole specification):**
```bash
cd src
python main_pole_placement.py
```

This demonstrates pole placement control where closed-loop poles are directly specified. Shows state trajectories, control force, and animation.

**Note:** All simulations automatically save plots to the `plots/` directory. Plots show true states (blue), noisy measurements (red dots), and filtered estimates (green).

## Measurement Noise and State Filtering

All simulations include realistic measurement noise and a two-stage state filtering approach:

### Measurement Noise
- **Position noise:** σ = 5mm (Gaussian)
- **Angle noise:** σ = 0.01 rad ≈ 0.6° (Gaussian)

Noise is added to position measurements (x and θ), simulating encoder noise. Velocities are not directly measured.

### Two-Stage State Filter

**Stage 1: Low-pass filtering of position**

The continuous-time first-order low-pass filter:
```
τ·ẏ(t) + y(t) = x(t)
```

In discrete time with sampling period Ts:
```
y[k] = α·y[k-1] + (1-α)·x[k]
```
where the smoothing factor α = τ/(τ + Ts)

**Stage 2: Velocity estimation (dirty derivative)**
```
ẏ[k] = (y[k] - y[k-1]) / Ts
```

**Default filter parameters:**
- Position filter: τ = 0.05s (α ≈ 0.714)
- Angle filter: τ = 0.02s (α ≈ 0.5)
- Sampling period: Ts = 0.02s (50 Hz)

This two-stage approach provides smooth position estimates and velocity estimates with reduced sensitivity to measurement noise.

## Control Algorithms

### PID Controller

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

### LQR Controller

The Linear Quadratic Regulator (LQR) provides optimal full-state feedback control by minimizing a cost function:

J = ∫(x'Qx + u'Ru)dt

where:
- **Q matrix** (4×4): State cost - penalizes deviations in [x, x_dot, theta, theta_dot]
- **R matrix** (1×1): Control cost - penalizes control effort

**Default cost matrices:**
- Q = diag([1, 1, 100, 10]) - heavily penalizes angle error
- R = [[0.1]] - moderate control cost

The optimal gains K are computed by solving the continuous-time algebraic Riccati equation (CARE). Control law: u = -Kx

**Features:**
- Optimal control for linearized system
- Full state feedback (uses all 4 states)
- Guaranteed stability for small angles
- Force saturation: ±100 N

### Pole Placement Controller

Pole Placement provides full-state feedback control by directly specifying where the closed-loop system poles should be located. This gives direct control over the system's transient response characteristics.

**Control law:**
u = -K(x - x_setpoint)

The gain matrix K is computed such that the closed-loop system:
```
ẋ = (A - BK)x
```
has eigenvalues at the specified pole locations.

**Default poles:**
- p₁ = -3, p₂ = -4, p₃ = -5, p₄ = -6 (all real, stable)

**Pole selection guidelines:**
- All poles must have negative real parts for stability
- More negative = faster response but more control effort
- Complex conjugate pairs give oscillatory response
- Poles should not be too fast (actuator limits) or too slow (poor disturbance rejection)

**Features:**
- Direct specification of closed-loop dynamics
- Full state feedback (uses all 4 states)
- Intuitive tuning via pole locations
- Force saturation: ±100 N

## Dependencies

- numpy
- scipy
- matplotlib
