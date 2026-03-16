# Inverted Pendulum (Cart-Pole)

Physical cart-pole robot with Python simulation and Arduino firmware. Implements LQR, PID, and Pole Placement controllers for balancing an inverted pendulum while moving the cart to a target position.

## Repository Structure

```
src/                            # Python simulation
├── cart_pole.py                # Nonlinear dynamics (RK4 integration)
├── controller.py               # PID, LQR, Pole Placement controllers
├── motor.py                    # DC motor model (back-EMF, voltage saturation)
├── state_filter.py             # Gaussian noise injection + low-pass/dirty-derivative filter
├── main_interactive.py         # Matplotlib GUI simulation (50 Hz)
├── generate_plots.py           # Controller comparison plots
├── robustness_test.py          # Max angle / disturbance tolerance tests
└── metrics.py                  # Performance metrics output

Sensors/Tests/                  # Arduino firmware (physical hardware)
├── LQRPendulum/                         # LQR controller (500 Hz, reference governor, LQI)
├── LQRJerkStartStabilisation/           # LQR with jerk-start swing-up sequence
├── PolePlacementPendulum/               # Pole placement controller
├── PolePlacementJerkStartStabilisation/ # Pole placement with jerk-start
├── PIDStabilisation/                    # Cascaded PID controller
├── PIDJerkStartStabilisation/           # Cascaded PID with jerk-start
├── Motors/                     # Motor direction / speed test
├── OpticalSensor/              # Pendulum encoder test
├── GetRawAngle/                # Read raw pendulum angle over serial
├── GetAngularVelAndAccel/      # Read angular velocity and acceleration
├── CheckAddresses/             # I2C address scan
├── HardwareDetectionTest/      # Motor shield detection
├── controller.py               # Controller code copy (used during hardware testing)
└── swing_test_analysis.py      # Offline swing-up data analysis
```

## Hardware

| Component | Part |
|-----------|------|
| Microcontroller | Arduino Giga / Mega |
| Motors | 4x Pololu #4862 (9.68:1 gearbox, 12 V nominal) |
| Motor driver | Motoron M3S550 (I2C, speed range -800 to +800) |
| Pendulum encoder | Broadcom AS22 (1000 CPR, x4 quadrature = 4000 counts/rev) |
| Cart encoder | Pololu motor encoder (464.64 counts/rev of gearbox shaft) |
| Supply voltage | 10.6 V |
| Wheel radius | 40 mm |

**Motor specs at 10.6 V:**
- Stall torque per motor: 0.147 N·m (gearbox output)
- Max force (4 motors, r = 40 mm): ~14.7 N
- Force-to-command scale: 54.33 cmd/N

## Physical Parameters

| Parameter | Simulation | Hardware |
|-----------|-----------|----------|
| Cart mass M | 1.0 kg | 1.2 kg |
| Pendulum mass m | 0.05 kg | 0.91 kg |
| Rod length to CoM l | 0.8 m | 0.5 m |
| Cart friction b | 0.1 N·s/m | — |
| Rotational damping c | 0.01 N·m·s/rad | — |

**State vector:** `[x (m), x_dot (m/s), theta (rad), theta_dot (rad/s)]`

## Control Algorithms

All controllers share dual objectives: keep the pendulum upright (θ = 0) and drive the cart to a target position.

### Cascaded PID (simulation)

Position outer loop → angle setpoint → force inner loop.

| Gain | Value |
|------|-------|
| Kp (angle) | 55 |
| Ki (angle) | 2 |
| Kd (angle) | 26 |
| Kp\_pos | 0.05 |
| Ki\_pos | 0.004 |
| Kd\_pos | 0.11 |
| Max angle setpoint | ±0.09 rad |

### LQR (simulation and hardware)

Solves the continuous algebraic Riccati equation (CARE) offline; applies `u = -K(x - x_ref)`.

**Simulation gains** — Q = diag([1, 1, 100, 10]), R = 0.5, setpoint x = 2 m.

**Hardware gains** — Q = diag([5, 1, 100, 10]), R = 0.01, M=1.2, m=0.91, l=0.5:

| Gain | Value |
|------|-------|
| K1 (position) | -22.36 N/m |
| K2 (velocity) | -31.32 N·s/m |
| K3 (angle) | -213.97 N/rad |
| K4 (angular rate) | -20.28 N·s/rad |
| K5 (LQI integral, optional) | 0 (disabled by default) |

The hardware firmware runs at 500 Hz and uses a reference governor that advances the position setpoint at 0.1 m/s only when |θ| < 0.10 rad and |θ_dot| < 0.75 rad/s. Motors cut out if |θ| > 0.52 rad (~30°).

### Pole Placement (simulation and hardware)

Desired poles: `[-1.5, -1.7, -1.9, -2.1]`

## Simulation Usage

```bash
cd src
python main_interactive.py   # Interactive GUI (50 Hz, matplotlib)
python generate_plots.py     # Controller comparison plots → plots/
python robustness_test.py    # Max recoverable angle / disturbance tests
python metrics.py            # Print performance metrics
```

## Hardware Usage

1. Compute LQR gains by running the Python script embedded in the header of [LQRPendulum.ino](Sensors/Tests/LQRPendulum/LQRPendulum.ino) and paste the K values into the firmware.
2. Flash `LQRPendulum` to the Arduino Giga via the Arduino IDE.
3. Open the serial monitor at 115200 baud.
4. Hold the pendulum upright and press `r` to calibrate.

**Serial commands:**

| Key | Action |
|-----|--------|
| `r` | Recalibrate (hold pendulum upright first) |
| `t` / `y` / `u` / `i` | Set target to 0.0 / 0.5 / 1.0 / 2.0 m |
| `1` / `2` | K1 stronger / weaker |
| `3` / `4` | K2 stronger / weaker |
| `5` / `6` | K3 stronger / weaker |
| `7` / `8` | K4 stronger / weaker |
| `9` / `0` | K5 (LQI) stronger / weaker |
| `p` | Print current gains |

## Noise and Filtering (Simulation)

- Measurement noise: σ_pos = 5 mm, σ_angle = 0.01 rad (Gaussian)
- Low-pass filter: α = τ/(τ + Ts), τ_pos = 0.1 s, τ_angle = 0.08 s
- Velocity estimate: dirty derivative at 50 Hz

**Hardware filtering (LQRPendulum):**
- 5-sample moving average on raw encoder readings
- Exponential low-pass on derivatives: α_theta = 0.20, α_x = 0.10

## Dependencies

**Python:** numpy, scipy, matplotlib

**Arduino:** [Motoron library](https://github.com/pololu/motoron-arduino), Wire
