# Inverted Pendulum (Cart-Pole)

Physical cart-pole robot with Python simulation and Arduino firmware. Implements LQR, PID, and Pole Placement controllers for balancing an inverted pendulum while moving the cart to a target position.

## Repository Structure

```
src/                            # Python simulation
├── cart_pole.py                # Nonlinear dynamics (forward Euler, 500 Hz)
├── controller.py               # PID, LQR, Pole Placement controllers
├── motor.py                    # DC motor model (back-EMF, voltage saturation)
├── state_filter.py             # Gaussian noise + MA(5) → derivative → exponential LP filter
├── main_interactive.py         # Matplotlib GUI simulation (500 Hz)
├── generate_plots.py           # Controller comparison plots
├── generate_videos.py          # Side-by-side controller comparison videos → videos/
├── robustness_test.py          # Max angle / disturbance tolerance tests
└── metrics.py                  # Performance metrics output

Sensors/Tests/                  # Arduino firmware (physical hardware)
├── LQRPendulum/                         # LQR controller (500 Hz, reference governor, LQI)
├── LQRJerkStartStabilisation/           # LQR with jerk-start swing-up sequence
├── PolePlacementPendulum/               # Pole placement controller
├── PolePlacementJerkStartStabilisation/ # Pole placement with jerk-start
├── Motors/                     # Motor direction / speed test
├── OpticalSensor/              # Pendulum encoder test
├── GetRawAngle/                # Read raw pendulum angle over serial
├── GetAngularVelAndAccel/      # Read angular velocity and acceleration
├── CheckAddresses/             # I2C address scan
└── HardwareDetectionTest/      # Motor shield detection
```

## Hardware

| Component | Part |
|-----------|------|
| Microcontroller | Arduino Giga |
| Motors | 4× Pololu #4862 (9.68:1 gearbox, 12 V nominal) |
| Motor driver | 2× Motoron M3S550 (I2C, speed range −800 to +800) |
| Pendulum encoder | Broadcom AS22 (1000 CPR × 4 quadrature = 4000 counts/rev) |
| Cart encoder | Pololu motor encoder (464.64 counts/rev) |
| Supply voltage | 10.6 V (simulation) / 12.3 V (hardware) |
| Wheel radius | 40 mm |

**Motor limits (firmware, 12.3 V supply):**
- Stall torque per motor at 12.3 V: 0.171 N·m
- Max force (4 motors, r = 40 mm): 17.1 N
- MOTOR\_MAX: 730 (current-limited; Motoron 1.7 A / motor 1.845 A stall at 12.3 V)
- Force-to-command scale: 18.72 cmd/N (empirically tuned; geometric derivation gives 42.72)

## Physical Parameters

Simulation and hardware use the same physical parameters:

| Parameter | Value |
|-----------|-------|
| Cart mass M | 1.2 kg |
| Pendulum mass m | 0.91 kg |
| Rod length to CoM l | 0.5 m |
| Full rod length | 0.6 m (visualisation only) |
| Cart friction b | 0.1 N·s/m |
| Rotational damping c | 0.01 N·m·s/rad |
| Gravity g | 9.81 m/s² |

**State vector:** `[x (m), x_dot (m/s), theta (rad), theta_dot (rad/s)]`

Upright equilibrium: θ = 0. Positive θ is a lean to the right.

## Control Algorithms

All controllers share dual objectives: keep the pendulum upright (θ = 0) and drive the cart to a target position.

### Cascaded PID (simulation only)

Position outer loop → angle setpoint → angle inner loop → force.

| Gain | Value |
|------|-------|
| Kp (angle inner) | 55 |
| Ki (angle inner) | 2 |
| Kd (angle inner) | 26 |
| Kp\_pos (outer) | 0.05 |
| Ki\_pos (outer) | 0.004 |
| Kd\_pos (outer) | 0.11 |
| Max angle setpoint | ±0.09 rad |

### LQR (simulation and hardware)

Solves the continuous algebraic Riccati equation (CARE) offline; applies `u = -K(x − x_ref)`.

Cost matrices (both simulation and hardware): **Q = diag([5, 1, 100, 10]), R = 0.01**

**Simulation gains** (Python `controller.py`, setpoint x = 2 m):

| Gain | Value |
|------|-------|
| K1 (position) | −22.36 N/m |
| K2 (velocity) | −31.32 N·s/m |
| K3 (angle) | −213.97 N/rad |
| K4 (angular rate) | −51.28 N·s/rad |

**Hardware gains** (firmware `LQRPendulum.ino`, control law `u = -(K1·(x−ref) + K2·ẋ + K3·θ + K4·θ̇)`):

| Gain | Value | Units |
|------|-------|-------|
| K1 | 22.3607 | N/m |
| K2 | 55.3222 | N·s/m |
| K3 | −220.9713 | N/rad |
| K4 | −55.2750 | N·s/rad |
| K5 (LQI integral, optional) | 0 (disabled by default) | N/(m·s) |

K3 and K4 are negative because B[3] = −1/(M·l) < 0 inverts their sign through the B matrix. The control law's leading minus sign then produces the correct physical response (cart chases the fall).

**Reference governor (hardware only):** advances the position setpoint at 0.1 m/s, but only while |θ| < 0.10 rad and |θ̇| < 0.75 rad/s, to avoid sacrificing angle stability during repositioning.

**Safety cutoff:** if |θ| > 0.52 rad (~30°), motors are set to zero and the integral is reset. The linear controller is only valid near upright.

### Pole Placement (simulation and hardware)

Poles are placed at the LQR closed-loop eigenvalues using Ackermann's formula (`scipy.signal.place_poles`). Because the system is fully controllable (SISO), this yields an identical gain vector to LQR. The firmware, reference governor, safety cutoff, and filter are identical to `LQRPendulum.ino`.

## State Estimation / Filtering

Both simulation and hardware use the same three-stage pipeline at 500 Hz:

1. **Moving average MA(5):** circular buffer over the last 5 raw encoder readings reduces quantisation noise.
2. **Numerical derivative:** `θ̇_raw = (θ_MA[k] − θ_MA[k−1]) / Ts`
3. **Exponential low-pass:** `θ̇_filt[k] = α·θ̇_raw + (1−α)·θ̇_filt[k−1]`
   - α\_θ = 0.20 (pendulum angular rate) — 0.15 in jerk-start variants
   - α\_x = 0.10 (cart velocity) — 0.08 in jerk-start variants

**Simulation noise injection** (before filtering):
- Position noise: σ = 1 mm
- Angle noise: σ = 2 mrad

## Simulation Usage

```bash
cd src
python main_interactive.py   # Interactive GUI (500 Hz, matplotlib)
python generate_plots.py     # Controller comparison plots → plots/
python generate_videos.py    # Side-by-side MP4 videos → videos/
python robustness_test.py    # Max recoverable angle / disturbance tests
python metrics.py            # Print performance metrics
```

## Hardware Usage

1. Compute LQR gains by running the Python script embedded in the header of [LQRPendulum.ino](Sensors/Tests/LQRPendulum/LQRPendulum.ino) and paste the K values into the firmware.
2. Flash `LQRPendulum` (or a jerk-start variant) to the Arduino Giga via the Arduino IDE.
3. Open the serial monitor at 115200 baud.
4. Hold the pendulum upright and press `r` to calibrate.
5. Press `i` to set target 2.0 m (or `t`/`y`/`u` for 0/0.5/1 m).

**Serial commands (LQR / Pole Placement):**

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

**Jerk-start variants** additionally accept `j` (trigger jerk launch) and `k` (cancel, return to idle). The cart automatically attempts a jerk launch 10 s after boot.

## Dependencies

**Python:** numpy, scipy, matplotlib

**Arduino:** [Motoron library](https://github.com/pololu/motoron-arduino), Wire
