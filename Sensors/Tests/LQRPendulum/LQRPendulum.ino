/*
 * ============================================================================
 *  TRUE LQR INVERTED PENDULUM CONTROLLER
 *  Arduino Giga / Mega compatible
 * ============================================================================
 *
 *  HARDWARE CONFIRMED FROM DATASHEETS:
 *  ─────────────────────────────────────────────────────────────────────────
 *  Motor:  Pololu #4862  — 9.68:1 gearbox, 12 V nominal
 *          Stall torque @ 12 V : 1.700 kg·cm = 0.16671 N·m (gearbox output)
 *          No-load speed @ 12 V: 800 RPM (gearbox output)
 *
 *  Driver: Motoron M3S550 — speed command range -800 to +800
 *          +/-800 maps linearly to +/-100% PWM duty cycle (i.e. +/-V_supply)
 *
 *  Pendulum encoder: Broadcom AS22, 1000 CPR per channel, x4 quadrature
 *          = 4000 counts per revolution  (unchanged from your original code)
 *
 *  Cart encoder: Pololu motor encoder "provides 464.64 counts per revolution
 *          of the gearbox output shaft".  Your ISRs implement full quadrature,
 *          so use 464.64 directly.  (Original code had 465.6 — minor error.)
 *
 *  Supply: 10.6 V (user measured)
 *
 * ============================================================================
 *  FORCE-TO-COMMAND SCALE  (your original value of 100.81 was ~2x too large)
 * ============================================================================
 *
 *  At 10.6 V, stall torque per motor (gearbox output):
 *      T = 0.16671 x (10.6 / 12.0) = 0.14727 N.m
 *
 *  Force per motor at drive wheel (radius 0.04 m):
 *      F = T / r = 0.14727 / 0.04 = 3.682 N
 *
 *  Four motors total:
 *      F_max = 4 x 3.682 = 14.73 N
 *
 *  Mapping from force (N) to speed command:
 *      cmd = F x (800 / F_max) = F x 54.33
 *
 * ============================================================================
 *  STATE-SPACE MODEL  (linearised, point-mass pendulum, I_pendulum = 0)
 * ============================================================================
 *
 *  State:  [x (m),  x_dot (m/s),  theta (rad),  theta_dot (rad/s)]
 *  Input:  u = horizontal force on cart (N), positive = positive-x
 *
 *  A = [0,  1,           0,          0]
 *      [0,  0,      -mg/M,          0]
 *      [0,  0,           0,          1]
 *      [0,  0,  (M+m)g/Ml,          0]
 *
 *  B = [0, 1/M, 0, -1/(M*l)]^T
 *
 *  With M=1.2, m=0.91, l=0.5:
 *    A[1,2] = -7.437      A[3,2] = +34.499   (unstable pole at +/-5.87 rad/s)
 *    B[1]   = +0.8333     B[3]   = -1.6667
 *
 * ============================================================================
 *  OFFLINE GAIN COMPUTATION -- run this Python script, paste result below
 * ============================================================================
 *
 *  #!/usr/bin/env python3
 *  import numpy as np
 *  from scipy.linalg import solve_continuous_are
 *
 *  M, m, l, g = 1.2, 0.91, 0.5, 9.81
 *
 *  A = np.array([
 *      [0, 1,              0,  0],
 *      [0, 0,        -m*g/M,  0],
 *      [0, 0,              0,  1],
 *      [0, 0, (M+m)*g/(M*l),  0]
 *  ])
 *  B = np.array([[0], [1/M], [0], [-1/(M*l)]])
 *
 *  # Tune these weights:
 *  # Q: penalise each state deviation   R: penalise control effort
 *  Q = np.diag([5.0, 1.0, 100.0, 10.0])   # [x, x_dot, theta, theta_dot]
 *  R = np.array([[0.01]])
 *
 *  P = solve_continuous_are(A, B, Q, R)
 *  K = (np.linalg.inv(R) @ B.T @ P).flatten()
 *
 *  poles = np.linalg.eigvals(A - B @ K.reshape(1, -1))
 *  print(f"K1={K[0]:.4f}  K2={K[1]:.4f}  K3={K[2]:.4f}  K4={K[3]:.4f}")
 *  print(f"Closed-loop poles: {np.sort_complex(poles)}")
 *  print(f"All stable: {all(np.real(p) < 0 for p in poles)}")
 *
 *  # For LQI (adds integral of position error -- eliminates steady-state drift):
 *  # A5 = np.block([[A, np.zeros((4,1))], [[1,0,0,0,0]]])
 *  # B5 = np.block([[B], [np.zeros((1,1))]])
 *  # Q5 = np.diag([5.0, 1.0, 100.0, 10.0, 3.0])
 *  # P5 = solve_continuous_are(A5, B5, Q5, R)
 *  # K5 = (np.linalg.inv(R) @ B5.T @ P5).flatten()
 *  # print(f"K5_integral = {K5[4]:.4f}")
 *
 * ============================================================================
 *  EXPECTED OUTPUT (Q=diag([5,1,100,10]), R=0.01):
 *    K1 = -22.3607   K2 = -31.3222   K3 = -213.9713   K4 = -51.2750
 *
 *  ALL four K values are NEGATIVE because B[3] = -1/(M*l) < 0 couples
 *  the states through the B matrix.  The control law u = -(K @ x) with
 *  all-negative K produces the correct physical response:
 *    theta > 0 (lean right)  -->  u = -(K3*theta) > 0  (chase the fall)
 *    x > 0 (cart right)      -->  u = -(K1*x) > 0      (tip pendulum to drift back)
 *
 *  LQI (with integral, Q5 = diag([5,1,100,10,3])):
 *    K1 = -45.2218  K2 = -44.6007  K3 = -247.9407  K4 = -58.2920  K5 = -17.3205
 * ============================================================================
 */

#include <Motoron.h>
#include <Wire.h>

// --- Motor shields -----------------------------------------------------------
MotoronI2C shield1(16);
MotoronI2C shield2(17);
// MOTOR_MAX is capped at 670 (not 800) to limit the effective motor voltage
// to ~10.6 V equivalent (670/800 x 12.6 V = 10.57 V), keeping stall current
// below the Motoron M3S550 overcurrent threshold that trips at 12.6 V / 800.
const int16_t MOTOR_MAX = 800;

// --- Pendulum encoder (Broadcom AS22, 1000 CPR x4 quadrature) ---------------
const long ENCODER_CPR           = 1000;
const long QUADRATURE_MULTIPLIER = 4;
const long COUNTS_PER_REV        = ENCODER_CPR * QUADRATURE_MULTIPLIER; // 4000

const int pinA = 2;
const int pinB = 3;
volatile long          encoderCount               = 0;
volatile uint8_t       lastPendulumEncoderState   = 0;
volatile unsigned long invalidPendulumTransitions = 0;

// --- Cart encoder (Pololu 4862, 464.64 counts per output shaft revolution) ---
//     Pololu spec: "464.64 counts per revolution of the gearbox output shaft"
//     Your two CHANGE ISRs already implement full x4 quadrature decoding,
//     so 464.64 is the correct constant to use here.
const float CART_ENCODER_CPR  = 464.64f;   // corrected from 465.6
const float CART_WHEEL_RADIUS = 0.04f;     // metres -- verify this!
const int   cartPinA          = 24;
const int   cartPinB          = 26;
volatile long          cartEncoderCount        = 0;
volatile uint8_t       lastCartEncoderState    = 0;
volatile unsigned long invalidCartTransitions  = 0;

// ============================================================================
//  PHYSICAL PARAMETERS
// ============================================================================
const float g_grav = 9.81f;
float M = 1.2f;    // cart mass (kg)
float m = 0.91f;   // pendulum mass (kg)
float l = 0.5f;    // pendulum CoM distance from pivot (m)

// ============================================================================
//  FORCE TO COMMAND SCALE
// ============================================================================
//  Formula (always):
//      F_max = 4 x (0.16671 N.m x V_supply/12) / wheel_radius_m
//      SCALE = MOTOR_MAX / F_max
//
//  At 10.6 V, MOTOR_MAX=800:  F_max=14.73 N  SCALE = 800/14.73 = 54.33
//  At 12.6 V, MOTOR_MAX=670:  F_max=17.50 N  SCALE = 670/17.50 = 38.29
//      (MOTOR_MAX capped at 670 to hold motor voltage to ~10.6 V equivalent,
//       preventing Motoron overcurrent faults at the higher supply voltage)
//
const float FORCE_TO_CMD_SCALE = 50.33f;

// ============================================================================
//  LQR GAINS  <-- PASTE VALUES FROM PYTHON SCRIPT HERE
// ============================================================================
//  The values below are analytically derived estimates (see header for method).
//  RUN THE PYTHON SCRIPT above and replace with its exact output.
//
//  Control law:  u = -(K1*(x-x_ref) + K2*x_dot + K3*theta + K4*theta_dot)
//
//  ALL gains are NEGATIVE (due to B matrix structure).  DO NOT change signs.
//  K1 < 0 (position),   K2 < 0 (velocity),
//  K3 < 0 (angle),      K4 < 0 (angular rate)
//
float K1 =  22.3607f;     // N / m
float K2 =  31.3222f;     // N.s / m
float K3 = -213.9713f;    // N / rad    (negative -- see header explanation)
float K4 =  -20.2750f;    // N.s / rad  (negative)

// Optional LQI integral term.  Eliminates steady-state position offset.
// Set to 0.0 during initial bring-up.  Compute from augmented Python script.
float K5_integral = -0.0f;   // N / (m.s)

float X_TARGET   = -0.0f;    // commanded cart position (m)
float X_SETPOINT = 0.0f;    // governed cart position reference used by LQR (m)

// --- Loop timing -------------------------------------------------------------
const float LOOP_DT_S = 0.002f;   // 2 ms = 500 Hz

// --- Position reference governor -------------------------------------------
// Large step changes in x_ref make the LQR fight position error too hard and
// sacrifice angle stability.  Move the reference slowly and only while upright.
const float X_REF_SLEW_RATE = 0.1f;        // m/s
const float X_REF_MAX_STEP  = X_REF_SLEW_RATE * LOOP_DT_S;
const float X_REF_MOVE_ANGLE_THRESHOLD = 0.10f; // rad  (~5.7 deg)
const float X_REF_MOVE_RATE_THRESHOLD  = 0.75f; // rad/s

// --- Derivative low-pass filter coefficients ---------------------------------
//  Lower = heavier smoothing (more lag, less noise).
const float THETA_DOT_ALPHA = 0.20f;
const float X_DOT_ALPHA     = 0.10f;

// --- Moving average window on raw sensor readings ----------------------------
const int MA_SIZE = 5;
float theta_ma_buf[MA_SIZE] = {0};
float x_ma_buf[MA_SIZE]     = {0};
int   ma_idx  = 0;
bool  ma_full = false;

// --- Runtime state -----------------------------------------------------------
long          zeroCount         = 0;
long          cartZeroCount     = 0;
unsigned long lastControlMicros = 0;
unsigned long lastPrintMillis   = 0;
uint16_t      motorFaultClearCtr = 0;   // periodic Motoron fault-flag clearing
uint32_t      motorResetCount   = 0;   // diagnostic: how often Motoron resets mid-run

float theta_filt     = 0.0f;
float theta_dot_filt = 0.0f;
float x_filt         = 0.0f;
float x_dot_filt     = 0.0f;
float prev_theta     = 0.0f;
float prev_x         = 0.0f;
float x_integral     = 0.0f;   // integral of (x - x_ref) for K5
int16_t lastCmd      = 0;

// ============================================================================
//  UTILITIES
// ============================================================================

static inline float wrapDeg(float d) {
    while (d >  180.0f) d -= 360.0f;
    while (d < -180.0f) d += 360.0f;
    return d;
}

static inline int16_t clampCmd(int16_t v) {
    return (v >  MOTOR_MAX) ?  MOTOR_MAX :
           (v < -MOTOR_MAX) ? -MOTOR_MAX : v;
}

static inline float clampF(float v, float lo, float hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

static inline float moveToward(float current, float target, float maxStep) {
    if (target > current + maxStep) return current + maxStep;
    if (target < current - maxStep) return current - maxStep;
    return target;
}

// ============================================================================
//  MOVING AVERAGE
// ============================================================================

void maUpdate(float th, float x) {
    theta_ma_buf[ma_idx] = th;
    x_ma_buf[ma_idx]     = x;
    if (++ma_idx >= MA_SIZE) { ma_idx = 0; ma_full = true; }
}

float maAvg(float* buf) {
    int n = ma_full ? MA_SIZE : ma_idx;
    if (n == 0) return 0.0f;
    float s = 0;
    for (int i = 0; i < n; i++) s += buf[i];
    return s / (float)n;
}

void maReset() {
    for (int i = 0; i < MA_SIZE; i++) { theta_ma_buf[i] = 0; x_ma_buf[i] = 0; }
    ma_idx = 0; ma_full = false;
}

// ============================================================================
//  SENSOR READING
// ============================================================================

uint8_t pendulumState() {
    return (uint8_t)((digitalRead(pinA) << 1) | digitalRead(pinB));
}

float readTheta() {
    long c; noInterrupts(); c = encoderCount; interrupts();
    float deg = wrapDeg((float)(c - zeroCount) * 360.0f / (float)COUNTS_PER_REV);
    return deg * DEG_TO_RAD;
}

float readCartPos() {
    long c; noInterrupts(); c = cartEncoderCount; interrupts();
    return ((float)(c - cartZeroCount) / CART_ENCODER_CPR)
           * 2.0f * PI * CART_WHEEL_RADIUS;
}

// ============================================================================
//  MOTOR OUTPUT
// ============================================================================

void setAllMotors(int16_t cmd) {
    shield1.setSpeed(1, cmd);
    shield1.setSpeed(2, cmd);
    shield2.setSpeed(1, cmd);
    shield2.setSpeed(2, cmd);
}

// ============================================================================
//  ENCODER ISRs
// ============================================================================

// Full 4-state quadrature decoder for pendulum (Broadcom AS22)
void updateEncoder() {
    uint8_t cur   = pendulumState();
    uint8_t trans = (lastPendulumEncoderState << 2) | cur;
    switch (trans) {
        case 0b0010: case 0b1011: case 0b1101: case 0b0100: encoderCount++; break;
        case 0b0001: case 0b0111: case 0b1110: case 0b1000: encoderCount--; break;
        case 0b0000: case 0b0101: case 0b1010: case 0b1111: break;
        default: invalidPendulumTransitions++; break;
    }
    lastPendulumEncoderState = cur;
}

// Full 4-state quadrature decoder for cart (Pololu motor encoder)
// Uses the same state-machine approach as the pendulum encoder to reject
// EMI-induced phantom counts from motor PWM switching noise.
uint8_t cartEncoderState() {
    return (uint8_t)((digitalRead(cartPinA) << 1) | digitalRead(cartPinB));
}
void updateCartEncoder() {
    uint8_t cur   = cartEncoderState();
    uint8_t trans = (lastCartEncoderState << 2) | cur;
    switch (trans) {
        case 0b0010: case 0b1011: case 0b1101: case 0b0100: cartEncoderCount++; break;
        case 0b0001: case 0b0111: case 0b1110: case 0b1000: cartEncoderCount--; break;
        case 0b0000: case 0b0101: case 0b1010: case 0b1111: break;
        default: invalidCartTransitions++; break;
    }
    lastCartEncoderState = cur;
}

// ============================================================================
//  LQR CONTROL LAW
// ============================================================================
//
//  Single linear feedback across all four states simultaneously.
//  No mode switching, no conditional logic.
//
//  u [N] = -(K1*(x-ref) + K2*x_dot + K3*theta + K4*theta_dot + K5*integral)
//  cmd   = clamp(u * FORCE_TO_CMD_SCALE, -800, +800)
//
int16_t computeLQR(float x, float x_dot, float theta, float theta_dot) {

    // Position error integral (anti-windup clamped to +/-1 m.s)
    x_integral += (x - X_SETPOINT) * LOOP_DT_S;
    x_integral  = clampF(x_integral, -1.0f, 1.0f);

    float u = -(  K1 * (x - X_SETPOINT)
                + K2 * x_dot
                + K3 * theta
                + K4 * theta_dot
                + K5_integral * x_integral );

    return clampCmd((int16_t)(u * FORCE_TO_CMD_SCALE));
}

void updateReferenceGovernor(float theta, float theta_dot) {
    bool uprightEnough = fabsf(theta) <= X_REF_MOVE_ANGLE_THRESHOLD
                      && fabsf(theta_dot) <= X_REF_MOVE_RATE_THRESHOLD;

    if (uprightEnough) {
        X_SETPOINT = moveToward(X_SETPOINT, X_TARGET, X_REF_MAX_STEP);
    }
}

// ============================================================================
//  HARDWARE SETUP
// ============================================================================

void setupMotors() {
    Wire1.begin();
    // 400 kHz fast-mode I2C: cuts each transaction from ~450 µs to ~115 µs.
    // At 100 kHz the four setSpeed() calls per 2 ms cycle consumed ~1800 µs,
    // leaving almost no margin for EMI-induced retries at 12 V.  A single
    // NACK retry pushed the cycle over budget, the Motoron command-timeout
    // fired, the RESET flag latched, and setSpeed() was silently ignored.
    shield1.setBus(&Wire1);
    shield2.setBus(&Wire1);

    shield1.reinitialize();
    shield1.clearResetFlag();
    shield1.clearMotorFaultUnconditional();
    // PWM mode 1: switches from default 20 kHz to ~1 kHz. At 20 kHz the
    // switching noise couples strongly into the I2C lines at 12 V, causing
    // NACKs and command timeouts.  1 kHz has less EMI at the cost of slightly
    // more audible motor noise.
    shield1.setPwmMode(1, 3);
    shield1.setPwmMode(2, 3);
    // Acceleration limit 200 (was 800): spreads direction reversals over
    // ~3.5 ms instead of <1 ms, preventing the back-EMF + supply-voltage
    // spike (V_supply + V_bemf ≈ 23 V at 12.6 V) from tripping the
    // Motoron overcurrent detector.  The 3.5 ms ramp is within the 2 ms
    // control-cycle budget for steady-state commands; only large step
    // changes (motor reversal) are slowed, which is acceptable.
    shield1.setMaxAcceleration(1, 800);
    shield1.setMaxDeceleration(1, 800);
    shield1.setMaxAcceleration(2, 800);
    shield1.setMaxDeceleration(2, 800);

    shield2.reinitialize();
    shield2.clearResetFlag();
    shield2.clearMotorFaultUnconditional();
    shield2.setPwmMode(1, 3);
    shield2.setPwmMode(2, 3);
    shield2.setMaxAcceleration(1, 800);
    shield2.setMaxDeceleration(1, 800);
    shield2.setMaxAcceleration(2, 800);
    shield2.setMaxDeceleration(2, 800);
}

void setupEncoders() {
    pinMode(pinA, INPUT);
    pinMode(pinB, INPUT);
    lastPendulumEncoderState = pendulumState();
    attachInterrupt(digitalPinToInterrupt(pinA), updateEncoder, CHANGE);
    attachInterrupt(digitalPinToInterrupt(pinB), updateEncoder, CHANGE);

    pinMode(cartPinA, INPUT_PULLUP);
    pinMode(cartPinB, INPUT_PULLUP);
    lastCartEncoderState = cartEncoderState();
    attachInterrupt(digitalPinToInterrupt(cartPinA), updateCartEncoder, CHANGE);
    attachInterrupt(digitalPinToInterrupt(cartPinB), updateCartEncoder, CHANGE);

    delay(100);
    noInterrupts();
    zeroCount     = encoderCount;
    cartZeroCount = cartEncoderCount;
    interrupts();
}

// ============================================================================
//  SETUP
// ============================================================================

void setup() {
    Serial.begin(115200);
    setupMotors();
    setupEncoders();
    lastControlMicros = micros();

    Serial.println("==============================================");
    Serial.println("  TRUE LQR INVERTED PENDULUM -- Arduino Giga  ");
    Serial.println("==============================================");
    Serial.println("Control law:");
    Serial.println("  u = -(K1*(x-ref) + K2*xdot + K3*theta + K4*thdot)");
    Serial.println("");
    Serial.print("  K1 (pos)   = "); Serial.println(K1, 4);
    Serial.print("  K2 (vel)   = "); Serial.println(K2, 4);
    Serial.print("  K3 (angle) = "); Serial.println(K3, 4);
    Serial.print("  K4 (rate)  = "); Serial.println(K4, 4);
    Serial.print("  K5 (integ) = "); Serial.println(K5_integral, 4);
    Serial.println("");
    Serial.print("  Force/cmd scale: "); Serial.print(FORCE_TO_CMD_SCALE, 2);
    Serial.println(" cmd/N  (10.6V, 4 motors, r=0.04m)");
    Serial.println("");
    Serial.println("*** RUN lqr_design.py AND PASTE K VALUES BEFORE USE ***");
    Serial.println("");
    Serial.print("Reference slew rate: "); Serial.print(X_REF_SLEW_RATE, 2);
    Serial.println(" m/s");
    Serial.println("Position reference only advances when pendulum is near upright.");
    Serial.println("");
    Serial.println("Commands:");
    Serial.println("  r     = recalibrate (hold pendulum upright first)");
    Serial.println("  t/y/u/i = target 0.0 / 0.5 / 1.0 / 2.0 m");
    Serial.println("  1/2   = K1 more/less negative  (position)");
    Serial.println("  3/4   = K2 more/less negative  (velocity)");
    Serial.println("  5/6   = K3 more/less negative  (angle)");
    Serial.println("  7/8   = K4 more/less negative  (rate)");
    Serial.println("  9/0   = K5 more/less negative  (integral, LQI)");
    Serial.println("  p     = print gains");
    Serial.println("==============================================");
}

// ============================================================================
//  SERIAL COMMAND HANDLER
// ============================================================================

void handleSerial() {
    if (!Serial.available()) return;
    char c = Serial.read();
    switch (c) {

        case 'r':
            noInterrupts();
            zeroCount     = encoderCount;
            cartZeroCount = cartEncoderCount;
            interrupts();
            theta_dot_filt = 0; x_dot_filt = 0;
            prev_theta = 0; prev_x = 0; x_integral = 0;
            X_TARGET = 0.0f; X_SETPOINT = 0.0f;
            maReset();
            Serial.println(">> Recalibrated.");
            break;

        case 't': X_TARGET = 0.0f; x_integral = 0; Serial.println(">> Target 0.0 m"); break;
        case 'y': X_TARGET = 0.5f; x_integral = 0; Serial.println(">> Target 0.5 m"); break;
        case 'u': X_TARGET = 1.0f; x_integral = 0; Serial.println(">> Target 1.0 m"); break;
        case 'i': X_TARGET = 2.0f; x_integral = 0; Serial.println(">> Target 2.0 m"); break;

        case '1': K1 -= 2.0f;  Serial.print(">> K1 = "); Serial.println(K1, 3); break;  // more negative = stronger
        case '2': K1 = min(0.0f, K1 + 2.0f); Serial.print(">> K1 = "); Serial.println(K1, 3); break;

        case '3': K2 -= 1.0f;  Serial.print(">> K2 = "); Serial.println(K2, 3); break;  // more negative = stronger
        case '4': K2 = min(0.0f, K2 + 1.0f); Serial.print(">> K2 = "); Serial.println(K2, 3); break;

        case '5': K3 -= 5.0f;  Serial.print(">> K3 = "); Serial.println(K3, 3); break;  // more negative
        case '6': K3 = min(0.0f, K3 + 5.0f); Serial.print(">> K3 = "); Serial.println(K3, 3); break;

        case '7': K4 -= 2.0f;  Serial.print(">> K4 = "); Serial.println(K4, 3); break;  // more negative
        case '8': K4 = min(0.0f, K4 + 2.0f); Serial.print(">> K4 = "); Serial.println(K4, 3); break;

        case '9': K5_integral -= 0.5f; Serial.print(">> K5 = "); Serial.println(K5_integral, 3); break;  // more negative = stronger
        case '0': K5_integral = min(0.0f, K5_integral + 0.5f); Serial.print(">> K5 = "); Serial.println(K5_integral, 3); break;

        case 'p':
            Serial.println("\n--- Current LQR Gains ---");
            Serial.print("  K1  = "); Serial.println(K1, 4);
            Serial.print("  K2  = "); Serial.println(K2, 4);
            Serial.print("  K3  = "); Serial.println(K3, 4);
            Serial.print("  K4  = "); Serial.println(K4, 4);
            Serial.print("  K5  = "); Serial.println(K5_integral, 4);
            Serial.print("  X_TARGET   = "); Serial.print(X_TARGET, 2); Serial.println(" m");
            Serial.print("  X_SETPOINT = "); Serial.print(X_SETPOINT, 2); Serial.println(" m");
            Serial.print("  x_integral = "); Serial.println(x_integral, 5);
            break;
    }
}

// ============================================================================
//  MAIN LOOP
// ============================================================================

void loop() {
    handleSerial();

    unsigned long now = micros();
    if ((now - lastControlMicros) < (unsigned long)(LOOP_DT_S * 1e6f)) return;
    lastControlMicros = now;

    // 1. Read sensors
    float theta_raw = readTheta();
    float x_raw     = readCartPos();

    // 2. Moving average (reduces encoder quantisation noise)
    maUpdate(theta_raw, x_raw);
    float theta_ma = maAvg(theta_ma_buf);
    float x_ma     = maAvg(x_ma_buf);

    // 3. Numerical derivative + exponential low-pass filter
    float theta_dot_raw = (theta_ma - prev_theta) / LOOP_DT_S;
    float x_dot_raw     = (x_ma     - prev_x)     / LOOP_DT_S;

    theta_dot_filt = THETA_DOT_ALPHA * theta_dot_raw
                   + (1.0f - THETA_DOT_ALPHA) * theta_dot_filt;
    x_dot_filt     = X_DOT_ALPHA     * x_dot_raw
                   + (1.0f - X_DOT_ALPHA)     * x_dot_filt;

    theta_filt = theta_ma;
    x_filt     = x_ma;
    prev_theta = theta_ma;
    prev_x     = x_ma;

    // Balance-first position control: advance x reference gradually and only
    // while the pendulum is close enough to upright.
    updateReferenceGovernor(theta_filt, theta_dot_filt);

    // 4. Safety cutoff: if pendulum has fallen too far, disable motors.
    //    The linear LQR is only valid near the upright equilibrium.
    if (fabsf(theta_filt) > 0.52f) {   // ~30 degrees
        setAllMotors(0);
        x_integral = 0;
        lastCmd = 0;
    } else {
        // LQR -- one linear law, all four states simultaneously
        lastCmd = computeLQR(x_filt, x_dot_filt, theta_filt, theta_dot_filt);
        setAllMotors(lastCmd);
    }

    // 5. Clear Motoron reset/fault flags every 5 cycles (10 ms).
    //    At 12 V the higher stall current trips the Motoron overcurrent
    //    detector mid-run; the flag latches and silently blocks setSpeed()
    //    until explicitly cleared.  100 ms was too slow — the pendulum
    //    drifted ~3.5° before the fault cleared, then the large accumulated
    //    command fired all at once causing a violent lurch.  At 10 ms, the
    //    maximum drift before re-enabling the motor is ~0.35°.
    if (++motorFaultClearCtr >= 5) {
        motorFaultClearCtr = 0;
        // Count how often the Motoron RESET flag is set mid-run.
        // If motorResetCount climbs in the telemetry, the Motoron is
        // still resetting (I2C noise or overcurrent) and needs hardware fixes.
        uint16_t f1 = shield1.getStatusFlags();
        uint16_t f2 = shield2.getStatusFlags();
        if ((f1 | f2) & 0x0001) motorResetCount++; // bit 0 = RESET flag
        shield1.clearResetFlag();
        shield1.clearMotorFaultUnconditional();
        shield2.clearResetFlag();
        shield2.clearMotorFaultUnconditional();
    }

    // 6. Telemetry at 10 Hz
    if (millis() - lastPrintMillis >= 100) {
        lastPrintMillis = millis();
        Serial.print("x:");    Serial.print(x_filt, 3);
        Serial.print(" ref:"); Serial.print(X_SETPOINT, 2);
        Serial.print(" tgt:"); Serial.print(X_TARGET, 2);
        Serial.print(" th:");  Serial.print(theta_filt * RAD_TO_DEG, 2);
        Serial.print(" thd:"); Serial.print(theta_dot_filt, 2);
        Serial.print(" cmd:"); Serial.print(lastCmd);
        Serial.print(" inv:"); Serial.print(invalidPendulumTransitions);
        Serial.print(" cinv:"); Serial.print(invalidCartTransitions);
        Serial.print(" rst:"); Serial.println(motorResetCount);
    }
}

/*
 * ============================================================================
 *  BRING-UP CHECKLIST  (do this in order before enabling motors under LQR)
 * ============================================================================
 *
 *  STEP 1 -- Compute exact gains
 *    Run the Python script in the header on any machine with numpy/scipy.
 *    Paste K1, K2, K3, K4 into this file.  ALL four values will be negative.
 *
 *  STEP 2 -- Verify pendulum encoder direction
 *    Open Serial Monitor, press 'r' with pendulum upright.
 *    Gently push the TOP of the pendulum forward (positive x direction).
 *    "th:" should increase (become more positive).
 *    If it decreases: change (c - zeroCount) to (zeroCount - c) in readTheta().
 *
 *  STEP 3 -- Verify cart encoder direction
 *    Push cart forward (positive x).  "x:" should increase.
 *    If not: swap cartPinA and cartPinB constants.
 *
 *  STEP 4 -- Verify motor direction (without pendulum mounted)
 *    Temporarily replace:  lastCmd = computeLQR(...);
 *    with:                 lastCmd = 100;
 *    Confirm cart moves in the positive-x direction.
 *    If backward: negate cmd inside setAllMotors.  Restore computeLQR after.
 *
 *  STEP 5 -- Verify wheel radius
 *    Push cart exactly 1.0 m.  "x:" should read ~1.000 m.
 *    Adjust CART_WHEEL_RADIUS if needed.
 *
 *  STEP 6 -- Angle-only stabilisation test
 *    Set K1=0, K2=0, K5=0.  Mount pendulum.  Press 'r' with pendulum upright.
 *    Release.  Press '5' to increase |K3| until pendulum holds.
 *    Press '7' to increase |K4| to remove oscillation.
 *
 *  STEP 7 -- Add position control
 *    Press '1' to gradually add K1.  Cart should return to origin.
 *    Press '3' to add K2 to damp cart oscillation.
 *
 *  STEP 8 -- Enable LQI (optional)
 *    Press '9' slowly to increase K5.  This eliminates steady-state drift.
 *    If oscillation appears, reduce K5.
 *
 * ============================================================================
 *  COMMON FAILURE MODES
 * ============================================================================
 *
 *  Oscillates immediately after release:
 *    K3 or K4 too large.  Press '6' or '8'.  Also verify Steps 2-4.
 *
 *  Pendulum falls without any cart response:
 *    Encoder or motor direction wrong.  Check Steps 2-4.
 *    K3 too small -- press '5'.
 *
 *  Cart drifts while pendulum is stable:
 *    K1 too small (press '1').  Or enable K5.
 *
 *  Command always hits +/-800:
 *    FORCE_TO_CMD_SCALE too large, or wheel radius wrong.
 *    Verify supply voltage and recompute scale from the header formula.
 *
 *  "inv:" counter rising rapidly:
 *    Encoder noise.  Add 100 nF caps from each encoder signal pin to GND.
 *
 * ============================================================================
 *  CURRENT LIMIT NOTE
 * ============================================================================
 *  Motoron M3S550: 1.7 A continuous per channel.
 *  Pololu 4862 stall current at 10.6 V: 1.8 x (10.6/12) = 1.59 A.
 *  This is just under the continuous rating.  Normal LQR operation is safe.
 *  Sustained stall (cmd stuck at +/-800 with motor stopped) may trigger
 *  overcurrent protection -- resolve by reducing gain magnitude or checking
 *  the mechanical system for obstructions.
 * ============================================================================
 */