/*
 * Pole placement inverted pendulum controller
 * Arduino Giga
 *
 * Hardware identical to LQRPendulum.ino.
 * Poles placed at LQR closed-loop eigenvalues (Q=diag([5,1,100,10]), R=0.01)
 * using Ackermann's formula -- gains come out identical to LQR.
 *
 * State:  [x (m), x_dot (m/s), theta (rad), theta_dot (rad/s)]
 *
 *  A = [0,  1,           0,  0]        B = [0      ]
 *      [0,  0,      -mg/M,  0]             [1/M    ]
 *      [0,  0,           0,  1]             [0      ]
 *      [0,  0,  (M+m)g/Ml,  0]             [-1/(Ml)]
 *
 *  With M=1.2, m=0.91, l=0.5:
 *    A[1,2] = -7.437   A[3,2] = +34.499   (unstable at ±5.87 rad/s)
 *
 * Gain computation -- run this Python script, paste K values below:
 *
 *  #!/usr/bin/env python3
 *  import numpy as np
 *  from scipy.linalg import solve_continuous_are
 *  from scipy.signal import place_poles
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
 *  # ── Step 1: LQR to obtain the target closed-loop poles ──────────────────
 *  Q = np.diag([5.0, 1.0, 100.0, 10.0])
 *  R = np.array([[0.01]])
 *  P      = solve_continuous_are(A, B, Q, R)
 *  K_lqr  = (np.linalg.inv(R) @ B.T @ P).flatten()
 *  lqr_poles = np.linalg.eigvals(A - B @ K_lqr.reshape(1,-1))
 *  print(f"LQR K  : {K_lqr}")
 *  print(f"LQR poles: {np.sort_complex(lqr_poles)}")
 *
 *  # ── Step 2: Pole Placement to the same poles ─────────────────────────────
 *  result = place_poles(A, B, lqr_poles)
 *  K = result.gain_matrix.flatten()
 *  pp_poles = np.linalg.eigvals(A - B @ K.reshape(1,-1))
 *
 *  print(f"\nPole Placement K:")
 *  print(f"K1={K[0]:.4f}  K2={K[1]:.4f}  K3={K[2]:.4f}  K4={K[3]:.4f}")
 *  print(f"Closed-loop poles: {np.sort_complex(pp_poles)}")
 *  print(f"All stable: {all(np.real(p) < 0 for p in pp_poles)}")
 *  print(f"Gains match LQR: {np.allclose(K, K_lqr, rtol=1e-4)}")
 *
 * Expected output (Q=diag([5,1,100,10]), R=0.01):
 *   K1 = -22.3607   K2 = -31.3222   K3 = -213.9713   K4 = -51.2750
 *   All gains negative because B[3] = -1/(M*l) < 0.
 */

#include <Motoron.h>
#include <Wire.h>

// --- Motor shields -----------------------------------------------------------
MotoronI2C shield1(16);
MotoronI2C shield2(17);
// 1.7 A Motoron limit / 1.845 A stall @ 12.3 V = 92% → 730 with margin
const int16_t MOTOR_MAX = 730;

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
const float CART_ENCODER_CPR  = 464.64f;
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

// Force-to-command scale (12.3 V, 4 motors, r=0.04 m). Geometric = 42.72,
// empirically tuned to 18.72 for this hardware.
const float FORCE_TO_CMD_SCALE = 18.72f;

// Pole placement gains -- paste output of Python script here
// u = -(K1*(x-x_ref) + K2*x_dot + K3*theta + K4*theta_dot)
float K1 =  22.3607f;     // N/m
float K2 =  55.3222f;     // N·s/m
float K3 = -220.9713f;    // N/rad
float K4 =  -55.2750f;    // N·s/rad

// Optional integral term.  Eliminates steady-state position offset.
// Set to 0.0 during initial bring-up.  Compute from augmented Python script.
float K5_integral = -0.0f;   // N / (m.s)

float X_TARGET   = -2.0f;    // commanded cart position (m)
float X_SETPOINT = 0.0f;    // governed cart position reference used by PP (m)

// --- Loop timing -------------------------------------------------------------
const float LOOP_DT_S = 0.002f;   // 2 ms = 500 Hz

// --- Position reference governor -------------------------------------------
// Large step changes in x_ref make the controller fight position error too
// hard and sacrifice angle stability.  Move the reference slowly and only
// while upright.
const float X_REF_SLEW_RATE = 0.1f;        // m/s
const float X_REF_MAX_STEP  = X_REF_SLEW_RATE * LOOP_DT_S;
const float X_REF_MOVE_ANGLE_THRESHOLD = 0.10f; // rad  (~5.7 deg)
const float X_REF_MOVE_RATE_THRESHOLD  = 0.75f; // rad/s

// --- Derivative low-pass filter coefficients ---------------------------------
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

float theta_filt     = 0.0f;
float theta_dot_filt = 0.0f;
float x_filt         = 0.0f;
float x_dot_filt     = 0.0f;
float prev_theta     = 0.0f;
float prev_x         = 0.0f;
float x_integral     = 0.0f;   // integral of (x - x_ref) for K5
int16_t lastCmd      = 0;
uint16_t motorFaultClearCtr = 0;
uint32_t motorResetCount    = 0;   // diagnostic: how often Motoron resets mid-run

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
//  POLE PLACEMENT CONTROL LAW
// ============================================================================
//
//  Single linear feedback across all four states simultaneously.
//  No mode switching, no conditional logic.
//
//  Gains K1-K4 are computed offline by placing the closed-loop poles of
//  (A - B*K) at the desired locations (identical to the LQR closed-loop poles).
//
//  u [N] = -(K1*(x-ref) + K2*x_dot + K3*theta + K4*theta_dot + K5*integral)
//  cmd   = clamp(u * FORCE_TO_CMD_SCALE, -800, +800)
//
int16_t computePP(float x, float x_dot, float theta, float theta_dot) {

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
    Wire1.setClock(400000);   // 400 kHz fast-mode: prevents I2C timeout at 12.3 V
    shield1.setBus(&Wire1);
    shield2.setBus(&Wire1);

    shield1.reinitialize();
    shield1.clearResetFlag();
    shield1.clearMotorFaultUnconditional();
    // Max acceleration: no ramping — commands must take effect within one 2 ms timestep.
    shield1.setMaxAcceleration(1, 800);
    shield1.setMaxDeceleration(1, 800);
    shield1.setMaxAcceleration(2, 800);
    shield1.setMaxDeceleration(2, 800);

    shield2.reinitialize();
    shield2.clearResetFlag();
    shield2.clearMotorFaultUnconditional();
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
    Serial.println("  POLE PLACEMENT INVERTED PENDULUM -- Arduino Giga  ");
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
    Serial.println("*** RUN pole_placement_design.py AND PASTE K VALUES BEFORE USE ***");
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
    Serial.println("  9/0   = K5 more/less negative  (integral)");
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

        case '1': K1 -= 2.0f;  Serial.print(">> K1 = "); Serial.println(K1, 3); break;
        case '2': K1 = min(0.0f, K1 + 2.0f); Serial.print(">> K1 = "); Serial.println(K1, 3); break;

        case '3': K2 -= 1.0f;  Serial.print(">> K2 = "); Serial.println(K2, 3); break;
        case '4': K2 = min(0.0f, K2 + 1.0f); Serial.print(">> K2 = "); Serial.println(K2, 3); break;

        case '5': K3 -= 5.0f;  Serial.print(">> K3 = "); Serial.println(K3, 3); break;
        case '6': K3 = min(0.0f, K3 + 5.0f); Serial.print(">> K3 = "); Serial.println(K3, 3); break;

        case '7': K4 -= 2.0f;  Serial.print(">> K4 = "); Serial.println(K4, 3); break;
        case '8': K4 = min(0.0f, K4 + 2.0f); Serial.print(">> K4 = "); Serial.println(K4, 3); break;

        case '9': K5_integral -= 0.5f; Serial.print(">> K5 = "); Serial.println(K5_integral, 3); break;
        case '0': K5_integral = min(0.0f, K5_integral + 0.5f); Serial.print(">> K5 = "); Serial.println(K5_integral, 3); break;

        case 'p':
            Serial.println("\n--- Current Pole Placement Gains ---");
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
    //    The linearised PP law is only valid near the upright equilibrium.
    if (fabsf(theta_filt) > 0.52f) {   // ~30 degrees
        setAllMotors(0);
        x_integral = 0;
        lastCmd = 0;
    } else {
        // Pole Placement -- one linear law, all four states simultaneously
        lastCmd = computePP(x_filt, x_dot_filt, theta_filt, theta_dot_filt);
        setAllMotors(lastCmd);
    }

    // 5. Clear Motoron reset/fault flags every 5 cycles (10 ms)
    if (++motorFaultClearCtr >= 5) {
        motorFaultClearCtr = 0;
        uint16_t f1 = shield1.getStatusFlags();
        uint16_t f2 = shield2.getStatusFlags();
        if ((f1 | f2) & 0x0001) motorResetCount++;
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

