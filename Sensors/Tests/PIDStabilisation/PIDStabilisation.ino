#include <Motoron.h>
#include <Wire.h>

// ============================================================================
//  CASCADED PID INVERTED PENDULUM CONTROLLER
//  Arduino Giga / Mega compatible
// ============================================================================
//
//  Matches src/controller.py PIDController exactly:
//    Outer loop:  position error  -->  angle setpoint
//    Inner loop:  angle error     -->  force (N)
//
//  Hardware identical to LQRPendulum.ino:
//    Motors  : 4x Pololu #4862, 9.68:1 gearbox, 12 V nominal
//    Driver  : Motoron M3S550, speed range -800..+800
//    Pendulum: Broadcom AS22, 1000 CPR x4 = 4000 counts/rev
//    Cart    : Pololu motor encoder, 464.64 counts/rev of gearbox shaft
//    Supply  : 10.6 V  -->  F_max = 14.73 N  -->  FORCE_TO_CMD = 54.33
//
// ============================================================================

// --- Motor shields -----------------------------------------------------------
MotoronI2C shield1(16);
MotoronI2C shield2(17);
const int16_t MOTOR_MAX = 800;

// --- Pendulum encoder (Broadcom AS22, 1000 CPR x4 quadrature) ---------------
const long COUNTS_PER_REV = 4000;
const int pinA = 2;
const int pinB = 3;
volatile long          encoderCount               = 0;
volatile uint8_t       lastPendulumEncoderState   = 0;
volatile unsigned long invalidPendulumTransitions = 0;

// --- Cart encoder (Pololu 4862, 464.64 counts/rev) ---------------------------
const float CART_ENCODER_CPR  = 464.64f;
const float CART_WHEEL_RADIUS = 0.04f;   // metres
const int   cartPinA          = 24;
const int   cartPinB          = 26;
volatile long          cartEncoderCount       = 0;
volatile uint8_t       lastCartEncoderState   = 0;
volatile unsigned long invalidCartTransitions = 0;

// ============================================================================
//  PHYSICAL PARAMETERS  (match LQRPendulum.ino / src/controller.py)
// ============================================================================
// Not used in the PID law itself, but kept here for reference / future use.
const float M_CART = 1.2f;   // kg
const float M_PEND = 0.91f;  // kg
const float L_PEND = 0.5f;   // m  (pivot to CoM)

// ============================================================================
//  FORCE-TO-COMMAND SCALE  (identical to LQRPendulum.ino)
// ============================================================================
const float FORCE_TO_CMD_SCALE = 54.33f;

// ============================================================================
//  CASCADED PID GAINS  (from src/controller.py PIDController defaults)
// ============================================================================
//  Inner loop  -- angle PID  (u in Newtons)
float KP     = 55.0f;    // N / rad
float KI     = 2.0f;     // N / (rad·s)
float KD     = 26.0f;    // N·s / rad   (applied to theta_dot directly)

//  Outer loop  -- position PID  --> desired angle setpoint (rad)
float KP_POS = 0.05f;   // rad / m
float KI_POS = 0.004f;  // rad / (m·s)
float KD_POS = 0.11f;   // rad·s / m   (applied to x_dot directly)

const float MAX_ANGLE_SETPOINT = 0.09f;   // rad  (~5.2 deg) -- matches sim clamp
const float ANGLE_INTEGRAL_CLAMP = 100.0f;
const float POS_INTEGRAL_CLAMP   = 10.0f;

float X_TARGET   = 0.0f;   // commanded cart position (m)
float X_SETPOINT = 0.0f;   // governed reference sent to PID outer loop (m)

// --- Position reference governor -------------------------------------------
// Advances X_SETPOINT toward X_TARGET at a limited slew rate, and only while
// the pendulum is close enough to upright (matches LQRPendulum.ino behaviour).
const float X_REF_SLEW_RATE            = 0.1f;                          // m/s
const float X_REF_MAX_STEP             = X_REF_SLEW_RATE * 0.002f;      // m/tick @ 500 Hz
const float X_REF_MOVE_ANGLE_THRESHOLD = 0.10f;                         // rad (~5.7 deg)
const float X_REF_MOVE_RATE_THRESHOLD  = 0.75f;                         // rad/s

// ============================================================================
//  LOOP TIMING
// ============================================================================
const float LOOP_DT_S = 0.002f;   // 2 ms = 500 Hz  (same as LQRPendulum)

// ============================================================================
//  FILTER PARAMETERS  (identical to LQRPendulum.ino)
// ============================================================================
const float THETA_DOT_ALPHA = 0.20f;
const float X_DOT_ALPHA     = 0.10f;
const int   MA_SIZE         = 5;

float theta_ma_buf[MA_SIZE] = {0};
float x_ma_buf[MA_SIZE]     = {0};
int   ma_idx  = 0;
bool  ma_full = false;

// ============================================================================
//  RUNTIME STATE
// ============================================================================
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

float angle_integral = 0.0f;
float pos_integral   = 0.0f;

int16_t lastCmd = 0;

// ============================================================================
//  UTILITIES
// ============================================================================

static inline float wrapRad(float r) {
    while (r >  PI) r -= 2.0f * PI;
    while (r < -PI) r += 2.0f * PI;
    return r;
}

static inline int16_t clampCmd(int16_t v) {
    return (v >  MOTOR_MAX) ?  MOTOR_MAX :
           (v < -MOTOR_MAX) ? -MOTOR_MAX : v;
}

static inline float clampF(float v, float lo, float hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

static inline float moveToward(float cur, float tgt, float step) {
    if (tgt > cur + step) return cur + step;
    if (tgt < cur - step) return cur - step;
    return tgt;
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
    float deg = (float)(c - zeroCount) * 360.0f / (float)COUNTS_PER_REV;
    while (deg >  180.0f) deg -= 360.0f;
    while (deg < -180.0f) deg += 360.0f;
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
//  ENCODER ISRs  (identical to LQRPendulum.ino)
// ============================================================================

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
//  CASCADED PID CONTROL LAW  (matches src/controller.py PIDController)
// ============================================================================
//
//  Outer loop: angle_setpoint = KP_POS*(x_ref-x) + KI_POS*pos_integral - KD_POS*x_dot
//  Inner loop: force = KP*angle_error + KI*angle_integral + KD*theta_dot
//
int16_t computePID(float x, float x_dot, float theta, float theta_dot) {

    // --- Outer loop: position --> angle setpoint ----------------------------
    float pos_error = X_SETPOINT - x;
    pos_integral = clampF(pos_integral + pos_error * LOOP_DT_S,
                          -POS_INTEGRAL_CLAMP, POS_INTEGRAL_CLAMP);
    float angle_setpoint = KP_POS * pos_error
                         + KI_POS * pos_integral
                         - KD_POS * x_dot;
    angle_setpoint = clampF(angle_setpoint, -MAX_ANGLE_SETPOINT, MAX_ANGLE_SETPOINT);

    // --- Inner loop: angle --> force ----------------------------------------
    float angle_error = theta - angle_setpoint;
    angle_integral = clampF(angle_integral + angle_error * LOOP_DT_S,
                            -ANGLE_INTEGRAL_CLAMP, ANGLE_INTEGRAL_CLAMP);

    float force = KP * angle_error
                + KI * angle_integral
                + KD * theta_dot;
    force = clampF(force, -100.0f, 100.0f);

    return clampCmd((int16_t)(force * FORCE_TO_CMD_SCALE));
}

// ============================================================================
//  REFERENCE GOVERNOR  (identical to LQRPendulum.ino / PolePlacementPendulum.ino)
// ============================================================================

void updateReferenceGovernor(float theta, float theta_dot) {
    bool uprightEnough = fabsf(theta)     <= X_REF_MOVE_ANGLE_THRESHOLD
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
    shield1.setBus(&Wire1);
    shield2.setBus(&Wire1);

    shield1.reinitialize();
    shield1.clearResetFlag();
    shield1.setMaxAcceleration(1, 800);
    shield1.setMaxDeceleration(1, 800);
    shield1.setMaxAcceleration(2, 800);
    shield1.setMaxDeceleration(2, 800);

    shield2.reinitialize();
    shield2.clearResetFlag();
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
    Serial.println("  CASCADED PID INVERTED PENDULUM -- 500 Hz   ");
    Serial.println("==============================================");
    Serial.println("Control law:");
    Serial.println("  angle_sp = KP_POS*(ref-x) + KI_POS*int - KD_POS*xdot");
    Serial.println("  force    = KP*angle_err + KI*int + KD*thdot");
    Serial.println("");
    Serial.print("  KP      = "); Serial.println(KP, 4);
    Serial.print("  KI      = "); Serial.println(KI, 4);
    Serial.print("  KD      = "); Serial.println(KD, 4);
    Serial.print("  KP_POS  = "); Serial.println(KP_POS, 4);
    Serial.print("  KI_POS  = "); Serial.println(KI_POS, 4);
    Serial.print("  KD_POS  = "); Serial.println(KD_POS, 4);
    Serial.print("  Max angle setpoint: +/-"); Serial.print(MAX_ANGLE_SETPOINT * RAD_TO_DEG, 2);
    Serial.println(" deg");
    Serial.println("");
    Serial.println("Commands:");
    Serial.println("  r       = recalibrate (hold pendulum upright first)");
    Serial.println("  t/y/u/i = target 0.0 / 0.5 / 1.0 / 2.0 m");
    Serial.println("  p       = print gains");
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
            prev_theta = 0; prev_x = 0;
            angle_integral = 0; pos_integral = 0;
            X_TARGET = 0.0f; X_SETPOINT = 0.0f;
            maReset();
            Serial.println(">> Recalibrated.");
            break;

        case 't': X_TARGET = 0.0f; Serial.println(">> Target 0.0 m"); break;
        case 'y': X_TARGET = 0.5f; Serial.println(">> Target 0.5 m"); break;
        case 'u': X_TARGET = 1.0f; Serial.println(">> Target 1.0 m"); break;
        case 'i': X_TARGET = 2.0f; Serial.println(">> Target 2.0 m"); break;

        case 'p':
            Serial.println("\n--- Current PID Gains ---");
            Serial.print("  KP     = "); Serial.println(KP, 4);
            Serial.print("  KI     = "); Serial.println(KI, 4);
            Serial.print("  KD     = "); Serial.println(KD, 4);
            Serial.print("  KP_POS = "); Serial.println(KP_POS, 4);
            Serial.print("  KI_POS = "); Serial.println(KI_POS, 4);
            Serial.print("  KD_POS = "); Serial.println(KD_POS, 4);
            Serial.print("  X_TARGET   = "); Serial.print(X_TARGET, 2); Serial.println(" m");
            Serial.print("  X_SETPOINT = "); Serial.print(X_SETPOINT, 2); Serial.println(" m");
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

    // 2. Moving average
    maUpdate(theta_raw, x_raw);
    float theta_ma = maAvg(theta_ma_buf);
    float x_ma     = maAvg(x_ma_buf);

    // 3. Numerical derivative + exponential LP
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

    // 4. Reference governor: advance X_SETPOINT toward X_TARGET while upright
    updateReferenceGovernor(theta_filt, theta_dot_filt);

    // 5. Safety cutoff: PID linearisation only valid near upright
    if (fabsf(theta_filt) > 0.52f) {   // ~30 degrees
        setAllMotors(0);
        angle_integral = 0;
        pos_integral   = 0;
        lastCmd = 0;
    } else {
        lastCmd = computePID(x_filt, x_dot_filt, theta_filt, theta_dot_filt);
        setAllMotors(lastCmd);
    }

    // 6. Telemetry at 10 Hz
    if (millis() - lastPrintMillis >= 100) {
        lastPrintMillis = millis();
        Serial.print("x:");    Serial.print(x_filt, 3);
        Serial.print(" ref:"); Serial.print(X_SETPOINT, 2);
        Serial.print(" tgt:"); Serial.print(X_TARGET, 2);
        Serial.print(" th:");  Serial.print(theta_filt * RAD_TO_DEG, 2);
        Serial.print(" thd:"); Serial.print(theta_dot_filt, 2);
        Serial.print(" cmd:"); Serial.println(lastCmd);
    }
}
