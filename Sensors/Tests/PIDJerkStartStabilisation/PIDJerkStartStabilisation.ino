#include <Motoron.h>
#include <Wire.h>

// ============================================================================
//  PID JERK-START + STABILISATION CONTROLLER
//  Arduino Giga / Mega compatible
// ============================================================================
//
//  Adds a jerk-start launch sequence to PIDStabilisation.ino.
//  The cart gives a short kick in the lean direction, briefly brakes, then
//  hands off to the cascaded PID stabiliser once the pendulum re-enters the
//  capture window.
//
//  Hardware identical to LQRPendulum.ino / PIDStabilisation.ino.
//  PID gains identical to src/controller.py PIDController defaults.
//
// ============================================================================

// --- Motor shields -----------------------------------------------------------
MotoronI2C shield1(16);
MotoronI2C shield2(17);
const int16_t MOTOR_MAX = 800;
const uint16_t STABILIZE_RAMP_LIMIT = 800;
const uint16_t JERK_RAMP_LIMIT      = 2047;

// --- Pendulum encoder (Broadcom AS22, 1000 CPR x4 quadrature) ---------------
const long COUNTS_PER_REV = 4000;
const int pinA = 2;
const int pinB = 3;
volatile long          encoderCount               = 0;
volatile uint8_t       lastPendulumEncoderState   = 0;
volatile unsigned long invalidPendulumTransitions = 0;

// --- Cart encoder (Pololu 4862, 464.64 counts/rev) ---------------------------
const float CART_ENCODER_CPR  = 464.64f;
const float CART_WHEEL_RADIUS = 0.04f;
const int   cartPinA          = 24;
const int   cartPinB          = 26;
volatile long          cartEncoderCount       = 0;
volatile uint8_t       lastCartEncoderState   = 0;
volatile unsigned long invalidCartTransitions = 0;

// ============================================================================
//  FORCE-TO-COMMAND SCALE  (identical to LQRPendulum.ino)
// ============================================================================
const float FORCE_TO_CMD_SCALE = 54.33f;

// ============================================================================
//  CASCADED PID GAINS  (src/controller.py PIDController defaults)
// ============================================================================
float KP     = 55.0f;
float KI     = 2.0f;
float KD     = 26.0f;
float KP_POS = 0.05f;
float KI_POS = 0.004f;
float KD_POS = 0.11f;

const float MAX_ANGLE_SETPOINT   = 0.09f;
const float ANGLE_INTEGRAL_CLAMP = 100.0f;
const float POS_INTEGRAL_CLAMP   =  10.0f;

float X_TARGET = 0.0f;

float angle_integral = 0.0f;
float pos_integral   = 0.0f;

// ============================================================================
//  JERK-START PARAMETERS
// ============================================================================
const float          JERK_START_MIN_ANGLE     = 0.03f;   // ~1.7 deg minimum lean
const float          JERK_CAPTURE_ANGLE       = 0.05f;   // rad -- enter STABILIZE
const float          JERK_CAPTURE_RATE        = 1.0f;    // rad/s
const int16_t        JERK_INITIAL_CMD         = 160;
const int16_t        JERK_CMD_STEP            = 30;
const unsigned long  JERK_ACCEL_MS            = 200;
const unsigned long  JERK_BRAKE_MS            = 200;
const unsigned long  JERK_CAPTURE_TIMEOUT_MS  = 30000;

// ============================================================================
//  LOOP TIMING AND FILTER
// ============================================================================
const float LOOP_DT_S       = 0.002f;
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
int16_t lastCmd      = 0;

enum ControlMode { MODE_IDLE, MODE_JERK_ACCEL, MODE_JERK_BRAKE, MODE_STABILIZE };
ControlMode controlMode = MODE_IDLE;
unsigned long controlModeStartMillis  = 0;
unsigned long jerkSequenceStartMillis = 0;
int     jerkDirection   = 0;
int16_t jerkCurrentCmd  = JERK_INITIAL_CMD;

// ============================================================================
//  UTILITIES
// ============================================================================

static inline float clampF(float v, float lo, float hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

static inline int16_t clampCmd(int16_t v) {
    return (v >  MOTOR_MAX) ?  MOTOR_MAX :
           (v < -MOTOR_MAX) ? -MOTOR_MAX : v;
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

void setMotorRampLimits(uint16_t limit) {
    shield1.setMaxAcceleration(1, limit); shield1.setMaxDeceleration(1, limit);
    shield1.setMaxAcceleration(2, limit); shield1.setMaxDeceleration(2, limit);
    shield2.setMaxAcceleration(1, limit); shield2.setMaxDeceleration(1, limit);
    shield2.setMaxAcceleration(2, limit); shield2.setMaxDeceleration(2, limit);
}

// ============================================================================
//  ENCODER ISRs
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
//  CASCADED PID CONTROL LAW  (src/controller.py PIDController)
// ============================================================================

int16_t computePID(float x, float x_dot, float theta, float theta_dot) {
    float pos_error = X_TARGET - x;
    pos_integral = clampF(pos_integral + pos_error * LOOP_DT_S,
                          -POS_INTEGRAL_CLAMP, POS_INTEGRAL_CLAMP);
    float angle_setpoint = KP_POS * pos_error
                         + KI_POS * pos_integral
                         - KD_POS * x_dot;
    angle_setpoint = clampF(angle_setpoint, -MAX_ANGLE_SETPOINT, MAX_ANGLE_SETPOINT);

    float angle_error = theta - angle_setpoint;
    angle_integral = clampF(angle_integral + angle_error * LOOP_DT_S,
                            -ANGLE_INTEGRAL_CLAMP, ANGLE_INTEGRAL_CLAMP);

    float force = clampF(KP * angle_error + KI * angle_integral + KD * theta_dot,
                         -100.0f, 100.0f);
    return clampCmd((int16_t)(force * FORCE_TO_CMD_SCALE));
}

// ============================================================================
//  JERK-START STATE MACHINE
// ============================================================================

void resetControlStates() {
    theta_dot_filt = 0; x_dot_filt = 0;
    theta_filt = 0; x_filt = 0;
    prev_theta = 0; prev_x = 0;
    angle_integral = 0; pos_integral = 0;
    lastCmd = 0;
    maReset();
}

void setControlMode(ControlMode newMode) {
    controlMode = newMode;
    controlModeStartMillis = millis();
    setMotorRampLimits((newMode == MODE_JERK_ACCEL || newMode == MODE_JERK_BRAKE)
                       ? JERK_RAMP_LIMIT : STABILIZE_RAMP_LIMIT);
}

bool startJerkSequence(float theta, float theta_dot) {
    if (fabsf(theta) < JERK_START_MIN_ANGLE) {
        Serial.print(">> Jerk refused. Lean pendulum at least ");
        Serial.print(JERK_START_MIN_ANGLE * RAD_TO_DEG, 1);
        Serial.println(" deg first.");
        return false;
    }
    jerkDirection   = (theta >= 0.0f) ? 1 : -1;
    jerkCurrentCmd  = JERK_INITIAL_CMD;
    resetControlStates();
    prev_theta = theta;
    jerkSequenceStartMillis = millis();
    setControlMode(MODE_JERK_ACCEL);
    setAllMotors((int16_t)(jerkDirection * jerkCurrentCmd));
    Serial.print(">> Jerk start at "); Serial.print(theta * RAD_TO_DEG, 1);
    Serial.print(" deg, dir="); Serial.println(jerkDirection > 0 ? "right" : "left");
    return true;
}

int16_t updateJerkStart(float theta, float theta_dot) {
    unsigned long elapsed    = millis() - controlModeStartMillis;
    unsigned long jerkElapsed = millis() - jerkSequenceStartMillis;

    // Timeout
    if (controlMode == MODE_JERK_ACCEL || controlMode == MODE_JERK_BRAKE) {
        if (jerkElapsed >= JERK_CAPTURE_TIMEOUT_MS) {
            setControlMode(MODE_IDLE);
            Serial.println(">> Jerk timed out. Motors idle.");
            return 0;
        }
    }

    // Capture check
    if (controlMode != MODE_IDLE && controlMode != MODE_STABILIZE &&
        fabsf(theta) <= JERK_CAPTURE_ANGLE && fabsf(theta_dot) <= JERK_CAPTURE_RATE) {
        // Rezero cart so PID starts from x=0
        noInterrupts(); cartZeroCount = cartEncoderCount; interrupts();
        x_filt = 0; x_dot_filt = 0; prev_x = 0;
        for (int i = 0; i < MA_SIZE; i++) x_ma_buf[i] = 0;
        X_TARGET = 0.0f;
        angle_integral = 0; pos_integral = 0;
        setControlMode(MODE_STABILIZE);
        Serial.println(">> Captured. Cart zeroed. PID stabilisation engaged.");
        return computePID(0.0f, 0.0f, theta, theta_dot);
    }

    switch (controlMode) {
        case MODE_JERK_ACCEL:
            if (elapsed >= JERK_ACCEL_MS) {
                setControlMode(MODE_JERK_BRAKE);
                return (int16_t)(-jerkDirection * jerkCurrentCmd);
            }
            return (int16_t)(jerkDirection * jerkCurrentCmd);

        case MODE_JERK_BRAKE:
            if (elapsed >= JERK_BRAKE_MS) {
                jerkCurrentCmd = (int16_t)min((int)jerkCurrentCmd + JERK_CMD_STEP, (int)MOTOR_MAX);
                setControlMode(MODE_JERK_ACCEL);
                return (int16_t)(jerkDirection * jerkCurrentCmd);
            }
            return (int16_t)(-jerkDirection * jerkCurrentCmd);

        case MODE_STABILIZE:
            return computePID(x_filt, x_dot_filt, theta, theta_dot);

        case MODE_IDLE:
        default:
            return 0;
    }
}

const char* modeName() {
    switch (controlMode) {
        case MODE_JERK_ACCEL:  return "JERK_ACCEL";
        case MODE_JERK_BRAKE:  return "JERK_BRAKE";
        case MODE_STABILIZE:   return "STABILIZE";
        case MODE_IDLE: default: return "IDLE";
    }
}

// ============================================================================
//  HARDWARE SETUP
// ============================================================================

void setupMotors() {
    Wire1.begin();
    shield1.setBus(&Wire1); shield2.setBus(&Wire1);
    shield1.reinitialize(); shield1.clearResetFlag();
    shield2.reinitialize(); shield2.clearResetFlag();
    setMotorRampLimits(STABILIZE_RAMP_LIMIT);
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
    zeroCount = encoderCount; cartZeroCount = cartEncoderCount;
    interrupts();
}

// ============================================================================
//  SETUP
// ============================================================================

void setup() {
    Serial.begin(115200);
    setupMotors();
    setupEncoders();
    setControlMode(MODE_IDLE);
    lastControlMicros = micros();

    Serial.println("==============================================");
    Serial.println("  PID JERK-START + STABILISATION -- 500 Hz   ");
    Serial.println("==============================================");
    Serial.print("  KP="); Serial.print(KP);
    Serial.print("  KI="); Serial.print(KI);
    Serial.print("  KD="); Serial.println(KD);
    Serial.print("  KP_POS="); Serial.print(KP_POS);
    Serial.print("  KI_POS="); Serial.print(KI_POS);
    Serial.print("  KD_POS="); Serial.println(KD_POS);
    Serial.println("");
    Serial.println("Commands:");
    Serial.println("  r       = recalibrate, return to idle");
    Serial.println("  j       = jerk launch from current lean");
    Serial.println("  k       = cancel jerk, motors idle");
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
            zeroCount = encoderCount; cartZeroCount = cartEncoderCount;
            interrupts();
            resetControlStates();
            X_TARGET = 0.0f;
            setControlMode(MODE_IDLE);
            setAllMotors(0);
            Serial.println(">> Recalibrated. Idle.");
            break;

        case 'j': startJerkSequence(readTheta(), theta_dot_filt); break;

        case 'k':
            setControlMode(MODE_IDLE);
            angle_integral = 0; pos_integral = 0;
            setAllMotors(0);
            Serial.println(">> Jerk cancelled. Idle.");
            break;

        case 't': X_TARGET = 0.0f; pos_integral = 0; Serial.println(">> Target 0.0 m"); break;
        case 'y': X_TARGET = 0.5f; pos_integral = 0; Serial.println(">> Target 0.5 m"); break;
        case 'u': X_TARGET = 1.0f; pos_integral = 0; Serial.println(">> Target 1.0 m"); break;
        case 'i': X_TARGET = 2.0f; pos_integral = 0; Serial.println(">> Target 2.0 m"); break;

        case 'p':
            Serial.println("\n--- PID Gains ---");
            Serial.print("  KP="); Serial.print(KP, 3);
            Serial.print("  KI="); Serial.print(KI, 3);
            Serial.print("  KD="); Serial.println(KD, 3);
            Serial.print("  KP_POS="); Serial.print(KP_POS, 4);
            Serial.print("  KI_POS="); Serial.print(KI_POS, 4);
            Serial.print("  KD_POS="); Serial.println(KD_POS, 4);
            Serial.print("  X_TARGET="); Serial.print(X_TARGET, 2);
            Serial.print(" m  mode="); Serial.println(modeName());
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

    // 3. Derivative + exponential LP
    float theta_dot_raw = (theta_ma - prev_theta) / LOOP_DT_S;
    float x_dot_raw     = (x_ma     - prev_x)     / LOOP_DT_S;

    theta_dot_filt = THETA_DOT_ALPHA * theta_dot_raw + (1.0f - THETA_DOT_ALPHA) * theta_dot_filt;
    x_dot_filt     = X_DOT_ALPHA     * x_dot_raw     + (1.0f - X_DOT_ALPHA)     * x_dot_filt;

    theta_filt = theta_ma;
    x_filt     = x_ma;
    prev_theta = theta_ma;
    prev_x     = x_ma;

    // 4. Safety cutoff in STABILIZE; jerk modes are intentionally large angle
    if (controlMode == MODE_STABILIZE && fabsf(theta_filt) > 0.52f) {
        setAllMotors(0);
        angle_integral = 0; pos_integral = 0;
        setControlMode(MODE_IDLE);
        lastCmd = 0;
        Serial.println(">> Fell. Motors off. Idle.");
    } else {
        lastCmd = updateJerkStart(theta_filt, theta_dot_filt);
        setAllMotors(lastCmd);
    }

    // 5. Telemetry at 10 Hz
    if (millis() - lastPrintMillis >= 100) {
        lastPrintMillis = millis();
        Serial.print("x:");    Serial.print(x_filt, 3);
        Serial.print(" tgt:"); Serial.print(X_TARGET, 2);
        Serial.print(" th:");  Serial.print(theta_filt * RAD_TO_DEG, 2);
        Serial.print(" thd:"); Serial.print(theta_dot_filt, 2);
        Serial.print(" cmd:"); Serial.print(lastCmd);
        Serial.print(" mode:"); Serial.println(modeName());
    }
}
