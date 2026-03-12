#include <Motoron.h>
#include <Wire.h>

// ============================================================================
// LQR JERK-START + STABILISATION CONTROLLER
// ============================================================================
// Adds a simple launch sequence for cases where the pole starts near upright
// but outside the normal capture region. The cart gives a short kick in the
// lean direction, briefly brakes, then hands off to the stabiliser once the
// pole re-enters the capture window.
// ============================================================================

// ---------- Motor configuration ----------
MotoronI2C shield1(16);
MotoronI2C shield2(17);
const int16_t MOTOR_MAX = 800;
const int16_t MOTOR_DEADBAND = 0;
const uint16_t STABILIZE_RAMP_LIMIT = 400;
const uint16_t JERK_RAMP_LIMIT = 2047;

// ---------- Pendulum angle encoder (optical sensor) ----------
const long ENCODER_CPR = 1000;
const long QUADRATURE_MULTIPLIER = 4;
const long COUNTS_PER_REV = ENCODER_CPR * QUADRATURE_MULTIPLIER;
const int pinA = 2;
const int pinB = 3;
volatile long encoderCount = 0;
volatile unsigned long invalidPendulumTransitionCount = 0;
volatile uint8_t lastPendulumEncoderState = 0;

// ---------- Cart position encoder ----------
const float CART_ENCODER_CPR = 465.6f;
const float CART_WHEEL_RADIUS = 0.04f;
const int cartPinA = 24;
const int cartPinB = 26;
volatile long cartEncoderCount = 0;

// ============================================================================
// PHYSICAL SYSTEM PARAMETERS
// ============================================================================
const float g = 9.81f;
float M = 1.2f;    // Cart mass (kg)
float m = 0.91f;   // Pendulum mass (kg)
float l = 0.5f;    // Pendulum length to CoM (m)

// ============================================================================
// CONTROL GAINS - CASCADED STRUCTURE
// ============================================================================
// Primary loop: Angle stabilization (always active)
// Secondary loop: Position control (only when stable)
//
// The key insight: Position control MUST be much slower/weaker than angle
// control, otherwise it destabilizes the pendulum.
// ============================================================================

// --- ANGLE STABILIZATION GAINS (Primary - always active) ---
// These keep the pole upright. K3 and K4 are most critical.
float K3_theta = -110.0f;      // Gain on angle error (proportional)
float K4_theta_dot = -0.0f;  // Gain on angular velocity (derivative/damping)
                               // THIS MUST BE NON-ZERO! Provides essential damping.

// --- POSITION CONTROL GAINS (Secondary - conditional) ---
// These are intentionally weak to avoid fighting the stabilization loop.
// Position control generates a "desired lean angle" that modifies theta setpoint.
float Kp_pos = 0.0f;          // Position proportional gain (m -> rad lean)
float Kd_pos = 0.0f;          // Position derivative gain (m/s -> rad lean)

// Maximum lean angle that position control can request (radians)
// ~5 degrees - keeps pole in linear region
const float MAX_LEAN_ANGLE = 0.087f;

// --- ACTIVATION THRESHOLDS ---
// Position control only activates when pole is nearly vertical
const float STABLE_ANGLE_THRESHOLD = 0.15f;  // ~8.5 degrees
const float STABLE_RATE_THRESHOLD = 1.0f;    // rad/s

// --- JERK-START PARAMETERS ---
const float JERK_START_MIN_ANGLE = 0.03f;      // ~1.7 deg
const float JERK_CAPTURE_ANGLE = 0.09f;        // ~8.0 deg
const float JERK_CAPTURE_RATE = 3.0f;          // rad/s
const int16_t JERK_INITIAL_CMD = 220;
const int16_t JERK_CMD_STEP = 45;
const unsigned long JERK_ACCEL_MS = 250;
const unsigned long JERK_BRAKE_MS = 250;
const unsigned long JERK_CAPTURE_TIMEOUT_MS = 30000;

// Scaling factor: force (N) to motor command
const float FORCE_TO_CMD_SCALE = 100.81f;

// ---------- Setpoints ----------
float THETA_SETPOINT = 0.0f;   // Modified by position controller
const float THETA_SETPOINT_BASE = 0.0f;
float X_SETPOINT = 0.0f;       // Target position (meters) - adjustable

// ---------- Control loop timing ----------
const float LOOP_DT_S = 0.002f;  // 2ms = 500 Hz (slightly slower for stability)

// ---------- Filtering ----------
// Heavier filtering on derivatives to reduce noise
const float THETA_DOT_FILTER_ALPHA = 0.15f;  // More aggressive filtering
const float X_DOT_FILTER_ALPHA = 0.08f;      // Even more for position (slower dynamics)

// Moving average for position measurements
const int MA_FILTER_SIZE = 5;
float theta_ma_buffer[MA_FILTER_SIZE] = {0};
float x_ma_buffer[MA_FILTER_SIZE] = {0};
int ma_buffer_index = 0;
bool ma_buffer_filled = false;

// ---------- State variables ----------
long zeroCount = 0;
long cartZeroCount = 0;
unsigned long lastControlMicros = 0;
unsigned long lastPrintMillis = 0;

// Filtered estimates
float theta_filtered = 0.0f;
float theta_dot_filtered = 0.0f;
float x_filtered = 0.0f;
float x_dot_filtered = 0.0f;
float prev_theta = 0.0f;
float prev_x = 0.0f;

// State tracking
float state[4] = {0.0f, 0.0f, 0.0f, 0.0f};
bool positionControlActive = false;
float positionControlBlend = 0.0f;  // 0 = off, 1 = full

enum ControlMode {
  MODE_IDLE = 0,
  MODE_JERK_ACCEL,
  MODE_JERK_BRAKE,
  MODE_STABILIZE
};

ControlMode controlMode = MODE_IDLE;
unsigned long controlModeStartMillis = 0;
unsigned long jerkSequenceStartMillis = 0;
int jerkDirection = 0;
int jerkPulseIndex = 0;
int16_t jerkCurrentCmd = JERK_INITIAL_CMD;

// ============================================================================
// MOVING AVERAGE FILTER
// ============================================================================

float computeMovingAverage(float* buffer, int size, bool filled) {
  float sum = 0.0f;
  int count = filled ? size : ma_buffer_index;
  if (count == 0) return 0.0f;
  for (int i = 0; i < count; i++) {
    sum += buffer[i];
  }
  return sum / (float)count;
}

void updateMovingAverageBuffers(float theta_raw, float x_raw) {
  theta_ma_buffer[ma_buffer_index] = theta_raw;
  x_ma_buffer[ma_buffer_index] = x_raw;
  ma_buffer_index++;
  if (ma_buffer_index >= MA_FILTER_SIZE) {
    ma_buffer_index = 0;
    ma_buffer_filled = true;
  }
}

float getFilteredTheta() {
  return computeMovingAverage(theta_ma_buffer, MA_FILTER_SIZE, ma_buffer_filled);
}

float getFilteredX() {
  return computeMovingAverage(x_ma_buffer, MA_FILTER_SIZE, ma_buffer_filled);
}

void resetMovingAverageFilters() {
  for (int i = 0; i < MA_FILTER_SIZE; i++) {
    theta_ma_buffer[i] = 0.0f;
    x_ma_buffer[i] = 0.0f;
  }
  ma_buffer_index = 0;
  ma_buffer_filled = false;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

static inline float wrapAngleDeg(float angleDeg) {
  while (angleDeg > 180.0f) angleDeg -= 360.0f;
  while (angleDeg < -180.0f) angleDeg += 360.0f;
  return angleDeg;
}

static inline int16_t clampMotorCmd(int16_t cmd) {
  if (cmd > MOTOR_MAX) return MOTOR_MAX;
  if (cmd < -MOTOR_MAX) return -MOTOR_MAX;
  return cmd;
}

static inline float clampFloat(float val, float minVal, float maxVal) {
  if (val > maxVal) return maxVal;
  if (val < minVal) return minVal;
  return val;
}

uint8_t readPendulumEncoderState() {
  return (digitalRead(pinA) << 1) | digitalRead(pinB);
}

float readPendulumAngleRad() {
  long countSnapshot;
  noInterrupts();
  countSnapshot = encoderCount;
  interrupts();
  long relativeCount = countSnapshot - zeroCount;
  float angleDeg = (relativeCount * 360.0f) / (float)COUNTS_PER_REV;
  angleDeg = wrapAngleDeg(angleDeg);
  return angleDeg * DEG_TO_RAD;
}

float readCartPosition() {
  long countSnapshot;
  noInterrupts();
  countSnapshot = cartEncoderCount;
  interrupts();
  long relativeCount = countSnapshot - cartZeroCount;
  float revolutions = (float)relativeCount / CART_ENCODER_CPR;
  return revolutions * 2.0f * PI * CART_WHEEL_RADIUS;
}

void setAllMotors(int16_t speedCmd) {
  shield1.setSpeed(1, speedCmd);
  shield1.setSpeed(2, speedCmd);
  shield2.setSpeed(1, speedCmd);
  shield2.setSpeed(2, speedCmd);
}

void setMotorRampLimits(uint16_t limit) {
  shield1.setMaxAcceleration(1, limit);
  shield1.setMaxDeceleration(1, limit);
  shield1.setMaxAcceleration(2, limit);
  shield1.setMaxDeceleration(2, limit);

  shield2.setMaxAcceleration(1, limit);
  shield2.setMaxDeceleration(1, limit);
  shield2.setMaxAcceleration(2, limit);
  shield2.setMaxDeceleration(2, limit);
}

void resetControlStates() {
  theta_dot_filtered = 0.0f;
  x_dot_filtered = 0.0f;
  theta_filtered = 0.0f;
  x_filtered = 0.0f;
  prev_theta = 0.0f;
  prev_x = 0.0f;
  for (int i = 0; i < 4; i++) state[i] = 0.0f;
  resetMovingAverageFilters();
  positionControlBlend = 0.0f;
  positionControlActive = false;
  THETA_SETPOINT = THETA_SETPOINT_BASE;
}

void setControlMode(ControlMode newMode) {
  controlMode = newMode;
  controlModeStartMillis = millis();

  if (newMode == MODE_JERK_ACCEL || newMode == MODE_JERK_BRAKE) {
    setMotorRampLimits(JERK_RAMP_LIMIT);
  } else {
    setMotorRampLimits(STABILIZE_RAMP_LIMIT);
  }
}

int16_t getCurrentJerkCmd() {
  return clampMotorCmd(jerkCurrentCmd);
}

void advanceJerkCommand() {
  int nextCmd = jerkCurrentCmd + JERK_CMD_STEP;
  if (nextCmd > MOTOR_MAX) {
    nextCmd = MOTOR_MAX;
  }
  jerkCurrentCmd = (int16_t)nextCmd;
}

bool startJerkSequence(float theta, float theta_dot) {
  float absTheta = fabs(theta);
  if (absTheta < JERK_START_MIN_ANGLE) {
    Serial.print(">> Jerk start refused. Move pole at least ");
    Serial.print(JERK_START_MIN_ANGLE * RAD_TO_DEG, 1);
    Serial.println(" deg away from zero.");
    return false;
  }

  jerkDirection = (theta >= 0.0f) ? 1 : -1;
  jerkPulseIndex = 0;
  jerkCurrentCmd = JERK_INITIAL_CMD;
  resetControlStates();
  prev_theta = theta;
  state[2] = theta;
  state[3] = theta_dot;
  jerkSequenceStartMillis = millis();
  setControlMode(MODE_JERK_ACCEL);
  setAllMotors((int16_t)(jerkDirection * getCurrentJerkCmd()));

  Serial.print(">> Jerk start armed at angle ");
  Serial.print(theta * RAD_TO_DEG, 1);
  Serial.print(" deg, theta_dot ");
  Serial.print(theta_dot, 2);
  Serial.print(" rad/s, direction: ");
  Serial.print(jerkDirection > 0 ? "right" : "left");
  Serial.print(", start cmd: ");
  Serial.print(jerkCurrentCmd);
  Serial.println(", pulsing until capture.");
  return true;
}

int16_t updateJerkStart(float theta, float theta_dot) {
  unsigned long elapsedMs = millis() - controlModeStartMillis;
  unsigned long jerkElapsedMs = millis() - jerkSequenceStartMillis;

  if ((controlMode == MODE_JERK_ACCEL || controlMode == MODE_JERK_BRAKE) &&
      jerkElapsedMs >= JERK_CAPTURE_TIMEOUT_MS) {
    setControlMode(MODE_IDLE);
    Serial.println(">> Jerk timed out before capture. Returning to idle with motors off.");
    return 0;
  }

  if (controlMode != MODE_IDLE &&
      controlMode != MODE_STABILIZE &&
      fabs(theta) <= JERK_CAPTURE_ANGLE &&
      fabs(theta_dot) <= JERK_CAPTURE_RATE) {
    setControlMode(MODE_STABILIZE);
    positionControlBlend = 0.0f;
    Serial.println(">> Capture window reached. Stabilisation engaged.");
    return computeCascadedControl(x_filtered, x_dot_filtered, theta, theta_dot);
  }

  switch (controlMode) {
    case MODE_JERK_ACCEL:
      if (elapsedMs >= JERK_ACCEL_MS) {
        setControlMode(MODE_JERK_BRAKE);
        return (int16_t)(-jerkDirection * getCurrentJerkCmd());
      }
      return (int16_t)(jerkDirection * getCurrentJerkCmd());

    case MODE_JERK_BRAKE:
      if (elapsedMs >= JERK_BRAKE_MS) {
        jerkPulseIndex++;
        advanceJerkCommand();
        setControlMode(MODE_JERK_ACCEL);
        return (int16_t)(jerkDirection * getCurrentJerkCmd());
      }
      return (int16_t)(-jerkDirection * getCurrentJerkCmd());

    case MODE_IDLE:
      return 0;

    case MODE_STABILIZE:
    default:
      return computeCascadedControl(x_filtered, x_dot_filtered, theta, theta_dot);
  }
}

const char* controlModeName() {
  switch (controlMode) {
    case MODE_JERK_ACCEL:
      return "JERK_ACCEL";
    case MODE_JERK_BRAKE:
      return "JERK_BRAKE";
    case MODE_IDLE:
      return "IDLE";
    case MODE_STABILIZE:
    default:
      return "STABILIZE";
  }
}

// ============================================================================
// ENCODER ISRs
// ============================================================================

void updateEncoder() {
  uint8_t currentState = readPendulumEncoderState();
  uint8_t transition = (lastPendulumEncoderState << 2) | currentState;

  switch (transition) {
    case 0b0010:
    case 0b1011:
    case 0b1101:
    case 0b0100:
      encoderCount++;
      break;

    case 0b0001:
    case 0b0111:
    case 0b1110:
    case 0b1000:
      encoderCount--;
      break;

    case 0b0000:
    case 0b0101:
    case 0b1010:
    case 0b1111:
      break;

    default:
      invalidPendulumTransitionCount++;
      break;
  }

  lastPendulumEncoderState = currentState;
}

void updateCartEncoderA() {
  int stateA = digitalRead(cartPinA);
  int stateB = digitalRead(cartPinB);
  if (stateA == stateB) {
    cartEncoderCount++;
  } else {
    cartEncoderCount--;
  }
}

void updateCartEncoderB() {
  int stateA = digitalRead(cartPinA);
  int stateB = digitalRead(cartPinB);
  if (stateA != stateB) {
    cartEncoderCount++;
  } else {
    cartEncoderCount--;
  }
}

// ============================================================================
// CASCADED CONTROL COMPUTATION
// ============================================================================
// Two-level control architecture:
//
// Level 1 (Position -> Lean angle reference):
//   If stable, compute desired lean angle to move toward target position
//   lean_ref = -Kp_pos * (x - x_target) - Kd_pos * x_dot
//   This lean angle becomes the new theta setpoint
//
// Level 2 (Angle stabilization):
//   Standard LQR-style control to maintain theta at setpoint
//   u = -K3 * (theta - theta_setpoint) - K4 * theta_dot
// ============================================================================

int16_t computeCascadedControl(float x_pos, float x_dot, float theta, float theta_dot) {
  // --- Check if pole is stable enough for position control ---
  bool isStable = (fabs(theta) < STABLE_ANGLE_THRESHOLD) && 
                  (fabs(theta_dot) < STABLE_RATE_THRESHOLD);
  
  // Smooth blend in/out of position control (prevents sudden jumps)
  if (isStable) {
    positionControlBlend += 0.002f;  // Ramp up slowly (~0.5s to full)
    if (positionControlBlend > 1.0f) positionControlBlend = 1.0f;
  } else {
    positionControlBlend -= 0.01f;   // Ramp down faster when unstable
    if (positionControlBlend < 0.0f) positionControlBlend = 0.0f;
  }
  positionControlActive = (positionControlBlend > 0.01f);
  
  // --- Level 1: Position control (generates lean angle reference) ---
  float lean_ref = 0.0f;
  if (positionControlActive) {
    float pos_error = x_pos - X_SETPOINT;
    
    // PD control on position -> desired lean angle
    // Negative sign: to move right (positive x), lean left (negative theta)
    lean_ref = -Kp_pos * pos_error - Kd_pos * x_dot;
    
    // Clamp lean angle to keep pendulum in linear operating region
    lean_ref = clampFloat(lean_ref, -MAX_LEAN_ANGLE, MAX_LEAN_ANGLE);
    
    // Apply blend factor
    lean_ref *= positionControlBlend;
  }
  
  // Update effective theta setpoint
  THETA_SETPOINT = THETA_SETPOINT_BASE + lean_ref;
  
  // --- Level 2: Angle stabilization ---
  float theta_error = theta - THETA_SETPOINT;
  
  // Control law: u = -K3*theta_error - K4*theta_dot
  float u = -K3_theta * theta_error - K4_theta_dot * theta_dot;
  
  // Convert to motor command
  int16_t cmd = (int16_t)(u * FORCE_TO_CMD_SCALE);
  return clampMotorCmd(cmd);
}

// ============================================================================
// SETUP
// ============================================================================

void setupMotors() {
  Wire1.begin();
  shield1.setBus(&Wire1);
  shield2.setBus(&Wire1);

  shield1.reinitialize();
  shield1.clearResetFlag();
  shield1.setMaxAcceleration(1, STABILIZE_RAMP_LIMIT);
  shield1.setMaxDeceleration(1, STABILIZE_RAMP_LIMIT);
  shield1.setMaxAcceleration(2, STABILIZE_RAMP_LIMIT);
  shield1.setMaxDeceleration(2, STABILIZE_RAMP_LIMIT);

  shield2.reinitialize();
  shield2.clearResetFlag();
  shield2.setMaxAcceleration(1, STABILIZE_RAMP_LIMIT);
  shield2.setMaxDeceleration(1, STABILIZE_RAMP_LIMIT);
  shield2.setMaxAcceleration(2, STABILIZE_RAMP_LIMIT);
  shield2.setMaxDeceleration(2, STABILIZE_RAMP_LIMIT);
}

void setupEncoders() {
  pinMode(pinA, INPUT);
  pinMode(pinB, INPUT);
  lastPendulumEncoderState = readPendulumEncoderState();
  attachInterrupt(digitalPinToInterrupt(pinA), updateEncoder, CHANGE);
  attachInterrupt(digitalPinToInterrupt(pinB), updateEncoder, CHANGE);
  
  pinMode(cartPinA, INPUT_PULLUP);
  pinMode(cartPinB, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(cartPinA), updateCartEncoderA, CHANGE);
  attachInterrupt(digitalPinToInterrupt(cartPinB), updateCartEncoderB, CHANGE);
  
  delay(100);
  
  noInterrupts();
  zeroCount = encoderCount;
  cartZeroCount = cartEncoderCount;
  interrupts();
}

void setup() {
  Serial.begin(115200);
  setupMotors();
  setupEncoders();
  resetControlStates();
  setControlMode(MODE_IDLE);
  
  lastControlMicros = micros();
  
  Serial.println("=============================================");
  Serial.println("LQR Jerk-Start + Stabilisation Controller");
  Serial.println("=============================================");
  Serial.print("Pendulum encoder CPR: "); Serial.println(ENCODER_CPR);
  Serial.print("Pendulum quadrature multiplier: x"); Serial.println(QUADRATURE_MULTIPLIER);
  Serial.println("Architecture: Position -> Lean angle -> Torque");
  Serial.println("Launch: short jerk -> brake -> capture -> stabilise");
  Serial.println("Idle behaviour: motors remain off until you send 'j'");
  Serial.print("Motor ramp limits stabilize/jerk: "); Serial.print(STABILIZE_RAMP_LIMIT); Serial.print("/"); Serial.println(JERK_RAMP_LIMIT);
  Serial.print("Jerk initial cmd: "); Serial.println(JERK_INITIAL_CMD);
  Serial.print("Jerk linear step: "); Serial.println(JERK_CMD_STEP);
  Serial.print("Jerk timing accel/brake ms: "); Serial.print(JERK_ACCEL_MS); Serial.print("/"); Serial.println(JERK_BRAKE_MS);
  Serial.println("");
  Serial.println("ANGLE STABILIZATION GAINS (always active):");
  Serial.print("  K3 (theta)     = "); Serial.println(K3_theta);
  Serial.print("  K4 (theta_dot) = "); Serial.println(K4_theta_dot);
  Serial.println("");
  Serial.println("POSITION CONTROL GAINS (when stable):");
  Serial.print("  Kp_pos = "); Serial.println(Kp_pos, 3);
  Serial.print("  Kd_pos = "); Serial.println(Kd_pos, 3);
  Serial.print("  Max lean angle = "); Serial.print(MAX_LEAN_ANGLE * RAD_TO_DEG, 1); Serial.println(" deg");
  Serial.println("");
  Serial.println("Commands:");
  Serial.println("  'r' = recalibrate zero position and return to idle");
  Serial.println("  'j' = start jerk launch from current lean angle");
  Serial.println("  'k' = cancel jerk mode and return to idle");
  Serial.println("  't' = set target to 0m (origin)");
  Serial.println("  'y' = set target to 0.5m");
  Serial.println("  'u' = set target to 1.0m");
  Serial.println("  'i' = set target to 2.0m");
  Serial.println("  '+/-' = adjust K3 (theta) by 5");
  Serial.println("  'd/s' = adjust K4 (theta_dot) by 1");
  Serial.println("  'q/a' = adjust Kp_pos by 0.05");
  Serial.println("  'w/e' = adjust Kd_pos by 0.05");
  Serial.println("  'p' = print current state");
  Serial.println("=============================================");
}

// ============================================================================
// SERIAL COMMAND HANDLER
// ============================================================================

void handleSerialCommands() {
  if (Serial.available() > 0) {
    char c = Serial.read();
    switch (c) {
      case 'r':  // Recalibrate
        noInterrupts();
        zeroCount = encoderCount;
        cartZeroCount = cartEncoderCount;
        interrupts();
        resetControlStates();
        setControlMode(MODE_IDLE);
        setAllMotors(0);
        Serial.println(">> Recalibrated. All states reset. Motors idle.");
        break;

      case 'j':
        startJerkSequence(readPendulumAngleRad(), theta_dot_filtered);
        break;

      case 'k':
        setControlMode(MODE_IDLE);
        positionControlBlend = 0.0f;
        setAllMotors(0);
        Serial.println(">> Jerk mode cancelled. Motors idle.");
        break;
        
      // Target position presets
      case 't':
        X_SETPOINT = 0.0f;
        Serial.println(">> Target: 0.0 m");
        break;
      case 'y':
        X_SETPOINT = 0.5f;
        Serial.println(">> Target: 0.5 m");
        break;
      case 'u':
        X_SETPOINT = 1.0f;
        Serial.println(">> Target: 1.0 m");
        break;
      case 'i':
        X_SETPOINT = 2.0f;
        Serial.println(">> Target: 2.0 m");
        break;
        
      // Angle gains
      case '+':
        K3_theta -= 5.0f;  // More negative = stronger
        Serial.print(">> K3 (theta) = "); Serial.println(K3_theta);
        break;
      case '-':
        K3_theta += 5.0f;
        Serial.print(">> K3 (theta) = "); Serial.println(K3_theta);
        break;
      case 'd':
        K4_theta_dot -= 1.0f;  // More negative = more damping
        Serial.print(">> K4 (theta_dot) = "); Serial.println(K4_theta_dot);
        break;
      case 's':
        K4_theta_dot += 1.0f;
        Serial.print(">> K4 (theta_dot) = "); Serial.println(K4_theta_dot);
        break;
        
      // Position gains
      case 'q':
        Kp_pos += 0.05f;
        Serial.print(">> Kp_pos = "); Serial.println(Kp_pos, 3);
        break;
      case 'a':
        Kp_pos -= 0.05f;
        if (Kp_pos < 0) Kp_pos = 0;
        Serial.print(">> Kp_pos = "); Serial.println(Kp_pos, 3);
        break;
      case 'w':
        Kd_pos += 0.05f;
        Serial.print(">> Kd_pos = "); Serial.println(Kd_pos, 3);
        break;
      case 'e':
        Kd_pos -= 0.05f;
        if (Kd_pos < 0) Kd_pos = 0;
        Serial.print(">> Kd_pos = "); Serial.println(Kd_pos, 3);
        break;
        
      case 'p':
        Serial.println("\n--- Current Configuration ---");
        Serial.print("Mode: "); Serial.println(controlModeName());
        Serial.print("Jerk pulse index: "); Serial.println(jerkPulseIndex);
        Serial.print("Angle gains: K3="); Serial.print(K3_theta);
        Serial.print(", K4="); Serial.println(K4_theta_dot);
        Serial.print("Position gains: Kp="); Serial.print(Kp_pos, 3);
        Serial.print(", Kd="); Serial.println(Kd_pos, 3);
        Serial.print("Target position: "); Serial.print(X_SETPOINT); Serial.println(" m");
        Serial.print("Position control blend: "); Serial.println(positionControlBlend, 2);
        break;
    }
  }
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  handleSerialCommands();
  
  unsigned long nowMicros = micros();
  if ((nowMicros - lastControlMicros) < (unsigned long)(LOOP_DT_S * 1000000.0f)) {
    return;
  }
  lastControlMicros = nowMicros;

  // Read sensors
  float theta_raw = readPendulumAngleRad();
  float x_raw = readCartPosition();
  
  // Apply moving average filter
  updateMovingAverageBuffers(theta_raw, x_raw);
  float theta_ma = getFilteredTheta();
  float x_ma = getFilteredX();

  // Compute derivatives with filtering
  float theta_dot_raw = (theta_ma - prev_theta) / LOOP_DT_S;
  float x_dot_raw = (x_ma - prev_x) / LOOP_DT_S;
  
  // Exponential moving average on derivatives
  theta_filtered = theta_ma;
  theta_dot_filtered = THETA_DOT_FILTER_ALPHA * theta_dot_raw + 
                       (1.0f - THETA_DOT_FILTER_ALPHA) * theta_dot_filtered;
  x_filtered = x_ma;
  x_dot_filtered = X_DOT_FILTER_ALPHA * x_dot_raw + 
                   (1.0f - X_DOT_FILTER_ALPHA) * x_dot_filtered;
  
  prev_theta = theta_ma;
  prev_x = x_ma;
  
  // Update state vector
  state[0] = x_filtered;
  state[1] = x_dot_filtered;
  state[2] = theta_filtered;
  state[3] = theta_dot_filtered;

  int16_t motorCmd = updateJerkStart(theta_filtered, theta_dot_filtered);
  
  setAllMotors(motorCmd);

  // Debug output at 10 Hz
  if (millis() - lastPrintMillis >= 100) {
    lastPrintMillis = millis();
    Serial.print("x:");
    Serial.print(state[0], 2);
    Serial.print("m(tgt:");
    Serial.print(X_SETPOINT, 1);
    Serial.print(") th:");
    Serial.print(state[2] * RAD_TO_DEG, 1);
    Serial.print("d(sp:");
    Serial.print(THETA_SETPOINT * RAD_TO_DEG, 1);
    Serial.print(") thd:");
    Serial.print(state[3], 1);
    Serial.print(" u:");
    Serial.print(motorCmd);
    Serial.print(" mode:");
    Serial.print(controlModeName());
    Serial.print(" pulse:");
    Serial.print(jerkPulseIndex);
    Serial.print(" inv:");
    Serial.print(invalidPendulumTransitionCount);
    Serial.print(" pos:");
    Serial.println(positionControlActive ? "ON" : "off");
  }
}
