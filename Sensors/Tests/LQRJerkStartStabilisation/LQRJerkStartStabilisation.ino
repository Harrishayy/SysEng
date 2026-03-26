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
// MOTOR_MAX = 730: Motoron limit 1.7 A / motor stall 1.845 A at 12.3 V → 93% of 800.
const int16_t MOTOR_MAX = 730;
const uint16_t STABILIZE_RAMP_LIMIT = 800;
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
const float CART_ENCODER_CPR = 464.64f;
const float CART_WHEEL_RADIUS = 0.04f;
const int cartPinA = 24;
const int cartPinB = 26;
volatile long          cartEncoderCount       = 0;
volatile uint8_t       lastCartEncoderState   = 0;
volatile unsigned long invalidCartTransitions = 0;

// ============================================================================
// PHYSICAL SYSTEM PARAMETERS
// ============================================================================
const float g = 9.81f;
float M = 1.2f;    // Cart mass (kg)
float m = 0.91f;   // Pendulum mass (kg)
float l = 0.5f;    // Pendulum length to CoM (m)

// ============================================================================
// LQR GAINS  (from working LQRPendulum.ino)
// ============================================================================
//  Control law:  u = -(K1*(x-x_ref) + K2*x_dot + K3*theta + K4*theta_dot)
float K1 =  22.3607f;     // N / m
float K2 =  55.3222f;     // N.s / m
float K3 = -220.9713f;    // N / rad
float K4 =  -55.2750f;    // N.s / rad
float K5_integral = 0.0f; // N / (m.s)  -- LQI integral term
float x_integral  = 0.0f; // accumulated integral of (x - x_ref)

// --- JERK-START PARAMETERS ---
const float JERK_START_MIN_ANGLE = 0.03f;      // ~1.7 deg
const float JERK_CAPTURE_ANGLE = 0.05f;        // ~8.0 deg
const float JERK_CAPTURE_RATE = 1.0f;          // rad/s
const int16_t JERK_INITIAL_CMD = 80;
const int16_t JERK_CMD_STEP = 30;
const unsigned long JERK_ACCEL_MS = 200;
const unsigned long JERK_BRAKE_MS = 200;
const unsigned long JERK_CAPTURE_TIMEOUT_MS = 30000;

// FORCE_TO_CMD_SCALE: geometric derivation gives 42.72 cmd/N; empirically tuned to 18.72.
const float FORCE_TO_CMD_SCALE = 18.72f;

// ---------- Setpoints ----------
float X_SETPOINT = 0.0f;       // Target position (meters) - adjustable

// ---------- Control loop timing ----------
const float LOOP_DT_S = 0.002f;  // 2ms = 500 Hz (slightly slower for stability)

// ---------- Filtering ----------
const float THETA_DOT_FILTER_ALPHA = 0.15f;
const float X_DOT_FILTER_ALPHA = 0.08f;

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
int16_t lastCmd = 0;
uint16_t motorFaultClearCtr = 0;
uint32_t motorResetCount    = 0;   // diagnostic: how often Motoron resets mid-run

enum ControlMode {
  MODE_IDLE = 0,
  MODE_JERK_ACCEL,
  MODE_JERK_BRAKE,
  MODE_STABILIZE
};

ControlMode controlMode = MODE_IDLE;
unsigned long controlModeStartMillis = 0;
bool autoJerkTriggered = false;
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
  x_integral = 0.0f;
  lastCmd = 0;
  for (int i = 0; i < 4; i++) state[i] = 0.0f;
  resetMovingAverageFilters();
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
    // Rezero cart position at capture so LQR starts from x=0
    noInterrupts();
    cartZeroCount = cartEncoderCount;
    interrupts();
    x_filtered = 0.0f;
    x_dot_filtered = 0.0f;
    prev_x = 0.0f;
    x_integral = 0.0f;
    X_SETPOINT = 0.0f;
    for (int i = 0; i < MA_FILTER_SIZE; i++) x_ma_buffer[i] = 0.0f;
    Serial.println(">> Capture window reached. Cart zeroed. LQR stabilisation engaged.");
    return computeLQR(0.0f, 0.0f, theta, theta_dot);
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
      return computeLQR(x_filtered, x_dot_filtered, theta, theta_dot);
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

// Full 4-state quadrature decoder for cart — rejects EMI phantom counts
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
// LQR CONTROL LAW  (from working LQRPendulum.ino)
// ============================================================================
//  u [N] = -(K1*(x-ref) + K2*x_dot + K3*theta + K4*theta_dot + K5*integral)
//  cmd   = clamp(u * FORCE_TO_CMD_SCALE, -800, +800)

int16_t computeLQR(float x, float x_dot, float theta, float theta_dot) {
  // Position error integral (anti-windup clamped to +/-1 m.s)
  x_integral += (x - X_SETPOINT) * LOOP_DT_S;
  x_integral  = clampFloat(x_integral, -1.0f, 1.0f);

  float u = -(  K1 * (x - X_SETPOINT)
              + K2 * x_dot
              + K3 * theta
              + K4 * theta_dot
              + K5_integral * x_integral );

  return clampMotorCmd((int16_t)(u * FORCE_TO_CMD_SCALE));
}

// ============================================================================
// SETUP
// ============================================================================

void setupMotors() {
  Wire1.begin();
  Wire1.setClock(400000);   // 400 kHz fast-mode: prevents I2C timeout at 12.3 V
  shield1.setBus(&Wire1);
  shield2.setBus(&Wire1);

  shield1.reinitialize();
  shield1.clearResetFlag();
  shield1.clearMotorFaultUnconditional();
  shield1.setMaxAcceleration(1, STABILIZE_RAMP_LIMIT);
  shield1.setMaxDeceleration(1, STABILIZE_RAMP_LIMIT);
  shield1.setMaxAcceleration(2, STABILIZE_RAMP_LIMIT);
  shield1.setMaxDeceleration(2, STABILIZE_RAMP_LIMIT);

  shield2.reinitialize();
  shield2.clearResetFlag();
  shield2.clearMotorFaultUnconditional();
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
  lastCartEncoderState = cartEncoderState();
  attachInterrupt(digitalPinToInterrupt(cartPinA), updateCartEncoder, CHANGE);
  attachInterrupt(digitalPinToInterrupt(cartPinB), updateCartEncoder, CHANGE);
  
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
  Serial.println("Architecture: Full 4-state LQR");
  Serial.println("Launch: short jerk -> brake -> capture -> LQR stabilise");
  Serial.println("Idle behaviour: motors remain off until you send 'j'");
  Serial.println("");
  Serial.println("LQR GAINS:");
  Serial.print("  K1 (pos)   = "); Serial.println(K1, 4);
  Serial.print("  K2 (vel)   = "); Serial.println(K2, 4);
  Serial.print("  K3 (angle) = "); Serial.println(K3, 4);
  Serial.print("  K4 (rate)  = "); Serial.println(K4, 4);
  Serial.print("  K5 (integ) = "); Serial.println(K5_integral, 4);
  Serial.print("  Force/cmd scale: "); Serial.println(FORCE_TO_CMD_SCALE, 2);
  Serial.println("");
  Serial.println("Commands:");
  Serial.println("  'r' = recalibrate zero position and return to idle");
  Serial.println("  'j' = start jerk launch from current lean angle");
  Serial.println("  'k' = cancel jerk mode and return to idle");
  Serial.println("  't/y/u/i' = target 0.0 / 0.5 / 1.0 / 2.0 m");
  Serial.println("  '1/2' = K1 more/less negative  (position)");
  Serial.println("  '3/4' = K2 more/less negative  (velocity)");
  Serial.println("  '5/6' = K3 more/less negative  (angle)");
  Serial.println("  '7/8' = K4 more/less negative  (rate)");
  Serial.println("  '9/0' = K5 more/less negative  (integral)");
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
        x_integral = 0.0f;
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
        
      // LQR gain tuning
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
        Serial.println("\n--- Current Configuration ---");
        Serial.print("Mode: "); Serial.println(controlModeName());
        Serial.print("K1="); Serial.print(K1, 3);
        Serial.print(" K2="); Serial.print(K2, 3);
        Serial.print(" K3="); Serial.print(K3, 3);
        Serial.print(" K4="); Serial.print(K4, 3);
        Serial.print(" K5="); Serial.println(K5_integral, 3);
        Serial.print("Target: "); Serial.print(X_SETPOINT); Serial.println(" m");
        Serial.print("x_integral: "); Serial.println(x_integral, 4);
        break;
    }
  }
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  handleSerialCommands();

  // Auto-start jerk sequence 10 s after boot if still idle
  if (!autoJerkTriggered && controlMode == MODE_IDLE && millis() >= 10000) {
    autoJerkTriggered = true;
    startJerkSequence(readPendulumAngleRad(), theta_dot_filtered);
  }

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

  // Safety cutoff: if pendulum has fallen too far, disable motors
  int16_t motorCmd;
  if (fabsf(theta_filtered) > 0.52f && controlMode == MODE_STABILIZE) {
    setAllMotors(0);
    x_integral = 0;
    motorCmd = 0;
  } else {
    motorCmd = updateJerkStart(theta_filtered, theta_dot_filtered);
    setAllMotors(motorCmd);
  }
  lastCmd = motorCmd;

  // Clear Motoron reset/fault flags every 5 cycles (10 ms)
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

  // Debug output at 10 Hz
  if (millis() - lastPrintMillis >= 100) {
    lastPrintMillis = millis();
    Serial.print("x:");    Serial.print(state[0], 3);
    Serial.print(" ref:"); Serial.print(X_SETPOINT, 2);
    Serial.print(" th:");  Serial.print(state[2] * RAD_TO_DEG, 2);
    Serial.print(" thd:"); Serial.print(state[3], 2);
    Serial.print(" cmd:"); Serial.print(lastCmd);
    Serial.print(" mode:"); Serial.print(controlModeName());
    Serial.print(" inv:"); Serial.print(invalidPendulumTransitionCount);
    Serial.print(" cinv:"); Serial.print(invalidCartTransitions);
    Serial.print(" rst:"); Serial.println(motorResetCount);
  }
}
