#include <Motoron.h>
#include <Wire.h>

// ============================================================================
// LQR POLE STABILISATION CONTROLLER
// ============================================================================
// Stabilises an inverted pendulum at 0 degrees (upright position).
// Uses direct motor command output (0-800 range).
// No reliance on physical system parameters - pure gain tuning.
// ============================================================================

// ---------- Motor configuration ----------
MotoronI2C shield1(16);
MotoronI2C shield2(17);
const int16_t MOTOR_MAX = 800;       // Maximum motor command (very fast)
const int16_t MOTOR_DEADBAND = 50;   // Minimum command to overcome friction

// ---------- Pendulum angle encoder (optical sensor) ----------
const float ENCODER_CPR = 1000.0f;   // Counts per revolution
const int pinA = 2;
const int pinB = 3;
volatile long encoderCount = 0;

// ---------- Cart position encoder ----------
const float CART_ENCODER_CPR = 465.6f;   // Counts per revolution (48 CPR * 9.7:1 gear ratio)
const float CART_WHEEL_RADIUS = 0.04f;   // Wheel radius in meters
const int cartPinA = 24;
const int cartPinB = 26;
volatile long cartEncoderCount = 0;

// ============================================================================
// CONTROLLER GAINS - TUNE THESE!
// ============================================================================
// Control law: motorCmd = K_theta * theta_error + K_theta_dot * theta_dot
//                       + K_x * x_error + K_x_dot * x_dot
// 
// Start by tuning K_theta and K_theta_dot to stabilise the pole.
// Once stable, enable K_x to maintain position (currently set to 0).
// ============================================================================

// Angle control gains (PRIMARY - tune these first!)
float K_theta     = 150.0f;   // Proportional gain on angle error (motor_cmd per radian)
float K_theta_dot = 15.0f;    // Derivative gain on angular velocity (damping)

// Position control gains (SECONDARY - tune after pole is stable)
float K_x         = 0.0f;     // Proportional gain on position error (SET TO 0 FOR NOW)
float K_x_dot     = 0.0f;     // Derivative gain on cart velocity

// Integral gain for angle (helps with steady-state offset)
float Ki_theta    = 5.0f;     // Integral gain on angle error
const float INTEGRAL_LIMIT = 50.0f;  // Anti-windup limit

// ---------- Setpoints ----------
const float THETA_SETPOINT = 0.0f;   // Target angle: 0 = upright (radians)
const float X_SETPOINT = 2.0f;       // Target position: 2 meters

// ---------- Control loop timing ----------
const float LOOP_DT_S = 0.002f;      // 2ms = 500 Hz control loop

// ---------- State estimation (simple low-pass filter for derivatives) ----------
const float VELOCITY_FILTER_ALPHA = 0.2f;  // Higher = more responsive, noisier

// ---------- State variables ----------
long zeroCount = 0;
long cartZeroCount = 0;
unsigned long lastControlMicros = 0;
unsigned long lastPrintMillis = 0;

// Filtered state estimates
float theta_filtered = 0.0f;
float theta_dot_filtered = 0.0f;
float x_filtered = 0.0f;
float x_dot_filtered = 0.0f;
float prev_theta = 0.0f;
float prev_x = 0.0f;

// Integral accumulator
float integral_theta = 0.0f;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Wrap angle to [-180, 180] degrees
static inline float wrapAngleDeg(float angleDeg) {
  while (angleDeg > 180.0f) angleDeg -= 360.0f;
  while (angleDeg < -180.0f) angleDeg += 360.0f;
  return angleDeg;
}

// Clamp motor command to valid range with deadband compensation
static inline int16_t clampMotorCmd(int16_t cmd) {
  // Apply deadband compensation (overcome static friction)
  if (cmd > 0 && cmd < MOTOR_DEADBAND) {
    cmd = MOTOR_DEADBAND;
  } else if (cmd < 0 && cmd > -MOTOR_DEADBAND) {
    cmd = -MOTOR_DEADBAND;
  }
  
  // Clamp to motor limits
  if (cmd > MOTOR_MAX) return MOTOR_MAX;
  if (cmd < -MOTOR_MAX) return -MOTOR_MAX;
  return cmd;
}

float readPendulumAngleRad() {
  long countSnapshot;
  noInterrupts();
  countSnapshot = encoderCount;
  interrupts();

  long relativeCount = countSnapshot - zeroCount;
  float angleDeg = (relativeCount * 360.0f) / ENCODER_CPR;
  angleDeg = wrapAngleDeg(angleDeg);
  return angleDeg * DEG_TO_RAD;  // Return in radians
}

float readCartPosition() {
  long countSnapshot;
  noInterrupts();
  countSnapshot = cartEncoderCount;
  interrupts();

  long relativeCount = countSnapshot - cartZeroCount;
  float revolutions = (float)relativeCount / CART_ENCODER_CPR;
  float position = revolutions * 2.0f * PI * CART_WHEEL_RADIUS;
  return position;
}

void setAllMotors(int16_t speedCmd) {
  shield1.setSpeed(1, speedCmd);
  shield1.setSpeed(2, speedCmd);
  shield2.setSpeed(1, speedCmd);
  shield2.setSpeed(2, speedCmd);
}

// ============================================================================
// ENCODER INTERRUPT SERVICE ROUTINES
// ============================================================================

void updateEncoder() {
  int stateB = digitalRead(pinB);
  if (stateB == LOW) {
    encoderCount++;
  } else {
    encoderCount--;
  }
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
// LQR CONTROL COMPUTATION
// ============================================================================
// Outputs motor command directly (not force).
// Control law: cmd = K_theta * theta_error + K_theta_dot * theta_dot
//                  + K_x * x_error + K_x_dot * x_dot
//                  + Ki_theta * integral(theta_error)
// ============================================================================

int16_t computeControl(float x, float x_dot, float theta, float theta_dot, float dt) {
  // Compute errors
  float error_theta = theta - THETA_SETPOINT;
  float error_x = x - X_SETPOINT;
  
  // Update integral term with anti-windup
  integral_theta += error_theta * dt;
  if (integral_theta > INTEGRAL_LIMIT) integral_theta = INTEGRAL_LIMIT;
  if (integral_theta < -INTEGRAL_LIMIT) integral_theta = -INTEGRAL_LIMIT;
  
  // Compute control output (direct motor command)
  // Positive command should move cart in direction to correct positive angle error
  float cmd_f = K_theta * error_theta 
              + K_theta_dot * theta_dot
              + K_x * error_x 
              + K_x_dot * x_dot
              + Ki_theta * integral_theta;
  
  // Convert to integer and apply limits/deadband
  int16_t cmd = (int16_t)cmd_f;
  
  // Only apply deadband compensation if command is meaningful
  if (abs(cmd) > 5) {
    cmd = clampMotorCmd(cmd);
  } else {
    cmd = 0;  // Too small to matter, avoid chattering
  }
  
  // Anti-windup: reduce integral if saturated
  if (abs(cmd) >= MOTOR_MAX) {
    integral_theta -= error_theta * dt * 0.5f;
  }
  
  return cmd;
}

// ============================================================================
// SETUP FUNCTIONS
// ============================================================================

void setupMotors() {
  Wire1.begin();
  shield1.setBus(&Wire1);
  shield2.setBus(&Wire1);

  shield1.reinitialize();
  shield1.clearResetFlag();
  shield1.setMaxAcceleration(1, 0);  // No accel limit - immediate response
  shield1.setMaxDeceleration(1, 0);
  shield1.setMaxAcceleration(2, 0);
  shield1.setMaxDeceleration(2, 0);

  shield2.reinitialize();
  shield2.clearResetFlag();
  shield2.setMaxAcceleration(1, 0);
  shield2.setMaxDeceleration(1, 0);
  shield2.setMaxAcceleration(2, 0);
  shield2.setMaxDeceleration(2, 0);
}

void setupEncoders() {
  // Pendulum angle encoder
  pinMode(pinA, INPUT_PULLUP);
  pinMode(pinB, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(pinA), updateEncoder, RISING);
  
  // Cart position encoder (quadrature)
  pinMode(cartPinA, INPUT_PULLUP);
  pinMode(cartPinB, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(cartPinA), updateCartEncoderA, CHANGE);
  attachInterrupt(digitalPinToInterrupt(cartPinB), updateCartEncoderB, CHANGE);
  
  delay(100);
  
  // Zero at current position (pole should be upright when starting)
  noInterrupts();
  zeroCount = encoderCount;
  cartZeroCount = cartEncoderCount;
  interrupts();
}

void setup() {
  Serial.begin(115200);
  setupMotors();
  setupEncoders();
  
  lastControlMicros = micros();
  
  Serial.println("====================================");
  Serial.println("LQR Pole Stabilisation Controller");
  Serial.println("====================================");
  Serial.println("Target: theta=0 deg (upright), x=2m");
  Serial.println("");
  Serial.println("Current gains:");
  Serial.print("  K_theta     = "); Serial.println(K_theta);
  Serial.print("  K_theta_dot = "); Serial.println(K_theta_dot);
  Serial.print("  Ki_theta    = "); Serial.println(Ki_theta);
  Serial.print("  K_x         = "); Serial.println(K_x);
  Serial.print("  K_x_dot     = "); Serial.println(K_x_dot);
  Serial.println("");
  Serial.println("Commands:");
  Serial.println("  'r' = recalibrate (set current as zero)");
  Serial.println("  'i' = reset integral term");
  Serial.println("  '+' = increase K_theta by 10");
  Serial.println("  '-' = decrease K_theta by 10");
  Serial.println("  'd' = increase K_theta_dot by 1");
  Serial.println("  's' = decrease K_theta_dot by 1");
  Serial.println("====================================");
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
        integral_theta = 0.0f;
        theta_dot_filtered = 0.0f;
        x_dot_filtered = 0.0f;
        Serial.println(">> Recalibrated. Integral reset.");
        break;
        
      case 'i':  // Reset integral
        integral_theta = 0.0f;
        Serial.println(">> Integral reset.");
        break;
        
      case '+':  // Increase K_theta
        K_theta += 10.0f;
        Serial.print(">> K_theta = "); Serial.println(K_theta);
        break;
        
      case '-':  // Decrease K_theta
        K_theta -= 10.0f;
        if (K_theta < 0) K_theta = 0;
        Serial.print(">> K_theta = "); Serial.println(K_theta);
        break;
        
      case 'd':  // Increase K_theta_dot
        K_theta_dot += 1.0f;
        Serial.print(">> K_theta_dot = "); Serial.println(K_theta_dot);
        break;
        
      case 's':  // Decrease K_theta_dot
        K_theta_dot -= 1.0f;
        if (K_theta_dot < 0) K_theta_dot = 0;
        Serial.print(">> K_theta_dot = "); Serial.println(K_theta_dot);
        break;
    }
  }
}

// ============================================================================
// MAIN CONTROL LOOP
// ============================================================================

void loop() {
  handleSerialCommands();
  
  // Timing: run at fixed rate
  unsigned long nowMicros = micros();
  if ((nowMicros - lastControlMicros) < (unsigned long)(LOOP_DT_S * 1000000.0f)) {
    return;
  }
  lastControlMicros = nowMicros;

  // Read sensors
  float theta_raw = readPendulumAngleRad();
  float x_raw = readCartPosition();

  // Simple low-pass filter derivative estimation
  // theta_dot ≈ (theta - prev_theta) / dt, then filtered
  float theta_dot_raw = (theta_raw - prev_theta) / LOOP_DT_S;
  float x_dot_raw = (x_raw - prev_x) / LOOP_DT_S;
  
  // Update filtered estimates (exponential moving average)
  theta_filtered = theta_raw;  // Angle doesn't need much filtering
  theta_dot_filtered = VELOCITY_FILTER_ALPHA * theta_dot_raw + (1.0f - VELOCITY_FILTER_ALPHA) * theta_dot_filtered;
  x_filtered = x_raw;
  x_dot_filtered = VELOCITY_FILTER_ALPHA * x_dot_raw + (1.0f - VELOCITY_FILTER_ALPHA) * x_dot_filtered;
  
  // Store for next iteration
  prev_theta = theta_raw;
  prev_x = x_raw;

  // Compute control
  int16_t motorCmd = computeControl(x_filtered, x_dot_filtered, theta_filtered, theta_dot_filtered, LOOP_DT_S);
  
  // Apply to motors
  setAllMotors(motorCmd);

  // Print debug info at 10 Hz
  if (millis() - lastPrintMillis >= 100) {
    lastPrintMillis = millis();
    Serial.print("theta:");
    Serial.print(theta_filtered * RAD_TO_DEG, 1);
    Serial.print("°  theta_dot:");
    Serial.print(theta_dot_filtered, 1);
    Serial.print("  x:");
    Serial.print(x_filtered, 3);
    Serial.print("m  cmd:");
    Serial.println(motorCmd);
  }
}