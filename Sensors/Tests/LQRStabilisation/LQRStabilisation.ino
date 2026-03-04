#include <Motoron.h>
#include <Wire.h>

// ============================================================================
// LQR POLE STABILISATION CONTROLLER
// ============================================================================
// Stabilises an inverted pendulum at 0 degrees (upright position).
// Uses linearized state-space model: x_dot = Ax + Bu
// Control law: u = -K_lqr * x, where x = [x, x_dot, theta, theta_dot]^T
// K_lqr gains are computed offline using solve_continuous_are (Python) or lqr (MATLAB)
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
// PHYSICAL SYSTEM PARAMETERS
// ============================================================================
// These define the linearized state-space model around theta = 0 (upright):
//   x_dot = A*x + B*u
// where x = [x, x_dot, theta, theta_dot]^T and u is force applied to cart.
//
// A = [0,  1,       0,           0      ]
//     [0,  0,      -m*g/M,       0      ]
//     [0,  0,       0,           1      ]
//     [0,  0,  g*(M+m)/(M*l),    0      ]
//
// B = [0, 1/M, 0, -1/(M*l)]^T
// ============================================================================

const float g = 9.81f;           // Gravitational acceleration (m/s^2)
float M = 1.0f;                  // Mass of cart (kg) - PLACEHOLDER
float m = 0.1f;                  // Mass of pendulum (kg) - PLACEHOLDER
float l = 0.3f;                  // Distance from pivot to pendulum CoM (m) - PLACEHOLDER

// ============================================================================
// LQR GAIN VECTOR - COMPUTED OFFLINE
// ============================================================================
// Control law: u = -K_lqr * x = -[K1, K2, K3, K4] * [x, x_dot, theta, theta_dot]^T
//
// Compute K_lqr offline using:
//   Python: scipy.linalg.solve_continuous_are() then K = R^-1 * B^T * P
//   MATLAB: [K, ~, ~] = lqr(A, B, Q, R)
//
// Q and R matrices define the cost function J = integral(x'Qx + u'Ru)dt
// Larger Q penalizes state deviations, larger R penalizes control effort.
// ============================================================================

// LQR gains: K_lqr = [K1, K2, K3, K4] corresponding to [x, x_dot, theta, theta_dot]
float K_lqr[4] = {
    0.0f,     // K1: gain on cart position (x)
    0.0f,     // K2: gain on cart velocity (x_dot)
    150.0f,   // K3: gain on pendulum angle (theta) - PLACEHOLDER
    15.0f     // K4: gain on angular velocity (theta_dot) - PLACEHOLDER
};

// Scaling factor to convert force (N) to motor command (0-800)
// motor_cmd = u * FORCE_TO_CMD_SCALE
const float FORCE_TO_CMD_SCALE = 1.0f;  // PLACEHOLDER - calibrate for your motor

// ---------- Setpoints ----------
const float THETA_SETPOINT = 0.0f;   // Target angle: 0 = upright (radians)
const float X_SETPOINT = 0.0f;       // Target position: 0 meters (origin)

// ---------- Control loop timing ----------
const float LOOP_DT_S = 0.002f;      // 2ms = 500 Hz control loop

// ---------- State estimation (simple low-pass filter for derivatives) ----------
const float VELOCITY_FILTER_ALPHA = 0.2f;  // Higher = more responsive, noisier

// ---------- Moving average filter for encoder measurements ----------
const int MA_FILTER_SIZE = 10;  // Number of samples for moving average
float theta_ma_buffer[MA_FILTER_SIZE] = {0};  // Buffer for pendulum angle
float x_ma_buffer[MA_FILTER_SIZE] = {0};      // Buffer for cart position
int ma_buffer_index = 0;                       // Current index in circular buffer
bool ma_buffer_filled = false;                 // True once buffer has been filled once

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

// State vector: x = [x, x_dot, theta, theta_dot]^T
float state[4] = {0.0f, 0.0f, 0.0f, 0.0f};

// ============================================================================
// MOVING AVERAGE FILTER
// ============================================================================
// Circular buffer implementation for noise reduction on encoder readings.
// Call updateMovingAverageBuffers() each control loop iteration.
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
  return angleDeg * DEG_TO_RAD;  // Return in radians (raw, unfiltered)
}

float readCartPosition() {
  long countSnapshot;
  noInterrupts();
  countSnapshot = cartEncoderCount;
  interrupts();

  long relativeCount = countSnapshot - cartZeroCount;
  float revolutions = (float)relativeCount / CART_ENCODER_CPR;
  float position = revolutions * 2.0f * PI * CART_WHEEL_RADIUS;
  return position;  // Raw, unfiltered
}

void resetMovingAverageFilters() {
  for (int i = 0; i < MA_FILTER_SIZE; i++) {
    theta_ma_buffer[i] = 0.0f;
    x_ma_buffer[i] = 0.0f;
  }
  ma_buffer_index = 0;
  ma_buffer_filled = false;
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
// Implements optimal state feedback control: u = -K_lqr * (x - x_setpoint)
// where x = [x, x_dot, theta, theta_dot]^T is the state vector.
// The force u is then scaled to a motor command.
// ============================================================================

int16_t computeControl(float x_pos, float x_dot, float theta, float theta_dot) {
  // Form the state error vector (deviation from setpoint)
  float state_error[4] = {
    x_pos - X_SETPOINT,     // Position error
    x_dot,                   // Velocity (setpoint is 0)
    theta - THETA_SETPOINT,  // Angle error
    theta_dot                // Angular velocity (setpoint is 0)
  };
  
  // Compute control input: u = -K_lqr * state_error
  // u = -(K1*x_err + K2*x_dot + K3*theta_err + K4*theta_dot)
  float u = 0.0f;
  for (int i = 0; i < 4; i++) {
    u -= K_lqr[i] * state_error[i];
  }
  
  // Scale force to motor command
  float cmd_f = u * FORCE_TO_CMD_SCALE;
  
  // Convert to integer and apply limits/deadband
  int16_t cmd = (int16_t)cmd_f;
  
  // Only apply deadband compensation if command is meaningful
  if (abs(cmd) > 5) {
    cmd = clampMotorCmd(cmd);
  } else {
    cmd = 0;  // Too small to matter, avoid chattering
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
  
  Serial.println("============================================");
  Serial.println("LQR Pole Stabilisation Controller");
  Serial.println("============================================");
  Serial.println("State-space model: x_dot = Ax + Bu");
  Serial.println("Control law: u = -K_lqr * x");
  Serial.println("State vector: x = [x, x_dot, theta, theta_dot]^T");
  Serial.println("");
  Serial.println("Physical parameters (PLACEHOLDERS):");
  Serial.print("  M (cart mass)      = "); Serial.print(M); Serial.println(" kg");
  Serial.print("  m (pendulum mass)  = "); Serial.print(m); Serial.println(" kg");
  Serial.print("  l (pendulum length)= "); Serial.print(l); Serial.println(" m");
  Serial.print("  g (gravity)        = "); Serial.print(g); Serial.println(" m/s^2");
  Serial.println("");
  Serial.println("LQR gains K_lqr = [K1, K2, K3, K4]:");
  Serial.print("  K1 (x)         = "); Serial.println(K_lqr[0]);
  Serial.print("  K2 (x_dot)     = "); Serial.println(K_lqr[1]);
  Serial.print("  K3 (theta)     = "); Serial.println(K_lqr[2]);
  Serial.print("  K4 (theta_dot) = "); Serial.println(K_lqr[3]);
  Serial.println("");
  Serial.println("Setpoints:");
  Serial.print("  x_setpoint     = "); Serial.print(X_SETPOINT); Serial.println(" m");
  Serial.print("  theta_setpoint = "); Serial.print(THETA_SETPOINT); Serial.println(" rad");
  Serial.println("");
  Serial.println("Commands:");
  Serial.println("  'r' = recalibrate (set current as zero)");
  Serial.println("  '1/2' = increase/decrease K1 (x) by 1");
  Serial.println("  '3/4' = increase/decrease K2 (x_dot) by 1");
  Serial.println("  '+/-' = increase/decrease K3 (theta) by 10");
  Serial.println("  'd/s' = increase/decrease K4 (theta_dot) by 1");
  Serial.println("============================================");
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
        theta_dot_filtered = 0.0f;
        x_dot_filtered = 0.0f;
        for (int i = 0; i < 4; i++) state[i] = 0.0f;
        resetMovingAverageFilters();
        prev_theta = 0.0f;
        prev_x = 0.0f;
        Serial.println(">> Recalibrated. State and filters reset.");
        break;
        
      case '1':  // Increase K1 (x)
        K_lqr[0] += 1.0f;
        Serial.print(">> K1 (x) = "); Serial.println(K_lqr[0]);
        break;
        
      case '2':  // Decrease K1 (x)
        K_lqr[0] -= 1.0f;
        Serial.print(">> K1 (x) = "); Serial.println(K_lqr[0]);
        break;
        
      case '3':  // Increase K2 (x_dot)
        K_lqr[1] += 1.0f;
        Serial.print(">> K2 (x_dot) = "); Serial.println(K_lqr[1]);
        break;
        
      case '4':  // Decrease K2 (x_dot)
        K_lqr[1] -= 1.0f;
        Serial.print(">> K2 (x_dot) = "); Serial.println(K_lqr[1]);
        break;
        
      case '+':  // Increase K3 (theta)
        K_lqr[2] += 10.0f;
        Serial.print(">> K3 (theta) = "); Serial.println(K_lqr[2]);
        break;
        
      case '-':  // Decrease K3 (theta)
        K_lqr[2] -= 10.0f;
        if (K_lqr[2] < 0) K_lqr[2] = 0;
        Serial.print(">> K3 (theta) = "); Serial.println(K_lqr[2]);
        break;
        
      case 'd':  // Increase K4 (theta_dot)
        K_lqr[3] += 1.0f;
        Serial.print(">> K4 (theta_dot) = "); Serial.println(K_lqr[3]);
        break;
        
      case 's':  // Decrease K4 (theta_dot)
        K_lqr[3] -= 1.0f;
        if (K_lqr[3] < 0) K_lqr[3] = 0;
        Serial.print(">> K4 (theta_dot) = "); Serial.println(K_lqr[3]);
        break;
        
      case 'p':  // Print current state and gains
        Serial.println("\n--- Current LQR Gains ---");
        Serial.print("K_lqr = ["); 
        Serial.print(K_lqr[0]); Serial.print(", ");
        Serial.print(K_lqr[1]); Serial.print(", ");
        Serial.print(K_lqr[2]); Serial.print(", ");
        Serial.print(K_lqr[3]); Serial.println("]");
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
  
  // Update moving average buffers and get filtered values
  updateMovingAverageBuffers(theta_raw, x_raw);
  float theta_ma = getFilteredTheta();
  float x_ma = getFilteredX();

  // Simple low-pass filter derivative estimation
  // theta_dot ≈ (theta - prev_theta) / dt, then filtered
  float theta_dot_raw = (theta_ma - prev_theta) / LOOP_DT_S;
  float x_dot_raw = (x_ma - prev_x) / LOOP_DT_S;
  
  // Update filtered estimates (exponential moving average for velocities)
  theta_filtered = theta_ma;  // Use MA-filtered angle
  theta_dot_filtered = VELOCITY_FILTER_ALPHA * theta_dot_raw + (1.0f - VELOCITY_FILTER_ALPHA) * theta_dot_filtered;
  x_filtered = x_ma;          // Use MA-filtered position
  x_dot_filtered = VELOCITY_FILTER_ALPHA * x_dot_raw + (1.0f - VELOCITY_FILTER_ALPHA) * x_dot_filtered;
  
  // Store for next iteration (use MA-filtered values for derivative calculation)
  prev_theta = theta_ma;
  prev_x = x_ma;
  
  // Update state vector: x = [x, x_dot, theta, theta_dot]^T
  state[0] = x_filtered;
  state[1] = x_dot_filtered;
  state[2] = theta_filtered;
  state[3] = theta_dot_filtered;

  // Compute LQR control: u = -K_lqr * x
  int16_t motorCmd = computeControl(x_filtered, x_dot_filtered, theta_filtered, theta_dot_filtered);
  
  // Apply to motors
  setAllMotors(motorCmd);

  // Print debug info at 10 Hz
  if (millis() - lastPrintMillis >= 100) {
    lastPrintMillis = millis();
    Serial.print("x:");
    Serial.print(state[0], 3);
    Serial.print("m  x_dot:");
    Serial.print(state[1], 2);
    Serial.print("  theta:");
    Serial.print(state[2] * RAD_TO_DEG, 1);
    Serial.print("deg  theta_dot:");
    Serial.print(state[3], 1);
    Serial.print("  u:");
    Serial.println(motorCmd);
  }
}