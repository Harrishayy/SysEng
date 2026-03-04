#include <Motoron.h>
#include <Wire.h>

// ---------- Motor configuration (from Motors.ino) ----------
MotoronI2C shield1(16);
MotoronI2C shield2(17);
const int16_t MOTOR_LIMIT = 1000;

// ---------- Optical sensor configuration (from OpticalSensor.ino) ----------
const float ENCODER_CPR = 1000.0f;
const int pinA = 2;
const int pinB = 3;
volatile long encoderCount = 0;

// ---------- PID tuning ----------
const float TARGET_ANGLE_DEG = -25.0f;
const float KP = 50.0f;
const float KI = 2.0f;
const float KD = 1.6f;
const float INTEGRAL_CLAMP = 250.0f;
const float LOOP_DT_S = 0.01f;  // 100 Hz

long zeroCount = 0;
float integralError = 0.0f;
float lastError = 0.0f;
unsigned long lastControlMicros = 0;
unsigned long lastPrintMillis = 0;

static inline float wrapDegSigned(float angleDeg) {
  while (angleDeg > 180.0f) angleDeg -= 360.0f;
  while (angleDeg < -180.0f) angleDeg += 360.0f;
  return angleDeg;
}

static inline int16_t clampMotorCommand(float u) {
  if (u > MOTOR_LIMIT) return MOTOR_LIMIT;
  if (u < -MOTOR_LIMIT) return -MOTOR_LIMIT;
  return (int16_t)u;
}

float readPendulumAngleDeg() {
  long countSnapshot;
  noInterrupts();
  countSnapshot = encoderCount;
  interrupts();

  long relativeCount = countSnapshot - zeroCount;
  float angleDeg = (relativeCount * 360.0f) / ENCODER_CPR;
  return wrapDegSigned(angleDeg);
}

void setAllMotors(int16_t speedCmd) {
  shield1.setSpeed(1, speedCmd);
  shield1.setSpeed(2, speedCmd);
  shield2.setSpeed(1, speedCmd);
  shield2.setSpeed(2, speedCmd);
}

void updateEncoder() {
  int stateB = digitalRead(pinB);
  if (stateB == LOW) {
    encoderCount++;
  } else {
    encoderCount--;
  }
}

void setupMotors() {
  Wire1.begin();
  shield1.setBus(&Wire1);
  shield2.setBus(&Wire1);

  shield1.reinitialize();
  shield1.clearResetFlag();
  shield1.setMaxAcceleration(1, 140);
  shield1.setMaxDeceleration(1, 300);
  shield1.setMaxAcceleration(2, 140);
  shield1.setMaxDeceleration(2, 300);

  shield2.reinitialize();
  shield2.clearResetFlag();
  shield2.setMaxAcceleration(1, 140);
  shield2.setMaxDeceleration(1, 300);
  shield2.setMaxAcceleration(2, 140);
  shield2.setMaxDeceleration(2, 300);
}

void setupEncoder() {
  pinMode(pinA, INPUT_PULLUP);
  pinMode(pinB, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(pinA), updateEncoder, RISING);
  delay(500);
  noInterrupts();
  zeroCount = encoderCount;
  interrupts();
}

void setup() {
  Serial.begin(115200);
  setupMotors();
  setupEncoder();
  lastControlMicros = micros();
  Serial.println("PID stabilisation ready (target = 0 deg).");
}

void loop() {
  unsigned long nowMicros = micros();
  if ((nowMicros - lastControlMicros) < (unsigned long)(LOOP_DT_S * 1000000.0f)) {
    return;
  }
  lastControlMicros = nowMicros;

  float angleDeg = readPendulumAngleDeg();
  float errorDeg = wrapDegSigned(TARGET_ANGLE_DEG - angleDeg);
  float derivative = (errorDeg - lastError) / LOOP_DT_S;

  integralError += errorDeg * LOOP_DT_S;
  if (integralError > INTEGRAL_CLAMP) integralError = INTEGRAL_CLAMP;
  if (integralError < -INTEGRAL_CLAMP) integralError = -INTEGRAL_CLAMP;

  float control = (KP * errorDeg) + (KI * integralError) + (KD * derivative);
  int16_t cmd = clampMotorCommand(-control);
  setAllMotors(cmd);

  lastError = errorDeg;

  if (millis() - lastPrintMillis >= 100) {
    lastPrintMillis = millis();
    Serial.print("angle[deg]: ");
    Serial.print(angleDeg, 2);
    Serial.print("  err: ");
    Serial.print(errorDeg, 2);
    Serial.print("  u: ");
    Serial.println(cmd);
  }
}
