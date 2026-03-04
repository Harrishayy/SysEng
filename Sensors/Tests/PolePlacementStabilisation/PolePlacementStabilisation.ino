#include <Motoron.h>
#include <Wire.h>

// ---------- Motor configuration (from Motors.ino) ----------
MotoronI2C shield1(16);
MotoronI2C shield2(17);
const int16_t MOTOR_LIMIT = 800;

// ---------- Optical sensor configuration (from OpticalSensor.ino) ----------
const float ENCODER_CPR = 1000.0f;
const int pinA = 2;
const int pinB = 3;
volatile long encoderCount = 0;

// ---------- Pole placement tuning ----------
// Assumed linearized model:
// xdot = [0 1; 0 0]x + [0; B_GAIN]u, where x = [theta; thetaDot].
// Desired poles are negative real values in rad/s.
const float POLE_1 = -6.5f;
const float POLE_2 = -8.0f;
const float B_GAIN = 14.0f;
const float LOOP_DT_S = 0.01f;  // 100 Hz
const float DERIVATIVE_ALPHA = 0.8f;

long zeroCount = 0;
float kTheta = 0.0f;
float kOmega = 0.0f;
float lastThetaRad = 0.0f;
float thetaDotFilt = 0.0f;
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

  // Characteristic equation: s^2 + (kOmega*B_GAIN)s + (kTheta*B_GAIN) = 0
  // Match to desired: (s - p1)(s - p2) = s^2 - (p1 + p2)s + p1*p2
  kOmega = -(POLE_1 + POLE_2) / B_GAIN;
  kTheta = (POLE_1 * POLE_2) / B_GAIN;

  lastControlMicros = micros();
  Serial.println("Pole placement stabilisation ready (target = 0 deg).");
  Serial.print("kTheta=");
  Serial.print(kTheta, 4);
  Serial.print("  kOmega=");
  Serial.println(kOmega, 4);
}

void loop() {
  unsigned long nowMicros = micros();
  if ((nowMicros - lastControlMicros) < (unsigned long)(LOOP_DT_S * 1000000.0f)) {
    return;
  }
  lastControlMicros = nowMicros;

  float thetaDeg = readPendulumAngleDeg();
  float thetaRad = thetaDeg * DEG_TO_RAD;

  float thetaDotRaw = (thetaRad - lastThetaRad) / LOOP_DT_S;
  thetaDotFilt = DERIVATIVE_ALPHA * thetaDotFilt + (1.0f - DERIVATIVE_ALPHA) * thetaDotRaw;

  float control = -((kTheta * thetaRad) + (kOmega * thetaDotFilt));
  int16_t cmd = clampMotorCommand(control);
  setAllMotors(cmd);

  lastThetaRad = thetaRad;

  if (millis() - lastPrintMillis >= 100) {
    lastPrintMillis = millis();
    Serial.print("theta[deg]: ");
    Serial.print(thetaDeg, 2);
    Serial.print("  thetaDot[rad/s]: ");
    Serial.print(thetaDotFilt, 2);
    Serial.print("  u: ");
    Serial.println(cmd);
  }
}
