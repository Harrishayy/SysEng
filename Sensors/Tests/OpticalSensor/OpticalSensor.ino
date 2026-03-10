// AS22 Optical Encoder Test Code for Arduino Uno R4

// --- CONFIGURATION ---
// Enter the CPR (Counts Per Revolution) from your specific part number.
// Refer to the ordering table on Page 6 of the datasheet.
// This code uses full quadrature decoding, so the effective counts per
// revolution is ENCODER_CPR * QUADRATURE_MULTIPLIER.
const long ENCODER_CPR = 1000;
const long QUADRATURE_MULTIPLIER = 4;
const long COUNTS_PER_REV = ENCODER_CPR * QUADRATURE_MULTIPLIER;

// --- PINS ---
const int pinA = 2; // Channel A connected to Pin 2
const int pinB = 3; // Channel B connected to Pin 3
// const int pinIndex = 4; // Optional Index pin

// --- VARIABLES ---
volatile long encoderCount = 0; // "volatile" because it changes inside an interrupt
volatile unsigned long invalidTransitionCount = 0;
volatile uint8_t lastEncoderState = 0;

long lastReportedCount = 0;
unsigned long lastReportedInvalidTransitions = 0;
unsigned long lastReportMs = 0;
bool hasReported = false;

uint8_t readEncoderState() {
  return (digitalRead(pinA) << 1) | digitalRead(pinB);
}

void setup() {
  Serial.begin(115200);
  while (!Serial); // Wait for Serial Monitor to open

  // The AS22 single-ended version provides TTL outputs, so external pullups
  // are not required for a healthy signal.
  pinMode(pinA, INPUT);
  pinMode(pinB, INPUT);

  lastEncoderState = readEncoderState();

  // Decode both channels on every edge so we can detect missed/invalid states.
  attachInterrupt(digitalPinToInterrupt(pinA), updateEncoder, CHANGE);
  attachInterrupt(digitalPinToInterrupt(pinB), updateEncoder, CHANGE);

  Serial.println("AS22 Encoder Test Initialized");
  Serial.print("Assumed CPR: ");
  Serial.println(ENCODER_CPR);
  Serial.print("Quadrature multiplier: x");
  Serial.println(QUADRATURE_MULTIPLIER);
  Serial.print("Effective counts/rev: ");
  Serial.println(COUNTS_PER_REV);
}

void loop() {
  long currentCount;
  unsigned long invalidTransitions;
  noInterrupts();
  currentCount = encoderCount;
  invalidTransitions = invalidTransitionCount;
  interrupts();

  unsigned long now = millis();
  if (hasReported &&
      currentCount == lastReportedCount &&
      invalidTransitions == lastReportedInvalidTransitions &&
      (now - lastReportMs) < 50) {
    return;
  }

  long wrappedCount = currentCount % COUNTS_PER_REV;
  float angle = wrappedCount * 360.0f / COUNTS_PER_REV;
  if (angle < 0) angle += 360.0;

  Serial.print("Count: ");
  Serial.print(currentCount);
  Serial.print("\tAngle: ");
  Serial.print(angle, 2); // Print with 2 decimal places
  Serial.print(" deg\tInvalid transitions: ");
  Serial.println(invalidTransitions);

  hasReported = true;
  lastReportedCount = currentCount;
  lastReportedInvalidTransitions = invalidTransitions;
  lastReportMs = now;
}

// --- INTERRUPT SERVICE ROUTINE (ISR) ---
// This runs every time Channel A or B changes.
void updateEncoder() {
  uint8_t currentState = readEncoderState();
  uint8_t transition = (lastEncoderState << 2) | currentState;

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
      invalidTransitionCount++;
      break;
  }

  lastEncoderState = currentState;
}