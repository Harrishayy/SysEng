// AS22 Optical Encoder Test Code for Arduino Uno R4

// --- CONFIGURATION ---
// Enter the CPR (Counts Per Revolution) from your specific part number.
// Refer to the "Resolution CPR" table on Page 6 of the datasheet.
// Common values: 360, 500, 1000, 2048[cite: 171].
const float ENCODER_CPR = 1000; 

// --- PINS ---
const int pinA = 2; // Channel A connected to Pin 2
const int pinB = 3; // Channel B connected to Pin 3
// const int pinIndex = 4; // Optional Index pin

// --- VARIABLES ---
volatile long encoderCount = 0; // "volatile" because it changes inside an interrupt
long lastReportedCount = 0;

void setup() {
  Serial.begin(9600);
  while (!Serial); // Wait for Serial Monitor to open

  // Configure pins
  pinMode(pinA, INPUT_PULLUP); // Use internal pullup to ensure clean high signal
  pinMode(pinB, INPUT_PULLUP);

  // Attach interrupt to Pin A
  // We trigger on the RISING edge of Channel A
  attachInterrupt(digitalPinToInterrupt(pinA), updateEncoder, RISING);

  Serial.println("AS22 Encoder Test Initialized");
  Serial.print("Assumed CPR: ");
  Serial.println(ENCODER_CPR);
}

void loop() {
  // Create a snapshot of the count safely
  // (Interrupts continue in background, so we grab a copy to print)
  long currentCount;
  noInterrupts();
  currentCount = encoderCount;
  interrupts();

  // Only print if the position has changed
  // if (currentCount != lastReportedCount) {
    
  // Calculate Angle in Degrees
  // Formula: (Count / CPR) * 360
  float angle = (currentCount % (long)ENCODER_CPR) * 360.0 / ENCODER_CPR;
  
  // Handle negative angles for neatness
  if (angle < 0) angle += 360.0;

  Serial.print("Count: ");
  Serial.print(currentCount);
  Serial.print("\t Angle: ");
  Serial.print(angle, 2); // Print with 2 decimal places
  Serial.println(" deg");

  lastReportedCount = currentCount;
  // }
  
  delay(10); // Short delay to prevent serial spamming
}

// --- INTERRUPT SERVICE ROUTINE (ISR) ---
// This runs every time Pin A goes from LOW to HIGH
void updateEncoder() {
  // Read the state of Channel B to determine direction
  // If A leads B (A=High, B=Low), it is usually Clockwise 
  int stateB = digitalRead(pinB);

  if (stateB == LOW) {
    encoderCount++; // Clockwise
  } else {
    encoderCount--; // Counter-Clockwise
  }
}