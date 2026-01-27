#include <Motoron.h>

MotoronI2C mc;

void setup() {
  //Wire.begin();

  // 1. Initialize the controller
  mc.reinitialize();
  mc.disableCrc();
  mc.clearResetFlag();

  // 2. Configure Motor 1
  // Set Max Acceleration and Deceleration (prevents jerky starts/stops)
  mc.setMaxAcceleration(1, 200);
  mc.setMaxDeceleration(1, 200);
}

void loop() {
  // --- Move FORWARD ---
  // Speed range is -800 to 800
  mc.setSpeed(1, 400);  
  delay(2000); // Run for 2 seconds

  // --- STOP ---
  mc.setSpeed(1, 0);
  delay(1000); // Wait 1 second

  // --- Move BACKWARD ---
  // Negative speed reverses direction
  mc.setSpeed(1, -400); 
  delay(2000); // Run for 2 seconds

  // --- STOP ---
  mc.setSpeed(1, 0);
  delay(1000); 
}