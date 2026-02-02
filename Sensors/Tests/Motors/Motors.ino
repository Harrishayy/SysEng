/*
 * Simple 4-Motor Control - Pololu Motoron M3S550
 * 
 * Controls 4 DC motors using 2 stacked shields on Arduino Giga.
 * Shield 1 (address 16): Motors A & B
 * Shield 2 (address 17): Motors C & D
 * 
 * Serial Commands: 'f' = forward, 'b' = backward, 's' = stop
 */

#include <Motoron.h>

// Arduino Giga uses Wire1 (change to Wire for Uno/Mega)
#define I2C_BUS Wire1

MotoronI2C shield1(16);
MotoronI2C shield2(17);

const int16_t SPEED = 400;  // Motor speed (0-800)

void setup() {
  Serial.begin(115200);
  I2C_BUS.begin();
  
  // Configure shields
  shield1.setBus(&I2C_BUS);
  shield2.setBus(&I2C_BUS);
  
  // Initialize shield 1
  shield1.reinitialize();
  shield1.clearResetFlag();
  shield1.setMaxAcceleration(1, 140);
  shield1.setMaxDeceleration(1, 300);
  shield1.setMaxAcceleration(2, 140);
  shield1.setMaxDeceleration(2, 300);
  
  // Initialize shield 2
  shield2.reinitialize();
  shield2.clearResetFlag();
  shield2.setMaxAcceleration(1, 140);
  shield2.setMaxDeceleration(1, 300);
  shield2.setMaxAcceleration(2, 140);
  shield2.setMaxDeceleration(2, 300);
  
  Serial.println("Ready! Commands: 'f'=forward, 'b'=backward, 's'=stop");
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    
    if (cmd == 'f' || cmd == 'F') {
      // All motors forward
      shield1.setSpeed(1, SPEED);
      shield1.setSpeed(2, SPEED);
      shield2.setSpeed(1, SPEED);
      shield2.setSpeed(2, SPEED);
      Serial.println("Forward");
    }
    else if (cmd == 'b' || cmd == 'B') {
      // All motors backward
      shield1.setSpeed(1, -SPEED);
      shield1.setSpeed(2, -SPEED);
      shield2.setSpeed(1, -SPEED);
      shield2.setSpeed(2, -SPEED);
      Serial.println("Backward");
    }
    else if (cmd == 's' || cmd == 'S') {
      // Stop all motors
      shield1.setSpeed(1, 0);
      shield1.setSpeed(2, 0);
      shield2.setSpeed(1, 0);
      shield2.setSpeed(2, 0);
      Serial.println("Stop");
    }
  }
  
  delay(10);
}
