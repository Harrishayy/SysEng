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
#include <Wire.h>

MotoronI2C shield1(16);
MotoronI2C shield2(17);

const int16_t SPEED = 800;  // Motor speed (0-800)

void setup() {
  Serial.begin(9600);
  Wire1.begin();
  
  // Configure shields to use Wire1 (SCL1/SDA1)
  shield1.setBus(&Wire1);
  shield2.setBus(&Wire1);
  
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
  shield1.setSpeed(1, SPEED);
  shield1.setSpeed(2, SPEED);
  shield2.setSpeed(1, SPEED);
  shield2.setSpeed(2, SPEED);
  delay(10);
}