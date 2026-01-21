#include <Wire.h>
#include <FIMU_ITG3200.h>

ITG3200 gyro;

// Previous values for calculating acceleration
float prev_gyro_x = 0, prev_gyro_y = 0, prev_gyro_z = 0;
unsigned long prev_time = 0;

void setup() {
  Serial.begin(9600);
  Wire.begin();
  
  delay(5);
  gyro.init(ITG3200_ADDR_AD0_LOW); // Initialize gyroscope
  delay(5);
}

void loop() {
  float gyro_x, gyro_y, gyro_z;
  unsigned long current_time = millis();
  
  // Read angular velocities (deg/s)
  gyro.readGyro(&gyro_x, &gyro_y, &gyro_z);
  
  // Calculate time difference in seconds
  float dt = (current_time - prev_time) / 1000.0;
  
  if (dt > 0) {
    // Calculate angular accelerations (deg/s²)
    float ang_accel_x = (gyro_x - prev_gyro_x) / dt;
    float ang_accel_y = (gyro_y - prev_gyro_y) / dt;
    float ang_accel_z = (gyro_z - prev_gyro_z) / dt;
    
    // Print angular velocities
    Serial.print("X: ");
    Serial.print(gyro_x);
    Serial.print(" Y: ");
    Serial.print(gyro_y);
    Serial.print(" Z: ");
    Serial.print(gyro_z);
    
    // Print angular accelerations
    Serial.print(" X_dot: ");
    Serial.print(ang_accel_x);
    Serial.print(" Y_dot: ");
    Serial.print(ang_accel_y);
    Serial.print(" Z_dot: ");
    Serial.println(ang_accel_z);
  }
  
  // Store current values for next iteration
  prev_gyro_x = gyro_x;
  prev_gyro_y = gyro_y;
  prev_gyro_z = gyro_z;
  prev_time = current_time;
  
  delay(100); // 100ms sampling rate
}