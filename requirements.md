1. Equations of Motion:
o Derive the nonlinear dynamics of the inverted pendulum using
Newton's laws or the Lagrangian method.
o Consider air drag and damping at the pivoting point. Simulate the
system with and without these.
o If needed, linearize the system around the equilibrium point
(small-angle approximation) for controller design.
2. Sensor Noise Simulation:
o Simulate sensor noise in the system (e.g., Gaussian noise in angle
and position readings).
o Implement a noise filtering technique to handle noisy
measurements.
3. Controller Design and Implementation:
o Design alternative types of controllers, including PID, pole
placement, and a nonlinear controller for stabilizing the pendulum.
At least two methods are needed for your project.
o Hints: Tune the PID gains using manual or automated methods
like Ziegler-Nichols. For pole placement, design a controller using
the linearized system.
o Test and compare the performance of the controllers under noise
and disturbances.
4. Python Implementation:
o Implement the nonlinear equations of motion and simulate the
system using Python or C++ (You may use scipy for solving
ODEs).
o Incorporate noise in the system and simulate its effect on the
control loop.
5. Visualization of the System:
o Create a dynamic visualization of the system showing the cart and
pendulum motion.
o The simulation should include a real-time animation of the cart
and pendulum.
o Disturbances (e.g., external force) should be visually represented
in the simulation, as well as in the graphs.
o Display real-time plots of the cart’s position, pendulum’s angle,
disturbance, and control force.
o Visualize the system’s behavior with tools/libraries such as
matplotlib, vpython, or pygame.
o Allow users to adjust controller parameters and observe changes
in real time.