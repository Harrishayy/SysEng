# Inverted Pendulum (Cart-Pole) Simulation

A simulation of an uncontrolled inverted pendulum on a cart using numerical integration.

## System Description

The cart-pole system consists of:
- A cart that moves horizontally on a frictionless track
- A rigid pendulum attached to the cart via a pivot

The simulation solves the coupled equations of motion for the cart position and pendulum angle.

## Project Structure

```
src/
├── cart_pole.py    # CartPole class - physical parameters and dynamics
├── simulator.py    # Simulator class - numerical integration (RK45)
├── visualizer.py   # Visualizer class - plots and animation
└── main.py         # Entry point
```

## Running the Simulation

```bash
cd src
python main.py
```

This runs an uncontrolled simulation starting with the pendulum at a small angle offset, showing the natural dynamics as it falls and oscillates.

## Dependencies

- numpy
- scipy
- matplotlib
