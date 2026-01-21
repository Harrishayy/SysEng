"""
Simulator class for running cart-pole simulations.
"""
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

from cart_pole import CartPole


@dataclass
class SimulationResult:
    """Container for simulation results."""
    time: np.ndarray
    states: np.ndarray  # Shape: (4, n_timesteps) - True states
    noisy_states: np.ndarray = None    # Noisy measurements (if noise enabled)
    filtered_states: np.ndarray = None  # Filtered states (if noise enabled)
    
    @property
    def x(self) -> np.ndarray:
        """Cart position."""
        return self.states[0]
    
    @property
    def x_dot(self) -> np.ndarray:
        """Cart velocity."""
        return self.states[1]
    
    @property
    def theta(self) -> np.ndarray:
        """Pendulum angle."""
        return self.states[2]
    
    @property
    def theta_dot(self) -> np.ndarray:
        """Pendulum angular velocity."""
        return self.states[3]


class Simulator:
    """
    Handles numerical integration of the cart-pole dynamics.
    """
    
    def __init__(self, cart_pole: CartPole):
        """
        Initialize the simulator with a cart-pole system.
        
        Args:
            cart_pole: CartPole instance to simulate
        """
        self.cart_pole = cart_pole
    
    def run(
        self,
        initial_state: np.ndarray,
        duration: float,
        dt: float = 0.01,
        force_func: Optional[Callable[[float], float]] = None,
        controller: Optional[object] = None,
        method: str = 'RK45'
    ) -> SimulationResult:
        """
        Run the simulation.
        
        Args:
            initial_state: Initial state [x, x_dot, theta, theta_dot]
            duration: Simulation duration (s)
            dt: Time step for output (s)
            force_func: Optional function f(t) -> force, defaults to zero force
            controller: Optional controller object with compute(theta, theta_dot, t) or compute(state, t) method
            method: Integration method ('RK45', 'RK23', 'DOP853', etc.)
            
        Returns:
            SimulationResult containing time and state arrays
        """
        # Determine force source
        if controller is not None:
            # Use controller
            def get_force(t, state):
                # Check if controller uses full state or just angle
                if hasattr(controller, 'compute'):
                    import inspect
                    sig = inspect.signature(controller.compute)
                    n_params = len(sig.parameters)
                    
                    if n_params == 3:  # theta, theta_dot, t
                        return controller.compute(state[2], state[3], t)
                    elif n_params == 2:  # state, t
                        return controller.compute(state, t)
                    else:
                        raise ValueError(f"Controller compute method has unexpected signature: {sig}")
                return 0.0
        elif force_func is not None:
            get_force = lambda t, state: force_func(t)
        else:
            get_force = lambda t, state: 0.0
        
        # Create wrapper for dynamics
        def dynamics_wrapper(t, state):
            force = get_force(t, state)
            return self.cart_pole.dynamics(t, state, force)
        
        # Time span and evaluation points
        t_span = (0, duration)
        t_eval = np.arange(0, duration + dt, dt)
        
        # Solve the ODE
        solution = solve_ivp(
            fun=dynamics_wrapper,
            t_span=t_span,
            y0=initial_state,
            t_eval=t_eval,
            method=method
        )
        
        return SimulationResult(
            time=solution.t,
            states=solution.y
        )
    
    def run_with_noise(
        self,
        initial_state: np.ndarray,
        duration: float,
        dt: float = 0.02,
        controller: Optional[object] = None,
        state_processor: Optional[object] = None,
        method: str = 'RK45'
    ) -> SimulationResult:
        """
        Run simulation with measurement noise and state filtering.
        
        Uses discrete time stepping to properly simulate:
        1. True dynamics integration
        2. Noisy measurements
        3. Filtered state estimation
        4. Controller using filtered states
        
        Args:
            initial_state: Initial state [x, x_dot, theta, theta_dot]
            duration: Simulation duration (s)
            dt: Time step (s)
            controller: Controller object with compute() method
            state_processor: NoisyStateProcessor for noise and filtering
            method: Integration method for each step
            
        Returns:
            SimulationResult with true, noisy, and filtered states
        """
        import inspect
        
        # Initialize storage
        n_steps = int(duration / dt) + 1
        time = np.zeros(n_steps)
        true_states = np.zeros((4, n_steps))
        noisy_states = np.zeros((4, n_steps))
        filtered_states = np.zeros((4, n_steps))
        
        # Initial conditions
        state = initial_state.copy()
        true_states[:, 0] = state
        time[0] = 0.0
        
        # Reset filter if provided
        if state_processor is not None:
            state_processor.reset()
            noisy, filtered = state_processor.process(state)
            noisy_states[:, 0] = noisy
            filtered_states[:, 0] = filtered
        else:
            noisy_states[:, 0] = state
            filtered_states[:, 0] = state
        
        # Determine controller signature
        controller_uses_state = False
        if controller is not None and hasattr(controller, 'compute'):
            sig = inspect.signature(controller.compute)
            n_params = len(sig.parameters)
            controller_uses_state = (n_params == 2)  # (state, t)
        
        # Simulate step by step
        for k in range(1, n_steps):
            t = (k - 1) * dt
            
            # Get control force based on filtered state (what controller "sees")
            if controller is not None:
                if controller_uses_state:
                    force = controller.compute(filtered_states[:, k-1], t)
                else:
                    # PID controller uses (theta, theta_dot, t)
                    force = controller.compute(
                        filtered_states[2, k-1],  # theta
                        filtered_states[3, k-1],  # theta_dot
                        t
                    )
            else:
                force = 0.0
            
            # Integrate true dynamics for one time step
            def dynamics(t_inner, s):
                return self.cart_pole.dynamics(t_inner, s, force)
            
            sol = solve_ivp(
                fun=dynamics,
                t_span=(t, t + dt),
                y0=state,
                method=method
            )
            
            # Update true state
            state = sol.y[:, -1]
            true_states[:, k] = state
            time[k] = t + dt
            
            # Process measurements (add noise and filter)
            if state_processor is not None:
                noisy, filtered = state_processor.process(state)
                noisy_states[:, k] = noisy
                filtered_states[:, k] = filtered
            else:
                noisy_states[:, k] = state
                filtered_states[:, k] = state
        
        return SimulationResult(
            time=time,
            states=true_states,
            noisy_states=noisy_states,
            filtered_states=filtered_states
        )
