"""Simulation engine for cart-pole system."""
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Callable, Optional

from cart_pole import CartPole


@dataclass
class SimulationResult:
    """Container for simulation results."""
    time: np.ndarray
    states: np.ndarray              # (4, n_steps) true states
    noisy_states: np.ndarray = None
    filtered_states: np.ndarray = None
    control_forces: np.ndarray = None
    disturbances: np.ndarray = None
    
    @property
    def x(self): return self.states[0]
    
    @property
    def x_dot(self): return self.states[1]
    
    @property
    def theta(self): return self.states[2]
    
    @property
    def theta_dot(self): return self.states[3]


class Simulator:
    """Numerical integration of cart-pole dynamics."""
    
    def __init__(self, cart_pole: CartPole):
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
        """Run simulation without noise."""
        # Determine force source
        if controller is not None:
            def get_force(t, state):
                if hasattr(controller, 'compute'):
                    import inspect
                    sig = inspect.signature(controller.compute)
                    n_params = len(sig.parameters)
                    if n_params == 3:
                        return controller.compute(state[2], state[3], t)
                    elif n_params == 2:
                        return controller.compute(state, t)
                return 0.0
        elif force_func is not None:
            get_force = lambda t, state: force_func(t)
        else:
            get_force = lambda t, state: 0.0
        
        def dynamics_wrapper(t, state):
            return self.cart_pole.dynamics(t, state, get_force(t, state))
        
        t_eval = np.arange(0, duration + dt, dt)
        solution = solve_ivp(
            fun=dynamics_wrapper,
            t_span=(0, duration),
            y0=initial_state,
            t_eval=t_eval,
            method=method
        )
        
        return SimulationResult(time=solution.t, states=solution.y)
    
    def run_with_noise(
        self,
        initial_state: np.ndarray,
        duration: float,
        dt: float = 0.02,
        controller: Optional[object] = None,
        state_processor: Optional[object] = None,
        disturbance_func: Optional[Callable[[float], float]] = None,
        method: str = 'RK45'
    ) -> SimulationResult:
        """Run simulation with noise, filtering, and disturbances."""
        import inspect
        
        n_steps = int(duration / dt) + 1
        time = np.zeros(n_steps)
        true_states = np.zeros((4, n_steps))
        noisy_states = np.zeros((4, n_steps))
        filtered_states = np.zeros((4, n_steps))
        control_forces = np.zeros(n_steps)
        disturbances = np.zeros(n_steps)
        
        state = initial_state.copy()
        true_states[:, 0] = state
        
        # Initialize filter
        if state_processor is not None:
            state_processor.reset()
            noisy, filtered = state_processor.process(state)
            noisy_states[:, 0] = noisy
            filtered_states[:, 0] = filtered
        else:
            noisy_states[:, 0] = state
            filtered_states[:, 0] = state
        
        # Check controller signature
        controller_uses_state = False
        if controller is not None and hasattr(controller, 'compute'):
            sig = inspect.signature(controller.compute)
            controller_uses_state = (len(sig.parameters) == 2)
        
        # Step-by-step simulation
        for k in range(1, n_steps):
            t = (k - 1) * dt
            
            # Compute control force
            if controller is not None:
                if controller_uses_state:
                    control_force = controller.compute(filtered_states[:, k-1], t)
                else:
                    control_force = controller.compute(
                        filtered_states[2, k-1],
                        filtered_states[3, k-1],
                        t
                    )
            else:
                control_force = 0.0
            
            # Get disturbance
            disturbance = disturbance_func(t) if disturbance_func else 0.0
            
            control_forces[k-1] = control_force
            disturbances[k-1] = disturbance
            total_force = control_force + disturbance
            
            # Integrate one step
            def dynamics(t_inner, s):
                return self.cart_pole.dynamics(t_inner, s, total_force)
            
            sol = solve_ivp(dynamics, (t, t + dt), state, method=method)
            state = sol.y[:, -1]
            true_states[:, k] = state
            time[k] = t + dt
            
            # Process measurements
            if state_processor is not None:
                noisy, filtered = state_processor.process(state)
                noisy_states[:, k] = noisy
                filtered_states[:, k] = filtered
            else:
                noisy_states[:, k] = state
                filtered_states[:, k] = state
        
        # Fill last force values
        control_forces[-1] = control_forces[-2] if n_steps > 1 else 0.0
        disturbances[-1] = disturbances[-2] if n_steps > 1 else 0.0
        
        return SimulationResult(
            time=time,
            states=true_states,
            noisy_states=noisy_states,
            filtered_states=filtered_states,
            control_forces=control_forces,
            disturbances=disturbances
        )
