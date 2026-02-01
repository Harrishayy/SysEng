"""
Controller tuning using Differential Evolution optimization.

Optimizes PID, LQR, and Pole Placement controllers for the cart-pole system
with the updated motor model (LP 12V motor specs).
"""
import numpy as np
from scipy.optimize import differential_evolution
from scipy.integrate import solve_ivp
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
import warnings

from cart_pole import CartPole
from controller import PIDController, LQRController, PolePlacementController
from motor import MotorModel

warnings.filterwarnings('ignore')


@dataclass
class OptimizationResult:
    """Results from optimization."""
    controller_name: str
    best_params: Dict[str, float]
    best_cost: float
    iterations: int
    convergence_history: List[float]
    early_stopped: bool


class EarlyStoppingCallback:
    """
    Callback for differential evolution with early stopping.
    
    Stops if:
    1. Best cost hasn't improved by min_delta for patience iterations
    2. Best cost reaches target threshold
    """
    
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        target_cost: float = 0.0,
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.target_cost = target_cost
        self.verbose = verbose
        
        self.best_cost = np.inf
        self.no_improvement_count = 0
        self.iteration = 0
        self.history = []
        self.start_time = time.time()
    
    def __call__(self, xk, convergence):
        """Called after each iteration of differential evolution."""
        self.iteration += 1
        current_cost = convergence  # In DE, convergence is the best fitness value
        
        # Track history
        self.history.append(current_cost)
        
        # Check for improvement
        if current_cost < self.best_cost - self.min_delta:
            self.best_cost = current_cost
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        if self.verbose and self.iteration % 5 == 0:
            elapsed = time.time() - self.start_time
            print(f"  Iter {self.iteration:3d} | Best: {self.best_cost:.6f} | "
                  f"No improvement: {self.no_improvement_count}/{self.patience} | "
                  f"Time: {elapsed:.1f}s")
        
        # Early stopping conditions
        if self.no_improvement_count >= self.patience:
            if self.verbose:
                print(f"  Early stopping: No improvement for {self.patience} iterations")
            return True
        
        if self.best_cost <= self.target_cost:
            if self.verbose:
                print(f"  Early stopping: Target cost {self.target_cost} reached")
            return True
        
        return False


class ControllerTuner:
    """
    Tunes controllers using differential evolution optimization.
    """
    
    def __init__(
        self,
        simulation_time: float = 10.0,
        dt: float = 0.02,  # 50 Hz to match interactive simulation
        initial_angle: float = 0.10,  # ~5.7 degrees (matching UI default)
        target_position: float = 2.0,
        verbose: bool = True,
        use_noise: bool = True  # Match interactive simulation
    ):
        self.simulation_time = simulation_time
        self.dt = dt
        self.initial_angle = initial_angle
        self.target_position = target_position
        self.verbose = verbose
        self.use_noise = use_noise
        
        # System parameters
        self.cart_pole = CartPole(
            cart_mass=1.0, pendulum_mass=0.05, rod_length=0.8,
            cart_friction=0.1, rotational_damping=0.01, gravity=9.81
        )
        
        # Motor model with updated specs (LP 12V)
        self.motor = MotorModel(
            num_motors=4, wheel_radius=0.03,
            voltage_min=1.0, voltage_max=12.0,
            rpm_at_nominal=170.0, voltage_nominal=12.0,
            motor_resistance=13.33, torque_constant=0.0403
        )
        
        # State filter matching interactive simulation
        from state_filter import NoisyStateProcessor
        self.state_processor = NoisyStateProcessor(
            position_noise_std=0.005, angle_noise_std=0.01,
            tau_position=0.1, tau_angle=0.08, dt=self.dt, seed=None
        )
    
    def _simulate(
        self,
        controller,
        initial_state: np.ndarray = None,
        use_motor: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run simulation and return time, states, forces, and voltages.
        Matches the interactive simulation exactly (Euler integration, noise, 50Hz).
        """
        if initial_state is None:
            initial_state = np.array([0.0, 0.0, self.initial_angle, 0.0])
        
        t_eval = np.arange(0, self.simulation_time, self.dt)
        
        states = [initial_state.copy()]
        forces = []
        voltages = []
        
        state = initial_state.copy()
        self.state_processor.reset()
        
        for i, t in enumerate(t_eval[:-1]):
            # Apply noise filtering like interactive sim
            if self.use_noise:
                _, filtered_state = self.state_processor.process(state)
            else:
                filtered_state = state.copy()
            
            # Get control force using filtered state
            desired_force = controller.compute(filtered_state, t)
            
            # Apply motor model
            if use_motor:
                motor_output = self.motor.compute_motor_output(desired_force, state[1])
                actual_force = motor_output['actual_force']
                voltage = motor_output['voltage']
            else:
                actual_force = desired_force
                voltage = 0.0
            
            forces.append(actual_force)
            voltages.append(voltage)
            
            # Euler integration (matching interactive sim)
            state_dot = self.cart_pole.dynamics(t, state, actual_force)
            state = state + state_dot * self.dt
            states.append(state.copy())
        
        return t_eval, np.array(states[:-1]), np.array(forces), np.array(voltages)
    
    def _compute_cost(
        self,
        times: np.ndarray,
        states: np.ndarray,
        forces: np.ndarray,
        voltages: np.ndarray
    ) -> float:
        """
        Compute cost function for optimization.
        
        Cost = settling_time_penalty + overshoot_penalty + 
               steady_state_error + control_effort + stability_penalty
        """
        x = states[:, 0]
        x_dot = states[:, 1]
        theta = states[:, 2]
        theta_dot = states[:, 3]
        
        # Check for instability (pendulum fell over)
        if np.any(np.abs(theta) > 0.5):  # ~28 degrees
            return 1000.0  # Large penalty for falling
        
        # 1. Settling time (time to reach and stay within 5% of target)
        position_error = np.abs(x - self.target_position)
        angle_error = np.abs(theta)
        
        settled_pos = position_error < 0.1  # Within 10cm
        settled_angle = angle_error < 0.02  # Within ~1 degree
        settled = settled_pos & settled_angle
        
        if np.any(settled):
            # Find first time we settle and stay settled
            settling_idx = None
            for i in range(len(settled)):
                if np.all(settled[i:min(i+50, len(settled))]):  # Stay settled for 0.5s
                    settling_idx = i
                    break
            
            if settling_idx is not None:
                settling_time = times[settling_idx]
            else:
                settling_time = self.simulation_time
        else:
            settling_time = self.simulation_time
        
        # 2. Steady-state errors (last 20% of simulation)
        final_idx = int(0.8 * len(times))
        ss_position_error = np.mean(np.abs(x[final_idx:] - self.target_position))
        ss_angle_error = np.mean(np.abs(theta[final_idx:]))
        
        # 3. Control effort (penalize high forces and voltages)
        control_effort = np.mean(np.abs(forces)) / 10.0  # Normalize
        
        # 4. Overshoot penalty
        max_position = np.max(x)
        overshoot = max(0, max_position - self.target_position - 0.1) * 5
        
        # 5. Oscillation penalty (penalize velocity variance in steady state)
        oscillation = np.std(x_dot[final_idx:]) + np.std(theta_dot[final_idx:])
        
        # Combined cost
        cost = (
            settling_time * 2.0 +           # Faster settling is better
            ss_position_error * 20.0 +       # Accurate position
            ss_angle_error * 50.0 +          # Accurate angle
            control_effort * 1.0 +           # Minimize control effort
            overshoot * 2.0 +                # Minimize overshoot
            oscillation * 5.0                # Minimize oscillations
        )
        
        return cost
    
    def tune_pid(
        self,
        maxiter: int = 100,
        popsize: int = 15,
        patience: int = 15
    ) -> OptimizationResult:
        """
        Tune PID controller parameters.
        
        Parameters to optimize:
        - kp, ki, kd (angle PID)
        - kp_pos, ki_pos, kd_pos (position PID)
        - max_angle_setpoint
        """
        if self.verbose:
            print("\n" + "="*60)
            print("Tuning PID Controller")
            print("="*60)
        
        # Parameter bounds: [kp, ki, kd, kp_pos, ki_pos, kd_pos, max_angle_setpoint]
        bounds = [
            (10.0, 100.0),   # kp
            (0.0, 5.0),      # ki
            (1.0, 30.0),     # kd
            (0.01, 0.5),     # kp_pos
            (0.0, 0.2),      # ki_pos
            (0.1, 5.0),      # kd_pos
            (0.05, 0.3)      # max_angle_setpoint
        ]
        
        def objective(params):
            kp, ki, kd, kp_pos, ki_pos, kd_pos, max_angle = params
            
            controller = PIDController(
                kp=kp, ki=ki, kd=kd,
                kp_pos=kp_pos, ki_pos=ki_pos, kd_pos=kd_pos,
                x_target=self.target_position,
                max_angle_setpoint=max_angle
            )
            
            try:
                times, states, forces, voltages = self._simulate(controller)
                cost = self._compute_cost(times, states, forces, voltages)
            except Exception:
                cost = 1000.0
            
            return cost
        
        callback = EarlyStoppingCallback(
            patience=patience, min_delta=0.01, verbose=self.verbose
        )
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            callback=callback,
            workers=1,
            updating='deferred',
            polish=True
        )
        
        param_names = ['kp', 'ki', 'kd', 'kp_pos', 'ki_pos', 'kd_pos', 'max_angle_setpoint']
        best_params = dict(zip(param_names, result.x))
        
        if self.verbose:
            print(f"\nPID Optimization Complete!")
            print(f"Best cost: {result.fun:.6f}")
            print("Optimal parameters:")
            for name, val in best_params.items():
                print(f"  {name}: {val:.6f}")
        
        return OptimizationResult(
            controller_name='PID',
            best_params=best_params,
            best_cost=result.fun,
            iterations=callback.iteration,
            convergence_history=callback.history,
            early_stopped=callback.no_improvement_count >= patience
        )
    
    def tune_lqr(
        self,
        maxiter: int = 100,
        popsize: int = 15,
        patience: int = 15
    ) -> OptimizationResult:
        """
        Tune LQR controller Q and R matrices.
        
        Parameters to optimize:
        - Q diagonal elements (4 values)
        - R value (1 value)
        """
        if self.verbose:
            print("\n" + "="*60)
            print("Tuning LQR Controller")
            print("="*60)
        
        # Parameter bounds: [q_x, q_x_dot, q_theta, q_theta_dot, r]
        bounds = [
            (0.1, 50.0),    # q_x
            (0.1, 20.0),    # q_x_dot
            (10.0, 200.0),  # q_theta
            (0.1, 30.0),    # q_theta_dot
            (0.01, 5.0)     # r
        ]
        
        def objective(params):
            q_x, q_x_dot, q_theta, q_theta_dot, r = params
            
            Q = np.diag([q_x, q_x_dot, q_theta, q_theta_dot])
            R = np.array([[r]])
            
            try:
                controller = LQRController(
                    cart_mass=self.cart_pole.M,
                    pendulum_mass=self.cart_pole.m,
                    rod_length=self.cart_pole.L,
                    cart_friction=self.cart_pole.b,
                    rotational_damping=self.cart_pole.c,
                    gravity=self.cart_pole.g,
                    Q=Q, R=R,
                    setpoint=np.array([self.target_position, 0.0, 0.0, 0.0])
                )
                
                times, states, forces, voltages = self._simulate(controller)
                cost = self._compute_cost(times, states, forces, voltages)
            except Exception:
                cost = 1000.0
            
            return cost
        
        callback = EarlyStoppingCallback(
            patience=patience, min_delta=0.01, verbose=self.verbose
        )
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            callback=callback,
            workers=1,
            updating='deferred',
            polish=True
        )
        
        param_names = ['q_x', 'q_x_dot', 'q_theta', 'q_theta_dot', 'r']
        best_params = dict(zip(param_names, result.x))
        
        if self.verbose:
            print(f"\nLQR Optimization Complete!")
            print(f"Best cost: {result.fun:.6f}")
            print("Optimal Q diagonal:", [f"{v:.4f}" for v in result.x[:4]])
            print(f"Optimal R: {result.x[4]:.4f}")
        
        return OptimizationResult(
            controller_name='LQR',
            best_params=best_params,
            best_cost=result.fun,
            iterations=callback.iteration,
            convergence_history=callback.history,
            early_stopped=callback.no_improvement_count >= patience
        )
    
    def tune_pole_placement(
        self,
        maxiter: int = 100,
        popsize: int = 15,
        patience: int = 15
    ) -> OptimizationResult:
        """
        Tune Pole Placement controller pole locations.
        
        Parameters to optimize:
        - 4 pole locations (real parts, all must be negative for stability)
        """
        if self.verbose:
            print("\n" + "="*60)
            print("Tuning Pole Placement Controller")
            print("="*60)
        
        # Parameter bounds: pole real parts (must be negative)
        bounds = [
            (-10.0, -0.5),  # pole 1
            (-10.0, -0.5),  # pole 2
            (-10.0, -0.5),  # pole 3
            (-10.0, -0.5)   # pole 4
        ]
        
        def objective(params):
            # Sort poles to ensure distinct values (helps numerical stability)
            poles = np.sort(params)
            
            # Add small offsets if poles are too close
            for i in range(1, len(poles)):
                if poles[i] - poles[i-1] < 0.1:
                    poles[i] = poles[i-1] + 0.1
            
            try:
                controller = PolePlacementController(
                    cart_mass=self.cart_pole.M,
                    pendulum_mass=self.cart_pole.m,
                    rod_length=self.cart_pole.L,
                    cart_friction=self.cart_pole.b,
                    rotational_damping=self.cart_pole.c,
                    gravity=self.cart_pole.g,
                    poles=poles,
                    setpoint=np.array([self.target_position, 0.0, 0.0, 0.0])
                )
                
                times, states, forces, voltages = self._simulate(controller)
                cost = self._compute_cost(times, states, forces, voltages)
            except Exception:
                cost = 1000.0
            
            return cost
        
        callback = EarlyStoppingCallback(
            patience=patience, min_delta=0.01, verbose=self.verbose
        )
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            callback=callback,
            workers=1,
            updating='deferred',
            polish=True
        )
        
        best_poles = np.sort(result.x)
        param_names = ['pole_1', 'pole_2', 'pole_3', 'pole_4']
        best_params = dict(zip(param_names, best_poles))
        
        if self.verbose:
            print(f"\nPole Placement Optimization Complete!")
            print(f"Best cost: {result.fun:.6f}")
            print("Optimal poles:", [f"{p:.4f}" for p in best_poles])
        
        return OptimizationResult(
            controller_name='PolePlacement',
            best_params=best_params,
            best_cost=result.fun,
            iterations=callback.iteration,
            convergence_history=callback.history,
            early_stopped=callback.no_improvement_count >= patience
        )
    
    def tune_all(
        self,
        maxiter: int = 100,
        popsize: int = 15,
        patience: int = 15
    ) -> Dict[str, OptimizationResult]:
        """Tune all controllers and return results."""
        results = {}
        
        print("\n" + "#"*60)
        print("# Starting Controller Optimization with Differential Evolution")
        print(f"# Max iterations: {maxiter}, Population: {popsize}, Patience: {patience}")
        print("#"*60)
        
        start_time = time.time()
        
        results['PID'] = self.tune_pid(maxiter, popsize, patience)
        results['LQR'] = self.tune_lqr(maxiter, popsize, patience)
        results['PolePlacement'] = self.tune_pole_placement(maxiter, popsize, patience)
        
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Total optimization time: {total_time:.1f}s")
        print()
        
        for name, res in results.items():
            print(f"{name}:")
            print(f"  Best cost: {res.best_cost:.6f}")
            print(f"  Iterations: {res.iterations}")
            print(f"  Early stopped: {res.early_stopped}")
            print(f"  Parameters: {res.best_params}")
            print()
        
        return results


def generate_tuned_code(results: Dict[str, OptimizationResult]) -> str:
    """Generate code snippet with optimized parameters."""
    
    pid = results['PID'].best_params
    lqr = results['LQR'].best_params
    pp = results['PolePlacement'].best_params
    
    code = f'''
# ============================================================
# OPTIMIZED CONTROLLER PARAMETERS
# Generated by differential evolution optimization
# ============================================================

# PID Controller (Cascaded position -> angle control)
pid_params = {{
    'kp': {pid['kp']:.4f},
    'ki': {pid['ki']:.4f},
    'kd': {pid['kd']:.4f},
    'kp_pos': {pid['kp_pos']:.4f},
    'ki_pos': {pid['ki_pos']:.4f},
    'kd_pos': {pid['kd_pos']:.4f},
    'max_angle_setpoint': {pid['max_angle_setpoint']:.4f}
}}

# LQR Controller
lqr_Q = np.diag([{lqr['q_x']:.4f}, {lqr['q_x_dot']:.4f}, {lqr['q_theta']:.4f}, {lqr['q_theta_dot']:.4f}])
lqr_R = np.array([[{lqr['r']:.4f}]])

# Pole Placement Controller
poles = np.array([{pp['pole_1']:.4f}, {pp['pole_2']:.4f}, {pp['pole_3']:.4f}, {pp['pole_4']:.4f}])
'''
    return code


if __name__ == '__main__':
    # Run optimization
    tuner = ControllerTuner(
        simulation_time=10.0,
        dt=0.01,
        initial_angle=0.15,
        target_position=2.0,
        verbose=True
    )
    
    # Tune all controllers
    results = tuner.tune_all(
        maxiter=100,
        popsize=15,
        patience=15
    )
    
    # Generate code with optimized parameters
    print("\n" + "="*60)
    print("OPTIMIZED PARAMETER CODE")
    print("="*60)
    print(generate_tuned_code(results))
    
    # Save results to file
    with open('tuned_parameters.txt', 'w') as f:
        f.write("Controller Tuning Results\n")
        f.write("="*60 + "\n\n")
        
        for name, res in results.items():
            f.write(f"{name} Controller:\n")
            f.write(f"  Best cost: {res.best_cost:.6f}\n")
            f.write(f"  Iterations: {res.iterations}\n")
            f.write(f"  Early stopped: {res.early_stopped}\n")
            f.write("  Parameters:\n")
            for pname, pval in res.best_params.items():
                f.write(f"    {pname}: {pval:.6f}\n")
            f.write("\n")
        
        f.write("\nGenerated Code:\n")
        f.write(generate_tuned_code(results))
    
    print("\nResults saved to 'tuned_parameters.txt'")
