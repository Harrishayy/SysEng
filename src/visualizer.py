"""
Visualization classes for cart-pole simulation.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.animation import FuncAnimation
from typing import Optional

from cart_pole import CartPole
from simulator import SimulationResult


class Visualizer:
    """
    Handles visualization and animation of cart-pole simulation results.
    """
    
    def __init__(self, cart_pole: CartPole):
        """
        Initialize the visualizer.
        
        Args:
            cart_pole: CartPole instance (needed for geometry)
        """
        self.cart_pole = cart_pole
        
        # Visual parameters
        self.cart_width = 0.4
        self.cart_height = 0.2
        self.wheel_radius = 0.05
        self.bob_radius = 0.08
        
    def plot_states(self, result: SimulationResult, figsize: tuple = (10, 8), save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the state variables over time.
        
        Args:
            result: SimulationResult from simulation
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Cart position
        axes[0].plot(result.time, result.x, 'b-', linewidth=1.5)
        axes[0].set_ylabel('Cart Position (m)')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Cart-Pole Simulation Results')
        
        # Cart velocity
        axes[1].plot(result.time, result.x_dot, 'b-', linewidth=1.5)
        axes[1].set_ylabel('Cart Velocity (m/s)')
        axes[1].grid(True, alpha=0.3)
        
        # Pendulum angle
        axes[2].plot(result.time, np.rad2deg(result.theta), 'r-', linewidth=1.5)
        axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[2].set_ylabel('Angle (deg)')
        axes[2].grid(True, alpha=0.3)
        
        # Pendulum angular velocity
        axes[3].plot(result.time, np.rad2deg(result.theta_dot), 'r-', linewidth=1.5)
        axes[3].set_ylabel('Angular Vel (deg/s)')
        axes[3].set_xlabel('Time (s)')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"State plot saved to {save_path}")
        
        return fig
    
    def plot_states_with_noise(
        self,
        result: SimulationResult,
        figsize: tuple = (12, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot true, noisy, and filtered states for comparison.
        
        Args:
            result: SimulationResult with noisy_states and filtered_states
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if result.noisy_states is None or result.filtered_states is None:
            raise ValueError("SimulationResult must contain noisy_states and filtered_states")
        
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Cart position
        axes[0].plot(result.time, result.states[0], 'b-', linewidth=1.5, label='True', alpha=0.8)
        axes[0].plot(result.time, result.noisy_states[0], 'r.', markersize=1, label='Noisy', alpha=0.3)
        axes[0].plot(result.time, result.filtered_states[0], 'g-', linewidth=1.5, label='Filtered', alpha=0.8)
        axes[0].set_ylabel('Cart Position (m)')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('State Estimation: True vs Noisy vs Filtered')
        axes[0].legend(loc='upper right')
        
        # Cart velocity
        axes[1].plot(result.time, result.states[1], 'b-', linewidth=1.5, label='True', alpha=0.8)
        axes[1].plot(result.time, result.filtered_states[1], 'g-', linewidth=1.5, label='Estimated', alpha=0.8)
        axes[1].set_ylabel('Cart Velocity (m/s)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper right')
        
        # Pendulum angle
        axes[2].plot(result.time, np.rad2deg(result.states[2]), 'b-', linewidth=1.5, label='True', alpha=0.8)
        axes[2].plot(result.time, np.rad2deg(result.noisy_states[2]), 'r.', markersize=1, label='Noisy', alpha=0.3)
        axes[2].plot(result.time, np.rad2deg(result.filtered_states[2]), 'g-', linewidth=1.5, label='Filtered', alpha=0.8)
        axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[2].set_ylabel('Angle (deg)')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='upper right')
        
        # Pendulum angular velocity
        axes[3].plot(result.time, np.rad2deg(result.states[3]), 'b-', linewidth=1.5, label='True', alpha=0.8)
        axes[3].plot(result.time, np.rad2deg(result.filtered_states[3]), 'g-', linewidth=1.5, label='Estimated', alpha=0.8)
        axes[3].set_ylabel('Angular Vel (deg/s)')
        axes[3].set_xlabel('Time (s)')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"State plot saved to {save_path}")
        
        return fig
    
    def plot_forces(
        self,
        result: SimulationResult,
        figsize: tuple = (10, 4),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot control forces and disturbances over time.
        
        Args:
            result: SimulationResult with control_forces and disturbances
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Control force
        if result.control_forces is not None:
            axes[0].plot(result.time, result.control_forces, 'g-', linewidth=1.5, label='Control Force')
        axes[0].set_ylabel('Control Force (N)')
        axes[0].set_title('Forces Applied to Cart')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[0].legend(loc='upper right')
        
        # Disturbance
        if result.disturbances is not None:
            axes[1].plot(result.time, result.disturbances, 'r-', linewidth=1.5, label='Disturbance')
            # Fill under disturbance curve to make it more visible
            axes[1].fill_between(result.time, 0, result.disturbances, alpha=0.3, color='red')
        axes[1].set_ylabel('Disturbance (N)')
        axes[1].set_xlabel('Time (s)')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[1].legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Force plot saved to {save_path}")
        
        return fig
    
    def plot_comprehensive(
        self,
        result: SimulationResult,
        figsize: tuple = (14, 12),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive plot showing all simulation data.
        
        Includes: cart position, pendulum angle, control force, disturbance.
        
        Args:
            result: SimulationResult with all data
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        has_noise = result.noisy_states is not None and result.filtered_states is not None
        has_forces = result.control_forces is not None
        has_disturbance = result.disturbances is not None
        
        n_plots = 4 if (has_forces or has_disturbance) else 2
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
        
        # Cart position
        if has_noise:
            axes[0].plot(result.time, result.states[0], 'b-', linewidth=1.5, label='True', alpha=0.8)
            axes[0].plot(result.time, result.noisy_states[0], 'r.', markersize=1, label='Noisy', alpha=0.3)
            axes[0].plot(result.time, result.filtered_states[0], 'g-', linewidth=1.5, label='Filtered', alpha=0.8)
            axes[0].legend(loc='upper right')
        else:
            axes[0].plot(result.time, result.x, 'b-', linewidth=1.5)
        axes[0].set_ylabel('Cart Position (m)')
        axes[0].set_title('Cart-Pole Simulation - Comprehensive View')
        axes[0].grid(True, alpha=0.3)
        
        # Pendulum angle
        if has_noise:
            axes[1].plot(result.time, np.rad2deg(result.states[2]), 'b-', linewidth=1.5, label='True', alpha=0.8)
            axes[1].plot(result.time, np.rad2deg(result.noisy_states[2]), 'r.', markersize=1, label='Noisy', alpha=0.3)
            axes[1].plot(result.time, np.rad2deg(result.filtered_states[2]), 'g-', linewidth=1.5, label='Filtered', alpha=0.8)
            axes[1].legend(loc='upper right')
        else:
            axes[1].plot(result.time, np.rad2deg(result.theta), 'r-', linewidth=1.5)
        axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[1].set_ylabel('Pendulum Angle (deg)')
        axes[1].grid(True, alpha=0.3)
        
        if n_plots > 2:
            # Control force
            if has_forces:
                axes[2].plot(result.time, result.control_forces, 'g-', linewidth=1.5, label='Control')
            axes[2].set_ylabel('Control Force (N)')
            axes[2].grid(True, alpha=0.3)
            axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
            axes[2].legend(loc='upper right')
            
            # Disturbance
            if has_disturbance:
                axes[3].plot(result.time, result.disturbances, 'r-', linewidth=1.5, label='Disturbance')
                axes[3].fill_between(result.time, 0, result.disturbances, alpha=0.3, color='red')
            axes[3].set_ylabel('Disturbance (N)')
            axes[3].grid(True, alpha=0.3)
            axes[3].axhline(0, color='k', linestyle='--', alpha=0.3)
            axes[3].legend(loc='upper right')
        
        axes[-1].set_xlabel('Time (s)')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comprehensive plot saved to {save_path}")
        
        return fig
    
    def animate(
        self,
        result: SimulationResult,
        interval: int = 20,
        save_path: Optional[str] = None
    ) -> FuncAnimation:
        """
        Create an animation of the cart-pole system.
        
        Args:
            result: SimulationResult from simulation
            interval: Time between frames in milliseconds
            save_path: Optional path to save animation (e.g., 'animation.gif')
            
        Returns:
            FuncAnimation object
        """
        # Calculate bounds for the plot
        x_min = np.min(result.x) - self.cart_width - self.cart_pole.L - 0.5
        x_max = np.max(result.x) + self.cart_width + self.cart_pole.L + 0.5
        y_min = -0.5
        y_max = self.cart_pole.L + 0.5
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Height (m)')
        ax.set_title('Cart-Pole Animation')
        
        # Draw ground
        ax.axhline(y=0, color='brown', linewidth=2)
        ax.fill_between([x_min, x_max], [-0.5, -0.5], [0, 0], 
                       color='burlywood', alpha=0.3)
        
        # Initialize drawing elements
        # Cart body
        cart = FancyBboxPatch(
            (0, 0), self.cart_width, self.cart_height,
            boxstyle="round,pad=0.02",
            facecolor='steelblue',
            edgecolor='darkblue',
            linewidth=2
        )
        ax.add_patch(cart)
        
        # Wheels
        wheel_left = Circle((0, 0), self.wheel_radius, 
                           facecolor='dimgray', edgecolor='black')
        wheel_right = Circle((0, 0), self.wheel_radius,
                            facecolor='dimgray', edgecolor='black')
        ax.add_patch(wheel_left)
        ax.add_patch(wheel_right)
        
        # Pendulum rod
        rod, = ax.plot([], [], 'o-', color='darkred', 
                       linewidth=4, markersize=0)
        
        # Pendulum bob
        bob = Circle((0, 0), self.bob_radius,
                    facecolor='crimson', edgecolor='darkred', linewidth=2)
        ax.add_patch(bob)
        
        # Time display
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Angle display
        angle_text = ax.text(0.02, 0.85, '', transform=ax.transAxes,
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Force display
        force_text = ax.text(0.02, 0.75, '', transform=ax.transAxes,
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Force arrows (control = green, disturbance = red)
        has_forces = result.control_forces is not None
        has_disturbance = result.disturbances is not None
        
        # Scale factor for force arrows (adjust for visibility)
        force_scale = 0.02  # meters per Newton
        
        # Control force arrow
        control_arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                                   arrowprops=dict(arrowstyle='->', color='green', lw=2))
        
        # Disturbance arrow
        disturbance_arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                                       arrowprops=dict(arrowstyle='->', color='red', lw=3))
        
        def init():
            """Initialize animation."""
            cart.set_x(0)
            cart.set_y(0)
            wheel_left.center = (0, 0)
            wheel_right.center = (0, 0)
            rod.set_data([], [])
            bob.center = (0, 0)
            time_text.set_text('')
            angle_text.set_text('')
            force_text.set_text('')
            return cart, wheel_left, wheel_right, rod, bob, time_text, angle_text, force_text
        
        def update(frame):
            """Update animation frame."""
            t = result.time[frame]
            state = result.states[:, frame]
            x = state[0]
            theta = state[2]
            
            # Cart position
            cart_x = x - self.cart_width / 2
            cart_y = self.wheel_radius
            cart.set_x(cart_x)
            cart.set_y(cart_y)
            
            # Wheel positions
            wheel_left.center = (x - self.cart_width / 4, self.wheel_radius)
            wheel_right.center = (x + self.cart_width / 4, self.wheel_radius)
            
            # Pendulum pivot point (top center of cart)
            pivot_x = x
            pivot_y = cart_y + self.cart_height
            
            # Pendulum bob position
            pend_x, pend_y = self.cart_pole.get_pendulum_position(state)
            pend_y += pivot_y  # Adjust for cart height
            
            # Update rod
            rod.set_data([pivot_x, pend_x], [pivot_y, pend_y])
            
            # Update bob
            bob.center = (pend_x, pend_y)
            
            # Update text
            time_text.set_text(f'Time: {t:.2f} s')
            angle_text.set_text(f'Angle: {np.rad2deg(theta):.1f}°')
            
            # Update force arrows and text
            force_info = []
            arrow_y = cart_y + self.cart_height / 2  # Middle of cart
            
            if has_forces and frame < len(result.control_forces):
                ctrl_force = result.control_forces[frame]
                force_info.append(f'Control: {ctrl_force:.1f} N')
                # Update control arrow
                arrow_length = ctrl_force * force_scale
                control_arrow.xy = (x + arrow_length, arrow_y)
                control_arrow.xytext = (x, arrow_y)
            
            if has_disturbance and frame < len(result.disturbances):
                dist_force = result.disturbances[frame]
                force_info.append(f'Disturbance: {dist_force:.1f} N')
                # Update disturbance arrow (show at top of cart, different y level)
                arrow_length = dist_force * force_scale
                dist_arrow_y = cart_y + self.cart_height
                disturbance_arrow.xy = (x + arrow_length, dist_arrow_y)
                disturbance_arrow.xytext = (x, dist_arrow_y)
            
            force_text.set_text('\n'.join(force_info) if force_info else '')
            
            return cart, wheel_left, wheel_right, rod, bob, time_text, angle_text, force_text
        
        # Create animation
        anim = FuncAnimation(
            fig, update, frames=len(result.time),
            init_func=init, blit=True, interval=interval
        )
        
        # Save if path provided
        if save_path:
            anim.save(save_path, writer='pillow', fps=1000//interval)
            print(f"Animation saved to {save_path}")
        
        return anim
    
    def show(self):
        """Display all figures."""
        plt.show()
