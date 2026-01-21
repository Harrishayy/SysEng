"""Visualization for cart-pole simulation."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.animation import FuncAnimation
from typing import Optional

from cart_pole import CartPole
from simulator import SimulationResult


class Visualizer:
    """Plotting and animation for cart-pole results."""
    
    def __init__(self, cart_pole: CartPole):
        self.cart_pole = cart_pole
        self.cart_width = 0.4
        self.cart_height = 0.2
        self.wheel_radius = 0.05
        self.bob_radius = 0.08
        
    def plot_states(self, result: SimulationResult, figsize: tuple = (10, 8), save_path: Optional[str] = None) -> plt.Figure:
        """Plot state variables over time."""
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        axes[0].plot(result.time, result.x, 'b-', linewidth=1.5)
        axes[0].set_ylabel('Cart Position (m)')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Cart-Pole Simulation Results')
        
        axes[1].plot(result.time, result.x_dot, 'b-', linewidth=1.5)
        axes[1].set_ylabel('Cart Velocity (m/s)')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(result.time, np.rad2deg(result.theta), 'r-', linewidth=1.5)
        axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[2].set_ylabel('Angle (deg)')
        axes[2].grid(True, alpha=0.3)
        
        axes[3].plot(result.time, np.rad2deg(result.theta_dot), 'r-', linewidth=1.5)
        axes[3].set_ylabel('Angular Vel (deg/s)')
        axes[3].set_xlabel('Time (s)')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        return fig
    
    def plot_states_with_noise(self, result: SimulationResult, figsize: tuple = (12, 10), save_path: Optional[str] = None) -> plt.Figure:
        """Plot true, noisy, and filtered states."""
        if result.noisy_states is None or result.filtered_states is None:
            raise ValueError("Result must contain noisy and filtered states")
        
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Position
        axes[0].plot(result.time, result.states[0], 'b-', linewidth=1.5, label='True', alpha=0.8)
        axes[0].plot(result.time, result.noisy_states[0], 'r.', markersize=1, label='Noisy', alpha=0.3)
        axes[0].plot(result.time, result.filtered_states[0], 'g-', linewidth=1.5, label='Filtered', alpha=0.8)
        axes[0].set_ylabel('Cart Position (m)')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('State Estimation')
        axes[0].legend(loc='upper right')
        
        # Velocity
        axes[1].plot(result.time, result.states[1], 'b-', linewidth=1.5, label='True', alpha=0.8)
        axes[1].plot(result.time, result.filtered_states[1], 'g-', linewidth=1.5, label='Estimated', alpha=0.8)
        axes[1].set_ylabel('Cart Velocity (m/s)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper right')
        
        # Angle
        axes[2].plot(result.time, np.rad2deg(result.states[2]), 'b-', linewidth=1.5, label='True', alpha=0.8)
        axes[2].plot(result.time, np.rad2deg(result.noisy_states[2]), 'r.', markersize=1, label='Noisy', alpha=0.3)
        axes[2].plot(result.time, np.rad2deg(result.filtered_states[2]), 'g-', linewidth=1.5, label='Filtered', alpha=0.8)
        axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[2].set_ylabel('Angle (deg)')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='upper right')
        
        # Angular velocity
        axes[3].plot(result.time, np.rad2deg(result.states[3]), 'b-', linewidth=1.5, label='True', alpha=0.8)
        axes[3].plot(result.time, np.rad2deg(result.filtered_states[3]), 'g-', linewidth=1.5, label='Estimated', alpha=0.8)
        axes[3].set_ylabel('Angular Vel (deg/s)')
        axes[3].set_xlabel('Time (s)')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(loc='upper right')
        
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        return fig
    
    def plot_forces(self, result: SimulationResult, figsize: tuple = (10, 4), save_path: Optional[str] = None) -> plt.Figure:
        """Plot control force and disturbance."""
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        if result.control_forces is not None:
            axes[0].plot(result.time, result.control_forces, 'g-', linewidth=1.5, label='Control')
        axes[0].set_ylabel('Control Force (N)')
        axes[0].set_title('Forces Applied to Cart')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[0].legend(loc='upper right')
        
        if result.disturbances is not None:
            axes[1].plot(result.time, result.disturbances, 'r-', linewidth=1.5, label='Disturbance')
            axes[1].fill_between(result.time, 0, result.disturbances, alpha=0.3, color='red')
        axes[1].set_ylabel('Disturbance (N)')
        axes[1].set_xlabel('Time (s)')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[1].legend(loc='upper right')
        
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        return fig
    
    def plot_comprehensive(self, result: SimulationResult, figsize: tuple = (14, 12), save_path: Optional[str] = None) -> plt.Figure:
        """Plot position, angle, control force, and disturbance."""
        has_noise = result.noisy_states is not None and result.filtered_states is not None
        has_forces = result.control_forces is not None
        has_disturbance = result.disturbances is not None
        
        n_plots = 4 if (has_forces or has_disturbance) else 2
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
        
        # Position
        if has_noise:
            axes[0].plot(result.time, result.states[0], 'b-', linewidth=1.5, label='True', alpha=0.8)
            axes[0].plot(result.time, result.noisy_states[0], 'r.', markersize=1, label='Noisy', alpha=0.3)
            axes[0].plot(result.time, result.filtered_states[0], 'g-', linewidth=1.5, label='Filtered', alpha=0.8)
            axes[0].legend(loc='upper right')
        else:
            axes[0].plot(result.time, result.x, 'b-', linewidth=1.5)
        axes[0].set_ylabel('Cart Position (m)')
        axes[0].set_title('Cart-Pole Simulation')
        axes[0].grid(True, alpha=0.3)
        
        # Angle
        if has_noise:
            axes[1].plot(result.time, np.rad2deg(result.states[2]), 'b-', linewidth=1.5, label='True', alpha=0.8)
            axes[1].plot(result.time, np.rad2deg(result.noisy_states[2]), 'r.', markersize=1, label='Noisy', alpha=0.3)
            axes[1].plot(result.time, np.rad2deg(result.filtered_states[2]), 'g-', linewidth=1.5, label='Filtered', alpha=0.8)
            axes[1].legend(loc='upper right')
        else:
            axes[1].plot(result.time, np.rad2deg(result.theta), 'r-', linewidth=1.5)
        axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[1].set_ylabel('Angle (deg)')
        axes[1].grid(True, alpha=0.3)
        
        if n_plots > 2:
            if has_forces:
                axes[2].plot(result.time, result.control_forces, 'g-', linewidth=1.5, label='Control')
            axes[2].set_ylabel('Control Force (N)')
            axes[2].grid(True, alpha=0.3)
            axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
            axes[2].legend(loc='upper right')
            
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
            print(f"Saved to {save_path}")
        return fig
    
    def animate(self, result: SimulationResult, interval: int = 20, save_path: Optional[str] = None) -> FuncAnimation:
        """Animate the cart-pole system."""
        x_min = np.min(result.x) - self.cart_width - self.cart_pole.L - 0.5
        x_max = np.max(result.x) + self.cart_width + self.cart_pole.L + 0.5
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.5, self.cart_pole.L + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Height (m)')
        ax.set_title('Cart-Pole Animation')
        
        # Ground
        ax.axhline(y=0, color='brown', linewidth=2)
        ax.fill_between([x_min, x_max], [-0.5, -0.5], [0, 0], color='burlywood', alpha=0.3)
        
        # Cart
        cart = FancyBboxPatch((0, 0), self.cart_width, self.cart_height,
                              boxstyle="round,pad=0.02", facecolor='steelblue',
                              edgecolor='darkblue', linewidth=2)
        ax.add_patch(cart)
        
        # Wheels
        wheel_left = Circle((0, 0), self.wheel_radius, facecolor='dimgray', edgecolor='black')
        wheel_right = Circle((0, 0), self.wheel_radius, facecolor='dimgray', edgecolor='black')
        ax.add_patch(wheel_left)
        ax.add_patch(wheel_right)
        
        # Pendulum
        rod, = ax.plot([], [], 'o-', color='darkred', linewidth=4, markersize=0)
        bob = Circle((0, 0), self.bob_radius, facecolor='crimson', edgecolor='darkred', linewidth=2)
        ax.add_patch(bob)
        
        # Text displays
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        angle_text = ax.text(0.02, 0.85, '', transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        force_text = ax.text(0.02, 0.75, '', transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        has_forces = result.control_forces is not None
        has_disturbance = result.disturbances is not None
        force_scale = 0.02
        
        control_arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                                   arrowprops=dict(arrowstyle='->', color='green', lw=2))
        disturbance_arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                                       arrowprops=dict(arrowstyle='->', color='red', lw=3))
        
        def init():
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
            t = result.time[frame]
            state = result.states[:, frame]
            x, theta = state[0], state[2]
            
            cart_x = x - self.cart_width / 2
            cart_y = self.wheel_radius
            cart.set_x(cart_x)
            cart.set_y(cart_y)
            
            wheel_left.center = (x - self.cart_width / 4, self.wheel_radius)
            wheel_right.center = (x + self.cart_width / 4, self.wheel_radius)
            
            pivot_x, pivot_y = x, cart_y + self.cart_height
            pend_x, pend_y = self.cart_pole.get_pendulum_position(state)
            pend_y += pivot_y
            
            rod.set_data([pivot_x, pend_x], [pivot_y, pend_y])
            bob.center = (pend_x, pend_y)
            
            time_text.set_text(f'Time: {t:.2f} s')
            angle_text.set_text(f'Angle: {np.rad2deg(theta):.1f}°')
            
            force_info = []
            arrow_y = cart_y + self.cart_height / 2
            
            if has_forces and frame < len(result.control_forces):
                ctrl = result.control_forces[frame]
                force_info.append(f'Control: {ctrl:.1f} N')
                control_arrow.xy = (x + ctrl * force_scale, arrow_y)
                control_arrow.xytext = (x, arrow_y)
            
            if has_disturbance and frame < len(result.disturbances):
                dist = result.disturbances[frame]
                force_info.append(f'Disturbance: {dist:.1f} N')
                disturbance_arrow.xy = (x + dist * force_scale, cart_y + self.cart_height)
                disturbance_arrow.xytext = (x, cart_y + self.cart_height)
            
            force_text.set_text('\n'.join(force_info) if force_info else '')
            return cart, wheel_left, wheel_right, rod, bob, time_text, angle_text, force_text
        
        anim = FuncAnimation(fig, update, frames=len(result.time), init_func=init, blit=True, interval=interval)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=1000//interval)
            print(f"Saved to {save_path}")
        return anim
    
    def show(self):
        """Display all figures."""
        plt.show()
