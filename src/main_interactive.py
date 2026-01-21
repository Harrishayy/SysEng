"""Interactive cart-pole simulation with real-time parameter adjustment."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.animation import FuncAnimation

from cart_pole import CartPole
from controller import PIDController, LQRController, PolePlacementController
from state_filter import NoisyStateProcessor


class InteractiveSimulation:
    """Interactive cart-pole simulation with real-time control."""
    
    def __init__(self):
        self.cart_pole = CartPole(
            cart_mass=1.0, pendulum_mass=0.05, rod_length=0.8,
            cart_friction=0.1, rotational_damping=0.01, gravity=9.81
        )
        
        self.dt = 0.02  # 50 Hz
        self.state = np.array([0.0, 0.0, 0.1, 0.0])
        self.filtered_state = self.state.copy()
        
        self.state_processor = NoisyStateProcessor(
            position_noise_std=0.005, angle_noise_std=0.01,
            tau_position=0.05, tau_angle=0.02, dt=self.dt, seed=None
        )
        
        # Controllers
        self.pid = PIDController(kp=100, ki=0.5, kd=20.0, setpoint=0.0)
        self.lqr = LQRController(
            cart_mass=self.cart_pole.M, pendulum_mass=self.cart_pole.m,
            rod_length=self.cart_pole.L, cart_friction=self.cart_pole.b,
            rotational_damping=self.cart_pole.c, gravity=self.cart_pole.g
        )
        self.pole_placement = PolePlacementController(
            cart_mass=self.cart_pole.M, pendulum_mass=self.cart_pole.m,
            rod_length=self.cart_pole.L, cart_friction=self.cart_pole.b,
            rotational_damping=self.cart_pole.c, gravity=self.cart_pole.g
        )
        
        self.active_controller = 'PID'
        self.disturbance = 0.0
        self.disturbance_duration = 0
        self.noise_enabled = True
        self.running = True
        self.t = 0.0
        
        # History for plots
        self.max_history = 500
        self.time_history = []
        self.theta_history = []
        self.force_history = []
        self.disturbance_history = []
        
        # Visual params
        self.cart_width = 0.4
        self.cart_height = 0.2
        self.wheel_radius = 0.05
        self.bob_radius = 0.08
        
        self._setup_figure()
    
    def _setup_figure(self):
        """Setup figure with all widgets."""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Interactive Cart-Pole Simulation', fontsize=14)
        
        # Animation axes
        self.ax_anim = self.fig.add_axes([0.05, 0.45, 0.55, 0.5])
        self.ax_anim.set_xlim(-3, 3)
        self.ax_anim.set_ylim(-0.5, 1.5)
        self.ax_anim.set_aspect('equal')
        self.ax_anim.grid(True, alpha=0.3)
        self.ax_anim.set_xlabel('Position (m)')
        self.ax_anim.set_ylabel('Height (m)')
        
        # Ground
        self.ax_anim.axhline(y=0, color='brown', linewidth=2)
        self.ax_anim.fill_between([-3, 3], [-0.5, -0.5], [0, 0], color='burlywood', alpha=0.3)
        
        # Cart and pendulum
        self.cart_patch = FancyBboxPatch((0, 0), self.cart_width, self.cart_height,
                                         boxstyle="round,pad=0.02", facecolor='steelblue',
                                         edgecolor='darkblue', linewidth=2)
        self.ax_anim.add_patch(self.cart_patch)
        
        self.wheel_left = Circle((0, 0), self.wheel_radius, facecolor='dimgray', edgecolor='black')
        self.wheel_right = Circle((0, 0), self.wheel_radius, facecolor='dimgray', edgecolor='black')
        self.ax_anim.add_patch(self.wheel_left)
        self.ax_anim.add_patch(self.wheel_right)
        
        self.rod_line, = self.ax_anim.plot([], [], 'o-', color='darkred', linewidth=4, markersize=0)
        self.bob_patch = Circle((0, 0), self.bob_radius, facecolor='crimson', edgecolor='darkred', linewidth=2)
        self.ax_anim.add_patch(self.bob_patch)
        
        self.info_text = self.ax_anim.text(0.02, 0.98, '', transform=self.ax_anim.transAxes, fontsize=10,
                                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        self.dist_arrow = self.ax_anim.annotate('', xy=(0, 0), xytext=(0, 0),
                                                arrowprops=dict(arrowstyle='->', color='red', lw=3))
        
        # Real-time plots
        self.ax_theta = self.fig.add_axes([0.65, 0.7, 0.32, 0.2])
        self.ax_theta.set_ylabel('Angle (deg)')
        self.ax_theta.set_xlim(0, 10)
        self.ax_theta.set_ylim(-30, 30)
        self.ax_theta.grid(True, alpha=0.3)
        self.ax_theta.axhline(0, color='k', linestyle='--', alpha=0.3)
        self.theta_line, = self.ax_theta.plot([], [], 'r-', linewidth=1.5)
        
        self.ax_force = self.fig.add_axes([0.65, 0.45, 0.32, 0.2])
        self.ax_force.set_ylabel('Force (N)')
        self.ax_force.set_xlabel('Time (s)')
        self.ax_force.set_xlim(0, 10)
        self.ax_force.set_ylim(-50, 50)
        self.ax_force.grid(True, alpha=0.3)
        self.ax_force.axhline(0, color='k', linestyle='--', alpha=0.3)
        self.force_line, = self.ax_force.plot([], [], 'g-', linewidth=1.5, label='Control')
        self.dist_line, = self.ax_force.plot([], [], 'r-', linewidth=1.5, label='Disturbance')
        self.ax_force.legend(loc='upper right', fontsize=8)
        
        # Controller selection
        self.ax_radio = self.fig.add_axes([0.05, 0.25, 0.12, 0.15])
        self.ax_radio.set_title('Controller')
        self.radio_controller = RadioButtons(self.ax_radio, ('PID', 'LQR', 'Pole Placement'))
        self.radio_controller.on_clicked(self._on_controller_change)
        
        # PID sliders
        self.ax_kp = self.fig.add_axes([0.25, 0.32, 0.25, 0.03])
        self.ax_ki = self.fig.add_axes([0.25, 0.27, 0.25, 0.03])
        self.ax_kd = self.fig.add_axes([0.25, 0.22, 0.25, 0.03])
        
        self.slider_kp = Slider(self.ax_kp, 'Kp', 0, 200, valinit=100)
        self.slider_ki = Slider(self.ax_ki, 'Ki', 0, 5, valinit=0.5)
        self.slider_kd = Slider(self.ax_kd, 'Kd', 0, 50, valinit=20)
        
        self.slider_kp.on_changed(self._on_pid_change)
        self.slider_ki.on_changed(self._on_pid_change)
        self.slider_kd.on_changed(self._on_pid_change)
        
        # Disturbance controls
        self.ax_dist_slider = self.fig.add_axes([0.25, 0.12, 0.25, 0.03])
        self.slider_disturbance = Slider(self.ax_dist_slider, 'Disturbance', -50, 50, valinit=0)
        
        self.ax_dist_btn = self.fig.add_axes([0.55, 0.11, 0.12, 0.05])
        self.btn_apply_dist = Button(self.ax_dist_btn, 'Apply Impulse')
        self.btn_apply_dist.on_clicked(self._on_apply_disturbance)
        
        # Reset button
        self.ax_reset = self.fig.add_axes([0.55, 0.22, 0.12, 0.05])
        self.btn_reset = Button(self.ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._on_reset)
        
        # Initial angle slider
        self.ax_init_angle = self.fig.add_axes([0.25, 0.05, 0.25, 0.03])
        self.slider_init_angle = Slider(self.ax_init_angle, 'Init Angle (deg)', -45, 45, valinit=5.7)
        
        # Noise toggle
        self.ax_noise = self.fig.add_axes([0.75, 0.25, 0.15, 0.1])
        self.ax_noise.set_title('Noise')
        self.radio_noise = RadioButtons(self.ax_noise, ('On', 'Off'))
        self.radio_noise.on_clicked(self._on_noise_toggle)
    
    def _on_controller_change(self, label):
        self.active_controller = label
        self.pid.reset()
    
    def _on_pid_change(self, val):
        self.pid.kp = self.slider_kp.val
        self.pid.ki = self.slider_ki.val
        self.pid.kd = self.slider_kd.val
        self.pid.reset()
    
    def _on_apply_disturbance(self, event):
        self.disturbance = self.slider_disturbance.val
        self.disturbance_duration = 10  # 0.2 seconds
    
    def _on_reset(self, event):
        init_angle = np.deg2rad(self.slider_init_angle.val)
        self.state = np.array([0.0, 0.0, init_angle, 0.0])
        self.filtered_state = self.state.copy()
        self.state_processor.reset()
        self.pid.reset()
        self.t = 0.0
        self.time_history.clear()
        self.theta_history.clear()
        self.force_history.clear()
        self.disturbance_history.clear()
    
    def _on_noise_toggle(self, label):
        self.noise_enabled = (label == 'On')
    
    def _get_controller(self):
        if self.active_controller == 'PID': return self.pid
        if self.active_controller == 'LQR': return self.lqr
        return self.pole_placement
    
    def _compute_control(self, filtered_state, t):
        controller = self._get_controller()
        if self.active_controller == 'PID':
            return controller.compute(filtered_state[2], filtered_state[3], t)
        return controller.compute(filtered_state, t)
    
    def _step(self):
        control_force = self._compute_control(self.filtered_state, self.t)
        
        dist = self.disturbance if self.disturbance_duration > 0 else 0.0
        if self.disturbance_duration > 0:
            self.disturbance_duration -= 1
        
        total_force = control_force + dist
        state_dot = self.cart_pole.dynamics(self.t, self.state, total_force)
        self.state = self.state + state_dot * self.dt
        
        if self.noise_enabled:
            _, self.filtered_state = self.state_processor.process(self.state)
        else:
            self.filtered_state = self.state.copy()
        
        self.t += self.dt
        self.time_history.append(self.t)
        self.theta_history.append(np.rad2deg(self.state[2]))
        self.force_history.append(control_force)
        self.disturbance_history.append(dist)
        
        if len(self.time_history) > self.max_history:
            self.time_history.pop(0)
            self.theta_history.pop(0)
            self.force_history.pop(0)
            self.disturbance_history.pop(0)
        
        return control_force, dist
    
    def _update_animation(self, frame):
        if not self.running:
            return []
        
        control_force, dist = self._step()
        x, theta = self.state[0], self.state[2]
        
        # Update cart
        cart_x = x - self.cart_width / 2
        cart_y = self.wheel_radius
        self.cart_patch.set_x(cart_x)
        self.cart_patch.set_y(cart_y)
        
        self.wheel_left.center = (x - self.cart_width / 4, self.wheel_radius)
        self.wheel_right.center = (x + self.cart_width / 4, self.wheel_radius)
        
        # Update pendulum
        pivot_x, pivot_y = x, cart_y + self.cart_height
        pend_x, pend_y = self.cart_pole.get_pendulum_position(self.state)
        pend_y += pivot_y
        
        self.rod_line.set_data([pivot_x, pend_x], [pivot_y, pend_y])
        self.bob_patch.center = (pend_x, pend_y)
        
        # Disturbance arrow
        if abs(dist) > 0.1:
            self.dist_arrow.xy = (x + dist * 0.02, cart_y + self.cart_height / 2)
            self.dist_arrow.xytext = (x, cart_y + self.cart_height / 2)
        else:
            self.dist_arrow.xy = (0, -10)
            self.dist_arrow.xytext = (0, -10)
        
        # Info text
        info = f"Time: {self.t:.1f}s\nAngle: {np.rad2deg(theta):.1f}°\nPos: {x:.2f}m\n{self.active_controller}\nControl: {control_force:.1f}N"
        if abs(dist) > 0.1:
            info += f"\nDist: {dist:.1f}N"
        self.info_text.set_text(info)
        
        # Update plots
        if self.time_history:
            t_min = max(0, self.t - 10)
            self.theta_line.set_data(self.time_history, self.theta_history)
            self.ax_theta.set_xlim(t_min, self.t + 0.5)
            
            self.force_line.set_data(self.time_history, self.force_history)
            self.dist_line.set_data(self.time_history, self.disturbance_history)
            self.ax_force.set_xlim(t_min, self.t + 0.5)
        
        return []
    
    def run(self):
        self.anim = FuncAnimation(self.fig, self._update_animation,
                                  interval=int(self.dt * 1000), blit=False)
        plt.show()


def main():
    print("Interactive Cart-Pole Simulation")
    print("=" * 40)
    print("Controls:")
    print("  - Select controller: PID / LQR / Pole Placement")
    print("  - Adjust PID gains with sliders")
    print("  - Apply disturbance impulses")
    print("  - Toggle noise on/off")
    print("=" * 40)
    
    sim = InteractiveSimulation()
    sim.run()


if __name__ == "__main__":
    main()
