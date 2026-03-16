"""Interactive cart-pole simulation with motor model and real-time parameter adjustment."""
import matplotlib
matplotlib.use('qtagg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.animation import FuncAnimation

from cart_pole import CartPole
from controller import PIDController, LQRController, PolePlacementController
from state_filter import NoisyStateProcessor
from motor import MotorModel

SAFETY_ANGLE_CUTOFF = 0.52      # rad (~30 deg)


class InteractiveSimulation:
    """Interactive cart-pole simulation with motor model."""

    def __init__(self):
        # Physical model — hardware parameters from LQRPendulum.ino
        # Dynamics use realistic friction; controller linearisation uses b=0, c=0
        self.cart_pole = CartPole(
            cart_mass=1.2, pendulum_mass=0.91, rod_length=0.5,
            cart_friction=0.1, rotational_damping=0.01, gravity=9.81,
            rod_full_length=0.6
        )

        # Motor model — Pololu #4862, 4 motors, 40mm wheels, 10.6V supply
        self.motor = MotorModel(
            num_motors=4, wheel_radius=0.04,
            voltage_min=0.0, voltage_max=10.6,
            rpm_at_nominal=800.0, voltage_nominal=12.0
        )

        self.dt = 0.002              # 500 Hz physics — matches hardware loop rate
        self.render_every = 10       # render at 50 Hz (every 10 physics steps)
        self.state = np.array([0.0, 0.0, 0.1, 0.0])
        self.filtered_state = self.state.copy()

        # Filter at dt=0.002s auto-scales to hardware-identical params:
        # MA=5, theta_dot_alpha=0.20, x_dot_alpha=0.10
        self.state_processor = NoisyStateProcessor(
            position_noise_std=0.001, angle_noise_std=0.002,
            dt=self.dt, seed=None
        )

        # Controllers — linearisation uses b=0, c=0 to match .ino A/B matrices
        self.pid = PIDController(x_target=2.0)
        self.lqr = LQRController(
            cart_mass=self.cart_pole.M, pendulum_mass=self.cart_pole.m,
            rod_length=self.cart_pole.L, cart_friction=0.0,
            rotational_damping=0.0, gravity=self.cart_pole.g,
            setpoint=[2.0, 0.0, 0.0, 0.0]
        )
        self.pole_placement = PolePlacementController(
            cart_mass=self.cart_pole.M, pendulum_mass=self.cart_pole.m,
            rod_length=self.cart_pole.L, cart_friction=0.0,
            rotational_damping=0.0, gravity=self.cart_pole.g,
            setpoint=[2.0, 0.0, 0.0, 0.0]
        )

        self.active_controller = 'None'
        self.disturbance = 0.0
        self.disturbance_duration = 0
        self.pendulum_disturbance = 0.0
        self.pendulum_disturbance_duration = 0
        self.noise_enabled = True
        self.motor_enabled = True
        self.running = True
        self.t = 0.0

        # History for plots
        self.max_history = 500
        self.time_history = []
        self.theta_history = []
        self.position_history = []
        self.desired_force_history = []
        self.actual_force_history = []
        self.voltage_history = []

        # Visual params
        self.cart_width = 0.4
        self.cart_height = 0.2
        self.wheel_radius_vis = 0.05
        self.bob_radius = 0.08

        self._setup_figure()

    def _setup_figure(self):
        """Setup figure with all widgets."""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Interactive Cart-Pole with Motor Model', fontsize=14)

        # Animation axes
        self.ax_anim = self.fig.add_axes([0.06, 0.45, 0.55, 0.5])
        self.ax_anim.set_xlim(-3, 3)
        self.ax_anim.set_ylim(-0.5, 1.5)
        self.ax_anim.set_aspect('equal')
        self.ax_anim.grid(True, alpha=0.3)
        self.ax_anim.set_xlabel('Position (m)')
        self.ax_anim.set_ylabel('Height (m)')

        self.ax_anim.axhline(y=0, color='brown', linewidth=2)
        self.ax_anim.fill_between([-3, 3], [-0.5, -0.5], [0, 0], color='burlywood', alpha=0.3)

        # Cart and pendulum
        self.cart_patch = FancyBboxPatch((0, 0), self.cart_width, self.cart_height,
                                         boxstyle="round,pad=0.02", facecolor='steelblue',
                                         edgecolor='darkblue', linewidth=2)
        self.ax_anim.add_patch(self.cart_patch)

        self.wheel_left = Circle((0, 0), self.wheel_radius_vis, facecolor='dimgray', edgecolor='black')
        self.wheel_right = Circle((0, 0), self.wheel_radius_vis, facecolor='dimgray', edgecolor='black')
        self.ax_anim.add_patch(self.wheel_left)
        self.ax_anim.add_patch(self.wheel_right)

        self.rod_line, = self.ax_anim.plot([], [], 'o-', color='darkred', linewidth=4, markersize=0)
        self.bob_patch = Circle((0, 0), self.bob_radius, facecolor='crimson', edgecolor='darkred', linewidth=2)
        self.ax_anim.add_patch(self.bob_patch)

        self.info_text = self.ax_anim.text(0.02, 0.98, '', transform=self.ax_anim.transAxes, fontsize=9,
                                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        self.dist_arrow = self.ax_anim.annotate('', xy=(0, 0), xytext=(0, 0),
                                                arrowprops=dict(arrowstyle='->', color='red', lw=3))

        # Real-time plots
        self.ax_theta = self.fig.add_axes([0.68, 0.77, 0.30, 0.13])
        self.ax_theta.set_ylabel('Angle (deg)')
        self.ax_theta.set_xlim(0, 10)
        self.ax_theta.set_ylim(-30, 30)
        self.ax_theta.grid(True, alpha=0.3)
        self.ax_theta.axhline(0, color='k', linestyle='--', alpha=0.3)
        self.ax_theta.tick_params(labelbottom=False)
        self.theta_line, = self.ax_theta.plot([], [], 'r-', linewidth=1.5)

        self.ax_position = self.fig.add_axes([0.68, 0.62, 0.30, 0.13])
        self.ax_position.set_ylabel('Position (m)')
        self.ax_position.set_xlim(0, 10)
        self.ax_position.set_ylim(-3, 3)
        self.ax_position.grid(True, alpha=0.3)
        self.ax_position.axhline(2.0, color='g', linestyle='--', alpha=0.5, label='Target')
        self.ax_position.axhline(0, color='k', linestyle='--', alpha=0.3)
        self.ax_position.tick_params(labelbottom=False)
        self.position_line, = self.ax_position.plot([], [], 'b-', linewidth=1.5)

        self.ax_force = self.fig.add_axes([0.68, 0.47, 0.30, 0.13])
        self.ax_force.set_ylabel('Force (N)')
        self.ax_force.set_xlim(0, 10)
        self.ax_force.set_ylim(-20, 20)
        self.ax_force.grid(True, alpha=0.3)
        self.ax_force.axhline(0, color='k', linestyle='--', alpha=0.3)
        self.ax_force.tick_params(labelbottom=False)
        self.desired_line, = self.ax_force.plot([], [], 'b--', linewidth=1, label='Desired', alpha=0.7)
        self.actual_line, = self.ax_force.plot([], [], 'g-', linewidth=1.5, label='Motor')
        self.ax_force.legend(loc='upper right', fontsize=8)

        self.ax_voltage = self.fig.add_axes([0.68, 0.32, 0.30, 0.13])
        self.ax_voltage.set_ylabel('Voltage (V)')
        self.ax_voltage.set_xlabel('Time (s)')
        self.ax_voltage.set_xlim(0, 10)
        self.ax_voltage.set_ylim(-0.5, 12)
        self.ax_voltage.grid(True, alpha=0.3)
        self.ax_voltage.axhline(10.6, color='r', linestyle=':', alpha=0.5)
        self.voltage_line, = self.ax_voltage.plot([], [], 'm-', linewidth=1.5)

        # Controller selection
        self.ax_radio = self.fig.add_axes([0.05, 0.15, 0.12, 0.18])
        self.ax_radio.set_title('Controller')
        self.radio_controller = RadioButtons(self.ax_radio, ('None', 'PID', 'LQR', 'Pole Placement'))
        self.radio_controller.on_clicked(self._on_controller_change)

        # PID sliders — angle inner loop (left) + position outer loop (right)
        self.ax_kp     = self.fig.add_axes([0.22, 0.28, 0.22, 0.025])
        self.ax_ki     = self.fig.add_axes([0.22, 0.24, 0.22, 0.025])
        self.ax_kd     = self.fig.add_axes([0.22, 0.20, 0.22, 0.025])
        self.ax_kp_pos = self.fig.add_axes([0.52, 0.28, 0.10, 0.025])
        self.ax_ki_pos = self.fig.add_axes([0.52, 0.24, 0.10, 0.025])
        self.ax_kd_pos = self.fig.add_axes([0.52, 0.20, 0.10, 0.025])
        self.pid_axes = [self.ax_kp, self.ax_ki, self.ax_kd,
                         self.ax_kp_pos, self.ax_ki_pos, self.ax_kd_pos]

        self.slider_kp     = Slider(self.ax_kp,     'Kp',     0,    100,  valinit=55.0)
        self.slider_ki     = Slider(self.ax_ki,     'Ki',     0,    5,    valinit=2.0)
        self.slider_kd     = Slider(self.ax_kd,     'Kd',     0,    40,   valinit=26.0)
        self.slider_kp_pos = Slider(self.ax_kp_pos, 'Kp_pos', 0,    0.5,  valinit=0.05)
        self.slider_ki_pos = Slider(self.ax_ki_pos, 'Ki_pos', 0,    0.05, valinit=0.004)
        self.slider_kd_pos = Slider(self.ax_kd_pos, 'Kd_pos', 0,    1.0,  valinit=0.11)

        for s in (self.slider_kp, self.slider_ki, self.slider_kd,
                  self.slider_kp_pos, self.slider_ki_pos, self.slider_kd_pos):
            s.on_changed(self._on_pid_change)

        # LQR sliders (Q diagonal and R) — defaults match .ino Python header
        self.lqr_axes = []
        self.ax_q1 = self.fig.add_axes([0.22, 0.28, 0.22, 0.025])
        self.ax_q2 = self.fig.add_axes([0.22, 0.24, 0.22, 0.025])
        self.ax_q3 = self.fig.add_axes([0.22, 0.20, 0.22, 0.025])
        self.ax_q4 = self.fig.add_axes([0.22, 0.16, 0.22, 0.025])
        self.ax_r = self.fig.add_axes([0.48, 0.28, 0.10, 0.025])
        self.lqr_axes = [self.ax_q1, self.ax_q2, self.ax_q3, self.ax_q4, self.ax_r]

        self.slider_q1 = Slider(self.ax_q1, 'Q[x]', 0.1, 20, valinit=5.0)
        self.slider_q2 = Slider(self.ax_q2, 'Q[v]', 0.1, 10, valinit=1.0)
        self.slider_q3 = Slider(self.ax_q3, 'Q[θ]', 1, 200, valinit=100.0)
        self.slider_q4 = Slider(self.ax_q4, 'Q[ω]', 0.1, 30, valinit=10.0)
        self.slider_r = Slider(self.ax_r, 'R', 0.001, 1.0, valinit=0.01)

        self.slider_q1.on_changed(self._on_lqr_change)
        self.slider_q2.on_changed(self._on_lqr_change)
        self.slider_q3.on_changed(self._on_lqr_change)
        self.slider_q4.on_changed(self._on_lqr_change)
        self.slider_r.on_changed(self._on_lqr_change)

        # Pole Placement sliders
        self.pole_axes = []
        self.ax_p1 = self.fig.add_axes([0.22, 0.28, 0.22, 0.025])
        self.ax_p2 = self.fig.add_axes([0.22, 0.24, 0.22, 0.025])
        self.ax_p3 = self.fig.add_axes([0.22, 0.20, 0.22, 0.025])
        self.ax_p4 = self.fig.add_axes([0.22, 0.16, 0.22, 0.025])
        self.pole_axes = [self.ax_p1, self.ax_p2, self.ax_p3, self.ax_p4]

        self.slider_p1 = Slider(self.ax_p1, 'p1', -20, -0.5, valinit=-3.0)
        self.slider_p2 = Slider(self.ax_p2, 'p2', -20, -0.5, valinit=-5.0)
        self.slider_p3 = Slider(self.ax_p3, 'p3', -20, -0.5, valinit=-7.0)
        self.slider_p4 = Slider(self.ax_p4, 'p4', -20, -0.5, valinit=-9.0)

        self.slider_p1.on_changed(self._on_pole_change)
        self.slider_p2.on_changed(self._on_pole_change)
        self.slider_p3.on_changed(self._on_pole_change)
        self.slider_p4.on_changed(self._on_pole_change)

        # Hide all controller sliders initially
        self._hide_all_sliders()

        # Reset button
        self.ax_reset = self.fig.add_axes([0.52, 0.15, 0.10, 0.04])
        self.btn_reset = Button(self.ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._on_reset)

        # Disturbance controls
        self.ax_pend_dist_slider = self.fig.add_axes([0.22, 0.10, 0.22, 0.025])
        self.slider_pend_disturbance = Slider(self.ax_pend_dist_slider, 'Pend (N·m)', -2, 2, valinit=0)

        self.ax_pend_dist_btn = self.fig.add_axes([0.48, 0.095, 0.10, 0.04])
        self.btn_apply_pend_dist = Button(self.ax_pend_dist_btn, 'Pend Impulse')
        self.btn_apply_pend_dist.on_clicked(self._on_apply_pendulum_disturbance)

        self.ax_dist_slider = self.fig.add_axes([0.22, 0.06, 0.22, 0.025])
        self.slider_disturbance = Slider(self.ax_dist_slider, 'Cart (N)', -30, 30, valinit=0)

        self.ax_dist_btn = self.fig.add_axes([0.48, 0.055, 0.10, 0.04])
        self.btn_apply_dist = Button(self.ax_dist_btn, 'Cart Impulse')
        self.btn_apply_dist.on_clicked(self._on_apply_disturbance)

        # Initial angle slider
        self.ax_init_angle = self.fig.add_axes([0.22, 0.02, 0.22, 0.025])
        self.slider_init_angle = Slider(self.ax_init_angle, 'Init θ (deg)', -45, 45, valinit=5.7)

        # Toggles
        self.ax_noise = self.fig.add_axes([0.65, 0.08, 0.10, 0.12])
        self.ax_noise.set_title('Noise', fontsize=9)
        self.radio_noise = RadioButtons(self.ax_noise, ('On', 'Off'), active=0)
        self.radio_noise.on_clicked(self._on_noise_toggle)

        self.ax_motor = self.fig.add_axes([0.78, 0.08, 0.10, 0.12])
        self.ax_motor.set_title('Motor', fontsize=9)
        self.radio_motor = RadioButtons(self.ax_motor, ('On', 'Off'), active=0)
        self.radio_motor.on_clicked(self._on_motor_toggle)

    def _hide_all_sliders(self):
        """Hide all controller-specific sliders."""
        for ax in self.pid_axes + self.lqr_axes + self.pole_axes:
            ax.set_visible(False)

    def _show_pid_sliders(self):
        """Show PID sliders only."""
        self._hide_all_sliders()
        for ax in self.pid_axes:
            ax.set_visible(True)

    def _show_lqr_sliders(self):
        """Show LQR sliders only."""
        self._hide_all_sliders()
        for ax in self.lqr_axes:
            ax.set_visible(True)

    def _show_pole_sliders(self):
        """Show Pole Placement sliders only."""
        self._hide_all_sliders()
        for ax in self.pole_axes:
            ax.set_visible(True)

    def _on_controller_change(self, label):
        self.active_controller = label
        self.pid.reset()

        if label == 'None':
            self._hide_all_sliders()
        elif label == 'PID':
            self._show_pid_sliders()
        elif label == 'LQR':
            self._show_lqr_sliders()
        elif label == 'Pole Placement':
            self._show_pole_sliders()

        self.fig.canvas.draw_idle()

    def _on_pid_change(self, val):
        self.pid.kp     = self.slider_kp.val
        self.pid.ki     = self.slider_ki.val
        self.pid.kd     = self.slider_kd.val
        self.pid.kp_pos = self.slider_kp_pos.val
        self.pid.ki_pos = self.slider_ki_pos.val
        self.pid.kd_pos = self.slider_kd_pos.val
        self.pid.reset()

    def _on_lqr_change(self, val):
        """Rebuild LQR controller with new Q and R values."""
        Q = np.diag([self.slider_q1.val, self.slider_q2.val,
                     self.slider_q3.val, self.slider_q4.val])
        R = np.array([[self.slider_r.val]])
        self.lqr = LQRController(
            cart_mass=self.cart_pole.M, pendulum_mass=self.cart_pole.m,
            rod_length=self.cart_pole.L, cart_friction=0.0,
            rotational_damping=0.0, gravity=self.cart_pole.g,
            Q=Q, R=R, setpoint=[2.0, 0.0, 0.0, 0.0]
        )

    def _on_pole_change(self, val):
        """Rebuild Pole Placement controller with new poles."""
        poles = np.array([self.slider_p1.val, self.slider_p2.val,
                         self.slider_p3.val, self.slider_p4.val])
        self.pole_placement = PolePlacementController(
            cart_mass=self.cart_pole.M, pendulum_mass=self.cart_pole.m,
            rod_length=self.cart_pole.L, cart_friction=0.0,
            rotational_damping=0.0, gravity=self.cart_pole.g,
            poles=poles, setpoint=[2.0, 0.0, 0.0, 0.0]
        )

    def _on_apply_disturbance(self, event):
        self.disturbance = self.slider_disturbance.val
        self.disturbance_duration = int(0.2 / self.dt)  # 0.2 seconds

    def _on_apply_pendulum_disturbance(self, event):
        self.pendulum_disturbance = self.slider_pend_disturbance.val
        self.pendulum_disturbance_duration = int(0.2 / self.dt)  # 0.2 seconds

    def _on_reset(self, event):
        init_angle = np.deg2rad(self.slider_init_angle.val)
        self.state = np.array([0.0, 0.0, init_angle, 0.0])
        self.filtered_state = self.state.copy()
        self.state_processor.reset()
        self.pid.reset()
        self.disturbance_duration = 0
        self.pendulum_disturbance_duration = 0
        self.t = 0.0
        self.time_history.clear()
        self.theta_history.clear()
        self.position_history.clear()
        self.desired_force_history.clear()
        self.actual_force_history.clear()
        self.voltage_history.clear()

    def _on_noise_toggle(self, label):
        self.noise_enabled = (label == 'On')

    def _on_motor_toggle(self, label):
        self.motor_enabled = (label == 'On')

    def _get_controller(self):
        if self.active_controller == 'None': return None
        if self.active_controller == 'PID': return self.pid
        if self.active_controller == 'LQR': return self.lqr
        return self.pole_placement

    def _compute_control(self, filtered_state, t):
        if self.active_controller == 'None':
            return 0.0

        # Safety cutoff: LQR/PP only valid near upright (matches .ino)
        if self.active_controller in ('LQR', 'Pole Placement'):
            if abs(filtered_state[2]) > SAFETY_ANGLE_CUTOFF:
                return 0.0
            return self._get_controller().compute(filtered_state, t)

        return self._get_controller().compute(filtered_state, t)

    def _step(self):
        """Single 500 Hz physics step. Returns values needed for display."""
        cart_velocity = self.state[1]

        desired_force = self._compute_control(self.filtered_state, self.t)

        if self.motor_enabled:
            motor_output = self.motor.compute_motor_output(desired_force, cart_velocity)
            actual_force = motor_output['actual_force']
            voltage = motor_output['voltage']
        else:
            actual_force = desired_force
            voltage = 0.0

        cart_dist = self.disturbance if self.disturbance_duration > 0 else 0.0
        if self.disturbance_duration > 0:
            self.disturbance_duration -= 1

        pend_dist = self.pendulum_disturbance if self.pendulum_disturbance_duration > 0 else 0.0
        if self.pendulum_disturbance_duration > 0:
            self.pendulum_disturbance_duration -= 1

        total_force = actual_force + cart_dist

        state_dot = self.cart_pole.dynamics(self.t, self.state, total_force, pend_dist)
        self.state = self.cart_pole.apply_constraints(self.state + state_dot * self.dt)

        if self.noise_enabled:
            _, self.filtered_state = self.state_processor.process(self.state)
        else:
            self.filtered_state = self.state.copy()

        self.t += self.dt

        return desired_force, actual_force, voltage, cart_dist, pend_dist

    def _update_animation(self, frame):
        if not self.running:
            return []

        # Run render_every physics steps at 500Hz per 50Hz render frame
        for _ in range(self.render_every - 1):
            self._step()
        desired_force, actual_force, voltage, cart_dist, pend_dist = self._step()

        # Record history at render rate (50Hz)
        self.time_history.append(self.t)
        self.theta_history.append(np.rad2deg(self.state[2]))
        self.position_history.append(self.state[0])
        self.desired_force_history.append(desired_force)
        self.actual_force_history.append(actual_force)
        self.voltage_history.append(voltage)
        if len(self.time_history) > self.max_history:
            self.time_history.pop(0)
            self.theta_history.pop(0)
            self.position_history.pop(0)
            self.desired_force_history.pop(0)
            self.actual_force_history.pop(0)
            self.voltage_history.pop(0)

        x, theta = self.state[0], self.state[2]

        # Update cart
        cart_x = x - self.cart_width / 2
        cart_y = self.wheel_radius_vis
        self.cart_patch.set_x(cart_x)
        self.cart_patch.set_y(cart_y)

        self.wheel_left.center = (x - self.cart_width / 4, self.wheel_radius_vis)
        self.wheel_right.center = (x + self.cart_width / 4, self.wheel_radius_vis)

        # Update pendulum — rod drawn to physical tip (0.6m), bob at CoM (0.5m)
        pivot_x, pivot_y = x, cart_y + self.cart_height
        com_x, com_y = self.cart_pole.get_pendulum_position(self.state)
        com_y += pivot_y
        tip_x, tip_y = self.cart_pole.get_rod_tip_position(self.state)
        tip_y += pivot_y

        self.rod_line.set_data([pivot_x, tip_x], [pivot_y, tip_y])
        self.bob_patch.center = (com_x, com_y)

        # Cart disturbance arrow
        if abs(cart_dist) > 0.1:
            self.dist_arrow.xy = (x + cart_dist * 0.02, cart_y + self.cart_height / 2)
            self.dist_arrow.xytext = (x, cart_y + self.cart_height / 2)
        else:
            self.dist_arrow.xy = (0, -10)
            self.dist_arrow.xytext = (0, -10)

        if abs(pend_dist) > 0.01:
            self.bob_patch.set_facecolor('orange')
        else:
            self.bob_patch.set_facecolor('crimson')

        # Info text
        ctrl_name = "Uncontrolled" if self.active_controller == 'None' else self.active_controller
        info = f"Time: {self.t:.1f}s\nAngle: {np.rad2deg(theta):.1f}°\nPos: {x:.2f}m\n"
        info += f"{ctrl_name}"
        if self.active_controller != 'None':
            info += f"\nDesired: {desired_force:.1f}N\nMotor: {actual_force:.1f}N"
            if self.motor_enabled:
                info += f"\nVoltage: {voltage:.1f}V"
        if abs(cart_dist) > 0.1:
            info += f"\nCart Dist: {cart_dist:.1f}N"
        if abs(pend_dist) > 0.01:
            info += f"\nPend Dist: {pend_dist:.2f}N·m"
        self.info_text.set_text(info)

        # Update plots
        if self.time_history:
            t_min = max(0, self.t - 10)

            self.theta_line.set_data(self.time_history, self.theta_history)
            self.ax_theta.set_xlim(t_min, self.t + 0.5)

            self.position_line.set_data(self.time_history, self.position_history)
            self.ax_position.set_xlim(t_min, self.t + 0.5)

            self.desired_line.set_data(self.time_history, self.desired_force_history)
            self.actual_line.set_data(self.time_history, self.actual_force_history)
            self.ax_force.set_xlim(t_min, self.t + 0.5)

            self.voltage_line.set_data(self.time_history, self.voltage_history)
            self.ax_voltage.set_xlim(t_min, self.t + 0.5)

        return []

    def run(self):
        self.anim = FuncAnimation(self.fig, self._update_animation,
                                  interval=int(self.dt * self.render_every * 1000), blit=False)
        plt.show()


def main():
    print("Interactive Cart-Pole Simulation with Motor Model")
    print("=" * 50)
    print("Controls:")
    print("  - Select controller: None / PID / LQR / Pole Placement")
    print("  - Adjust controller gains via sliders")
    print("  - Apply disturbance impulses to cart or pendulum")
    print("  - Toggle noise and motor model on/off")
    print()
    print("Objective: Stabilize pendulum upright AND move cart to x=2m")
    print("Hardware: M=1.2kg, m=0.91kg, L=0.5m")
    print("Motor: 4x Pololu #4862, 10.6V supply, 40mm wheels (~14.73N max)")
    print("Setpoint: x=2m directly (no slew rate governor)")
    print("=" * 50)

    sim = InteractiveSimulation()
    sim.run()


if __name__ == "__main__":
    main()
