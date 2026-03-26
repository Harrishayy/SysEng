"""Generate MP4 videos: 3 controllers side-by-side for 3 scenarios."""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.animation import FFMpegWriter
import numpy as np

# Use bundled ffmpeg from imageio if system ffmpeg is absent
try:
    import imageio.plugins.ffmpeg as _iff
    matplotlib.rcParams['animation.ffmpeg_path'] = _iff.get_exe()
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cart_pole import CartPole
from controller import PIDController, LQRController, PolePlacementController
from state_filter import NoisyStateProcessor
from motor import MotorModel

# ── Simulation constants (match main_interactive.py exactly) ─────────────────
SAFETY_ANGLE_CUTOFF = 0.52   # rad (~30°)
DT = 0.002                   # 500 Hz physics
RENDER_EVERY = 10            # render at 50 Hz
FPS = 50
DIST_STEPS = int(0.2 / DT)  # 0.2 s disturbance = 100 physics steps

# ── Visual constants ──────────────────────────────────────────────────────────
CART_W, CART_H = 0.4, 0.2
WHEEL_R = 0.05
BOB_R = 0.07
ROD_L_VIS = 0.6              # full rod length for drawing

CTRL_COLORS = {
    'PID':             '#e74c3c',
    'LQR':             '#27ae60',
    'Pole Placement':  '#2980b9',
}


# ─────────────────────────────────────────────────────────────────────────────
class SimInstance:
    """Independent simulation state for one controller."""

    def __init__(self, name: str, setpoint_x: float):
        self.name = name
        self.setpoint = [setpoint_x, 0.0, 0.0, 0.0]

        self.cp = CartPole(
            cart_mass=1.2, pendulum_mass=0.91, rod_length=0.5,
            cart_friction=0.1, rotational_damping=0.01, gravity=9.81,
            rod_full_length=0.6,
        )
        self.motor = MotorModel(
            num_motors=4, wheel_radius=0.04,
            voltage_min=0.0, voltage_max=10.6,
            rpm_at_nominal=800.0, voltage_nominal=12.0,
        )
        self.filt = NoisyStateProcessor(
            position_noise_std=0.001, angle_noise_std=0.002, dt=DT, seed=42,
        )

        if name == 'PID':
            self.ctrl = PIDController(x_target=setpoint_x)
        elif name == 'LQR':
            self.ctrl = LQRController(
                cart_mass=1.2, pendulum_mass=0.91, rod_length=0.5,
                cart_friction=0.0, rotational_damping=0.0, gravity=9.81,
                setpoint=self.setpoint,
            )
        else:
            self.ctrl = PolePlacementController(
                cart_mass=1.2, pendulum_mass=0.91, rod_length=0.5,
                cart_friction=0.0, rotational_damping=0.0, gravity=9.81,
                setpoint=self.setpoint,
            )

        self.state = np.zeros(4)
        self.filt_state = np.zeros(4)
        self.t = 0.0
        self.actual_force = 0.0
        self.time_hist: list = []
        self.theta_hist: list = []
        self.pos_hist: list = []

    def reset(self, initial_angle_deg: float) -> None:
        self.state = np.array([0.0, 0.0, np.deg2rad(initial_angle_deg), 0.0])
        self.filt_state = self.state.copy()
        self.filt.reset()
        if hasattr(self.ctrl, 'reset'):
            self.ctrl.reset()
        self.t = 0.0
        self.actual_force = 0.0
        self.time_hist.clear()
        self.theta_hist.clear()
        self.pos_hist.clear()

    def step(self, cart_dist: float = 0.0, pend_dist: float = 0.0) -> None:
        if (self.name in ('LQR', 'Pole Placement')
                and abs(self.filt_state[2]) > SAFETY_ANGLE_CUTOFF):
            desired = 0.0
        else:
            desired = self.ctrl.compute(self.filt_state, self.t)

        motor_out = self.motor.compute_motor_output(desired, self.state[1])
        actual = motor_out['actual_force']
        self.actual_force = actual

        ds = self.cp.dynamics(self.t, self.state, actual + cart_dist, pend_dist)
        self.state = self.cp.apply_constraints(self.state + ds * DT)
        _, self.filt_state = self.filt.process(self.state)
        self.t += DT

    def record(self) -> None:
        self.time_hist.append(self.t)
        self.theta_hist.append(np.rad2deg(self.state[2]))
        self.pos_hist.append(self.state[0])


# ─────────────────────────────────────────────────────────────────────────────
def _build_anim_axes(axes, sims, x_lim, setpoint_x):
    """Initialise cart-pole animation patches; return list of artist dicts."""
    artists = []
    for ax, sim in zip(axes, sims):
        ax.cla()
        ax.set_xlim(x_lim)
        ax.set_ylim(-0.3, 1.35)
        ax.set_aspect('equal')
        ax.set_xlabel('Position (m)', fontsize=8)
        ax.grid(True, alpha=0.25, color='#aaa')
        ax.axhline(0, color='#8B4513', lw=2)
        ax.fill_between(x_lim, [-0.3, -0.3], [0, 0], color='#d4a574', alpha=0.25)
        if setpoint_x != 0:
            ax.axvline(setpoint_x, color='#27ae60', lw=1.5, ls='--', alpha=0.8,
                       label=f'target {setpoint_x} m')
            ax.legend(loc='upper right', fontsize=7, framealpha=0.6)

        col = CTRL_COLORS[sim.name]
        cart = FancyBboxPatch((0, 0), CART_W, CART_H,
                              boxstyle='round,pad=0.02',
                              facecolor=col, edgecolor='#333', lw=1.5, alpha=0.9)
        ax.add_patch(cart)
        wl = Circle((0, 0), WHEEL_R, facecolor='#555', edgecolor='#222', lw=1)
        wr = Circle((0, 0), WHEEL_R, facecolor='#555', edgecolor='#222', lw=1)
        ax.add_patch(wl)
        ax.add_patch(wr)
        rod, = ax.plot([], [], '-', color='#5c2c0a', lw=4)
        bob = Circle((0, 0), BOB_R, facecolor='#c0392b', edgecolor='#7b241c', lw=1.5)
        ax.add_patch(bob)
        info = ax.text(0.02, 0.97, '', transform=ax.transAxes, fontsize=7.5,
                       va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.75))
        # Disturbance label – persistent banner shown for 2.5 s after each event
        dlbl = ax.text(0.5, 0.04, '', transform=ax.transAxes,
                       ha='center', va='bottom', fontsize=10, fontweight='bold',
                       color='white',
                       bbox=dict(boxstyle='round,pad=0.4', fc='#c0392b',
                                 ec='#922b21', lw=1.5, alpha=0.0))

        artists.append(dict(cart=cart, wl=wl, wr=wr, rod=rod,
                            bob=bob, info=info, dlbl=dlbl))
    return artists


def _update_anim(art, sim, pend_dist_phys, cart_label: str, pend_label: str):
    """Update all dynamic artists for one controller panel."""
    x, _, theta, _ = sim.state
    cy = WHEEL_R
    art['cart'].set_x(x - CART_W / 2)
    art['cart'].set_y(cy)
    art['wl'].center = (x - CART_W / 4, WHEEL_R)
    art['wr'].center = (x + CART_W / 4, WHEEL_R)

    pivot_y = cy + CART_H
    com_x = x + 0.5 * np.sin(theta)
    com_y = pivot_y + 0.5 * np.cos(theta)
    tip_x = x + ROD_L_VIS * np.sin(theta)
    tip_y = pivot_y + ROD_L_VIS * np.cos(theta)

    art['rod'].set_data([x, tip_x], [pivot_y, tip_y])
    art['bob'].center = (com_x, com_y)
    art['bob'].set_facecolor('#e67e22' if abs(pend_dist_phys) > 0.01 else '#c0392b')

    # Persistent disturbance banner (shown for 2.5 s after each event)
    parts = []
    if cart_label:
        parts.append(cart_label)
    if pend_label:
        parts.append(pend_label)
    if parts:
        art['dlbl'].set_text('\n'.join(parts))
        art['dlbl'].get_bbox_patch().set_alpha(0.88)
    else:
        art['dlbl'].set_text('')
        art['dlbl'].get_bbox_patch().set_alpha(0.0)

    art['info'].set_text(
        f"t = {sim.t:.1f} s\n"
        f"θ = {np.rad2deg(sim.state[2]):.1f}°\n"
        f"x = {sim.state[0]:.2f} m\n"
        f"F = {sim.actual_force:.1f} N"
    )


# ─────────────────────────────────────────────────────────────────────────────
def run_scenario(name, initial_angle_deg, setpoint_x,
                 dist_schedule, duration, out_dir):
    """
    Render one side-by-side (3-column) video and save as MP4.

    dist_schedule : list of (t_trigger_s, cart_N, pend_Nm)
                    Each event is held for 0.2 s.
    """
    os.makedirs(out_dir, exist_ok=True)
    fname = name.replace(' ', '_').lower() + '.mp4'
    out_path = os.path.join(out_dir, fname)

    sims = [SimInstance(n, setpoint_x) for n in ('PID', 'LQR', 'Pole Placement')]
    for s in sims:
        s.reset(initial_angle_deg)

    n_frames = int(duration / (DT * RENDER_EVERY))
    x_lim = (-2.8, 3.5) if setpoint_x > 1 else (-2.5, 2.5)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('#f7f7f7')

    # Reserve top strip for scenario title + column headers
    outer = gridspec.GridSpec(1, 3, figure=fig,
                              wspace=0.12, left=0.04, right=0.98,
                              top=0.84, bottom=0.07)
    anim_axes, theta_axes, pos_axes = [], [], []

    # Scenario title
    fig.text(0.5, 0.975, f'Cart-Pole Simulation  ·  {name}',
             ha='center', va='top', fontsize=14, fontweight='bold')

    # Column header x-positions (figure coordinates, centred on each column)
    col_centres = [0.20, 0.51, 0.82]

    for col, sim in enumerate(sims):
        inner = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer[col],
            height_ratios=[3, 1, 1], hspace=0.45,
        )
        ax_a = fig.add_subplot(inner[0])
        ax_t = fig.add_subplot(inner[1])
        ax_p = fig.add_subplot(inner[2])

        # ── Large column header banner ─────────────────────────────────────
        fig.text(col_centres[col], 0.90,
                 sim.name,
                 ha='center', va='center',
                 fontsize=16, fontweight='bold', color='white',
                 bbox=dict(boxstyle='round,pad=0.45',
                           fc=CTRL_COLORS[sim.name], ec='none'))

        # Small coloured label on each sub-plot so the algo is always visible
        for ax, ylabel in ((ax_t, 'Angle (deg)'), (ax_p, 'Position (m)')):
            ax.text(0.01, 1.0, sim.name,
                    transform=ax.transAxes,
                    ha='left', va='bottom',
                    fontsize=8, fontweight='bold', color=CTRL_COLORS[sim.name])

        # Angle plot
        ax_t.set_ylabel('θ (deg)', fontsize=8)
        ax_t.set_ylim(-36, 36)
        ax_t.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.6)
        ax_t.axhline(30, color='#e74c3c', lw=0.8, ls=':', alpha=0.5)
        ax_t.axhline(-30, color='#e74c3c', lw=0.8, ls=':', alpha=0.5)
        ax_t.grid(True, alpha=0.3)
        ax_t.tick_params(labelsize=7)

        # Position plot
        ax_p.set_ylabel('x (m)', fontsize=8)
        ax_p.set_xlabel('Time (s)', fontsize=8)
        if setpoint_x != 0:
            ax_p.axhline(setpoint_x, color='#27ae60', lw=1.2,
                         ls='--', alpha=0.8, label=f'target {setpoint_x} m')
            ax_p.legend(loc='upper right', fontsize=7, framealpha=0.6)
        ax_p.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.6)
        ax_p.grid(True, alpha=0.3)
        ax_p.tick_params(labelsize=7)

        anim_axes.append(ax_a)
        theta_axes.append(ax_t)
        pos_axes.append(ax_p)

    cart_artists = _build_anim_axes(anim_axes, sims, x_lim, setpoint_x)

    theta_lines = [ax.plot([], [], color=CTRL_COLORS[s.name], lw=1.4)[0]
                   for ax, s in zip(theta_axes, sims)]
    pos_lines   = [ax.plot([], [], color=CTRL_COLORS[s.name], lw=1.4)[0]
                   for ax, s in zip(pos_axes, sims)]

    # Physics disturbance counters (physics steps)
    cd_active = [0, 0, 0]
    pd_active = [0, 0, 0]
    cd_val    = [0.0, 0.0, 0.0]
    pd_val    = [0.0, 0.0, 0.0]

    # Label persistence counters (render frames) – banner stays 2.5 s
    LABEL_FRAMES = int(2.5 / (DT * RENDER_EVERY))   # 125 render frames
    cd_lframes = [0, 0, 0]
    pd_lframes = [0, 0, 0]
    cd_lstr    = ['', '', '']
    pd_lstr    = ['', '', '']

    # ── Y limits (fixed per scenario for stable video) ────────────────────────
    pos_ylim = (-0.8, 3.2) if setpoint_x > 1 else (-1.6, 1.6)
    for ax in pos_axes:
        ax.set_ylim(*pos_ylim)

    HIST_WIN = 10.0  # seconds of history shown in scrolling plots

    writer = FFMpegWriter(
        fps=FPS,
        metadata={'title': name, 'artist': 'CartPoleSim'},
        extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20'],
    )

    print(f"  Rendering '{name}': {n_frames} frames …", flush=True)
    with writer.saving(fig, out_path, dpi=120):
        for frame in range(n_frames):
            t_now = frame * DT * RENDER_EVERY  # start-of-frame simulation time

            # Trigger scheduled disturbances
            for (t_trig, c_N, p_Nm) in dist_schedule:
                if abs(t_now - t_trig) < DT * RENDER_EVERY * 0.5:
                    for i in range(3):
                        if c_N != 0:
                            cd_active[i]  = DIST_STEPS
                            cd_val[i]     = c_N
                            cd_lframes[i] = LABEL_FRAMES
                            direction     = '→' if c_N > 0 else '←'
                            cd_lstr[i]    = f'Cart push  {direction}  {abs(c_N):.0f} N'
                        if p_Nm != 0:
                            pd_active[i]  = DIST_STEPS
                            pd_val[i]     = p_Nm
                            direction     = '↻' if p_Nm > 0 else '↺'
                            pd_lframes[i] = LABEL_FRAMES
                            pd_lstr[i]    = f'Pend tap  {direction}  {abs(p_Nm):.2f} N·m'

            # Advance physics RENDER_EVERY steps
            for _ in range(RENDER_EVERY):
                for i, sim in enumerate(sims):
                    cd = cd_val[i] if cd_active[i] > 0 else 0.0
                    pd = pd_val[i] if pd_active[i] > 0 else 0.0
                    sim.step(cd, pd)
                    if cd_active[i] > 0:
                        cd_active[i] -= 1
                    if pd_active[i] > 0:
                        pd_active[i] -= 1

            for sim in sims:
                sim.record()

            # Scrolling x-axis
            t_end   = sims[0].t
            t_start = max(0.0, t_end - HIST_WIN)
            for ax in theta_axes + pos_axes:
                ax.set_xlim(t_start, t_end + 0.3)

            # Tick down label persistence counters
            for i in range(3):
                if cd_lframes[i] > 0:
                    cd_lframes[i] -= 1
                else:
                    cd_lstr[i] = ''
                if pd_lframes[i] > 0:
                    pd_lframes[i] -= 1
                else:
                    pd_lstr[i] = ''

            # Update artists
            for i, sim in enumerate(sims):
                pd_phys = pd_val[i] if pd_active[i] > 0 else 0.0
                _update_anim(cart_artists[i], sim,
                             pd_phys, cd_lstr[i], pd_lstr[i])
                if sim.time_hist:
                    theta_lines[i].set_data(sim.time_hist, sim.theta_hist)
                    pos_lines[i].set_data(sim.time_hist, sim.pos_hist)

            fig.canvas.draw()
            writer.grab_frame()

    print(f"  Saved → {out_path}")
    plt.close(fig)


# ── Scenario definitions ──────────────────────────────────────────────────────
SCENARIOS = [
    dict(
        name='Disturbance Rejection',
        initial_angle_deg=0.0,
        setpoint_x=0.0,
        duration=22.0,
        # (t_trigger_s, cart_N, pend_Nm)
        dist_schedule=[
            (2.0,   8.0,   0.0),   # small cart push forward
            (5.0,  -8.0,   0.0),   # small cart push backward
            (8.0,   0.0,   0.5),   # medium pendulum tap
            (11.0,  14.0,  0.0),   # larger cart push
            (14.0, -14.0,  0.0),   # larger cart push back
            (17.5,  0.0,  -0.8),   # strong pendulum tap
            (20.5,  18.0,  0.0),   # maximum cart push
        ],
    ),
    dict(
        name='Recovery from 15 Degrees',
        initial_angle_deg=15.0,
        setpoint_x=0.0,
        duration=10.0,
        dist_schedule=[],
    ),
    dict(
        name='2m Sprint',
        initial_angle_deg=0.0,
        setpoint_x=2.0,
        duration=15.0,
        dist_schedule=[],
    ),
]


if __name__ == '__main__':
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'videos'
    )

    print('Cart-Pole Simulation Video Generator')
    print('=' * 45)
    for s in SCENARIOS:
        run_scenario(
            s['name'], s['initial_angle_deg'], s['setpoint_x'],
            s['dist_schedule'], s['duration'], out_dir,
        )
    print('\nAll videos saved to:', os.path.abspath(out_dir))
