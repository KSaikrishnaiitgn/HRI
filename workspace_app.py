import math
import random
import tkinter as tk
from tkinter import colorchooser, messagebox, ttk


class CableWorkspace2R:
    """2R transverse-plane workspace with selectable three/four-cable layouts."""

    HUMAN_LIMITS = ((-140.0, 90.0), (0.0, 140.0))
    FREE_LIMITS = ((-180.0, 180.0), (-180.0, 180.0))

    def __init__(self, root):
        self.root = root
        root.title("2R Workspace with Selectable Cables")
        root.configure(bg="#f1f5f9")
        root.minsize(820, 790)

        self.width, self.height = 900, 520
        self.canvas = tk.Canvas(root, width=self.width, height=self.height,
                                bg="#fbfcfe", highlightthickness=0)
        self.canvas.pack(padx=12, pady=(8, 2))

        self.base = (self.width / 2, self.height / 2)
        self.link1, self.link2 = 105, 90
        self.theta1 = 0.0
        self.theta2 = 0.0
        self.cable_mode = None
        self.constraint_enabled = tk.BooleanVar(value=True)
        self.trace_enabled = tk.BooleanVar(value=False)
        self.trace_color = "#a855f7"
        self.trace_color_swatch = None
        self.joint_vars = [tk.StringVar(value="0.0"), tk.StringVar(value="0.0")]
        self.iteration_var = tk.StringVar(value="1000")
        self.scales = []
        self.workspace_points = []
        self.workspace_layers = []
        self.workspace_plot_color = self.trace_color
        self.overlay_enabled = tk.BooleanVar(value=False)
        self.workspace_target_count = 1000
        self.workspace_attempts = 0
        self.workspace_running = False
        self.tip_trace = []
        self.status_message = "Select a cable layout or plot the workspace."
        self.workspace_message = ""

        tk.Label(root, text="2R Manipulator Workspace — Transverse Plane",
                 font=("Segoe UI", 15, "bold"), bg="#f1f5f9",
                 fg="#0f172a").pack(pady=(2, 0))
        tk.Label(root, text="Hand motion in transverse plane",
                 font=("Segoe UI", 10, "bold"), bg="#f1f5f9",
                 fg="#475569").pack(pady=(0, 1))
        tk.Label(root,
                 text="Shoulder θ₁: −90° to 140°   |   Ankle θ₂: 0° to 140°   "
                      "|   Positive angles rotate anticlockwise",
                 font=("Segoe UI", 9), fg="#475569", bg="#f1f5f9").pack(pady=(1, 2))

        self._build_controls(root)
        self.status = tk.StringVar(value="")
        tk.Label(root, textvariable=self.status, font=("Segoe UI", 9),
                 bg="#f1f5f9", fg="#334155", pady=2,
                 wraplength=self.width - 30).pack()
        self._configure_scale_limits()
        self.draw()
        self.sync_controls()

    def _build_controls(self, root):
        controls = tk.Frame(root, bg="#f1f5f9")
        controls.pack(fill="x", padx=26, pady=(2, 0))

        for index, (label, color) in enumerate((("Shoulder (θ₁)", "#2563eb"),
                                                ("Ankle (θ₂)", "#ea580c"))):
            row = tk.Frame(controls, bg="#f1f5f9")
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, width=17, anchor="w", fg=color,
                     font=("Segoe UI", 9, "bold"), bg="#f1f5f9").pack(side="left")
            scale = tk.Scale(row, from_=self.HUMAN_LIMITS[index][0],
                             to=self.HUMAN_LIMITS[index][1], resolution=1,
                             orient="horizontal", showvalue=False, length=390,
                             bg="#f1f5f9", highlightthickness=0,
                             command=lambda value, i=index: self.on_slider(i, value))
            scale.set(0)
            scale.pack(side="left", fill="x", expand=True, padx=5)
            self.scales.append(scale)
            entry = tk.Entry(row, textvariable=self.joint_vars[index], width=7,
                             justify="right", font=("Segoe UI", 9))
            entry.pack(side="left", padx=(3, 2))
            entry.bind("<Return>", lambda _event, i=index: self.apply_entry(i))
            tk.Label(row, text="°", bg="#f1f5f9", fg="#475569").pack(side="left")
            ttk.Button(row, text="Set", command=lambda i=index: self.apply_entry(i),
                       width=4).pack(side="left", padx=4)

        modes = tk.Frame(controls, bg="#f1f5f9")
        modes.pack(fill="x", pady=(3, 1))
        tk.Label(modes, text="Cable layout:", bg="#f1f5f9", fg="#334155",
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 6))
        ttk.Button(modes, text="3 cables", command=lambda: self.activate_cables(3)).pack(
            side="left", padx=3)
        ttk.Button(modes, text="4 cables", command=lambda: self.activate_cables(4)).pack(
            side="left", padx=3)
        ttk.Button(modes, text="No cables", command=lambda: self.activate_cables(None)).pack(
            side="left", padx=3)
        tk.Label(modes, text="(choose a button to draw a cable layout)",
                 bg="#f1f5f9", fg="#64748b", font=("Segoe UI", 8)).pack(side="left", padx=5)

        options = tk.Frame(controls, bg="#f1f5f9")
        options.pack(fill="x", pady=1)
        ttk.Checkbutton(options, text="Enforce joint and cable constraints",
                        variable=self.constraint_enabled,
                        command=self.on_constraint_toggle).pack(side="left")
        ttk.Checkbutton(options, text="Trace end effector",
                        variable=self.trace_enabled).pack(side="left", padx=(12, 6))
        self.trace_color_swatch = tk.Button(
            options, text="Trace color", command=self.choose_trace_color,
            bg=self.trace_color, fg="white", activebackground=self.trace_color,
            relief="flat", padx=8, pady=2)
        self.trace_color_swatch.pack(side="left")

        workspace = tk.Frame(controls, bg="#f1f5f9")
        workspace.pack(fill="x", pady=(2, 0))
        tk.Label(workspace, text="Workspace iterations:", bg="#f1f5f9",
                 fg="#334155", font=("Segoe UI", 9)).pack(side="left")
        tk.Entry(workspace, textvariable=self.iteration_var, width=9,
                 justify="right").pack(side="left", padx=5)
        ttk.Button(workspace, text="Plot workspace", command=self.plot_workspace).pack(
            side="left", padx=4)
        ttk.Checkbutton(workspace, text="Overlay", variable=self.overlay_enabled).pack(
            side="left", padx=4)
        ttk.Button(workspace, text="Clear plot", command=self.clear_workspace).pack(
            side="left", padx=4)
        ttk.Button(workspace, text="Clear trace", command=self.clear_trace).pack(
            side="right")

    def current_limits(self):
        return self.HUMAN_LIMITS if self.constraint_enabled.get() else self.FREE_LIMITS

    def _configure_scale_limits(self):
        for scale, (minimum, maximum) in zip(self.scales, self.current_limits()):
            scale.configure(from_=minimum, to=maximum)

    def point(self, origin, radius, angle):
        # Canvas y is inverted; positive joint angles are counterclockwise.
        return (origin[0] + radius * math.cos(angle),
                origin[1] - radius * math.sin(angle))

    def geometry(self, theta1=None, theta2=None):
        theta1 = self.theta1 if theta1 is None else theta1
        theta2 = self.theta2 if theta2 is None else theta2
        shoulder = self.base
        ankle = self.point(shoulder, self.link1, theta1)
        foot = self.point(ankle, self.link2, theta1 + theta2)
        link1_points = (self.point(shoulder, self.link1 * 0.32, theta1),
                        self.point(shoulder, self.link1 * 0.72, theta1))
        link2_points = (self.point(ankle, self.link2 * 0.30, theta1 + theta2),
                        self.point(ankle, self.link2 * 0.70, theta1 + theta2))
        return shoulder, ankle, foot, link1_points + link2_points

    def cable_layout(self, mode=None):
        mode = self.cable_mode if mode is None else mode
        if mode == 3:
            return (-230, -260, 230), ("#0891b2", "#db2777", "#be185d"), (0, 2, 3)
        if mode == 4:
            return (230, -230, 260, -260), (
                "#0891b2", "#0e7490", "#db2777", "#be185d"), (0, 1, 2, 3)
        return (), (), ()

    @staticmethod
    def point_to_segment_distance(point, start, end):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length_squared = dx * dx + dy * dy
        if length_squared == 0:
            return math.hypot(point[0] - start[0], point[1] - start[1])
        t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
        t = max(0.0, min(1.0, t))
        nearest = (start[0] + t * dx, start[1] + t * dy)
        return math.hypot(point[0] - nearest[0], point[1] - nearest[1])

    @staticmethod
    def segments_intersect(a, b, c, d):
        def orient(p, q, r):
            return ((q[0] - p[0]) * (r[1] - p[1])
                    - (q[1] - p[1]) * (r[0] - p[0]))

        def on_segment(p, q, r):
            return (min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
                    and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9)

        o1, o2 = orient(a, b, c), orient(a, b, d)
        o3, o4 = orient(c, d, a), orient(c, d, b)
        if ((o1 > 0 > o2 or o2 > 0 > o1)
                and (o3 > 0 > o4 or o4 > 0 > o3)):
            return True
        return ((abs(o1) <= 1e-9 and on_segment(a, c, b))
                or (abs(o2) <= 1e-9 and on_segment(a, d, b))
                or (abs(o3) <= 1e-9 and on_segment(c, a, d))
                or (abs(o4) <= 1e-9 and on_segment(c, b, d)))

    def pose_violation(self, theta1, theta2, mode=None, check_joint_limits=True):
        if not self.constraint_enabled.get():
            return None
        mode = self.cable_mode if mode is None else mode
        limits = self.current_limits()
        d1, d2 = math.degrees(theta1), math.degrees(theta2)
        if check_joint_limits:
            if not limits[0][0] <= d1 <= limits[0][1]:
                return "shoulder angle limit"
            if not limits[1][0] <= d2 <= limits[1][1]:
                return "ankle angle limit"
        if mode not in (3, 4):
            return None

        shoulder, ankle, _foot, all_points = self.geometry(theta1, theta2)
        offsets, colors, indices = self.cable_layout(mode)
        attachments = [all_points[i] for i in indices]
        x0, y0 = self.base
        anchors = [(x0, y0 + offset) for offset in offsets]
        joints = ((shoulder, "shoulder joint"), (ankle, "ankle joint"))
        clearance = 7.0
        for index, (anchor, attachment) in enumerate(zip(anchors, attachments), start=1):
            for joint, name in joints:
                if self.point_to_segment_distance(joint, anchor, attachment) <= clearance:
                    return f"cable C{index} would touch the {name}"

        if mode == 4:
            for first in range(len(anchors)):
                for second in range(first + 1, len(anchors)):
                    if self.segments_intersect(anchors[first], attachments[first],
                                               anchors[second], attachments[second]):
                        return f"cables C{first + 1} and C{second + 1} would cross"
        return None

    def motion_violation(self, start_pose, end_pose, mode=None, check_joint_limits=True):
        delta1 = end_pose[0] - start_pose[0]
        delta2 = end_pose[1] - start_pose[1]
        degrees_moved = max(abs(math.degrees(delta1)), abs(math.degrees(delta2)))
        steps = max(1, math.ceil(degrees_moved))
        for step in range(1, steps + 1):
            fraction = step / steps
            violation = self.pose_violation(start_pose[0] + fraction * delta1,
                                            start_pose[1] + fraction * delta2, mode,
                                            check_joint_limits)
            if violation:
                return violation
        return None

    def draw(self):
        self.canvas.delete("all")
        x0, y0 = self.base
        shoulder, ankle, foot, all_points = self.geometry()

        self.canvas.create_line(45, y0, self.width - 45, y0,
                                fill="#94a3b8", width=1, arrow=tk.LAST)
        self.canvas.create_line(x0, self.height - 8, x0, 8,
                                fill="#94a3b8", width=1, arrow=tk.LAST)
        self.canvas.create_text(self.width - 50, y0 - 12, text="X", fill="#64748b",
                                font=("Segoe UI", 9, "bold"))
        self.canvas.create_text(x0 + 12, 14, text="Y", fill="#64748b",
                                font=("Segoe UI", 9, "bold"))

        for color, points in self.workspace_layers:
            for px, py in points:
                self.canvas.create_oval(px - 1.5, py - 1.5, px + 1.5, py + 1.5,
                                        fill=color, outline="")
        if len(self.tip_trace) > 1:
            coords = [coordinate for point in self.tip_trace for coordinate in point]
            self.canvas.create_line(*coords, fill=self.trace_color, width=2,
                                    smooth=True, splinesteps=18)

        offsets, cable_colors, attachment_indices = self.cable_layout()
        if offsets:
            attachments = [all_points[i] for i in attachment_indices]
            anchors = [(x0, y0 + offset) for offset in offsets]
            for index, (anchor, attachment, color) in enumerate(
                    zip(anchors, attachments, cable_colors), start=1):
                self.canvas.create_line(*anchor, *attachment, fill=color, width=2)
                ax, ay = anchor
                self.canvas.create_oval(ax - 3, ay - 3, ax + 3, ay + 3,
                                        fill=color, outline="white", width=1)
                self.canvas.create_text(ax + 8, ay, text=f"C{index}", anchor="w",
                                        fill=color, font=("Segoe UI", 8, "bold"))

        self.canvas.create_line(*shoulder, *ankle, fill="#2563eb", width=9,
                                capstyle=tk.ROUND)
        self.canvas.create_line(*ankle, *foot, fill="#f97316", width=8,
                                capstyle=tk.ROUND)
        if offsets:
            for index, color in zip(attachment_indices, cable_colors):
                ax, ay = all_points[index]
                self.canvas.create_oval(ax - 4, ay - 4, ax + 4, ay + 4,
                                        fill=color, outline="white", width=1)

        self.canvas.create_oval(shoulder[0] - 6, shoulder[1] - 6,
                                shoulder[0] + 6, shoulder[1] + 6,
                                fill="#0f172a", outline="white", width=1)
        self.canvas.create_oval(ankle[0] - 6, ankle[1] - 6,
                                ankle[0] + 6, ankle[1] + 6,
                                fill="white", outline="#1e293b", width=2)
        self.canvas.create_oval(foot[0] - 5, foot[1] - 5,
                                foot[0] + 5, foot[1] + 5,
                                fill="#f97316", outline="white", width=1)
        self.canvas.create_text(shoulder[0] - 10, shoulder[1] + 17,
                                text="shoulder", anchor="e", fill="#334155",
                                font=("Segoe UI", 8))
        self.canvas.create_text(ankle[0] + 9, ankle[1] - 12,
                                text="ankle", anchor="w", fill="#334155",
                                font=("Segoe UI", 8))
        self.canvas.create_text(foot[0] + 10, foot[1] - 10,
                                text="end effector", anchor="w", fill="#334155",
                                font=("Segoe UI", 8))

        cable_label = "none" if self.cable_mode is None else str(self.cable_mode)
        limits = self.current_limits()
        self.status.set(
            f"θ₁ {math.degrees(self.theta1):.1f}° [{limits[0][0]:g},{limits[0][1]:g}]   "
            f"θ₂ {math.degrees(self.theta2):.1f}° [{limits[1][0]:g},{limits[1][1]:g}]   "
            f"Cables: {cable_label}   Constraints: "
            f"{'on' if self.constraint_enabled.get() else 'off'}   "
            f"{self.status_message} {self.workspace_message}")

    def sync_controls(self):
        self.updating_controls = True
        try:
            for index, angle in enumerate((self.theta1, self.theta2)):
                self.joint_vars[index].set(f"{math.degrees(angle):.1f}")
                self.scales[index].set(math.degrees(angle))
        finally:
            self.updating_controls = False

    def set_pose(self, theta1, theta2):
        if self.constraint_enabled.get():
            violation = self.motion_violation(
                (self.theta1, self.theta2), (theta1, theta2),
                check_joint_limits=False)
            if violation:
                self.status_message = f"Blocked: {violation}."
                self.sync_controls()
                self.draw()
                return
        self.theta1, self.theta2 = theta1, theta2
        self.status_message = ""
        self.sync_controls()
        self.record_tip()
        self.draw()

    def on_slider(self, index, value):
        if self.updating_controls:
            return
        if self.workspace_running:
            self.sync_controls()
            return
        angle = math.radians(float(value))
        theta1, theta2 = self.theta1, self.theta2
        if index == 0:
            theta1 = angle
        else:
            theta2 = angle
        self.set_pose(theta1, theta2)

    def apply_entry(self, index):
        if self.workspace_running:
            self.sync_controls()
            return
        try:
            degrees = float(self.joint_vars[index].get())
        except ValueError:
            messagebox.showerror("Invalid angle", "Enter a joint angle in degrees.")
            self.sync_controls()
            return
        minimum, maximum = self.current_limits()[index]
        if not minimum <= degrees <= maximum:
            messagebox.showerror("Angle outside limits",
                                 f"Enter an angle from {minimum:g}° to {maximum:g}°.")
            self.sync_controls()
            return
        theta1, theta2 = self.theta1, self.theta2
        if index == 0:
            theta1 = math.radians(degrees)
        else:
            theta2 = math.radians(degrees)
        self.set_pose(theta1, theta2)

    def on_constraint_toggle(self):
        if self.workspace_running:
            self.constraint_enabled.set(not self.constraint_enabled.get())
            return
        if self.constraint_enabled.get():
            # Enabling constraints checks the current cable geometry only. The
            # current pose is preserved even when it is outside the angle range.
            violation = self.pose_violation(self.theta1, self.theta2,
                                            check_joint_limits=False)
            if violation:
                self.constraint_enabled.set(False)
                self.status_message = (
                    f"Cannot enable constraints: {violation}; reposition the limb.")
                self.draw()
                return
            self.status_message = "Joint and cable constraints enabled."
        else:
            self.status_message = "Joint and cable constraints disabled."
        self._configure_scale_limits()
        self.sync_controls()
        self.draw()

    def activate_cables(self, mode):
        if self.workspace_running:
            return
        old_mode = self.cable_mode
        self.cable_mode = mode
        violation = (self.pose_violation(self.theta1, self.theta2, mode,
                                         check_joint_limits=False)
                     if mode in (3, 4) else None)
        if violation:
            self.cable_mode = old_mode
            self.status_message = f"Could not activate {mode}-cable layout: {violation}."
        else:
            self.status_message = f"{mode}-cable layout activated."
        self.draw()

    def choose_trace_color(self):
        selected = colorchooser.askcolor(color=self.trace_color,
                                         title="Choose end-effector trace color")[1]
        if selected:
            self.trace_color = selected
            self.trace_color_swatch.configure(bg=selected, activebackground=selected)
            self.draw()

    def record_tip(self):
        if self.trace_enabled.get():
            _shoulder, _ankle, foot = self.geometry()[:3]
            self.tip_trace.append(foot)

    def clear_trace(self):
        self.tip_trace.clear()
        self.draw()

    def clear_workspace(self):
        self.workspace_points.clear()
        self.workspace_layers.clear()
        self.workspace_message = "Workspace plot cleared."
        self.draw()

    def plot_workspace(self):
        if self.workspace_running:
            return
        try:
            iterations = int(self.iteration_var.get())
            if iterations <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid iteration count",
                                 "Enter a positive whole number of iterations.")
            return

        self.workspace_target_count = iterations
        self.workspace_attempts = 0
        if not self.overlay_enabled.get():
            self.workspace_layers.clear()
        self.workspace_points = []
        self.workspace_plot_color = self.trace_color
        self.workspace_layers.append((self.workspace_plot_color, self.workspace_points))
        self.workspace_running = True
        self.workspace_message = f"Sampling 0 / {iterations:,}."
        self.status_message = ""
        self.draw()
        self.root.after(1, self._workspace_step)

    def _workspace_step(self):
        batch_size = 10
        max_attempts = max(1000, self.workspace_target_count * 100)
        limits = self.current_limits()
        for _ in range(batch_size):
            if (len(self.workspace_points) >= self.workspace_target_count
                    or self.workspace_attempts >= max_attempts):
                break
            theta1 = math.radians(random.uniform(*limits[0]))
            theta2 = math.radians(random.uniform(*limits[1]))
            self.workspace_attempts += 1
            if self.constraint_enabled.get():
                if self.motion_violation((self.theta1, self.theta2), (theta1, theta2)):
                    continue
            self.theta1, self.theta2 = theta1, theta2
            self.sync_controls()
            _shoulder, _ankle, foot = self.geometry()[:3]
            self.workspace_points.append(foot)
            if self.trace_enabled.get():
                self.tip_trace.append(foot)

        complete = (len(self.workspace_points) >= self.workspace_target_count
                    or self.workspace_attempts >= max_attempts)
        if complete:
            self.workspace_running = False
            if len(self.workspace_points) == self.workspace_target_count:
                self.workspace_message = f"Workspace plotted: {len(self.workspace_points):,} points."
            else:
                self.workspace_message = (
                    f"Plotted {len(self.workspace_points):,} of "
                    f"{self.workspace_target_count:,} points; remaining moves were blocked.")
        else:
            self.workspace_message = (
                f"Sampling workspace: {len(self.workspace_points):,} / "
                f"{self.workspace_target_count:,} points.")
        self.draw()
        if not complete:
            self.root.after(10, self._workspace_step)


if __name__ == "__main__":
    window = tk.Tk()
    CableWorkspace2R(window)
    window.mainloop()
