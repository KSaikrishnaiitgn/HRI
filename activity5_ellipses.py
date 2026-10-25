"""
ES 656 Human-Robot Interaction - Activity 5
Measuring a limb's ability in the task space.

Planar two-link arm (n = 2 joints, m = 3 spring-muscles).

For each of five real-world postures the script computes
    J    : 2x2 end-point Jacobian            (dx = J dq)
    A    : 2x3 structure matrix              (tau = A T,  ldot = -A^T qdot)
    K_q  : 2x2 joint stiffness              K_q = A K_m A^T       (T = 0)
    K_X  : 2x2 task-space stiffness         K_X = J^{-T} K_q J^{-1}  (F = 0)

and plots velocity manipulability, force manipulability, and task-space
stiffness ellipses for all five postures.

Conventions (same as Activity 4):
    origin = shoulder joint, x = horizontal forward, y = vertically up
    theta1 = upper-arm angle measured from +x axis (CCW positive)
    theta2 = elbow flexion relative to upper arm (0 = fully extended)

Sagittal plane joint limits:  theta1 in [-140, +90] deg,  theta2 in [0, 140] deg
Transverse plane joint limits: theta1 in [-40, +90] deg,  theta2 in [0, 140] deg

Cable anchor grounding: anchors are placed OUTSIDE the workspace donut
(outer radius = L1+L2 = 0.54 m) at r = 0.75 m from shoulder, equally
spaced at 120-deg intervals. This maximises workspace coverage because
every arm configuration can tension at least one cable regardless of
direction, as the cable lines always cross the reachable workspace.

Units: SI throughout (m, N, rad).
Run:  python activity5_ellipses.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# 1.  PARAMETERS
# =============================================================================

# Link lengths measured on Kathroju Saikrishna (humerus / forearm).
L1 = 0.28   # upper arm [m]
L2 = 0.26   # forearm   [m]

# ---- Cable anchor grounding (world frame, metres) ---------------------------
# Placed outside the workspace donut (outer radius = L1+L2 = 0.54 m) at
# r = 0.75 m from the shoulder, separated by 120 deg for maximum coverage.
R_ANCHOR = 0.75   # anchor radius [m]  (> L1+L2 = 0.54 m)
_ang = np.radians([90.0, 210.0, 330.0])
ANCHOR_WORLD = [R_ANCHOR * np.array([np.cos(a), np.sin(a)]) for a in _ang]

# ---- Cable attachment points on the links (link frame, metres) --------------
# Fractions along each link, small lateral offset so moment arm is non-zero.
# Cable 1 -> link 1 (upper arm), Cable 2 -> link 2 (forearm, proximal),
# Cable 3 -> link 2 (forearm, distal).
ATTACH_FRAC   = [0.55, 0.35, 0.65]   # along-link fraction
ATTACH_OFFSET = [0.02, 0.02, 0.02]   # lateral offset [m]
ATTACH_LINKS  = [1,    2,    2    ]   # which link (1 = upper arm, 2 = forearm)

# Build (link, uv) list where uv = attachment in link frame
CABLES = []
fracs  = ATTACH_FRAC
offsets= ATTACH_OFFSET
for i, (frac, off, link) in enumerate(zip(fracs, offsets, ATTACH_LINKS)):
    if link == 1:
        uv = np.array([frac * L1, off])
    else:
        uv = np.array([frac * L2, off])
    CABLES.append((f"C{i+1}", link, ANCHOR_WORLD[i], uv))

# ---- Muscle spring stiffnesses [N/m] ----------------------------------------
# Extension springs; values representative of human arm muscle-tendon units.
K_MUSCLE = np.diag([1200.0, 1500.0, 1000.0])

# ---- Five real-world postures (NAME, theta1 [deg], theta2 [deg]) -----------
# Angles read directly from posture_with_angles photographs.
# Boxing & Dart & Defense: sagittal plane.
# Carrom & Chess: transverse (top-down) plane.
CONFIGS = [
    ("Boxing (sagittal)",    -10.0,  55.0),
    ("Carrom (transverse)",    0.0,  39.0),
    ("Chess (transverse)",   -22.0,  56.0),
    ("Dart (sagittal)",      -15.0,  65.0),
    ("Defense (sagittal)",   -52.0, 140.0),
]

FD_STEP = 1e-7   # central-difference step [rad]

# =============================================================================
# 2.  KINEMATICS
# =============================================================================

def rot2(a):
    """2-D rotation matrix for angle a."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def pos_elbow(q):
    """Elbow position in world frame."""
    return L1 * np.array([np.cos(q[0]), np.sin(q[0])])


def pos_hand(q):
    """Hand (end-effector) position in world frame."""
    return pos_elbow(q) + L2 * np.array([np.cos(q[0]+q[1]), np.sin(q[0]+q[1])])


def jacobian(q):
    """
    Analytic 2x2 end-point Jacobian  J = d(pos_hand)/dq.

    J = [[-L1 sin(q1) - L2 sin(q1+q2),  -L2 sin(q1+q2)],
         [ L1 cos(q1) + L2 cos(q1+q2),   L2 cos(q1+q2)]]
    """
    s1,  c1  = np.sin(q[0]),      np.cos(q[0])
    s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
    return np.array([
        [-L1*s1 - L2*s12, -L2*s12],
        [ L1*c1 + L2*c12,  L2*c12],
    ])


def attach_world(q, link, uv):
    """World position of a cable attachment given in link frame (uv)."""
    if link == 1:
        return rot2(q[0]) @ uv
    return pos_elbow(q) + rot2(q[0]+q[1]) @ uv


def attach_jacobian(q, link, uv):
    """
    Analytic d(attach_world)/dq for one attachment point.
    Used to build the structure matrix A column by column.
    """
    u, v = uv
    if link == 1:
        s1, c1 = np.sin(q[0]), np.cos(q[0])
        dr_dq1 = np.array([-u*s1 - v*c1,  u*c1 - v*s1])
        return np.column_stack([dr_dq1, np.zeros(2)])
    s1,  c1  = np.sin(q[0]),      np.cos(q[0])
    sp,  cp  = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
    col2 = np.array([-u*sp - v*cp,  u*cp - v*sp])
    col1 = np.array([-L1*s1, L1*c1]) + col2
    return np.column_stack([col1, col2])


def cable_lengths(q):
    """Scalar length of each cable at configuration q."""
    return np.array([
        np.linalg.norm(anchor - attach_world(q, link, uv))
        for _, link, anchor, uv in CABLES
    ])


def structure_matrix(q):
    """
    Structure matrix A  (2 x m) such that  tau = A T.

    Column j = (d r_j / d q)^T  l_hat_j
    where l_hat_j = (anchor_j - r_j) / |anchor_j - r_j|  (unit vector along cable).

    Note: this matches the sign convention  l_dot = -A^T q_dot.
    """
    cols = []
    for _, link, anchor, uv in CABLES:
        diff   = anchor - attach_world(q, link, uv)
        l_hat  = diff / np.linalg.norm(diff)
        dr_dq  = attach_jacobian(q, link, uv)   # 2 x 2
        cols.append(dr_dq.T @ l_hat)            # 2 x 1
    return np.column_stack(cols)                 # 2 x m


# =============================================================================
# 3.  STIFFNESS
# =============================================================================

def joint_stiffness(q):
    """
    Joint-space stiffness  K_q = A K_m A^T   [N·m/rad].

    Derivation (T = 0 assumption, as required by the activity):
      K_q dq = A K_m A^T dq     (tension-dependent term  [dA/dq T] dq  dropped)
    """
    A = structure_matrix(q)
    return A @ K_MUSCLE @ A.T


def task_stiffness(q):
    """
    Task-space stiffness  K_X = J^{-T} K_q J^{-1}   [N/m].

    Derivation (F = 0 assumption):
      K_q = J^T K_X J   =>   K_X = J^{-T} K_q J^{-1}
    The geometric term  [dJ^T/dq · F] vanishes because F = 0.
    """
    J    = jacobian(q)
    Jinv = np.linalg.inv(J)
    Kq   = joint_stiffness(q)
    return Jinv.T @ Kq @ Jinv


# =============================================================================
# 4.  FINITE-DIFFERENCE VERIFICATION
# =============================================================================

def _fd(f, q, h=FD_STEP):
    """Central-difference Jacobian of a vector function f: R^2 -> R^n."""
    cols = []
    for i in range(len(q)):
        dq      = np.zeros_like(q, dtype=float)
        dq[i]   = h
        cols.append((f(q + dq) - f(q - dq)) / (2*h))
    return np.column_stack(cols)


def fd_checks(q):
    """Return relative max-abs error between analytic and FD quantities."""
    l0    = cable_lengths(q)   # rest lengths = current lengths  =>  T = 0

    # Torque from springs: tau = A K_m (l - l0)
    def torque(x):
        T = K_MUSCLE @ (cable_lengths(x) - l0)
        return structure_matrix(x) @ T

    # Force at hand: F = J^{-T} tau
    def hand_force(x):
        return np.linalg.solve(jacobian(x).T, torque(x))

    J_fd   = _fd(pos_hand, q)
    A_fd   = -_fd(cable_lengths, q).T              # l_dot = -A^T q_dot
    Kq_fd  = -_fd(torque, q)                       # K_q = -d tau / dq
    KX_fd  = -_fd(hand_force, q) @ np.linalg.inv(jacobian(q))

    def relerr(analytic, fd):
        denom = max(np.abs(fd).max(), 1e-12)
        return np.abs(analytic - fd).max() / denom

    return {
        "J"  : np.abs(jacobian(q)        - J_fd ).max(),
        "A"  : np.abs(structure_matrix(q)- A_fd ).max(),
        "K_q": relerr(joint_stiffness(q),  Kq_fd),
        "K_X": relerr(task_stiffness(q),   KX_fd),
    }


# =============================================================================
# 5.  ELLIPSE COMPUTATION
# =============================================================================

def ellipse_axes(M_or_q, kind):
    """
    Compute semi-axis lengths and directions of a manipulability / stiffness ellipse.

    Velocity manipulability: unit sphere in JS  |q_dot|=1
        x_dot = J q_dot  =>  x_dot^T (J J^T)^{-1} x_dot = 1
        M = J J^T,  semi-axes = sqrt(eigenvalues of M)

    Force manipulability:   unit sphere in JS  |tau|=1  (tau = J^T F)
        F^T (J J^T) F = 1
        M = J J^T,  semi-axes = 1 / sqrt(eigenvalues of M)

    Task-space stiffness:   unit deflection sphere |dx|=1
        dF = K_X dx  =>  dF^T (K_X^{-T} K_X^{-1}) dF = 1
        => ellipse in dF-space, but conventionally plotted as:
        dF^T K_X^{-1} dF = 1  (K_X symmetric positive-definite)
        semi-axes = 1 / sqrt(eigenvalues of K_X)
        (larger axis = direction of LEAST resistance to deflection)

    kind : "velocity" | "force" | "stiffness"
    M_or_q : the 2x2 matrix already formed (JJT or KX)
    """
    lam, vec = np.linalg.eigh(M_or_q)   # lam sorted ascending
    lam = np.maximum(lam, 1e-15)        # numerical safety

    if kind == "velocity":
        semi = np.sqrt(lam)
    elif kind == "force":
        semi = 1.0 / np.sqrt(lam)
    else:  # stiffness
        # stiffness ellipse: dF^T K_X^{-1} dF = 1
        # semi-axis lengths = 1/sqrt(lam_i of K_X)
        semi = 1.0 / np.sqrt(lam)

    # Sort: largest semi-axis first (major axis)
    order = np.argsort(semi)[::-1]
    return semi[order], vec[:, order]


def ellipse_curve(center, semi, vec, scale, n=300):
    """Points on the scaled ellipse boundary."""
    t = np.linspace(0, 2*np.pi, n)
    return center[:, None] + scale * vec @ (semi[:, None] * np.vstack([np.cos(t), np.sin(t)]))


def axis_angle_deg(v):
    """Direction of vector v in degrees, folded to (-90, +90]."""
    ang = np.degrees(np.arctan2(v[1], v[0]))
    return (ang + 90) % 180 - 90


# =============================================================================
# 6.  REPORTING
# =============================================================================

def fmt(M):
    return np.array2string(np.atleast_2d(M), precision=4,
                           suppress_small=True, floatmode="fixed")


def hand_calculation_report(q, name):
    """Print every intermediate step for the hand-verification requirement."""
    print("\n" + "="*72)
    print(f"HAND-VERIFICATION: {name}")
    print(f"  theta1 = {np.degrees(q[0]):.1f} deg,  theta2 = {np.degrees(q[1]):.1f} deg")
    print("="*72)
    print(f"Elbow  = {fmt(pos_elbow(q))} m")
    print(f"Hand   = {fmt(pos_hand(q))} m")
    print(f"\nJ =\n{fmt(jacobian(q))}")
    print(f"det(J) = {np.linalg.det(jacobian(q)):.4f}")

    A = structure_matrix(q)
    for i, (cname, link, anchor, uv) in enumerate(CABLES):
        r    = attach_world(q, link, uv)
        diff = anchor - r
        lhat = diff / np.linalg.norm(diff)
        drq  = attach_jacobian(q, link, uv)
        print(f"\n{cname} (link {link}): anchor {fmt(anchor)}, attach_uv {fmt(uv)}")
        print(f"  r_j = {fmt(r)}  |cable| = {np.linalg.norm(diff):.4f} m")
        print(f"  l_hat = {fmt(lhat)}")
        print(f"  A col = (dr/dq)^T l_hat = {fmt(drq.T @ lhat)}")

    Kq = joint_stiffness(q)
    KX = task_stiffness(q)
    print(f"\nA =\n{fmt(A)}")
    print(f"K_m = diag{tuple(np.diag(K_MUSCLE))} N/m")
    print(f"K_q = A K_m A^T =\n{fmt(Kq)}  N·m/rad")
    print(f"\nJ^{{-1}} =\n{fmt(np.linalg.inv(jacobian(q)))}")
    print(f"K_X = J^{{-T}} K_q J^{{-1}} =\n{fmt(KX)}  N/m")


# =============================================================================
# 7.  PLOTTING HELPERS
# =============================================================================

def draw_arm(ax, q):
    """Draw the two-link arm and cable lines."""
    O = np.array([0.0, 0.0])
    E = pos_elbow(q)
    H = pos_hand(q)
    ax.plot([O[0], E[0]], [O[1], E[1]], '-',  color='#2463eb', lw=4,
            solid_capstyle='round', zorder=3)
    ax.plot([E[0], H[0]], [E[1], H[1]], '-',  color='#f08c00', lw=4,
            solid_capstyle='round', zorder=3)
    ax.plot([O[0], E[0], H[0]], [O[1], E[1], H[1]], 'ko', ms=5, zorder=4)
    for _, link, anchor, uv in CABLES:
        r = attach_world(q, link, uv)
        ax.plot([anchor[0], r[0]], [anchor[1], r[1]], '-', color='0.55', lw=0.8, zorder=1)
        ax.plot(*anchor, 's', color='0.3', ms=3, zorder=2)


# =============================================================================
# 8.  MAIN
# =============================================================================

def main():
    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(exist_ok=True)

    kinds   = ["velocity", "force", "stiffness"]
    TITLES  = {"velocity" : "Velocity Manipulability",
               "force"    : "Force Manipulability",
               "stiffness": "Task-Space Stiffness"}
    COLORS  = {"velocity" : "#1f77b4",
               "force"    : "#d62728",
               "stiffness": "#2ca02c"}
    UNITS   = {"velocity" : "m/s per (rad/s)",
               "force"    : "N per (N·m)",
               "stiffness": "N/m (resistance)"}

    # -------------------------------------------------------------------------
    # 8.1  Compute everything
    # -------------------------------------------------------------------------
    results = []
    for cfg_name, t1_deg, t2_deg in CONFIGS:
        q = np.radians([t1_deg, t2_deg])
        J   = jacobian(q)
        det = np.linalg.det(J)
        if abs(det) < 1e-6:
            print(f"WARNING: '{cfg_name}' is near-singular (det J = {det:.2e}). "
                  "Consider adjusting the configuration.")
        JJT = J @ J.T
        KX  = task_stiffness(q)
        results.append({
            "name"    : cfg_name,
            "q"       : q,
            "J"       : J,
            "A"       : structure_matrix(q),
            "K_q"     : joint_stiffness(q),
            "K_X"     : KX,
            "checks"  : fd_checks(q),
            "ellipses": {
                "velocity" : ellipse_axes(JJT, "velocity"),
                "force"    : ellipse_axes(JJT, "force"),
                "stiffness": ellipse_axes(KX,  "stiffness"),
            },
        })

    # -------------------------------------------------------------------------
    # 8.2  Console report
    # -------------------------------------------------------------------------
    print("="*72)
    print(f"Activity 5 - Limb ability in task space")
    print(f"L1 = {L1} m,  L2 = {L2} m,  outer workspace radius = {L1+L2:.2f} m")
    print(f"Cable anchors at r = {R_ANCHOR} m (outside donut), 120 deg apart")
    print(f"K_m = diag{tuple(np.diag(K_MUSCLE))} N/m")
    print("="*72)

    for i, r in enumerate(results, 1):
        t1, t2 = np.degrees(r["q"])
        h = pos_hand(r["q"])
        print(f"\n[{i}] {r['name']}")
        print(f"    theta1={t1:.1f} deg, theta2={t2:.1f} deg | "
              f"hand=({h[0]:.3f}, {h[1]:.3f}) m")
        for k in kinds:
            semi, vec = r["ellipses"][k]
            ratio = semi[0]/semi[1] if semi[1] > 1e-12 else float('inf')
            print(f"    {k:10s}: major={semi[0]:.4g}, minor={semi[1]:.4g} "
                  f"| ratio={ratio:.2f}, major axis @ {axis_angle_deg(vec[:,0]):.1f} deg")
        c = r["checks"]
        print(f"    FD checks: J={c['J']:.1e}m/rad, A={c['A']:.1e}m, "
              f"K_q={c['K_q']:.1e}(rel), K_X={c['K_X']:.1e}(rel)")

    worst = max(max(r["checks"].values()) for r in results)
    status = "PASSED" if worst < 1e-4 else "FAILED"
    print(f"\nAll FD checks {status} (worst = {worst:.1e})")

    # Full hand-verification for configuration 1 (Boxing)
    hand_calculation_report(results[0]["q"], results[0]["name"])

    # -------------------------------------------------------------------------
    # 8.3  Shared scale per ellipse type so sizes are comparable across postures
    # -------------------------------------------------------------------------
    TARGET = 0.12   # desired largest semi-axis in metres on the plot
    scales = {k: TARGET / max(r["ellipses"][k][0][0] for r in results)
              for k in kinds}

    lim = (L1 + L2) * 1.3

    # -------------------------------------------------------------------------
    # 8.4  Combined grid figure (3 rows x 5 cols)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 5, figsize=(22, 13))
    fig.suptitle(
        "Activity 5 – Velocity, Force and Stiffness Ellipses\n"
        "(shared scale per row; shoulder at origin)",
        fontsize=14, fontweight='bold')

    for col, r in enumerate(results):
        for row, k in enumerate(kinds):
            ax = axes[row, col]
            draw_arm(ax, r["q"])
            semi, vec = r["ellipses"][k]
            pts = ellipse_curve(pos_hand(r["q"]), semi, vec, scales[k])
            ax.fill(pts[0], pts[1], color=COLORS[k], alpha=0.25)
            ax.plot(pts[0], pts[1], color=COLORS[k], lw=1.8)
            # Major axis line
            ctr = pos_hand(r["q"])
            mv  = scales[k] * semi[0] * vec[:, 0]
            ax.plot([ctr[0]-mv[0], ctr[0]+mv[0]],
                    [ctr[1]-mv[1], ctr[1]+mv[1]],
                    '--', color=COLORS[k], lw=1.0)
            ax.set_xlim(-lim*0.4, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.grid(alpha=0.25)
            ax.axhline(0, color='0.75', lw=0.5)
            ax.axvline(0, color='0.75', lw=0.5)
            t1, t2 = np.degrees(r["q"])
            ratio = semi[0]/semi[1] if semi[1]>1e-12 else np.inf
            if row == 0:
                ax.set_title(
                    f"[{col+1}] {r['name']}\n"
                    f"θ₁={t1:.0f}°, θ₂={t2:.0f}°",
                    fontsize=9, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f"{TITLES[k]}\ny [m]", fontsize=8)
            if row == 2:
                ax.set_xlabel("x [m]", fontsize=8)
            ax.text(0.02, 0.02,
                    f"ratio {ratio:.2f}\naxis {axis_angle_deg(vec[:,0]):.0f}°",
                    transform=ax.transAxes, fontsize=7, va='bottom',
                    color=COLORS[k])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    grid_path = out_dir / "all_ellipses.png"
    fig.savefig(grid_path, dpi=150, bbox_inches='tight')
    print(f"\nGrid figure saved: {grid_path}")

    # -------------------------------------------------------------------------
    # 8.5  Individual figure per posture
    # -------------------------------------------------------------------------
    for i, r in enumerate(results, 1):
        f2, axs = plt.subplots(1, 3, figsize=(15, 5))
        t1, t2 = np.degrees(r["q"])
        f2.suptitle(f"[{i}] {r['name']}   (θ₁={t1:.0f}°, θ₂={t2:.0f}°)",
                    fontsize=12, fontweight='bold')
        for ax, k in zip(axs, kinds):
            draw_arm(ax, r["q"])
            semi, vec = r["ellipses"][k]
            pts = ellipse_curve(pos_hand(r["q"]), semi, vec, scales[k])
            ax.fill(pts[0], pts[1], color=COLORS[k], alpha=0.25)
            ax.plot(pts[0], pts[1], color=COLORS[k], lw=2.0)
            ratio = semi[0]/semi[1] if semi[1]>1e-12 else np.inf
            ax.set_title(
                f"{TITLES[k]}\n"
                f"major={semi[0]:.3g}, minor={semi[1]:.3g} {UNITS[k]}\n"
                f"ratio={ratio:.2f}, axis @ {axis_angle_deg(vec[:,0]):.1f}°",
                fontsize=9)
            ax.set_xlim(-lim*0.4, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.grid(alpha=0.25)
            ax.set_xlabel("x [m]")
        axs[0].set_ylabel("y [m]")
        f2.tight_layout()
        indiv_path = out_dir / f"config_{i}.png"
        f2.savefig(indiv_path, dpi=150, bbox_inches='tight')

    print(f"Individual figures saved to {out_dir}")
    plt.show()


if __name__ == "__main__":
    main()
