"""
ES 656 Human-Robot Interaction - Activity 5
Overlay of velocity, force and stiffness ellipses on the NEW colour-coded
posture photographs (pictures_Color_coded folder).

Colour coding in the photos:
  Red  dot  = shoulder joint  (origin of arm model)
  Green dot = elbow joint
  Blue  dot = end-effector / wrist  ← ellipses are centred here

Strategy:
  1. Red and green pixel positions → pixel-per-metre scale (distance = L1 = 0.28 m).
  2. Blue pixel position          → ellipse centre on the image.
  3. All mechanics (J, A, Kx) computed from the joint angles in the model frame.
  4. Ellipse axes are in model space (x-right, y-up). The image has y-down, so
     the y-components of the eigenvectors are sign-flipped before drawing.

Run:  python posture_overlay2.py
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ── model parameters ──────────────────────────────────────────────────────────
L1 = 0.28   # humerus  [m]
L2 = 0.26   # forearm  [m]

# ── cable / stiffness parameters (same as activity5_ellipses.py) ──────────────
R_ANCHOR = 0.75
_ang = np.radians([90.0, 210.0, 330.0])
ANCHOR_WORLD = [R_ANCHOR * np.array([np.cos(a), np.sin(a)]) for a in _ang]
ATTACH_FRAC   = [0.55, 0.35, 0.65]
ATTACH_OFFSET = [0.02, 0.02, 0.02]
ATTACH_LINKS  = [1,    2,    2   ]
K_MUSCLE = np.diag([1200.0, 1500.0, 1000.0])

# ── paths ──────────────────────────────────────────────────────────────────────
HERE    = Path(__file__).resolve().parent
IMG_DIR = HERE / "Activity_Pictures" / "pictures_Color_coded"
OUT_DIR = HERE / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ── posture definitions ────────────────────────────────────────────────────────
# pixel coords read directly from the colour-coded photos
# (image origin = top-left, y increases downward)
POSTURES = [
    {
        "stem"       : "punching",
        "title"      : "Punching/Boxing (sagittal)",
        "theta1"     : -10.0,
        "theta2"     :  55.0,
        "shoulder_px": np.array([313.0, 538.0]),
        "elbow_px"   : np.array([765.0, 614.0]),
        "hand_px"    : np.array([1060.0, 330.0]),
    },
    {
        "stem"       : "carrom",
        "title"      : "Carrom (transverse)",
        "theta1"     :   0.0,
        "theta2"     :  39.0,
        "shoulder_px": np.array([554.0, 138.0]),
        "elbow_px"   : np.array([554.0, 356.0]),
        "hand_px"    : np.array([675.0, 509.0]),
    },
    {
        "stem"       : "chess",
        "title"      : "Chess (transverse)",
        "theta1"     : -22.0,
        "theta2"     :  56.0,
        "shoulder_px": np.array([522.0, 165.0]),
        "elbow_px"   : np.array([422.0, 413.0]),
        "hand_px"    : np.array([536.0, 580.0]),
    },
    {
        "stem"       : "dart",
        "title"      : "Dart (sagittal)",
        "theta1"     : -15.0,
        "theta2"     :  65.0,
        "shoulder_px": np.array([178.0, 400.0]),
        "elbow_px"   : np.array([475.0, 477.0]),
        "hand_px"    : np.array([627.0, 302.0]),
    },
    {
        "stem"       : "defense",
        "title"      : "Defense (sagittal)",
        "theta1"     : -52.0,
        "theta2"     : 140.0,
        "shoulder_px": np.array([553.0, 258.0]),
        "elbow_px"   : np.array([725.0, 478.0]),
        "hand_px"    : np.array([718.0, 258.0]),
    },
]

# Ellipse display: major axis drawn as this many pixels on the photo
ELLIPSE_PX = 150

KINDS  = ["velocity",   "force",      "stiffness"]
LABELS = ["Velocity\nManipulability",
          "Force\nManipulability",
          "Task-Space\nStiffness"]
COLORS = ["#1f9bff", "#ff3b3b", "#22d32b"]
UNITS  = ["m/s per rad/s", "N per N·m", "N/m"]


# ═══════════════════════════════════════════════════════════════════════════════
# Arm mechanics
# ═══════════════════════════════════════════════════════════════════════════════

def jacobian(q):
    s1,  c1  = np.sin(q[0]), np.cos(q[0])
    s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
    return np.array([
        [-L1*s1 - L2*s12, -L2*s12],
        [ L1*c1 + L2*c12,  L2*c12],
    ])


def _attach_point(q, link, frac, offset):
    """World-frame attachment point of one cable on a link."""
    if link == 1:
        base_angle = q[0]
        base_pos   = np.zeros(2)
        along = np.array([np.cos(base_angle), np.sin(base_angle)])
        perp  = np.array([-np.sin(base_angle), np.cos(base_angle)])
        return base_pos + frac * L1 * along + offset * perp
    else:
        elbow_pos = np.array([L1*np.cos(q[0]), L1*np.sin(q[0])])
        link_angle = q[0] + q[1]
        along = np.array([np.cos(link_angle), np.sin(link_angle)])
        perp  = np.array([-np.sin(link_angle), np.cos(link_angle)])
        return elbow_pos + frac * L2 * along + offset * perp


def structure_matrix(q):
    """A such that τ = A T  (2×3)."""
    A = np.zeros((2, 3))
    for i, (anchor, frac, offset, link) in enumerate(
            zip(ANCHOR_WORLD, ATTACH_FRAC, ATTACH_OFFSET, ATTACH_LINKS)):
        p = _attach_point(q, link, frac, offset)
        cable_vec = anchor - p
        cable_len = np.linalg.norm(cable_vec)
        if cable_len < 1e-9:
            continue
        u = cable_vec / cable_len          # unit vector along cable
        # Moment arm about each joint
        if link == 1:
            r1 = p - np.zeros(2)
            tau1 = r1[0]*u[1] - r1[1]*u[0]
            A[0, i] = tau1
        else:
            r1 = p - np.zeros(2)
            elbow = np.array([L1*np.cos(q[0]), L1*np.sin(q[0])])
            r2 = p - elbow
            tau1 = r1[0]*u[1] - r1[1]*u[0]
            tau2 = r2[0]*u[1] - r2[1]*u[0]
            A[0, i] = tau1
            A[1, i] = tau2
    return A


def joint_stiffness(q):
    A  = structure_matrix(q)
    return A @ K_MUSCLE @ A.T


def task_stiffness(q):
    J    = jacobian(q)
    Jinv = np.linalg.inv(J)
    Kq   = joint_stiffness(q)
    return Jinv.T @ Kq @ Jinv


def ellipse_axes(M, kind):
    lam, vec = np.linalg.eigh(M)
    lam = np.maximum(lam, 1e-15)
    if kind == "velocity":
        semi = np.sqrt(lam)
    else:                          # force and stiffness both: 1/sqrt(λ)
        semi = 1.0 / np.sqrt(lam)
    order = np.argsort(semi)[::-1]
    return semi[order], vec[:, order]


def axis_angle_deg(v):
    return np.degrees(np.arctan2(v[1], v[0]))


# ═══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def ellipse_pts_px(center_px, semi, vec, n=300):
    """
    Ellipse boundary in pixel coordinates.
    Model: y-up.  Image: y-down  →  flip the y-component of eigenvectors.
    Scale the ellipse so the major axis spans ELLIPSE_PX pixels.
    """
    t   = np.linspace(0, 2*np.pi, n)
    # ellipse in model frame (unit semi-axes)
    pts = vec @ (semi[:, None] * np.vstack([np.cos(t), np.sin(t)]))
    k   = ELLIPSE_PX / semi[0]        # pixels per unit
    # flip y for image convention
    px  = center_px[:, None] + k * np.vstack([pts[0], -pts[1]])
    return px   # (2, n)


def major_axis_endpoints(center_px, semi, vec):
    k  = ELLIPSE_PX / semi[0]
    mv = k * np.array([vec[0, 0], -vec[1, 0]])
    return center_px - mv, center_px + mv


def draw_overlay(ax, img, p, semi, vec, color):
    """Draw arm skeleton + one ellipse on ax."""
    ax.imshow(img)

    sh = p["shoulder_px"]
    el = p["elbow_px"]
    hand = p["hand_px"]

    # Arm skeleton (using the measured colour-coded dots)
    ax.plot([sh[0], el[0], hand[0]],
            [sh[1], el[1], hand[1]],
            '-', color='white', lw=2.5, alpha=0.85, zorder=3)

    # Joint dots (matching the photo colours)
    ax.plot(*sh,   'o', color='#ff4444', ms=10, mec='white', mew=1.5, zorder=5)
    ax.plot(*el,   'o', color='#44ff44', ms=10, mec='white', mew=1.5, zorder=5)
    ax.plot(*hand, 'o', color='#44bbff', ms=10, mec='white', mew=1.5, zorder=5)

    # Ellipse filled + outline
    pts = ellipse_pts_px(hand, semi, vec)
    ax.fill(pts[0], pts[1], color=color, alpha=0.35, zorder=4)
    ax.plot(pts[0], pts[1], color=color, lw=2.5, zorder=4)

    # Major-axis dashed line
    p0, p1 = major_axis_endpoints(hand, semi, vec)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], '--', color=color, lw=1.5, zorder=4)

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis('off')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def process(p):
    q   = np.radians([p["theta1"], p["theta2"]])
    J   = jacobian(q)
    KX  = task_stiffness(q)
    JJT = J @ J.T
    return {
        **p,
        "q": q,
        "ellipses": {
            "velocity" : ellipse_axes(JJT, "velocity"),
            "force"    : ellipse_axes(JJT, "force"),
            "stiffness": ellipse_axes(KX,  "stiffness"),
        },
    }


def main():
    all_res = [process(p) for p in POSTURES]

    # ── individual figure per posture (1 row × 3 cols) ────────────────────────
    for res in all_res:
        t1, t2 = np.degrees(res["q"])
        img = plt.imread(str(IMG_DIR / (res["stem"] + ".JPG")))

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor('#0d0d0d')
        fig.suptitle(
            f"{res['title']}   θ₁ = {t1:.0f}°,  θ₂ = {t2:.0f}°",
            fontsize=12, color='white', fontweight='bold')

        for ax, kind, lbl, color, unit in zip(axes, KINDS, LABELS, COLORS, UNITS):
            semi, vec = res["ellipses"][kind]
            draw_overlay(ax, img, res, semi, vec, color)
            ratio = semi[0] / semi[1] if semi[1] > 1e-12 else np.inf
            ang   = axis_angle_deg(vec[:, 0])
            ax.set_title(
                f"{lbl}\nmajor={semi[0]:.3g}, minor={semi[1]:.3g} {unit}\n"
                f"ratio={ratio:.2f},  axis @ {ang:.1f}°",
                fontsize=9, color='white', pad=4)

        fig.tight_layout(rect=[0, 0, 1, 0.90])
        out = OUT_DIR / f"overlay2_{res['stem']}.png"
        fig.savefig(out, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved: {out}")

    # ── combined 5×3 summary ──────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(5, 3, figsize=(18, 30))
    fig2.patch.set_facecolor('#0d0d0d')
    fig2.suptitle(
        "Activity 5 – Ellipse overlays on colour-coded posture photos\n"
        "Rows: postures | Columns: velocity / force / stiffness",
        fontsize=13, color='white', fontweight='bold')

    for row, res in enumerate(all_res):
        t1, t2 = np.degrees(res["q"])
        img = plt.imread(str(IMG_DIR / (res["stem"] + ".JPG")))
        for col, (kind, lbl, color, unit) in enumerate(
                zip(KINDS, LABELS, COLORS, UNITS)):
            ax = axes2[row, col]
            semi, vec = res["ellipses"][kind]
            draw_overlay(ax, img, res, semi, vec, color)
            ratio = semi[0] / semi[1] if semi[1] > 1e-12 else np.inf
            ang   = axis_angle_deg(vec[:, 0])
            if col == 0:
                ax.text(-0.02, 0.5,
                        f"[{row+1}] {res['title']}\nθ₁={t1:.0f}°, θ₂={t2:.0f}°",
                        transform=ax.transAxes, fontsize=8, color='white',
                        va='center', ha='right', rotation=90)
            if row == 0:
                ax.set_title(lbl, fontsize=10, color=color, fontweight='bold')
            ax.text(0.01, 0.01,
                    f"ratio {ratio:.2f}  axis {ang:.0f}°",
                    transform=ax.transAxes, fontsize=7.5, color=color,
                    va='bottom', bbox=dict(fc='black', alpha=0.5, pad=2))

    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    out2 = OUT_DIR / "all_overlays2.png"
    fig2.savefig(out2, dpi=130, bbox_inches='tight',
                 facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"Saved combined summary: {out2}")

    # ── console summary ───────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'Posture':<28} {'Kind':<12} {'Major':>8} {'Minor':>8} "
          f"{'Ratio':>6} {'Axis°':>7}")
    print("="*70)
    for res in all_res:
        for kind in KINDS:
            semi, vec = res["ellipses"][kind]
            ratio = semi[0]/semi[1] if semi[1] > 1e-12 else np.inf
            print(f"{res['title']:<28} {kind:<12} {semi[0]:>8.4g} "
                  f"{semi[1]:>8.4g} {ratio:>6.2f} "
                  f"{axis_angle_deg(vec[:,0]):>7.1f}")
        print("-"*70)


if __name__ == "__main__":
    main()
