# ES 656: Human-Robot Interaction — Activities 4 & 5
## Workspace Analysis and Manipulability of a Cable-Actuated Human Upper Limb

**IIT Gandhinagar | 2026**

**Team:**
- Kalyan Baidya — 25210065
- Kathroju Saikrishna — 25210068

---

## Overview

This repository contains the simulation code and output for Activities 4 and 5 of ES 656. The goal is to model a planar two-link cable-actuated mechanism (representing the human upper limb) and:
- **Activity 4:** Analyse its workspace and cable-tension feasibility across different cable configurations and arm dimensions.
- **Activity 5:** Analyse velocity manipulability, force manipulability, and task-space stiffness for five real-world postures, with ellipse overlays on photographs.

Two arm models are studied throughout:
- **Original sketch** — $L_1 = 16.5$ cm, $L_2 = 14.5$ cm (physical mechanism from Activity 2)
- **Human reference** — $L_1 = 28$ cm, $L_2 = 26$ cm (measured from Kathroju Saikrishna)

---

---

# Activity 4: Workspace Analysis

## Files

| File | Description |
|------|-------------|
| `workspace_analysis.py` | Main simulation script (all steps) |
| `run_output.txt` | Full console output from the simulation run |
| `workspace_results.txt` | Numerical workspace extents and feasibility summary |

---

## Code Structure (`workspace_analysis.py`)

The script is organised into sequential steps matching the assignment:

### Step 1 — Model parameters and forward kinematics
- Two arm models defined (`MODELS` dict): `original_sketch` and `human_reference`
- Joint limits: $\theta_1 \in [-140°, 90°]$, $\theta_2 \in [0°, 140°]$
- `forward_kinematics(theta1, theta2, L1, L2)` — computes wrist endpoint $[x, y]$
- `within_joint_limits(theta1, theta2)` — checks admissibility

### Step 2 — Cable geometry
- 4 cables defined (`CABLES` dict): T1, T2, T3, T4
- Each cable has a fixed world-frame anchor and a local attachment coordinate on either link 1 or link 2
- `attachment_position(theta1, theta2, L1, local, link)` — returns world-frame attachment point
- `attachment_jacobian(theta1, theta2, L1, local, link)` — returns $\partial r_j / \partial \mathbf{q} \in \mathbb{R}^{2 \times 2}$

### Step 3 — Structure matrix
- `structure_matrix(theta1, theta2, L1, cables)` — assembles the full $A \in \mathbb{R}^{2 \times m}$ matrix
- Each column: $A_{:j} = J_j^T \hat{\ell}_j$, where $\hat{\ell}_j$ is the unit vector from attachment to anchor
- Validated against finite-difference Jacobian; max error $< 3 \times 10^{-9}$ cm/rad

### Step 4A — Torque-direction feasibility at reference posture
- Tests 360 uniformly sampled torque directions at $\theta_1 = \theta_2 = 0$
- For each direction, solves a linear programme:
  - Maximise slack $\delta$ subject to $A\mathbf{T} = \boldsymbol{\tau}(\alpha)$, $T_j \geq \delta \geq 0$
  - Uses `scipy.optimize.linprog`
- Reports fraction of directions feasible for all 5 configurations (4 subsets + full 4-cable)

### Step 4B — Posture-dependent feasibility
- Sweeps a $9 \times 9$ grid of joint angles over the full admissible range
- 36 torque directions tested at each of the 81 postures
- Reports mean / min / max feasibility and count of fully-feasible postures per configuration

### Step 5 — Workspace sampling and plotting
- Samples $231 \times 141$ joint-angle grid → 32,571 endpoint positions per model
- `planar_workspace_comparison.png` — side-by-side scatter of all reachable endpoints
- `workspace_feasibility_heatmaps.png` — colour-coded feasibility % over $(\theta_1, \theta_2)$ grid for 3-cable and 4-cable cases, both models
- `cable_feasible_workspace.png` — endpoint positions where all 36 directions pass the LP, overlaid for 3-cable (T1–T2–T4) and 4-cable (T1–T2–T3–T4)

---

## How to Run

```bash
python3 workspace_analysis.py
```

Requires: `numpy`, `scipy`, `matplotlib`

Output figures are saved to the working directory. Console output is captured in `run_output.txt`.

---

## Key Results

| Configuration | Original sketch (mean feasibility) | Human reference (mean feasibility) |
|---|---|---|
| T1, T2, T3 | 38.51% | 39.61% |
| **T1, T2, T4** | **60.36%** | **51.75%** |
| T1, T3, T4 | 48.15% | 42.63% |
| T2, T3, T4 | 45.13% | 44.55% |
| T1, T2, T3, T4 | 66.77% | 55.21% |

T1–T2–T4 is the best three-cable subset; the four-cable configuration outperforms all subsets. Feasibility drops for the human-reference model because longer links reduce the elbow moment arms of T1 and T4.

---

---

# Activity 5: Manipulability and Task-Space Stiffness Analysis

## System Parameters

- **Human-reference arm:** $L_1 = 28$ cm (humerus), $L_2 = 26$ cm (forearm)
- **DOF:** $n = 2$, **Actuators:** $m = 3$ (redundant actuation)
- **Cable anchors:** Fixed in world frame at radius $R = 0.75$ m from shoulder origin, at 90°, 210°, 330° (120° equal spacing). $R > L_1 + L_2 = 0.54$ m ensures anchors lie outside the entire workspace, guaranteeing cable tension throughout.
- **Cable attachments:**
  - Cable 1: Link 1 at 55% of $L_1$, 0.02 m lateral offset
  - Cable 2: Link 2 at 35% of $L_2$, 0.02 m lateral offset
  - Cable 3: Link 2 at 65% of $L_2$, 0.02 m lateral offset
- **Spring stiffness:** $K_m = \mathrm{diag}(1200,\ 1500,\ 1000)$ N/m

## Five Postures

| # | Activity | Plane | $\theta_1$ | $\theta_2$ |
|---|----------|-------|-----------|-----------|
| 1 | Punching / Boxing | Sagittal | −10° | 55° |
| 2 | Carrom | Transverse | 0° | 39° |
| 3 | Chess | Transverse | −22° | 56° |
| 4 | Dart throwing | Sagittal | −15° | 65° |
| 5 | Defense / Guard | Sagittal | −52° | 140° |

Joint angles measured from colour-coded photographs of a team member (red dot = shoulder, green dot = elbow, blue dot = end-effector).

---

## Files

| File | Description |
|------|-------------|
| `activity5_ellipses.py` | Standalone ellipse plots (no photo background). Computes and plots velocity, force, and task-space stiffness ellipses for all 5 postures on clean matplotlib axes. |
| `posture_overlay2.py` | **Main deliverable.** Self-contained overlay script. Reads colour-coded photos from `Activity_Pictures/pictures_Color_coded/`, computes ellipses, and draws them centred on the blue (end-effector) dot. Outputs individual figures and a combined 5×3 summary grid. |
| `figures/overlay2_punching.png` | Overlay figure — Punching/Boxing posture (velocity, force, stiffness ellipses) |
| `figures/overlay2_carrom.png` | Overlay figure — Carrom posture |
| `figures/overlay2_chess.png` | Overlay figure — Chess posture |
| `figures/overlay2_dart.png` | Overlay figure — Dart throwing posture |
| `figures/overlay2_defense.png` | Overlay figure — Defense/Guard posture |
| `figures/all_overlays2.png` | Combined 5×3 grid: all postures × all ellipse types |
| `Activity_Pictures/pictures_Color_coded/*.JPG` | Colour-coded posture photographs (1280×720) |

---

## Code Structure

### `activity5_ellipses.py` — Traditional Standalone Plots

| Function | Description |
|----------|-------------|
| `jacobian(q)` | Returns $J \in \mathbb{R}^{2\times2}$: standard planar two-link Jacobian |
| `structure_matrix(q)` | Returns $A \in \mathbb{R}^{2\times3}$: maps cable tensions to joint torques via moment arms |
| `joint_stiffness(q)` | Returns $K_q = A K_m A^T \in \mathbb{R}^{2\times2}$ |
| `task_stiffness(q)` | Returns $K_x = J^{-T} K_q J^{-1} \in \mathbb{R}^{2\times2}$ |
| `ellipse_axes(M, kind)` | Eigendecomposition of $M$; returns semi-axes and eigenvectors for velocity / force / stiffness ellipses |
| `plot_ellipses(...)` | Draws all three ellipses for a posture on matplotlib axes |

### `posture_overlay2.py` — Photo Overlay (Main Script)

| Function | Description |
|----------|-------------|
| `jacobian(q)` | Same as above (self-contained, no imports from other scripts) |
| `_attach_point(q, link, frac, offset)` | World-frame cable attachment point on a given link |
| `structure_matrix(q)` | Assembles $A$ column by column using moment arm cross-products |
| `joint_stiffness(q)` | $K_q = A K_m A^T$ |
| `task_stiffness(q)` | $K_x = J^{-T} K_q J^{-1}$ |
| `ellipse_axes(M, kind)` | Semi-axes via eigendecomposition; velocity: $\sqrt{\lambda_i}$; force/stiffness: $1/\sqrt{\lambda_i}$ |
| `ellipse_pts_px(center_px, semi, vec)` | Converts ellipse from model frame (y-up) to pixel coordinates (y-down) by flipping y-components of eigenvectors; scales major axis to `ELLIPSE_PX = 150` pixels |
| `draw_overlay(ax, img, p, semi, vec, color)` | Draws skeleton + coloured ellipse on a photo axis |
| `process(p)` | Runs all mechanics for one posture; returns ellipse data |
| `main()` | Generates all individual figures and the combined 5×3 grid |

---

## Key Mathematical Relations

$$\dot{\mathbf{x}} = J\dot{\mathbf{q}}, \qquad \boldsymbol{\tau} = A\mathbf{T}, \qquad K_q = AK_mA^T, \qquad K_x = J^{-T}K_qJ^{-1}$$

| Ellipse | Governing equation | Semi-axes |
|---------|-------------------|-----------|
| Velocity manipulability | $\dot{\mathbf{x}}^T (JJ^T)^{-1} \dot{\mathbf{x}} = 1$ | $\sqrt{\lambda_i}$ of $JJ^T$ |
| Force manipulability | $\mathbf{f}^T (JJ^T) \mathbf{f} = 1$ | $1/\sqrt{\lambda_i}$ of $JJ^T$ |
| Task-space stiffness | $\delta\mathbf{F}^T K_x^{-1} \delta\mathbf{F} = 1$ | $1/\sqrt{\lambda_i}$ of $K_x$ |

Assumptions: $\mathbf{T} = \mathbf{0}$ (zero nominal tension → geometric stiffness term vanishes), $\mathbf{F} = \mathbf{0}$ (no external force).

---

## How to Run

```bash
# Standalone ellipse plots (no photos needed)
python3 activity5_ellipses.py

# Photo overlays (requires Activity_Pictures/pictures_Color_coded/*.JPG)
python3 posture_overlay2.py
```

Requires: `numpy`, `matplotlib`

Output figures are saved to `figures/`.

---
