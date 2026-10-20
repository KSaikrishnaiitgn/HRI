# ES 656: Human-Robot Interaction — Activity 4
## Workspace Analysis of a Cable-Actuated Human Upper Limb

**IIT Gandhinagar | 2026**

**Team:**
- Kalyan Baidya — 25210065
- Kathroju Saikrishna — 25210068

---

## Overview

This repository contains the simulation code and output for Activity 4 of ES 656. The goal is to model a planar two-link cable-actuated mechanism (representing the human upper limb) and analyse its workspace and cable-tension feasibility across different cable configurations and arm dimensions.

Two arm models are studied:
- **Original sketch** — $L_1 = 16.5$ cm, $L_2 = 14.5$ cm (physical mechanism from Activity 2)
- **Human reference** — $L_1 = 28$ cm, $L_2 = 26$ cm (measured from Kathroju Saikrishna)

---

## Files

| File | Description |
|------|-------------|
| `workspace_analysis.py` | Main simulation script (all steps) |
| `run_output.txt` | Full console output from the simulation run |
| `workspace_results.txt` | Numerical workspace extents and feasibility summary |
| `planar_workspace_comparison.png` | Geometric workspace: original sketch vs human reference |
| `workspace_feasibility_heatmaps.png` | Posture-dependent feasibility heatmaps over joint-angle grid |
| `cable_feasible_workspace.png` | Cable-feasible workspace: 3-cable vs 4-cable overlay |
| `mechanism_photo.png` | Photo of the physical mechanism from Activity 2 |

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
