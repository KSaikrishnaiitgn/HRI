
"""
ES 656 Activity 4
Step 1: Model parameters, joint limits, and forward kinematics.

Units:
    Link lengths and endpoint coordinates: cm
    Joint angles: radians internally
"""

import numpy as np
from scipy.optimize import linprog
# --------------------------------------------------
# 1. MODEL PARAMETERS
# --------------------------------------------------

MODELS = {
    "original_sketch": {
        "L1": 16.5,  # Upper arm (cm)
        "L2": 14.5,  # Forearm (cm)
    },
    "human_reference": {
        "L1": 28.0,   # Humerus reference length (cm)
        "L2": 26.0,   # Forearm reference length (cm)
    },
}

# Joint limits, specified in degrees and converted to radians.
THETA1_LIMITS_DEG = (-140.0, 90.0)
THETA2_LIMITS_DEG = (0.0, 140.0)

THETA1_LIMITS = np.deg2rad(THETA1_LIMITS_DEG)
THETA2_LIMITS = np.deg2rad(THETA2_LIMITS_DEG)


# --------------------------------------------------
# 2. JOINT-LIMIT CHECK
# --------------------------------------------------

def within_joint_limits(theta1, theta2):
    """
    Check whether both joint angles satisfy their limits.

    Parameters
    ----------
    theta1, theta2 : float
        Shoulder and relative elbow angles (radians).

    Returns
    -------
    bool
        True if both angles lie within their limits.
    """
    shoulder_ok = (
        THETA1_LIMITS[0] <= theta1 <= THETA1_LIMITS[1]
    )

    elbow_ok = (
        THETA2_LIMITS[0] <= theta2 <= THETA2_LIMITS[1]
    )

    return shoulder_ok and elbow_ok


# --------------------------------------------------
# 3. FORWARD KINEMATICS
# --------------------------------------------------

def forward_kinematics(theta1, theta2, L1, L2):
    """
    Calculate the wrist/hand reference point B.

    theta1: shoulder angle relative to the positive x-axis
    theta2: elbow angle relative to the upper arm
    L1, L2: link lengths in cm

    Returns
    -------
    numpy.ndarray, shape (2,)
        Endpoint coordinates [x_B, y_B] in cm.
    """

    # Absolute orientation of the forearm.
    theta_forearm = theta1 + theta2

    x_B = (
        L1 * np.cos(theta1)
        + L2 * np.cos(theta_forearm)
    )

    y_B = (
        L1 * np.sin(theta1)
        + L2 * np.sin(theta_forearm)
    )

    return np.array([x_B, y_B])


# --------------------------------------------------
# 4. BASIC VALIDATION TESTS
# --------------------------------------------------

def run_step1_tests():
    """Verify the reference configurations."""

    q0 = (0.0, 0.0)

    print("=" * 55)
    print("STEP 1: MODEL AND FORWARD-KINEMATICS VALIDATION")
    print("=" * 55)

    for name, model in MODELS.items():
        L1 = model["L1"]
        L2 = model["L2"]

        B = forward_kinematics(
            q0[0], q0[1], L1, L2
        )

        expected_reach = L1 + L2

        print(f"\nModel: {name}")
        print(f"L1 = {L1:.2f} cm")
        print(f"L2 = {L2:.2f} cm")
        print(f"Endpoint B = ({B[0]:.4f}, "
              f"{B[1]:.4f}) cm")
        print(f"Expected straight reach = "
              f"{expected_reach:.2f} cm")

        assert np.allclose(
            B,
            np.array([expected_reach, 0.0]),
            atol=1e-10,
        ), "Forward-kinematics reference test failed."

    # Check the joint-limit function.
    assert within_joint_limits(
        np.deg2rad(0), np.deg2rad(90)
    )

    assert not within_joint_limits(
        np.deg2rad(100), np.deg2rad(90)
    )

    print("\nAll Step 1 tests passed.")

# --------------------------------------------------
# 5. CABLE GEOMETRY
# --------------------------------------------------

# Fixed anchors in world coordinates (cm).
CABLES = {
    "T1": {
        "anchor": np.array([0.0, 14.0]),
        "attachment_local": (6.5, 1.0),
        "link": 2,
    },
    "T2": {
        "anchor": np.array([0.0, 5.5]),
        "attachment_local": (7.0, 2.0),
        "link": 1,
    },
    "T3": {
        "anchor": np.array([0.0, -6.5]),
        "attachment_local": (5.0, -2.0),
        "link": 1,
    },
    "T4": {
        "anchor": np.array([0.0, -16.0]),
        "attachment_local": (4.5, -1.0),
        "link": 2,
    },
}


# --------------------------------------------------
# 6. MOVING ATTACHMENT POSITION
# --------------------------------------------------

def attachment_position(theta1, theta2, L1, local, link):
    """
    Return the world-frame position of a cable attachment.

    Parameters
    ----------
    theta1, theta2 : float
        Joint angles in radians.
    L1 : float
        Upper-arm length in cm.
    local : tuple
        Attachment coordinates (a, b) in the link frame, cm.
    link : int
        1 for upper arm, 2 for forearm.

    Returns
    -------
    numpy.ndarray, shape (2,)
        World coordinates in cm.
    """
    a, b = local

    c1, s1 = np.cos(theta1), np.sin(theta1)

    if link == 1:
        return np.array([
            a * c1 - b * s1,
            a * s1 + b * c1,
        ])

    if link == 2:
        angle = theta1 + theta2
        c12, s12 = np.cos(angle), np.sin(angle)

        return np.array([
            L1 * c1 + a * c12 - b * s12,
            L1 * s1 + a * s12 + b * c12,
        ])

    raise ValueError("link must be 1 or 2")


# --------------------------------------------------
# 7. ANALYTIC ATTACHMENT JACOBIAN
# --------------------------------------------------

def attachment_jacobian(theta1, theta2, L1, local, link):
    """
    Return J = d(r)/d(q), with shape (2, 2).

    Rows correspond to x and y.
    Columns correspond to theta1 and theta2.

    Units: cm/rad.
    """
    a, b = local

    c1, s1 = np.cos(theta1), np.sin(theta1)

    if link == 1:
        return np.array([
            [-a * s1 - b * c1, 0.0],
            [ a * c1 - b * s1, 0.0],
        ])

    if link == 2:
        angle = theta1 + theta2
        c12, s12 = np.cos(angle), np.sin(angle)

        return np.array([
            [
                -L1 * s1 - a * s12 - b * c12,
                -a * s12 - b * c12,
            ],
            [
                L1 * c1 + a * c12 - b * s12,
                a * c12 - b * s12,
            ],
        ])

    raise ValueError("link must be 1 or 2")


# --------------------------------------------------
# 8. TENDON LENGTH AND UNIT VECTOR
# --------------------------------------------------

def tendon_geometry(theta1, theta2, L1, cable):
    """
    Calculate the moving point, tendon length, and unit vector.

    The unit vector points FROM the moving attachment
    TOWARD the fixed anchor.

    Returns a dictionary of geometric quantities.
    """
    s = cable["anchor"]

    r = attachment_position(
        theta1, theta2, L1,
        cable["attachment_local"],
        cable["link"],
    )

    J = attachment_jacobian(
        theta1, theta2, L1,
        cable["attachment_local"],
        cable["link"],
    )

    cable_vector = s - r
    length = np.linalg.norm(cable_vector)

    if not np.isfinite(length) or length <= 1e-12:
        raise ValueError(
            "Degenerate cable geometry: length is zero "
            "or numerically invalid."
        )

    unit_vector = cable_vector / length

    # d(length)/dq = -unit_vector.T @ J
    length_gradient = -unit_vector @ J

    return {
        "anchor": s.copy(),
        "attachment": r,
        "jacobian": J,
        "cable_vector": cable_vector,
        "length": length,
        "unit_vector": unit_vector,
        "length_gradient": length_gradient,
    }


# --------------------------------------------------
# 9. FINITE-DIFFERENCE VERIFICATION
# --------------------------------------------------

def finite_difference_attachment_jacobian(
    theta1, theta2, L1, local, link, h=1e-6
):
    """
    Central-difference approximation of the attachment Jacobian.

    h is in radians.
    """
    q = np.array([theta1, theta2], dtype=float)
    J_fd = np.zeros((2, 2))

    for k in range(2):
        dq = np.zeros(2)
        dq[k] = h

        r_plus = attachment_position(
            *(q + dq), L1, local, link
        )
        r_minus = attachment_position(
            *(q - dq), L1, local, link
        )

        J_fd[:, k] = (r_plus - r_minus) / (2.0 * h)

    return J_fd


def finite_difference_tendon_gradient(
    theta1, theta2, L1, cable, h=1e-6
):
    """
    Central-difference approximation of d(length)/dq.
    """
    q = np.array([theta1, theta2], dtype=float)
    gradient_fd = np.zeros(2)

    for k in range(2):
        dq = np.zeros(2)
        dq[k] = h

        length_plus = tendon_geometry(
            *(q + dq), L1, cable
        )["length"]

        length_minus = tendon_geometry(
            *(q - dq), L1, cable
        )["length"]

        gradient_fd[k] = (
            length_plus - length_minus
        ) / (2.0 * h)

    return gradient_fd


# --------------------------------------------------
# 10. STEP 2 VALIDATION
# --------------------------------------------------

def run_step2_tests():
    """
    Verify analytic Jacobians and tendon-length derivatives
    for both link geometries and several postures.
    """
    test_postures_deg = [
        (0.0, 0.0),
        (-30.0, 45.0),
        (40.0, 80.0),
        (-100.0, 120.0),
    ]

    tolerance = 1e-6
    max_jacobian_error = 0.0
    max_length_gradient_error = 0.0

    print("\n" + "=" * 55)
    print("STEP 2: CABLE-KINEMATICS VALIDATION")
    print("=" * 55)

    for model_name, model in MODELS.items():
        L1 = model["L1"]

        for theta1_deg, theta2_deg in test_postures_deg:
            theta1 = np.deg2rad(theta1_deg)
            theta2 = np.deg2rad(theta2_deg)

            for cable_name, cable in CABLES.items():

                # Analytic and numerical attachment Jacobians.
                J_analytic = attachment_jacobian(
                    theta1, theta2, L1,
                    cable["attachment_local"],
                    cable["link"],
                )

                J_numeric = finite_difference_attachment_jacobian(
                    theta1, theta2, L1,
                    cable["attachment_local"],
                    cable["link"],
                )

                J_error = np.max(
                    np.abs(J_analytic - J_numeric)
                )

                max_jacobian_error = max(
                    max_jacobian_error, J_error
                )

                # Analytic and numerical length gradients.
                geometry = tendon_geometry(
                    theta1, theta2, L1, cable
                )

                gradient_numeric = finite_difference_tendon_gradient(
                    theta1, theta2, L1, cable
                )

                gradient_error = np.max(
                    np.abs(
                        geometry["length_gradient"]
                        - gradient_numeric
                    )
                )

                max_length_gradient_error = max(
                    max_length_gradient_error,
                    gradient_error,
                )

                assert np.all(np.isfinite(J_analytic))
                assert np.isfinite(geometry["length"])
                assert geometry["length"] > 0.0

                assert J_error < tolerance, (
                    f"Jacobian mismatch: {model_name}, "
                    f"{cable_name}, "
                    f"posture=({theta1_deg}, "
                    f"{theta2_deg}) deg, "
                    f"error={J_error:.3e}"
                )

                assert gradient_error < tolerance, (
                    f"Length-gradient mismatch: "
                    f"{model_name}, {cable_name}, "
                    f"error={gradient_error:.3e}"
                )

    print(
        "Maximum absolute Jacobian error: "
        f"{max_jacobian_error:.3e} cm/rad"
    )

    print(
        "Maximum absolute length-gradient error: "
        f"{max_length_gradient_error:.3e} cm/rad"
    )

    print(f"Acceptance tolerance: {tolerance:.1e}")

    print("All Step 2 numerical checks passed.")


# --------------------------------------------------
# 11. REFERENCE-POSTURE CABLE SUMMARY
# --------------------------------------------------

def print_reference_cable_geometry(model_name):
    """Print cable geometry at q = (0, 0)."""
    model = MODELS[model_name]
    L1 = model["L1"]

    print(f"\nReference cable geometry: {model_name}")

    for cable_name, cable in CABLES.items():
        result = tendon_geometry(
            0.0, 0.0, L1, cable
        )

        r = result["attachment"]
        s = result["anchor"]

        print(
            f"{cable_name}: "
            f"anchor={s}, attachment={r}, "
            f"length={result['length']:.4f} cm"
        )

# --------------------------------------------------
# 12. STRUCTURE MATRIX
# --------------------------------------------------

def structure_matrix(theta1, theta2, L1, cable_names=None):
    """
    Construct A in R^(2 x m) such that:

        length_rate = A.T @ q_dot
        cable_generated_torque = -A @ tensions

    The columns are derived from the actual cable geometry.
    """
    if cable_names is None:
        cable_names = list(CABLES.keys())

    A = np.zeros((2, len(cable_names)))

    for j, name in enumerate(cable_names):
        cable = CABLES[name]

        geom = tendon_geometry(
            theta1, theta2, L1, cable
        )

        # The gradient of tendon length with respect to q.
        A[:, j] = geom["length_gradient"]

    return A


# --------------------------------------------------
# 13. THREE-TENDON SUBSETS
# --------------------------------------------------

THREE_TENDON_SUBSETS = {
    "T1_T2_T3": ["T1", "T2", "T3"],
    "T1_T2_T4": ["T1", "T2", "T4"],
    "T1_T3_T4": ["T1", "T3", "T4"],
    "T2_T3_T4": ["T2", "T3", "T4"],
}


# --------------------------------------------------
# 14. STRUCTURE-MATRIX AND VIRTUAL-WORK TESTS
# --------------------------------------------------

def run_step3_tests():
    """
    Verify:
      1. Structure-matrix dimensions.
      2. A.T @ q_dot equals numerical length rates.
      3. Cable-generated torque equals -A @ tensions.
      4. The structure matrix is finite at the reference posture.
    """
    q_dot = np.array([0.23, -0.17])  # rad/s
    tensions = np.array([3.0, 4.0, 5.0, 6.0])  # N

    tolerance = 1e-6
    max_rate_error = 0.0
    max_torque_error = 0.0

    print("\n" + "=" * 55)
    print("STEP 3: STRUCTURE-MATRIX VALIDATION")
    print("=" * 55)

    for model_name, model in MODELS.items():
        L1 = model["L1"]

        for theta1_deg, theta2_deg in [
            (0.0, 0.0),
            (-30.0, 45.0),
            (40.0, 80.0),
        ]:
            theta1 = np.deg2rad(theta1_deg)
            theta2 = np.deg2rad(theta2_deg)

            A4 = structure_matrix(
                theta1, theta2, L1
            )

            assert A4.shape == (2, 4)
            assert np.all(np.isfinite(A4))

            # Verify the length-rate equation by finite difference.
            predicted_rates = A4.T @ q_dot

            h = 1e-6
            q = np.array([theta1, theta2])
            q_plus = q + h * q_dot
            q_minus = q - h * q_dot

            lengths_plus = np.array([
                tendon_geometry(
                    q_plus[0], q_plus[1], L1, CABLES[name]
                )["length"]
                for name in CABLES
            ])

            lengths_minus = np.array([
                tendon_geometry(
                    q_minus[0], q_minus[1], L1, CABLES[name]
                )["length"]
                for name in CABLES
            ])

            numerical_rates = (
                lengths_plus - lengths_minus
            ) / (2.0 * h)

            rate_error = np.max(
                np.abs(predicted_rates - numerical_rates)
            )

            max_rate_error = max(max_rate_error, rate_error)

            assert rate_error < tolerance, (
                f"Length-rate check failed for {model_name}: "
                f"{rate_error:.3e}"
            )

            # Verify the tendon-generated torque independently.
            # Cable force on attachment = tension * unit vector.
            tau_cable_direct = np.zeros(2)

            for name, tension in zip(CABLES, tensions):
                geom = tendon_geometry(
                    theta1, theta2, L1, CABLES[name]
                )

                tau_cable_direct += (
                    geom["jacobian"].T
                    @ geom["unit_vector"]
                    * tension
                )

            tau_cable_matrix = -A4 @ tensions

            torque_error = np.max(
                np.abs(tau_cable_direct - tau_cable_matrix)
            )

            max_torque_error = max(
                max_torque_error, torque_error
            )

            assert torque_error < tolerance, (
                f"Virtual-work torque check failed "
                f"for {model_name}: {torque_error:.3e}"
            )

            # Confirm that selecting three columns works.
            for subset in THREE_TENDON_SUBSETS.values():
                A3 = structure_matrix(
                    theta1, theta2, L1, subset
                )
                assert A3.shape == (2, 3)

    print(
        "Maximum absolute length-rate error: "
        f"{max_rate_error:.3e} cm/s"
    )
    print(
        "Maximum absolute torque-mapping error: "
        f"{max_torque_error:.3e} N*cm"
    )
    print(f"Acceptance tolerance: {tolerance:.1e}")
    print("All Step 3 numerical checks passed.")


# --------------------------------------------------
# 15. REFERENCE MATRIX FOR HAND VERIFICATION
# --------------------------------------------------

def print_reference_structure_matrix(model_name):
    """
    Print reference-posture cable geometry and A at q = (0, 0).

    A uses the convention:
        length_rate = A.T @ q_dot
        cable_generated_torque = -A @ tensions
    """
    model = MODELS[model_name]
    L1 = model["L1"]

    A4 = structure_matrix(0.0, 0.0, L1)

    print("\n" + "-" * 65)
    print(f"REFERENCE STRUCTURE MATRIX: {model_name}")
    print(f"L1 = {L1:.2f} cm")
    print("Posture: theta1 = 0 deg, theta2 = 0 deg")
    print("Rows: shoulder, elbow")
    print("Columns: T1, T2, T3, T4")
    print("-" * 65)

    for name in CABLES:
        geom = tendon_geometry(
            0.0, 0.0, L1, CABLES[name]
        )

        print(f"\n{name}")
        print("  Anchor:", geom["anchor"])
        print("  Attachment:", geom["attachment"])
        print("  Cable vector:", geom["cable_vector"])
        print(f"  Length: {geom['length']:.6f} cm")
        print("  Unit vector:", geom["unit_vector"])
        print("  Jacobian:\n", geom["jacobian"])
        print("  A column:", geom["length_gradient"])

    print("\nFull four-tendon matrix A (2 x 4):")
    print(np.array2string(A4, precision=6, suppress_small=True))

    for subset_name, subset in THREE_TENDON_SUBSETS.items():
        A3 = structure_matrix(0.0, 0.0, L1, subset)
        print(f"\n{subset_name}, A (2 x 3):")
        print(np.array2string(A3, precision=6, suppress_small=True))


# --------------------------------------------------
# 16. STEP 4A: NORMALIZED TORQUE-DIRECTION FEASIBILITY
# --------------------------------------------------

def positive_tension_feasibility(A, tau_direction, tolerance=1e-8):
    """
    Check whether A @ T = tau_direction has a solution
    with every tendon tension strictly positive.

    We maximize delta subject to:
        A @ T = tau_direction
        T[i] >= delta
        delta >= 0

    A positive optimal delta indicates that all tensions
    can be strictly positive.

    tau_direction is a normalized mathematical direction,
    NOT a measured or assumed physical load.
    """

    A = np.asarray(A, dtype=float)
    tau_direction = np.asarray(tau_direction, dtype=float)

    n_tendons = A.shape[1]

    # Decision variables are [T1, ..., Tm, delta].
    # Minimize -delta to maximize the smallest tension.
    objective = np.zeros(n_tendons + 1)
    objective[-1] = -1.0

    # A @ T = tau_direction
    A_eq = np.hstack([
        A,
        np.zeros((A.shape[0], 1))
    ])

    # T[i] >= delta  -->  -T[i] + delta <= 0
    A_ub = np.zeros((n_tendons, n_tendons + 1))

    for i in range(n_tendons):
        A_ub[i, i] = -1.0
        A_ub[i, -1] = 1.0

    result = linprog(
        c=objective,
        A_ub=A_ub,
        b_ub=np.zeros(n_tendons),
        A_eq=A_eq,
        b_eq=tau_direction,
        bounds=[(0, None)] * (n_tendons + 1),
        method="highs"
    )

    if result.status == 3:
        # The minimum tension can increase without bound
        # while preserving equilibrium: feasible.
        return True

    if not result.success:
        return False

    delta = result.x[-1]
    return delta > tolerance


def run_step4a_tests(n_directions=360):
    """
    Compare four three-tendon subsets and the full
    four-tendon configuration at the reference posture.

    The metric is the fraction of sampled torque directions
    that admit strictly positive tendon tensions.

    This is a normalized geometric feasibility metric,
    not a percentage of real-world loads that can be carried.
    """

    configurations = {
        **THREE_TENDON_SUBSETS,
        "T1_T2_T3_T4": list(CABLES.keys())
    }

    # Avoid counting both 0 and 2*pi as separate directions.
    angles = np.linspace(
        0.0, 2.0 * np.pi, n_directions, endpoint=False
    )

    directions = [
        np.array([np.cos(phi), np.sin(phi)])
        for phi in angles
    ]

    print("\n" + "=" * 70)
    print("STEP 4A: NORMALIZED TORQUE-DIRECTION FEASIBILITY")
    print("=" * 70)
    print("Posture: theta1 = 0 deg, theta2 = 0 deg")
    print(f"Sampled directions: {n_directions}")
    print("Tension condition: every selected tendon must have T > 0")
    print("No physical load magnitude or tension limit is assumed.")

    for model_name, model in MODELS.items():
        print("\n" + "-" * 70)
        print(f"Arm geometry: {model_name}")
        print("-" * 70)

        for config_name, cable_names in configurations.items():
            A = structure_matrix(
                0.0,
                0.0,
                model["L1"],
                cable_names
            )

            feasible_count = 0

            for tau_direction in directions:
                if positive_tension_feasibility(
                    A, tau_direction
                ):
                    feasible_count += 1

            fraction = feasible_count / n_directions
            percentage = 100.0 * fraction

            print(
                f"{config_name:12s}: "
                f"{feasible_count:3d}/{n_directions} directions "
                f"({percentage:6.2f}%)"
            )

    print("\nStep 4A completed.")
    print(
        "Interpretation: results describe sampled torque-direction "
        "feasibility at one posture only."
    )


# --------------------------------------------------
# 17. STEP 4B: POSTURE-DEPENDENT FEASIBILITY
# --------------------------------------------------

def run_step4b_tests(n_postures_per_joint=9, n_directions=36):
    """
    Survey positive-tension torque-direction feasibility
    over a coarse grid inside the specified joint limits.

    Reports:
      - Mean feasible-direction percentage across postures.
      - Minimum and maximum percentage at any sampled posture.
      - Number of sampled postures with all directions feasible.

    This remains a normalized geometric analysis. It does not
    establish physical load capacity or feasibility between
    the sampled postures.
    """

    configurations = {
        **THREE_TENDON_SUBSETS,
        "T1_T2_T3_T4": list(CABLES.keys())
    }

    theta1_values = np.linspace(-140.0, 90.0, n_postures_per_joint)
    theta2_values = np.linspace(0.0, 140.0, n_postures_per_joint)

    angles = np.linspace(
        0.0, 2.0 * np.pi, n_directions, endpoint=False
    )

    directions = [
        np.array([np.cos(phi), np.sin(phi)])
        for phi in angles
    ]

    print("\n" + "=" * 75)
    print("STEP 4B: POSTURE-DEPENDENT FEASIBILITY")
    print("=" * 75)
    print(f"Posture grid: {n_postures_per_joint} x "
          f"{n_postures_per_joint}")
    print(f"Torque directions per posture: {n_directions}")
    print("Joint limits: theta1 [-140, 90] deg; theta2 [0, 140] deg")
    print("Criterion: all selected tendon tensions strictly positive")

    for model_name, model in MODELS.items():
        print("\n" + "-" * 75)
        print(f"Arm geometry: {model_name}")
        print("-" * 75)

        for config_name, cable_names in configurations.items():
            posture_percentages = []

            for theta1_deg in theta1_values:
                for theta2_deg in theta2_values:
                    A = structure_matrix(
                        np.deg2rad(theta1_deg),
                        np.deg2rad(theta2_deg),
                        model["L1"],
                        cable_names
                    )

                    feasible_count = sum(
                        positive_tension_feasibility(A, direction)
                        for direction in directions
                    )

                    posture_percentages.append(
                        100.0 * feasible_count / n_directions
                    )

            posture_percentages = np.asarray(posture_percentages)

            mean_pct = np.mean(posture_percentages)
            min_pct = np.min(posture_percentages)
            max_pct = np.max(posture_percentages)

            fully_feasible = np.count_nonzero(
                posture_percentages >= 100.0 - 1e-9
            )

            print(
                f"{config_name:12s}: "
                f"mean={mean_pct:6.2f}%, "
                f"min={min_pct:6.2f}%, "
                f"max={max_pct:6.2f}%, "
                f"all-directions-feasible postures="
                f"{fully_feasible}/{len(posture_percentages)}"
            )

    print("\nStep 4B completed.")
    print(
        "These are sampled results only; they do not prove "
        "feasibility at every point in the continuous workspace."
    )


# --------------------------------------------------
# STEP 5: POSTURE-DEPENDENT FEASIBILITY HEATMAPS
# --------------------------------------------------

def plot_step5_feasibility_heatmaps(
    n_postures_per_joint=9,
    n_directions=36,
    save_path="workspace_feasibility_heatmaps.png",
):
    """
    Plot sampled torque-direction feasibility across joint space.

    The color at each posture represents the percentage of sampled
    normalized torque directions for which a strictly positive
    tendon-tension solution exists.

    This is a sampled mathematical feasibility metric, not a
    physical load-capacity measurement.

    BUG FIX: theta1_values and theta2_values are now sampled in
    degrees (consistent with Step 4B) and converted to radians
    inside the loop via np.deg2rad(). The previous version passed
    THETA1_LIMITS (already radians) to np.linspace and then applied
    np.deg2rad() again inside the loop, causing all postures to be
    evaluated near (0, 0) and producing artifically high feasibility.
    """
    import matplotlib.pyplot as plt

    # Sample in degrees — convert to radians inside the loop.
    # This is consistent with run_step4b_tests().
    theta1_values = np.linspace(-140.0, 90.0, n_postures_per_joint)
    theta2_values = np.linspace(0.0, 140.0, n_postures_per_joint)

    # Use the two configurations selected for comparison.
    configurations = {
        "Four tendons (T1-T2-T3-T4)": list(CABLES.keys()),
        "Three tendons (T1-T2-T4)": THREE_TENDON_SUBSETS[
            "T1_T2_T4"
        ],
    }

    # Unit torque directions, sampled uniformly around a circle.
    angles = np.linspace(
        0.0, 2.0 * np.pi, n_directions, endpoint=False
    )
    torque_directions = [
        np.array([np.cos(phi), np.sin(phi)])
        for phi in angles
    ]

    results = {}

    print("\n" + "=" * 70)
    print("STEP 5: POSTURE-DEPENDENT FEASIBILITY HEATMAPS")
    print("=" * 70)
    print(f"Posture grid: {n_postures_per_joint} x "
          f"{n_postures_per_joint}")
    print(f"Torque directions per posture: {n_directions}")
    print("Values are percentages of sampled directions.")

    for model_name, model in MODELS.items():
        L1 = model["L1"]

        for config_name, cable_names in configurations.items():
            # Rows correspond to theta2; columns correspond to theta1.
            feasibility_map = np.zeros(
                (len(theta2_values), len(theta1_values))
            )

            for i, theta2_deg in enumerate(theta2_values):
                for j, theta1_deg in enumerate(theta1_values):
                    A = structure_matrix(
                        np.deg2rad(theta1_deg),
                        np.deg2rad(theta2_deg),
                        L1,
                        cable_names,
                    )

                    feasible_count = sum(
                        positive_tension_feasibility(A, direction)
                        for direction in torque_directions
                    )

                    feasibility_map[i, j] = (
                        100.0 * feasible_count / n_directions
                    )

            results[(model_name, config_name)] = feasibility_map
            print(
                f"DEBUG {model_name} | {config_name}: "
                f"shape={feasibility_map.shape}, "
                f"unique={np.unique(feasibility_map)}, "
                f"nonzero={np.count_nonzero(feasibility_map)}/"
                f"{feasibility_map.size}"
)
            print(
                f"{model_name:18s} | {config_name:30s} | "
                f"mean={np.mean(feasibility_map):6.2f}% | "
                f"min={np.min(feasibility_map):6.2f}% | "
                f"max={np.max(feasibility_map):6.2f}%"
            )

    fig, axes = plt.subplots(
        2, 2, figsize=(13, 9), constrained_layout=True
    )

    panels = [
        ("original_sketch", "Four tendons (T1-T2-T3-T4)", axes[0, 0]),
        ("original_sketch", "Three tendons (T1-T2-T4)", axes[0, 1]),
        ("human_reference", "Four tendons (T1-T2-T3-T4)", axes[1, 0]),
        ("human_reference", "Three tendons (T1-T2-T4)", axes[1, 1]),
    ]

    image = None

    for model_name, config_name, ax in panels:
        values = results[(model_name, config_name)]

        image = ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=[
                theta1_values[0],
                theta1_values[-1],
                theta2_values[0],
                theta2_values[-1],
            ],
            vmin=0,
            vmax=100,
            interpolation="nearest",
        )

        geometry_title = (
            "Original sketch"
            if model_name == "original_sketch"
            else "Human-reference geometry"
        )

        ax.set_title(f"{geometry_title}\n{config_name}")
        ax.set_xlabel("Shoulder angle, theta1 (deg)")
        ax.set_ylabel("Elbow angle, theta2 (deg)")
        ax.set_xticks(np.linspace(-140, 90, 6))
        ax.set_yticks(np.linspace(0, 140, 8))
        ax.grid(False)

    colorbar = fig.colorbar(
        image, ax=axes.ravel().tolist(), shrink=0.90
    )
    colorbar.set_label(
        "Feasible sampled torque directions (%)"
    )

    fig.suptitle(
        "Posture-dependent positive-tension feasibility",
        fontsize=14,
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nHeatmap saved to: {save_path}")

    plt.show()

    return results


# --------------------------------------------------
# STEP 5: PLANAR GEOMETRIC WORKSPACE ANALYSIS
# --------------------------------------------------

def plot_planar_workspace(
    n_theta1=231,
    n_theta2=141,
    save_path="planar_workspace_comparison.png",
):
    """
    Generate planar reachable-workspace plots for both arm geometries.

    Each point represents the wrist/hand endpoint B generated by
    an admissible pair of shoulder and elbow angles.

    This is geometric workspace only. Cable-force feasibility
    and physical load capacity are not imposed here.

    BUG FIX: THETA1_LIMITS and THETA2_LIMITS are already stored in
    radians (converted at module level). The previous version wrapped
    them in np.deg2rad() again, shrinking the angles by a factor of
    ~57.3 and producing a narrow vertical cluster near maximum reach.
    The fix removes the redundant np.deg2rad() calls here.
    """

    import matplotlib.pyplot as plt

    # THETA1_LIMITS and THETA2_LIMITS are already in radians.
    # Do NOT apply np.deg2rad() here — that was the bug.
    theta1_values = np.linspace(
        THETA1_LIMITS[0],
        THETA1_LIMITS[1],
        n_theta1,
    )

    theta2_values = np.linspace(
        THETA2_LIMITS[0],
        THETA2_LIMITS[1],
        n_theta2,
    )

    # --- Numerical validation of sampling ---
    print("\n" + "=" * 55)
    print("WORKSPACE SAMPLING VALIDATION")
    print("=" * 55)
    print(
        f"Shoulder angle range: "
        f"{np.rad2deg(theta1_values[0]):.2f} deg to "
        f"{np.rad2deg(theta1_values[-1]):.2f} deg "
        f"({n_theta1} samples)"
    )
    print(
        f"Elbow angle range: "
        f"{np.rad2deg(theta2_values[0]):.2f} deg to "
        f"{np.rad2deg(theta2_values[-1]):.2f} deg "
        f"({n_theta2} samples)"
    )
    assert np.isclose(np.rad2deg(theta1_values[0]), -140.0, atol=0.1), \
        "Shoulder lower limit incorrect"
    assert np.isclose(np.rad2deg(theta1_values[-1]), 90.0, atol=0.1), \
        "Shoulder upper limit incorrect"
    assert np.isclose(np.rad2deg(theta2_values[0]), 0.0, atol=0.1), \
        "Elbow lower limit incorrect"
    assert np.isclose(np.rad2deg(theta2_values[-1]), 140.0, atol=0.1), \
        "Elbow upper limit incorrect"

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 6),
        constrained_layout=True,
    )

    workspace_results = {}

    for ax, (model_name, model) in zip(
        axes, MODELS.items()
    ):

        L1 = model["L1"]
        L2 = model["L2"]
        expected_reach = L1 + L2

        x_points = []
        y_points = []
        ex_points = []  # elbow (joint 2) positions
        ey_points = []

        # Sample the complete permitted joint-angle domain.
        for theta1 in theta1_values:
            for theta2 in theta2_values:

                B = forward_kinematics(
                    theta1,
                    theta2,
                    L1,
                    L2,
                )

                x_points.append(B[0])
                y_points.append(B[1])

                # Elbow position = end of link 1
                ex_points.append(L1 * np.cos(theta1))
                ey_points.append(L1 * np.sin(theta1))

        x_points = np.asarray(x_points)
        y_points = np.asarray(y_points)
        ex_points = np.asarray(ex_points)
        ey_points = np.asarray(ey_points)

        workspace_results[model_name] = {
            "x": x_points,
            "y": y_points,
        }

        # --- Numerical workspace checks ---
        x_range = x_points.max() - x_points.min()
        y_range = y_points.max() - y_points.min()
        radii = np.sqrt(x_points**2 + y_points**2)
        max_radius = radii.max()

        print(f"\nWorkspace: {model_name}")
        print(f"  L1={L1} cm, L2={L2} cm, expected reach={expected_reach:.2f} cm")
        print(f"  Sampled endpoint positions: {len(x_points)}")
        print(f"  x range: {x_points.min():.3f} to {x_points.max():.3f} cm  (span={x_range:.3f})")
        print(f"  y range: {y_points.min():.3f} to {y_points.max():.3f} cm  (span={y_range:.3f})")
        print(f"  Max radial distance: {max_radius:.4f} cm (expected <= {expected_reach:.2f} cm)")

        assert x_range > expected_reach * 0.5, \
            f"x_B barely varies ({x_range:.3f} cm) — joint sampling is likely wrong"
        assert y_range > expected_reach * 0.5, \
            f"y_B barely varies ({y_range:.3f} cm) — joint sampling is likely wrong"
        assert max_radius <= expected_reach + 1e-4, \
            f"Max radius {max_radius:.4f} exceeds full reach {expected_reach:.2f}"
        assert np.isclose(max_radius, expected_reach, atol=0.1), \
            f"Max radius {max_radius:.4f} is far from expected reach {expected_reach:.2f}"

        print(f"  Workspace sampling checks PASSED for {model_name}")

        ax.scatter(
            x_points,
            y_points,
            s=0.5,
            alpha=0.25,
            color="steelblue",
            label="Hand (wrist)",
            rasterized=True,
        )

        ax.scatter(
            ex_points,
            ey_points,
            s=0.5,
            alpha=0.20,
            color="darkorange",
            label="Elbow",
            rasterized=True,
        )

        # Shoulder reference.
        ax.scatter(
            0.0, 0.0,
            color="red",
            s=18,
            marker="o",
            label="Shoulder",
            zorder=5,
        )

        ax.set_title(
            f"{model_name.replace('_', ' ').title()}\n"
            f"L1={L1:.1f} cm, L2={L2:.1f} cm"
        )

        ax.set_xlabel("Horizontal position, x_B (cm)")
        ax.set_ylabel("Vertical position, y_B (cm)")

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

        # Small legend placed outside the plot area (upper right corner)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0,
            fontsize=7,
            markerscale=1.5,
            framealpha=0.7,
        )

        ax.set_xlim(
            -(L1 + L2) - 2,
            (L1 + L2) + 2,
        )

        ax.set_ylim(
            -(L1 + L2) - 2,
            (L1 + L2) + 2,
        )

    fig.suptitle(
        "Planar Geometric Workspace of the Two-Link Upper Limb",
        fontsize=15,
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"\nWorkspace figure saved: {save_path}")

    plt.show()

    return workspace_results


# --------------------------------------------------
# STEP 6: CABLE-FEASIBLE CARTESIAN WORKSPACE
# --------------------------------------------------

def plot_cable_feasible_workspace(
    n_theta1=51,
    n_theta2=41,
    n_directions=36,
    feasibility_threshold=100.0,
    save_path="cable_feasible_workspace.png",
):
    """
    Plot the cable-feasible Cartesian workspace for:
      - 3-tendon case: T1-T2-T4 (best subset, one redundancy, m=n+1)
      - 4-tendon case: T1-T2-T3-T4 (two redundancies, m=n+2)

    Assignment condition (Activity 4):
      "Consider a posture part of the workspace only if it lies
       within the joint limits and the cables can hold it with
       all tensions positive (T > 0)."

    Physical interpretation:
      Cables act as agonist/antagonist pairs — they can only pull,
      never push. A posture is cable-feasible ONLY IF every cable
      remains taut (T_i > 0) simultaneously for ALL possible torque
      directions. This is the wrench-closure condition:
        feasibility_threshold = 100% of sampled directions must pass.

    The LP for each direction maximises the minimum tension margin δ:
        max δ  s.t.  A·T = τ_dir,  T_i ≥ δ ≥ 0
    Feasible ↔ optimal δ > 0 (all cables taut with positive margin).

    Endpoints of feasible postures are plotted in colour.
    Geometrically reachable but cable-infeasible endpoints are shown
    as a faint grey background for reference.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    configurations = {
        "3-tendon: T1–T2–T4\n(m = n+1, one redundancy)": ["T1", "T2", "T4"],
        "4-tendon: T1–T2–T3–T4\n(m = n+2, two redundancies)": list(CABLES.keys()),
    }

    theta1_deg_vals = np.linspace(-140.0, 90.0, n_theta1)
    theta2_deg_vals = np.linspace(0.0, 140.0, n_theta2)

    angles = np.linspace(0.0, 2.0 * np.pi, n_directions, endpoint=False)
    directions = [np.array([np.cos(a), np.sin(a)]) for a in angles]

    print("\n" + "=" * 70)
    print("STEP 6: CABLE-FEASIBLE CARTESIAN WORKSPACE")
    print("=" * 70)
    print(f"Joint grid: {n_theta1} x {n_theta2} = {n_theta1*n_theta2} postures")
    print(f"Torque directions per posture: {n_directions}")
    print(f"Feasibility threshold: {feasibility_threshold:.0f}% of directions (wrench-closure: all T_i > 0 for every torque direction)")
    print("Configurations: 3-tendon T1-T2-T4 and 4-tendon T1-T2-T3-T4")

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 12),
        constrained_layout=True,
    )

    for row_idx, (model_name, model) in enumerate(MODELS.items()):
        L1 = model["L1"]
        L2 = model["L2"]
        reach = L1 + L2

        geometry_label = (
            "Original Sketch"
            if model_name == "original_sketch"
            else "Human Reference"
        )

        for col_idx, (config_label, cable_names) in enumerate(
            configurations.items()
        ):
            ax = axes[row_idx, col_idx]

            x_feasible, y_feasible = [], []
            x_infeasible, y_infeasible = [], []

            feasible_count_total = 0
            total_postures = n_theta1 * n_theta2

            for t1_deg in theta1_deg_vals:
                for t2_deg in theta2_deg_vals:
                    t1 = np.deg2rad(t1_deg)
                    t2 = np.deg2rad(t2_deg)

                    B = forward_kinematics(t1, t2, L1, L2)

                    A = structure_matrix(t1, t2, L1, cable_names)

                    # Wrench-closure: ALL directions must admit T > 0.
                    # Short-circuit as soon as any direction fails.
                    wrench_closed = all(
                        positive_tension_feasibility(A, d)
                        for d in directions
                    )
                    pct = 100.0 if wrench_closed else 0.0

                    if wrench_closed:
                        x_feasible.append(B[0])
                        y_feasible.append(B[1])
                        feasible_count_total += 1
                    else:
                        x_infeasible.append(B[0])
                        y_infeasible.append(B[1])

            x_feasible = np.asarray(x_feasible)
            y_feasible = np.asarray(y_feasible)
            x_infeasible = np.asarray(x_infeasible)
            y_infeasible = np.asarray(y_infeasible)

            pct_feasible = 100.0 * feasible_count_total / total_postures

            print(
                f"\n{geometry_label} | {cable_names}: "
                f"{feasible_count_total}/{total_postures} postures feasible "
                f"({pct_feasible:.1f}%)"
            )
            if len(x_feasible):
                print(
                    f"  Feasible x range: {x_feasible.min():.2f} to "
                    f"{x_feasible.max():.2f} cm"
                )
                print(
                    f"  Feasible y range: {y_feasible.min():.2f} to "
                    f"{y_feasible.max():.2f} cm"
                )

            # Grey background: full geometric workspace
            if len(x_infeasible):
                ax.scatter(
                    x_infeasible, y_infeasible,
                    s=2, alpha=0.12, color="grey",
                    rasterized=True, label="Geom. reachable (infeasible)",
                )

            # Coloured foreground: cable-feasible postures
            if len(x_feasible):
                ax.scatter(
                    x_feasible, y_feasible,
                    s=4, alpha=0.55,
                    color="steelblue" if col_idx == 0 else "darkorange",
                    rasterized=True,
                    label=f"Cable-feasible ({pct_feasible:.0f}% of postures)",
                )

            # Shoulder marker
            ax.scatter(0, 0, color="red", s=18, zorder=6, label="Shoulder")

            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-(reach + 3), reach + 3)
            ax.set_ylim(-(reach + 3), reach + 3)
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("x_B (cm)")
            ax.set_ylabel("y_B (cm)")
            ax.set_title(
                f"{geometry_label}  |  L1={L1:.1f}, L2={L2:.1f} cm\n"
                f"{config_label}",
                fontsize=9,
            )
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0,
                fontsize=7,
                markerscale=1.5,
                framealpha=0.75,
            )

    fig.suptitle(
        "Cable-Feasible Cartesian Workspace (Wrench-Closure Condition)\n"
        "Posture included only if ALL 36 sampled torque directions"
        r" admit all-positive tensions ($T_i > 0$, no cable slack)",
        fontsize=11,
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nCable-feasible workspace figure saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    run_step1_tests()
    run_step2_tests()
    run_step3_tests()

    print_reference_cable_geometry("original_sketch")
    print_reference_cable_geometry("human_reference")

    print_reference_structure_matrix("original_sketch")
    print_reference_structure_matrix("human_reference")

    run_step4a_tests()
    run_step4b_tests()

    plot_planar_workspace()
    plot_step5_feasibility_heatmaps()
    plot_cable_feasible_workspace()
