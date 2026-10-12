#!/usr/bin/env python3
"""Check each edge of a mapper3d run's pose graph against KITTI ground truth.

regression_test.py gives one APE number per trajectory. This script shows where that number comes
from. It reads the .g2o dumps and keyframe_stamps.txt that every run writes, looks up each keyframe
in the ground truth by its timestamp, and prints:

  * Odometry edges: how far each keyframe-to-keyframe edge is from the ground truth, per axis.
    Set the odometry sigmas from these numbers.
  * Loop closures: how far each loop closure is from the ground truth, split into loop closures close to
    the ground truth and far from it (see below).
  * Edges over the GNC limit: every edge whose residual in the optimized graph is above the limit
    past which GNC's TLS loss gives an edge zero weight. GNC's weights are not saved, so this is
    how the script judges which edges GNC rejected.
  * Keyframe APE of the unoptimized and optimized keyframe trajectories, split into xy and z.

Close and far: KITTI does not say which loop closures are right. Its ground truth is one GPS/IMU pose per
scan. So the script compares a loop closure's measured relative pose with the relative pose of its two
keyframes in the ground truth, and counts it as far from the ground truth when the translation error is
over --far-translation-error or the rotation error is over --far-rotation-error.

Needs evo (pip install evo), as regression_test.py does.

Usage:
    check_pose_graph.py <regression_runs>/<timestamp>/07
    check_pose_graph.py <regression_runs>/<timestamp>/0*     # every sequence of one run
"""

import argparse
import pathlib
from dataclasses import dataclass

import numpy as np
from evo.core import lie_algebra, metrics, sync
from evo.core.trajectory import Plane, PoseTrajectory3D
from evo.tools import file_interface
from scipy.linalg import logm
from scipy.spatial.transform import Rotation

from kitti_gt_to_tum import read_ground_truth

# GTSAM's GncOptimizer sets each factor's limit with setInlierCostThresholdsAtProbability(0.99), so a
# 6-DoF edge is rejected once its squared whitened residual is above chi2inv(0.99, 6).
GNC_CHI2_LIMIT = 16.812

# Timestamp tolerance when looking a keyframe up in times.txt, the same as regression_test.py's.
TIME_TOLERANCE = 0.01


@dataclass
class GraphEdge:
    """One EDGE_SE3:QUAT line of a .g2o file."""

    keyframe_i: int
    keyframe_j: int
    # 4x4 pose of keyframe j in the frame of keyframe i, as the front end or a loop closure measured it.
    measured_relative_pose: np.ndarray
    # 6x6 information matrix. Its rows and columns are ordered translation first: x, y, z, rx, ry, rz.
    information: np.ndarray


@dataclass
class CheckedEdge:
    """One graph edge compared with the ground truth.

    The errors are the measured pose of keyframe j, in the frame of keyframe j's ground-truth pose, so
    they are all zero for a perfect edge. In the velodyne frame x points forward, y left and z up.
    """

    keyframe_i: int
    keyframe_j: int
    # The edge's information matrix, as in GraphEdge.
    information: np.ndarray
    # Distance between the two keyframes in the ground truth (m).
    ground_truth_distance: float
    # Length of the edge's measured translation (m).
    measured_distance: float
    # x, y, z (m).
    translation_error_vector: np.ndarray
    # Rotation vector, the rotation axis times the angle: x, y, z (degrees).
    rotation_error_vector_deg: np.ndarray
    # Magnitude of translation_error_vector (m).
    translation_error_magnitude: float
    # Magnitude of the x and y parts of translation_error_vector (m).
    translation_error_magnitude_xy: float
    # Magnitude of the z part of translation_error_vector (m).
    translation_error_magnitude_z: float
    # Angle of the rotation error (degrees).
    rotation_error_magnitude_deg: float
    # Squared whitened residual at the optimized poses, the number GNC compares with GNC_CHI2_LIMIT.
    # None when there is no optimized graph, or when it lacks either keyframe.
    chi2: float | None


@dataclass
class FarFromGroundTruthLimits:
    """The errors past which a loop closure counts as far from the ground truth."""

    # Limit on the translation error's magnitude (m).
    translation_error: float
    # Limit on the rotation error's angle (degrees).
    rotation_error_deg: float

    def is_far(self, translation_error_magnitude, rotation_error_magnitude_deg):
        """Return True when either error is over its limit."""
        return (translation_error_magnitude > self.translation_error
                or rotation_error_magnitude_deg > self.rotation_error_deg)

    def describe(self):
        """Return the limits as text, to print next to the counts they give."""
        return f"error > {self.translation_error:g} m or > {self.rotation_error_deg:g} deg"


def pose_from_values(values):
    """Return the 4x4 pose for the seven numbers 'x y z qx qy qz qw', the layout used in g2o lines."""
    x, y, z, qx, qy, qz, qw = values
    # scipy takes the quaternion with its scalar last, the same order as g2o.
    rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    return lie_algebra.se3(rotation, np.array([x, y, z]))


def information_from_upper_triangle(values):
    """Return the symmetric 6x6 matrix whose upper triangle, read row by row, is the 21 numbers in values.

    g2o writes only the upper triangle, diagonal included, because the matrix is symmetric. The
    numbers in values fill it in this order:

         0  1  2  3  4  5
            6  7  8  9 10
              11 12 13 14
                 15 16 17
                    18 19
                       20
    """
    size = 6
    expected_count = size * (size + 1) // 2
    if len(values) != expected_count:
        raise ValueError(f"expected {expected_count} upper-triangle numbers, got {len(values)}")

    # triu_indices lists the upper triangle's (row, column) pairs row by row, the same order g2o
    # writes them in, so values can be assigned in one step.
    rows, columns = np.triu_indices(size)

    information = np.zeros((size, size))
    information[rows, columns] = values
    # Mirror to the lower triangle. The diagonal is written twice with the same number.
    information[columns, rows] = values
    return information


def read_g2o_vertices(path):
    """Return the poses of a .g2o file's vertices as {keyframe id: 4x4 pose}."""
    poses = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) == 0:
            continue
        # VERTEX_SE3:QUAT id x y z qx qy qz qw
        if fields[0] == "VERTEX_SE3:QUAT":
            keyframe_id = int(fields[1])
            poses[keyframe_id] = pose_from_values(np.array(fields[2:9], dtype=float))
    return poses


def read_g2o_edges(path):
    """Return the edges of a .g2o file as a list of GraphEdge, in the order of the file."""
    edges = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) == 0:
            continue
        # EDGE_SE3:QUAT id_i id_j x y z qx qy qz qw, then the 21 numbers of the information matrix's
        # upper triangle.
        if fields[0] == "EDGE_SE3:QUAT":
            edge = GraphEdge(
                keyframe_i=int(fields[1]),
                keyframe_j=int(fields[2]),
                measured_relative_pose=pose_from_values(np.array(fields[3:10], dtype=float)),
                information=information_from_upper_triangle(np.array(fields[10:31], dtype=float)),
            )
            edges.append(edge)
    return edges


def find_ground_truth_lines(stamps_path, times):
    """Return {keyframe id: ground-truth line} for every keyframe in keyframe_stamps.txt.

    A ground-truth line is a line number, counted from 0, of both times.txt and poses/NN.txt, so it
    indexes both arrays that read_ground_truth returns. Raises ValueError if any keyframe is more than
    TIME_TOLERANCE from every line, which usually means the run is not of this sequence.
    """
    keyframe_ids = []
    keyframe_times = []
    for line in stamps_path.read_text().splitlines():
        # A '#' header line, then one line per keyframe: 'keyframe_id point_cloud_id timestamp'.
        if line.startswith("#") or line.strip() == "":
            continue
        fields = line.split()
        keyframe_ids.append(int(fields[0]))
        keyframe_times.append(float(fields[2]))

    # For each keyframe, evo finds the line of times.txt nearest in time, and keeps the pair only if the
    # two times are within TIME_TOLERANCE. ape_rmse pairs its poses with the same evo function.
    matched_keyframes, matched_lines = sync.matching_time_indices(np.array(keyframe_times), times,
                                                                  max_diff=TIME_TOLERANCE)
    unmatched_count = len(keyframe_ids) - len(matched_keyframes)
    if unmatched_count > 0:
        raise ValueError(f"{unmatched_count} timestamps in {stamps_path} are more than {TIME_TOLERANCE} s "
                         f"from any ground-truth time - is this the right sequence?")

    # The two lists are the same length. Entry k pairs keyframe number matched_keyframes[k] of the file
    # with line matched_lines[k].
    ground_truth_lines = {}
    for k in range(len(matched_keyframes)):
        keyframe_id = keyframe_ids[matched_keyframes[k]]
        ground_truth_lines[keyframe_id] = matched_lines[k]
    return ground_truth_lines


def edge_chi2(edge, pose_i, pose_j):
    """Return a GraphEdge's chi2, its squared whitened residual, with its keyframes at pose_i and pose_j.

    GTSAM is built with GTSAM_POSE3_EXPMAP, so BetweenFactor's residual is gtsam::Pose3::Logmap of the
    error pose below.
    """
    # Keyframe j in the frame of keyframe i at these poses, what gtsam::Pose3::between() returns.
    relative_pose = lie_algebra.relative_se3(pose_i, pose_j)
    error_pose = lie_algebra.relative_se3(edge.measured_relative_pose, relative_pose)

    # logm returns the matrix log of the 4x4 error pose, which always has this form:
    #     [  0   -wz   wy   ux ]
    #     [  wz   0   -wx   uy ]
    #     [ -wy   wx   0    uz ]
    #     [  0    0    0    0  ]
    # Pose3::Logmap returns the same six numbers as (wx, wy, wz, ux, uy, uz). w is the rotation vector:
    # its direction is the rotation axis and its length the angle in radians. u is not the error pose's
    # translation. The log treats the pose as one motion that turns while it moves, so the path curves:
    # the translation is the straight line from start to end, and u is the movement along the curve.
    # They are equal only with no rotation. The residual uses u, because GTSAM's residual does.
    twist = logm(error_pose)
    # Within about 1e-5 rad of 180 degrees, logm can return a complex matrix instead of the real log,
    # and numpy would then compare a complex chi2 with the limit without an error.
    if np.iscomplexobj(twist):
        raise ValueError(f"edge {edge.keyframe_i}->{edge.keyframe_j}: the error rotation is too close to "
                         f"180 degrees for logm")
    # vee reads (wx, wy, wz) out of the top-left 3x3 block.
    wx, wy, wz = lie_algebra.vee(twist[0:3, 0:3])
    ux, uy, uz = twist[0:3, 3]

    # gtsam::writeG2o swaps the two halves of the information matrix, putting translation first, so the
    # residual has to be in that order too.
    residual = np.array([ux, uy, uz, wx, wy, wz])
    return residual @ edge.information @ residual


def check_edge(edge, true_pose_i, true_pose_j, optimized_poses):
    """Compare a GraphEdge with the ground truth, and return a CheckedEdge.

    true_pose_i and true_pose_j are the ground-truth poses of the edge's two keyframes. optimized_poses
    is {keyframe id: 4x4 pose} from the optimized graph, or None when there is no optimized graph.
    """
    # The ground-truth pose of keyframe j, in the frame of keyframe i's ground-truth pose. This is what
    # edge.measured_relative_pose would be if the measurement had no error.
    true_relative_pose = lie_algebra.relative_se3(true_pose_i, true_pose_j)
    # How far the measurement is from the ground truth. Keyframe i is placed at the same pose in both,
    # and this is where the edge puts keyframe j, in the frame of where the ground truth puts it. It is
    # the identity matrix when the measurement has no error.
    error_pose = lie_algebra.relative_se3(true_relative_pose, edge.measured_relative_pose)
    translation_error_vector = error_pose[0:3, 3]
    rotation_error_vector_deg = np.degrees(lie_algebra.so3_log(error_pose[0:3, 0:3]))

    chi2 = None
    if optimized_poses is not None:
        # Keyframes added after the last optimize() are in the unoptimized graph only, so an edge that
        # joins one of them gets no chi2. The "keyframes N, optimized M" line printed for the run says
        # how many are missing.
        if edge.keyframe_i in optimized_poses and edge.keyframe_j in optimized_poses:
            chi2 = edge_chi2(edge, optimized_poses[edge.keyframe_i], optimized_poses[edge.keyframe_j])

    return CheckedEdge(
        keyframe_i=edge.keyframe_i,
        keyframe_j=edge.keyframe_j,
        information=edge.information,
        ground_truth_distance=np.linalg.norm(true_relative_pose[0:3, 3]),
        measured_distance=np.linalg.norm(edge.measured_relative_pose[0:3, 3]),
        translation_error_vector=translation_error_vector,
        rotation_error_vector_deg=rotation_error_vector_deg,
        translation_error_magnitude=np.linalg.norm(translation_error_vector),
        translation_error_magnitude_xy=np.linalg.norm(translation_error_vector[0:2]),
        translation_error_magnitude_z=abs(translation_error_vector[2]),
        rotation_error_magnitude_deg=np.linalg.norm(rotation_error_vector_deg),
        chi2=chi2,
    )


def is_over_gnc_limit(edge):
    """Return True when a CheckedEdge's chi2 is over GNC_CHI2_LIMIT, the sign that GNC rejected it."""
    if edge.chi2 is None:
        return False
    return edge.chi2 > GNC_CHI2_LIMIT


def is_within_gnc_limit(edge):
    """Return True when a CheckedEdge's chi2 is within GNC_CHI2_LIMIT, the sign that GNC kept it."""
    if edge.chi2 is None:
        return False
    return edge.chi2 <= GNC_CHI2_LIMIT


def translation_ape_rmse(reference, estimate):
    """Return the RMSE of evo's APE on the translation part, the rmse that evo_ape prints."""
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((reference, estimate))
    return ape.get_statistic(metrics.StatisticsType.rmse)


def ape_rmse(trajectory_path, ground_truth_poses, ground_truth_times):
    """Return the APE RMSE of a TUM trajectory in metres, as (xyz, xy, z).

    xyz is the rmse that `evo_ape --align` prints, which is the number regression_test.py reports. xy is
    the one that `evo_ape --align --project_to_plane xy` prints.
    """
    reference = PoseTrajectory3D(poses_se3=list(ground_truth_poses), timestamps=ground_truth_times)
    estimate = file_interface.read_tum_trajectory_file(trajectory_path)
    # Pair each estimated pose with the ground-truth pose nearest it in time, and drop every pose left
    # without a partner.
    reference, estimate = sync.associate_trajectories(reference, estimate, max_diff=TIME_TOLERANCE)
    # Move the estimate onto the ground truth by the rotation and translation that fit it best.
    estimate.align(reference)

    rmse_xyz = translation_ape_rmse(reference, estimate)

    # evo has no APE for a single axis, so the z error is worked out here.
    dz = estimate.positions_xyz[:, 2] - reference.positions_xyz[:, 2]
    rmse_z = np.sqrt(np.mean(dz**2))

    # Flatten both trajectories onto the xy plane, setting every z to 0. project() changes the
    # trajectories in place, so it comes after everything that needs z.
    reference.project(Plane.XY)
    estimate.project(Plane.XY)
    rmse_xy = translation_ape_rmse(reference, estimate)

    return rmse_xyz, rmse_xy, rmse_z


def format_p50_p95_max(values):
    """Return the median, 95th percentile and maximum of values as the text 'p50 / p95 / max'."""
    p50 = np.percentile(values, 50)
    p95 = np.percentile(values, 95)
    maximum = np.max(values)
    return f"{p50:.3f} / {p95:.3f} / {maximum:.3f}"


def describe_edge(edge):
    """Return a one-line summary of a CheckedEdge."""
    text = (f"{edge.keyframe_i}->{edge.keyframe_j}: "
            f"GT distance {edge.ground_truth_distance:.2f} m, "
            f"measured distance {edge.measured_distance:.2f} m, "
            f"translation error {edge.translation_error_magnitude:.3f} m "
            f"(xy {edge.translation_error_magnitude_xy:.3f} m, "
            f"z {edge.translation_error_magnitude_z:.3f} m), "
            f"rotation error {edge.rotation_error_magnitude_deg:.2f} deg")
    if edge.chi2 is not None:
        text = text + f", chi2 {edge.chi2:.1f}"
    return text


def print_sigmas_in_graph(edges):
    """Print the sigma that the information matrices of a list of CheckedEdge give each axis.

    When the edges do not all have the same information matrix, print each axis's median over the edges.
    """
    # The covariance is the inverse of the information matrix, and the square root of each diagonal entry
    # of the covariance is one axis's sigma.
    sigmas_per_edge = []
    all_information_equal = True
    for edge in edges:
        covariance = np.linalg.inv(edge.information)
        sigmas_per_edge.append(np.sqrt(np.diag(covariance)))
        if not np.array_equal(edge.information, edges[0].information):
            all_information_equal = False

    if all_information_equal:
        label = "sigmas in graph"
        sigmas = sigmas_per_edge[0]
    else:
        label = "sigmas in graph p50"
        sigmas = np.median(np.array(sigmas_per_edge), axis=0)

    translation_sigmas = sigmas[0:3]
    rotation_sigmas_deg = np.degrees(sigmas[3:6])
    print(f"     {label:<20} xyz {np.round(translation_sigmas, 4)} m, "
          f"rxryrz {np.round(rotation_sigmas_deg, 3)} deg")


def print_odometry_edges(odometry_edges):
    """Print how far the odometry edges are from the ground truth, the numbers to set their sigmas from."""
    if len(odometry_edges) == 0:
        return

    ground_truth_distances = []
    translation_error_magnitudes = []
    rotation_error_magnitudes_deg = []
    translation_error_vectors = []
    rotation_error_vectors_deg = []
    for edge in odometry_edges:
        ground_truth_distances.append(edge.ground_truth_distance)
        translation_error_magnitudes.append(edge.translation_error_magnitude)
        rotation_error_magnitudes_deg.append(edge.rotation_error_magnitude_deg)
        translation_error_vectors.append(edge.translation_error_vector)
        rotation_error_vectors_deg.append(edge.rotation_error_vector_deg)

    # One row per edge and one column per axis, so axis=0 gives the error's RMS on each axis: three
    # numbers to compare with the sigmas print_sigmas_in_graph prints. RMS and not std, because std
    # subtracts the mean error, and a noise model whose mean is zero has to cover that mean too.
    translation_error_rms_per_axis = np.sqrt(np.mean(np.array(translation_error_vectors)**2, axis=0))
    rotation_error_rms_per_axis_deg = np.sqrt(np.mean(np.array(rotation_error_vectors_deg)**2, axis=0))

    print(f"   odometry edges {len(odometry_edges)}, GT length p50 {np.median(ground_truth_distances):.2f} m")
    print_sigmas_in_graph(odometry_edges)
    print(f"     error rms per axis   xyz {np.round(translation_error_rms_per_axis, 4)} m, "
          f"rxryrz {np.round(rotation_error_rms_per_axis_deg, 3)} deg")
    print(f"     error p50/p95/max    {format_p50_p95_max(translation_error_magnitudes)} m, "
          f"{format_p50_p95_max(rotation_error_magnitudes_deg)} deg")


def print_loop_closures(loop_closure_edges, close_loop_closures, far_loop_closures, far_limits,
                        all_loop_closures):
    """Print how far the loop closures are from the ground truth, and list the ones far from it.

    loop_closure_edges holds every loop closure, in graph order. close_loop_closures and far_loop_closures
    are the same loop closures, split by far_limits, a FarFromGroundTruthLimits. all_loop_closures lists the
    close loop closures as well.
    """
    print(f"   loop closures {len(loop_closure_edges)}, "
          f"far from GT ({far_limits.describe()}) {len(far_loop_closures)}")
    if len(loop_closure_edges) > 0:
        print_sigmas_in_graph(loop_closure_edges)

    if len(close_loop_closures) > 0:
        ground_truth_distances = []
        measured_distances = []
        translation_error_magnitudes_xy = []
        translation_error_magnitudes_z = []
        rotation_error_magnitudes_deg = []
        for edge in close_loop_closures:
            ground_truth_distances.append(edge.ground_truth_distance)
            measured_distances.append(edge.measured_distance)
            translation_error_magnitudes_xy.append(edge.translation_error_magnitude_xy)
            translation_error_magnitudes_z.append(edge.translation_error_magnitude_z)
            rotation_error_magnitudes_deg.append(edge.rotation_error_magnitude_deg)
        print(f"     close loop closures, p50/p95/max: "
              f"GT distance {format_p50_p95_max(ground_truth_distances)} m, "
              f"measured {format_p50_p95_max(measured_distances)} m")
        print(f"       error xy {format_p50_p95_max(translation_error_magnitudes_xy)} m, "
              f"z {format_p50_p95_max(translation_error_magnitudes_z)} m, "
              f"rotation {format_p50_p95_max(rotation_error_magnitudes_deg)} deg")

    for edge in far_loop_closures:
        print(f"     far    {describe_edge(edge)}")
    if all_loop_closures:
        for edge in close_loop_closures:
            print(f"     close  {describe_edge(edge)}")


def print_edges_over_gnc_limit(odometry_edges, close_loop_closures, far_loop_closures):
    """Print the edges that GNC rejected, judged by their chi2 in the optimized graph.

    Also lists the loop closures far from the ground truth that GNC kept.
    """
    rejected_odometry_edges = []
    for edge in odometry_edges:
        if is_over_gnc_limit(edge):
            rejected_odometry_edges.append(edge)

    rejected_close_loop_closures = []
    for edge in close_loop_closures:
        if is_over_gnc_limit(edge):
            rejected_close_loop_closures.append(edge)

    # A loop closure with no chi2 goes in neither list.
    rejected_far_loop_closures = []
    kept_far_loop_closures = []
    for edge in far_loop_closures:
        if is_over_gnc_limit(edge):
            rejected_far_loop_closures.append(edge)
        elif is_within_gnc_limit(edge):
            kept_far_loop_closures.append(edge)

    print(f"   over the GNC limit (chi2 > {GNC_CHI2_LIMIT}): odometry edges {len(rejected_odometry_edges)}, "
          f"close loop closures {len(rejected_close_loop_closures)} of {len(close_loop_closures)}, "
          f"far loop closures {len(rejected_far_loop_closures)} of {len(far_loop_closures)}")
    for edge in rejected_odometry_edges:
        print(f"     odometry  {describe_edge(edge)}")
    for edge in rejected_close_loop_closures:
        print(f"     close loop closure rejected  {describe_edge(edge)}")
    for edge in kept_far_loop_closures:
        print(f"     far loop closure kept  {describe_edge(edge)}")


def print_keyframe_ape(run_dir, ground_truth_poses, ground_truth_times):
    """Print the APE of the unoptimized and optimized keyframe trajectories, split into xy and z."""
    print("   keyframe APE (m)          xyz      xy       z")
    for trajectory_name in ["unoptimized_keyframes_tum.txt", "optimized_keyframes_tum.txt"]:
        trajectory_path = run_dir / trajectory_name
        if trajectory_path.is_file():
            rmse_xyz, rmse_xy, rmse_z = ape_rmse(trajectory_path, ground_truth_poses, ground_truth_times)
            print(f"     {trajectory_name:<30} {rmse_xyz:7.3f}  {rmse_xy:7.3f}  {rmse_z:7.3f}")
        else:
            print(f"     {trajectory_name:<30} not found")


def check_run(run_dir, sequence, dataset_root, far_limits, all_loop_closures):
    """Check the pose graph of one sequence directory of a run against the ground truth, and print it."""
    ground_truth_path = dataset_root / "original" / "poses" / f"{sequence}.txt"
    sequence_dir = dataset_root / "original" / "sequences" / sequence
    unoptimized_graph = run_dir / "unoptimized_graph.g2o"
    optimized_graph = run_dir / "optimized_graph.g2o"
    stamps_path = run_dir / "keyframe_stamps.txt"

    # optimized_graph.g2o is not required: the node writes it only when an optimize has run.
    required_files = [
        ground_truth_path,
        sequence_dir / "times.txt",
        sequence_dir / "calib.txt",
        unoptimized_graph,
        stamps_path,
    ]
    for path in required_files:
        if not path.is_file():
            print(f"== {run_dir}: {path} not found, skipped")
            return

    try:
        ground_truth_poses, ground_truth_times = read_ground_truth(ground_truth_path, sequence_dir)
        ground_truth_lines = find_ground_truth_lines(stamps_path, ground_truth_times)
    except ValueError as error:
        print(f"== {run_dir}: skipped: {error}")
        return
    if len(ground_truth_lines) == 0:
        print(f"== {run_dir}: no keyframes in {stamps_path}, skipped")
        return

    # Both .g2o files are written from the same graph, so they hold the same edges and differ only in the
    # vertex poses. The edges are read from the unoptimized graph, which is always written.
    keyframe_count = len(read_g2o_vertices(unoptimized_graph))
    edges = read_g2o_edges(unoptimized_graph)
    optimized_poses = None
    if optimized_graph.is_file():
        optimized_poses = read_g2o_vertices(optimized_graph)

    print(f"== {run_dir} (sequence {sequence})")
    if optimized_poses is None:
        print(f"   keyframes {keyframe_count}, optimized none")
    elif len(optimized_poses) < keyframe_count:
        print(f"   keyframes {keyframe_count}, optimized {len(optimized_poses)}"
              f"   <- optimized graph is missing keyframes")
    else:
        print(f"   keyframes {keyframe_count}, optimized {len(optimized_poses)}")

    # Compare every edge with the ground truth. The odometry edges join each keyframe to the next one,
    # and every other edge is a loop closure.
    odometry_edges = []
    loop_closure_edges = []
    for edge in edges:
        true_pose_i = ground_truth_poses[ground_truth_lines[edge.keyframe_i]]
        true_pose_j = ground_truth_poses[ground_truth_lines[edge.keyframe_j]]
        checked_edge = check_edge(edge, true_pose_i, true_pose_j, optimized_poses)
        if edge.keyframe_j == edge.keyframe_i + 1:
            odometry_edges.append(checked_edge)
        else:
            loop_closure_edges.append(checked_edge)

    close_loop_closures = []
    far_loop_closures = []
    for edge in loop_closure_edges:
        if far_limits.is_far(edge.translation_error_magnitude, edge.rotation_error_magnitude_deg):
            far_loop_closures.append(edge)
        else:
            close_loop_closures.append(edge)

    print_odometry_edges(odometry_edges)
    print_loop_closures(loop_closure_edges, close_loop_closures, far_loop_closures, far_limits,
                        all_loop_closures)
    if optimized_poses is not None:
        print_edges_over_gnc_limit(odometry_edges, close_loop_closures, far_loop_closures)
    print_keyframe_ape(run_dir, ground_truth_poses, ground_truth_times)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", type=pathlib.Path, nargs="+",
                        help="sequence directories of a run, e.g. <regression_runs>/<timestamp>/07")
    parser.add_argument("--sequence",
                        help="KITTI sequence number (default: the name of each run directory)")
    parser.add_argument("--dataset-root", type=pathlib.Path,
                        default=pathlib.Path.home() / "projects" / "datasets" / "kitti",
                        help="KITTI dataset root, holding original/poses and original/sequences")
    parser.add_argument("--far-translation-error", type=float, default=8.0,
                        help="count a loop closure as far from the ground truth when its translation error "
                             "is over this (m, default: 8)")
    parser.add_argument("--far-rotation-error", type=float, default=3.0,
                        help="count a loop closure as far from the ground truth when its rotation error is "
                             "over this (degrees, default: 3)")
    parser.add_argument("--all-loop-closures", action="store_true",
                        help="print every loop closure, not only the ones far from the ground truth")
    args = parser.parse_args()
    far_limits = FarFromGroundTruthLimits(translation_error=args.far_translation_error,
                                          rotation_error_deg=args.far_rotation_error)

    for run_dir in args.run_dirs:
        if not run_dir.is_dir():
            print(f"== {run_dir}: not a directory, skipped")
            continue
        check_run(run_dir, args.sequence or run_dir.name, args.dataset_root, far_limits,
                  args.all_loop_closures)


if __name__ == "__main__":
    main()
