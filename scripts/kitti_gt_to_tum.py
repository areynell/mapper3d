#!/usr/bin/env python3
"""Convert KITTI odometry ground truth into a TUM trajectory in the velodyne frame.

The KITTI benchmark includes ground truth as `poses/NN.txt`: one line per scan, twelve numbers
forming the 3x4 pose of the *left camera*, and no timestamps. mapper3d writes its trajectories
in the *velodyne* frame, stamped with the scan time. Two things therefore have to happen before
evo can compare them:

  * Frame: G_velo = Tr^-1 @ G_cam @ Tr, with Tr (velodyne -> left camera) taken from `calib.txt`.
    Skipping this leaves a fixed rigid offset that `--align` only partly masks.
  * Time: Line i of `poses/NN.txt` corresponds to line i of `times.txt`. That is what lets evo
    associate by timestamp rather than by line number - the only association that survives if a scan is dropped.

Usage:
    kitti_gt_to_tum.py --gt <poses>/NN.txt --sequence-dir <sequences>/NN -o gt_NN_tum.txt
"""

import argparse
import pathlib
import sys

import numpy as np


def read_calib_tr(calib_path):
    """Return Tr (velodyne -> left camera) from calib.txt as a 4x4 matrix."""
    for line in calib_path.read_text().splitlines():
        if line.startswith("Tr:"):
            values = np.fromstring(line[3:], sep=" ")
            if values.size != 12:
                raise ValueError(f"{calib_path}: Tr has {values.size} numbers, expected 12")
            transform = np.eye(4)
            transform[:3, :4] = values.reshape(3, 4)
            return transform
    raise ValueError(f"{calib_path}: no 'Tr:' line found")


def read_poses(poses_path):
    """Return the ground-truth camera poses as an (N, 4, 4) array."""
    rows = np.loadtxt(poses_path)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.shape[1] != 12:
        raise ValueError(f"{poses_path}: {rows.shape[1]} numbers per line, expected 12")

    poses = np.tile(np.eye(4), (rows.shape[0], 1, 1))
    poses[:, :3, :4] = rows.reshape(-1, 3, 4)
    return poses


def read_ground_truth(gt_path, sequence_dir, camera_frame=False):
    """Return the ground-truth poses as an (N, 4, 4) array, and their timestamps from times.txt.

    Line i of the poses file takes the timestamp on line i of times.txt. The poses are moved into the
    velodyne frame (G_velo = Tr^-1 @ G_cam @ Tr) unless camera_frame is set.
    """
    poses = read_poses(gt_path)
    times_path = sequence_dir / "times.txt"
    times = np.atleast_1d(np.loadtxt(times_path))
    if poses.shape[0] != times.size:
        raise ValueError(f"{gt_path} has {poses.shape[0]} poses but {times_path} has {times.size} "
                         f"timestamps - are they the same sequence?")

    if not camera_frame:
        tr = read_calib_tr(sequence_dir / "calib.txt")
        poses = np.linalg.inv(tr) @ poses @ tr
    return poses, times


def rotation_to_quaternion(rotation):
    """Convert a 3x3 rotation matrix to [x, y, z, w] using Shepperd's method.

    Uses a numerically stable branch based on the largest quaternion
    component to avoid dividing by a value close to zero.
    """

    trace = np.trace(rotation)

    # If the trace is positive, w is the safest component to compute first.
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    # Otherwise choose the diagonal entry corresponding to the
    # largest quaternion component for better numerical stability.
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s

    quaternion = np.array([x, y, z, w])
    # Normalize to compensate for small numerical errors.
    return quaternion / np.linalg.norm(quaternion)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", type=pathlib.Path, required=True,
                        help="KITTI ground-truth poses file, e.g. <poses>/NN.txt")
    parser.add_argument("--sequence-dir", type=pathlib.Path, required=True,
                        help="sequence directory holding calib.txt and times.txt")
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True,
                        help="TUM trajectory to write")
    parser.add_argument("--camera-frame", action="store_true",
                        help="skip the Tr transform and leave the poses in the left-camera frame")
    args = parser.parse_args()

    times_path = args.sequence_dir / "times.txt"
    calib_path = args.sequence_dir / "calib.txt"
    for path in (args.gt, times_path, calib_path):
        if not path.is_file():
            sys.exit(f"error: {path} not found")

    try:
        poses, times = read_ground_truth(args.gt, args.sequence_dir, args.camera_frame)
    except ValueError as error:
        sys.exit(f"error: {error}")

    frame = "left camera" if args.camera_frame else "velodyne"
    with args.output.open("w") as out:
        out.write(f"# KITTI ground truth from {args.gt.name}, {frame} frame\n")
        out.write("# timestamp tx ty tz qx qy qz qw\n")
        for timestamp, pose in zip(times, poses):
            tx, ty, tz = pose[:3, 3]
            qx, qy, qz, qw = rotation_to_quaternion(pose[:3, :3])
            out.write(f"{timestamp:.9f} {tx:.9f} {ty:.9f} {tz:.9f} "
                      f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")

    print(f"Wrote {poses.shape[0]} poses to {args.output} ({frame} frame)")


if __name__ == "__main__":
    main()
