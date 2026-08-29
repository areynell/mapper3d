#!/usr/bin/env python3
"""Run the KITTI sequences through mapper3d and check for regressions against a saved baseline.

For each sequence the regression test starts `mapper3d_node` in its own directory, plays the matching bag,
records how much CPU and memory the node used, and scores the lidar odometry and optimized keyframe trajectories against
KITTI ground truth with `evo_ape`. The scores are compared against a baseline score computed from a known
good commit, and the run fails if any metric is more than THRESHOLD worse.

Usage:
    regression_test.py                          # score every sequence against the baseline
    regression_test.py --sequences 07           # score one sequence only

To refresh the baseline, run this normally, check the results, and then manually copy
the run's results.json over test/regression_baseline.json.
"""

import argparse
import datetime
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
import zipfile

from evo import EvoException
from evo.core import sync
from evo.tools import file_interface

import matplotlib
# Chosen before pyplot is imported, so the run never depends on a display being present.
# Do NOT import evo.tools.plot anywhere in this file: merely importing it re-runs evo's own
# mpl.use() with an interactive backend (qtagg/TkAgg by default) and undoes this, plus applies
# seaborn styling and rcParam changes process-wide. evo.tools.file_interface and evo.core.sync,
# used below, do not touch matplotlib at all.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_PATH = pathlib.Path(__file__).resolve()
PACKAGE_DIR = SCRIPT_PATH.parents[1]
WORKSPACE_DIR = SCRIPT_PATH.parents[3]

# KITTI sequences 00-10, as used in the KITTI odometry benchmark, since they have accompanying ground truth data.
ALL_SEQUENCES = [f"{index:02d}" for index in range(11)]
POINT_CLOUD_TOPIC = "/velodyne_points"
# One entry per file to grade: a short name, the file to read, and whether a missing file should
# fail the sequence. optimized_keyframes_tum.txt is allowed to be missing, since it is only written
# when a loop closure was found.
SCORED_TRAJECTORIES = [
    ("odometry", "lidar_odometry_tum.txt", True),
    ("optimized", "optimized_keyframes_tum.txt", False),
]

# The two scores compare() checks against the baseline. A sequence counts as regressed if either
# score gets worse than THRESHOLD.
COMPARED_METRICS = ["odometry_ape_rmse", "optimized_ape_rmse"]
CLOCK_TICKS_PER_SECOND = os.sysconf("SC_CLK_TCK")
GT_CONVERTER = PACKAGE_DIR / "scripts" / "kitti_gt_to_tum.py"
NODE_BINARY = WORKSPACE_DIR / "install/mapper3d/lib/mapper3d/mapper3d_node"

# The branch a baseline is expected to come from.
MAIN_BRANCH = "master"

# How much worse than the baseline a sequence may score before it fails.
THRESHOLD = 0.20

PLAYBACK_RATE = 1.0
STARTUP_TIMEOUT = 60.0
SHUTDOWN_TIMEOUT = 300.0

# Timestamp association tolerance, for both scoring and plotting.
TIME_TOLERANCE = 0.01

def resolve_inputs(sequence, dataset_root):
    """Return the bag, ground truth and sequence directory for a sequence, plus any missing paths."""
    inputs = {
        "bag": dataset_root / "ros2" / f"kitti_sequence_{sequence}",
        "ground_truth": dataset_root / "original" / "poses" / f"{sequence}.txt",
        "sequence_dir": dataset_root / "original" / "sequences" / sequence,
    }
    missing = [str(path) for path in inputs.values() if not path.exists()]
    return inputs, missing

def wait_for_subscriber(node, timeout):
    """Wait until the node has subscribed to the point cloud topic. Returns True on success.

    The node takes a while to come up, and anything published before it subscribes is lost.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if node.poll() is not None:
            return False

        # Ask the ROS graph who is attached to the topic. The node's own subscription is what shows
        # up here, so no publisher has to exist yet.
        result = subprocess.run(
            ["ros2", "topic", "info", POINT_CLOUD_TOPIC],
            capture_output=True,
            text=True,
        )
        # 'ros2 topic info' prints a "Subscription count: N" line. A non-zero count means the node
        # has finished starting and is listening, so the bag can be played without losing scans.
        match = re.search(r"Subscription count:\s*(\d+)", result.stdout)
        if match and int(match.group(1)) > 0:
            return True

        time.sleep(1.0)

    return False

def sample_resource_usage(pid):
    """Return the peak resident memory (MB) and CPU time (s) of a running process.

    Read from /proc while the process is still alive - it disappears on shutdown.
    """
    # Left as None if anything below cannot be read, so a missing number never fails the run.
    resource_usage = {"peak_rss_mb": None, "cpu_seconds": None}
    try:
        # VmHWM is the high water mark: the most memory the process has ever held, in kB.
        status = pathlib.Path(f"/proc/{pid}/status").read_text()
        match = re.search(r"^VmHWM:\s+(\d+) kB", status, re.MULTILINE)
        if match:
            resource_usage["peak_rss_mb"] = round(int(match.group(1)) / 1024.0, 1)

        # /proc/<pid>/stat is one line of numbered fields. Fields 14 and 15 are utime and stime, in
        # clock ticks: time spent in the process's own code and in the kernel on its behalf, summed
        # over all its threads.
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text()
        # Field 2 is the command name in parentheses and may itself contain spaces, which would
        # throw the field numbering off. Skip past its closing parenthesis before splitting.
        # That drops fields 1 and 2, so field N is at index N - 3: utime at 11, stime at 12.
        fields = stat[stat.rindex(")") + 2 :].split()
        ticks = int(fields[11]) + int(fields[12])
        resource_usage["cpu_seconds"] = round(ticks / CLOCK_TICKS_PER_SECOND, 1)
    except (OSError, ValueError, IndexError):
        # The process can exit part way through the read, which empties /proc/<pid>.
        pass

    return resource_usage

def shut_down_node(node, timeout):
    """Send SIGINT to the node and wait for it to write its results. Returns True on a clean exit."""
    try:
        # SIGINT is the "save and quit" signal, which the node handles by writing its trajectories and shutting down cleanly.
        os.killpg(os.getpgid(node.pid), signal.SIGINT)
    except ProcessLookupError:
        # The node has already exited.
        return node.poll() == 0

    try:
        # Wait for the shutdown work: a final optimization, then the trajectory and map writes.
        node.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        # A kill leaves half-written files, so the caller scores this as a failed run rather than
        # trusting whatever reached disk.
        print(f"  node did not exit within {timeout:.0f}s, killing it - its output is incomplete")
        node.kill()
        node.wait()
        return False

def last_line(output):
    """Return the last non-blank line of a command's output, for use in an error message."""
    lines = output.strip().splitlines()
    if not lines:
        return ""
    return lines[-1].strip()

def count_poses(path):
    """Return the number of poses in a TUM trajectory, ignoring the comment header."""
    count = 0
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue  # skip blank lines and comments
            count += 1
    return count

def run_evo_ape(run_dir, filename, results_zip):
    """Score one trajectory file against ground truth. Returns (rmse, error message)."""
    command = ["evo_ape", "tum", "gt.txt", filename, "--align", "--t_max_diff", str(TIME_TOLERANCE), "--save_results", results_zip, "--silent", "--no_warnings",]
    result = subprocess.run(command, cwd=run_dir, capture_output=True, text=True)
    if result.returncode != 0:
        # Errors usually go to stderr, but some tools print them to stdout instead.
        message = last_line(result.stderr)
        if not message:
            message = last_line(result.stdout)
        if not message:
            message = "no output"
        return None, message

    with zipfile.ZipFile(run_dir / results_zip) as archive:
        stats = json.loads(archive.read("stats.json"))
    return round(stats["rmse"], 4), None

def score_trajectories(run_dir):
    """Score the front-end and back-end trajectories against ground truth.

    Each one contributes a `<name>_ape_rmse` and a `<name>_poses` field.
    """
    result = {"status": "ok"}

    for name, filename, required in SCORED_TRAJECTORIES:
        trajectory = run_dir / filename
        if not trajectory.is_file():
            if required:
                return {"status": "missing_trajectory", "detail": f"{filename} was not written"}
            result.setdefault("notes", []).append(f"{filename} was not written")
            continue

        rmse, error = run_evo_ape(run_dir, filename, f"{name}_ape.zip")
        if error is not None:
            return {"status": "scoring_failed", "detail": f"{filename}: {error}"}

        result[f"{name}_ape_rmse"] = rmse
        result[f"{name}_poses"] = count_poses(trajectory)

    return result

def plot_trajectories(run_dir, sequence, result):
    """Draw the front-end and back-end trajectories over ground truth. Returns an error message or None.

    Reads and aligns trajectories with evo's own file_interface, sync and PosePath3D.align - the
    exact same functions `evo_ape tum gt.txt <filename> --align --t_max_diff TIME_TOLERANCE` runs
    internally, so the plotted pose pairs and alignment match what the reported APE was computed
    from exactly.
    """
    figure = None
    try:
        gt_traj = file_interface.read_tum_trajectory_file(run_dir / "gt.txt")
        gt_positions = gt_traj.positions_xyz

        figure, axes = plt.subplots(figsize=(9, 9))
        axes.plot(gt_positions[:, 0], gt_positions[:, 1], "--", color="0.4", linewidth=1.5, label="ground truth")

        for (name, filename, _), colour in zip(SCORED_TRAJECTORIES, ["tab:blue", "tab:green"]):
            trajectory_path = run_dir / filename
            # Skip plotting if the file is not found (e.g., optimized_keyframes_tum.txt is not written when no loop closures are found, since no optimization is run).
            if not trajectory_path.is_file():
                continue

            estimate_traj = file_interface.read_tum_trajectory_file(trajectory_path)
            try:
                matched_gt, matched_estimate = sync.associate_trajectories(gt_traj, estimate_traj, TIME_TOLERANCE, first_name="gt.txt", snd_name=filename)
                # Align the estimate to the ground truth.
                matched_estimate.align(matched_gt, correct_scale=False)
            except EvoException as error:
                result.setdefault("notes", []).append(f"{filename}: not plotted, could not align ({error})")
                continue

            rmse = result.get(f"{name}_ape_rmse")
            if rmse is not None:
                label = f"{name} (APE {rmse:.2f} m)"
            else:
                label = name
            aligned_positions = matched_estimate.positions_xyz
            axes.plot(aligned_positions[:, 0], aligned_positions[:, 1], color=colour, linewidth=1.2, label=label)

        axes.plot(gt_positions[0, 0], gt_positions[0, 1], "ko", markersize=7, label="start")

        axes.set_title(f"KITTI sequence {sequence} - trajectories aligned to ground truth")
        axes.set_xlabel("x (m)")
        axes.set_ylabel("y (m)")
        axes.set_aspect("equal", adjustable="datalim")
        axes.grid(True, linewidth=0.4, alpha=0.5)
        axes.legend(loc="best")

        figure.tight_layout()
        figure.savefig(run_dir / "trajectories.png", dpi=150)
    except Exception as error:
        return f"plotting failed: {error}"
    finally:
        if figure is not None:
            plt.close(figure)

    return None

def run_sequence(sequence, run_dir, inputs, args):
    """Run one sequence end to end and return its result fields."""
    run_dir.mkdir(parents=True, exist_ok=True)

    convert = subprocess.run(
        [sys.executable, str(GT_CONVERTER), "--gt", str(inputs["ground_truth"]),
         "--sequence-dir", str(inputs["sequence_dir"]), "-o", "gt.txt"],
        cwd=run_dir, capture_output=True, text=True)
    if convert.returncode != 0:
        return {"status": "ground_truth_failed", "detail": convert.stderr.strip()}

    started = time.monotonic()
    # Non-blocking call to start the node so the bag can be played with it listening.
    node = subprocess.Popen(
        [str(NODE_BINARY)],
        cwd=run_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    # Play the bag into the node, then stop it cleanly. Any failure along the way returns a status
    # instead of a score, so a partial run is never mistaken for a good one. The `finally` covers the
    # early returns below it, which would otherwise leave the node running.
    try:
        # Wait for the node to subscribe to the point cloud topic before playing the bag, or it will miss scans.
        if not wait_for_subscriber(node, STARTUP_TIMEOUT):
            if node.poll() is not None:
                reason = "the node exited immediately"
            else:
                reason = (f"the node did not subscribe to {POINT_CLOUD_TOPIC} "
                          f"within {STARTUP_TIMEOUT:.0f}s")
            return {"status": "startup_failed", "detail": reason}

        print(f"  playing {inputs['bag'].name}")
        # Blocking call, that lasts until the bag is done playing. The node may exit part way through, which is caught below.
        playback = subprocess.run(
            ["ros2", "bag", "play", str(inputs["bag"]), "--rate", str(PLAYBACK_RATE)]
        )
        if playback.returncode != 0:
            return {
                "status": "playback_failed",
                "detail": f"'ros2 bag play' exited with code {playback.returncode}",
            }

        # The bag finishing does not mean the node survived it - it may have crashed part way
        # through, leaving a trajectory that covers only some of the sequence.
        if node.poll() is not None:
            return {"status": "node_died", "detail": "the node exited during playback"}

        # Sample resource usage before shutting down, since /proc/<pid> is gone once the node exits.
        resource_usage = sample_resource_usage(node.pid)
        # Shutdown the node cleanly, so it writes its trajectories and map. If it does not exit in time, kill it and report a failure.
        clean_exit = shut_down_node(node, SHUTDOWN_TIMEOUT)
    finally:
        # Never leave a node running: the next sequence would find the topic already subscribed and
        # play its bag into both nodes.
        if node.poll() is None:
            node.kill()
            node.wait()

    if not clean_exit:
        return {"status": "shutdown_failed", "detail": "the node had to be killed, so its output is truncated"}

    result = score_trajectories(run_dir)
    result.update(resource_usage)
    result["wall_seconds"] = round(time.monotonic() - started, 1)
    # CPU time is only meaningful here as a rate: 100% is one core busy for the whole run.
    cpu_seconds = result.pop("cpu_seconds", None)
    if cpu_seconds and result["wall_seconds"]:
        result["cpu_percent"] = round(cpu_seconds / result["wall_seconds"] * 100, 1)

    if result["status"] == "ok":
        error = plot_trajectories(run_dir, sequence, result)
        if error is not None:
            result.setdefault("notes", []).append(error)

    return result

def git_output(*arguments):
    """Return the output of a git command, or None if it fails."""
    result = subprocess.run(["git", *arguments], cwd=PACKAGE_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip()

def describe_current_commit():
    """Return the commit the results were produced from."""
    return {
        "git_sha": git_output("rev-parse", "HEAD"),
        "branch": git_output("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git_output("status", "--porcelain")),
    }

def compare(results, baseline):
    """Print a comparison table and return the sequences that regressed.

    A sequence the baseline recorded as "ok" has regressed when any one of these is true:
      1. Failed:       A run was attempted but did not finish cleanly (node_died, startup_failed, ...).
      2. Metric gone:  A run finished, but a COMPARED_METRICS entry was not scored - usually
                       optimized_ape_rmse, which needs a loop closure to exist at all.
      3. Metric worse: A run finished, but a COMPARED_METRICS entry became worse by more than THRESHOLD.

    A sequence that was not run at all (e.g. --sequences left it out) is not a regression: compare()
    prints NOT RUN and carries on.
    """

    # Warn if the baseline came from an older commit than master's current tip.
    baseline_sha = baseline.get("git_sha")
    main_sha = git_output("rev-parse", MAIN_BRANCH)
    if main_sha and baseline_sha != main_sha:
        print()
        print("=" * 96)
        print(f"WARNING: the baseline is not from the head of {MAIN_BRANCH}.")
        print(f"  baseline:        {baseline_sha}")
        print(f"  {MAIN_BRANCH} head: {main_sha}")
        print("=" * 96)

    print()
    print(f"Comparison against {baseline.get('created', 'unknown date')} "
          f"({(baseline_sha or 'unknown')[:8]}), threshold {THRESHOLD:.0%}")
    print()
    print(f"{'seq':<5}{'metric':<22}{'baseline':>12}{'current':>12}{'change':>10}  verdict")
    print("-" * 96)

    regressed = []
    for sequence, baseline_run in sorted(baseline.get("sequences", {}).items()):
        if baseline_run.get("status") != "ok":
            continue

        current_run = results["sequences"].get(sequence)

        # This sequence has no result in this run (e.g. it was not added to --sequences), so there is
        # nothing to compare it to. Not a regression: skip it and move to the next sequence.
        if current_run is None:
            print(f"{sequence:<5}{'-':<22}{'-':>12}{'-':>12}{'-':>10}  NOT RUN")
            continue

        # The baseline run passed, but this run did not (e.g. the node crashed). That is a
        # regression on its own, so record it and skip the per-metric checks below.
        if current_run.get("status") != "ok":
            print(f"{sequence:<5}{'-':<22}{'-':>12}{'-':>12}{'-':>10}  FAIL ({current_run['status']})")
            regressed.append(sequence)
            continue

        for metric in COMPARED_METRICS:
            baseline_value, current_value = baseline_run.get(metric), current_run.get(metric)
            # No baseline number to compare against (or a bad one), so this metric can't regress.
            if baseline_value is None or baseline_value <= 0:
                continue

            if current_value is None:
                # The baseline scored this but the current run did not.
                print(f"{sequence:<5}{metric:<22}{baseline_value:>12.2f}{'missing':>12}{'-':>10}  FAIL")
                regressed.append(sequence)
                continue

            # How much worse (positive) or better (negative) the new value is, as a fraction of
            # the baseline. Only "worse than THRESHOLD" counts as a regression.
            value_change = (current_value - baseline_value) / baseline_value
            failed = value_change > THRESHOLD
            if failed:
                regressed.append(sequence)
            print(f"{sequence:<5}{metric:<22}{baseline_value:>12.2f}{current_value:>12.2f}{value_change:>9.1%}  "
                  f"{'FAIL' if failed else 'ok'}")

    print("-" * 96)
    # A sequence can land in `regressed` twice (e.g. both metrics failed), so drop duplicates
    # before returning the list of sequence names.
    return sorted(set(regressed))

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sequences", nargs="+", default=ALL_SEQUENCES,
                        help=f"sequences to run (default: {' '.join(ALL_SEQUENCES)})")
    parser.add_argument("--baseline", type=pathlib.Path,
                        default=PACKAGE_DIR / "test" / "regression_baseline.json",
                        help="baseline results file to compare against")
    parser.add_argument("--out-root", type=pathlib.Path,
                        default=WORKSPACE_DIR / "regression_runs",
                        help="directory to write run outputs into")
    parser.add_argument("--dataset-root", type=pathlib.Path,
                        default=pathlib.Path.home() / "projects" / "datasets" / "kitti",
                        help="KITTI dataset root, holding ros2/ and original/")
    return parser.parse_args()

def print_sequence_result(sequence, result, out_dir):
    """Print the per-sequence summary lines that run_all_sequences prints after each sequence."""
    if result["status"] != "ok":
        print(f"  {result['status']}: {result.get('detail', '')}")
        return

    for name, _, _ in SCORED_TRAJECTORIES:
        rmse = result.get(f"{name}_ape_rmse")
        if rmse is None:
            print(f"  {name + ' APE':<14} not written")
        else:
            print(f"  {name + ' APE':<14} {rmse:.3f} m over {result[f'{name}_poses']} poses")
    print(f"  {'resources':<14} peak RSS {result['peak_rss_mb']} MB, "
          f"CPU {result.get('cpu_percent', 0):.0f}%")
    plot = out_dir / sequence / "trajectories.png"
    if plot.is_file():
        print(f"  {'plot':<14} {plot}")
    for note in result.get("notes", []):
        print(f"  {'note':<14} {note}")

def run_all_sequences(args, out_dir, results):
    """Run every requested sequence and fill in results["sequences"] as it goes."""
    for sequence in args.sequences:
        print(f"\n=== sequence {sequence} ===")
        inputs, missing = resolve_inputs(sequence, args.dataset_root)
        if missing:
            print(f"  unavailable: {', '.join(missing)}")
            results["sequences"][sequence] = {"status": "unavailable", "missing": missing}
            continue

        result = run_sequence(sequence, out_dir / sequence, inputs, args)
        results["sequences"][sequence] = result
        print_sequence_result(sequence, result, out_dir)

def check_against_baseline(args, results):
    """Compare results against the saved baseline and report unavailable/regressed/failed sequences."""
    if not args.baseline.is_file():
        print("\nerror: no baseline at " + str(args.baseline) +
              " - run this once, then copy its results.json over test/regression_baseline.json",
              file=sys.stderr)
        return 1

    baseline = json.loads(args.baseline.read_text())
    regressed = compare(results, baseline)

    # Sort each sequence into  "unavailable" (no bag for it) vs. "failed" (ran bag, but program output was not okay, e.g. node crashed).
    unavailable = []
    failed = []
    for name in results["sequences"]:
        run = results["sequences"][name]
        status = run["status"]
        if status == "unavailable":
            unavailable.append(name)
        elif status != "ok":
            failed.append(name)

    if len(regressed) > 0:
        print("REGRESSION in " + ", ".join(regressed) + ".")
        return 1

    if len(failed) > 0:
        print("Runs failed for " + ", ".join(sorted(failed)) + ".")
        return 1

    if len(unavailable) > 0:
        print("No data for " + ", ".join(sorted(unavailable)) + ".")
        return 2

    print("All sequences within threshold.")
    return 0

def main():
    args = parse_arguments()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    results = describe_current_commit()
    results["created"] = datetime.datetime.now().isoformat(timespec="seconds")
    results["rate"] = PLAYBACK_RATE
    results["sequences"] = {}

    run_all_sequences(args, out_dir, results)

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults written to {results_path}")

    # Score this run against the saved baseline. To update the baseline,
    # manually copy this run's results.json over test/regression_baseline.json.
    return check_against_baseline(args, results)

if __name__ == "__main__":
    sys.exit(main())
