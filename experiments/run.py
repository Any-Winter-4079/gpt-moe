import argparse
from datetime import datetime
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ORDER = ["baseline", "012", "014", "017", "000", "001", "018", "002", "003", "005", "019", "020", "021", "009", "013", "015", "016", "004", "010", "006", "011", "007", "008"]
DEPENDENCIES = {
    "009": ["005"],
    "010": ["004"],
    "013": ["005", "009"],
    "014": ["012"],
    "015": ["005", "009", "013"],
    "016": ["005", "009", "013", "015"],
    "017": ["012", "014"],
    "018": ["000"],
    "019": ["005"],
    "020": ["005"],
    "021": ["005"],
}


def run_command(command, log):
    log.write(f"\n$ {shlex.join(command)}\n")
    log.flush()
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        return process.wait()
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise


def stop(signum, frame):
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser(description="Apply experiments, train, upload to the Hub and reverse the patches")
    parser.add_argument("experiments", nargs="*", metavar="ID", help="experiment IDs in execution order; baseline always runs first; omitted runs all experiments")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="number of training GPUs (default: 1)")
    args = parser.parse_args()
    if any(experiment not in ORDER for experiment in args.experiments):
        parser.error(f"experiment IDs must be one of: {', '.join(ORDER)}")

    os.chdir(ROOT)
    if not (os.environ.get("hf_user") or os.environ.get("HF_USER")):
        parser.error("set hf_user or HF_USER before starting")
    if not (os.environ.get("hf_token") or os.environ.get("HF_TOKEN")):
        parser.error("set hf_token or HF_TOKEN before starting")
    if args.nproc_per_node < 1:
        parser.error("--nproc-per-node must be positive")

    signal.signal(signal.SIGTERM, stop)
    batch = datetime.now().strftime("%Y%m%d_%H%M%S")
    patches = {path.name[:3]: path for path in (ROOT / "experiments").glob("*.patch")}
    runs_dir = ROOT / "configs_and_logs"
    runs_dir.mkdir(exist_ok=True)
    summary_path = ROOT / f"experiments-{batch}.log"
    failed = False

    print(f"summary: {summary_path}", flush=True)
    with summary_path.open("x") as summary:
        summary.write("experiment\ttrain_exit\tupload_exit\ttimestamp\n")
        summary.flush()
        experiments = ["baseline"] + [experiment for experiment in (args.experiments or ORDER) if experiment != "baseline"]
        for index, experiment in enumerate(experiments, 1):
            applied = []
            console_path = ROOT / f"experiment-{experiment}-{batch}-{index:02d}.log"
            print(f"experiment {experiment}; training output: {console_path}", flush=True)
            try:
                with console_path.open("x") as console:
                    patch_ids = [] if experiment == "baseline" else DEPENDENCIES.get(experiment, []) + [experiment]
                    for patch_id in patch_ids:
                        patch = patches[patch_id]
                        if run_command(["git", "apply", "--check", str(patch)], console):
                            break
                        # finish applying and recording a patch before handling an interrupt
                        previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
                        try:
                            result = run_command(["git", "apply", str(patch)], console)
                            if result == 0:
                                applied.append(patch)
                        finally:
                            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
                        if result:
                            break
                    else:
                        before = {path.name for path in runs_dir.iterdir() if path.is_dir()}
                        train_exit = run_command([
                            "torchrun", "--standalone", f"--nproc_per_node={args.nproc_per_node}", "train.py",
                        ], console)
                        console.write(f"\ntraining exit code: {train_exit}\n")
                        created = {path.name for path in runs_dir.iterdir() if path.is_dir()} - before
                        if len(created) > 1:
                            raise RuntimeError("multiple new run directories; cannot identify this experiment's results")
                        timestamp = next(iter(created)) if created else f"{batch}_exp{experiment}_{index:02d}_failed"
                        run_dir = runs_dir / timestamp
                        if not created:
                            run_dir.mkdir()
                        console.flush()
                        console_path.rename(run_dir / "console.log")

                        upload_path = ROOT / f"experiment-{experiment}-{batch}-{index:02d}-upload.log"
                        print(f"experiment {experiment}: training exit {train_exit}; uploading {timestamp}", flush=True)
                        with upload_path.open("x") as upload_log:
                            upload_exit = run_command([
                                sys.executable, "push_to_hub.py", "--timestamp", timestamp,
                                "--repo-prefix", "nanogpt-baseline" if experiment == "baseline" else f"nanogpt-exp{experiment}",
                            ], upload_log)
                        summary.write(f"{experiment}\t{train_exit}\t{upload_exit}\t{timestamp}\n")
                        summary.flush()
                        failed = failed or train_exit != 0 or upload_exit != 0
                        print(f"experiment {experiment}: upload exit {upload_exit}; upload output: {upload_path}", flush=True)
                        continue

                failed = True
                summary.write(f"{experiment}\tpatch_failed\tnot_attempted\t-\n")
                summary.flush()
                print(f"experiment {experiment}: patch failed; see {console_path}", flush=True)
            finally:
                # keep cleanup running until the applied patch stack is fully reversed
                previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
                try:
                    for patch in reversed(applied):
                        subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], check=True)
                        subprocess.run(["git", "apply", "--reverse", str(patch)], check=True)
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous)

    print(f"finished; summary: {summary_path}", flush=True)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
