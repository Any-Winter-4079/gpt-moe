import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ORDER = ["baseline", "012", "014", "017", "000", "001", "018", "002", "003", "005", "019", "020", "022", "023", "021", "009", "013", "015", "016", "004", "010", "006", "011", "007", "008"]
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
    "022": ["005", "019"],
    "023": ["005"],
}
COMBINATIONS = {
    "5+20+11": ["005", "020", "011"],
    "5+20+11+6": ["005", "020", "011", "006"],
    "5+20+11+12": ["024"],
    "5+20+11+6+12": ["025"],
    "5+20+11+12+19": ["026"],
    "5+20+11+12+19+21": ["027"],
    "5+20+11+12+19+21+exact-swa": ["027", "036"],
    "5+20+11+12+19+21+exact-swa+fa4": ["027", "036", "037"],
    "5+20+11+12+19+21+fa4-device": ["027", "036", "037", "046"],
    "5+20+11+12+19+21+mla": ["027", "038"],
    "5+20+11+12+19+21+qk-rope": ["027", "039"],
    "5+20+11+12+19+21+exact-swa+flex-fa4": ["027", "036", "041"],
    "5+20+11+12+19+21+attention": ["027", "028"],
    "5+20+11+12+19+21+attention-fused": ["027", "029"],
    "5+20+11+12+19+21+attention-dg": ["027", "029", "030"],
    "5+20+11+12+19+21+attention-mx": ["027", "029", "030", "035"],
    "5+20+11+12+19+21+attention-cuda": ["027", "029", "030", "040"],
    "5+20+11+12+19+21+attention-wide": ["027", "029", "030", "040", "050"],
    "5+20+11+12+19+21+qkv-epilogue": ["027", "029", "030", "040", "045"],
    "5+20+11+12+19+21+qkv-backquant": ["027", "029", "030", "044"],
    "5+20+11+12+19+21+attention-dw": ["027", "029", "030", "047"],
    "5+20+11+12+19+21+attention-dx16": ["027", "029", "030", "031"],
    "5+20+11+12+19+21+attention-pipe": ["027", "029", "032"],
    "5+20+11+12+19+21+attention-qkv": ["027", "029", "030", "033"],
    "5+20+11+12+19+21+attention-out": ["027", "029", "030", "034"],
    "5+20+11+12+19+21+inverse-route": ["027", "042"],
    "5+20+11+12+19+21+inverse-route+qk": ["027", "042", "039"],
    "5+20+11+12+19+21+route+qk+fa4": ["027", "042", "039", "041"],
    "5+20+11+12+19+21+inverse-route+tri": ["027", "042", "048"],
    "5+20+11+12+19+21+polar-triangle": ["027", "048"],
    "5+20+11+12+19+21+attention-te": ["027", "049"],
}
PREVIOUS_5_20_GAIN = 8.64


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


def read_training_result(run_dir):
    train_minutes = None
    val_loss = None
    with (run_dir / "console.log").open() as log:
        for line in log:
            train = re.search(r"^step:\s*[\d,]+\s*\|\s*train loss:.*\|\s*total train time:\s*([\d,.]+)\s*min", line)
            validation = re.search(r"^step:\s*[\d,]+\s*\|\s*val loss:\s*(\S+)", line)
            if train:
                train_minutes = float(train.group(1).replace(",", ""))
            if validation:
                val_loss = float(validation.group(1))
    if train_minutes is None or not math.isfinite(train_minutes) or train_minutes <= 0:
        raise ValueError("missing or invalid total training time")
    if val_loss is None or not math.isfinite(val_loss) or val_loss > 3.28:
        raise ValueError("run did not reach validation loss 3.28")
    return train_minutes, val_loss


def main():
    parser = argparse.ArgumentParser(description="Apply experiments, train, upload to the Hub and reverse the patches")
    parser.add_argument("experiments", nargs="*", metavar="ID", help="experiment IDs or registered combinations in execution order; include baseline explicitly; omitted runs all experiments with baseline first")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="number of training GPUs (default: 1)")
    parser.add_argument("--adaptive", action="store_true", help="run baseline and 3+5, select against 5+20's historical 8.64%% saving, then test fused RoPE and Polar Express fusion")
    args = parser.parse_args()
    if args.adaptive and args.experiments:
        parser.error("--adaptive selects its own experiments; do not supply IDs")
    if any(experiment not in ORDER and experiment not in COMBINATIONS for experiment in args.experiments):
        parser.error(f"experiment IDs must be one of: {', '.join([*ORDER, *COMBINATIONS])}")

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
        experiments = ["baseline", "023"] if args.adaptive else args.experiments or ORDER
        combinations = COMBINATIONS.copy()
        for index, experiment in enumerate(experiments, 1):
            applied = []
            label = "3+5" if args.adaptive and experiment == "023" else experiment
            console_path = ROOT / f"experiment-{experiment}-{batch}-{index:02d}.log"
            print(f"experiment {label}; training output: {console_path}", flush=True)
            try:
                with console_path.open("x") as console:
                    patch_ids = [] if experiment == "baseline" else combinations.get(experiment, DEPENDENCIES.get(experiment, []) + [experiment])
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
                        timestamp = next(iter(created)) if created else f"{batch}_exp{index:02d}_failed"
                        run_dir = runs_dir / timestamp
                        if not created:
                            run_dir.mkdir()
                        console.flush()
                        console_path.rename(run_dir / "console.log")

                        upload_path = ROOT / f"experiment-{experiment}-{batch}-{index:02d}-upload.log"
                        print(f"experiment {label}: training exit {train_exit}; uploading {timestamp}", flush=True)
                        with upload_path.open("x") as upload_log:
                            upload_exit = run_command([
                                sys.executable, "push_to_hub.py", "--timestamp", timestamp,
                                "--repo-prefix", "nanogpt-baseline" if experiment == "baseline" else f"nanogpt-exp{label.replace('+', '-')}",
                            ], upload_log)
                        summary.write(f"{label}\t{train_exit}\t{upload_exit}\t{timestamp}\n")
                        summary.flush()
                        failed = failed or train_exit != 0 or upload_exit != 0
                        print(f"experiment {label}: upload exit {upload_exit}; upload output: {upload_path}", flush=True)
                        if args.adaptive:
                            if train_exit != 0:
                                print(f"experiment {label}: adaptive sequence stopped after training failure", flush=True)
                                return 1
                            try:
                                train_minutes, val_loss = read_training_result(run_dir)
                            except (OSError, ValueError) as error:
                                print(f"experiment {label}: adaptive sequence stopped: {error}", flush=True)
                                return 1
                            messages = [f"{label}: total train time {train_minutes:.2f} min; final validation loss {val_loss:.8f}"]
                            if index == 1:
                                baseline_minutes = train_minutes
                            elif index == 2:
                                gain = 100 * (1 - train_minutes / baseline_minutes)
                                # normalize the historical result to this machine's baseline
                                winner_minutes = baseline_minutes * (1 - PREVIOUS_5_20_GAIN / 100)
                                if train_minutes < winner_minutes:
                                    winner = "3+5"
                                    winner_patches = ["005", "023"]
                                    winner_minutes = train_minutes
                                else:
                                    winner = "5+20"
                                    winner_patches = ["005", "020"]
                                messages.append(f"3+5 saved {gain:.4f}% versus this baseline; 5+20 saved {PREVIOUS_5_20_GAIN:.2f}% versus its historical baseline; selected {winner}")
                                messages.append(f"{winner}: reference time {winner_minutes:.4f} min; next add 8 (fused RoPE)")
                                experiments.append(f"{winner}+8")
                                combinations[experiments[-1]] = winner_patches + ["008"]
                            elif index == 3:
                                keep_rope = train_minutes < winner_minutes
                                next_experiment = f"{winner}+8+11" if keep_rope else f"{winner}+11"
                                combinations[next_experiment] = winner_patches + (["008"] if keep_rope else []) + ["011"]
                                experiments.append(next_experiment)
                                messages.append(f"8 (fused RoPE): {'retain' if keep_rope else 'omit'}; {train_minutes:.2f} min versus {winner_minutes:.4f} min; next {next_experiment} adds 11 (Polar Express fusion)")
                            for message in messages:
                                print(message, flush=True)
                                summary.write(f"# {message}\n")
                            summary.flush()
                        continue

                failed = True
                summary.write(f"{label}\tpatch_failed\tnot_attempted\t-\n")
                summary.flush()
                print(f"experiment {label}: patch failed; see {console_path}", flush=True)
                if args.adaptive:
                    return 1
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
