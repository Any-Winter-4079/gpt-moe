import argparse
import importlib.util
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time


REFERENCE_TREE = "8d1ab71bd105b70a882af0f3d6b82f9e9a37504f"
STAGES = [
    ("bf16-attention", "5+20+11+12+19+21+inverse-route+qk", 25),
    ("te-attention", "5+20+11+12+19+21+route+qk+te", 25),
    ("dense-wgrad", "5+20+11+12+19+21+route+qk+dw", 25),
    ("qkv-backquant", "5+20+11+12+19+21+route+qkv-bq", None),
]


def stop_training(process):
    # torchrun workers can start their own sessions
    children = {}
    rows = subprocess.check_output(["ps", "-eo", "pid=,ppid=,pgid="], text=True)
    for row in rows.splitlines():
        pid, parent, group = map(int, row.split())
        children.setdefault(parent, []).append((pid, group))
    groups = {process.pid}
    pids = {process.pid}
    pending = [process.pid]
    while pending:
        for pid, group in children.get(pending.pop(), []):
            groups.add(group)
            pids.add(pid)
            pending.append(pid)
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            states = subprocess.run(
                ["ps", "-o", "stat=", "-p", ",".join(map(str, pids))],
                capture_output=True, text=True,
            ).stdout.splitlines()
            if not any(state.strip() and not state.strip().startswith("Z") for state in states):
                return True
        time.sleep(0.2)
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()
    return False


def run_training(command, log, name, cache_dir, stop_step):
    log.write(f"\ncache replay stage: {name}\nshared cache directory: {cache_dir}\n")
    log.write(f"stop after training step: {stop_step if stop_step is not None else 'full run'}\n")
    log.write(f"\n$ {shlex.join(command)}\n")
    log.flush()
    stopped = False
    step_pattern = re.compile(rf"^step:\s*{stop_step}\s*\|\s*train loss:", re.MULTILINE) if stop_step is not None else None
    with Path(log.name).open(encoding="utf-8", errors="replace") as reader:
        reader.seek(log.tell())
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        pending = ""
        try:
            while process.poll() is None:
                chunk = reader.read()
                if chunk:
                    print(chunk, end="", flush=True)
                    pending += chunk
                    if step_pattern is not None and step_pattern.search(pending):
                        message = f"cache replay: intentional stop after training step {stop_step}; not a completed training run"
                        print(message, flush=True)
                        log.write(f"\n{message}\n")
                        log.flush()
                        stopped = stop_training(process)
                        if not stopped:
                            log.write("\ncache replay: forced shutdown; remaining stages will not run\n")
                            log.flush()
                        break
                    pending = pending.rsplit("\n", 1)[-1]
                time.sleep(0.2)
        except BaseException:
            stop_training(process)
            raise
        finally:
            print(reader.read(), end="", flush=True)
    exit_code = process.wait()
    log.write(f"\ncache replay intentional stop: {stopped}; actual training exit: {exit_code}\n")
    log.flush()
    return exit_code, stopped


def main():
    parser = argparse.ArgumentParser(description="Replay the historical cache sequence without modifying the old training code")
    parser.add_argument("reference", nargs="?", default=".", type=Path, help="old checkout at e43c991 or history-before-squash (default: current directory)")
    args = parser.parse_args()
    reference = args.reference.resolve()
    runner_path = reference / "experiments" / "run.py"
    if not runner_path.is_file():
        parser.error(f"old experiment runner not found: {runner_path}")
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=reference, text=True).strip()
    if tree != REFERENCE_TREE:
        parser.error("reference must contain the old e43c991/history-before-squash code")
    for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        if subprocess.run(command, cwd=reference).returncode:
            parser.error("old checkout has tracked changes; wait for the active runner to finish and reverse its patches")

    cache_dir = Path(tempfile.mkdtemp(prefix="gpt-moe-cache-replay-", dir=reference.parent))
    os.environ.update({
        "TORCHINDUCTOR_CACHE_DIR": str(cache_dir / "inductor"),
        "TRITON_CACHE_DIR": str(cache_dir / "triton"),
        "TRITON_PRINT_AUTOTUNING": "1",
        "PYTHONUNBUFFERED": "1",
        "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE": "0",
        "TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE": "0",
        "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE": "0",
    })
    print(f"reference: {reference}\nshared fresh caches: {cache_dir}", flush=True)

    spec = importlib.util.spec_from_file_location("reference_experiment_runner", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    original_run_command = runner.run_command

    for index, (name, experiment, stop_step) in enumerate(STAGES, 1):
        training_result = None
        upload_exit = None

        def run_command(command, log):
            nonlocal training_result, upload_exit
            if command[0] == "torchrun":
                training_result = run_training(command, log, name, cache_dir, stop_step)
                return training_result[0]
            if len(command) > 1 and command[1] == "push_to_hub.py":
                command = command.copy()
                command[command.index("--repo-prefix") + 1] = f"nanogpt-cache-replay-{name}"
                upload_exit = original_run_command(command, log)
                return upload_exit
            return original_run_command(command, log)

        runner.run_command = run_command
        sys.argv = [str(runner_path), experiment]
        print(f"\ncache replay {index}/{len(STAGES)}: {name}; experiment {experiment}", flush=True)
        result = runner.main()
        expected_stop = training_result is not None and training_result[1]
        success = expected_stop if stop_step is not None else training_result == (0, False) and result == 0
        if not success or upload_exit != 0:
            print(f"cache replay stopped: {name}; training={training_result}, upload={upload_exit}, runner={result}", flush=True)
            return 1
        print(f"cache replay {name}: {'intentional cutoff' if expected_stop else 'full training completed'}; upload succeeded; patches reversed", flush=True)

    print(f"cache replay finished; preserved caches: {cache_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
