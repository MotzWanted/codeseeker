"""Spawns multiple vLLMs instances."""

import copy
import math
import os
import pathlib
import re
import subprocess
import time

import rich
from loguru import logger

from docgen.tools.arguantic import Arguantic


def get_gpu_pids() -> dict[str, list[str]]:
    """Return a dict of PIDs by GPU index."""
    command_output = subprocess.check_output(
        ["nvidia-smi", "-L"]  # noqa: S603, S607
    )
    uuids_to_index = {}
    for line in command_output.decode().split("\n"):
        if not line or "MIG" in line:
            continue
        # Example: GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-c7ab3bb0-c597-a392-9037-db2f567e46b6)
        matches = re.match(r"GPU (?P<index>\d+):.*\(UUID: (?P<uuid>.*)\)", line)
        if not matches:
            raise ValueError(f"Cannot parse line: `{line}`")
        uuid = matches["uuid"]
        index = matches["index"]
        uuids_to_index[uuid] = index

    command_output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"]  # noqa: S603, S607
    )
    ids_to_users = {i: [] for i in uuids_to_index.values()}
    for line in command_output.decode().split("\n"):
        if not line:
            continue
        uuid, pids = line.split(",")
        index = uuids_to_index[uuid]
        ids_to_users[index].append(pids)

    return ids_to_users


# Example usage of the function
try:
    gpu_pids = get_gpu_pids()
    print("GPU PID mapping:", gpu_pids)
except ValueError as e:
    print(str(e))


class Args(Arguantic):
    """Args for the script."""

    ports: str = "8001,8002,8003,8004"
    pypath: str = "/home/amo/miniconda3/envs/vllm/bin/python"
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    quantization: str = ""
    gpu_per_model: int = 1
    gpu_usage: float = 0.95
    max_length: int = -1  # Override model max length
    cache_dir: str = "/mnt/md0/amo/cache"
    dtype: str = "auto"
    log2file: int = 0
    disable_logging: int = 1
    enforce_eager: bool = False


def spawn_servers(args: Args) -> None:
    """Spawn multiple vLLM servers."""
    rich.print(args)
    service_name = args.model_name.replace("/", "-")
    gpus_by_port = _distribute_gpus(args)
    rich.print(gpus_by_port)

    procs = []
    try:
        for i, port in enumerate(gpus_by_port):
            env_vars = copy.copy(os.environ)
            env_vars.update({"CUDA_VISIBLE_DEVICES": ",".join(str(x) for x in gpus_by_port[port])})
            command = _make_command(args, port)
            logger.info(f"Spawning server `{i}` on port `{port}` with GPUs `{gpus_by_port[port]}`")
            logger.info(f"Command: `{' '.join(command)}`")

            # setup log files
            if args.log2file:
                stdout_file, stderr_file = _setup_log_files(service_name, f"{i}-{port}")
            stdout_file = stderr_file = None

            # spawn server
            server_proc = subprocess.Popen(
                command,  # noqa: S603
                env=env_vars,
                stdout=stdout_file.open("w") if stdout_file else None,
                stderr=stderr_file.open("w") if stderr_file else None,
            )
            procs.append(server_proc)

        # Wait indefinitely
        while True:
            time.sleep(1)
            # Check if any of the processes have died
            for proc in procs:
                if proc.poll() is not None:
                    raise RuntimeError(f"Process `{proc.pid}` died with return code `{proc.returncode}`")

    except Exception as e:
        for proc in procs:
            proc.terminate()
        raise e

    # Terminate
    for proc in procs:
        proc.terminate()


def _distribute_gpus(args: Args) -> dict[int, list[int]]:
    """Distribute available GPUs to instances (ports)."""
    ports = [int(p) for p in args.ports.split(",")]  # one port per instance
    available_gpus_ids = sorted(
        (int(i) for i, pids in get_gpu_pids().items() if len(pids) == 0),
        reverse=True,
    )
    logger.info(f"Available GPUs: `{available_gpus_ids}`")
    requested_gpus = math.ceil(len(ports) * args.gpu_per_model)
    logger.info(f"Requested `{requested_gpus}` GPUs")
    if requested_gpus > len(available_gpus_ids):
        raise ValueError(f"Requested `{requested_gpus}` GPUs but only `{len(available_gpus_ids)}` are available.")

    # Allocate GPUs to instances
    gpus_availability = {i: 1.0 for i in available_gpus_ids}
    gpus_by_port: dict[int, list[int]] = {p: [] for p in ports}

    for port in ports:
        allocated_gpus = 0
        for gpu in sorted(gpus_availability, key=lambda gpu: gpus_availability[gpu], reverse=True):
            if allocated_gpus >= args.gpu_per_model:
                break
            if gpus_availability[gpu] > 0:
                gpus_by_port[port].append(gpu)
                allocated_gpus += 1
                gpus_availability[gpu] -= 1
        if len(gpus_by_port[port]) < args.gpu_per_model:
            raise RuntimeError(
                f"Could not allocate `{args.gpu_per_model}` GPUs to port `{port}`. "
                f"Available GPUs: `{gpus_availability}`"
            )

    return gpus_by_port


def _setup_log_files(service_name: str, spec: str) -> tuple[pathlib.Path, pathlib.Path]:
    stdout_file = pathlib.Path(f"vllm-{service_name}-{spec}.stdout.log")
    logger.debug(f"Writing stdout to `{stdout_file.absolute()}`")
    if stdout_file.exists():
        stdout_file.unlink()
    stderr_file = pathlib.Path(f"vllm-{service_name}-{spec}.stderr.log")
    logger.debug(f"Writing stderr to `{stderr_file.absolute()}`")
    if stderr_file.exists():
        stderr_file.unlink()
    return stdout_file, stderr_file


def _make_command(args: Args, port: int) -> list[str]:
    command = [
        args.pypath,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--port",
        str(port),
        "--model",
        args.model_name,
        "--download-dir",
        args.cache_dir,
        "--dtype",
        args.dtype,
        "--tensor-parallel-size",
        str(args.gpu_per_model),
    ]
    if args.max_length > 0:
        command += ["--max-model-len", str(args.max_length)]
    if args.gpu_usage > 0:
        gpu_usage = args.gpu_usage * min(args.gpu_per_model, 1)
        command += ["--gpu-memory-utilization", str(gpu_usage)]
    if len(args.quantization) > 0:
        command += ["--quantization", args.quantization]
    if args.disable_logging:
        command += ["--disable-log-requests", "--disable-log-stats"]

    return command


if __name__ == "__main__":
    spawn_servers(Args.parse())
