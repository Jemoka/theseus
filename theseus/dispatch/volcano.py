"""
Kubernetes Volcano dispatch utilities.

Provides kubectl wrappers for submitting Volcano Jobs, shipping code
to PVCs, and rendering the volcano_job.yaml template.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml
from loguru import logger

from theseus.dispatch.ssh import RunResult

if TYPE_CHECKING:
    from theseus.dispatch.config import VolcanoHostConfig

# Path to Volcano Job template
VOLCANO_JOB_TEMPLATE = Path(__file__).parent / "volcano_job.yaml"

# Helper pod image for PVC file operations
_BUSYBOX_IMAGE = "busybox:1.36"


def _kubectl_base(
    kubeconfig: str | None = None,
    context: str | None = None,
) -> list[str]:
    """Build base kubectl command with optional kubeconfig/context."""
    cmd = ["kubectl"]
    if kubeconfig:
        cmd += ["--kubeconfig", kubeconfig]
    if context:
        cmd += ["--context", context]
    return cmd


def _run_kubectl(
    args: list[str],
    kubeconfig: str | None = None,
    context: str | None = None,
    timeout: float | None = None,
    stdin_data: bytes | None = None,
) -> RunResult:
    """Run a kubectl command and return RunResult."""
    cmd = _kubectl_base(kubeconfig, context) + args
    logger.debug(f"VOLCANO | running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=stdin_data is None,
            input=stdin_data,
            timeout=timeout,
        )
        stdout = (
            result.stdout
            if isinstance(result.stdout, str)
            else result.stdout.decode("utf-8", errors="replace")
        )
        stderr = (
            result.stderr
            if isinstance(result.stderr, str)
            else result.stderr.decode("utf-8", errors="replace")
        )
        return RunResult(
            returncode=result.returncode,
            stdout=stdout,
            stderr=stderr,
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            returncode=-1,
            stdout="",
            stderr=f"kubectl timed out after {timeout}s",
        )


def get_queue_availability(
    queue_name: str,
    gpu_resource_key: str = "nvidia.com/gpu",
    kubeconfig: str | None = None,
    context: str | None = None,
    timeout: float = 30.0,
) -> tuple[int, int] | None:
    """Return (available, capacity) GPU count for a Volcano queue.

    Fetches ``spec.capability`` and ``status.allocated`` from the Queue CRD.
    Returns ``None`` on error (queue not found, kubectl failure, etc.).
    """
    result = _run_kubectl(
        ["get", "queue", queue_name, "-o", "json"],
        kubeconfig=kubeconfig,
        context=context,
        timeout=timeout,
    )
    if not result.ok:
        logger.debug(f"VOLCANO | failed to get queue '{queue_name}': {result.stderr}")
        return None
    try:
        data = json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError):
        logger.debug(f"VOLCANO | failed to parse queue JSON for '{queue_name}'")
        return None

    capacity = int(
        data.get("spec", {}).get("capability", {}).get(gpu_resource_key, "0")
    )
    allocated = int(
        data.get("status", {}).get("allocated", {}).get(gpu_resource_key, "0")
    )
    available = max(capacity - allocated, 0)
    logger.debug(
        f"VOLCANO | queue '{queue_name}': {available}/{capacity} GPUs available "
        f"({allocated} allocated)"
    )
    return (available, capacity)


def apply_job(
    yaml_content: str,
    namespace: str = "default",
    kubeconfig: str | None = None,
    context: str | None = None,
    timeout: float = 60.0,
) -> RunResult:
    """Submit a Volcano Job via kubectl apply -f -."""
    result = _run_kubectl(
        ["apply", "-n", namespace, "-f", "-"],
        kubeconfig=kubeconfig,
        context=context,
        timeout=timeout,
        stdin_data=yaml_content.encode("utf-8"),
    )
    if result.ok:
        logger.info("VOLCANO | job applied successfully")
    else:
        logger.error(f"VOLCANO | apply failed: {result.stderr}")
    return result


def delete_job(
    job_name: str,
    namespace: str = "default",
    kubeconfig: str | None = None,
    context: str | None = None,
    timeout: float = 60.0,
) -> RunResult:
    """Delete a Volcano Job."""
    return _run_kubectl(
        ["delete", "vcjob", job_name, "-n", namespace, "--ignore-not-found"],
        kubeconfig=kubeconfig,
        context=context,
        timeout=timeout,
    )


def get_job_status(
    job_name: str,
    namespace: str = "default",
    kubeconfig: str | None = None,
    context: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any] | None:
    """Get Volcano Job status as a dict, or None if not found."""
    result = _run_kubectl(
        ["get", "vcjob", job_name, "-n", namespace, "-o", "json"],
        kubeconfig=kubeconfig,
        context=context,
        timeout=timeout,
    )
    if not result.ok:
        return None
    try:
        data: dict[str, Any] = json.loads(result.stdout)
        return data
    except json.JSONDecodeError:
        logger.warning("VOLCANO | failed to parse job status JSON")
        return None


def get_pod_logs(
    job_name: str,
    namespace: str = "default",
    kubeconfig: str | None = None,
    context: str | None = None,
    timeout: float = 30.0,
    tail_lines: int = 100,
) -> RunResult:
    """Get logs from pods belonging to a Volcano Job."""
    return _run_kubectl(
        [
            "logs",
            "-n",
            namespace,
            "-l",
            f"volcano.sh/job-name={job_name}",
            "--tail",
            str(tail_lines),
            "--all-containers",
        ],
        kubeconfig=kubeconfig,
        context=context,
        timeout=timeout,
    )


def wait_completed(
    job_name: str,
    namespace: str = "default",
    kubeconfig: str | None = None,
    context: str | None = None,
    poll_interval: float = 10.0,
    timeout: float | None = None,
) -> str | None:
    """Poll until a Volcano Job reaches a terminal state.

    Returns the final state string ("Completed", "Failed", "Aborted", etc.)
    or None on timeout.
    """
    terminal_states = {"Completed", "Failed", "Aborted", "Terminated"}
    start = time.time()
    while True:
        status = get_job_status(job_name, namespace, kubeconfig, context)
        if status:
            phase: str = status.get("status", {}).get("state", {}).get("phase", "")
            logger.debug(f"VOLCANO | job '{job_name}' phase: {phase}")
            if phase in terminal_states:
                return phase
        if timeout is not None and (time.time() - start) > timeout:
            logger.warning(f"VOLCANO | timed out waiting for job '{job_name}'")
            return None
        time.sleep(poll_interval)


def render_volcano_job(
    job_name: str,
    host_config: VolcanoHostConfig,
    bootstrap_command: str,
    work_dir: str,
) -> str:
    """Render the volcano_job.yaml template with concrete values.

    Returns the rendered YAML string ready for kubectl apply.
    """
    template = VOLCANO_JOB_TEMPLATE.read_text()

    num_replicas = host_config.num_nodes

    # Basic substitutions
    rendered = template.replace("__JOB_NAME__", job_name)
    rendered = rendered.replace("__NAMESPACE__", host_config.namespace)
    rendered = rendered.replace("__QUEUE__", host_config.queue or "default")
    rendered = rendered.replace("__NUM_REPLICAS__", str(num_replicas))
    rendered = rendered.replace("__IMAGE__", host_config.image)
    rendered = rendered.replace("__PVC_NAME__", host_config.pvc_name)
    rendered = rendered.replace("__PVC_MOUNT_PATH__", host_config.pvc_mount_path)
    rendered = rendered.replace("__BOOTSTRAP_COMMAND__", bootstrap_command)

    # Labels
    if host_config.labels:
        label_lines = ["labels:"]
        for k, v in host_config.labels.items():
            label_lines.append(f'  {k}: "{v}"')
        rendered = rendered.replace(
            "__LABELS__",
            "\n    ".join(label_lines),
        )
    else:
        rendered = rendered.replace("  __LABELS__\n", "")

    # Priority class
    if host_config.priority_class:
        rendered = rendered.replace(
            "__PRIORITY_CLASS__",
            f"priorityClassName: {host_config.priority_class}",
        )
    else:
        rendered = rendered.replace("          __PRIORITY_CLASS__\n", "")

    # Service account
    if host_config.service_account:
        rendered = rendered.replace(
            "__SERVICE_ACCOUNT__",
            f"serviceAccountName: {host_config.service_account}",
        )
    else:
        rendered = rendered.replace("          __SERVICE_ACCOUNT__\n", "")

    # Node selector
    if host_config.node_selector:
        ns_lines = ["nodeSelector:"]
        for k, v in host_config.node_selector.items():
            ns_lines.append(f'  {k}: "{v}"')
        rendered = rendered.replace(
            "__NODE_SELECTOR__",
            "\n            ".join(ns_lines),
        )
    else:
        rendered = rendered.replace("          __NODE_SELECTOR__\n", "")

    # Tolerations
    if host_config.tolerations:
        tol_lines = ["tolerations:"]
        for t in host_config.tolerations:
            tol_lines.append(f'- key: "{t.get("key", "")}"')
            if "operator" in t:
                tol_lines.append(f'  operator: "{t["operator"]}"')
            if "value" in t:
                tol_lines.append(f'  value: "{t["value"]}"')
            if "effect" in t:
                tol_lines.append(f'  effect: "{t["effect"]}"')
        rendered = rendered.replace(
            "__TOLERATIONS__",
            "\n            ".join(tol_lines),
        )
    else:
        rendered = rendered.replace("          __TOLERATIONS__\n", "")

    # Resources
    resource_lines = []
    if host_config.cpu:
        resource_lines.append(f'cpu: "{host_config.cpu}"')
    if host_config.memory:
        resource_lines.append(f'memory: "{host_config.memory}"')
    if host_config.gpus_per_node > 0:
        resource_lines.append(
            f"{host_config.gpu_resource_key}: {host_config.gpus_per_node}"
        )
    if host_config.rdma:
        resource_lines.append(f"rdma/rdma_shared_device_a: {host_config.rdma_per_node}")
    rendered = rendered.replace(
        "__RESOURCES__",
        "\n                  ".join(resource_lines) if resource_lines else "{}",
    )

    # Environment variables
    env_lines = []
    # Volcano mode flag for bootstrap
    all_env = {"THESEUS_VOLCANO_MODE": "1"}
    all_env.update(host_config.env)
    for k, v in all_env.items():
        env_lines.append(f'- name: "{k}"')
        env_lines.append(f'  value: "{v}"')
    rendered = rendered.replace(
        "__ENV_VARS__",
        "\n                ".join(env_lines),
    )

    # Shared memory (/dev/shm) volume
    if host_config.shm_size:
        rendered = rendered.replace(
            "__SHM_MOUNT__",
            "- name: dshm\n                  mountPath: /dev/shm",
        )
        rendered = rendered.replace(
            "__SHM_VOLUME__",
            f"- name: dshm\n              emptyDir:\n                medium: Memory\n                sizeLimit: {host_config.shm_size}",
        )
    else:
        rendered = rendered.replace("                __SHM_MOUNT__\n", "")
        rendered = rendered.replace("            __SHM_VOLUME__\n", "")

    return rendered


def ship_and_write_to_pvc(
    tarball: bytes,
    script: str,
    bootstrap_pys: dict[str, str],
    pvc_name: str,
    remote_subdir: str,
    queue: str,
    namespace: str = "default",
    pvc_mount_path: str = "/workspace",
    kubeconfig: str | None = None,
    context: str | None = None,
    timeout: float = 120.0,
) -> RunResult:
    """Ship code + bootstrap files to a PVC via a temporary Volcano Job.

    Uses a single Volcano Job (vcjob) as the helper instead of a plain Pod,
    so it goes through the Volcano scheduler like real workloads.  Code is
    streamed via ``tar | kubectl exec -i … tar -xpf -`` (no intermediate
    files).  Bootstrap scripts are written via ``kubectl exec -i … cat``.
    The helper vcjob is cleaned up on exit.
    """
    helper_name = f"theseus-pvc-loader-{int(time.time())}"
    dest_path = f"{pvc_mount_path}/{remote_subdir}"

    # 1. Create helper Volcano Job — single replica, sleeps until we're done
    helper_yaml = yaml.dump(
        {
            "apiVersion": "batch.volcano.sh/v1alpha1",
            "kind": "Job",
            "metadata": {
                "name": helper_name,
                "namespace": namespace,
            },
            "spec": {
                "schedulerName": "volcano",
                "queue": queue or "default",
                "minAvailable": 1,
                "plugins": {"ssh": [], "svc": [], "env": []},
                "tasks": [
                    {
                        "name": "loader",
                        "replicas": 1,
                        "template": {
                            "spec": {
                                "schedulerName": "volcano",
                                "restartPolicy": "Never",
                                "containers": [
                                    {
                                        "name": "loader",
                                        "image": _BUSYBOX_IMAGE,
                                        "command": ["sh", "-c"],
                                        "args": [
                                            f"mkdir -p {dest_path} && "
                                            f"echo 'ready' && "
                                            f"sleep infinity"
                                        ],
                                        "volumeMounts": [
                                            {
                                                "name": "workspace",
                                                "mountPath": pvc_mount_path,
                                            }
                                        ],
                                    }
                                ],
                                "volumes": [
                                    {
                                        "name": "workspace",
                                        "persistentVolumeClaim": {
                                            "claimName": pvc_name
                                        },
                                    }
                                ],
                            },
                        },
                    }
                ],
            },
        }
    )

    logger.debug(f"VOLCANO | creating helper vcjob '{helper_name}'")
    create_result = _run_kubectl(
        ["create", "-n", namespace, "-f", "-"],
        kubeconfig=kubeconfig,
        context=context,
        timeout=timeout,
        stdin_data=helper_yaml.encode("utf-8"),
    )
    if not create_result.ok:
        return create_result

    # Helper cleanup on any exit path
    def _cleanup() -> None:
        logger.debug(f"VOLCANO | cleaning up helper vcjob '{helper_name}'")
        _run_kubectl(
            ["delete", "vcjob", helper_name, "-n", namespace, "--ignore-not-found"],
            kubeconfig=kubeconfig,
            context=context,
            timeout=60.0,
        )

    # 2. Wait for the loader pod to be scheduled and Ready
    logger.debug("VOLCANO | waiting for helper pod to be scheduled")
    pod_name: str | None = None
    poll_start = time.time()
    while pod_name is None:
        if (time.time() - poll_start) > timeout:
            _cleanup()
            return RunResult(
                returncode=-1,
                stdout="",
                stderr=f"helper vcjob pod not scheduled within {timeout}s",
            )
        list_result = _run_kubectl(
            [
                "get",
                "pods",
                "-n",
                namespace,
                "-l",
                f"volcano.sh/job-name={helper_name}",
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ],
            kubeconfig=kubeconfig,
            context=context,
            timeout=30.0,
        )
        name_candidate = list_result.stdout.strip()
        if list_result.ok and name_candidate and not name_candidate.startswith("{"):
            pod_name = name_candidate
        else:
            time.sleep(5)

    logger.debug(f"VOLCANO | helper pod: {pod_name}, waiting for Ready")
    wait_result = _run_kubectl(
        [
            "wait",
            "--for=condition=Ready",
            f"pod/{pod_name}",
            "-n",
            namespace,
            "--timeout=300s",
        ],
        kubeconfig=kubeconfig,
        context=context,
        timeout=360.0,
    )
    if not wait_result.ok:
        _cleanup()
        return wait_result

    # 3. Stream tarball to PVC via tar pipe
    logger.debug(f"VOLCANO | streaming code tarball to {dest_path}")
    pipe_cmd = _kubectl_base(kubeconfig, context) + [
        "exec",
        "-i",
        pod_name,
        "-n",
        namespace,
        "--",
        "sh",
        "-c",
        f"tar -C '{dest_path}' -xzf - -m",
    ]
    try:
        pipe_result = subprocess.run(
            pipe_cmd,
            input=tarball,
            capture_output=True,
            timeout=timeout,
        )
        if pipe_result.returncode != 0:
            raw_stderr = pipe_result.stderr
            stderr_str = (
                raw_stderr.decode("utf-8", errors="replace")
                if isinstance(raw_stderr, bytes)
                else raw_stderr
            )
            _cleanup()
            return RunResult(
                returncode=pipe_result.returncode,
                stdout="",
                stderr=stderr_str,
            )
    except subprocess.TimeoutExpired:
        _cleanup()
        return RunResult(
            returncode=-1,
            stdout="",
            stderr=f"tar streaming timed out after {timeout}s",
        )

    logger.info(f"VOLCANO | code shipped to PVC '{pvc_name}' at {dest_path}")

    # 4. Write bootstrap.sh and Python files via kubectl exec
    logger.debug("VOLCANO | writing bootstrap files")
    write_result = _run_kubectl(
        [
            "exec",
            "-i",
            pod_name,
            "-n",
            namespace,
            "--",
            "sh",
            "-c",
            f"cat > {dest_path}/_bootstrap.sh && chmod +x {dest_path}/_bootstrap.sh",
        ],
        kubeconfig=kubeconfig,
        context=context,
        timeout=timeout,
        stdin_data=script.encode("utf-8"),
    )
    if not write_result.ok:
        _cleanup()
        return write_result

    for filename, content in bootstrap_pys.items():
        file_result = _run_kubectl(
            [
                "exec",
                "-i",
                pod_name,
                "-n",
                namespace,
                "--",
                "sh",
                "-c",
                f"cat > {dest_path}/{filename}",
            ],
            kubeconfig=kubeconfig,
            context=context,
            timeout=timeout,
            stdin_data=content.encode("utf-8"),
        )
        if not file_result.ok:
            _cleanup()
            return file_result

    logger.info(f"VOLCANO | bootstrap files written to PVC at {dest_path}")

    # 5. Cleanup helper
    _cleanup()
    return RunResult(returncode=0, stdout="", stderr="")
