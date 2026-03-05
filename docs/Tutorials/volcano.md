# Volcano Dispatch Guide

Volcano is a CNCF batch scheduler for Kubernetes that adds **gang scheduling**
(all-or-nothing pod scheduling), queue-based resource management, and
distributed training plugins. Theseus uses it to run multi-node JAX training
jobs on Kubernetes GPU clusters.

## Prerequisites

Before using the Volcano backend you need:

1. **A Kubernetes cluster with Volcano installed.**
   Check with: `kubectl get deployment -n volcano-system`
   If missing, install via Helm:
   ```bash
   helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
   helm install volcano volcano-sh/volcano -n volcano-system --create-namespace
   ```

2. **kubectl configured** to talk to the cluster
   (`~/.kube/config` or a custom kubeconfig path).

3. **A PersistentVolumeClaim (PVC)** that all pods can mount.
   This is where theseus ships your code and bootstrap scripts.
   ```bash
   # Check existing PVCs:
   kubectl get pvc -n <namespace>

   # Example: create a 100Gi PVC (adjust storageClassName for your cluster)
   kubectl apply -f - <<'EOF'
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: training-data
     namespace: training
   spec:
     accessModes: [ReadWriteMany]   # RWX so all pods can mount simultaneously
     storageClassName: your-sc      # e.g. "azurefile-csi-premium", "efs-sc", "nfs-client"
     resources:
       requests:
         storage: 100Gi
   EOF
   ```

4. **A container image** with your CUDA/JAX stack.
   Popular choices:
   - `nvcr.io/nvidia/jax:24.04-py3` (NVIDIA JAX)
   - `nvcr.io/nvidia/pytorch:24.07-py3` (if you need PyTorch too)
   - Or your own image with `uv`, Python, and CUDA drivers.

5. **(Optional) A Volcano Queue** for resource quotas:
   ```bash
   # List queues:
   kubectl get queue

   # Create one:
   kubectl apply -f - <<'EOF'
   apiVersion: scheduling.volcano.sh/v1beta1
   kind: Queue
   metadata:
     name: training
   spec:
     weight: 1
     capability:
       nvidia.com/gpu: 64
   EOF
   ```

## Dispatch Config

Add a Volcano host to your `~/.theseus.yaml` (or whatever dispatch config you
use). Here is a walkthrough of every field:

```yaml
clusters:
  # The cluster entry defines paths *inside the pods* (on the PVC).
  # These are used by the bootstrap script for checkpoints, logs, etc.
  k8s:
    root: /workspace                  # root of PVC — must match pvc_mount_path
    work: /workspace/projects         # scratch/working directory on the PVC
    log: /workspace/projects          # log directory

hosts:
  gpu-cluster:
    type: volcano

    # Must match a cluster entry above.
    cluster: k8s

    # Kubernetes namespace where the Volcano Job will be created.
    # Must exist already:  kubectl get ns training
    namespace: training

    # Volcano queue name. Must exist:  kubectl get queue
    # If your cluster only has a "default" queue, use "default".
    queue: default

    # Container image. Must be pullable from the cluster.
    # Tip: use `kubectl describe pod <pod>` to debug ImagePullBackOff.
    image: nvcr.io/nvidia/jax:24.04-py3

    # Name of the PVC where code + data is stored.
    # Must be ReadWriteMany if num_nodes > 1.
    pvc_name: training-data

    # Where the PVC is mounted inside each pod.
    # Must match the cluster `root` so all paths are consistent.
    pvc_mount_path: /workspace

    # Number of pods (= nodes) to schedule.
    # Gang scheduling ensures ALL are placed or NONE.
    num_nodes: 4

    # GPUs requested per pod.
    gpus_per_node: 8

    # Chip mapping used by the solver to match hardware requests.
    # Keys must match theseus chip names (h100, a100-sxm4-80gb, etc.).
    # Values = count per node (should match gpus_per_node for GPU chips).
    chips:
      h100: 8

    # Kubernetes resource key for GPUs. Almost always "nvidia.com/gpu".
    # Some clusters use "amd.com/gpu" or custom device plugins.
    gpu_resource_key: nvidia.com/gpu

    # CPU and memory requests/limits per pod.
    # These are Kubernetes resource quantities.
    # Tip: check what's available: kubectl describe node <node> | grep Allocatable
    cpu: "32"         # number of CPU cores
    memory: "256Gi"   # RAM

    # Shared memory size. NCCL and multiprocessing use /dev/shm.
    # If you see "bus error" or NCCL hangs, increase this.
    # Rule of thumb: at least 1/4 of memory, often 64Gi+.
    shm_size: "64Gi"

    # --- Optional fields ---

    # Kubernetes PriorityClass for scheduling priority.
    # List available: kubectl get priorityclass
    # priority_class: high-priority

    # ServiceAccount if your pods need specific RBAC permissions.
    # service_account: training-sa

    # If you have multiple kubeconfigs or contexts:
    # kubeconfig: /path/to/kubeconfig
    # context: my-cluster-context

    # Node selector to pin pods to specific node pools.
    # Check labels: kubectl get nodes --show-labels
    # node_selector:
    #   accelerator: nvidia-h100
    #   kubernetes.io/os: linux

    # Tolerations for tainted GPU nodes.
    # Most GPU clusters taint nodes so only GPU workloads land on them.
    # Check taints: kubectl describe node <node> | grep Taint
    # tolerations:
    #   - key: nvidia.com/gpu
    #     operator: Exists
    #     effect: NoSchedule
    #   - key: rdma
    #     operator: Exists
    #     effect: NoSchedule

    # RDMA (Remote Direct Memory Access) for high-bandwidth inter-node communication.
    # Enable this if your nodes have InfiniBand or RoCE NICs and the K8s RDMA
    # device plugin is installed (check: kubectl get nodes -o json | jq '.items[0].status.capacity | keys').
    # When true, each pod requests `rdma/rdma_shared_device_a` resources.
    # NCCL uses RDMA automatically for GPU-to-GPU transfers across nodes,
    # giving much higher throughput than TCP.
    # rdma: false
    # rdma_per_node: 8             # number of RDMA devices per node (default 8)

    # Extra environment variables injected into every pod.
    # env:
    #   NCCL_DEBUG: INFO           # useful for debugging multi-node comms
    #   NCCL_IB_DISABLE: "0"       # set to "1" if no InfiniBand
    #   OMP_NUM_THREADS: "8"

    # uv dependency groups to sync in the bootstrap script.
    uv_groups: [gpu]
```

## Submitting a Job

```bash
# Basic submit (uses dispatch config from ~/.theseus.yaml):
theseus submit my-run experiment.yaml

# Override the container image (e.g. testing a new image):
theseus submit my-run experiment.yaml --volcano-image myrepo/jax:latest

# Override the namespace:
theseus submit my-run experiment.yaml --volcano-namespace dev

# Request specific hardware (solver matches against chips config):
theseus submit my-run experiment.yaml --chip h100 -n 32

# Include uncommitted code changes (default is --dirty):
theseus submit my-run experiment.yaml --dirty

# Target a specific cluster if you have multiple:
theseus submit my-run experiment.yaml --cluster k8s
```

## Monitoring Jobs

```bash
# List Volcano jobs:
kubectl get vcjob -n training

# Watch job status:
kubectl get vcjob <job-name> -n training -w

# Check pod status:
kubectl get pods -l volcano.sh/job-name=<job-name> -n training

# Stream logs from all pods:
kubectl logs -l volcano.sh/job-name=<job-name> -n training --all-containers -f

# Stream logs from a single pod:
kubectl logs <pod-name> -n training -f

# Get detailed pod info (useful for debugging scheduling/image issues):
kubectl describe pod <pod-name> -n training

# Check events (scheduling failures, image pull errors, etc.):
kubectl get events -n training --sort-by=.lastTimestamp | grep <job-name>
```

## Deleting Jobs

```bash
# Delete a specific job (also deletes its pods):
kubectl delete vcjob <job-name> -n training

# Delete all jobs by a label:
kubectl delete vcjob -l submitter=myuser -n training
```

## How It Works Under the Hood

When you run `theseus submit` targeting a Volcano host:

1. **Solver** checks that the Volcano host has enough chips
   (`chips_per_node * num_nodes >= requested chips`). It computes
   `num_nodes = ceil(requested_chips / chips_per_node)` automatically.

2. **Code shipping**: theseus creates a temporary Volcano Job (vcjob) as
   a helper — this goes through the Volcano scheduler like real workloads,
   which is important on clusters where all workloads must be scheduled
   through Volcano queues. The helper mounts the PVC, receives a tarball
   of your repo via streaming `tar | kubectl exec -i … tar -xpf -` (no
   intermediate files), and then receives bootstrap scripts
   (`_bootstrap.sh`, `_bootstrap_dispatch.py`) via `kubectl exec -i … cat`.
   The helper vcjob is deleted after all files are written.

3. **Volcano Job YAML** is rendered from a template with your config values
   and submitted via `kubectl apply`.

4. **Gang scheduling**: Volcano ensures all `num_nodes` pods are scheduled
   simultaneously (via `minAvailable`). If the cluster doesn't have enough
   GPUs, pods wait in the queue until resources free up.

5. **JAX distributed init**: Each pod gets `THESEUS_VOLCANO_MODE=1` and
   Volcano's `env` plugin injects `VC_WORKER_HOSTS` (comma-separated pod
   hostnames) and `VC_TASK_INDEX` (0-based pod index). The bootstrap script
   uses these to call `jax.distributed.initialize()` with:
   - `coordinator_address` = first host in `VC_WORKER_HOSTS`, port 1234
   - `num_processes` = length of `VC_WORKER_HOSTS`
   - `process_id` = `VC_TASK_INDEX`

**Note on JuiceFS**: If your cluster config has a `mount:` field (JuiceFS
Redis URL), it will be ignored for Volcano dispatch — a warning is logged.
Volcano uses the PVC for all storage. If you need JuiceFS-backed storage in
K8s, configure it at the infrastructure level via the
[JuiceFS CSI driver](https://juicefs.com/docs/csi/introduction/) backing
the PVC itself.

## Key Volcano Concepts

| Concept | What it means |
|---|---|
| **Volcano Job** (`vcjob`) | A custom K8s resource that groups pods ("tasks") for batch scheduling |
| **Gang scheduling** | `minAvailable: N` means all N pods get placed at once, or none do |
| **Queue** | Resource pool with optional quotas (GPU, CPU, memory limits) |
| **Plugins** | `env` injects `VC_*` vars; `svc` creates DNS between pods; `ssh` sets up passwordless SSH |
| **Task** | A group of identical pod replicas within a job. Theseus uses a single "worker" task group since JAX processes are symmetric |

## Troubleshooting

### Pods stuck in Pending
```bash
# Check why scheduling failed:
kubectl describe pod <pod-name> -n training | grep -A5 Events

# Common causes:
# - Not enough GPUs (check queue capacity and node resources)
# - Node taints without matching tolerations
# - PVC can't be mounted (wrong access mode or storage class)
```

### ImagePullBackOff
```bash
# Check the image name is correct and accessible:
kubectl describe pod <pod-name> -n training | grep -A3 "image"

# If using a private registry, ensure imagePullSecrets are configured
```

### NCCL errors / multi-node communication failures
- Increase `shm_size` (try `"64Gi"` or more)
- Set `env: { NCCL_DEBUG: INFO }` to see detailed NCCL logs
- Check if RDMA/InfiniBand is available; if not, set `NCCL_IB_DISABLE: "1"`
- Verify pods can reach each other: the `svc` plugin creates headless services for DNS

### Job completes but shows as Failed
- The `TaskCompleted` policy on the worker task triggers `CompleteJob`.
  Check that your training script exits with code 0.
- Check logs: `kubectl logs <pod> -n training`

### Code changes not reflected
- Make sure you're using `--dirty` to include uncommitted changes
- The code is shipped to PVC at submit time; re-submit to pick up new changes
