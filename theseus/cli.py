"""
a basic cli called `theseus` implemented in click
with nice `rich` driven printouts, etc. not intended
to be used for i.e. logging; and instead meant for
user-facing interactions
"""

import sys
import click
import inspect
from pathlib import Path
from omegaconf import OmegaConf
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
import random
import numpy as np

from typing import Any, List, Type

from theseus.registry import JOBS
from theseus.config import build, configuration
from theseus.job import RestoreableJob
from theseus.base.job import ExecutionSpec, JobSpec
from theseus.base.chip import SUPPORTED_CHIPS
from theseus.base.hardware import HardwareRequest

console = Console()

random.seed(0)
np.random.seed(0)

load_dotenv()


def setup_logging(verbose: bool) -> None:
    """Configure loguru with appropriate log level."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> |"
        "<level>{level: ^8}</level>| "
        "<magenta>({name}:{line})</magenta> <level>{message}</level>",
        level="DEBUG" if verbose else "INFO",
        colorize=True,
        enqueue=True,
        filter=lambda x: x["extra"].get("task", "") != "plot",
    )


@click.group()  # type: ignore[misc]
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")  # type: ignore[misc]
def theseus(verbose: bool) -> None:
    """Theseus CLI for managing and running jobs."""
    setup_logging(verbose)


@theseus.command()  # type: ignore[misc]
def jobs() -> None:
    """List all available jobs in the registry."""
    if not JOBS:
        console.print("\n[yellow]No jobs registered[/yellow]\n")
        return

    table = Table(show_header=True, header_style="bold green")
    table.add_column("Job Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")

    for job_name in sorted(JOBS.keys()):
        job_class = JOBS[job_name]
        docstring = inspect.getdoc(job_class)

        # Get first line of docstring as description
        if docstring:
            description = docstring.split("\n")[0].strip()
        else:
            description = "[dim]No description available[/dim]"

        table.add_row(job_name, description)

    console.print()
    console.print(table)
    console.print()


@theseus.command()  # type: ignore[misc]
@click.argument("job")  # type: ignore[misc]
@click.argument("out_yaml")  # type: ignore[misc]
@click.option(  # type: ignore[misc]
    "-p", "--previous", default=None, help="Previous YAML config to use as base"
)
@click.option(  # type: ignore[misc]
    "--chip",
    default=None,
    help=f"Chip type for hardware request ({', '.join(SUPPORTED_CHIPS.keys())})",
)
@click.option(  # type: ignore[misc]
    "-n",
    "--chips",
    type=int,
    default=None,
    help="Minimum number of chips for hardware request",
)
@click.argument("overrides", nargs=-1)  # type: ignore[misc]
def configure(
    job: str,
    out_yaml: str,
    previous: str | None,
    chip: str | None,
    chips: int | None,
    overrides: tuple[str, ...],
) -> None:
    """Generate a configuration YAML for a job.

    JOB: Name of the job to generate config for
    OUT_YAML: Output path for the generated YAML config
    OVERRIDES: Optional config overrides in key=value format
    """
    # Validate job exists
    if job not in JOBS:
        console.print(f"\n[red]Error: Job '{job}' not found in registry[/red]")
        console.print(f"[yellow]Available jobs: {', '.join(JOBS.keys())}[/yellow]\n")
        sys.exit(1)

    # Validate chip if specified
    if chip and chip not in SUPPORTED_CHIPS:
        console.print(f"\n[red]Error: Unknown chip '{chip}'[/red]")
        console.print(
            f"[yellow]Available chips: {', '.join(SUPPORTED_CHIPS.keys())}[/yellow]\n"
        )
        sys.exit(1)

    job_obj = JOBS[job]

    # if job obj is iterable, spread it into build, otherwise just pass directly
    if isinstance(job_obj.config(), (list, tuple)):
        cfgs: List[Type[Any]] = job_obj.config()  # type: ignore
        config = build(*cfgs)
    else:
        config = build(job_obj.config())

    # If previous config provided, merge it
    if previous:
        if not Path(previous).exists():
            console.print(
                f"\n[red]Error: Previous config file '{previous}' not found[/red]\n"
            )
            sys.exit(1)

        prev_config = OmegaConf.load(previous)
        # Merge only fields that exist in the new config
        config = OmegaConf.merge(config, OmegaConf.masked_copy(prev_config, config))

    # Apply CLI overrides
    if overrides:
        cfg_cli = OmegaConf.from_dotlist(list(overrides))
        config = OmegaConf.merge(config, cfg_cli)

    # Add job name to config
    OmegaConf.set_struct(config, False)
    config.job = job

    # Add hardware request if specified
    if chip or chips:
        config.request = OmegaConf.create({})
        if chip:
            config.request.chip = chip
        if chips:
            config.request.min_chips = chips

    OmegaConf.set_struct(config, True)

    # Validate that output path parent exists
    out_path = Path(out_yaml)
    if not out_path.parent.exists():
        console.print(
            f"\n[red]Error: Parent directory '{out_path.parent}' does not exist[/red]\n"
        )
        sys.exit(1)

    # Write the config
    yaml_str = OmegaConf.to_yaml(config)
    out_path.write_text(yaml_str)

    # Print the output prettily
    console.print()
    if overrides:
        console.print("[yellow]Applied overrides:[/yellow]")
        for override in overrides:
            console.print(f"[yellow]  • {override}[/yellow]")
        console.print()
    console.print(f"[green]Generated config for job '{job}':[/green]")
    console.print(
        Syntax(yaml_str, "yaml", line_numbers=True, background_color="default")
    )
    console.print(f"Config written to: {out_yaml}")
    console.print()


@theseus.command()  # type: ignore[misc]
@click.argument("name")  # type: ignore[misc]
@click.argument("yaml_path")  # type: ignore[misc]
@click.argument("out_path")  # type: ignore[misc]
@click.option(
    "-j", "--job", default=None, help="Job name (read from YAML if not specified)"
)  # type: ignore[misc]
@click.option("-p", "--project", default=None, help="Project this run belongs to")  # type: ignore[misc]
@click.option(
    "-g", "--group", default=None, help="Group under the project this run belongs to"
)  # type: ignore[misc]
@click.argument("overrides", nargs=-1)  # type: ignore[misc]
def run(
    name: str,
    yaml_path: str,
    out_path: str,
    job: str | None,
    project: str | None,
    group: str | None,
    overrides: tuple[str, ...],
) -> None:
    """Run a job with a configuration file.

    NAME: Name of the job run
    YAML_PATH: Path to the configuration YAML file
    OUT_PATH: Output path for job results
    OVERRIDES: Optional config overrides in key=value format
    """
    # Load config file
    if not Path(yaml_path).exists():
        console.print(f"\n[red]Error: Config file '{yaml_path}' not found[/red]\n")
        sys.exit(1)

    cfg = OmegaConf.load(yaml_path)

    # Get job name from option or config
    if job is None:
        if "job" not in cfg:
            console.print(
                "\n[red]Error: No job specified and 'job' not found in config[/red]"
            )
            console.print(
                "[yellow]Use -j/--job option or add 'job: <name>' to your YAML[/yellow]\n"
            )
            sys.exit(1)
        job = cfg.job
    else:
        OmegaConf.set_struct(cfg, False)
        cfg.job = job
        OmegaConf.set_struct(cfg, True)

    # Validate job exists
    if job not in JOBS:
        console.print(f"\n[red]Error: Job '{job}' not found in registry[/red]")
        console.print(f"[yellow]Available jobs: {', '.join(JOBS.keys())}[/yellow]\n")
        sys.exit(1)

    job_obj = JOBS[job]

    # Apply CLI overrides
    if overrides:
        cfg_cli = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(cfg, cfg_cli)

        console.print()
        console.print("[yellow]Config Overrides:[/yellow]")
        for override in overrides:
            console.print(f"[yellow]  • {override}[/yellow]")
        console.print()

    # Print what we're about to dispatch
    console.print()
    console.print(f"[blue]Running job '{job}':[/blue]")
    console.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", background_color="default"))
    console.print(f"[blue]Output path:[/blue] {out_path}")
    console.print()

    # Run the job within the configuration context
    # BasicJob.__init__ will call configure() to hydrate args from the context
    with configuration(cfg):
        job_instance = job_obj.local(out_path, name=name, project=project, group=group)
        job_instance()

    console.print()
    console.print(f"\n[green]Job '{job}' completed successfully[/green]")
    console.print()


@theseus.command()  # type: ignore[misc]
@click.argument("name")  # type: ignore[misc]
@click.argument("yaml_path")  # type: ignore[misc]
@click.option(
    "-d",
    "--dispatch-config",
    default=None,
    help="Path to dispatch.yaml (default: ~/.theseus.yaml)",
)  # type: ignore[misc]
@click.option(
    "-j", "--job", default=None, help="Job name (read from YAML if not specified)"
)  # type: ignore[misc]
@click.option("-p", "--project", default=None, help="Project this run belongs to")  # type: ignore[misc]
@click.option(
    "-g", "--group", default=None, help="Group under the project this run belongs to"
)  # type: ignore[misc]
@click.option(
    "--chip", default=None, help=f"Chip type ({', '.join(SUPPORTED_CHIPS.keys())})"
)  # type: ignore[misc]
@click.option("-n", "--chips", type=int, default=None, help="Minimum number of chips")  # type: ignore[misc]
@click.option("--mem", default=None, help="Memory per job (e.g., '64G', '128G')")  # type: ignore[misc]
@click.option(
    "--cluster", default=None, help="Only use these clusters (comma-separated)"
)  # type: ignore[misc]
@click.option(
    "--exclude-cluster", default=None, help="Exclude these clusters (comma-separated)"
)  # type: ignore[misc]
@click.option("--dirty", is_flag=True, help="Include uncommitted changes")  # type: ignore[misc]
@click.argument("overrides", nargs=-1)  # type: ignore[misc]
def submit(
    name: str,
    yaml_path: str,
    dispatch_config: str | None,
    job: str | None,
    project: str | None,
    group: str | None,
    chip: str | None,
    chips: int | None,
    mem: str | None,
    cluster: str | None,
    exclude_cluster: str | None,
    dirty: bool,
    overrides: tuple[str, ...],
) -> None:
    """Submit a job to remote infrastructure via dispatch.

    NAME: Name of the job run
    YAML_PATH: Path to the configuration YAML file
    OVERRIDES: Optional config overrides in key=value format
    """
    from theseus.dispatch import dispatch, load_dispatch_config

    # Load config file
    if not Path(yaml_path).exists():
        console.print(f"\n[red]Error: Config file '{yaml_path}' not found[/red]\n")
        sys.exit(1)

    # Find dispatch config: CLI flag > ~/.theseus.yaml
    if dispatch_config is None:
        default_config = Path.home() / ".theseus.yaml"
        if default_config.exists():
            dispatch_config = str(default_config)
            console.print(f"[dim]Using dispatch config: {dispatch_config}[/dim]")
        else:
            console.print("\n[red]Error: No dispatch config specified[/red]")
            console.print(
                "[yellow]Use -d/--dispatch-config or create ~/.theseus.yaml[/yellow]\n"
            )
            sys.exit(1)

    # Load dispatch config
    if not Path(dispatch_config).exists():
        console.print(
            f"\n[red]Error: Dispatch config '{dispatch_config}' not found[/red]\n"
        )
        sys.exit(1)

    cfg = OmegaConf.load(yaml_path)
    dispatch_cfg = load_dispatch_config(dispatch_config)

    # Get job name from option or config
    if job is None:
        if "job" not in cfg:
            console.print(
                "\n[red]Error: No job specified and 'job' not found in config[/red]"
            )
            console.print(
                "[yellow]Use -j/--job option or add 'job: <name>' to your YAML[/yellow]\n"
            )
            sys.exit(1)
        job = cfg.job
    else:
        OmegaConf.set_struct(cfg, False)
        cfg.job = job
        OmegaConf.set_struct(cfg, True)

    # Validate job exists
    if job not in JOBS:
        console.print(f"\n[red]Error: Job '{job}' not found in registry[/red]")
        console.print(f"[yellow]Available jobs: {', '.join(JOBS.keys())}[/yellow]\n")
        sys.exit(1)

    # Get hardware request from CLI flags or config
    request_chip = chip
    request_chips = chips

    # Fall back to config.request if CLI flags not specified
    if request_chip is None and "request" in cfg and "chip" in cfg.request:
        request_chip = cfg.request.chip
    if request_chips is None and "request" in cfg and "min_chips" in cfg.request:
        request_chips = cfg.request.min_chips

    # Validate we have hardware request
    if request_chip is None:
        console.print("\n[red]Error: No chip specified[/red]")
        console.print(
            "[yellow]Use --chip option or add 'request.chip' to your YAML[/yellow]\n"
        )
        sys.exit(1)
    if request_chips is None:
        console.print("\n[red]Error: No chip count specified[/red]")
        console.print(
            "[yellow]Use -n/--chips option or add 'request.min_chips' to your YAML[/yellow]\n"
        )
        sys.exit(1)

    # Validate chip
    if request_chip not in SUPPORTED_CHIPS:
        console.print(f"\n[red]Error: Unknown chip '{request_chip}'[/red]")
        console.print(
            f"[yellow]Available chips: {', '.join(SUPPORTED_CHIPS.keys())}[/yellow]\n"
        )
        sys.exit(1)

    # Apply CLI overrides
    if overrides:
        cfg_cli = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(cfg, cfg_cli)

        console.print()
        console.print("[yellow]Config Overrides:[/yellow]")
        for override in overrides:
            console.print(f"[yellow]  • {override}[/yellow]")
        console.print()

    # Parse cluster filters
    preferred_clusters = [c.strip() for c in cluster.split(",")] if cluster else []
    forbidden_clusters = (
        [c.strip() for c in exclude_cluster.split(",")] if exclude_cluster else []
    )

    # Build hardware request
    hardware_request = HardwareRequest(
        chip=SUPPORTED_CHIPS[request_chip],
        min_chips=request_chips,
        preferred_clusters=preferred_clusters,
        forbidden_clusters=forbidden_clusters,
    )

    # Build job spec
    spec = JobSpec(
        name=name,
        project=project,
        group=group,
    )

    # Print what we're about to dispatch
    console.print()
    console.print(f"[blue]Submitting job '{job}':[/blue]")
    console.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", background_color="default"))
    console.print(f"[blue]Hardware:[/blue] {request_chips}x {request_chip}")
    if mem:
        console.print(f"[blue]Memory:[/blue] {mem}")
    if preferred_clusters:
        console.print(f"[blue]Clusters:[/blue] {', '.join(preferred_clusters)}")
    if forbidden_clusters:
        console.print(
            f"[blue]Excluded clusters:[/blue] {', '.join(forbidden_clusters)}"
        )
    console.print(f"[blue]Dispatch config:[/blue] {dispatch_config}")
    console.print(f"[blue]Dirty:[/blue] {dirty}")

    # Dispatch the job
    result = dispatch(
        cfg=cfg,
        spec=spec,
        hardware=hardware_request,
        dispatch_config=dispatch_cfg,
        dirty=dirty,
        mem=mem,
    )

    if not result.ok:
        console.print("\n[red]Job submission failed[/red]")
        stderr = getattr(result, "stderr", None) or getattr(
            getattr(result, "ssh_result", None), "stderr", ""
        )
        if stderr:
            console.print(f"[red]{stderr}[/red]")
        sys.exit(1)


@theseus.command()  # type: ignore[misc]
@click.argument("name")  # type: ignore[misc]
@click.argument("yaml_path")  # type: ignore[misc]
@click.argument("out_script")  # type: ignore[misc]
@click.option(
    "-j", "--job", default=None, help="Job name (read from YAML if not specified)"
)  # type: ignore[misc]
@click.option("-p", "--project", default=None, help="Project this run belongs to")  # type: ignore[misc]
@click.option(
    "-g", "--group", default=None, help="Group under the project this run belongs to"
)  # type: ignore[misc]
@click.option(
    "--chip", default=None, help=f"Chip type ({', '.join(SUPPORTED_CHIPS.keys())})"
)  # type: ignore[misc]
@click.option("-n", "--chips", type=int, default=None, help="Minimum number of chips")  # type: ignore[misc]
@click.option(
    "--root",
    default=None,
    help="Cluster root path (optional; if omitted, generated script requires --root at runtime)",
)  # type: ignore[misc]
@click.option("--work", default=None, help="Cluster work path (default: <root>/work)")  # type: ignore[misc]
@click.option("--log", default=None, help="Cluster log path (default: <work>/logs)")  # type: ignore[misc]
@click.option(
    "--mount",
    default=None,
    help="JuiceFS Redis URL to mount at root (default: disabled)",
)  # type: ignore[misc]
@click.option("--cache-size", default=None, help="JuiceFS cache size")  # type: ignore[misc]
@click.option("--cache-dir", default=None, help="JuiceFS cache directory")  # type: ignore[misc]
@click.option("--dirty", is_flag=True, help="Include uncommitted changes")  # type: ignore[misc]
@click.argument("overrides", nargs=-1)  # type: ignore[misc]
def bootstrap(
    name: str,
    yaml_path: str,
    out_script: str,
    job: str | None,
    project: str | None,
    group: str | None,
    chip: str | None,
    chips: int | None,
    root: str | None,
    work: str | None,
    log: str | None,
    mount: str | None,
    cache_size: str | None,
    cache_dir: str | None,
    dirty: bool,
    overrides: tuple[str, ...],
) -> None:
    """Generate a standalone bootstrap script (like SLURM submit payload).

    NAME: Name of the job run
    YAML_PATH: Path to the configuration YAML file
    OUT_SCRIPT: Output path for the generated bootstrap shell script
    OVERRIDES: Optional config overrides in key=value format
    """
    import subprocess

    from theseus.base.hardware import Cluster, ClusterMachine, HardwareResult
    from theseus.dispatch.config import JuiceFSMount
    from theseus.dispatch.dispatch import _generate_bootstrap
    from theseus.dispatch.slurm import SlurmJob
    from theseus.dispatch.sync import snapshot

    def normalize_path(path: str) -> str:
        path = path.strip()
        if path == "/":
            return "/"
        return path.rstrip("/")

    if not Path(yaml_path).exists():
        console.print(f"\n[red]Error: Config file '{yaml_path}' not found[/red]\n")
        sys.exit(1)

    cfg = OmegaConf.load(yaml_path)

    # Get job name from option or config
    if job is None:
        if "job" not in cfg:
            console.print(
                "\n[red]Error: No job specified and 'job' not found in config[/red]"
            )
            console.print(
                "[yellow]Use -j/--job option or add 'job: <name>' to your YAML[/yellow]\n"
            )
            sys.exit(1)
        job = cfg.job
    else:
        OmegaConf.set_struct(cfg, False)
        cfg.job = job
        OmegaConf.set_struct(cfg, True)

    if job not in JOBS:
        console.print(f"\n[red]Error: Job '{job}' not found in registry[/red]")
        console.print(f"[yellow]Available jobs: {', '.join(JOBS.keys())}[/yellow]\n")
        sys.exit(1)

    request_chip = chip
    request_chips = chips
    if request_chip is None and "request" in cfg and "chip" in cfg.request:
        request_chip = cfg.request.chip
    if request_chips is None and "request" in cfg and "min_chips" in cfg.request:
        request_chips = cfg.request.min_chips

    if request_chip is None:
        console.print("\n[red]Error: No chip specified[/red]")
        console.print(
            "[yellow]Use --chip option or add 'request.chip' to your YAML[/yellow]\n"
        )
        sys.exit(1)
    if request_chips is None:
        console.print("\n[red]Error: No chip count specified[/red]")
        console.print(
            "[yellow]Use -n/--chips option or add 'request.min_chips' to your YAML[/yellow]\n"
        )
        sys.exit(1)
    if request_chip not in SUPPORTED_CHIPS:
        console.print(f"\n[red]Error: Unknown chip '{request_chip}'[/red]")
        console.print(
            f"[yellow]Available chips: {', '.join(SUPPORTED_CHIPS.keys())}[/yellow]\n"
        )
        sys.exit(1)
    if request_chips < 1:
        console.print("\n[red]Error: --chips must be >= 1[/red]\n")
        sys.exit(1)

    if overrides:
        cfg_cli = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(cfg, cfg_cli)

        console.print()
        console.print("[yellow]Config Overrides:[/yellow]")
        for override in overrides:
            console.print(f"[yellow]  • {override}[/yellow]")
        console.print()

    root_placeholder = "__THESEUS_RUNTIME_ROOT__"
    if root is None:
        root_path = root_placeholder
        require_root_at_runtime = True
    else:
        root_path = normalize_path(root)
        require_root_at_runtime = False

    if not root_path:
        console.print("\n[red]Error: --root cannot be empty[/red]\n")
        sys.exit(1)
    work_path = normalize_path(work) if work else f"{root_path}/work"
    log_path = normalize_path(log) if log else f"{work_path}/logs"
    project_name = project or "general"
    group_name = group or "default"
    work_dir = f"{work_path}/{project_name}/{group_name}/{name}"

    cluster = Cluster(name="bootstrap", root=root_path, work=work_path, log=log_path)
    chip_obj = SUPPORTED_CHIPS[request_chip]
    hardware = HardwareResult(
        chip=chip_obj,
        hosts=[
            ClusterMachine(
                name="bootstrap",
                cluster=cluster,
                resources={chip_obj: request_chips},
            )
        ],
        total_chips=request_chips,
    )

    spec = JobSpec(name=name, project=project, group=group)
    bootstrap_py_content = _generate_bootstrap(cfg, hardware, spec)

    juicefs_mount = (
        JuiceFSMount(
            redis_url=mount,
            mount_point=root_path,
            cache_size=cache_size,
            cache_dir=cache_dir,
        )
        if mount
        else None
    )
    job_name = f"{project_name}-{group_name}-{name}"
    slurm_job = SlurmJob(
        name=job_name,
        command="python _bootstrap_dispatch.py",
        is_slurm=False,
        env={"THESEUS_DISPATCH_REQUIRE_ROOT": "1"} if require_root_at_runtime else {},
        payload_extract_to=work_dir,
        juicefs_mount=juicefs_mount,
        bootstrap_py=bootstrap_py_content,
    )

    if dirty:
        stash_result = subprocess.run(
            ["git", "stash", "create"],
            capture_output=True,
            text=True,
        )
        ref = stash_result.stdout.strip() or "HEAD"
    else:
        ref = "HEAD"
    tarball = snapshot(".", ref)
    script = slurm_job.pack(tarball).to_bootstrap_script()

    out_path = Path(out_script)
    if not out_path.parent.exists():
        console.print(
            f"\n[red]Error: Parent directory '{out_path.parent}' does not exist[/red]\n"
        )
        sys.exit(1)
    out_path.write_text(script)
    out_path.chmod(0o755)

    console.print()
    console.print(f"[green]Bootstrap script generated:[/green] {out_script}")
    console.print(f"[blue]Job:[/blue] {job}")
    console.print(f"[blue]Hardware:[/blue] {request_chips}x {request_chip}")
    root_display = (
        root_path if not require_root_at_runtime else "<runtime --root required>"
    )
    console.print(f"[blue]Root:[/blue] {root_display}")
    console.print(f"[blue]Work:[/blue] {work_path}")
    console.print(f"[blue]Log:[/blue] {log_path}")
    console.print(f"[blue]Mount:[/blue] {mount or 'disabled'}")
    console.print("[dim]Runtime overrides: --root/--work/--log[/dim]")
    console.print()


@theseus.command()  # type: ignore[misc]
@click.argument("name")  # type: ignore[misc]
@click.argument("out_path")  # type: ignore[misc]
@click.option("-p", "--project", default=None, help="Project this run belongs to")  # type: ignore[misc]
@click.option("-g", "--group", default=None, help="Group under the project")  # type: ignore[misc]
def checkpoints(
    name: str, out_path: str, project: str | None, group: str | None
) -> None:
    """List available checkpoints for a job.

    NAME: Name of the job
    OUT_PATH: Output path where checkpoints are stored
    """
    spec = ExecutionSpec.local(out_path, name=name, project=project, group=group)
    ckpts = RestoreableJob.checkpoints(spec)

    if not ckpts:
        console.print(f"\n[yellow]No checkpoints found for job '{name}'[/yellow]\n")
        return

    console.print()
    console.print(f"[blue]Checkpoints for job '{name}':[/blue]")
    for ckpt in sorted(ckpts):
        console.print(f"  [cyan]• {ckpt}[/cyan]")
    console.print()


@theseus.command()  # type: ignore[misc]
@click.argument("name")  # type: ignore[misc]
@click.argument("checkpoint")  # type: ignore[misc]
@click.argument("out_path")  # type: ignore[misc]
@click.option("-p", "--project", default=None, help="Project this run belongs to")  # type: ignore[misc]
@click.option("-g", "--group", default=None, help="Group under the project")  # type: ignore[misc]
def restore(
    name: str, checkpoint: str, out_path: str, project: str | None, group: str | None
) -> None:
    """Restore and run a job from a checkpoint.

    NAME: Name of the job
    CHECKPOINT: Checkpoint to restore from
    OUT_PATH: Output path where checkpoints are stored
    """
    spec = ExecutionSpec.local(out_path, name=name, project=project, group=group)

    console.print()
    console.print(
        f"[blue]Restoring job '{name}' from checkpoint '{checkpoint}'...[/blue]"
    )

    job: Any
    cfg: Any

    job, cfg = RestoreableJob.from_checkpoint(checkpoint, spec)

    console.print(
        f"[green]Restored job '{name}' from checkpoint '{checkpoint}'[/green]"
    )
    console.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", background_color="default"))
    console.print()

    with configuration(cfg):
        job()

    console.print(f"\n[green]Job '{name}' completed successfully[/green]\n")


if __name__ == "__main__":
    theseus()
