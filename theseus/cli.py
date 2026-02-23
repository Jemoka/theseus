"""
a basic cli called `theseus` implemented in click
with nice `rich` driven printouts, etc. not intended
to be used for i.e. logging; and instead meant for
user-facing interactions
"""

import sys
import click
import inspect
from datetime import datetime
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


def _resolve_request_hardware(
    cfg: Any, chip: str | None, n_chips: int | None
) -> tuple[str | None, int]:
    """Resolve effective chip/min_chips from CLI + config with submit semantics."""
    request_chip = chip
    request_chips = n_chips

    # Fall back to config.request if CLI flags not specified
    if request_chip is None and "request" in cfg and "chip" in cfg.request:
        request_chip = cfg.request.chip
    if request_chips is None and "request" in cfg and "min_chips" in cfg.request:
        request_chips = cfg.request.min_chips

    # Default missing chip count to 1 for generic-GPU dispatch/bootstrap
    if request_chips is None:
        request_chips = 1

    if request_chips < 0:
        console.print("\n[red]Error: --n_chips must be >= 0[/red]\n")
        sys.exit(1)

    # Validate chip when provided
    if request_chip is not None and request_chip not in SUPPORTED_CHIPS:
        console.print(f"\n[red]Error: Unknown chip '{request_chip}'[/red]")
        console.print(
            f"[yellow]Available chips: {', '.join(SUPPORTED_CHIPS.keys())}[/yellow]\n"
        )
        sys.exit(1)

    # n_chips=0 means CPU job; ignore chip selector.
    if request_chips == 0:
        request_chip = None

    return request_chip, request_chips


def _jobs_registry() -> Any:
    """Lazy-load job registry to avoid heavy imports at CLI import time."""
    from theseus.registry import JOBS

    return JOBS


def _build_and_configuration() -> tuple[Any, Any]:
    """Lazy-load config helpers to avoid importing JAX-linked modules eagerly."""
    from theseus.config import build, configuration

    return build, configuration


def _restoreable_job() -> Any:
    """Lazy-load RestoreableJob to avoid heavy imports at CLI import time."""
    from theseus.job import RestoreableJob

    return RestoreableJob


@click.group()  # type: ignore[misc]
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")  # type: ignore[misc]
def theseus(verbose: bool) -> None:
    """Theseus CLI for managing and running jobs."""
    setup_logging(verbose)


@theseus.command()  # type: ignore[misc]
def jobs() -> None:
    """List all available jobs in the registry."""
    jobs = _jobs_registry()
    if not jobs:
        console.print("\n[yellow]No jobs registered[/yellow]\n")
        return

    table = Table(show_header=True, header_style="bold green")
    table.add_column("Job Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")

    for job_name in sorted(jobs.keys()):
        job_class = jobs[job_name]
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
    "--n_chips",
    type=int,
    default=None,
    help="Minimum number of chips for hardware request",
)
@click.option(  # type: ignore[misc]
    "--n_shards",
    type=int,
    default=None,
    help="Number of tensor parallel shards for the model",
)
@click.argument("overrides", nargs=-1)  # type: ignore[misc]
def configure(
    job: str,
    out_yaml: str,
    previous: str | None,
    chip: str | None,
    n_chips: int | None,
    n_shards: int | None,
    overrides: tuple[str, ...],
) -> None:
    """Generate a configuration YAML for a job.

    JOB: Name of the job to generate config for
    OUT_YAML: Output path for the generated YAML config
    OVERRIDES: Optional config overrides in key=value format
    """
    jobs = _jobs_registry()
    build, _ = _build_and_configuration()

    # Validate job exists
    if job not in jobs:
        console.print(f"\n[red]Error: Job '{job}' not found in registry[/red]")
        console.print(f"[yellow]Available jobs: {', '.join(jobs.keys())}[/yellow]\n")
        sys.exit(1)

    # Validate chip if specified
    if chip and chip not in SUPPORTED_CHIPS:
        console.print(f"\n[red]Error: Unknown chip '{chip}'[/red]")
        console.print(
            f"[yellow]Available chips: {', '.join(SUPPORTED_CHIPS.keys())}[/yellow]\n"
        )
        sys.exit(1)

    job_obj = jobs[job]

    # if job obj is iterable, spread it into build, otherwise just pass directly
    if isinstance(job_obj.config(), (list, tuple)):
        cfgs: List[Type[Any]] = job_obj.config()
        config = build(*cfgs)
    else:
        config = build(job_obj.config())

    def _keep_known_keys(src: Any, template: Any) -> Any:
        """Return a copy of src containing only keys present in template.

        This lets us safely merge a previous config onto a new schema while
        dropping fields the new job doesn't know about. For nested containers we
        recurse, and for leaves we keep the src value.
        """

        from omegaconf import DictConfig, ListConfig

        # If template is a dict-like config, only keep intersecting keys
        if isinstance(template, DictConfig):
            if not isinstance(src, (DictConfig, dict)):
                return OmegaConf.create({})
            kept: dict[str, Any] = {}
            for key in template.keys():
                if key in src:
                    kept[key] = _keep_known_keys(src[key], template[key])
            return OmegaConf.create(kept)

        # If template is a list-like config, keep aligned positions only.
        # When the new schema leaves the list empty, carry over the whole src list.
        if isinstance(template, ListConfig):
            if not isinstance(src, (ListConfig, list, tuple)):
                return OmegaConf.create([])
            if len(template) == 0:
                return OmegaConf.create(list(src))
            kept_list = []
            for idx, tmpl_item in enumerate(template):
                if idx < len(src):
                    kept_list.append(_keep_known_keys(src[idx], tmpl_item))
            return OmegaConf.create(kept_list)

        # Leaf node: use value from src
        return src

    # If previous config provided, merge it
    if previous:
        if not Path(previous).exists():
            console.print(
                f"\n[red]Error: Previous config file '{previous}' not found[/red]\n"
            )
            sys.exit(1)

        prev_config = OmegaConf.load(previous)
        # Merge only fields that exist in the new config schema
        safe_prev = _keep_known_keys(prev_config, config)
        config = OmegaConf.merge(config, safe_prev)

    # Apply CLI overrides
    if overrides:
        cfg_cli = OmegaConf.from_dotlist(list(overrides))
        config = OmegaConf.merge(config, cfg_cli)

    # Add job name to config
    OmegaConf.set_struct(config, False)
    config.job = job

    if n_chips is not None and n_chips < 0:
        console.print("\n[red]Error: --n_chips must be >= 0[/red]\n")
        sys.exit(1)

    # n_chips=0 means CPU request; chip selector is ignored.
    if n_chips == 0:
        chip = None

    # Add hardware request if specified
    if chip is not None or n_chips is not None or n_shards is not None:
        config.request = OmegaConf.create({})
        if chip:
            config.request.chip = chip
        if n_chips is not None:
            config.request.min_chips = n_chips
        if n_shards is not None:
            config.request.n_shards = n_shards

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
    jobs = _jobs_registry()
    _, configuration = _build_and_configuration()

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
    if job not in jobs:
        console.print(f"\n[red]Error: Job '{job}' not found in registry[/red]")
        console.print(f"[yellow]Available jobs: {', '.join(jobs.keys())}[/yellow]\n")
        sys.exit(1)

    job_obj = jobs[job]

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
@click.option("-n", "--n_chips", type=int, default=None, help="Minimum number of chips")  # type: ignore[misc]
@click.option(
    "--n_shards",
    type=int,
    default=None,
    help="Number of tensor parallel shards for the model",
)  # type: ignore[misc]
@click.option("--mem", default=None, help="Memory per job (e.g., '64G', '128G')")  # type: ignore[misc]
@click.option(
    "--cluster", default=None, help="Only use these clusters (comma-separated)"
)  # type: ignore[misc]
@click.option(
    "--exclude-cluster", default=None, help="Exclude these clusters (comma-separated)"
)  # type: ignore[misc]
@click.option(
    "--dirty/--clean",
    default=True,
    help="Include uncommitted changes (default: --dirty)",
)  # type: ignore[misc]
@click.argument("overrides", nargs=-1)  # type: ignore[misc]
def submit(
    name: str,
    yaml_path: str,
    dispatch_config: str | None,
    job: str | None,
    project: str | None,
    group: str | None,
    chip: str | None,
    n_chips: int | None,
    n_shards: int | None,
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

    jobs = _jobs_registry()

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
    if job not in jobs:
        console.print(f"\n[red]Error: Job '{job}' not found in registry[/red]")
        console.print(f"[yellow]Available jobs: {', '.join(jobs.keys())}[/yellow]\n")
        sys.exit(1)

    # Get hardware request from CLI flags or config
    request_chip, request_chips = _resolve_request_hardware(cfg, chip, n_chips)
    request_n_shards = n_shards

    # Fall back to config.request for shards if CLI flag not specified
    if request_n_shards is None and "request" in cfg and "n_shards" in cfg.request:
        request_n_shards = cfg.request.n_shards

    if request_n_shards is not None:
        OmegaConf.set_struct(cfg, False)
        if "request" not in cfg:
            cfg.request = OmegaConf.create({})
        cfg.request.n_shards = request_n_shards
        OmegaConf.set_struct(cfg, True)

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
        chip=SUPPORTED_CHIPS[request_chip] if request_chip is not None else None,
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
    if request_chips == 0:
        console.print("[blue]Hardware:[/blue] cpu-only (n_chips=0)")
    elif request_chip is None:
        console.print(f"[blue]Hardware:[/blue] {request_chips}x any-gpu")
    else:
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
@click.option(
    "-d",
    "--dispatch-config",
    default=None,
    help="Path to dispatch.yaml (default: ~/.theseus.yaml)",
)  # type: ignore[misc]
@click.option(
    "--chip",
    default=None,
    help=f"Chip type ({', '.join(SUPPORTED_CHIPS.keys())})",
)  # type: ignore[misc]
@click.option(
    "-n",
    "--n_chips",
    type=int,
    default=None,
    help="Minimum number of chips",
)  # type: ignore[misc]
@click.option(
    "--n_shards",
    type=int,
    default=None,
    help="Number of tensor parallel shards for the model (accepted for parity)",
)  # type: ignore[misc]
@click.option("--mem", default=None, help="Memory per job (e.g., '64G', '128G')")  # type: ignore[misc]
@click.option(
    "--cluster", default=None, help="Only use these clusters (comma-separated)"
)  # type: ignore[misc]
@click.option(
    "--exclude-cluster", default=None, help="Exclude these clusters (comma-separated)"
)  # type: ignore[misc]
@click.option(
    "--dirty/--clean",
    default=True,
    help="Include uncommitted changes (default: --dirty)",
)  # type: ignore[misc]
@click.option(
    "--sync",
    "sync_mode",
    is_flag=True,
    help="Launch REPL with mailbox sync sidecar enabled",
)  # type: ignore[misc]
@click.option(
    "--update",
    "update_mode",
    is_flag=True,
    help="Send mailbox patches to active synced REPL jobs",
)  # type: ignore[misc]
@click.option(
    "--port",
    type=int,
    default=8888,
    help="Local port to use for notebook access (SSH target)",
)  # type: ignore[misc]
@click.option(
    "--startup-timeout",
    type=float,
    default=180.0,
    help="Seconds to wait for Jupyter token/URL in logs",
)  # type: ignore[misc]
@click.option(
    "--slurm-wait-timeout",
    type=float,
    default=None,
    help="Optional timeout (seconds) while waiting for SLURM allocation",
)  # type: ignore[misc]
def repl(
    dispatch_config: str | None,
    chip: str | None,
    n_chips: int | None,
    n_shards: int | None,
    mem: str | None,
    cluster: str | None,
    exclude_cluster: str | None,
    dirty: bool,
    sync_mode: bool,
    update_mode: bool,
    port: int,
    startup_timeout: float,
    slurm_wait_timeout: float | None,
) -> None:
    """Start a remote Jupyter REPL on selected dispatch infrastructure."""
    from theseus.dispatch import dispatch_repl, load_dispatch_config
    from theseus.dispatch.mailbox import (
        ensure_local_mount,
        is_local_juicefs_mounted,
        list_active_entries,
        mailbox_display_root,
        publish_updates,
        register_synced_repl,
        require_git_repo,
    )

    console.print()
    if sync_mode and update_mode:
        console.print(
            "\n[red]Error: --sync and --update are mutually exclusive[/red]\n"
        )
        sys.exit(1)

    request_chip, request_chips = _resolve_request_hardware(
        OmegaConf.create({}), chip, n_chips
    )
    if port <= 0:
        console.print("\n[red]Error: --port must be > 0[/red]\n")
        sys.exit(1)

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

    if not Path(dispatch_config).exists():
        console.print(
            f"\n[red]Error: Dispatch config '{dispatch_config}' not found[/red]\n"
        )
        sys.exit(1)

    dispatch_cfg = load_dispatch_config(dispatch_config)
    local_mount = Path(dispatch_cfg.mount).expanduser() if dispatch_cfg.mount else None
    proxy = dispatch_cfg.proxy

    preferred_clusters = [c.strip() for c in cluster.split(",")] if cluster else []
    forbidden_clusters = (
        [c.strip() for c in exclude_cluster.split(",")] if exclude_cluster else []
    )

    if update_mode:
        if local_mount is None and proxy is None:
            console.print(
                "\n[red]Error: dispatch config top-level 'mount' or 'proxy' is required for --update[/red]\n"
            )
            sys.exit(1)
        try:
            repo_root = require_git_repo(Path.cwd())
        except RuntimeError as exc:
            console.print(f"\n[red]Error: {exc}[/red]\n")
            sys.exit(1)

        entries, _ = list_active_entries(
            local_mount=local_mount,
            proxy=proxy,
            include_clusters=set(preferred_clusters) if preferred_clusters else None,
            exclude_clusters=set(forbidden_clusters) if forbidden_clusters else None,
        )

        logger.debug(
            f"REPL UPDATE | active synced entries after filter: {len(entries)}"
        )
        if not entries:
            console.print("\n[yellow]No active synced REPL jobs found.[/yellow]\n")
            return

        backend_ids: set[str] = set()
        for entry in entries:
            backend = str(entry.get("backend", "")).strip()
            if backend:
                backend_ids.add(backend)

        if len(backend_ids) > 1:
            console.print(
                "\n[red]Error: active synced REPLs span multiple JuiceFS backends.[/red]"
            )
            for backend in sorted(backend_ids):
                console.print(f"[red]- {backend}[/red]")
            console.print()
            sys.exit(1)

        backend = next(iter(backend_ids), "")
        if not backend:
            console.print(
                "\n[red]Error: missing cluster.mount backend for active synced REPLs[/red]\n"
            )
            sys.exit(1)
        if proxy is None:
            assert local_mount is not None
            try:
                ensure_local_mount(local_mount, backend)
            except RuntimeError as exc:
                console.print(f"\n[red]Error: {exc}[/red]\n")
                sys.exit(1)

        sender_host = Path.cwd().name
        any_sent = False
        # console.print()
        console.print("[blue]Publishing mailbox updates:[/blue]")
        logger.debug("REPL UPDATE | publishing active mailbox entries")
        summaries, mailbox_root = publish_updates(
            local_mount=local_mount,
            proxy=proxy,
            repo_root=repo_root,
            sender_host=sender_host,
            include_clusters=set(preferred_clusters) if preferred_clusters else None,
            exclude_clusters=set(forbidden_clusters) if forbidden_clusters else None,
        )
        if summaries:
            console.print(
                f"[blue]Mailbox:[/blue] {mailbox_display_root(local_mount=local_mount, proxy=proxy)}"
            )
            for item in summaries:
                console.print(f"  {item.job_id}: {item.status} ({item.detail})")
                if item.status == "sent":
                    any_sent = True
        if not any_sent:
            console.print("[yellow]No patches sent.[/yellow]")
        console.print()
        return
    hardware_request = HardwareRequest(
        chip=SUPPORTED_CHIPS[request_chip] if request_chip is not None else None,
        min_chips=request_chips,
        preferred_clusters=preferred_clusters,
        forbidden_clusters=forbidden_clusters,
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    chip_label = (
        request_chip
        if request_chip is not None
        else "cpu"
        if request_chips == 0
        else "any"
    )
    spec = JobSpec(
        name=f"repl-{chip_label}-{timestamp}",
        project="repl",
        group="interactive",
    )

    console.print()
    console.print("[blue]Starting remote Jupyter REPL:[/blue]")
    if request_chips == 0:
        console.print("[blue]Hardware:[/blue] cpu-only (n_chips=0)")
    elif request_chip is None:
        console.print(f"[blue]Hardware:[/blue] {request_chips}x any-gpu")
    else:
        console.print(f"[blue]Hardware:[/blue] {request_chips}x {request_chip}")
    console.print(f"[blue]Dispatch config:[/blue] {dispatch_config}")
    console.print(f"[blue]Dirty:[/blue] {dirty}")
    console.print(f"[blue]Local port:[/blue] {port}")
    if n_shards is not None:
        console.print(
            "[dim]--n_shards is accepted for parity and ignored in repl[/dim]"
        )
    if mem:
        console.print(f"[blue]Memory:[/blue] {mem}")
    if preferred_clusters:
        console.print(f"[blue]Clusters:[/blue] {', '.join(preferred_clusters)}")
    if forbidden_clusters:
        console.print(
            f"[blue]Excluded clusters:[/blue] {', '.join(forbidden_clusters)}"
        )
    if sync_mode:
        console.print("[blue]Sync mode:[/blue] enabled")

    result = dispatch_repl(
        spec=spec,
        hardware=hardware_request,
        dispatch_config=dispatch_cfg,
        local_port=port,
        dirty=dirty,
        mem=mem,
        startup_timeout=startup_timeout,
        slurm_wait_timeout=slurm_wait_timeout,
        sync_enabled=sync_mode,
    )

    if not result.ok:
        console.print("\n[red]REPL launch failed[/red]")
        if result.stderr:
            console.print(f"[red]{result.stderr}[/red]")
        if result.log_path:
            console.print(
                f"[yellow]Log path:[/yellow] {result.ssh_host}:{result.log_path}"
            )
        sys.exit(1)

    if sync_mode:
        if local_mount is None and proxy is None:
            console.print(
                "\n[yellow]REPL is running, but sync registration failed: dispatch config top-level 'mount' or 'proxy' is missing.[/yellow]"
            )
            console.print(
                "[yellow]Run --update only after adding top-level mount/proxy.[/yellow]\n"
            )
        elif not result.cluster_root or not result.mailbox_job_id:
            console.print(
                "\n[yellow]REPL is running, but sync registration skipped due to missing cluster/job metadata.[/yellow]\n"
            )
        else:
            sync_repo_root: Path | None
            try:
                sync_repo_root = require_git_repo(Path.cwd())
            except RuntimeError as exc:
                console.print(
                    f"\n[yellow]REPL is running, but sync registration failed: {exc}[/yellow]\n"
                )
                sync_repo_root = None
            backend = (result.cluster_mount or "").strip()
            if sync_repo_root is None:
                pass
            elif not backend:
                console.print(
                    "\n[yellow]REPL is running, but sync registration failed: target cluster has no JuiceFS backend (clusters.<name>.mount).[/yellow]\n"
                )
            else:
                try:
                    if proxy is None:
                        assert local_mount is not None
                        local_mount_path = local_mount
                    if proxy is None and not is_local_juicefs_mounted(local_mount_path):
                        console.print(
                            "\n[yellow]REPL is running, but sync registration skipped: local top-level mount is not currently mounted as JuiceFS. Use --update to mount/send.[/yellow]\n"
                        )
                    else:
                        console.print(
                            "[dim]Registering synced REPL mailbox state...[/dim]"
                        )
                        register_synced_repl(
                            local_mount=local_mount,
                            proxy=proxy,
                            cluster_root=result.cluster_root,
                            cluster_name=result.cluster_name or result.selected_host,
                            job_id=result.mailbox_job_id,
                            ssh_alias=result.ssh_host,
                            host_key=result.selected_host,
                            workdir=result.work_dir or "",
                            backend=backend,
                            repo_root=sync_repo_root,
                            initialize_shadow=False,
                        )
                except RuntimeError as exc:
                    console.print(
                        f"\n[yellow]REPL is running, but sync registration failed: {exc}[/yellow]\n"
                    )
                else:
                    if proxy is not None:
                        ready = True
                    else:
                        assert local_mount is not None
                        local_mount_path = local_mount
                        ready = is_local_juicefs_mounted(local_mount_path)
                    if ready:
                        console.print("[dim]Sync registration complete.[/dim]")
                        console.print(
                            f"[blue]Mailbox root:[/blue] {mailbox_display_root(local_mount=local_mount, proxy=proxy)}"
                        )
                        console.print(
                            f"[blue]Mailbox job id:[/blue] {result.mailbox_job_id} (registered for --update)"
                        )
                        if result.work_dir:
                            console.print(
                                f"[blue]Sidecar log:[/blue] {result.ssh_host}:{result.work_dir}/.theseus_repl_sidecar.log"
                            )

    console.print()
    if result.is_slurm:
        console.print("[green]REPL started on SLURM[/green]")
        console.print(f"[blue]SLURM login host:[/blue] {result.ssh_host}")
        console.print(f"[blue]SLURM host key:[/blue] {result.selected_host}")
        if result.job_id is not None:
            console.print(f"[blue]SLURM job id:[/blue] {result.job_id}")
        if result.allocated_hostname:
            console.print(
                f"[blue]Allocated hostname:[/blue] {result.allocated_hostname}"
            )
        if result.remote_port:
            console.print(f"[blue]Remote notebook port:[/blue] {result.remote_port}")
        if result.token:
            console.print(f"[blue]Notebook token:[/blue] {result.token}")
        if result.allocated_hostname and result.remote_port:
            url = f"http://{result.allocated_hostname}:{result.remote_port}/lab"
            if result.token:
                url = f"{url}?token={result.token}"
            console.print(f"[blue]Notebook URL:[/blue] {url}")
        if not result.token:
            console.print(
                "[dim]No token detected in logs; authenticate with password if prompted.[/dim]"
            )
        console.print(f"[blue]Log path:[/blue] {result.ssh_host}:{result.log_path}")
        console.print()
        console.print("[yellow]Stop this notebook:[/yellow]")
        if result.job_id is not None:
            console.print(f"  ssh {result.ssh_host} 'scancel {result.job_id}'")
            console.print(f"  ssh {result.ssh_host} 'squeue -j {result.job_id}'")
    else:
        console.print("[green]REPL started on SSH target[/green]")
        console.print(f"[blue]SSH host key:[/blue] {result.selected_host}")
        console.print(f"[blue]SSH alias:[/blue] {result.ssh_host}")
        if result.remote_port:
            console.print(f"[blue]Remote notebook port:[/blue] {result.remote_port}")
        if result.token:
            console.print(f"[blue]Notebook token:[/blue] {result.token}")
        else:
            console.print(
                "[dim]No token detected in logs; authenticate with password if prompted.[/dim]"
            )
        if result.local_url:
            console.print(f"[blue]Open locally:[/blue] {result.local_url}")
        if result.remote_pid:
            console.print(f"[blue]Remote notebook PID:[/blue] {result.remote_pid}")
        if result.tunnel_pid:
            console.print(f"[blue]Local tunnel PID:[/blue] {result.tunnel_pid}")
        console.print(f"[blue]Log path:[/blue] {result.ssh_host}:{result.log_path}")
        console.print()
        console.print("[yellow]Stop this notebook:[/yellow]")
        if result.tunnel_pid:
            console.print(f"  kill {result.tunnel_pid}")
        if result.remote_pid:
            console.print(f"  ssh {result.ssh_host} 'kill {result.remote_pid}'")
        if result.remote_port:
            console.print(
                f"  ssh {result.ssh_host} 'lsof -ti :{result.remote_port} | xargs -r kill'"
            )

    console.print()


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
@click.option("-n", "--n_chips", type=int, default=None, help="Minimum number of chips")  # type: ignore[misc]
@click.option(
    "--n_shards",
    type=int,
    default=None,
    help="Number of tensor parallel shards for the model",
)  # type: ignore[misc]
@click.option(
    "--target",
    "uv_targets",
    multiple=True,
    help="Dependency target group (cpu/cuda12/cuda13/tpu); can repeat. Always includes 'all' by default.",
)  # type: ignore[misc]
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
@click.option(
    "--dirty/--clean",
    default=True,
    help="Include uncommitted changes (default: --dirty)",
)  # type: ignore[misc]
@click.argument("overrides", nargs=-1)  # type: ignore[misc]
def bootstrap(
    name: str,
    yaml_path: str,
    out_script: str,
    job: str | None,
    project: str | None,
    group: str | None,
    chip: str | None,
    n_chips: int | None,
    n_shards: int | None,
    root: str | None,
    work: str | None,
    log: str | None,
    mount: str | None,
    cache_size: str | None,
    cache_dir: str | None,
    dirty: bool,
    overrides: tuple[str, ...],
    uv_targets: tuple[str, ...],
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

    jobs = _jobs_registry()

    custom_header = """#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# About This Script:
#   Theseus is a CLI + Python framework for configuring, running, and dispatching
#   machine-learning jobs locally or on remote infrastructure (SLURM/SSH).
#   This file is a generated bootstrap runner that executes one packed Theseus job.
#
#   *NOTE*: If you think "wow someone handed me a shell script that dumps a scary base64
#   encoded string and then runs it" you should indeed be afraid and then use a sanitizer
#   to decode the blob. And then you will find (disappointingly) that its just a tarball of a
#   git archive of all source files required to run this program which instead of being some
#   kind of sick zero day is just a pile of Python that trains an LLM.
#
# Usage:
#   ./bootstrap.sh --root /path/to/root [--work /path/to/work] [--log /path/to/log]
#   ./bootstrap.sh --help
#
# Note:
#   If this script was generated without --root, runtime --root is required.
#   
# Generated by github.com/Jemoka/theseus

"""

    def with_custom_header(script_text: str) -> str:
        lines = script_text.splitlines()
        if not lines:
            return script_text

        shebang = lines[0] if lines[0].startswith("#!") else "#!/bin/bash"

        body_start = 1
        while (
            body_start < len(lines) and lines[body_start].strip() != "set -euo pipefail"
        ):
            body_start += 1
        body_lines = lines[body_start:] if body_start < len(lines) else lines[1:]
        body_text = "\n".join(body_lines)
        return f"{shebang}\n{custom_header}{body_text}\n"

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

    if job not in jobs:
        console.print(f"\n[red]Error: Job '{job}' not found in registry[/red]")
        console.print(f"[yellow]Available jobs: {', '.join(jobs.keys())}[/yellow]\n")
        sys.exit(1)

    request_chip, request_chips = _resolve_request_hardware(cfg, chip, n_chips)
    request_n_shards = n_shards
    if request_n_shards is None and "request" in cfg and "n_shards" in cfg.request:
        request_n_shards = cfg.request.n_shards

    if request_n_shards is not None:
        OmegaConf.set_struct(cfg, False)
        if "request" not in cfg:
            cfg.request = OmegaConf.create({})
        cfg.request.n_shards = request_n_shards
        OmegaConf.set_struct(cfg, True)

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
    chip_obj = SUPPORTED_CHIPS[request_chip] if request_chip is not None else None
    resources = {chip_obj: request_chips} if chip_obj is not None else {}
    hardware = HardwareResult(
        chip=chip_obj,
        hosts=[
            ClusterMachine(
                name="bootstrap",
                cluster=cluster,
                resources=resources,
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
    slurm_env: dict[str, str] = {}
    if require_root_at_runtime:
        slurm_env["THESEUS_DISPATCH_REQUIRE_ROOT"] = "1"
    # Always include 'all'; append user targets if provided.
    effective_uv_groups = ["all"]
    if uv_targets:
        effective_uv_groups.extend(list(uv_targets))

    slurm_job = SlurmJob(
        name=job_name,
        command="python _bootstrap_dispatch.py",
        is_slurm=False,
        env=slurm_env,
        root_dir=root_path,
        payload_extract_to=work_dir,
        juicefs_mount=juicefs_mount,
        bootstrap_py=bootstrap_py_content,
        uv_groups=effective_uv_groups,
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
    script = with_custom_header(script)

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
    if request_chips == 0:
        console.print("[blue]Hardware:[/blue] cpu-only (n_chips=0)")
    elif request_chip is None:
        console.print(f"[blue]Hardware:[/blue] {request_chips}x any-gpu")
    else:
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
    RestoreableJob = _restoreable_job()
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
    RestoreableJob = _restoreable_job()
    _, configuration = _build_and_configuration()
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
