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

from theseus.registry import JOBS
from theseus.config import build, hydrate

console = Console()

random.seed(0)
np.random.seed(0)

load_dotenv()
logger.remove()
logger.add(
    sys.stderr,
    format="<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> |"
    "<level>{level: ^8}</level>| "
    "<magenta>({name}:{line})</magenta> <level>{message}</level>",
    level="INFO",
    colorize=True,
    enqueue=True,
    filter=lambda x: x["extra"].get("task", "") != "plot",
)


@click.group()  # type: ignore[misc]
def theseus() -> None:
    """Theseus CLI for managing and running jobs."""
    pass


@theseus.command()  # type: ignore[misc]
def jobs() -> None:
    """List all available jobs in the registry."""
    if not JOBS:
        console.print("\n[yellow]No jobs registered[/yellow]\n")
        return

    table = Table(title="Available Jobs", show_header=True, header_style="bold green")
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
@click.argument("overrides", nargs=-1)  # type: ignore[misc]
def generate(
    job: str, out_yaml: str, previous: str | None, overrides: tuple[str, ...]
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

    job_obj = JOBS[job]
    config = build(job_obj.config)

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
@click.argument("job")  # type: ignore[misc]
@click.argument("yaml_path")  # type: ignore[misc]
@click.argument("out_path")  # type: ignore[misc]
@click.option("--help-config", is_flag=True, help="Show config schema for the job")  # type: ignore[misc]
@click.argument("overrides", nargs=-1)  # type: ignore[misc]
def run(
    job: str,
    yaml_path: str,
    out_path: str,
    help_config: bool,
    overrides: tuple[str, ...],
) -> None:
    """Run a job with a configuration file.

    JOB: Name of the job to run
    YAML_PATH: Path to the configuration YAML file
    OUT_PATH: Output path for job results
    OVERRIDES: Optional config overrides in key=value format
    """
    # Validate job exists
    if job not in JOBS:
        console.print(f"\n[red]Error: Job '{job}' not found in registry[/red]")
        console.print(f"[yellow]Available jobs: {', '.join(JOBS.keys())}[/yellow]\n")
        sys.exit(1)

    job_obj = JOBS[job]

    # Handle help-config flag
    if help_config:
        config_schema = build(job_obj.config)
        yaml_str = OmegaConf.to_yaml(config_schema)
        console.print()
        console.print(f"[cyan]Config schema for job '{job}':[/cyan]")
        console.print(
            Syntax(yaml_str, "yaml", line_numbers=True, background_color="default")
        )
        console.print()
        return

    # Load config file
    if not Path(yaml_path).exists():
        console.print(f"\n[red]Error: Config file '{yaml_path}' not found[/red]\n")
        sys.exit(1)

    cfg = OmegaConf.load(yaml_path)

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

    # Hydrate and run the job
    cfg = hydrate(job_obj.config, cfg)
    job_instance = job_obj.local(cfg, out_path)
    job_instance()

    console.print()
    console.print(f"\n[green]Job '{job}' completed successfully[/green]")
    console.print()


if __name__ == "__main__":
    theseus()
