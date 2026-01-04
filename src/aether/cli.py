"""
AETHER CLI - Command Line Interface

The primary interface for the AETHER Band Engine.

Commands:
    aether new-project <name> [--genre <genre>]
    aether build-track <title> --genre <genre> --bpm <bpm> ...
    aether build-album --config <album.yaml>
    aether list-genres
    aether genre-info <genre_id>
    aether qa-check <track_id>
    aether originality-report <track_id>
"""

from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from aether import __version__
from aether.config import init_config, get_config

console = Console()


def print_banner() -> None:
    """Print the AETHER banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     █████╗ ███████╗████████╗██╗  ██╗███████╗██████╗          ║
    ║    ██╔══██╗██╔════╝╚══██╔══╝██║  ██║██╔════╝██╔══██╗         ║
    ║    ███████║█████╗     ██║   ███████║█████╗  ██████╔╝         ║
    ║    ██╔══██║██╔══╝     ██║   ██╔══██║██╔══╝  ██╔══██╗         ║
    ║    ██║  ██║███████╗   ██║   ██║  ██║███████╗██║  ██║         ║
    ║    ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝         ║
    ║                                                               ║
    ║    Autonomous Ensemble for Thoughtful                         ║
    ║    Harmonic Expression and Rendering                          ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


@click.group()
@click.version_option(version=__version__, prog_name="aether")
@click.option("--config", "-c", type=click.Path(exists=False), help="Config file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.pass_context
def main(ctx: click.Context, config: Optional[str], verbose: bool, debug: bool) -> None:
    """
    AETHER Band Engine - AI Music Creation System

    Generate commercially viable, 100% original music across any genre.
    """
    ctx.ensure_object(dict)

    # Initialize configuration
    config_path = Path(config) if config else None
    cfg = init_config(config_path)
    cfg.verbose = verbose
    cfg.debug = debug

    ctx.obj["config"] = cfg

    if verbose:
        print_banner()


# ============================================================================
# Project Commands
# ============================================================================


def _sanitize_project_name(name: str) -> str:
    """Sanitize project name to prevent path traversal attacks."""
    # Only allow alphanumeric, hyphen, and underscore
    sanitized = "".join(c for c in name if c.isalnum() or c in "-_")
    # Ensure not empty after sanitization
    if not sanitized:
        raise click.BadParameter("Project name must contain at least one alphanumeric character")
    # Ensure doesn't start with hyphen (could be interpreted as flag)
    if sanitized.startswith("-"):
        sanitized = "_" + sanitized[1:]
    return sanitized


@main.command("new-project")
@click.argument("name")
@click.option("--genre", "-g", default=None, help="Primary genre for the project")
@click.pass_context
def new_project(ctx: click.Context, name: str, genre: Optional[str]) -> None:
    """Create a new AETHER project."""
    config = ctx.obj["config"]

    # Sanitize project name to prevent path traversal
    safe_name = _sanitize_project_name(name)
    if safe_name != name:
        console.print(f"[yellow]Note:[/yellow] Project name sanitized to '{safe_name}'")

    project_dir = config.paths.get_absolute(config.paths.projects_dir) / safe_name

    if project_dir.exists():
        console.print(f"[red]Error:[/red] Project '{safe_name}' already exists")
        raise SystemExit(1)

    # Create project structure
    project_dir.mkdir(parents=True)
    (project_dir / "tracks").mkdir()
    (project_dir / "stems").mkdir()
    (project_dir / "masters").mkdir()
    (project_dir / "specs").mkdir()

    # Create project config
    project_config = {
        "name": safe_name,
        "genre": genre,
        "tracks": [],
    }

    import yaml

    with open(project_dir / "project.yaml", "w") as f:
        yaml.dump(project_config, f)

    console.print(
        Panel(
            f"[green]Created project:[/green] {safe_name}\n"
            f"[dim]Location:[/dim] {project_dir}\n"
            f"[dim]Genre:[/dim] {genre or 'Not specified'}",
            title="New Project",
            border_style="green",
        )
    )


@main.command("list-projects")
@click.pass_context
def list_projects(ctx: click.Context) -> None:
    """List all AETHER projects."""
    config = ctx.obj["config"]
    projects_dir = config.paths.get_absolute(config.paths.projects_dir)

    if not projects_dir.exists():
        console.print("[yellow]No projects found.[/yellow]")
        return

    table = Table(title="AETHER Projects")
    table.add_column("Name", style="cyan")
    table.add_column("Genre", style="green")
    table.add_column("Tracks", style="yellow")
    table.add_column("Path", style="dim")

    import yaml

    for project_dir in sorted(projects_dir.iterdir()):
        if project_dir.is_dir():
            config_file = project_dir / "project.yaml"
            if config_file.exists():
                with open(config_file) as f:
                    proj = yaml.safe_load(f)
                table.add_row(
                    proj.get("name", project_dir.name),
                    proj.get("genre", "-"),
                    str(len(proj.get("tracks", []))),
                    str(project_dir),
                )

    console.print(table)


# ============================================================================
# Track Commands
# ============================================================================


@main.command("build-track")
@click.argument("title")
@click.option("--genre", "-g", required=True, help="Genre ID")
@click.option(
    "--bpm", "-b", type=int, default=None, help="Tempo in BPM (optional, uses genre default)"
)
@click.option("--key", "-k", default=None, help="Musical key (e.g., 'Am', 'C')")
@click.option("--mood", "-m", default=None, help="Primary mood")
@click.option("--duration", "-d", type=int, default=210, help="Target duration in seconds")
@click.option("--brief", default=None, help="Creative brief text")
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility")
@click.option("--no-vocals", is_flag=True, help="Generate instrumental track (no vocals)")
@click.option("--no-save-state", is_flag=True, help="Don't save workflow state")
@click.pass_context
def build_track(
    ctx: click.Context,
    title: str,
    genre: str,
    bpm: Optional[int],
    key: Optional[str],
    mood: Optional[str],
    duration: int,
    brief: Optional[str],
    seed: Optional[int],
    no_vocals: bool,
    no_save_state: bool,
) -> None:
    """
    Build a complete track through the AETHER pipeline.

    This executes the full 10-agent pipeline:
    1. Creative Direction
    2. Composition (Harmony + Melody)
    3. Arrangement
    4. Lyrics (if vocals)
    5. Vocal Planning (if vocals)
    6. Sound Design
    7. Mixing
    8. Mastering
    9. QA
    10. Release Packaging
    """
    import asyncio
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.live import Live
    from aether.orchestration import MusicPipeline, WorkflowStatus

    # Build creative brief
    creative_brief = brief or f"A {mood or 'compelling'} {genre} track"

    console.print(
        Panel(
            f"[bold]Building Track:[/bold] {title}\n\n"
            f"[cyan]Genre:[/cyan] {genre}\n"
            f"[cyan]BPM:[/cyan] {bpm or 'Auto'}\n"
            f"[cyan]Key:[/cyan] {key or 'Auto'}\n"
            f"[cyan]Mood:[/cyan] {mood or 'Auto'}\n"
            f"[cyan]Duration:[/cyan] {duration}s\n"
            f"[cyan]Vocals:[/cyan] {'No' if no_vocals else 'Yes'}\n"
            f"[cyan]Seed:[/cyan] {seed or 'Random'}\n"
            f"[cyan]Brief:[/cyan] {creative_brief}",
            title="AETHER Track Builder",
            border_style="cyan",
        )
    )

    async def run_pipeline():
        pipeline = MusicPipeline()

        # Subscribe to events for progress display
        stages_completed = []
        current_stage = [None]

        def on_event(event):
            if event.event_type == "task_started":
                current_stage[0] = event.task_id
                console.print(f"  [cyan]▶[/cyan] Starting: {event.task_id}")
            elif event.event_type == "task_completed":
                stages_completed.append(event.task_id)
                duration_ms = event.data.get("duration_ms", 0)
                console.print(
                    f"  [green]✓[/green] Completed: {event.task_id} ({duration_ms:.0f}ms)"
                )
            elif event.event_type == "task_failed":
                error = event.data.get("error", "Unknown")
                console.print(f"  [red]✗[/red] Failed: {event.task_id} - {error}")
            elif event.event_type == "task_retrying":
                retry = event.data.get("retry_count", 0)
                console.print(f"  [yellow]↻[/yellow] Retrying: {event.task_id} (attempt {retry})")

        try:
            result = await pipeline.generate(
                title=title,
                genre=genre,
                creative_brief=creative_brief,
                bpm=bpm,
                key=key,
                mood=mood,
                duration_seconds=duration,
                has_vocals=not no_vocals,
                random_seed=seed,
                save_state=not no_save_state,
            )
            return result
        except Exception as e:
            console.print(f"\n[red]Pipeline failed:[/red] {e}")
            raise

    console.print("\n[bold]Executing Pipeline...[/bold]\n")

    try:
        result = asyncio.run(run_pipeline())

        # Display results
        status = result.get("status", "unknown")
        if status == "completed":
            console.print(
                Panel(
                    f"[bold green]Track Generated Successfully![/bold green]\n\n"
                    f"[cyan]Song ID:[/cyan] {result.get('song_id')}\n"
                    f"[cyan]Title:[/cyan] {result.get('title')}\n"
                    f"[cyan]Genre:[/cyan] {result.get('genre')}\n"
                    f"[cyan]QA Passed:[/cyan] {result.get('qa_report', {}).get('passed', 'N/A')}\n"
                    f"[cyan]Ready for Distribution:[/cyan] {result.get('ready_for_distribution', False)}",
                    title="Pipeline Complete",
                    border_style="green",
                )
            )

            # Show task summary
            task_results = result.get("task_results", {})
            if task_results:
                table = Table(title="Task Summary")
                table.add_column("Stage", style="cyan")
                table.add_column("Status", style="green")
                table.add_column("Duration", style="yellow")

                for task_id, task_data in task_results.items():
                    status_style = "green" if task_data.get("status") == "completed" else "red"
                    table.add_row(
                        task_id,
                        f"[{status_style}]{task_data.get('status', 'unknown')}[/{status_style}]",
                        f"{task_data.get('duration_ms', 0):.0f}ms",
                    )
                console.print(table)
        else:
            console.print(f"\n[red]Pipeline finished with status:[/red] {status}")

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        raise SystemExit(1)


@main.command("pipeline-list")
@click.pass_context
def pipeline_list(ctx: click.Context) -> None:
    """List all saved pipeline states."""
    from pathlib import Path
    import json

    state_dir = Path.home() / ".aether" / "workflows"

    if not state_dir.exists():
        console.print("[yellow]No pipelines found.[/yellow]")
        return

    table = Table(title="Saved Pipelines")
    table.add_column("Workflow ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Created", style="dim")

    for state_file in sorted(state_dir.glob("workflow_*.json"), reverse=True):
        try:
            with open(state_file) as f:
                state = json.load(f)
            status = state.get("status", "unknown")
            status_style = (
                "green" if status == "completed" else "yellow" if status == "running" else "red"
            )
            table.add_row(
                state.get("workflow_id", "unknown")[:8] + "...",
                state.get("name", "unnamed"),
                f"[{status_style}]{status}[/{status_style}]",
                state.get("created_at", "unknown")[:19],
            )
        except Exception:
            continue

    if table.row_count == 0:
        console.print("[yellow]No pipelines found.[/yellow]")
    else:
        console.print(table)
        console.print(f"\n[dim]State directory: {state_dir}[/dim]")


@main.command("pipeline-status")
@click.argument("workflow_id")
@click.pass_context
def pipeline_status(ctx: click.Context, workflow_id: str) -> None:
    """Show detailed status of a pipeline."""
    from pathlib import Path
    import json

    state_dir = Path.home() / ".aether" / "workflows"

    # Find matching workflow
    matching = list(state_dir.glob(f"workflow_{workflow_id}*.json"))
    if not matching:
        # Try partial match
        matching = [f for f in state_dir.glob("workflow_*.json") if workflow_id in f.name]

    if not matching:
        console.print(f"[red]Pipeline not found:[/red] {workflow_id}")
        return

    state_file = matching[0]
    with open(state_file) as f:
        state = json.load(f)

    # Display status panel
    status = state.get("status", "unknown")
    status_style = "green" if status == "completed" else "yellow" if status == "running" else "red"

    console.print(
        Panel(
            f"[bold]Workflow:[/bold] {state.get('name', 'unnamed')}\n\n"
            f"[cyan]ID:[/cyan] {state.get('workflow_id', 'unknown')}\n"
            f"[cyan]Status:[/cyan] [{status_style}]{status}[/{status_style}]\n"
            f"[cyan]Created:[/cyan] {state.get('created_at', 'unknown')}\n"
            f"[cyan]Updated:[/cyan] {state.get('updated_at', 'unknown')}\n"
            f"[cyan]Completed Tasks:[/cyan] {len(state.get('completed_tasks', []))}\n"
            f"[cyan]Failed Tasks:[/cyan] {len(state.get('failed_tasks', []))}",
            title="Pipeline Status",
            border_style=status_style,
        )
    )

    # Show task breakdown
    tasks = state.get("tasks", {})
    if tasks:
        table = Table(title="Tasks")
        table.add_column("Task ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Agent", style="magenta")
        table.add_column("Status", style="yellow")

        for task_id, task_data in tasks.items():
            task_status = task_data.get("status", "unknown")
            ts = (
                "green"
                if task_status == "completed"
                else (
                    "yellow"
                    if task_status in ["running", "queued"]
                    else "red" if task_status == "failed" else "dim"
                )
            )
            table.add_row(
                task_id,
                task_data.get("name", ""),
                task_data.get("agent_type", ""),
                f"[{ts}]{task_status}[/{ts}]",
            )
        console.print(table)


@main.command("pipeline-resume")
@click.argument("workflow_id")
@click.pass_context
def pipeline_resume(ctx: click.Context, workflow_id: str) -> None:
    """Resume a paused or failed pipeline."""
    import asyncio
    from pathlib import Path
    from aether.orchestration import WorkflowOrchestrator

    state_dir = Path.home() / ".aether" / "workflows"

    # Find matching workflow
    matching = list(state_dir.glob(f"workflow_{workflow_id}*.json"))
    if not matching:
        matching = [f for f in state_dir.glob("workflow_*.json") if workflow_id in f.name]

    if not matching:
        console.print(f"[red]Pipeline not found:[/red] {workflow_id}")
        return

    state_file = matching[0]
    console.print(f"[cyan]Loading pipeline from:[/cyan] {state_file}")

    try:
        workflow = WorkflowOrchestrator.load_state(state_file)

        async def run_resumed():
            # Re-register agent executors
            from aether.orchestration import PipelineAgentExecutor
            from aether.agents import get_pipeline_agents

            executor = PipelineAgentExecutor()
            for agent_type in get_pipeline_agents():
                workflow.register_agent(agent_type, executor)

            # Subscribe to events
            def on_event(event):
                if event.event_type == "task_started":
                    console.print(f"  [cyan]▶[/cyan] Resuming: {event.task_id}")
                elif event.event_type == "task_completed":
                    console.print(f"  [green]✓[/green] Completed: {event.task_id}")
                elif event.event_type == "task_failed":
                    console.print(f"  [red]✗[/red] Failed: {event.task_id}")

            workflow.event_bus.subscribe("*", on_event)

            return await workflow.run()

        console.print("\n[bold]Resuming Pipeline...[/bold]\n")
        results = asyncio.run(run_resumed())

        # Save updated state
        workflow.save_state()

        console.print(
            Panel(
                f"[bold green]Pipeline Resumed Successfully[/bold green]\n\n"
                f"[cyan]Status:[/cyan] {workflow.status.value}\n"
                f"[cyan]Completed:[/cyan] {len(workflow._completed_tasks)}\n"
                f"[cyan]Failed:[/cyan] {len(workflow._failed_tasks)}",
                title="Resume Complete",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(f"[red]Failed to resume pipeline:[/red] {e}")
        raise SystemExit(1)


@main.command("build-album")
@click.option(
    "--config", "-c", type=click.Path(exists=True), required=True, help="Album config YAML"
)
@click.pass_context
def build_album(ctx: click.Context, config: str) -> None:
    """Build a complete album from configuration."""
    console.print(f"[yellow]Album builder not yet implemented.[/yellow]")
    console.print(f"[dim]Config file: {config}[/dim]")


# ============================================================================
# Genre Commands
# ============================================================================


@main.command("list-genres")
@click.pass_context
def list_genres(ctx: click.Context) -> None:
    """List all available genre profiles."""
    config = ctx.obj["config"]
    genres_dir = config.paths.get_absolute(config.paths.genres_dir)

    table = Table(title="Available Genres")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("BPM Range", style="yellow")
    table.add_column("Era", style="magenta")

    # Built-in genres (will be loaded from files in Phase 3)
    built_in = [
        ("hip-hop-boom-bap", "Boom Bap", "80-100", "1990s"),
        ("synthwave", "Synthwave", "80-118", "1980s"),
        ("lo-fi-hip-hop", "Lo-Fi Hip Hop", "70-90", "2010s"),
    ]

    for genre_id, name, bpm, era in built_in:
        table.add_row(genre_id, name, bpm, era)

    console.print(table)
    console.print(f"\n[dim]Genre profiles directory: {genres_dir}[/dim]")


@main.command("genre-info")
@click.argument("genre_id")
@click.pass_context
def genre_info(ctx: click.Context, genre_id: str) -> None:
    """Show detailed information about a genre."""
    # TODO: Load actual genre profile
    console.print(f"[yellow]Genre info for '{genre_id}' not yet implemented.[/yellow]")
    console.print("[dim]Phase 3 will add genre profile loading.[/dim]")


# ============================================================================
# QA Commands
# ============================================================================


@main.command("qa-check")
@click.argument("track_id")
@click.pass_context
def qa_check(ctx: click.Context, track_id: str) -> None:
    """Run QA checks on a track."""
    console.print(f"[yellow]QA check for '{track_id}' not yet implemented.[/yellow]")
    console.print("[dim]Phase 7 will add the QA system.[/dim]")


@main.command("originality-report")
@click.argument("track_id")
@click.pass_context
def originality_report(ctx: click.Context, track_id: str) -> None:
    """Generate an originality report for a track."""
    console.print(f"[yellow]Originality report for '{track_id}' not yet implemented.[/yellow]")
    console.print("[dim]Phase 7 will add originality checking.[/dim]")


# ============================================================================
# Utility Commands
# ============================================================================


@main.command("init")
@click.pass_context
def init(ctx: click.Context) -> None:
    """Initialize AETHER in the current directory."""
    config = ctx.obj["config"]
    config.ensure_setup()

    tree = Tree("[bold]AETHER Initialized[/bold]")
    tree.add(f"[cyan]Base directory:[/cyan] {config.paths.base_dir}")
    tree.add(f"[cyan]Projects:[/cyan] {config.paths.get_absolute(config.paths.projects_dir)}")
    tree.add(f"[cyan]Output:[/cyan] {config.paths.get_absolute(config.paths.output_dir)}")
    tree.add(f"[cyan]Genres:[/cyan] {config.paths.get_absolute(config.paths.genres_dir)}")

    console.print(tree)


@main.command("status")
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show AETHER system status."""
    config = ctx.obj["config"]

    console.print(
        Panel(
            f"[bold]AETHER Band Engine[/bold] v{__version__}\n\n"
            f"[cyan]Status:[/cyan] Operational\n"
            f"[cyan]LLM Provider:[/cyan] {config.providers.llm_provider}\n"
            f"[cyan]Audio Provider:[/cyan] {config.providers.audio_provider}\n"
            f"[cyan]Debug Mode:[/cyan] {config.debug}\n",
            title="System Status",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
