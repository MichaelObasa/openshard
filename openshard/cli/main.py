import click


@click.group()
@click.version_option()
def cli():
    """OpenShard — intelligent task routing and execution."""


@cli.command()
@click.argument("task")
def plan(task: str):
    """Analyse TASK and produce an execution plan."""
    click.echo(f"[plan] Analysing task: {task!r}")


@cli.command()
@click.argument("task")
def run(task: str):
    """Execute TASK using the appropriate model tier."""
    click.echo(f"[run] Executing task: {task!r}")


@cli.command()
def report():
    """Display a summary report of recent executions."""
    click.echo("[report] Fetching execution report...")
