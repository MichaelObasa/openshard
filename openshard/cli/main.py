import click

from openshard.execution.runner import TaskRunner
from openshard.providers.openrouter import AuthError, OpenRouterError, RateLimitError


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
    try:
        runner = TaskRunner()
    except ValueError:
        raise click.ClickException(
            "OpenRouter API key is not configured.\n"
            "Set openrouter_api_key in config.yml or point the "
            "OPENSHARD_CONFIG environment variable to a config file that contains it."
        )

    try:
        response = runner.run(task)
    except AuthError:
        raise click.ClickException(
            "Authentication failed. Check that openrouter_api_key in config.yml is valid."
        )
    except RateLimitError:
        raise click.ClickException("Rate limit exceeded. Wait a moment then try again.")
    except OpenRouterError as exc:
        raise click.ClickException(f"API error: {exc}")

    click.echo(response.content)
    click.echo(
        f"\nTokens — prompt: {response.usage.prompt_tokens}, "
        f"completion: {response.usage.completion_tokens}, "
        f"total: {response.usage.total_tokens}"
    )


@cli.command()
def report():
    """Display a summary report of recent executions."""
    click.echo("[report] Fetching execution report...")
