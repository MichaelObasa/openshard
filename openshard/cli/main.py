import click

from openshard.execution.runner import TaskRunner
from openshard.planning.generator import PlanGenerator
from openshard.providers.openrouter import AuthError, OpenRouterError, RateLimitError


@click.group()
@click.version_option()
def cli():
    """OpenShard - intelligent task routing and execution."""


@cli.command()
@click.argument("task")
def plan(task: str):
    """Analyse TASK and produce a structured execution plan."""
    try:
        generator = PlanGenerator()
    except ValueError as exc:
        raise click.ClickException(str(exc))

    try:
        result = generator.generate(task)
    except AuthError:
        raise click.ClickException(
            "Authentication failed. Check that OPENROUTER_API_KEY is valid."
        )
    except RateLimitError:
        raise click.ClickException("Rate limit exceeded. Wait a moment then try again.")
    except OpenRouterError as exc:
        raise click.ClickException(f"API error: {exc}")

    click.echo(f"\nTask: {task}\n")
    click.echo(f"Summary: {result.summary}\n")
    for i, stage in enumerate(result.stages, 1):
        click.echo(f"Stage {i}: {stage.name} [{stage.tier}]")
        click.echo(f"  {stage.reasoning}")


@cli.command()
@click.argument("task")
def run(task: str):
    """Execute TASK using the appropriate model tier."""
    try:
        runner = TaskRunner()
    except ValueError:
        raise click.ClickException(
            "OpenRouter API key is not configured.\n"
            "Set OPENROUTER_API_KEY environment variable."
        )

    try:
        response = runner.run(task)
    except AuthError:
        raise click.ClickException(
            "Authentication failed. Check that OPENROUTER_API_KEY is valid."
        )
    except RateLimitError:
        raise click.ClickException("Rate limit exceeded. Wait a moment then try again.")
    except OpenRouterError as exc:
        raise click.ClickException(f"API error: {exc}")

    click.echo(response.content)
    click.echo(
        f"\nTokens - prompt: {response.usage.prompt_tokens}, "
        f"completion: {response.usage.completion_tokens}, "
        f"total: {response.usage.total_tokens}"
    )


@cli.command()
def report():
    """Display a summary report of recent executions."""
    click.echo("[report] Fetching execution report...")

if __name__ == "__main__":
    cli()