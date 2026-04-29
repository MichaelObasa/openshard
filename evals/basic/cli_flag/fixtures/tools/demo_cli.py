import click


@click.group()
def cli():
    pass


@cli.command()
def report():
    """Print a summary report."""
    click.echo("Summary: 3 runs, 2 passed, 1 failed")


if __name__ == "__main__":
    cli()
