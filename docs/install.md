# Installing OpenShard

## Recommended: pipx

pipx installs OpenShard in an isolated environment, so its dependencies don't conflict with other Python tools.

```sh
pipx install git+https://github.com/MichaelObasa/openshard.git
```

Run the TUI:

```sh
openshard tui
```

Upgrade later:

```sh
pipx upgrade openshard
```

## Alternative: uv tool

If you use [uv](https://docs.astral.sh/uv/):

```sh
uv tool install git+https://github.com/MichaelObasa/openshard.git
```

## Local development

```sh
git clone https://github.com/MichaelObasa/openshard.git
cd openshard
pip install -e .
```

## Notes

- **pipx** is recommended for CLI users — isolated environment, clean upgrades, no conflicts with system Python.
- **uv tool** is also supported where available.
- **pip install** works but is less ideal for end users (installs into the active environment).
- PyPI, Homebrew, and curl installers are future release steps.
