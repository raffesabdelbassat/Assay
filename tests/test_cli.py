from typer.testing import CliRunner

from assay.cli import app


def test_cli_help_does_not_require_provider_sdk() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
