import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _dry_run(*configfiles: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "snakemake",
            "--snakefile",
            "workflow/Snakefile",
            "--configfile",
            *configfiles,
            "-n",
            "all",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_snakemake_dry_run_main_config():
    result = _dry_run("config/config.yaml")
    assert result.returncode == 0, result.stderr


def test_snakemake_dry_run_smoke_config():
    result = _dry_run("config/config.yaml", "config/smoke.yaml")
    assert result.returncode == 0, result.stderr
