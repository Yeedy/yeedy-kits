from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = SKILL_ROOT / "scripts" / "memory_baseline.py"
SPEC = importlib.util.spec_from_file_location("memory_baseline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
memory_baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(memory_baseline)


class MemoryBaselineTests(unittest.TestCase):
    def clean_env(self, home: Path) -> dict[str, str]:
        env = os.environ.copy()
        for name in (
            "CODEX_HOME",
            "PI_CODING_AGENT_DIR",
            "PI_CONFIG_DIR",
            "OMP_PROFILE",
            "PI_PROFILE",
        ):
            env.pop(name, None)
        env["HOME"] = str(home)
        return env

    def make_roots(self, directory: Path) -> tuple[Path, Path]:
        codex_dir = directory / "codex"
        omp_dir = directory / "omp" / "agent"
        (codex_dir / "memories").mkdir(parents=True)
        (omp_dir / "memories").mkdir(parents=True)
        for agent_dir in (codex_dir, omp_dir):
            (agent_dir / "skills").mkdir()
            (agent_dir / "skills" / memory_baseline.SKILL_NAME).symlink_to(
                SKILL_ROOT, target_is_directory=True
            )
        return codex_dir, omp_dir

    def run_script(
        self, script: Path, env: dict[str, str], *arguments: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(script), *arguments],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_installed_copy_defaults_to_lexical_host(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            codex_dir, _ = self.make_roots(directory)
            script = (
                codex_dir
                / "skills"
                / memory_baseline.SKILL_NAME
                / "scripts"
                / "memory_baseline.py"
            )
            env = self.clean_env(directory)
            env["CODEX_HOME"] = str(codex_dir)

            result = self.run_script(script, env)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["runtime"], "codex")

    def test_installed_copy_defaults_to_omp_lexical_host(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            _, omp_dir = self.make_roots(directory)
            script = (
                omp_dir
                / "skills"
                / memory_baseline.SKILL_NAME
                / "scripts"
                / "memory_baseline.py"
            )
            env = self.clean_env(directory)
            env["PI_CODING_AGENT_DIR"] = str(omp_dir)

            result = self.run_script(script, env)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["runtime"], "omp")

    def test_repository_copy_requires_explicit_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            env = self.clean_env(Path(raw_directory))

            result = self.run_script(SCRIPT, env)

            self.assertEqual(result.returncode, 2)
            self.assertIn("--runtime is required", result.stderr)

    def test_shared_source_is_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            codex_dir, omp_dir = self.make_roots(directory)
            env = self.clean_env(directory)
            env["CODEX_HOME"] = str(codex_dir)
            env["PI_CODING_AGENT_DIR"] = str(omp_dir)

            result = self.run_script(SCRIPT, env)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Ambiguous installed runtime", result.stderr)
            self.assertIn("codex, omp", result.stderr)

    def test_invalid_omp_profile_only_ignored_from_codex_lexical_entry(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            codex_dir, omp_dir = self.make_roots(directory)
            env = self.clean_env(directory)
            env["CODEX_HOME"] = str(codex_dir)
            env["PI_CONFIG_DIR"] = str(omp_dir.parent)
            env["OMP_PROFILE"] = "bad/profile"
            codex_script = (
                codex_dir
                / "skills"
                / memory_baseline.SKILL_NAME
                / "scripts"
                / "memory_baseline.py"
            )
            omp_script = (
                omp_dir
                / "skills"
                / memory_baseline.SKILL_NAME
                / "scripts"
                / "memory_baseline.py"
            )

            codex_result = self.run_script(codex_script, env)
            omp_result = self.run_script(omp_script, env)

            self.assertEqual(codex_result.returncode, 0, codex_result.stderr)
            self.assertEqual(json.loads(codex_result.stdout)["runtime"], "codex")
            self.assertNotEqual(omp_result.returncode, 0)
            self.assertIn("Invalid OMP profile name", omp_result.stderr)

    def test_both_output_is_array_and_dual_codex_databases_fail_loud(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            codex_dir, omp_dir = self.make_roots(directory)
            selected_db = codex_dir / "memories_1.sqlite"
            with sqlite3.connect(selected_db):
                pass
            env = self.clean_env(directory)

            both_result = self.run_script(
                SCRIPT,
                env,
                "--runtime",
                "both",
                "--omp-dir",
                str(omp_dir),
                "--codex-dir",
                str(codex_dir),
                "--codex-db",
                str(selected_db),
            )
            self.assertEqual(both_result.returncode, 0, both_result.stderr)
            self.assertEqual(
                [item["runtime"] for item in json.loads(both_result.stdout)],
                ["omp", "codex"],
            )

            stale_db = codex_dir / "sqlite" / "memories_1.sqlite"
            stale_db.parent.mkdir()
            with sqlite3.connect(stale_db):
                pass
            dual_result = self.run_script(
                SCRIPT,
                env,
                "--runtime",
                "codex",
                "--codex-dir",
                str(codex_dir),
            )
            self.assertNotEqual(dual_result.returncode, 0)
            self.assertIn("Multiple Codex memory databases", dual_result.stderr)
            self.assertIn(str(selected_db), dual_result.stderr)
            self.assertIn(str(stale_db), dual_result.stderr)

    def test_blank_path_environment_values_use_defaults(self) -> None:
        home = Path("/tmp/test-memory-baseline-home")
        updates = {
            "CODEX_HOME": " \t ",
            "PI_CODING_AGENT_DIR": "  ",
            "PI_CONFIG_DIR": "\t ",
            "OMP_PROFILE": " ",
            "PI_PROFILE": " ",
        }
        with mock.patch.dict(os.environ, updates, clear=False):
            self.assertEqual(memory_baseline.resolve_codex_dir(home), home / ".codex")
            self.assertEqual(
                memory_baseline.resolve_omp_agent_dir(home), home / ".omp" / "agent"
            )

        with mock.patch.dict(
            os.environ,
            {
                "CODEX_HOME": " /real/codex ",
                "PI_CODING_AGENT_DIR": " /real/omp/agent ",
            },
            clear=False,
        ):
            self.assertEqual(
                memory_baseline.resolve_codex_dir(home), Path("/real/codex")
            )
            self.assertEqual(
                memory_baseline.resolve_omp_agent_dir(home), Path("/real/omp/agent")
            )

    def test_query_error_does_not_trigger_copy_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            db_path = Path(raw_directory) / "jobs.sqlite"
            with sqlite3.connect(db_path) as db:
                db.execute("CREATE TABLE jobs (kind TEXT)")

            with mock.patch.object(
                memory_baseline.tempfile, "TemporaryDirectory"
            ) as tempdir:
                with self.assertRaisesRegex(sqlite3.OperationalError, "no such column"):
                    memory_baseline.read_jobs(db_path)
            tempdir.assert_not_called()

    def test_encoded_path_uses_read_only_fallback_when_uri_connect_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            db_path = Path(raw_directory) / "path #? with spaces" / "jobs.sqlite"
            db_path.parent.mkdir()
            with sqlite3.connect(db_path) as db:
                db.execute(
                    """
                    CREATE TABLE jobs (
                        kind TEXT, job_key TEXT, status TEXT, retry_remaining INTEGER,
                        last_error TEXT, input_watermark INTEGER,
                        last_success_watermark INTEGER
                    )
                    """
                )
                db.execute(
                    "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("memory", "global", "done", 0, None, 2, 2),
                )

            real_connect = memory_baseline.sqlite3.connect

            def fail_uri_connect(target: str, **kwargs: object) -> sqlite3.Connection:
                if kwargs.get("uri"):
                    raise sqlite3.OperationalError("forced URI failure")
                return real_connect(target, **kwargs)

            with mock.patch.object(
                memory_baseline.sqlite3, "connect", side_effect=fail_uri_connect
            ) as connect:
                result = memory_baseline.read_jobs(db_path)

            self.assertEqual(result["job_count"], 1)
            self.assertEqual(result["status_counts"], {"done": 1})
            self.assertEqual(connect.call_count, 2)
            self.assertTrue(connect.call_args_list[0].kwargs["uri"])


if __name__ == "__main__":
    unittest.main()
