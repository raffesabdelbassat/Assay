"""SQLite persistence for runs, examples, responses, scores, and cache."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from assay.models import AssayConfig, DatasetExample, ProviderResponse, ScoreResult


class Storage:
    """Small synchronous SQLite repository; runner calls it between awaits."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL,
            config_json TEXT NOT NULL,
            input_price_per_token REAL NOT NULL,
            output_price_per_token REAL NOT NULL,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            example_key TEXT NOT NULL,
            input TEXT NOT NULL,
            expected TEXT,
            UNIQUE(run_id, example_key)
        );
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            example_id INTEGER NOT NULL UNIQUE REFERENCES examples(id) ON DELETE CASCADE,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt TEXT NOT NULL,
            system_prompt TEXT NOT NULL,
            output TEXT,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cost REAL NOT NULL DEFAULT 0,
            latency_ms REAL NOT NULL DEFAULT 0,
            attempts INTEGER NOT NULL DEFAULT 0,
            cache_hit INTEGER NOT NULL DEFAULT 0,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            response_id INTEGER NOT NULL REFERENCES responses(id) ON DELETE CASCADE,
            scorer TEXT NOT NULL,
            passed INTEGER NOT NULL,
            score REAL NOT NULL,
            details_json TEXT NOT NULL,
            UNIQUE(response_id, scorer)
        );
        CREATE TABLE IF NOT EXISTS response_cache (
            cache_key TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            system_prompt TEXT NOT NULL,
            prompt TEXT NOT NULL,
            temperature REAL NOT NULL,
            max_tokens INTEGER NOT NULL,
            output TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_examples_run ON examples(run_id);
        CREATE INDEX IF NOT EXISTS idx_scores_response ON scores(response_id);
        """)
        columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(runs)")}
        if "input_price_per_token" not in columns:
            self.connection.execute("ALTER TABLE runs ADD COLUMN input_price_per_token REAL NOT NULL DEFAULT 0")
            if "input_price_per_million" in columns:
                self.connection.execute("UPDATE runs SET input_price_per_token=input_price_per_million / 1000000.0")
        if "output_price_per_token" not in columns:
            self.connection.execute("ALTER TABLE runs ADD COLUMN output_price_per_token REAL NOT NULL DEFAULT 0")
            if "output_price_per_million" in columns:
                self.connection.execute("UPDATE runs SET output_price_per_token=output_price_per_million / 1000000.0")
        self.connection.commit()

    def start_run(self, config: AssayConfig, started_at: str) -> int:
        cursor = self.connection.execute(
            "INSERT INTO runs(started_at,status,config_json,input_price_per_token,output_price_per_token) VALUES(?,?,?,?,?)",
            (started_at, "running", config.model_dump_json(), config.prices.input_per_token, config.prices.output_per_token),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def add_example(self, run_id: int, example: DatasetExample) -> int:
        cursor = self.connection.execute(
            "INSERT INTO examples(run_id,example_key,input,expected) VALUES(?,?,?,?)",
            (run_id, example.id, example.input, example.expected),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def get_cached(self, key: str) -> ProviderResponse | None:
        row = self.connection.execute("SELECT output,input_tokens,output_tokens FROM response_cache WHERE cache_key=?", (key,)).fetchone()
        return None if row is None else ProviderResponse(**dict(row))

    def put_cached(self, key: str, *, provider: str, model: str, system_prompt: str, prompt: str,
                   temperature: float, max_tokens: int, response: ProviderResponse, created_at: str) -> None:
        self.connection.execute("""INSERT OR REPLACE INTO response_cache
            (cache_key,provider,model,system_prompt,prompt,temperature,max_tokens,output,input_tokens,output_tokens,created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)""", (key, provider, model, system_prompt, prompt, temperature, max_tokens,
            response.output, response.input_tokens, response.output_tokens, created_at))
        self.connection.commit()

    def add_response(self, *, example_db_id: int, provider: str, model: str, prompt: str, system_prompt: str,
                     response: ProviderResponse | None, cost: float, latency_ms: float, attempts: int,
                     cache_hit: bool, error: str | None) -> int:
        cursor = self.connection.execute("""INSERT INTO responses
            (example_id,provider,model,prompt,system_prompt,output,input_tokens,output_tokens,cost,latency_ms,attempts,cache_hit,error)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""", (example_db_id, provider, model, prompt, system_prompt,
            response.output if response else None, response.input_tokens if response else 0,
            response.output_tokens if response else 0, cost, latency_ms, attempts, int(cache_hit), error))
        self.connection.commit()
        return int(cursor.lastrowid)

    def add_scores(self, response_id: int, scores: list[ScoreResult]) -> None:
        self.connection.executemany("INSERT INTO scores(response_id,scorer,passed,score,details_json) VALUES(?,?,?,?,?)",
            [(response_id, s.scorer, int(s.passed), s.score, json.dumps(s.details)) for s in scores])
        self.connection.commit()

    def finish_run(self, run_id: int, finished_at: str, *, status: str = "completed", error: str | None = None) -> None:
        self.connection.execute("UPDATE runs SET finished_at=?,status=?,error=? WHERE id=?", (finished_at, status, error, run_id))
        self.connection.commit()

    def summary(self, run_id: int) -> dict[str, Any]:
        run = self.connection.execute("SELECT config_json FROM runs WHERE id=?", (run_id,)).fetchone()
        exact_match_enabled = True
        if run is not None:
            config = json.loads(run["config_json"])
            exact_match_enabled = config.get("scorers", {}).get("exact_match", True)
        row = self.connection.execute("""SELECT COUNT(DISTINCT e.id) AS examples,
            AVG(COALESCE(s.passed, 0)) AS accuracy, AVG(r.latency_ms) AS average_latency_ms, SUM(r.cost) AS total_cost
            FROM examples e LEFT JOIN responses r ON r.example_id=e.id
            LEFT JOIN scores s ON s.response_id=r.id AND s.scorer='exact_match' WHERE e.run_id=?""", (run_id,)).fetchone()
        result = dict(row)
        if not exact_match_enabled:
            result["accuracy"] = None
        return result

    def close(self) -> None:
        self.connection.close()
