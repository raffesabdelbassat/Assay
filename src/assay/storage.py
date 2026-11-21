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
            comparison_id TEXT,
            prompt_version TEXT,
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
            provider TEXT,
            model TEXT,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cost REAL NOT NULL DEFAULT 0,
            latency_ms REAL NOT NULL DEFAULT 0,
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
        if "comparison_id" not in columns:
            self.connection.execute("ALTER TABLE runs ADD COLUMN comparison_id TEXT")
        if "prompt_version" not in columns:
            self.connection.execute("ALTER TABLE runs ADD COLUMN prompt_version TEXT")
        score_columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(scores)")}
        for name, declaration in {
            "provider": "TEXT",
            "model": "TEXT",
            "input_tokens": "INTEGER NOT NULL DEFAULT 0",
            "output_tokens": "INTEGER NOT NULL DEFAULT 0",
            "cost": "REAL NOT NULL DEFAULT 0",
            "latency_ms": "REAL NOT NULL DEFAULT 0",
        }.items():
            if name not in score_columns:
                self.connection.execute(f"ALTER TABLE scores ADD COLUMN {name} {declaration}")
        self.connection.commit()

    def start_run(self, config: AssayConfig, started_at: str) -> int:
        cursor = self.connection.execute(
            "INSERT INTO runs(started_at,status,config_json,input_price_per_token,output_price_per_token,comparison_id,prompt_version) VALUES(?,?,?,?,?,?,?)",
            (started_at, "running", config.model_dump_json(), config.prices.input_per_token, config.prices.output_per_token,
             config.comparison_id, config.prompt_version),
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
        self.connection.executemany("""INSERT INTO scores
            (response_id,scorer,passed,score,details_json,provider,model,input_tokens,output_tokens,cost,latency_ms)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            [(response_id, s.scorer, int(s.passed), s.score, json.dumps(s.details), s.provider, s.model,
              s.input_tokens, s.output_tokens, s.cost, s.latency_ms) for s in scores])
        self.connection.commit()

    def finish_run(self, run_id: int, finished_at: str, *, status: str = "completed", error: str | None = None) -> None:
        self.connection.execute("UPDATE runs SET finished_at=?,status=?,error=? WHERE id=?", (finished_at, status, error, run_id))
        self.connection.commit()

    def summary(self, run_id: int) -> dict[str, Any]:
        run = self.connection.execute("SELECT config_json FROM runs WHERE id=?", (run_id,)).fetchone()
        judge_enabled = False
        if run is not None:
            config = json.loads(run["config_json"])
            judge_enabled = config.get("judge") is not None
        row = self.connection.execute("""SELECT COUNT(DISTINCT e.id) AS examples,
            AVG(CASE WHEN sj.passed IS NOT NULL THEN sj.passed ELSE COALESCE(se.passed, 0) END) AS accuracy,
            AVG(COALESCE(r.latency_ms, 0) + COALESCE((SELECT SUM(s2.latency_ms) FROM scores s2
                WHERE s2.response_id=r.id), 0)) AS average_latency_ms,
            COALESCE(SUM(r.cost), 0) + COALESCE((SELECT SUM(s3.cost) FROM scores s3
                JOIN responses r3 ON r3.id=s3.response_id JOIN examples e3 ON e3.id=r3.example_id
                WHERE e3.run_id=?), 0) AS total_cost
            FROM examples e LEFT JOIN responses r ON r.example_id=e.id
            LEFT JOIN scores se ON se.response_id=r.id AND se.scorer='exact_match'
            LEFT JOIN scores sj ON sj.response_id=r.id AND sj.scorer='llm_judge' WHERE e.run_id=?""", (run_id, run_id)).fetchone()
        result = dict(row)
        config = json.loads(run["config_json"]) if run is not None else {}
        exact_match_enabled = config.get("scorers", {}).get("exact_match", True)
        if not exact_match_enabled and not judge_enabled:
            result["accuracy"] = None
        result["average_judge_score"] = None
        if judge_enabled:
            result["average_judge_score"] = self.connection.execute("""SELECT AVG(s.score) FROM scores s
                JOIN responses r ON r.id=s.response_id JOIN examples e ON e.id=r.example_id
                WHERE e.run_id=? AND s.scorer='llm_judge'""", (run_id,)).fetchone()[0]
        return result

    def close(self) -> None:
        self.connection.close()
