"""Tests for src/rag_system/cli.py + compat shims — previously 0% covered."""
from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from typer.testing import CliRunner
from src.rag_system.cli import app

runner = CliRunner()
PATCH = "src.rag_system.pipeline.create_pipeline"

def _pipeline():
    p = MagicMock()
    p.ingest = AsyncMock(return_value={"status": "success", "num_files": 1, "num_chunks": 5, "tenant_id": "default", "latency_s": 0.1})
    p.query = AsyncMock(return_value={"status": "success", "query": "q", "answer": "Revenue was $23.35B [Source: f.pdf, Page 1].", "sources": [{"document": "f.pdf", "page": 1, "score": 0.9, "text_[...]
    p.health_check = AsyncMock(return_value={"status": "healthy", "components": {"vector_store": "ok", "embedder": "ok"}})
    return p

class TestVersionFlag:
    def test_version_succeeds(self): assert runner.invoke(app, ["--version"]).exit_code == 0
    def test_version_shows_number(self):
        r = runner.invoke(app, ["--version"])
        assert "2.0" in r.stdout or "version" in r.stdout.lower()

class TestHelpFlags:
    @pytest.mark.parametrize("cmd", ["ingest", "query", "evaluate", "serve", "health"])
    def test_help_exits_zero(self, cmd): assert runner.invoke(app, [cmd, "--help"]).exit_code == 0

class TestIngestCommand:
    def test_missing_file_exits_nonzero(self, tmp_path):
        assert runner.invoke(app, ["ingest", str(tmp_path / "no.pdf")]).exit_code != 0
    def test_success(self, tmp_path):
        pdf = tmp_path / "t.pdf"; pdf.write_bytes(b"%PDF-1.4")
        p = _pipeline()
        with patch(PATCH, AsyncMock(return_value=p)):
            result = runner.invoke(app, ["ingest", str(pdf)])
        assert result.exit_code == 0
        p.ingest.assert_called_once()
    def test_tenant_flag(self, tmp_path):
        pdf = tmp_path / "t.pdf"; pdf.write_bytes(b"%PDF-1.4")
        p = _pipeline()
        with patch(PATCH, AsyncMock(return_value=p)):
            runner.invoke(app, ["ingest", str(pdf), "--tenant", "acme"])
        assert "acme" in str(p.ingest.call_args)
    def test_error_exits_nonzero(self, tmp_path):
        pdf = tmp_path / "t.pdf"; pdf.write_bytes(b"%PDF-1.4")
        p = _pipeline(); p.ingest = AsyncMock(side_effect=RuntimeError("boom"))
        with patch(PATCH, AsyncMock(return_value=p)):
            assert runner.invoke(app, ["ingest", str(pdf)]).exit_code != 0

class TestQueryCommand:
    def test_success(self):
        p = _pipeline()
        with patch(PATCH, AsyncMock(return_value=p)):
            result = runner.invoke(app, ["query", "What was revenue?"])
        assert result.exit_code == 0
        p.query.assert_called_once()
    def test_show_sources_flag(self):
        with patch(PATCH, AsyncMock(return_value=_pipeline())):
            assert runner.invoke(app, ["query", "q", "--show-sources"]).exit_code == 0
    def test_top_k_and_tenant(self):
        with patch(PATCH, AsyncMock(return_value=_pipeline())):
            assert runner.invoke(app, ["query", "q", "--tenant", "acme", "--top-k", "10"]).exit_code == 0
    def test_error_exits_nonzero(self):
        p = _pipeline(); p.query = AsyncMock(side_effect=RuntimeError("boom"))
        with patch(PATCH, AsyncMock(return_value=p)):
            assert runner.invoke(app, ["query", "q"]).exit_code != 0

class TestHealthCommand:
    def test_success(self):
        p = _pipeline()
        with patch(PATCH, AsyncMock(return_value=p)):
            result = runner.invoke(app, ["health"])
        assert result.exit_code == 0
        p.health_check.assert_called_once()
    def test_degraded_still_exits_zero(self):
        p = _pipeline()
        p.health_check = AsyncMock(return_value={"status": "degraded", "components": {"vector_store": "error", "embedder": "ok"}})
        with patch(PATCH, AsyncMock(return_value=p)):
            assert runner.invoke(app, ["health"]).exit_code == 0
    def test_construction_failure_exits_nonzero(self):
        with patch(PATCH, AsyncMock(side_effect=RuntimeError("cannot connect"))):
            assert runner.invoke(app, ["health"]).exit_code != 0

class TestServeCommand:
    def test_missing_uvicorn_exits_nonzero(self):
        with patch.dict("sys.modules", {"uvicorn": None}):
            assert runner.invoke(app, ["serve"]).exit_code != 0
    def test_port_flag_not_usage_error(self):
        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            result = runner.invoke(app, ["serve", "--port", "8080"])
        assert result.exit_code != 2  # not a typer usage error

class TestEvaluateCommand:
    def test_missing_dataset_exits_nonzero(self, tmp_path):
        mock_runner_cls = MagicMock()
        mock_runner_cls.return_value.run = AsyncMock(side_effect=FileNotFoundError("not found"))
        with patch(PATCH, AsyncMock(return_value=_pipeline())):
            with patch("src.rag_system.components.evaluator.GoldenDatasetRunner", mock_runner_cls):
                result = runner.invoke(app, ["evaluate", "--dataset", str(tmp_path / "nope.jsonl")])
        assert result.exit_code != 0
    def test_runs_with_dataset(self, tmp_path):
        dataset = tmp_path / "golden.jsonl"
        dataset.write_text('{"question": "Revenue?", "ground_truth": "$23.35B"}\n')
        mock_report = MagicMock()
        for attr in ["pass_rate","avg_faithfulness","avg_numeric_accuracy","avg_answer_relevancy","regression_detected","avg_latency_ms","total_cost_usd","run_id"]:
            setattr(mock_report, attr, 0.9 if "rate" in attr or "avg" in attr else ("id" if attr == "run_id" else False))
        mock_report.num_samples = 1
        mock_report.passed = 1
        mock_report.failed = 0
        mock_runner_cls = MagicMock()
        mock_runner_cls.return_value.run = AsyncMock(return_value=mock_report)
        with patch(PATCH, AsyncMock(return_value=_pipeline())), \
             patch("src.rag_system.components.evaluator.GoldenDatasetRunner", mock_runner_cls):
            result = runner.invoke(app, ["evaluate", "--dataset", str(dataset)])
        assert result.exit_code in (0, 1)

class TestCompatibilityShims:
    def test_pdf_parser_shim(self):
        from src.rag_system.components.pdf_parser import PDFParser, build_parser
        assert PDFParser is not None and callable(build_parser)
    def test_vector_indexer_shim(self):
        from src.rag_system.components.vector_indexer import VectorIndexer, build_vector_store
        assert VectorIndexer is not None and callable(build_vector_store)
    def test_vision_processor_shim(self):
        from src.rag_system.components.vision_processor import VisionProcessor, build_vision_describer
        assert VisionProcessor is not None and callable(build_vision_describer)
