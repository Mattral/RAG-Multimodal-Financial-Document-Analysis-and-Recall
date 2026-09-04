"""Component compatibility tests — upgraded to v2.0 architecture.

These tests ensure the new pluggable component ABCs, data models, and
utility modules work correctly as a system, replacing the legacy v1
component-specific tests.
"""

import re

import pytest
from pydantic import ValidationError

from src.rag_system.components.base import (
    BaseEmbedder,
    BaseGenerator,
    BaseParser,
    BaseReranker,
    BaseRetriever,
    BaseVectorStore,
    DocumentElement,
    GeneratedAnswer,
    RetrievedChunk,
)
from src.rag_system.components.layout_parser import (
    LayoutAwareParser,
    _wrap_figure,
    _wrap_table,
)
from src.rag_system.components.pot_executor import (
    FINANCIAL_TEMPLATES,
    ASTSandboxValidator,
    PoTExecutor,
)

# ── Data Model Tests ────────────────────────────────────────────────────────


class TestDocumentElement:
    def test_creation_minimal(self):
        e = DocumentElement(type="text", text="Revenue was $42B.", source_document="doc.pdf")
        assert e.type == "text"
        assert e.source_document == "doc.pdf"

    def test_creation_full(self):
        e = DocumentElement(
            type="table",
            text="| A | B |",
            source_document="10k.pdf",
            page_number=12,
            content_hash="abc123",
            tenant_id="acme",
            metadata={"parser": "docling"},
        )
        assert e.page_number == 12
        assert e.metadata["parser"] == "docling"

    def test_immutable(self):
        e = DocumentElement(type="text", text="x", source_document="d.pdf")
        with pytest.raises(ValidationError):
            e.text = "mutated"  # type: ignore

    def test_roundtrip(self):
        e = DocumentElement(
            type="graph", text="chart", source_document="d.pdf", page_number=3
        )
        restored = DocumentElement(**e.model_dump())
        assert restored.page_number == 3


class TestRetrievedChunk:
    def test_creation(self):
        c = RetrievedChunk(text="Revenue $42B.", score=0.91, source_document="doc.pdf")
        assert c.score == 0.91

    def test_roundtrip(self):
        c = RetrievedChunk(
            text="x", score=0.5, source_document="d.pdf", page_number=7, chunk_id="c1"
        )
        d = c.model_dump()
        assert d["chunk_id"] == "c1"


class TestGeneratedAnswer:
    def test_creation(self):
        c = RetrievedChunk(text="Revenue $42B.", score=0.9, source_document="doc.pdf")
        a = GeneratedAnswer(
            answer="Revenue was $42B.",
            citations=[c],
            model_used="gpt-4o-mini",
            prompt_tokens=200,
            completion_tokens=50,
            estimated_cost_usd=0.0001,
            latency_ms=1200.0,
        )
        assert len(a.citations) == 1
        assert a.estimated_cost_usd > 0


# ── ABC Enforcement Tests ─────────────────────────────────────────────────────


class TestAbstractBaseClasses:
    """Verify ABCs cannot be instantiated and require implementation."""

    def test_base_parser_not_instantiable(self):
        with pytest.raises(TypeError):
            BaseParser()  # type: ignore

    def test_base_embedder_not_instantiable(self):
        with pytest.raises(TypeError):
            BaseEmbedder()  # type: ignore

    def test_base_vector_store_not_instantiable(self):
        with pytest.raises(TypeError):
            BaseVectorStore()  # type: ignore

    def test_base_retriever_not_instantiable(self):
        with pytest.raises(TypeError):
            BaseRetriever()  # type: ignore

    def test_base_reranker_not_instantiable(self):
        with pytest.raises(TypeError):
            BaseReranker()  # type: ignore

    def test_base_generator_not_instantiable(self):
        with pytest.raises(TypeError):
            BaseGenerator()  # type: ignore

    def test_incomplete_implementation_raises(self):
        class IncompleteParser(BaseParser):
            @property
            def name(self):
                return "incomplete"

            async def parse(self, file_path, tenant_id=None):
                return []

            # Missing parse_batch — should raise TypeError at instantiation

        with pytest.raises(TypeError):
            IncompleteParser()  # type: ignore

    def test_complete_implementation_works(self):
        class MinimalParser(BaseParser):
            @property
            def name(self):
                return "minimal"

            async def parse(self, file_path, tenant_id=None):
                return []

            async def parse_batch(self, file_paths, tenant_id=None):
                return []

        p = MinimalParser()
        assert p.name == "minimal"


# ── PoT Executor Component Tests ──────────────────────────────────────────────


class TestASTSandboxValidator:
    def test_blocks_import(self):
        v = ASTSandboxValidator()
        assert v.validate("import os\nresult=1") is not None

    def test_blocks_exec(self):
        v = ASTSandboxValidator()
        assert v.validate("exec('1+1')\nresult=1") is not None

    def test_allows_arithmetic(self):
        v = ASTSandboxValidator()
        assert v.validate("x=10\ny=20\nresult=x+y") is None

    def test_allows_builtins(self):
        v = ASTSandboxValidator()
        assert v.validate("result=round(3.14159, 2)") is None

    def test_blocks_open(self):
        v = ASTSandboxValidator()
        assert v.validate("f=open('/etc/passwd')\nresult=f.read()") is not None


@pytest.mark.asyncio
async def test_pot_executor_basic():
    ex = PoTExecutor()
    r = await ex.execute_code("result = 2 ** 10")
    assert r.success
    assert r.result == 1024.0


@pytest.mark.asyncio
async def test_pot_executor_all_templates():
    ex = PoTExecutor()
    for name in FINANCIAL_TEMPLATES:
        template_code = FINANCIAL_TEMPLATES[name]
        # Replace placeholders with dummy values
        dummy_code = re.sub(r"\{[^}]+\}", "10.0", template_code)

        # Validation passes...
        err = ASTSandboxValidator().validate(dummy_code.strip())
        assert err is None, f"Template {name} failed validation: {err}"

        # ...and the template actually executes successfully end-to-end
        result = await ex.execute_code(dummy_code.strip())
        assert result.success, f"Template {name} failed execution: {result.error}"
        assert result.result is not None, f"Template {name} produced no result"


# ── Layout Parser Component Tests ────────────────────────────────────────────


class TestLayoutParserComponents:
    def test_wrap_table_html_structure(self):
        html = _wrap_table("| A | B |")
        assert "<table" in html
        assert "</table>" in html

    def test_wrap_figure_html_structure(self):
        html = _wrap_figure("Chart description")
        assert "<figure" in html
        assert "</figure>" in html

    def test_parser_empty_input(self):
        parser = LayoutAwareParser()
        assert parser.parse([]) == []

    def test_parser_handles_all_element_types(self):
        from src.rag_system.components.base import DocumentElement

        elements = [
            DocumentElement(
                type="text",
                text="Narrative text.",
                source_document="d.pdf",
                page_number=1,
            ),
            DocumentElement(
                type="table", text="| X | Y |", source_document="d.pdf", page_number=2
            ),
            DocumentElement(
                type="graph",
                text="Chart: bar chart.",
                source_document="d.pdf",
                page_number=3,
            ),
            DocumentElement(
                type="image",
                text="Photo of facility.",
                source_document="d.pdf",
                page_number=4,
            ),
        ]
        parser = LayoutAwareParser()
        chunks = parser.parse(elements)
        types_seen = {t for c in chunks for t in c.element_types}
        assert "text" in types_seen
        assert "table" in types_seen
        assert len(chunks) >= 3  # At minimum text, table, graph chunks

    def test_to_document_elements_has_html_in_text(self):
        from src.rag_system.components.base import DocumentElement

        elements = [
            DocumentElement(
                type="table",
                text="| Revenue | $42B |",
                source_document="d.pdf",
                page_number=1,
            ),
        ]
        parser = LayoutAwareParser()
        chunks = parser.parse(elements)
        doc_elems = parser.to_document_elements(chunks, tenant_id="t1")
        assert any("<table" in e.text for e in doc_elems)
