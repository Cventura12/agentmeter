"""Capability taxonomy definitions and metadata registry."""

from __future__ import annotations

from enum import Enum

TAXONOMY_VERSION = "0.1.0"


class Capability(str, Enum):
    """Canonical capability identifiers used across agentmeter events."""

    CLASSIFICATION_DOCUMENT = "classification.document"
    CLASSIFICATION_EMAIL = "classification.email"
    CLASSIFICATION_INTENT = "classification.intent"

    EXTRACTION_INVOICE = "extraction.invoice"
    EXTRACTION_CONTRACT = "extraction.contract"
    EXTRACTION_FORM = "extraction.form"
    EXTRACTION_RECEIPT = "extraction.receipt"

    GENERATION_EMAIL = "generation.email"
    GENERATION_REPORT = "generation.report"
    GENERATION_CODE = "generation.code"
    GENERATION_SUMMARY = "generation.summary"

    PLANNING_MULTI_STEP = "planning.multi_step"
    PLANNING_ROUTING = "planning.routing"
    PLANNING_DECOMPOSITION = "planning.decomposition"

    RETRIEVAL_SEMANTIC = "retrieval.semantic"
    RETRIEVAL_STRUCTURED = "retrieval.structured"

    TRANSFORMATION_FORMAT = "transformation.format"
    TRANSFORMATION_TRANSLATE = "transformation.translate"

    VERIFICATION_FACTCHECK = "verification.factcheck"
    VERIFICATION_SCHEMA = "verification.schema"

    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> Capability:
        """Return an exact-match capability or `UNKNOWN` when unmatched."""
        if not isinstance(value, str):
            return cls.UNKNOWN
        member = cls._value2member_map_.get(value)
        return member if isinstance(member, cls) else cls.UNKNOWN

    @property
    def category(self) -> str:
        """Return the dot-prefix category for this capability."""
        return self.value.split(".", 1)[0]


CAPABILITY_REGISTRY: dict[Capability, dict[str, str | int | float]] = {
    Capability.CLASSIFICATION_DOCUMENT: {
        "description": "Classify general-purpose document types and intents",
        "typical_latency_ms": 250,
        "typical_cost_usd": 0.0006,
    },
    Capability.CLASSIFICATION_EMAIL: {
        "description": "Classify inbound email into operational categories",
        "typical_latency_ms": 180,
        "typical_cost_usd": 0.0004,
    },
    Capability.CLASSIFICATION_INTENT: {
        "description": "Identify user intent from natural-language requests",
        "typical_latency_ms": 140,
        "typical_cost_usd": 0.0003,
    },
    Capability.EXTRACTION_INVOICE: {
        "description": "Extract structured data from invoice documents",
        "typical_latency_ms": 800,
        "typical_cost_usd": 0.003,
    },
    Capability.EXTRACTION_CONTRACT: {
        "description": "Extract clauses, entities, and terms from contracts",
        "typical_latency_ms": 1200,
        "typical_cost_usd": 0.0055,
    },
    Capability.EXTRACTION_FORM: {
        "description": "Extract normalized fields from standardized forms",
        "typical_latency_ms": 600,
        "typical_cost_usd": 0.0021,
    },
    Capability.EXTRACTION_RECEIPT: {
        "description": "Extract merchant, line-item, and totals from receipts",
        "typical_latency_ms": 550,
        "typical_cost_usd": 0.0018,
    },
    Capability.GENERATION_EMAIL: {
        "description": "Generate context-aware emails with structured tone control",
        "typical_latency_ms": 350,
        "typical_cost_usd": 0.0012,
    },
    Capability.GENERATION_REPORT: {
        "description": "Generate long-form analytical reports from source inputs",
        "typical_latency_ms": 1400,
        "typical_cost_usd": 0.0075,
    },
    Capability.GENERATION_CODE: {
        "description": "Generate code snippets and implementation drafts",
        "typical_latency_ms": 900,
        "typical_cost_usd": 0.0048,
    },
    Capability.GENERATION_SUMMARY: {
        "description": "Generate concise summaries from long-form content",
        "typical_latency_ms": 320,
        "typical_cost_usd": 0.0011,
    },
    Capability.PLANNING_MULTI_STEP: {
        "description": "Plan multi-step tasks into executable action sequences",
        "typical_latency_ms": 700,
        "typical_cost_usd": 0.0029,
    },
    Capability.PLANNING_ROUTING: {
        "description": "Route tasks to best-fit agents or workflows",
        "typical_latency_ms": 220,
        "typical_cost_usd": 0.0007,
    },
    Capability.PLANNING_DECOMPOSITION: {
        "description": "Decompose complex objectives into atomic subtasks",
        "typical_latency_ms": 500,
        "typical_cost_usd": 0.002,
    },
    Capability.RETRIEVAL_SEMANTIC: {
        "description": "Retrieve semantically similar documents or passages",
        "typical_latency_ms": 160,
        "typical_cost_usd": 0.0005,
    },
    Capability.RETRIEVAL_STRUCTURED: {
        "description": "Retrieve records from structured stores and APIs",
        "typical_latency_ms": 120,
        "typical_cost_usd": 0.0004,
    },
    Capability.TRANSFORMATION_FORMAT: {
        "description": "Transform content between output schemas and formats",
        "typical_latency_ms": 280,
        "typical_cost_usd": 0.001,
    },
    Capability.TRANSFORMATION_TRANSLATE: {
        "description": "Translate text across languages while preserving meaning",
        "typical_latency_ms": 340,
        "typical_cost_usd": 0.0014,
    },
    Capability.VERIFICATION_FACTCHECK: {
        "description": "Verify claims against trusted evidence sources",
        "typical_latency_ms": 950,
        "typical_cost_usd": 0.0042,
    },
    Capability.VERIFICATION_SCHEMA: {
        "description": "Validate content against schema and policy constraints",
        "typical_latency_ms": 130,
        "typical_cost_usd": 0.0005,
    },
    Capability.UNKNOWN: {
        "description": "Fallback capability for uncategorized operations",
        "typical_latency_ms": 0,
        "typical_cost_usd": 0.0,
    },
}
