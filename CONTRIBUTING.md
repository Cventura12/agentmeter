# Contributing

Run this sequence locally before opening a PR:

```bash
pip install -e ".[dev]"
ruff check agentmeter/ tests/
mypy agentmeter/ --strict
pytest --cov=agentmeter --cov-fail-under=80
```

## Core Model Rule

No vertical-specific fields in core models. Use `tags` and `metadata` for vertical context.
Example of automatic PR rejection: `is_invoice_agent = true`.

## PR Checklist

- Tests pass.
- `mypy` passes.
- `SPAN_KINDS.md` is updated if `SpanKind` changed.
