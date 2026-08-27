# DSC 2026 Vietnamese LegalQA

Deterministic, provider-neutral Vietnamese legal question answering for DSC 2026 Task
2. The authoritative contract is [`tasks/spec.md`](tasks/spec.md), and the active
implementation sequence is [`LEGAL_RAG_EXEC_PLAN.md`](LEGAL_RAG_EXEC_PLAN.md).

## Current status

MIL-001 through MIL-004 and Pre-MIL-005 are owner-approved and complete. D-049 opens
the conditional end-to-end sequence, and D-050 approves the exact audited Qwen3
embedding/generator/reranker profile. G3 QLoRA was evaluated once and rejected; the
base G1-512 generator passes its numeric, grounding, and clean-reproducibility gates.

The single G1-512 generator ablation raises METEOR from 0.201985 to 0.206535 while
ROUGE-L changes from 0.327514 to 0.326640. The original R2 retrieval output remains
the strongest measured answer path but lacks enough runtime provenance for unseen
public inference. Its exact reproducible rebaseline fails downstream answer/resource
gates, so exact/BM25 R0 is the deployable fallback. One of 60 R0 grounding rows is
prefilled and 59 await approval before the checkpointed 1,000-question public run.
Real organizer/derived data must stay off Modal while OQ-003 is unresolved, and
OQ-001 still blocks a submission-ready claim.

## Local environment

The project virtual environment is `.venv`. It uses the tested CPython 3.12.7
Windows x86-64 runtime managed inside the project:

```powershell
.\.venv\Scripts\Activate.ps1
```

OQ-007 candidate testing is recorded under `artifacts/reports/mil-001`. The reviewed
lock can be restored and checked without changing dependency versions:

```powershell
.\.venv\Scripts\uv.exe sync --all-extras --frozen
.\.venv\Scripts\legal-rag.exe doctor --config configs\fixture.yaml --execution-mode local-offline
.\.venv\Scripts\legal-rag.exe scorer parity --official Scoring-Program-Task-LegalQA\scoring.py --fixtures data\fixtures\scoring
```

Do not install project packages globally. Dependency and resource acquisition belongs
only to the declared `prepare-online` bootstrap; accepted runtime checks are offline.

## Planning and progress

- Current milestone plan: [`tasks/plan.md`](tasks/plan.md)
- Executable checklist: [`tasks/todo.md`](tasks/todo.md)
- Doctor CLI contract: [`docs/interfaces/doctor.v1.md`](docs/interfaces/doctor.v1.md)
- Execution-mode schemas: [`docs/interfaces/execution-mode.v1.md`](docs/interfaces/execution-mode.v1.md)
- Scorer-parity CLI: [`docs/interfaces/scorer-parity.v1.md`](docs/interfaces/scorer-parity.v1.md)
- Blocked organizer/legal-version/operations facts remain in spec section 17 and must
  not be guessed.

Organizer data under `data/` is local-only and immutable. Only synthetic files under
`data/fixtures/` may be version-controlled.
