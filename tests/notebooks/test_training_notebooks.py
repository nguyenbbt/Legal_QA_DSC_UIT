from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS = (
    "00_model_training_preflight.ipynb",
    "10_mil005_qwen3_embedding_finetune.ipynb",
    "20_mil005_qwen3_reranker_finetune.ipynb",
    "30_mil006_qwen3_generator_qlora.ipynb",
    "90_training_results_review.ipynb",
)
ROOT = Path(__file__).parents[2] / "notebooks" / "training"


def _load(name: str) -> dict[str, object]:
    return json.loads((ROOT / name).read_bytes())


def test_all_training_notebooks_are_valid_safe_python_notebooks() -> None:
    for name in NOTEBOOKS:
        notebook = _load(name)
        assert notebook["nbformat"] == 4
        assert notebook["metadata"]["kernelspec"]["name"] == "python3"
        cells = notebook["cells"]
        assert cells
        for cell in cells:
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None
                assert cell.get("outputs") == []


def test_notebook_defaults_cannot_start_network_gpu_modal_or_training() -> None:
    for name in NOTEBOOKS:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "ALLOW_NETWORK = False" in text
        assert "ALLOW_GPU = False" in text
        assert "ALLOW_MODAL_REAL_DATA = False" in text
        assert "ALLOW_FINETUNE = False" in text
        assert "CONFIRM_REMOTE_EXECUTION = False" in text
        assert "modal run" not in text.lower()
        assert "m-gpux dev" not in text.lower()


def test_generator_notebook_contains_all_required_sections() -> None:
    text = (ROOT / "30_mil006_qwen3_generator_qlora.ipynb").read_text(encoding="utf-8")
    sections = tuple(f"G{index:02d}" for index in range(1, 17))
    for section in sections:
        assert section in text
