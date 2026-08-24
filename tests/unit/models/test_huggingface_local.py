from __future__ import annotations

from pathlib import Path

import pytest
import torch

import legal_rag.models.huggingface_local as huggingface_local
from legal_rag.models.huggingface_local import (
    Qwen3RerankerBackend,
    _last_token_pool,
    _local_adapter_directory,
    _local_directory,
)
from legal_rag.models.manifest import ModelManifestError


def test_local_directory_rejects_container_without_checkpoint_files(tmp_path: Path) -> None:
    (tmp_path / "revision").mkdir()

    with pytest.raises(ModelManifestError) as caught:
        _local_directory(tmp_path)

    assert caught.value.code == "MODEL_ARTIFACT_UNSUPPORTED"


def test_local_adapter_directory_requires_config_and_weights(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ModelManifestError) as caught:
        _local_adapter_directory(tmp_path)

    assert caught.value.code == "MODEL_ADAPTER_ARTIFACT_UNSUPPORTED"


def test_local_adapter_directory_accepts_complete_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"fixture")

    assert _local_adapter_directory(tmp_path) == str(tmp_path.resolve())


def test_reranker_only_materializes_last_token_logits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
        (tmp_path / name).write_bytes(b"fixture")

    class _Inputs(dict):
        def to(self, _device: str):
            return self

    class _Tokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def encode(self, value: str, *, add_special_tokens: bool):
            assert add_special_tokens is False
            return [1 if value == "yes" else 0]

        def __call__(self, batch, **_kwargs):
            return _Inputs(input_ids=torch.ones((len(batch), 2), dtype=torch.long))

    class _Model:
        last_kwargs = None

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def to(self, _device: str):
            return self

        def eval(self) -> None:
            return None

        def __call__(self, **kwargs):
            self.last_kwargs = kwargs
            return type("Output", (), {"logits": torch.tensor([[[0.0, 1.0]]])})()

    monkeypatch.setattr(
        huggingface_local,
        "_dependencies",
        lambda: (torch, object(), _Model, _Tokenizer),
    )
    backend = Qwen3RerankerBackend(
        tmp_path,
        model_id="fixture/reranker",
        model_revision="revision-1",
        instruction="Judge support.",
        device="cpu",
        batch_size=1,
        maximum_length=32,
    )

    assert backend.score("query", ("document",))[0] > 0.5
    assert backend._model.last_kwargs["logits_to_keep"] == 1


def test_last_token_pool_handles_right_padding() -> None:
    hidden = torch.tensor([[[1.0], [2.0], [99.0]], [[3.0], [4.0], [5.0]]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    pooled = _last_token_pool(hidden, mask)

    assert pooled.tolist() == [[2.0], [5.0]]


def test_last_token_pool_handles_left_padding() -> None:
    hidden = torch.tensor([[[99.0], [1.0], [2.0]], [[98.0], [3.0], [4.0]]])
    mask = torch.tensor([[0, 1, 1], [0, 1, 1]])

    pooled = _last_token_pool(hidden, mask)

    assert pooled.tolist() == [[2.0], [4.0]]
