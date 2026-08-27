from __future__ import annotations

from pathlib import Path

import pytest
import torch

import legal_rag.models.huggingface_local as huggingface_local
from legal_rag.models.huggingface_local import (
    Qwen3AdapterRerankerBackend,
    Qwen3RerankerBackend,
    Qwen25GeneratorBackend,
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


def test_adapter_reranker_wraps_one_local_base_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_path = tmp_path / "base"
    adapter_path = tmp_path / "adapter"
    base_path.mkdir()
    adapter_path.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
        (base_path / name).write_bytes(b"fixture")
    (adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_path / "adapter_model.safetensors").write_bytes(b"adapter")

    class _Tokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            assert _kwargs["local_files_only"] is True
            return cls()

        def encode(self, value: str, *, add_special_tokens: bool):
            assert add_special_tokens is False
            return [1 if value == "yes" else 0]

    class _Base:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            assert _kwargs["local_files_only"] is True
            return cls()

        def to(self, _device: str):
            return self

        def eval(self) -> None:
            return None

    class _Peft:
        called = None

        @classmethod
        def from_pretrained(cls, base, adapter, **kwargs):
            cls.called = (base, adapter, kwargs)
            return base

    monkeypatch.setattr(
        huggingface_local,
        "_dependencies",
        lambda: (torch, object(), _Base, _Tokenizer),
    )
    monkeypatch.setattr(huggingface_local, "_peft_dependency", lambda: _Peft)

    backend = Qwen3AdapterRerankerBackend(
        base_path,
        adapter_path,
        model_id="fixture/reranker",
        model_revision="revision-1",
        adapter_id="fixture/adapter",
        instruction="Judge support.",
        device="cpu",
    )

    assert backend.adapter_id == "fixture/adapter"
    assert _Peft.called is not None
    assert _Peft.called[2] == {"is_trainable": False, "local_files_only": True}


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


def test_qwen25_generator_uses_local_fp16_cpu_offload_without_thinking_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
        (checkpoint / name).write_bytes(b"fixture")
    offload = tmp_path / "offload"

    class _Inputs(dict):
        def to(self, device):
            assert str(device) == "cpu"
            return self

    class _Tokenizer:
        eos_token_id = 2
        pad_token_id = None
        template_kwargs = None

        @classmethod
        def from_pretrained(cls, *_args, **kwargs):
            assert kwargs == {"local_files_only": True, "trust_remote_code": False}
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["role"] == "system"
            type(self).template_kwargs = kwargs
            return "rendered"

        def __call__(self, *_args, **kwargs):
            assert kwargs["max_length"] == 2048
            return _Inputs(input_ids=torch.tensor([[10, 11]]))

        def decode(self, token_ids, *, skip_special_tokens: bool):
            assert token_ids.tolist() == [12]
            assert skip_special_tokens is True
            return "câu trả lời"

    class _Embedding:
        weight = torch.zeros(1)

    class _Model:
        load_kwargs = None
        moved = False

        @classmethod
        def from_pretrained(cls, *_args, **kwargs):
            cls.load_kwargs = kwargs
            return cls()

        def to(self, _device):
            type(self).moved = True
            return self

        def eval(self) -> None:
            return None

        def get_input_embeddings(self):
            return _Embedding()

        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            assert kwargs["pad_token_id"] == 2
            return torch.tensor([[10, 11, 12]])

    monkeypatch.setattr(
        huggingface_local,
        "_dependencies",
        lambda: (torch, object(), _Model, _Tokenizer),
    )

    backend = Qwen25GeneratorBackend(
        checkpoint,
        model_id="Qwen/Qwen2.5-3B-Instruct",
        model_revision="revision-1",
        device_map="auto",
        max_memory={0: "5GiB", "cpu": "6GiB"},
        offload_folder=offload,
        maximum_input_tokens=2048,
        maximum_new_tokens=512,
    )

    assert (
        backend.generate(system_prompt="system", question="question", evidence=("evidence",))
        == "câu trả lời"
    )
    assert _Model.moved is False
    assert _Model.load_kwargs == {
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": torch.float16,
        "device_map": "auto",
        "max_memory": {0: "5GiB", "cpu": "6GiB"},
        "offload_folder": str(offload.resolve()),
        "offload_state_dict": True,
    }
    assert _Tokenizer.template_kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
    }
