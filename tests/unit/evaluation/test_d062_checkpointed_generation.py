from __future__ import annotations

from pathlib import Path

from scripts.run_d062_full_dev_generation import (
    CheckpointedGeneratorBackend,
    GenerationExpectation,
    checkpoint_checksum,
)


class _Backend:
    model_id = "model"
    model_revision = "revision"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def generate(self, *, system_prompt: str, question: str, evidence: tuple[str, ...]) -> str:
        if self.fail:
            raise AssertionError("checkpoint replay invoked the model")
        self.calls += 1
        return f"answer-{question}"


def _wrapper(backend: _Backend, checkpoint_directory: Path) -> CheckpointedGeneratorBackend:
    return CheckpointedGeneratorBackend(
        backend=backend,
        checkpoint_directory=checkpoint_directory,
        run_id="D062-r0-G1A512-development-716-v1",
        run_fingerprint="sha256:" + "1" * 64,
        expectations=(
            GenerationExpectation(
                position=0,
                question_id="q1",
                question="Question",
                evidence_ids=("e1",),
                evidence=("Evidence",),
            ),
        ),
    )


def test_checkpointed_generator_replays_without_model_invocation(tmp_path: Path) -> None:
    backend = _Backend()
    first = _wrapper(backend, tmp_path)

    assert (
        first.generate(system_prompt="Prompt", question="Question", evidence=("Evidence",))
        == "answer-Question"
    )
    assert backend.calls == 1
    assert first.generated_count == 1
    assert first.resumed_count == 0

    replay = _wrapper(_Backend(fail=True), tmp_path)
    assert (
        replay.generate(system_prompt="Prompt", question="Question", evidence=("Evidence",))
        == "answer-Question"
    )
    assert replay.generated_count == 0
    assert replay.resumed_count == 1


def test_checkpointed_generator_rejects_changed_evidence(tmp_path: Path) -> None:
    wrapper = _wrapper(_Backend(), tmp_path)

    try:
        wrapper.generate(system_prompt="Prompt", question="Question", evidence=("Changed",))
    except ValueError as error:
        assert str(error) == "D-062 generation call differs from frozen inputs"
    else:
        raise AssertionError("changed evidence was accepted")


def test_checkpoint_checksum_binds_index_and_every_shard(tmp_path: Path) -> None:
    (tmp_path / "model-00001.safetensors").write_bytes(b"first")
    (tmp_path / "model-00002.safetensors").write_bytes(b"second")
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"model-00001.safetensors","b":"model-00002.safetensors"}}',
        encoding="utf-8",
    )

    before = checkpoint_checksum(tmp_path)
    (tmp_path / "model-00002.safetensors").write_bytes(b"changed")

    assert checkpoint_checksum(tmp_path) != before
