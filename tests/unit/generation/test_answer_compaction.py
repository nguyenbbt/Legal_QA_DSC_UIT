from __future__ import annotations

import json

import pytest

from legal_rag.generation.answer_compaction import (
    AnswerCompactionError,
    build_deletion_only_grounding_proof,
    compact_answer,
    derive_answer_compaction_policy,
)


def _train(answers: tuple[str, ...]) -> bytes:
    return json.dumps(
        {
            str(index): {"question": f"q{index}", "answer": answer}
            for index, answer in enumerate(answers, start=1)
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()


def test_policy_uses_nearest_rank_train_median_and_is_deterministic() -> None:
    data = _train(("một", "một hai", "một hai ba", "một hai ba bốn", "năm từ ở đây nhé"))

    first, first_manifest = derive_answer_compaction_policy(data, split="train")
    second, second_manifest = derive_answer_compaction_policy(data, split="train")

    assert first.maximum_whitespace_tokens == 3
    assert first.source_row_count == 5
    assert first == second
    assert first_manifest == second_manifest


def test_policy_rejects_any_non_train_calibration() -> None:
    with pytest.raises(AnswerCompactionError) as caught:
        derive_answer_compaction_policy(_train(("answer",)), split="development")

    assert caught.value.code == "ANSWER_COMPACTION_SPLIT_INVALID"


def test_compaction_returns_only_a_complete_literal_prefix() -> None:
    policy, _ = derive_answer_compaction_policy(
        _train(("a", "a b c d e", "a b c d e", "a b c d e f", "a b c d e f g")),
        split="train",
    )
    answer = "Câu thứ nhất kết thúc. Câu thứ hai còn rất nhiều từ và sẽ bị cắt."

    compacted = compact_answer(answer, policy)

    assert compacted == "Câu thứ nhất kết thúc."
    assert " ".join(answer.split()).startswith(compacted)


def test_compaction_keeps_long_answer_when_no_complete_sentence_is_within_cap() -> None:
    policy, _ = derive_answer_compaction_policy(
        _train(("a", "a b", "a b c", "a b c d", "a b c d e")),
        split="train",
    )
    answer = "một hai ba bốn năm sáu"

    assert compact_answer(answer, policy) == answer


def test_deletion_only_proof_rejects_rewritten_or_added_text() -> None:
    baseline = json.dumps({"q1": {"answer": "Một. Hai."}}, ensure_ascii=False).encode()
    valid = json.dumps({"q1": {"answer": "Một."}}, ensure_ascii=False).encode()
    invalid = json.dumps({"q1": {"answer": "Một. Ba."}}, ensure_ascii=False).encode()

    proof = json.loads(build_deletion_only_grounding_proof(baseline, valid))
    assert proof["proof_state"] == "passed"
    assert proof["changed_answer_count"] == 1

    with pytest.raises(AnswerCompactionError) as caught:
        build_deletion_only_grounding_proof(baseline, invalid)
    assert caught.value.code == "ANSWER_COMPACTION_NOT_PREFIX_ONLY"
