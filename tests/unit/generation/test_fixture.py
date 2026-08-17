from __future__ import annotations

from legal_rag.domain.checksums import checksum_bytes
from legal_rag.domain.models import ComponentScores, Evidence, QuestionRecord
from legal_rag.generation.fixture import FIXED_REFUSAL, FixtureExtractiveGenerator


def question_record() -> QuestionRecord:
    return QuestionRecord.model_validate(
        {
            "schema_version": "internal.question.v1",
            "question_id": "q1",
            "original_id": "q1",
            "original_id_kind": "object_key_string",
            "source_position": 0,
            "source_artifact": "fixtures/questions.json",
            "source_checksum": checksum_bytes(b"questions"),
            "question": "Điều 1 quy định ai được cấp thẻ?",
            "answer": None,
            "answer_state": "unlabeled",
        }
    )


def evidence(text: str) -> Evidence:
    return Evidence.model_validate(
        {
            "schema_version": "internal.evidence.v1",
            "evidence_id": "chunk_aaaaaaaaaaaaaaaaaaaaaaaa",
            "context_id": "1",
            "source_url": "https://example.invalid/1",
            "hierarchy_path": ("Điều 1",),
            "canonical_start": 0,
            "canonical_end": len(text),
            "display_text": text,
            "retrieval_text": "nội dung",
            "rank": 1,
            "component_scores": ComponentScores(
                exact_reference_match=True,
                sparse_score=1.0,
                dense_score=None,
                reranker_score=None,
            ),
            "chunk_checksum": checksum_bytes(text.encode()),
            "context_checksum": checksum_bytes(b"context"),
            "integrity_status": "valid",
            "claim_support": "unknown",
            "version_validity": "unknown",
        }
    )


def test_generator_removes_only_explicit_heading_only_first_line() -> None:
    generator = FixtureExtractiveGenerator()

    heading_only = generator.generate(
        question_record(),
        (evidence("Điều 1\nNgười đủ 18 tuổi được cấp thẻ."),),
    )
    heading_with_title = generator.generate(
        question_record(),
        (evidence("Điều 1. Phạm vi áp dụng.\nNội dung tiếp theo."),),
    )

    assert heading_only.answer_text == "Người đủ 18 tuổi được cấp thẻ."
    assert heading_with_title.answer_text == "Điều 1."


def test_generator_has_no_abbreviation_decimal_or_enumeration_exception() -> None:
    answer = FixtureExtractiveGenerator().generate(
        question_record(),
        (evidence("Ông A. Mức tiền là 1. 000 đồng. Nội dung sau."),),
    )

    assert answer.answer_text == "Ông A."


def test_generator_counts_non_bmp_code_points_and_terminator_at_position_800() -> None:
    text = "😀" + ("a" * 798) + ". Nội dung sau."

    answer = FixtureExtractiveGenerator().generate(question_record(), (evidence(text),))

    assert answer.answer_text == text[:800]
    assert len(answer.answer_text) == 800


def test_generator_truncates_no_whitespace_prefix_and_appends_one_ellipsis() -> None:
    text = "x" * 801

    answer = FixtureExtractiveGenerator().generate(question_record(), (evidence(text),))

    assert answer.answer_text == ("x" * 800) + "…"


def test_generator_returns_fixed_refusal_without_evidence_or_verifier() -> None:
    answer = FixtureExtractiveGenerator().generate(question_record(), ())

    assert answer.answer_text == FIXED_REFUSAL
    assert answer.used_evidence_ids == ()
    assert answer.material_claims == ()
    assert answer.competition_policy == "baseline.v1"


def test_generator_uses_highest_ranked_integrity_valid_evidence() -> None:
    lower_ranked = evidence("Câu trả lời sai.").model_copy(
        update={"evidence_id": "chunk_bbbbbbbbbbbbbbbbbbbbbbbb", "rank": 2}
    )
    highest_ranked = evidence("Câu trả lời đúng.")

    answer = FixtureExtractiveGenerator().generate(
        question_record(),
        (lower_ranked, highest_ranked),
    )

    assert answer.answer_text == "Câu trả lời đúng."
    assert answer.used_evidence_ids == (highest_ranked.evidence_id,)
