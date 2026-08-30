from __future__ import annotations

from legal_rag.retrieval.legal_citations import (
    mask_legal_reference_numbers,
    normalize_legal_ordinal,
    parse_legal_citations,
)


def test_parser_expands_clause_list_and_normalizes_article_clause_point() -> None:
    text = (
        "Căn cứ khoản 3, khoản 05 Điều 017 Nghị định 90/2017/NĐ-CP; "
        "điểm A khoản 9 Điều 3 Nghị định 07/2022/NĐ-CP."
    )

    first = parse_legal_citations(text)
    second = parse_legal_citations(text)

    assert first == second
    assert [(item.document_number, item.article, item.clause, item.point) for item in first] == [
        ("90/2017/NĐ-CP", "17", "3", None),
        ("90/2017/NĐ-CP", "17", "5", None),
        ("07/2022/NĐ-CP", "3", "9", "a"),
    ]
    assert all(text[item.canonical_start : item.canonical_end] for item in first)


def test_parser_extracts_law_identity_and_other_canonical_coordinates() -> None:
    citations = parse_legal_citations(
        "Theo Chương IV Điều 30 Bộ luật Tố tụng dân sự 2015 và "
        "tiểu mục 3.1 Mục 3 Công văn 1949/BTP-BTTP."
    )

    assert citations[0].law_identity == "bo luat to tung dan su 2015"
    assert citations[0].article == "30"
    assert ("chapter", "iv") in citations[0].other_coordinates
    assert citations[1].document_number == "1949/BTP-BTTP"
    assert citations[1].other_coordinates == (("section", "3"), ("subsection", "3.1"))


def test_explicit_document_number_wins_over_overlapping_law_name_text() -> None:
    citations = parse_legal_citations("Theo Điều 3 Luật số 45/2013/QH13 quy định.")

    assert len(citations) == 1
    assert citations[0].document_number == "45/2013/QH13"
    assert citations[0].law_identity is None


def test_ordinal_normalization_is_kind_aware_and_fail_closed() -> None:
    assert normalize_legal_ordinal("article", "0017") == "17"
    assert normalize_legal_ordinal("clause", "05a") == "5a"
    assert normalize_legal_ordinal("point", "Đ") == "đ"
    assert normalize_legal_ordinal("chapter", "IV") == "iv"
    assert normalize_legal_ordinal("article", "not-an-ordinal") is None


def test_reference_mask_keeps_semantic_numbers_only() -> None:
    text = "Điều 50 Nghị định 90/2017/NĐ-CP và Bộ luật Hình sự 2015 phạt 15% trong 30 ngày."

    masked = mask_legal_reference_numbers(text)

    assert "50" not in masked
    assert "90/2017/NĐ-CP" not in masked
    assert "2015" not in masked
    assert "15%" in masked
    assert "30 ngày" in masked
