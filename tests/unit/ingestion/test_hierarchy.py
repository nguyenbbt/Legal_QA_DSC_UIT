from __future__ import annotations

import unicodedata

from legal_rag.ingestion.hierarchy import hierarchy_regex_manifest, parse_hierarchy


def test_hierarchy_parser_applies_explicit_precedence_and_paths() -> None:
    passage = (
        "Phần I. Quy định chung\n"
        "Chương I. Phạm vi\n"
        "Mục 1. Chủ thể\n"
        "Tiểu mục 1A. Điều kiện\n"
        "Điều 2. Cấp thẻ\n"
        "Khoản 1. Người được cấp\n"
        "Điểm a. Đủ tuổi\n"
    )

    result = parse_hierarchy(passage)

    assert [node.rule_id for node in result.nodes] == [
        "HIER_PART",
        "HIER_CHAPTER",
        "HIER_SECTION",
        "HIER_SUBSECTION",
        "HIER_ARTICLE",
        "HIER_CLAUSE",
        "HIER_POINT",
    ]
    assert result.nodes[-1].hierarchy_path == (
        "Phần I",
        "Chương I",
        "Mục 1",
        "Tiểu mục 1A",
        "Điều 2",
        "Khoản 1",
        "Điểm a",
    )
    assert result.nodes[-1].title == "Đủ tuổi"


def test_implicit_clause_and_point_require_active_parent_scope() -> None:
    passage = "1. ngoài phạm vi\nĐiều 1\n1. Nội dung khoản\na) Nội dung điểm\n"

    result = parse_hierarchy(passage)

    assert [node.rule_id for node in result.nodes] == [
        "HIER_ARTICLE",
        "IMPLICIT_CLAUSE",
        "IMPLICIT_POINT",
    ]
    assert result.nodes[1].hierarchy_path == ("Điều 1", "Khoản 1")
    assert result.nodes[2].hierarchy_path == ("Điều 1", "Khoản 1", "Điểm a")
    assert result.nodes[0].heading_only is True
    assert result.nodes[1].heading_only is False


def test_hierarchy_offsets_reconstruct_canonical_nfc_lines() -> None:
    passage = unicodedata.normalize("NFC", unicodedata.normalize("NFD", "Điều 1. Điều kiện"))

    node = parse_hierarchy(passage).nodes[0]

    assert passage[node.canonical_start : node.canonical_end] == "Điều 1. Điều kiện"
    assert node.title == "Điều kiện"


def test_discontinuous_numbering_warns_without_renumbering() -> None:
    passage = "Điều 1\n1. Một\n3. Ba\n"

    result = parse_hierarchy(passage)

    assert [node.ordinal for node in result.nodes] == ["1", "1", "3"]
    assert [warning.code for warning in result.warnings] == ["HIER_NUMBERING_DISCONTINUOUS"]


def test_hierarchy_regex_manifest_pins_expanded_patterns_and_checksum() -> None:
    manifest = hierarchy_regex_manifest()

    assert manifest["schema_version"] == "hierarchy-regex.v1"
    assert [rule["rule_id"] for rule in manifest["rules"]][-2:] == [
        "IMPLICIT_CLAUSE",
        "IMPLICIT_POINT",
    ]
    assert manifest["patterns_checksum"].startswith("sha256:")
