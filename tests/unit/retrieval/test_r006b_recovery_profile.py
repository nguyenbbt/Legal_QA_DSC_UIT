from __future__ import annotations

import pytest
from scripts.run_r006b_packing_recovery import _require_checksum

from legal_rag.domain.checksums import checksum_bytes


def test_r006b_profile_rejects_changed_artifact_bytes() -> None:
    expected = checksum_bytes(b"approved")

    _require_checksum(b"approved", expected, "fixture")
    with pytest.raises(ValueError, match="D-058 fixture checksum changed"):
        _require_checksum(b"changed", expected, "fixture")
