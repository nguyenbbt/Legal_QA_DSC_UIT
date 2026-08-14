"""Offline reproduction and isolated parity execution of the supplied scorer."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib
import io
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn, cast

import nltk  # type: ignore[import-untyped]
import numpy as np
from nltk.translate.meteor_score import meteor_score  # type: ignore[import-untyped]

_SUPPLIED_SCORER_SHA256 = "f04843fbfad26d41356506d8e49692a7c8a0ed1b9f065a3a8472fa6398a5aa95"
_SUPPLIED_ROUGE_SHA256 = {
    "__init__.py": "7d007905e3e38fc10b41821cf4f49807732208e683dec8b2ff48ac1c4b2cbc91",
    "rouge_scorer.py": "9484c5fd05e22cd28b5053bf9de586b3620cf65b0c38052b8690badd51f31d1a",
    "scoring.py": "b3fc153499e484665294ddbaeb876452b40a8fa2120e824b804928c96e3f2c1a",
    "tokenize.py": "dc91cea8f09507f744549160c458031d0956e35fa230c80d01f990eba20a7403",
    "tokenizers.py": "2b7b9dae505ce8739064ba8e4791b871b13aa1f65bc54e25692f1a9ba8600508",
}
_REQUIRED_NLTK_ARCHIVES = (
    ("wordnet", Path("corpora/wordnet.zip")),
    ("omw-1.4", Path("corpora/omw-1.4.zip")),
)


class ScorerError(Exception):
    """Safe typed failure at the exact-scorer boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _fail(code: str, message: str) -> NoReturn:
    raise ScorerError(code, message)


@dataclass(frozen=True, slots=True)
class PerQueryScore:
    """One ordered official-exact metric pair."""

    question_id: str
    rouge_l: float
    meteor: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Ordered per-query and exact NumPy macro results."""

    question_ids: tuple[str, ...]
    per_query: tuple[PerQueryScore, ...]
    macro_rouge_l: float
    macro_meteor: float


def _require_offline_resources(nltk_data_root: Path) -> Path:
    root = nltk_data_root.resolve(strict=False)
    for resource_id, relative_path in _REQUIRED_NLTK_ARCHIVES:
        candidate = root / relative_path
        if candidate.is_symlink() or not candidate.is_file():
            _fail(
                "OFFLINE_RESOURCE_MISSING",
                f"required manifested resource is missing: {resource_id}",
            )
    return root


@contextlib.contextmanager
def _offline_nltk_path(root: Path) -> Iterator[None]:
    original = list(nltk.data.path)
    nltk.data.path[:] = [str(root)]
    try:
        yield
    finally:
        nltk.data.path[:] = original


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def _verified_file_bytes(path: Path, expected_sha256: str, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        _fail("SCORER_SOURCE_INVALID", f"reviewed {label} source is missing")
    try:
        source_bytes = path.read_bytes()
    except OSError as exc:
        raise ScorerError(
            "SCORER_SOURCE_INVALID", f"reviewed {label} source is unreadable"
        ) from exc
    if hashlib.sha256(source_bytes).hexdigest() != expected_sha256:
        _fail(
            "SCORER_SOURCE_CHECKSUM_MISMATCH",
            f"reviewed {label} source checksum does not match",
        )
    return source_bytes


def _verify_rouge_bundle(scorer_root: Path) -> None:
    package_root = scorer_root / "rouge_score"
    for filename, expected_sha256 in _SUPPLIED_ROUGE_SHA256.items():
        _verified_file_bytes(
            package_root / filename,
            expected_sha256,
            label=f"rouge_score/{filename}",
        )


def _verified_scoring_source(scoring_path: Path, scorer_root: Path) -> bytes:
    source_bytes = _verified_file_bytes(
        scoring_path,
        _SUPPLIED_SCORER_SHA256,
        label="scoring.py",
    )
    try:
        parent = scoring_path.parent.resolve(strict=True)
        expected_parent = scorer_root.resolve(strict=True)
    except OSError as exc:
        raise ScorerError("SCORER_SOURCE_INVALID", "supplied scorer path is invalid") from exc
    if parent != expected_parent:
        _fail(
            "SCORER_SOURCE_INVALID",
            "scoring.py and its supplied rouge_score package must share one directory",
        )
    return source_bytes


def _load_bundled_rouge(scorer_root: Path) -> ModuleType:
    _verify_rouge_bundle(scorer_root)
    resolved_root = scorer_root.resolve(strict=True)
    existing = sys.modules.get("rouge_score.rouge_scorer")
    if existing is not None:
        module_path = getattr(existing, "__file__", None)
        if not isinstance(module_path, str) or not _is_within(Path(module_path), resolved_root):
            _fail(
                "SCORER_IMPLEMENTATION_MISMATCH",
                "loaded rouge_score does not come from the supplied scorer",
            )
        return existing

    sys.path.insert(0, str(resolved_root))
    try:
        module = importlib.import_module("rouge_score.rouge_scorer")
    except (ImportError, OSError) as exc:
        raise ScorerError(
            "SCORER_IMPLEMENTATION_MISSING",
            "supplied rouge_score implementation cannot be loaded",
        ) from exc
    finally:
        del sys.path[0]
    module_path = getattr(module, "__file__", None)
    if not isinstance(module_path, str) or not _is_within(Path(module_path), resolved_root):
        _fail(
            "SCORER_IMPLEMENTATION_MISMATCH",
            "rouge_score resolved outside the supplied scorer directory",
        )
    return module


def _validated_inputs(
    predictions: Mapping[str, Mapping[str, object]],
    references: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if len(predictions) != len(references):
        _fail(
            "SCORER_SAMPLE_COUNT_MISMATCH",
            "prediction and reference sample counts do not match",
        )
    if not predictions:
        _fail("SCORER_EMPTY_INPUT", "official scoring requires at least one sample")
    question_ids = tuple(predictions)
    prediction_text: list[str] = []
    reference_text: list[str] = []
    for question_id in question_ids:
        prediction = predictions[question_id]
        if not isinstance(prediction, Mapping) or "answer" not in prediction:
            _fail(
                "SCORER_PREDICTION_INVALID",
                "every prediction must contain an answer member",
            )
        if question_id not in references:
            _fail(
                "SCORER_QUESTION_ID_MISMATCH",
                "prediction question ID is absent from references",
            )
        prediction_text.append(str(prediction["answer"]))
        reference_text.append(str(references[question_id]))
    return question_ids, tuple(prediction_text), tuple(reference_text)


def evaluate_official_exact(
    predictions: Mapping[str, Mapping[str, object]],
    references: Mapping[str, object],
    *,
    scorer_root: Path,
    nltk_data_root: Path,
) -> EvaluationResult:
    """Reproduce ``eval_qa`` and expose its ordered per-query metric values."""

    resource_root = _require_offline_resources(nltk_data_root)
    rouge_module = _load_bundled_rouge(scorer_root)
    question_ids, prediction_text, reference_text = _validated_inputs(predictions, references)
    scorer_type = rouge_module.RougeScorer
    scorer = scorer_type(["rougeL"], use_stemmer=False)
    rows: list[PerQueryScore] = []
    with _offline_nltk_path(resource_root):
        for question_id, prediction, reference in zip(
            question_ids, prediction_text, reference_text, strict=True
        ):
            rouge_l = float(scorer.score(reference, prediction)["rougeL"].fmeasure)
            meteor = float(meteor_score([reference.split()], prediction.split()))
            rows.append(PerQueryScore(question_id=question_id, rouge_l=rouge_l, meteor=meteor))
    macro_rouge_l = float(np.array([row.rouge_l for row in rows]).mean())
    macro_meteor = float(np.array([row.meteor for row in rows]).mean())
    return EvaluationResult(
        question_ids=question_ids,
        per_query=tuple(rows),
        macro_rouge_l=macro_rouge_l,
        macro_meteor=macro_meteor,
    )


def _extract_supplied_eval_qa(
    scoring_path: Path, scorer_root: Path, rouge_module: ModuleType
) -> Callable[[Mapping[str, Mapping[str, object]], Mapping[str, object]], Mapping[str, object]]:
    try:
        source = _verified_scoring_source(scoring_path, scorer_root).decode("utf-8")
        parsed = ast.parse(source, filename=scoring_path.name)
    except (UnicodeError, SyntaxError) as exc:
        raise ScorerError("SCORER_SOURCE_INVALID", "supplied scoring.py cannot be parsed") from exc
    functions = [
        node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "eval_qa"
    ]
    if len(functions) != 1:
        _fail("SCORER_SOURCE_INVALID", "supplied scoring.py must contain exactly one eval_qa")
    function_statements: list[ast.stmt] = list(functions)
    isolated = ast.fix_missing_locations(ast.Module(body=function_statements, type_ignores=[]))
    namespace: dict[str, Any] = {
        "np": np,
        "meteor_score": meteor_score,
        "rouge_scorer": rouge_module,
    }
    exec(compile(isolated, scoring_path.name, "exec"), namespace)
    return cast(
        Callable[[Mapping[str, Mapping[str, object]], Mapping[str, object]], Mapping[str, object]],
        namespace["eval_qa"],
    )


def evaluate_supplied_scorer(
    predictions: Mapping[str, Mapping[str, object]],
    references: Mapping[str, object],
    *,
    scoring_path: Path,
    scorer_root: Path,
    nltk_data_root: Path,
) -> EvaluationResult:
    """Execute only the supplied ``eval_qa`` body, excluding unsafe module setup."""

    resource_root = _require_offline_resources(nltk_data_root)
    rouge_module = _load_bundled_rouge(scorer_root)
    question_ids, _, _ = _validated_inputs(predictions, references)
    eval_qa = _extract_supplied_eval_qa(scoring_path, scorer_root, rouge_module)

    def invoke(
        selected_predictions: Mapping[str, Mapping[str, object]],
        selected_references: Mapping[str, object],
    ) -> tuple[float, float]:
        with _offline_nltk_path(resource_root), contextlib.redirect_stdout(io.StringIO()):
            result = eval_qa(selected_predictions, selected_references)
        return float(cast(Any, result["rouge"])), float(cast(Any, result["meteor"]))

    macro_rouge_l, macro_meteor = invoke(predictions, references)
    rows: list[PerQueryScore] = []
    for question_id in question_ids:
        rouge_l, meteor = invoke(
            {question_id: predictions[question_id]},
            {question_id: references[question_id]},
        )
        rows.append(PerQueryScore(question_id=question_id, rouge_l=rouge_l, meteor=meteor))
    return EvaluationResult(
        question_ids=question_ids,
        per_query=tuple(rows),
        macro_rouge_l=macro_rouge_l,
        macro_meteor=macro_meteor,
    )


def supplied_scorer_provenance(scoring_path: Path) -> dict[str, object]:
    """Return reviewed source identities without exposing local absolute paths."""

    source_bytes = _verified_scoring_source(scoring_path, scoring_path.parent)
    try:
        display_path = scoring_path.resolve(strict=True).relative_to(
            Path.cwd().resolve(strict=True)
        )
    except (OSError, ValueError) as exc:
        raise ScorerError(
            "SCORER_SOURCE_UNSUPPORTED",
            "supplied scorer must be a regular file inside the project",
        ) from exc
    return {
        "path": display_path.as_posix(),
        "checksum": f"sha256:{hashlib.sha256(source_bytes).hexdigest()}",
        "entrypoint": "eval_qa",
        "rouge_source_checksums": {
            filename: f"sha256:{checksum}"
            for filename, checksum in sorted(_SUPPLIED_ROUGE_SHA256.items())
        },
        "top_level_downloads_executed": 0,
    }
