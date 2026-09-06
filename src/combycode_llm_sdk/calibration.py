"""What this model actually replies with, at this input size.

Transposed from `unified-library-ts/src/helpers/calibration-store.ts`.

A static "assume 512 output tokens" is a number no run has ever confirmed. Once
real completions have been recorded, the estimator can stop guessing and use
what this model replies with at this input size -- which is often an order of
magnitude away from the default and changes what a run can afford.

Learning is keyed by `provider/model#input-bucket`, not by exact input size: a
2,001-token prompt and a 2,002-token one are the same question, and keying on
the exact number would mean every prompt is a fresh key that has learned
nothing.

Two statistics per key, because two bounds need them. An EWMA for the EXPECTED
reply -- smoothed, so one unusually long answer does not move the estimate far
-- and a coarse histogram for the HIGH bound, since a p90 needs the shape of the
distribution and a mean cannot describe a tail.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Upper edges (exclusive) for input-size buckets. The last bucket is open.
INPUT_SIZE_BUCKET_EDGES: tuple[int, ...] = (500, 2_000, 8_000, 32_000)

INPUT_SIZE_BUCKET_LABELS: tuple[str, ...] = (
    "0-500",
    "500-2000",
    "2000-8000",
    "8000-32000",
    "32000+",
)

#: Smoothing for the mean. Higher adapts faster; lower is steadier. At 0.15 a
#: single outlier moves the estimate by about a seventh of its distance.
CALIBRATION_EWMA_ALPHA = 0.15

P90_HISTOGRAM_BIN_COUNT = 32
P90_HISTOGRAM_BIN_WIDTH = 256
CALIBRATION_HIGH_QUANTILE = 0.9

KEY_PREFIX = "output-calibration:"


@dataclass(frozen=True)
class CalibrationObservation:
    """One finished call, as the four numbers calibration learns from."""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int


def observation_from_completion(ctx: Mapping[str, Any]) -> CalibrationObservation:
    """What an `onCompletion` payload says, as something `record()` accepts.

    The input count falls back to the ESTIMATE when the provider reported zero.
    Several providers omit input tokens entirely, and a 0 would not merely be
    missing -- it would file the call under the smallest input bucket and drag
    that bucket's learned output length toward a value no real call produced.
    """
    usage = ctx.get("usage") or {}
    reported = int(usage.get("inputTokens") or usage.get("input_tokens") or 0)
    estimated = int(ctx.get("estimatedInputTokens") or ctx.get("estimated_input_tokens") or 0)
    return CalibrationObservation(
        provider=str(ctx.get("provider") or ""),
        model=str(ctx.get("model") or ""),
        input_tokens=reported or estimated,
        output_tokens=int(usage.get("outputTokens") or usage.get("output_tokens") or 0),
    )


def input_bucket_label(input_tokens: int) -> str:
    """Which size bucket a prompt of this length falls in."""
    for index, edge in enumerate(INPUT_SIZE_BUCKET_EDGES):
        if input_tokens < edge:
            return INPUT_SIZE_BUCKET_LABELS[index]
    return INPUT_SIZE_BUCKET_LABELS[-1]


def calibration_key(provider: str, model: str, bucket: str) -> str:
    return f"{KEY_PREFIX}{provider}/{model}#{bucket}"


@dataclass
class OutputCalibrationEntry:
    """Running statistics for one key."""

    key: str
    #: The smoothed mean of observed output tokens.
    ewma_mean: float = 0.0
    #: Fixed-width bins; the last is open-ended.
    histogram: list[int] = field(default_factory=lambda: [0] * P90_HISTOGRAM_BIN_COUNT)
    count: int = 0
    last_updated: float = 0.0

    def quantile(self, q: float = CALIBRATION_HIGH_QUANTILE) -> int:
        """The bin edge below which `q` of the observations fall.

        Coarse on purpose: an exact quantile needs every observation kept, and
        the point of this is to be cheap enough to update on every completion.
        """
        if self.count == 0:
            return 0
        target = q * self.count
        seen = 0
        for index, bin_count in enumerate(self.histogram):
            seen += bin_count
            if seen >= target:
                return (index + 1) * P90_HISTOGRAM_BIN_WIDTH
        return P90_HISTOGRAM_BIN_COUNT * P90_HISTOGRAM_BIN_WIDTH

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "ewmaMean": self.ewma_mean,
            "histogram": list(self.histogram),
            "count": self.count,
            "lastUpdated": self.last_updated,
        }

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> OutputCalibrationEntry:
        histogram = list(raw.get("histogram") or [])
        if len(histogram) != P90_HISTOGRAM_BIN_COUNT:
            histogram = [0] * P90_HISTOGRAM_BIN_COUNT
        return OutputCalibrationEntry(
            key=str(raw.get("key") or ""),
            ewma_mean=float(raw.get("ewmaMean") or 0.0),
            histogram=histogram,
            count=int(raw.get("count") or 0),
            last_updated=float(raw.get("lastUpdated") or 0.0),
        )


class OutputCalibrationStore:
    """Per-key statistics, in memory or in any `Persistence`."""

    def __init__(self, persistence: Any = None, *, ewma_alpha: float = CALIBRATION_EWMA_ALPHA) -> None:
        from .persistence import MemoryPersistence

        self.persistence = persistence if persistence is not None else MemoryPersistence()
        self.ewma_alpha = ewma_alpha

    def record(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> None:
        """Fold one observed completion into this key's statistics."""
        key = calibration_key(provider, model, input_bucket_label(input_tokens))
        raw = self.persistence.get(key)
        entry = (
            OutputCalibrationEntry.from_dict(raw)
            if isinstance(raw, Mapping)
            # Seeded WITH the observation rather than from zero: an EWMA started
            # at 0 spends its first dozen samples climbing out of a number
            # nobody observed.
            else OutputCalibrationEntry(key=key, ewma_mean=float(output_tokens))
        )
        if entry.count > 0:
            entry.ewma_mean = (
                self.ewma_alpha * output_tokens + (1 - self.ewma_alpha) * entry.ewma_mean
            )
        entry.count += 1
        index = min(output_tokens // P90_HISTOGRAM_BIN_WIDTH, P90_HISTOGRAM_BIN_COUNT - 1)
        entry.histogram[index] += 1
        entry.last_updated = time.time() * 1000
        self.persistence.set(key, entry.to_dict())

    def get(self, provider: str, model: str, input_tokens: int) -> OutputCalibrationEntry | None:
        """What has been learned for this key, or None when nothing has."""
        key = calibration_key(provider, model, input_bucket_label(input_tokens))
        raw = self.persistence.get(key)
        return OutputCalibrationEntry.from_dict(raw) if isinstance(raw, Mapping) else None

    def keys(self) -> Sequence[str]:
        return list(self.persistence.list(KEY_PREFIX))


__all__ = [
    "CALIBRATION_EWMA_ALPHA",
    "CALIBRATION_HIGH_QUANTILE",
    "INPUT_SIZE_BUCKET_EDGES",
    "INPUT_SIZE_BUCKET_LABELS",
    "KEY_PREFIX",
    "P90_HISTOGRAM_BIN_COUNT",
    "P90_HISTOGRAM_BIN_WIDTH",
    "OutputCalibrationEntry",
    "OutputCalibrationStore",
    "calibration_key",
    "input_bucket_label",
]
