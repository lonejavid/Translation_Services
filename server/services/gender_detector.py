"""
Pitch-based gender detection from a mono WAV (voice reference clip).

Uses ``librosa.pyin`` (Probabilistic YIN) for accurate fundamental-frequency (F0)
estimation. Unlike ``piptrack``, pyin is designed for monophonic pitch tracking and
returns voiced/unvoiced flags so only actual speech frames are used.

Research-based speaking-voice F0 ranges (typical adult):
  Male:    85–165 Hz  (median ~120 Hz)
  Female: 185–255 Hz  (median ~210 Hz)
  Overlap / neutral: 165–185 Hz

``GENDER_F0_MALE_FLOOR_OVERRIDE`` (default ``1``): when median F0 is above the female
threshold but the 25th-percentile voiced F0 is still low (chest voice / phrase ends),
prefer **male** — reduces **male → female** errors on bright or heavily compressed speech.

Falls back to ``piptrack`` median if pyin raises (very old librosa build).
"""
from __future__ import annotations

import os

import librosa
import numpy as np

# Lower threshold so detected male/female labels pass through more often.
# 0.45 still gates very uncertain reads while preventing borderline males
# from being collapsed to "unknown" (which could route to wrong Edge voice).
GENDER_CONFIDENCE_THRESHOLD_DEFAULT = 0.45


def _male_floor_override_enabled() -> bool:
    return os.environ.get("GENDER_F0_MALE_FLOOR_OVERRIDE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def gender_confidence_threshold() -> float:
    raw = os.environ.get("GENDER_CONFIDENCE_THRESHOLD", "").strip()
    if not raw:
        return GENDER_CONFIDENCE_THRESHOLD_DEFAULT
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return GENDER_CONFIDENCE_THRESHOLD_DEFAULT


def apply_gender_confidence_gate(result: dict) -> dict:
    """
    If male/female confidence is below threshold, treat as unknown for voice routing
    (neutral Edge + default XTTS prosody).
    """
    g = (result.get("gender") or "unknown").strip().lower()
    conf = float(result.get("confidence") or 0.0)
    th = gender_confidence_threshold()
    if g in ("male", "female") and conf < th:
        out = {**result, "gender": "unknown", "gender_gated": True}
        out["raw_gender"] = g
        out["raw_confidence"] = conf
        return out
    return result


def clone_output_mismatches_expected(
    expected_gender: str | None,
    detected: dict,
    *,
    min_verify_conf: float | None = None,
) -> bool:
    """
    True if synthesized audio gender (from pitch heuristic) disagrees with expected
    male/female strongly enough to fall back to Edge.
    """
    exp = (expected_gender or "").strip().lower()
    if exp not in ("male", "female"):
        return False
    raw = os.environ.get("GENDER_CLONE_VERIFY_MIN_CONF", "").strip()
    if min_verify_conf is None:
        try:
            min_verify_conf = float(raw) if raw else 0.55
        except ValueError:
            min_verify_conf = 0.55
    min_verify_conf = max(0.35, min(0.95, float(min_verify_conf)))

    dg = (detected.get("gender") or "unknown").strip().lower()
    conf = float(detected.get("confidence") or 0.0)
    if conf < min_verify_conf:
        return False
    if dg in ("unknown", "neutral"):
        return False
    return dg != exp


class GenderDetector:
    # Speaking-voice F0 decision boundaries (Hz)
    # Raised female lower boundary: YouTube/podcast male speakers often
    # deliver with elevated pitch (190–210 Hz median). 195 Hz gives more
    # room before classifying a male as female.
    _MALE_UPPER: float = 168.0    # above this → ambiguous / female
    _FEMALE_LOWER: float = 198.0  # below this → ambiguous / male (raised from 185)

    def detect_gender(self, audio_path: str) -> dict:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        return self.detect_gender_numpy(y, sr, min_sec_after_trim=0.5)

    def detect_gender_numpy(
        self,
        y: np.ndarray,
        sr: int,
        *,
        min_sec_after_trim: float = 0.35,
    ) -> dict:
        """
        Pitch-based gender from a mono float waveform (e.g. one STT segment slice).

        ``min_sec_after_trim`` is lower than the full-file path (0.5s) so short
        subtitle windows can still yield a read when enough voiced frames exist.
        """
        y = np.asarray(y, dtype=np.float32).ravel()
        if y.size < int(sr * 0.08):
            return {"gender": "unknown", "confidence": 0.0, "avg_pitch_hz": 0.0}

        y, _ = librosa.effects.trim(y, top_db=20)
        min_samples = int(max(sr * float(min_sec_after_trim), sr * 0.2))
        if len(y) < min_samples:
            return {"gender": "unknown", "confidence": 0.0, "avg_pitch_hz": 0.0}

        f0_stats = self._estimate_f0_pyin(y, sr)
        if f0_stats is None or f0_stats[0] <= 0:
            return {"gender": "unknown", "confidence": 0.0, "avg_pitch_hz": 0.0}

        return self._classify_f0(*f0_stats)

    # ------------------------------------------------------------------
    # F0 estimation
    # ------------------------------------------------------------------

    def _estimate_f0_pyin(self, y: np.ndarray, sr: int) -> tuple[float, float, float] | None:
        """
        Probabilistic YIN (pyin) — accurate monophonic F0 estimation.
        Returns ``(median_f0, p25, p75)`` over voiced frames, or ``None`` on failure.
        """
        try:
            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=50.0,    # below any human speaking voice
                fmax=500.0,   # well above any human speaking voice
                sr=sr,
                frame_length=2048,
            )
            # Keep only voiced frames with a finite, non-zero F0
            mask = voiced_flag & np.isfinite(f0) & (f0 > 50.0)
            voiced_f0 = f0[mask]
            if len(voiced_f0) < 5:
                return None
            med = float(np.median(voiced_f0))
            p25 = float(np.percentile(voiced_f0, 25))
            p75 = float(np.percentile(voiced_f0, 75))
            return med, p25, p75
        except Exception:
            pt = self._estimate_f0_piptrack(y, sr)
            if pt <= 0:
                return None
            return pt, pt, pt

    def _estimate_f0_piptrack(self, y: np.ndarray, sr: int) -> float:
        """Fallback: piptrack median (used only when pyin fails)."""
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=50, fmax=400)
        pitch_values: list[float] = []
        for t in range(pitches.shape[1]):
            index = int(magnitudes[:, t].argmax())
            pitch = float(pitches[index, t])
            if 50 < pitch < 400:
                pitch_values.append(pitch)
        if not pitch_values:
            return 0.0
        return float(np.median(pitch_values))

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_f0(self, f0: float, p25: float | None = None, p75: float | None = None) -> dict:
        """
        Map median F0 → gender + calibrated confidence.

        Confidence design:
          Male  boundary (165 Hz) → ~0.55  |  clear male  (85 Hz) → 0.95
          Female boundary (185 Hz) → ~0.52  |  clear female (250 Hz) → 0.95
          Overlap zone (165–185 Hz) → neutral, 0.45 — unless quartiles break the tie.
        """
        observed_median = float(f0)

        # Resolve ambiguous median using pitch distribution.
        # Stricter resolution: always push out of neutral zone using quartile evidence.
        # "neutral" wastes the voice-clone gender hint — force male or female.
        if self._MALE_UPPER <= f0 <= self._FEMALE_LOWER and p25 is not None and p75 is not None:
            if p75 > 190.0:
                # Upper quartile clearly in female range → female
                f0 = self._FEMALE_LOWER + 10.0
            elif p25 < 155.0:
                # Lower quartile clearly in male range → male
                f0 = self._MALE_UPPER - 10.0
            elif f0 < (self._MALE_UPPER + self._FEMALE_LOWER) / 2:
                # Closer to male boundary → default male
                f0 = self._MALE_UPPER - 5.0
            else:
                # Closer to female boundary → default female
                f0 = self._FEMALE_LOWER + 5.0

        if f0 < self._MALE_UPPER:
            gender = "male"
            # Linear decay from 0.95 at 85 Hz to ~0.55 at 165 Hz
            raw_conf = 0.95 - (f0 - 85.0) / (self._MALE_UPPER - 85.0) * 0.40
            confidence = float(min(0.95, max(0.45, raw_conf)))
        elif f0 > self._FEMALE_LOWER:
            # Male with bright mic / compression: median tracks high harmonics (~190–220 Hz)
            # while p25 stays in chest-voice range.
            if (
                _male_floor_override_enabled()
                and p25 is not None
                and p25 < 168.0   # raised from 155 — catches energetic male speakers
                and observed_median < 240.0
            ):
                gender = "male"
                raw_conf = 0.58 + max(0.0, (155.0 - p25) / 60.0) * 0.26
                confidence = float(min(0.86, max(0.50, raw_conf)))
            else:
                gender = "female"
                raw_conf = 0.52 + (observed_median - self._FEMALE_LOWER) / 65.0 * 0.43
                confidence = float(min(0.95, max(0.45, raw_conf)))
        else:
            # 165–185 Hz overlap zone — too ambiguous to commit
            gender = "neutral"
            confidence = 0.45

        result = {
            "gender": gender,
            "confidence": confidence,
            "avg_pitch_hz": observed_median,
        }
        return apply_gender_confidence_gate(result)


_detector: GenderDetector | None = None


def get_gender_detector() -> GenderDetector:
    global _detector
    if _detector is None:
        _detector = GenderDetector()
    return _detector


def detect_gender_from_audio(audio_path: str) -> dict:
    return get_gender_detector().detect_gender(audio_path)
