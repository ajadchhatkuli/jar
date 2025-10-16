from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Deque, Optional, Tuple, Union

import cv2
import numpy as np

try:  # Optional dependency; raise helpful error only when used.
    from imquality import brisque as _brisque
except ImportError:  # pragma: no cover - handled at runtime.
    _brisque = None
else:
    # imquality still depends on the deprecated ``multichannel`` kwarg in skimage.
    try:
        from skimage import transform as _sk_transform
        from skimage import color as _sk_color
        import inspect

        _rescale_sig = inspect.signature(_sk_transform.rescale)
        if "multichannel" not in _rescale_sig.parameters:
            _orig_rescale = _sk_transform.rescale

            def _rescale_compat(
                image,
                scale,
                *args,
                multichannel=None,
                **kwargs,
            ):
                if multichannel is not None and "channel_axis" not in kwargs:
                    kwargs["channel_axis"] = -1 if multichannel else None
                return _orig_rescale(image, scale, *args, **kwargs)

            _sk_transform.rescale = _rescale_compat  # type: ignore[assignment]

        _rgb2gray_sig = inspect.signature(_sk_color.rgb2gray)
        if "channel_axis" in _rgb2gray_sig.parameters:
            _orig_rgb2gray = _sk_color.rgb2gray

            def _rgb2gray_compat(image, *args, **kwargs):
                if getattr(image, "ndim", 3) < 3:
                    arr = np.asarray(image)
                    if arr.dtype == np.uint8:
                        arr = arr / 255.0
                    return arr
                if "channel_axis" not in kwargs:
                    kwargs["channel_axis"] = -1
                return _orig_rgb2gray(image, *args, **kwargs)

            _sk_color.rgb2gray = _rgb2gray_compat  # type: ignore[assignment]

        try:
            import scipy  # type: ignore

            if not hasattr(scipy, "ndarray"):
                scipy.ndarray = np.ndarray  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass


class FrameQualityMethod(str, Enum):
    """Supported methods for frame quality assessment."""

    LAPLACIAN = "laplacian"
    BRISQUE = "brisque"


def _laplacian_score(image: np.ndarray) -> float:
    """Return the variance of Laplacian focus score (higher means sharper)."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _prune_buffer(
    buffer: Deque[Tuple[float, bytes]],
    current_time: float,
    max_age_seconds: float,
) -> None:
    """Drop frames older than ``max_age_seconds`` relative to ``current_time``."""
    if max_age_seconds <= 0:
        buffer.clear()
        return
    threshold = current_time - max_age_seconds
    while buffer and buffer[0][0] < threshold:
        buffer.popleft()


def _require_brisque() -> None:
    if _brisque is None:
        raise RuntimeError(
            "queue_frame_with_brisque requires the 'imquality' package. "
            "Install it with `pip install imquality`."
        )


def _brisque_score(image: np.ndarray) -> float:
    """
    Return the BRISQUE score for ``image``.

    Lower scores indicate higher quality. This helper attempts the native score
    call first and falls back to grayscale if the implementation requires it.
    """
    _require_brisque()
    try:
        return float(_brisque.score(image))
    except Exception:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(_brisque.score(gray))


def queue_frame_with_quality(
    buffer: Deque[Tuple[float, bytes]],
    frame: np.ndarray,
    timestamp: float,
    target_fps: float,
    *,
    max_age_seconds: float,
    last_sample_time: Optional[float],
    method: Union[str, FrameQualityMethod] = FrameQualityMethod.LAPLACIAN,
    brisque_threshold: float = 60.0,
    laplacian_threshold: float = 120.0,
    jpeg_bytes: Optional[bytes] = None,
    jpeg_quality: int = 85,
) -> Tuple[Optional[float], Optional[bytes]]:
    """
    Conditionally append ``frame`` to ``buffer`` according to sampling rate and
    a quality metric.

    Args:
        buffer: Deque storing ``(timestamp, jpeg_bytes)`` tuples.
        frame: BGR ``numpy.ndarray`` already resized if desired.
        timestamp: Absolute timestamp (seconds) of the frame.
        target_fps: Desired sampling rate; <=0 disables rate limiting.
        max_age_seconds: Sliding window size to retain in the buffer.
        last_sample_time: Timestamp of the last accepted frame.
        method: Frame quality assessment strategy (`laplacian` or `brisque`).
        brisque_threshold: Maximum BRISQUE score allowed (lower is sharper).
        laplacian_threshold: Minimum variance of Laplacian allowed (higher is sharper).
        jpeg_bytes: Optional pre-encoded JPEG payload. If omitted, the frame is
            encoded using ``jpeg_quality``.
        jpeg_quality: JPEG quality used when encoding ``frame``.

    Returns:
        A tuple ``(new_last_sample_time, encoded_bytes_if_added)``. When the
        frame is rejected, ``new_last_sample_time`` equals the incoming
        ``last_sample_time`` and the second element is ``None``.
    """
    interval = 0.0 if target_fps <= 0 else 1.0 / target_fps
    if (
        last_sample_time is not None
        and interval > 0
        and (timestamp - last_sample_time) < interval - 1e-6
    ):
        _prune_buffer(buffer, timestamp, max_age_seconds)
        return last_sample_time, None

    method_enum = (
        method if isinstance(method, FrameQualityMethod) else FrameQualityMethod(method)
    )

    if method_enum is FrameQualityMethod.BRISQUE:
        score = _brisque_score(frame)
        if score > brisque_threshold:
            _prune_buffer(buffer, timestamp, max_age_seconds)
            return last_sample_time, None
    else:
        score = _laplacian_score(frame)
        if score < laplacian_threshold:
            _prune_buffer(buffer, timestamp, max_age_seconds)
            return last_sample_time, None

    if jpeg_bytes is None:
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        )
        if not ok:
            _prune_buffer(buffer, timestamp, max_age_seconds)
            return last_sample_time, None
        jpeg_bytes = encoded.tobytes()

    buffer.append((timestamp, jpeg_bytes))
    _prune_buffer(buffer, timestamp, max_age_seconds)
    return timestamp, jpeg_bytes


def queue_frame_with_brisque(
    buffer: Deque[Tuple[float, bytes]],
    frame: np.ndarray,
    timestamp: float,
    target_fps: float,
    *,
    max_age_seconds: float,
    last_sample_time: Optional[float],
    brisque_threshold: float = 60.0,
    jpeg_bytes: Optional[bytes] = None,
    jpeg_quality: int = 85,
) -> Tuple[Optional[float], Optional[bytes]]:
    """
    Backwards-compatible alias for BRISQUE-only frame selection.
    """
    return queue_frame_with_quality(
        buffer,
        frame,
        timestamp,
        target_fps,
        max_age_seconds=max_age_seconds,
        last_sample_time=last_sample_time,
        method=FrameQualityMethod.BRISQUE,
        brisque_threshold=brisque_threshold,
        jpeg_bytes=jpeg_bytes,
        jpeg_quality=jpeg_quality,
    )
