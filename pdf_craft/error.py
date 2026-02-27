from typing import Callable

from .metering import InterruptedKind, OCRTokensMetering


class PDFError(Exception):
    def __init__(self, message: str, page_index: int | None = None) -> None:
        super().__init__(message)
        self.page_index: int | None = page_index


class OCRError(Exception):
    def __init__(self, message: str, page_index: int, step_index: int) -> None:
        super().__init__(message)
        self.page_index: int = page_index
        self.step_index: int = step_index


class TokenLimitError(Exception):
    """Token 用量超限（用于 OCR 前的预检查或 API 后端）。"""

    def __init__(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        super().__init__()
        self.input_tokens: int = input_tokens
        self.output_tokens: int = output_tokens


class AbortError(Exception):
    """用户中止 OCR（用于 API 后端或未安装 doc-page-extractor 时的 check_aborted）。"""

    input_tokens: int = 0
    output_tokens: int = 0


def is_inline_error(error: Exception) -> bool:
    return isinstance(error, (PDFError, OCRError))


IgnorePDFErrorsChecker = bool | Callable[[PDFError], bool]
IgnoreOCRErrorsChecker = bool | Callable[[OCRError], bool]


# 不可直接用 doc-page-extractor 的 Error，该库的一切都是懒加载，若暴露，则无法懒加载
class InterruptedError(Exception):
    """Raised when the operation is interrupted by the user."""

    def __init__(self, kind: InterruptedKind, metering: OCRTokensMetering) -> None:
        super().__init__()
        self._kind: InterruptedKind = kind
        self._metering: OCRTokensMetering = metering

    @property
    def kind(self) -> InterruptedKind:
        return self._kind

    @property
    def metering(self) -> OCRTokensMetering:
        return self._metering


def to_interrupted_error(error: Exception) -> InterruptedError | None:
    if isinstance(error, TokenLimitError):
        return InterruptedError(
            kind=InterruptedKind.TOKEN_LIMIT_EXCEEDED,
            metering=OCRTokensMetering(
                input_tokens=error.input_tokens,
                output_tokens=error.output_tokens,
            ),
        )
    if isinstance(error, AbortError):
        return InterruptedError(
            kind=InterruptedKind.ABORT,
            metering=OCRTokensMetering(
                input_tokens=getattr(error, "input_tokens", 0),
                output_tokens=getattr(error, "output_tokens", 0),
            ),
        )
    try:
        from doc_page_extractor import (
            AbortError as DPEAbortError,
            ExtractionAbortedError,
            TokenLimitError as DPETokenLimitError,
        )
    except ImportError:
        return None
    if isinstance(error, ExtractionAbortedError):
        kind: InterruptedKind | None = None
        if isinstance(error, DPEAbortError):
            kind = InterruptedKind.ABORT
        elif isinstance(error, DPETokenLimitError):
            kind = InterruptedKind.TOKEN_LIMIT_EXCEEDED

        if kind is not None:
            return InterruptedError(
                kind=kind,
                metering=OCRTokensMetering(
                    input_tokens=error.input_tokens,
                    output_tokens=error.output_tokens,
                ),
            )
    return None
