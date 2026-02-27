"""OCR 页面提取器后端抽象，支持本地模型与第三方 API 两种实现。"""

from pathlib import Path
from typing import Protocol

from PIL.Image import Image

from ..common import AssetHub
from ..metering import AbortedCheck
from .types import DeepSeekOCRSize, Page


class PageExtractorBackend(Protocol):
    """页面提取器后端协议。本地 doc-page-extractor 与第三方 API 均实现此接口。"""

    def download_models(self, revision: str | None) -> None:
        """预下载或刷新模型（API 后端可为 no-op）。"""
        ...

    def load_models(self) -> None:
        """加载模型到内存/显存（API 后端可为 no-op）。"""
        ...

    def image2page(
        self,
        image: Image,
        page_index: int,
        asset_hub: AssetHub,
        ocr_size: DeepSeekOCRSize,
        includes_footnotes: bool,
        includes_raw_image: bool,
        plot_path: Path | None,
        max_tokens: int | None,
        max_output_tokens: int | None,
        device_number: int | None,
        aborted: AbortedCheck,
    ) -> Page:
        """将单页图像识别为结构化 Page（body_layouts、footnotes_layouts 等）。"""
        ...
