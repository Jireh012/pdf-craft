"""基于 OpenAI 兼容接口的 OCR 后端（如硅基流动 DeepSeek-OCR）。"""

import base64
import io
import os
from pathlib import Path

from openai import OpenAI
from PIL.Image import Image

from ..common import AssetHub, remove_surrogates
from ..error import OCRError
from ..metering import AbortedCheck, check_aborted
from .types import DeepSeekOCRSize, Page, PageLayout

# 硅基流动 DeepSeek-OCR 默认模型名，以平台实际为准
DEFAULT_OCR_API_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_OCR_API_MODEL = "DeepSeek/DeepSeek-OCR"


def _image_to_base64_data_url(image: Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return remove_surrogates(text).strip()


class APIPageExtractor:
    """基于第三方 OpenAI 兼容 API 的页面提取器（如硅基流动 DeepSeek-OCR）。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_OCR_API_BASE_URL,
        model: str = DEFAULT_OCR_API_MODEL,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    def download_models(self, revision: str | None) -> None:
        """API 后端无需下载模型， no-op。"""
        pass

    def load_models(self) -> None:
        """API 后端无需加载模型， no-op。"""
        pass

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
        """调用远程 API 将单页图像识别为 Page。当前采用单块降级：整页为一个 body_layout。"""
        check_aborted(aborted)
        raw_image: Image | None = image if includes_raw_image else None

        url = _image_to_base64_data_url(image)
        # 与硅基流动文档一致：带 grounding 的文档转 Markdown 提示
        prompt = "<image>\nConvert the document to markdown."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            client = self._get_client()
            kwargs: dict = {
                "model": self._model,
                "messages": messages,
            }
            if max_output_tokens is not None:
                kwargs["max_tokens"] = max_output_tokens
            response = client.chat.completions.create(**kwargs)
        except Exception as err:
            raise OCRError(
                f"OCR API request failed for page {page_index}: {err!s}",
                page_index=page_index,
                step_index=1,
            ) from err

        check_aborted(aborted)

        content = (
            (response.choices[0].message.content or "").strip()
            if response.choices
            else ""
        )
        input_tokens = 0
        output_tokens = 0
        if getattr(response, "usage", None) is not None:
            input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(response.usage, "completion_tokens", 0) or 0

        # 降级：整页作为单个 body_layout，无 bbox 时用整图范围
        width, height = image.size
        full_det = (0, 0, width, height)
        layout = PageLayout(
            ref="text",
            det=full_det,
            text=_normalize_text(content),
            order=0,
            hash=None,
        )

        return Page(
            index=page_index,
            image=raw_image,
            body_layouts=[layout],
            footnotes_layouts=[],  # API 模式暂不区分脚注
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def create_api_extractor_from_env() -> APIPageExtractor | None:
    """从环境变量创建 API 后端；未配置时返回 None。"""
    api_key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OCR_API_KEY")
    if not api_key:
        return None
    base_url = os.environ.get("OCR_API_BASE_URL", DEFAULT_OCR_API_BASE_URL)
    model = os.environ.get("OCR_API_MODEL", DEFAULT_OCR_API_MODEL)
    return APIPageExtractor(api_key=api_key, base_url=base_url, model=model)
