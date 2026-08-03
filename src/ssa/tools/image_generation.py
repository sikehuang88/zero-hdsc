"""Frontend-routed image generation capability for the main model tool loop."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import re
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ssa.tools.models import (
    ImageGenerationArguments,
    ImageGenerationRoute,
    ToolAutonomyContext,
    ToolOutcome,
)
from ssa.tools.registry import ToolCapability, ToolRegistry

_MAX_IMAGE_BYTES = 24 * 1024 * 1024
_MAX_JSON_BYTES = 36 * 1024 * 1024
_GENERATED_IMAGE_ROOT = Path(tempfile.gettempdir()) / "zero-generated-images"
_ASSET_ID_RE = re.compile(r"^[0-9a-f]{32}\.(?:png|jpg|webp|gif)$")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+|data:image/[^)\s]+)\)")
_DATA_URL_RE = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.+)$", re.DOTALL)
_EXTENSION_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}


@dataclass(frozen=True)
class GeneratedImageAsset:
    asset_id: str
    url: str
    mime_type: str
    size_bytes: int


@dataclass(frozen=True)
class _ImageReference:
    data: bytes | None = None
    url: str = ""
    mime_type: str = ""
    revised_prompt: str = ""


class ImageGenerationTransport(Protocol):
    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        *,
        timeout_seconds: int,
    ) -> dict[str, Any]: ...

    async def get_bytes(
        self,
        url: str,
        *,
        timeout_seconds: int,
        max_bytes: int,
    ) -> tuple[bytes, str]: ...


class UrllibImageGenerationTransport:
    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        *,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._post_json,
            url,
            payload,
            headers,
            timeout_seconds,
        )

    async def get_bytes(
        self,
        url: str,
        *,
        timeout_seconds: int,
        max_bytes: int,
    ) -> tuple[bytes, str]:
        return await asyncio.to_thread(self._get_bytes, url, timeout_seconds, max_bytes)

    @staticmethod
    def _post_json(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        request = Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                **headers,
                "Content-Type": "application/json",
                "User-Agent": "HDSC/0.5 image-generation",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read(_MAX_JSON_BYTES + 1)
        except HTTPError as exc:
            detail = exc.read(8_192).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"image provider returned HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"image provider connection failed: {exc.reason}") from exc
        if len(raw) > _MAX_JSON_BYTES:
            raise RuntimeError("image provider response exceeded the size limit")
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("image provider returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("image provider returned a non-object payload")
        return decoded

    @staticmethod
    def _get_bytes(url: str, timeout_seconds: int, max_bytes: int) -> tuple[bytes, str]:
        request = Request(url, headers={"User-Agent": "HDSC/0.5 image-generation"})
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = response.read(max_bytes + 1)
                mime_type = response.headers.get_content_type()
        except HTTPError as exc:
            raise RuntimeError(f"generated image download returned HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"generated image download failed: {exc.reason}") from exc
        if len(payload) > max_bytes:
            raise RuntimeError("generated image exceeded the size limit")
        return payload, mime_type


class GeneratedImageStore:
    def __init__(self, root: Path = _GENERATED_IMAGE_ROOT, *, max_files: int = 48) -> None:
        self._root = root
        self._max_files = max_files

    def save(self, payload: bytes, mime_type: str = "") -> GeneratedImageAsset:
        if not payload:
            raise ValueError("generated image payload is empty")
        if len(payload) > _MAX_IMAGE_BYTES:
            raise ValueError("generated image exceeded the size limit")
        normalized_mime, extension = _detect_image_type(payload, mime_type)
        self._root.mkdir(parents=True, exist_ok=True)
        asset_id = f"{uuid.uuid4().hex}.{extension}"
        target = self._root / asset_id
        temporary = self._root / f".{asset_id}.tmp"
        temporary.write_bytes(payload)
        temporary.replace(target)
        self._prune()
        return GeneratedImageAsset(
            asset_id=asset_id,
            url=f"/api/generated-images/{asset_id}",
            mime_type=normalized_mime,
            size_bytes=len(payload),
        )

    def _prune(self) -> None:
        cutoff = time.time() - 48 * 60 * 60
        try:
            files = sorted(
                (item for item in self._root.iterdir() if _ASSET_ID_RE.fullmatch(item.name)),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return
        for index, item in enumerate(files):
            try:
                if index >= self._max_files or item.stat().st_mtime < cutoff:
                    item.unlink(missing_ok=True)
            except OSError:
                continue


class ImageGenerationExecutor:
    """Generate one image through the active frontend route and cache it locally."""

    def __init__(
        self,
        *,
        transport: ImageGenerationTransport | None = None,
        store: GeneratedImageStore | None = None,
    ) -> None:
        self._transport = transport or UrllibImageGenerationTransport()
        self._store = store or GeneratedImageStore()

    def register_into(self, registry: ToolRegistry) -> None:
        registry.register(
            ToolCapability(
                name="generate_image",
                description=(
                    "Generate a new image when the user asks to draw, create, render, design, "
                    "or visualize a picture. Write a complete visual prompt, choose landscape, "
                    "portrait, or square size from the schema, and call this tool once. The "
                    "result is displayed automatically in Zero's main stage."
                ),
                arguments_model=ImageGenerationArguments,
                handler=self.generate_image,
            )
        )

    async def generate_image(
        self,
        raw_arguments: dict[str, Any],
        context: ToolAutonomyContext,
    ) -> ToolOutcome:
        arguments = ImageGenerationArguments.model_validate(raw_arguments)
        route = context.image_generation_route
        if route is None:
            return ToolOutcome(ok=False, error="image generation provider is not configured")
        try:
            url, payload, headers = _build_request(route, arguments)
            response = await self._transport.post_json(
                url,
                payload,
                headers,
                timeout_seconds=180,
            )
            reference = _extract_image_reference(response)
            if reference is None:
                raise RuntimeError("image provider response did not contain an image")
            if reference.data is not None:
                image_bytes = reference.data
                mime_type = reference.mime_type
            else:
                image_bytes, downloaded_mime = await self._transport.get_bytes(
                    reference.url,
                    timeout_seconds=60,
                    max_bytes=_MAX_IMAGE_BYTES,
                )
                mime_type = reference.mime_type or downloaded_mime
            asset = self._store.save(image_bytes, mime_type)
        except (RuntimeError, ValueError, binascii.Error) as exc:
            return ToolOutcome(ok=False, error=str(exc))
        return ToolOutcome(
            ok=True,
            output=(
                "Image generated and displayed in Zero's main stage. "
                f"Prompt: {reference.revised_prompt or arguments.prompt}"
            ),
            metadata={
                "image_generation": True,
                "provider": route.protocol,
                "model": route.model,
                "asset_url": asset.url,
                "asset_id": asset.asset_id,
                "mime_type": asset.mime_type,
                "size_bytes": asset.size_bytes,
                "size": arguments.size,
                "prompt": reference.revised_prompt or arguments.prompt,
            },
        )


def resolve_generated_image(asset_id: str) -> tuple[Path, str] | None:
    if _ASSET_ID_RE.fullmatch(asset_id) is None:
        return None
    path = (_GENERATED_IMAGE_ROOT / asset_id).resolve(strict=False)
    root = _GENERATED_IMAGE_ROOT.resolve(strict=False)
    if path.parent != root or not path.is_file():
        return None
    extension = path.suffix.lstrip(".")
    return path, _EXTENSION_MIME[extension]


def _build_request(
    route: ImageGenerationRoute,
    arguments: ImageGenerationArguments,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    api_key = route.api_key.get_secret_value().strip()
    if not api_key:
        raise ValueError("image generation API key is empty")
    parsed = urlparse(route.base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("image generation base URL must use HTTP or HTTPS")
    if route.protocol == "openai-responses":
        image_tool: dict[str, Any] = {
            "type": "image_generation",
            "size": arguments.size,
            "quality": arguments.quality,
        }
        return (
            _endpoint(route.base_url, "responses"),
            {
                "model": route.model,
                "input": arguments.prompt,
                "tools": [image_tool],
                "tool_choice": {"type": "image_generation"},
            },
            {"Authorization": f"Bearer {api_key}"},
        )
    if route.protocol == "anthropic-messages":
        return (
            _endpoint(route.base_url, "v1/messages"),
            {
                "model": route.model,
                "max_tokens": 1_024,
                "messages": [{"role": "user", "content": arguments.prompt}],
                "metadata": {
                    "image_size": arguments.size,
                    "image_quality": arguments.quality,
                },
            },
            {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
    return (
        _endpoint(route.base_url, "chat/completions"),
        {
            "model": route.model,
            "messages": [{"role": "user", "content": arguments.prompt}],
            "modalities": ["text", "image"],
            "image_config": {
                "size": arguments.size,
                "quality": arguments.quality,
            },
            "stream": False,
        },
        {"Authorization": f"Bearer {api_key}"},
    )


def _endpoint(base_url: str, suffix: str) -> str:
    base = base_url.strip().rstrip("/")
    normalized_suffix = suffix.strip("/")
    if base.lower().endswith(f"/{normalized_suffix.lower()}"):
        return base
    if normalized_suffix.startswith("v1/") and base.lower().endswith("/v1"):
        normalized_suffix = normalized_suffix[3:]
    return f"{base}/{normalized_suffix}"


def _extract_image_reference(payload: dict[str, Any]) -> _ImageReference | None:
    revised_prompt = _find_text(payload, {"revised_prompt", "prompt"})
    return _walk_for_image(payload, revised_prompt=revised_prompt)


def _walk_for_image(value: Any, *, revised_prompt: str = "") -> _ImageReference | None:
    if isinstance(value, dict):
        item_type = str(value.get("type") or "").lower()
        if item_type == "image_generation_call" and isinstance(value.get("result"), str):
            return _decode_image_string(value["result"], revised_prompt=revised_prompt)
        if isinstance(value.get("b64_json"), str):
            return _decode_image_string(value["b64_json"], revised_prompt=revised_prompt)
        source = value.get("source")
        if item_type in {"image", "output_image"} and isinstance(source, dict):
            source_data = source.get("data")
            if isinstance(source_data, str):
                return _decode_image_string(
                    source_data,
                    mime_type=str(source.get("media_type") or ""),
                    revised_prompt=revised_prompt,
                )
        image_url = value.get("image_url")
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        if isinstance(image_url, str):
            return _decode_image_string(image_url, revised_prompt=revised_prompt)
        if item_type in {"image", "output_image"} and isinstance(value.get("url"), str):
            return _decode_image_string(value["url"], revised_prompt=revised_prompt)
        for child in value.values():
            found = _walk_for_image(child, revised_prompt=revised_prompt)
            if found is not None:
                return found
        return None
    if isinstance(value, list):
        for child in value:
            found = _walk_for_image(child, revised_prompt=revised_prompt)
            if found is not None:
                return found
        return None
    if isinstance(value, str):
        markdown = _MARKDOWN_IMAGE_RE.search(value)
        if markdown is not None:
            return _decode_image_string(markdown.group(1), revised_prompt=revised_prompt)
    return None


def _decode_image_string(
    value: str,
    *,
    mime_type: str = "",
    revised_prompt: str = "",
) -> _ImageReference:
    candidate = value.strip()
    if candidate.startswith(("http://", "https://")):
        return _ImageReference(url=candidate, mime_type=mime_type, revised_prompt=revised_prompt)
    data_url = _DATA_URL_RE.match(candidate)
    if data_url is not None:
        return _ImageReference(
            data=base64.b64decode(data_url.group(2), validate=True),
            mime_type=data_url.group(1),
            revised_prompt=revised_prompt,
        )
    return _ImageReference(
        data=base64.b64decode(candidate, validate=True),
        mime_type=mime_type,
        revised_prompt=revised_prompt,
    )


def _find_text(value: Any, keys: set[str]) -> str:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in keys and isinstance(child, str) and child.strip():
                return child.strip()
        for child in value.values():
            found = _find_text(child, keys)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_text(child, keys)
            if found:
                return found
    return ""


def _detect_image_type(payload: bytes, declared_mime: str) -> tuple[str, str]:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", "png"
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", "jpg"
    if payload.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif", "gif"
    if len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "image/webp", "webp"
    declared = declared_mime.lower().split(";", 1)[0].strip()
    if declared in _EXTENSION_MIME.values():
        raise ValueError(f"generated payload did not match declared image type {declared}")
    raise ValueError("generated payload is not a supported PNG, JPEG, WebP, or GIF image")


__all__ = [
    "GeneratedImageAsset",
    "GeneratedImageStore",
    "ImageGenerationExecutor",
    "ImageGenerationTransport",
    "UrllibImageGenerationTransport",
    "resolve_generated_image",
]
