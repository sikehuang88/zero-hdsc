"""Windows media-session and Soda Music desktop controls."""

from __future__ import annotations

import asyncio
import ctypes
import gzip
import json
import os
import re
import subprocess
import tempfile
from ctypes import wintypes
from pathlib import Path
from typing import Any, ClassVar, cast

from ssa.tools.models import (
    SodaMusicControlArguments,
    SodaMusicNoArguments,
    SodaMusicSearchPlayArguments,
    ToolAutonomyContext,
    ToolOutcome,
)
from ssa.tools.registry import ToolCapability, ToolRegistry

_SODA_SESSION_NAME = "汽水音乐"
_DEFAULT_SODA_EXE = Path(r"E:\汽水\Soda Music\3.5.1\SodaMusic.exe")
_SODA_ENABLE_ENV = "HDSC_SODA_MUSIC_ENABLED"
_TRUTHY = {"1", "true", "yes", "on", "enabled"}
_LEGACY_NOTE = (
    "Legacy fallback: Mineradio (mineradio_*) is the primary music backend. "
    "This tool drives the separately installed Soda Music desktop client and is only "
    "registered when HDSC_SODA_MUSIC_ENABLED is set. "
)


def soda_music_enabled() -> bool:
    """Opt-in switch for the legacy Soda Music path.

    Mineradio is the primary backend and covers the same providers over HTTP.
    This path injects into a separately installed desktop client whose executable
    path is version-pinned, so it stays off unless explicitly asked for.
    """
    return os.environ.get(_SODA_ENABLE_ENV, "").strip().lower() in _TRUTHY
_MEDIA_KEYS = {
    "play": 0xB3,
    "pause": 0xB3,
    "toggle": 0xB3,
    "next": 0xB0,
    "previous": 0xB1,
    "volume_up": 0xAF,
    "volume_down": 0xAE,
    "mute": 0xAD,
}
_QUERY_SPLIT_RE = re.compile(r"[\s,\uFF0C\u3001/|\u00B7:\uFF1A\-\u2014]+|(?<=\S)的(?=\S)")
_QUERY_NOISE_RE = re.compile(r"[\s\W_]+", re.UNICODE)
_SODA_BRIDGE_PORT = 19091
_SODA_BRIDGE_LOCK = asyncio.Lock()
_SODA_BRIDGE_MARKER = "HDSC_SODA_IPC_BRIDGE_V3"
_SODA_BRIDGE_SCRIPT = r"""
/* HDSC_SODA_IPC_BRIDGE_V3 */
(() => {
  const listeners = new Set()
  const pending = new Map()
  const originalReceive = window.transportPort.receiveTransport
  window.transportPort.receiveTransport = (callback) => originalReceive((data) => {
    callback(data)
    listeners.forEach((listener) => listener(data))
  })
  function invoke(service, method, args) {
    const requestId = `hdsc-${Date.now()}-${Math.random().toString(36).slice(2)}`
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        pending.delete(requestId)
        reject(new Error(`Soda IPC timeout: ${service}.${method}`))
      }, 15000)
      pending.set(requestId, { resolve, reject, timer })
      window.transportPort.sendTransport({
        type: 'method.invoke', fromWorkerId: 'rendererMain',
        toServiceId: service, methodName: method, requestId,
        arguments: args, callbacks: {},
      })
    })
  }
  listeners.add((message) => {
    if (message?.type !== 'method.return') return
    const item = pending.get(message.requestId)
    if (!item) return
    pending.delete(message.requestId)
    clearTimeout(item.timer)
    if (message.return?.type === 'error') item.reject(new Error(message.return.error))
    else item.resolve(message.return?.result)
  })
  const normalize = (value) => String(value || '').toLocaleLowerCase().replace(/[\s\W_]+/gu, '')
  const trackScore = (query, track) => {
    const q = normalize(query)
    const title = normalize(track?.name)
    const artists = normalize((track?.artists || []).map((artist) => artist.name).join(' '))
    return (title.includes(q) ? 4 : 0) + (artists.includes(q) ? 3 : 0)
      + (q && title ? (q.split('的').every((part) => title.includes(normalize(part))) ? 2 : 0) : 0)
  }
  function toPlayable(track) {
    return {
      key: `track-${track.id}`, type: 'track', track, video: undefined,
      id: track.id, name: track.name, sub_name: track.sub_name,
      media_type: track.media_type || 'track', cover_url: track.album?.url_cover,
      artists: track.artists, album: track.album, duration: track.duration,
      vid: track.vid, colors: track.colors, status: track.status,
      stats: track.stats, digital_track_info: track.digital_track_info,
      preview: track.preview, limited_free_info: track.limited_free_info,
      bit_rates: track.bit_rates, chorus: track.chorus,
      sharable_platforms: track.sharable_platforms, vocal: track.vocal,
      song_maker_team: track.song_maker_team, label_info: track.label_info,
      state: track.state ? { is_collected: track.state.is_collected } : undefined,
    }
  }
  async function searchPlay(query) {
    const search = await invoke('request', 'request', ['Search', {
      q: query, search_id: crypto.randomUUID(), search_method: 'input',
      cursor: '0', search_type: 'track', debug_params: '',
      from_search_id: '', search_scene: '',
    }])
    const tracks = []
    for (const group of (search?.result_groups || [])) {
      for (const item of (group.data || [])) {
        if (item?.entity?.track) tracks.push(item.entity.track)
      }
    }
    if (!tracks.length) throw new Error('Soda Search returned no tracks')
    tracks.sort((a, b) => trackScore(query, b) - trackScore(query, a))
    const track = tracks[0]
    await invoke('queue', 'play', [{
      collectionParams: {
        type: 'related', playable: toPlayable(track),
        extraParams: { searchQuery: query, searchItemClickTime: Math.floor(Date.now() / 1000) },
      }, autoPlay: true,
    }])
    return { query, track, candidate_count: tracks.length }
  }
  async function loginStatus() {
    const value = await invoke('user', 'isLogged', [])
    const payload = value && typeof value === 'object' ? value : {}
    const candidate = typeof value === 'boolean'
      ? value
      : (payload.logged_in ?? payload.is_logged ?? payload.isLogged ?? payload.is_login)
    return { logged_in: typeof candidate === 'boolean' ? candidate : null, raw: value }
  }
  window.addEventListener('message', async (event) => {
    if (event.data?.type !== 'HDSC_SODA_REQUEST') return
    try {
      const request = JSON.parse(String(event.data.raw || ''))
      try {
        const result = request.action === 'search_play'
          ? await searchPlay(String(request.query || ''))
          : request.action === 'login_status'
            ? await loginStatus()
            : (() => { throw new Error('unsupported Soda bridge action') })()
        window.postMessage({ type: 'HDSC_SODA_RESPONSE', payload: { ok: true, result } }, '*')
      } catch (error) {
        window.postMessage({
          type: 'HDSC_SODA_RESPONSE',
          payload: { ok: false, error: String(error?.message || error) },
        }, '*')
      }
    } catch {}
  })
  function mainWorldSocketBridge() {
    if (window.__hdscSodaSocketBridge) return
    window.__hdscSodaSocketBridge = true
    let socket
    window.addEventListener('message', (event) => {
      if (event.data?.type !== 'HDSC_SODA_RESPONSE') return
      if (socket?.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(event.data.payload))
      }
    })
    function connect() {
      socket = new WebSocket('ws://127.0.0.1:19091')
      socket.onmessage = (event) => window.postMessage({
        type: 'HDSC_SODA_REQUEST', raw: String(event.data || ''),
      }, '*')
      socket.onclose = () => setTimeout(connect, 500)
      socket.onerror = () => socket.close()
    }
    connect()
  }
  function installMainWorldBridge() {
    const script = document.createElement('script')
    script.textContent = `(${mainWorldSocketBridge.toString()})()`
    ;(document.head || document.documentElement).appendChild(script)
    script.remove()
  }
  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', installMainWorldBridge, { once: true })
  } else {
    installMainWorldBridge()
  }
})()
"""

_MEDIA_SESSION_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
[void][Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager,Windows.Media.Control,ContentType=WindowsRuntime]
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
  $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1
})[0]
function Await($operation, [Type]$resultType) {
  $task = $asTask.MakeGenericMethod($resultType).Invoke($null, @($operation))
  $task.Wait()
  $task.Result
}
$manager = Await ([Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager]::RequestAsync()) ([Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager])
$sessions = @($manager.GetSessions())
$session = $sessions | Where-Object { $_.SourceAppUserModelId -eq '汽水音乐' } | Select-Object -First 1
if ($null -eq $session) { '{"found":false}'; exit 0 }
$media = Await ($session.TryGetMediaPropertiesAsync()) ([Windows.Media.Control.GlobalSystemMediaTransportControlsSessionMediaProperties])
$timeline = $session.GetTimelineProperties()
$playback = $session.GetPlaybackInfo()
[pscustomobject]@{
  found = $true
  source = $session.SourceAppUserModelId
  title = $media.Title
  artist = $media.Artist
  album = $media.AlbumTitle
  status = $playback.PlaybackStatus.ToString()
  position_ms = [int64]$timeline.Position.TotalMilliseconds
  duration_ms = [int64]$timeline.EndTime.TotalMilliseconds
} | ConvertTo-Json -Compress
"""


class _KEYBDINPUT(ctypes.Structure):
    _fields_: ClassVar[Any] = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _MOUSEINPUT(ctypes.Structure):
    _fields_: ClassVar[Any] = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_: ClassVar[Any] = [
        ("ki", _KEYBDINPUT),
        ("mi", _MOUSEINPUT),
    ]


class _INPUT(ctypes.Structure):
    _anonymous_: ClassVar[tuple[str, ...]] = ("union",)
    _fields_: ClassVar[Any] = [
        ("type", wintypes.DWORD),
        ("union", _INPUT_UNION),
    ]


class SodaMusicExecutor:
    """Control the user's existing Soda Music session without replacing its player."""

    def __init__(self, *, executable_path: str | None = None) -> None:
        configured = executable_path or os.environ.get("HDSC_SODA_MUSIC_EXE", "")
        self._executable = Path(configured) if configured else _DEFAULT_SODA_EXE
        self._bridge_activation_attempted = False

    def register_into(self, registry: ToolRegistry) -> None:
        if not soda_music_enabled():
            return
        registry.register(
            ToolCapability(
                name="soda_music_now_playing",
                description=(
                    _LEGACY_NOTE + "Read the real current track, artist, playback state, "
                    "and timeline from the Soda Music Windows media session."
                ),
                arguments_model=SodaMusicNoArguments,
                handler=self.now_playing,
            )
        )
        registry.register(
            ToolCapability(
                name="soda_music_login_status",
                description=(
                    _LEGACY_NOTE + "Read whether the Soda Music desktop account is currently "
                    "logged in through its existing renderer session."
                ),
                arguments_model=SodaMusicNoArguments,
                handler=self.login_status_tool,
            )
        )
        registry.register(
            ToolCapability(
                name="soda_music_control",
                description=(
                    _LEGACY_NOTE + "Control Soda Music playback with play, pause, toggle, "
                    "next, previous, volume_up, volume_down, or mute."
                ),
                arguments_model=SodaMusicControlArguments,
                handler=self.control,
            )
        )
        registry.register(
            ToolCapability(
                name="soda_music_search_play",
                description=(
                    _LEGACY_NOTE + "Open Soda Music, search for a requested song or artist, "
                    "verify candidates against the Windows media session, and return the "
                    "actual selected version."
                ),
                arguments_model=SodaMusicSearchPlayArguments,
                handler=self.search_play,
            )
        )

    async def now_playing(
        self,
        _raw_arguments: dict[str, Any],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        state = await self._read_session()
        if not state.get("found"):
            return ToolOutcome(ok=False, error="Soda Music media session was not found")
        return self._state_outcome(state, action="observe")

    async def login_status_tool(
        self,
        _raw_arguments: dict[str, Any],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        state = await self.login_status()
        if state.get("status") == "unavailable":
            return ToolOutcome(ok=False, error=str(state.get("error") or "Soda Music login status unavailable"))
        label = "logged in" if state.get("logged_in") else "not logged in"
        return ToolOutcome(ok=True, output=f"Soda Music is {label}", metadata=state)

    async def login_status(self) -> dict[str, Any]:
        """Return the current Soda account state without changing playback."""
        window = await asyncio.to_thread(self._find_window)
        if not window:
            return {
                "status": "unavailable",
                "logged_in": None,
                "error": "Soda Music window was not found",
            }
        bridge = await self._bridge_call(
            window,
            "login_status",
            reload=False,
            timeout_seconds=1.5,
            lock_timeout_seconds=0.25,
        )
        if not bridge.get("ok") and not self._bridge_activation_attempted:
            # The preload may have been installed while Soda was already open.
            # Activate it once, then reuse the reconnecting bridge on later polls.
            self._bridge_activation_attempted = True
            bridge = await self._bridge_call(
                window,
                "login_status",
                reload=True,
                lock_timeout_seconds=0.25,
            )
        if not bridge.get("ok"):
            try:
                cached = await asyncio.to_thread(self._read_login_cache_status)
            except (OSError, ValueError) as exc:
                return {
                    "status": "unavailable",
                    "logged_in": None,
                    "error": str(bridge.get("error") or exc),
                }
            return {
                "status": "logged_in" if cached["logged_in"] else "logged_out",
                **cached,
                "source": "user_info_cache",
                "bridge_connected": False,
            }
        result = bridge.get("result") or {}
        logged_in = result.get("logged_in") if isinstance(result, dict) else None
        if not isinstance(logged_in, bool):
            return {
                "status": "unavailable",
                "logged_in": None,
                "error": "Soda Music returned an invalid login state",
            }
        return {
            "status": "logged_in" if logged_in else "logged_out",
            "logged_in": logged_in,
            "source": "renderer_ipc",
            "bridge_connected": True,
        }

    @staticmethod
    def _read_login_cache_status() -> dict[str, Any]:
        """Read Soda's live user-info cache without exposing account identifiers."""
        app_data = os.environ.get("APPDATA")
        if not app_data:
            raise OSError("APPDATA is not configured")
        cache_path = Path(app_data) / "SodaMusic" / "LunaStorage" / "Config"
        raw = cache_path.read_bytes()
        if not raw.startswith(b"LUNA\x1f\x8b"):
            raise ValueError("Soda user-info cache has an unknown format")
        payload = json.loads(gzip.decompress(raw[4:]))
        user_cache = payload.get("userInfoStateCache")
        user_info = user_cache.get("my_info") if isinstance(user_cache, dict) else None
        if not isinstance(user_info, dict):
            return {"logged_in": False}
        account_id = str(user_info.get("id") or "").strip()
        display_name = str(
            user_info.get("public_name") or user_info.get("nickname") or ""
        ).strip()
        result: dict[str, Any] = {"logged_in": bool(account_id)}
        if display_name:
            result["display_name"] = display_name
        return result

    async def control(
        self,
        raw_arguments: dict[str, Any],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        arguments = SodaMusicControlArguments.model_validate(raw_arguments)
        before = await self._read_session()
        action = arguments.action
        if action == "play" and before.get("status") == "Playing":
            return self._state_outcome(before, action=action)
        if action == "pause" and before.get("status") == "Paused":
            return self._state_outcome(before, action=action)
        await asyncio.to_thread(self._send_virtual_key, _MEDIA_KEYS[action])
        await asyncio.sleep(0.35)
        after = await self._read_session()
        if not after.get("found"):
            return ToolOutcome(ok=True, output=f"Soda Music control sent: {action}")
        return self._state_outcome(after, action=action)

    async def search_play(
        self,
        raw_arguments: dict[str, Any],
        _context: ToolAutonomyContext,
    ) -> ToolOutcome:
        arguments = SodaMusicSearchPlayArguments.model_validate(raw_arguments)
        window = await asyncio.to_thread(self._find_or_start_window)
        if not window:
            return ToolOutcome(ok=False, error="Soda Music window was not found after launch")
        bridge = await self._bridge_call(window, "search_play", arguments.query)
        if not bridge.get("ok"):
            return ToolOutcome(ok=False, error=str(bridge.get("error") or "Soda IPC bridge failed"))
        await asyncio.sleep(0.8)
        state = await self._read_session()
        if not state.get("found"):
            return ToolOutcome(
                ok=False,
                error="Soda Music search completed but no playback session appeared",
                metadata={"query": arguments.query, "bridge": "renderer_ipc"},
            )
        outcome = self._state_outcome(state, action="search_play")
        outcome.metadata.update(
            {"query": arguments.query, "bridge": "renderer_ipc", **bridge.get("result", {})}
        )
        return outcome

    async def _bridge_call(
        self,
        window: int,
        action: str,
        query: str | None = None,
        reload: bool = True,
        timeout_seconds: float = 12,
        lock_timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        preload = Path(tempfile.gettempdir()) / "sodamusic-preloads" / "main.js"
        try:
            source = preload.read_text(encoding="utf-8")
        except OSError as exc:
            return {"ok": False, "error": f"Soda preload not found: {exc}"}
        base_source = source.split("\n/* HDSC_SODA_IPC_BRIDGE_", 1)[0].rstrip() + "\n"
        try:
            preload.write_text(base_source + _SODA_BRIDGE_SCRIPT, encoding="utf-8")
        except OSError as exc:
            return {"ok": False, "error": f"Soda preload bridge write failed: {exc}"}

        response: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()

        async def bridge_handler(websocket: Any) -> None:
            if response.done():
                await websocket.close()
                return
            try:
                await websocket.send(
                    json.dumps(
                        {"action": action, **({"query": query} if query is not None else {})},
                        ensure_ascii=False,
                    )
                )
                raw = await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
                if not response.done():
                    response.set_result(cast(dict[str, Any], json.loads(raw)))
            except Exception as exc:
                if not response.done():
                    response.set_result({"ok": False, "error": str(exc)})
            finally:
                await websocket.close()

        try:
            if lock_timeout_seconds is None:
                await _SODA_BRIDGE_LOCK.acquire()
            else:
                await asyncio.wait_for(
                    _SODA_BRIDGE_LOCK.acquire(),
                    timeout=lock_timeout_seconds,
                )
        except TimeoutError:
            return {"ok": False, "error": "Soda renderer IPC bridge is busy"}
        try:
            from websockets.asyncio.server import serve

            try:
                async with serve(
                    bridge_handler,
                    "127.0.0.1",
                    _SODA_BRIDGE_PORT,
                    close_timeout=1,
                ):
                    if reload:
                        await asyncio.to_thread(self._reload_window, window)
                    return await asyncio.wait_for(response, timeout=timeout_seconds + 2)
            except Exception as exc:
                detail = str(exc) or "renderer did not reconnect"
                return {
                    "ok": False,
                    "error": (
                        "Soda renderer IPC bridge is not active; fully restart Soda Music once "
                        f"and retry ({detail})"
                    ),
                }
        finally:
            _SODA_BRIDGE_LOCK.release()

    @staticmethod
    def _reload_window(window: int) -> None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.ShowWindow(window, 9)
        user32.SetForegroundWindow(window)
        user32.SetFocus(window)
        SodaMusicExecutor._send_virtual_key(0x52, modifiers=(0x11,))
        SodaMusicExecutor._send_virtual_key(0x74)

    async def _read_session(self) -> dict[str, Any]:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = await asyncio.create_subprocess_exec(
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            _MEDIA_SESSION_SCRIPT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=creationflags,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
        if process.returncode != 0:
            error = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(error or "Windows media-session query failed")
        payload = stdout.decode("utf-8-sig", errors="replace").strip()
        return cast(dict[str, Any], json.loads(payload or '{"found":false}'))

    @staticmethod
    def _state_outcome(state: dict[str, Any], *, action: str) -> ToolOutcome:
        title = str(state.get("title") or "Unknown track")
        artist = str(state.get("artist") or "Unknown artist")
        status = str(state.get("status") or "Unknown")
        return ToolOutcome(
            ok=True,
            output=f"{title} - {artist} ({status})",
            metadata={
                "host_observation": True,
                "provider": "windows_gsmtc",
                "observation_kind": "soda_music_session",
                "action": action,
                **state,
            },
        )

    def _find_or_start_window(self) -> int:
        window = self._find_window()
        if window:
            return window
        if not self._executable.is_file():
            return 0
        subprocess.Popen(
            [str(self._executable)],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        for _ in range(30):
            import time

            time.sleep(0.2)
            window = self._find_window()
            if window:
                return window
        return 0

    @staticmethod
    def _find_window() -> int:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        matches: list[int] = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def callback(window: int, _parameter: int) -> bool:
            length = user32.GetWindowTextLengthW(window)
            if length:
                title = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(window, title, len(title))
                if title.value == _SODA_SESSION_NAME and user32.IsWindowVisible(window):
                    matches.append(int(window))
            return True

        user32.EnumWindows(callback_type(callback), 0)
        return matches[0] if matches else 0

    @classmethod
    def _prepare_song_search(cls, window: int, query: str) -> tuple[int, int]:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.ShowWindow(window, 9)
        user32.SetForegroundWindow(window)
        rect = wintypes.RECT()
        if not user32.GetClientRect(window, ctypes.byref(rect)):
            raise ctypes.WinError(ctypes.get_last_error())
        width = max(1, rect.right - rect.left)
        height = max(1, rect.bottom - rect.top)
        cls._click_client(window, int(width * 0.49), max(24, int(height * 0.025)))
        cls._send_virtual_key(0x41, modifiers=(0x11,))
        cls._send_text(query)
        cls._send_virtual_key(0x0D)
        import time

        time.sleep(1.8)
        cls._send_virtual_key(0x1B)
        cls._click_client(window, int(width * 0.31), int(height * 0.12))
        time.sleep(2.2)
        return width, height

    @classmethod
    def _click_song_row(
        cls,
        window: int,
        width: int,
        height: int,
        row_index: int,
    ) -> None:
        row_y = int(height * (0.22 + row_index * 0.047))
        cls._click_client(window, int(width * 0.40), row_y, count=2)

    @staticmethod
    def _click_client(window: int, x: int, y: int, *, count: int = 1) -> None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        point = wintypes.POINT(x, y)
        if not user32.ClientToScreen(window, ctypes.byref(point)):
            raise ctypes.WinError(ctypes.get_last_error())
        user32.SetCursorPos(point.x, point.y)
        for _ in range(count):
            inputs = (_INPUT * 2)(
                _INPUT(type=0, mi=_MOUSEINPUT(dwFlags=0x0002)),
                _INPUT(type=0, mi=_MOUSEINPUT(dwFlags=0x0004)),
            )
            user32.SendInput(2, inputs, ctypes.sizeof(_INPUT))

    @staticmethod
    def _send_virtual_key(key: int, *, modifiers: tuple[int, ...] = ()) -> None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        inputs: list[_INPUT] = []
        for modifier in modifiers:
            inputs.append(_INPUT(type=1, ki=_KEYBDINPUT(wVk=modifier)))
        inputs.append(_INPUT(type=1, ki=_KEYBDINPUT(wVk=key)))
        inputs.append(_INPUT(type=1, ki=_KEYBDINPUT(wVk=key, dwFlags=0x0002)))
        for modifier in reversed(modifiers):
            inputs.append(_INPUT(type=1, ki=_KEYBDINPUT(wVk=modifier, dwFlags=0x0002)))
        payload = (_INPUT * len(inputs))(*inputs)
        user32.SendInput(len(payload), payload, ctypes.sizeof(_INPUT))

    @staticmethod
    def _send_text(text: str) -> None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        inputs: list[_INPUT] = []
        for character in text:
            code = ord(character)
            inputs.append(_INPUT(type=1, ki=_KEYBDINPUT(wScan=code, dwFlags=0x0004)))
            inputs.append(_INPUT(type=1, ki=_KEYBDINPUT(wScan=code, dwFlags=0x0006)))
        payload = (_INPUT * len(inputs))(*inputs)
        user32.SendInput(len(payload), payload, ctypes.sizeof(_INPUT))


def _track_identity(state: dict[str, Any]) -> tuple[str, str]:
    return (
        _normalize_match_text(str(state.get("title") or "")),
        _normalize_match_text(str(state.get("artist") or "")),
    )


def _query_match_score(query: str, state: dict[str, Any]) -> int:
    """Return 2 for all query parts, 1 for a useful partial match, otherwise 0."""
    if not state.get("found"):
        return 0
    title = _normalize_match_text(str(state.get("title") or ""))
    artist = _normalize_match_text(str(state.get("artist") or ""))
    explicit = re.fullmatch(r"\s*(.+?)\s*的\s*(.+?)\s*", query)
    if explicit is not None:
        requested_artist = _normalize_match_text(explicit.group(1))
        requested_title = _normalize_match_text(explicit.group(2))
        title_matches = bool(requested_title and requested_title in title)
        artist_matches = bool(requested_artist and requested_artist in artist)
        return 2 if title_matches and artist_matches else (1 if title_matches else 0)
    combined = _normalize_match_text(
        f"{state.get('title') or ''} {state.get('artist') or ''} {state.get('album') or ''}"
    )
    parts = [
        _normalize_match_text(part)
        for part in _QUERY_SPLIT_RE.split(query.strip())
        if len(_normalize_match_text(part)) >= 2
    ]
    if not parts:
        normalized_query = _normalize_match_text(query)
        parts = [normalized_query] if normalized_query else []
    matched = sum(part in combined for part in parts)
    if parts and matched == len(parts):
        return 2
    return 1 if matched else 0


def _normalize_match_text(value: str) -> str:
    return _QUERY_NOISE_RE.sub("", value).casefold()


__all__ = ["SodaMusicExecutor"]
