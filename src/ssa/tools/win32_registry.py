"""Read-only Windows native API capability registry."""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any, Protocol

from ssa.config import Win32Config
from ssa.tools.models import (
    ToolAutonomyContext,
    ToolOutcome,
    Win32ClipboardArguments,
    Win32DirectoryListArguments,
    Win32NoArguments,
    Win32ProcessListArguments,
    Win32WindowListArguments,
)
from ssa.tools.registry import ToolCapability, ToolRegistry

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_CF_UNICODETEXT = 13
_FILE_ATTRIBUTE_DIRECTORY = 0x10
_FILE_ATTRIBUTE_HIDDEN = 0x2
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]


class _WIN32_FIND_DATAW(ctypes.Structure):
    _fields_ = [
        ("dwFileAttributes", wintypes.DWORD),
        ("ftCreationTime", _FILETIME),
        ("ftLastAccessTime", _FILETIME),
        ("ftLastWriteTime", _FILETIME),
        ("nFileSizeHigh", wintypes.DWORD),
        ("nFileSizeLow", wintypes.DWORD),
        ("dwReserved0", wintypes.DWORD),
        ("dwReserved1", wintypes.DWORD),
        ("cFileName", wintypes.WCHAR * 260),
        ("cAlternateFileName", wintypes.WCHAR * 14),
    ]


class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]


class Win32Backend(Protocol):
    def active_window(self) -> dict[str, Any]: ...

    def visible_windows(self, *, limit: int, include_untitled: bool) -> dict[str, Any]: ...

    def processes(self, *, limit: int, include_paths: bool) -> dict[str, Any]: ...

    def drives(self) -> dict[str, Any]: ...

    def list_directory(
        self,
        path: str,
        *,
        pattern: str,
        limit: int,
        include_hidden: bool,
    ) -> dict[str, Any]: ...

    def clipboard_text(self, *, max_chars: int) -> dict[str, Any]: ...


class CtypesWin32Backend:
    """Direct user32/kernel32/psapi calls without PowerShell or pywin32."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Win32 native capabilities require Windows")
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        self._user32.GetForegroundWindow.restype = wintypes.HWND
        self._user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        self._user32.GetWindowTextLengthW.restype = ctypes.c_int
        self._user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self._user32.GetWindowTextW.restype = ctypes.c_int
        self._user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self._user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        self._user32.IsWindowVisible.argtypes = [wintypes.HWND]
        self._user32.IsWindowVisible.restype = wintypes.BOOL
        self._user32.GetLastInputInfo.argtypes = [ctypes.POINTER(_LASTINPUTINFO)]
        self._user32.GetLastInputInfo.restype = wintypes.BOOL
        self._user32.EnumWindows.restype = wintypes.BOOL
        self._user32.OpenClipboard.argtypes = [wintypes.HWND]
        self._user32.OpenClipboard.restype = wintypes.BOOL
        self._user32.CloseClipboard.restype = wintypes.BOOL
        self._user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
        self._user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
        self._user32.GetClipboardData.argtypes = [wintypes.UINT]
        self._user32.GetClipboardData.restype = wintypes.HANDLE
        self._user32.GetClipboardSequenceNumber.restype = wintypes.DWORD
        self._kernel32.GetTickCount.restype = wintypes.DWORD
        self._kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        self._kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self._kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL
        self._kernel32.FindFirstFileW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(_WIN32_FIND_DATAW)]
        self._kernel32.FindFirstFileW.restype = wintypes.HANDLE
        self._kernel32.FindNextFileW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_WIN32_FIND_DATAW)]
        self._kernel32.FindNextFileW.restype = wintypes.BOOL
        self._kernel32.FindClose.argtypes = [wintypes.HANDLE]
        self._kernel32.FindClose.restype = wintypes.BOOL
        self._kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        self._kernel32.GlobalLock.restype = wintypes.LPVOID
        self._kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        self._kernel32.GlobalUnlock.restype = wintypes.BOOL
        self._kernel32.GlobalSize.argtypes = [wintypes.HGLOBAL]
        self._kernel32.GlobalSize.restype = ctypes.c_size_t
        self._kernel32.GetLogicalDriveStringsW.argtypes = [wintypes.DWORD, wintypes.LPWSTR]
        self._kernel32.GetLogicalDriveStringsW.restype = wintypes.DWORD
        self._kernel32.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
        self._kernel32.GetDriveTypeW.restype = wintypes.UINT
        self._kernel32.GetDiskFreeSpaceExW.argtypes = [
            wintypes.LPCWSTR,
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ulonglong),
        ]
        self._kernel32.GetDiskFreeSpaceExW.restype = wintypes.BOOL
        self._kernel32.GetVolumeInformationW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPWSTR,
            wintypes.DWORD,
        ]
        self._kernel32.GetVolumeInformationW.restype = wintypes.BOOL
        self._psapi.EnumProcesses.argtypes = [
            ctypes.POINTER(wintypes.DWORD),
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self._psapi.EnumProcesses.restype = wintypes.BOOL

    def active_window(self) -> dict[str, Any]:
        window = self._user32.GetForegroundWindow()
        process_id = wintypes.DWORD()
        thread_id = (
            int(self._user32.GetWindowThreadProcessId(window, ctypes.byref(process_id)))
            if window
            else 0
        )
        process_path = self._process_path(int(process_id.value))
        return {
            "observed_at_ms": _now_ms(),
            "window_handle": _handle_value(window),
            "title": self._window_title(window) if window else "",
            "process_id": int(process_id.value),
            "thread_id": thread_id,
            "process_name": Path(process_path).name if process_path else "",
            "process_path": process_path,
            "idle_ms": self._idle_ms(),
        }

    def visible_windows(self, *, limit: int, include_untitled: bool) -> dict[str, Any]:
        windows: list[dict[str, Any]] = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        @callback_type
        def collect(window: int, _parameter: int) -> bool:
            if len(windows) >= limit:
                return False
            if not self._user32.IsWindowVisible(window):
                return True
            title = self._window_title(window)
            if not title and not include_untitled:
                return True
            process_id = wintypes.DWORD()
            self._user32.GetWindowThreadProcessId(window, ctypes.byref(process_id))
            process_path = self._process_path(int(process_id.value))
            windows.append(
                {
                    "window_handle": _handle_value(window),
                    "title": title,
                    "process_id": int(process_id.value),
                    "process_name": Path(process_path).name if process_path else "",
                    "process_path": process_path,
                }
            )
            return True

        self._user32.EnumWindows(collect, 0)
        return {
            "observed_at_ms": _now_ms(),
            "count": len(windows),
            "truncated": len(windows) >= limit,
            "windows": windows,
        }

    def processes(self, *, limit: int, include_paths: bool) -> dict[str, Any]:
        capacity = 4_096
        process_ids = (wintypes.DWORD * capacity)()
        bytes_written = wintypes.DWORD()
        ok = self._psapi.EnumProcesses(
            process_ids,
            ctypes.sizeof(process_ids),
            ctypes.byref(bytes_written),
        )
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        count = int(bytes_written.value // ctypes.sizeof(wintypes.DWORD))
        records: list[dict[str, Any]] = []
        for process_id in process_ids[:count]:
            pid = int(process_id)
            if pid == 0:
                continue
            process_path = self._process_path(pid)
            records.append(
                {
                    "process_id": pid,
                    "process_name": Path(process_path).name if process_path else "",
                    **({"process_path": process_path} if include_paths else {}),
                }
            )
        records.sort(key=lambda item: (str(item["process_name"]).casefold(), item["process_id"]))
        return {
            "observed_at_ms": _now_ms(),
            "count": min(len(records), limit),
            "total_enumerated": len(records),
            "truncated": len(records) > limit,
            "processes": records[:limit],
        }

    def drives(self) -> dict[str, Any]:
        buffer = ctypes.create_unicode_buffer(512)
        length = int(self._kernel32.GetLogicalDriveStringsW(len(buffer), buffer))
        if length <= 0:
            raise ctypes.WinError(ctypes.get_last_error())
        roots = [item for item in "".join(buffer[:length]).split("\x00") if item]
        drive_types = {
            0: "unknown",
            1: "no_root",
            2: "removable",
            3: "fixed",
            4: "remote",
            5: "cdrom",
            6: "ramdisk",
        }
        records: list[dict[str, Any]] = []
        for root in roots:
            volume_name = ctypes.create_unicode_buffer(261)
            filesystem = ctypes.create_unicode_buffer(261)
            serial = wintypes.DWORD()
            max_component = wintypes.DWORD()
            flags = wintypes.DWORD()
            self._kernel32.GetVolumeInformationW(
                root,
                volume_name,
                len(volume_name),
                ctypes.byref(serial),
                ctypes.byref(max_component),
                ctypes.byref(flags),
                filesystem,
                len(filesystem),
            )
            free_for_user = ctypes.c_ulonglong()
            total = ctypes.c_ulonglong()
            total_free = ctypes.c_ulonglong()
            space_ok = self._kernel32.GetDiskFreeSpaceExW(
                root,
                ctypes.byref(free_for_user),
                ctypes.byref(total),
                ctypes.byref(total_free),
            )
            records.append(
                {
                    "root": root,
                    "type": drive_types.get(int(self._kernel32.GetDriveTypeW(root)), "unknown"),
                    "volume_name": volume_name.value,
                    "filesystem": filesystem.value,
                    "serial_number": int(serial.value),
                    "total_bytes": int(total.value) if space_ok else None,
                    "free_bytes": int(total_free.value) if space_ok else None,
                    "available_bytes": int(free_for_user.value) if space_ok else None,
                }
            )
        return {"observed_at_ms": _now_ms(), "count": len(records), "drives": records}

    def list_directory(
        self,
        path: str,
        *,
        pattern: str,
        limit: int,
        include_hidden: bool,
    ) -> dict[str, Any]:
        directory = os.path.abspath(os.path.expanduser(path))
        query = os.path.join(directory, pattern)
        data = _WIN32_FIND_DATAW()
        handle = self._kernel32.FindFirstFileW(query, ctypes.byref(data))
        if _handle_value(handle) == _INVALID_HANDLE_VALUE:
            raise OSError(ctypes.get_last_error(), f"FindFirstFileW failed for {query}")
        entries: list[dict[str, Any]] = []
        truncated = False
        try:
            while True:
                name = str(data.cFileName)
                attributes = int(data.dwFileAttributes)
                if name not in {".", ".."} and (
                    include_hidden or not attributes & _FILE_ATTRIBUTE_HIDDEN
                ):
                    if len(entries) >= limit:
                        truncated = True
                        break
                    entries.append(
                        {
                            "name": name,
                            "is_directory": bool(attributes & _FILE_ATTRIBUTE_DIRECTORY),
                            "is_hidden": bool(attributes & _FILE_ATTRIBUTE_HIDDEN),
                            "is_reparse_point": bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT),
                            "size_bytes": (
                                (int(data.nFileSizeHigh) << 32) | int(data.nFileSizeLow)
                            ),
                            "modified_at_ms": _filetime_ms(data.ftLastWriteTime),
                        }
                    )
                if not self._kernel32.FindNextFileW(handle, ctypes.byref(data)):
                    break
        finally:
            self._kernel32.FindClose(handle)
        entries.sort(key=lambda item: (not item["is_directory"], str(item["name"]).casefold()))
        return {
            "observed_at_ms": _now_ms(),
            "path": directory,
            "pattern": pattern,
            "count": len(entries),
            "truncated": truncated,
            "entries": entries,
        }

    def clipboard_text(self, *, max_chars: int) -> dict[str, Any]:
        sequence = int(self._user32.GetClipboardSequenceNumber())
        if not self._user32.IsClipboardFormatAvailable(_CF_UNICODETEXT):
            return {
                "observed_at_ms": _now_ms(),
                "sequence_number": sequence,
                "text_available": False,
                "text": "",
                "truncated": False,
            }
        if not self._user32.OpenClipboard(None):
            raise OSError(ctypes.get_last_error(), "OpenClipboard failed")
        try:
            handle = self._user32.GetClipboardData(_CF_UNICODETEXT)
            if not handle:
                raise OSError(ctypes.get_last_error(), "GetClipboardData failed")
            pointer = self._kernel32.GlobalLock(handle)
            if not pointer:
                raise OSError(ctypes.get_last_error(), "GlobalLock failed")
            try:
                byte_size = int(self._kernel32.GlobalSize(handle))
                available_chars = max(0, byte_size // ctypes.sizeof(ctypes.c_wchar))
                read_chars = min(available_chars, max_chars + 1)
                text = ctypes.wstring_at(pointer, read_chars).split("\x00", 1)[0]
            finally:
                self._kernel32.GlobalUnlock(handle)
        finally:
            self._user32.CloseClipboard()
        truncated = len(text) > max_chars or available_chars > max_chars + 1
        return {
            "observed_at_ms": _now_ms(),
            "sequence_number": sequence,
            "text_available": True,
            "text": text[:max_chars],
            "truncated": truncated,
        }

    def _window_title(self, window: int) -> str:
        length = int(self._user32.GetWindowTextLengthW(window))
        if length <= 0:
            return ""
        buffer = ctypes.create_unicode_buffer(length + 1)
        self._user32.GetWindowTextW(window, buffer, len(buffer))
        return str(buffer.value).strip()

    def _process_path(self, process_id: int) -> str:
        if process_id <= 0:
            return ""
        handle = self._kernel32.OpenProcess(
            _PROCESS_QUERY_LIMITED_INFORMATION,
            False,
            process_id,
        )
        if not handle:
            return ""
        try:
            size = wintypes.DWORD(32_768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if not self._kernel32.QueryFullProcessImageNameW(
                handle,
                0,
                buffer,
                ctypes.byref(size),
            ):
                return ""
            return str(buffer.value)
        finally:
            self._kernel32.CloseHandle(handle)

    def _idle_ms(self) -> int:
        info = _LASTINPUTINFO(cbSize=ctypes.sizeof(_LASTINPUTINFO), dwTime=0)
        if not self._user32.GetLastInputInfo(ctypes.byref(info)):
            return 0
        current = int(self._kernel32.GetTickCount())
        return (current - int(info.dwTime)) & 0xFFFFFFFF


class Win32ToolExecutor:
    """Expose selected native observations through the shared tool registry."""

    def __init__(self, config: Win32Config, *, backend: Win32Backend | None = None) -> None:
        self._config = config
        self._backend = backend or CtypesWin32Backend()

    def register_into(self, registry: ToolRegistry) -> None:
        capabilities = [
            ToolCapability(
                name="win32_active_window",
                description=(
                    "Observe the current foreground window, owning process, and user idle time "
                    "through native user32/kernel32 APIs."
                ),
                arguments_model=Win32NoArguments,
                handler=self.active_window,
            ),
            ToolCapability(
                name="win32_visible_windows",
                description="List visible top-level Windows windows and their owning processes.",
                arguments_model=Win32WindowListArguments,
                handler=self.visible_windows,
            ),
            ToolCapability(
                name="win32_processes",
                description="Enumerate running Windows processes through psapi/kernel32.",
                arguments_model=Win32ProcessListArguments,
                handler=self.processes,
            ),
            ToolCapability(
                name="win32_drives",
                description="Inspect logical drives, volume labels, filesystems, and free space.",
                arguments_model=Win32NoArguments,
                handler=self.drives,
            ),
            ToolCapability(
                name="win32_list_directory",
                description=(
                    "Enumerate files and directories at any Windows path through "
                    "FindFirstFileW/FindNextFileW."
                ),
                arguments_model=Win32DirectoryListArguments,
                handler=self.list_directory,
            ),
        ]
        if self._config.clipboard_enabled:
            capabilities.append(
                ToolCapability(
                    name="win32_clipboard_text",
                    description=(
                        "Read current Unicode clipboard text and its sequence number through "
                        "native clipboard APIs."
                    ),
                    arguments_model=Win32ClipboardArguments,
                    handler=self.clipboard_text,
                )
            )
        for capability in capabilities:
            registry.register(capability)

    async def active_window(
        self, _arguments: dict[str, Any], _context: ToolAutonomyContext
    ) -> ToolOutcome:
        return await self._observe("active_window", self._backend.active_window)

    async def visible_windows(
        self, raw_arguments: dict[str, Any], _context: ToolAutonomyContext
    ) -> ToolOutcome:
        arguments = Win32WindowListArguments.model_validate(raw_arguments)
        limit = min(arguments.limit, self._config.max_windows)
        return await self._observe(
            "visible_windows",
            self._backend.visible_windows,
            limit=limit,
            include_untitled=arguments.include_untitled,
        )

    async def processes(
        self, raw_arguments: dict[str, Any], _context: ToolAutonomyContext
    ) -> ToolOutcome:
        arguments = Win32ProcessListArguments.model_validate(raw_arguments)
        return await self._observe(
            "processes",
            self._backend.processes,
            limit=min(arguments.limit, self._config.max_processes),
            include_paths=arguments.include_paths,
        )

    async def drives(
        self, _arguments: dict[str, Any], _context: ToolAutonomyContext
    ) -> ToolOutcome:
        return await self._observe("drives", self._backend.drives)

    async def list_directory(
        self, raw_arguments: dict[str, Any], _context: ToolAutonomyContext
    ) -> ToolOutcome:
        arguments = Win32DirectoryListArguments.model_validate(raw_arguments)
        return await self._observe(
            "directory",
            self._backend.list_directory,
            arguments.path,
            pattern=arguments.pattern,
            limit=min(arguments.limit, self._config.max_directory_entries),
            include_hidden=arguments.include_hidden,
        )

    async def clipboard_text(
        self, raw_arguments: dict[str, Any], _context: ToolAutonomyContext
    ) -> ToolOutcome:
        arguments = Win32ClipboardArguments.model_validate(raw_arguments)
        return await self._observe(
            "clipboard",
            self._backend.clipboard_text,
            max_chars=min(arguments.max_chars, self._config.max_clipboard_chars),
        )

    async def _observe(
        self,
        kind: str,
        operation: Any,
        *args: Any,
        **kwargs: Any,
    ) -> ToolOutcome:
        payload = await asyncio.to_thread(operation, *args, **kwargs)
        count = payload.get("count") if isinstance(payload, dict) else None
        return ToolOutcome(
            ok=True,
            output=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            metadata={
                "host_observation": True,
                "provider": "windows-win32",
                "observation_kind": kind,
                "observed_at_ms": (
                    payload.get("observed_at_ms") if isinstance(payload, dict) else _now_ms()
                ),
                "record_count": count if isinstance(count, int) else None,
            },
        )


def _now_ms() -> int:
    return int(time.time() * 1_000)


def _handle_value(handle: Any) -> int:
    value = getattr(handle, "value", handle)
    return int(value or 0)


def _filetime_ms(value: _FILETIME) -> int:
    ticks = (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)
    return max(0, (ticks - 116_444_736_000_000_000) // 10_000)


__all__ = ["CtypesWin32Backend", "Win32Backend", "Win32ToolExecutor"]
