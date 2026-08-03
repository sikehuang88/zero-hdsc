"""Binary event framing used by Doubao Seed TTS bidirectional WebSockets."""

from __future__ import annotations

import gzip
import io
import json
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class MsgType(IntEnum):
    INVALID = 0
    FULL_CLIENT_REQUEST = 0b0001
    AUDIO_ONLY_CLIENT = 0b0010
    FULL_SERVER_RESPONSE = 0b1001
    AUDIO_ONLY_SERVER = 0b1011
    FRONTEND_RESULT_SERVER = 0b1100
    ERROR = 0b1111


class Flags(IntEnum):
    NO_SEQUENCE = 0b0000
    POSITIVE_SEQUENCE = 0b0001
    LAST_NO_SEQUENCE = 0b0010
    NEGATIVE_SEQUENCE = 0b0011
    WITH_EVENT = 0b0100


class Serialization(IntEnum):
    RAW = 0
    JSON = 0b0001


class Compression(IntEnum):
    NONE = 0
    GZIP = 0b0001


class EventType(IntEnum):
    NONE = 0
    START_CONNECTION = 1
    FINISH_CONNECTION = 2
    CONNECTION_STARTED = 50
    CONNECTION_FAILED = 51
    CONNECTION_FINISHED = 52
    START_SESSION = 100
    CANCEL_SESSION = 101
    FINISH_SESSION = 102
    SESSION_STARTED = 150
    SESSION_CANCELED = 151
    SESSION_FINISHED = 152
    SESSION_FAILED = 153
    USAGE_RESPONSE = 154
    TASK_REQUEST = 200
    UPDATE_CONFIG = 201
    AUDIO_MUTED = 250
    TTS_SENTENCE_START = 350
    TTS_SENTENCE_END = 351
    TTS_RESPONSE = 352
    TTS_ENDED = 359
    TTS_SUBTITLE = 364


_CONNECTION_EVENTS = frozenset(
    {
        EventType.START_CONNECTION,
        EventType.FINISH_CONNECTION,
        EventType.CONNECTION_STARTED,
        EventType.CONNECTION_FAILED,
        EventType.CONNECTION_FINISHED,
    }
)


class DoubaoProtocolError(RuntimeError):
    """Malformed or unsupported Doubao binary frame."""


def _read_exact(buffer: io.BytesIO, size: int) -> bytes:
    data = buffer.read(size)
    if len(data) != size:
        raise DoubaoProtocolError(f"frame ended early: expected {size} bytes, got {len(data)}")
    return data


@dataclass
class Message:
    type: MsgType = MsgType.INVALID
    flag: Flags = Flags.NO_SEQUENCE
    serialization: Serialization = Serialization.JSON
    compression: Compression = Compression.NONE
    event: EventType | int = EventType.NONE
    session_id: str = ""
    connect_id: str = ""
    sequence: int = 0
    error_code: int = 0
    payload: bytes = field(default=b"")

    def json_payload(self) -> dict[str, Any] | None:
        if not self.payload:
            return None
        raw = self.payload
        try:
            if self.compression == Compression.GZIP:
                raw = gzip.decompress(raw)
            decoded = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return decoded if isinstance(decoded, dict) else None

    def marshal(self) -> bytes:
        buffer = io.BytesIO()
        buffer.write(
            bytes(
                [
                    (1 << 4) | 1,
                    (int(self.type) << 4) | int(self.flag),
                    (int(self.serialization) << 4) | int(self.compression),
                    0,
                ]
            )
        )
        if self.flag == Flags.WITH_EVENT:
            buffer.write(struct.pack(">i", int(self.event)))
            if self.event not in _CONNECTION_EVENTS:
                session = self.session_id.encode("utf-8")
                buffer.write(struct.pack(">I", len(session)))
                buffer.write(session)
        if self.type == MsgType.ERROR:
            buffer.write(struct.pack(">I", self.error_code & 0xFFFFFFFF))
        elif self.flag in {Flags.POSITIVE_SEQUENCE, Flags.NEGATIVE_SEQUENCE}:
            buffer.write(struct.pack(">i", self.sequence))
        buffer.write(struct.pack(">I", len(self.payload)))
        buffer.write(self.payload)
        return buffer.getvalue()

    @classmethod
    def unmarshal(cls, data: bytes) -> Message:
        if len(data) < 4:
            raise DoubaoProtocolError(f"frame is too short: {len(data)} bytes")
        version = data[0] >> 4
        header_size = (data[0] & 0x0F) * 4
        if version != 1 or header_size < 4 or header_size > len(data):
            raise DoubaoProtocolError(
                f"unsupported protocol header: version={version}, size={header_size}"
            )
        try:
            message = cls(
                type=MsgType(data[1] >> 4),
                flag=Flags(data[1] & 0x0F),
                serialization=Serialization(data[2] >> 4),
                compression=Compression(data[2] & 0x0F),
            )
        except ValueError as exc:
            raise DoubaoProtocolError(f"unknown frame header value: {exc}") from exc

        buffer = io.BytesIO(data[header_size:])
        if message.flag == Flags.WITH_EVENT:
            event_value = struct.unpack(">i", _read_exact(buffer, 4))[0]
            try:
                message.event = EventType(event_value)
            except ValueError:
                message.event = event_value
            if message.event not in _CONNECTION_EVENTS:
                session_size = struct.unpack(">I", _read_exact(buffer, 4))[0]
                if session_size:
                    message.session_id = _read_exact(buffer, session_size).decode(
                        "utf-8", errors="replace"
                    )
            elif message.type == MsgType.FULL_SERVER_RESPONSE:
                connect_size = struct.unpack(">I", _read_exact(buffer, 4))[0]
                if connect_size:
                    message.connect_id = _read_exact(buffer, connect_size).decode(
                        "utf-8", errors="replace"
                    )

        if message.type == MsgType.ERROR:
            message.error_code = struct.unpack(">I", _read_exact(buffer, 4))[0]
        elif message.flag in {Flags.POSITIVE_SEQUENCE, Flags.NEGATIVE_SEQUENCE}:
            message.sequence = struct.unpack(">i", _read_exact(buffer, 4))[0]

        payload_size = struct.unpack(">I", _read_exact(buffer, 4))[0]
        if payload_size:
            message.payload = _read_exact(buffer, payload_size)
        return message


def client_event(
    event: EventType,
    *,
    session_id: str = "",
    payload: bytes = b"{}",
) -> Message:
    return Message(
        type=MsgType.FULL_CLIENT_REQUEST,
        flag=Flags.WITH_EVENT,
        event=event,
        session_id=session_id,
        payload=payload,
    )
