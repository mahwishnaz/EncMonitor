from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FrameStats:
    header_bytes: int
    blob_bytes: int

    @property
    def wire_bytes(self) -> int:
        return 8 + self.header_bytes + self.blob_bytes


def _recvn(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    got = 0
    while got < n:
        chunk = sock.recv(n - got)
        if not chunk:
            raise ConnectionError("socket closed before frame completed")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def send_packet(sock: socket.socket, header: Mapping[str, Any], blob: bytes = b"") -> FrameStats:
    raw = json.dumps(dict(header), separators=(",", ":"), sort_keys=True).encode("utf-8")
    blob = b"" if blob is None else blob
    sock.sendall(struct.pack("!I", len(raw)))
    sock.sendall(raw)
    sock.sendall(struct.pack("!I", len(blob)))
    if blob:
        sock.sendall(blob)
    return FrameStats(len(raw), len(blob))


def recv_packet(sock: socket.socket) -> tuple[dict[str, Any], bytes, FrameStats]:
    hlen = struct.unpack("!I", _recvn(sock, 4))[0]
    header = json.loads(_recvn(sock, hlen).decode("utf-8"))
    blen = struct.unpack("!I", _recvn(sock, 4))[0]
    blob = _recvn(sock, blen) if blen else b""
    return header, blob, FrameStats(hlen, blen)
