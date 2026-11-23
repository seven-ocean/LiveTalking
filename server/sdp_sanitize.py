# -*- coding: utf-8 -*-
"""answer SDP sanitizer for strict Chromium/Electron parsers.

The server (aiortc) emits SDP that modern Chrome tolerates but Electron 25 /
Chromium 114 rejects, e.g.::

    Failed to parse SessionDescription. a=ice-pwd:... Invalid SDP line.
    Failed to parse SessionDescription. a=fingerprint:sha-256 ... Invalid SDP line.
    Failed to parse SessionDescription. a=setup:active Invalid SDP line.

This module rebuilds the answer into the conservative shape:

* ICE credentials are media-level only, one pair per ``m=`` section;
* exactly one session-level sha-256 ``a=fingerprint``;
* ``a=setup`` only at session level;
* malformed / concatenated attribute lines are repaired or dropped;
* unusable host candidates (tcp, loopback, link-local, docker bridge ``.1``)
  are removed when at least one usable candidate remains;
* CRLF line endings with a trailing CRLF.

It is dependency-free so it can be unit-tested without importing aiortc.
"""

import ipaddress
import re
from typing import Dict, List, Optional, Tuple

_ICE_ATTR_RE = re.compile(r"^(a=ice-(?:ufrag|pwd))\s*:\s*(.*)$", re.IGNORECASE)
_ICE_VALUE_RE = re.compile(r"^[A-Za-z0-9+/=]{1,256}$")
_ICE_GLUED_RE = re.compile(
    r"^(a=ice-(?:ufrag|pwd)\s*:\s*[A-Za-z0-9+/=]{1,256})(a=.*)$",
    re.IGNORECASE,
)
_FINGERPRINT_RE = re.compile(r"^(a=fingerprint)\s*:\s*([A-Za-z0-9-]+)\s+(\S+)$", re.IGNORECASE)
_CANDIDATE_RE = re.compile(
    r"^a=candidate:\S+ \d+ (udp|tcp) \d+ (\S+) (\d+) typ (host|srflx|prflx|relay)\b",
    re.IGNORECASE,
)
_LINE_RE = re.compile(r"^[a-z]=.*$")
_DOCKER_BRIDGE_RE = re.compile(r"^172\.(?:1[7-9]|2\d|3[01])\.\d+\.1$")


def _split_lines(sdp: str) -> List[str]:
    """Normalize line endings and drop empty / malformed / glued attribute lines."""
    text = (sdp or "").replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines: List[str] = []
    for raw in text.split("\n"):
        line = "".join(ch for ch in raw if ch >= " " or ch == "\t").strip()
        if not line:
            continue
        # aiortc never emits glued attributes, but broken proxies sometimes do.
        glued = _ICE_GLUED_RE.match(line)
        if glued:
            lines.append(glued.group(1).strip())
            lines.append(glued.group(2).strip())
            continue
        if not _LINE_RE.match(line):
            continue
        lines.append(line)
    return lines


def _split_sections(lines: List[str]) -> List[List[str]]:
    """Return [session, media1, media2, ...]."""
    sections: List[List[str]] = [[]]
    for line in lines:
        if line.startswith("m="):
            sections.append([])
        sections[-1].append(line)
    return sections


def _take_ice(lines: List[str]) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Remove ICE credential lines, returning (first valid ufrag, first valid pwd, rest)."""
    ufrag: Optional[str] = None
    pwd: Optional[str] = None
    rest: List[str] = []
    for line in lines:
        m = _ICE_ATTR_RE.match(line)
        if not m:
            rest.append(line)
            continue
        name = m.group(1).lower()
        value = m.group(2).strip()
        if not _ICE_VALUE_RE.match(value):
            continue
        if name.endswith("ufrag"):
            if ufrag is None:
                ufrag = value
        elif pwd is None:
            pwd = value
    return ufrag, pwd, rest


def _is_bad_host(ip: str, cand_type: str) -> bool:
    if cand_type != "host":
        return False
    if ip.startswith("127.") or ip.startswith("169.254.") or ip in ("0.0.0.0", "::1"):
        return True
    if ip.lower().startswith("fe80:"):
        return True
    return bool(_DOCKER_BRIDGE_RE.match(ip))


def _candidate_ip(line: str) -> Tuple[Optional[str], Optional[str]]:
    m = _CANDIDATE_RE.match(line)
    if not m:
        return None, None
    return m.group(2), m.group(4).lower()


def _filter_candidates(lines: List[str]) -> List[str]:
    candidates = [line for line in lines if line.startswith("a=candidate:")]
    if not candidates:
        return lines
    # Do not drop every host candidate if the box only has docker/loopback IPs.
    keep_non_docker = any(
        (ip is not None and not _is_bad_host(ip, typ))
        for ip, typ in (_candidate_ip(line) for line in candidates)
    )
    out: List[str] = []
    for line in lines:
        if not line.startswith("a=candidate:"):
            out.append(line)
            continue
        m = _CANDIDATE_RE.match(line)
        if not m:
            continue
        proto = m.group(1).lower()
        ip = m.group(2)
        typ = m.group(4).lower()
        if proto == "tcp":
            continue
        if keep_non_docker and _is_bad_host(ip, typ):
            continue
        out.append(line)
    return out


def _insert_ice(lines: List[str], ufrag: Optional[str], pwd: Optional[str]) -> List[str]:
    """Insert one ICE credential pair after ``a=mid`` (or after m=/c= lines)."""
    out = list(lines)
    insert_at = len(out)
    for i, line in enumerate(out):
        if line.startswith("a=mid:"):
            insert_at = i + 1
            break
    else:
        for i, line in enumerate(out):
            if line.startswith("a=mid:") or line.startswith("a=mid :"):
                insert_at = i + 1
                break
        else:
            for i in range(len(out) - 1, -1, -1):
                if out[i].startswith(("m=", "c=", "a=rtcp:")):
                    insert_at = i + 1
                    break
    add: List[str] = []
    if ufrag:
        add.append(f"a=ice-ufrag:{ufrag}")
    if pwd:
        add.append(f"a=ice-pwd:{pwd}")
    return out[:insert_at] + add + out[insert_at:]


def _insert_session_attrs(lines: List[str], fingerprint: Optional[str], setup: Optional[str]) -> List[str]:
    out = list(lines)
    insert_at = len(out)
    for i, line in enumerate(out):
        if line.startswith("m="):
            insert_at = i
            break
    # Place after t= (or v=/o=/s=) and before the first m=.
    t_index = next((i for i, line in enumerate(out[:insert_at]) if line.startswith("t=")), None)
    if t_index is not None:
        insert_at = t_index + 1
    add: List[str] = []
    if setup:
        add.append(f"a=setup:{setup}")
    if fingerprint:
        add.append(fingerprint)
    return out[:insert_at] + add + out[insert_at:]


def sanitize_answer_sdp(sdp: str) -> str:
    """Return a strict-parser-safe answer SDP. Never raises; falls back to input."""
    if not sdp or "v=" not in sdp:
        return sdp

    try:
        lines = _split_lines(sdp)
        if not lines:
            return sdp
        sections = _split_sections(lines)
        session_lines = sections[0]
        media_lines = sections[1:]

        session_ufrag, session_pwd, session_rest = _take_ice(session_lines)

        # fingerprints / setup may live at session or media level.
        fingerprint: Optional[str] = None
        setup: Optional[str] = None
        cleaned_media: List[List[str]] = []
        media_ice: List[Tuple[Optional[str], Optional[str]]] = []

        for media in media_lines:
            ufrag, pwd, rest = _take_ice(media)
            media_ice.append((ufrag, pwd))
            kept: List[str] = []
            for line in rest:
                if line.startswith("a=fingerprint:"):
                    m = _FINGERPRINT_RE.match(line)
                    if m and m.group(2).lower() == "sha-256" and fingerprint is None:
                        fingerprint = f"a=fingerprint:sha-256 {m.group(3)}"
                    continue
                if line.startswith("a=setup:"):
                    if setup is None:
                        setup = line.split(":", 1)[1].strip()
                    continue
                kept.append(line)
            cleaned_media.append(_filter_candidates(kept))

        # Session-level fingerprint/setup cleanup.
        session_out: List[str] = []
        for line in session_rest:
            if line.startswith("a=fingerprint:"):
                m = _FINGERPRINT_RE.match(line)
                if m and m.group(2).lower() == "sha-256" and fingerprint is None:
                    fingerprint = f"a=fingerprint:sha-256 {m.group(3)}"
                continue
            if line.startswith("a=setup:"):
                if setup is None:
                    setup = line.split(":", 1)[1].strip()
                continue
            session_out.append(line)
        session_out = _insert_session_attrs(session_out, fingerprint, setup)

        # Fill missing per-media ICE credentials from session/other media.
        fallback_ufrag = session_ufrag or next((u for u, _ in media_ice if u), None)
        fallback_pwd = session_pwd or next((p for _, p in media_ice if p), None)
        out_media: List[List[str]] = []
        for idx, media in enumerate(cleaned_media):
            ufrag, pwd = media_ice[idx]
            ufrag = ufrag or fallback_ufrag
            pwd = pwd or fallback_pwd
            out_media.append(_insert_ice(media, ufrag, pwd))

        out = session_out + [line for media in out_media for line in media]
        return "\r\n".join(out) + "\r\n"
    except Exception:  # pragma: no cover - never break the call because of SDP cleanup
        return sdp


def describe_sdp(sdp: str) -> Dict[str, int]:
    """Small helper for diagnostics/logging."""
    lines = _split_lines(sdp or "")
    return {
        "lines": len(lines),
        "media": sum(1 for line in lines if line.startswith("m=")),
        "ice_ufrag": sum(1 for line in lines if line.startswith("a=ice-ufrag:")),
        "ice_pwd": sum(1 for line in lines if line.startswith("a=ice-pwd:")),
        "fingerprints": sum(1 for line in lines if line.startswith("a=fingerprint:")),
        "setups": sum(1 for line in lines if line.startswith("a=setup:")),
        "candidates": sum(1 for line in lines if line.startswith("a=candidate:")),
        "tcp_candidates": sum(1 for line in lines if line.startswith("a=candidate:") and " tcp " in line),
    }
