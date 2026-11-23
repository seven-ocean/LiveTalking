# -*- coding: utf-8 -*-
"""Unit tests for the strict-parser answer SDP sanitizer.

Run from the zl-talking repo root:

    python3 tests/test_sdp_sanitize.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.sdp_sanitize import describe_sdp, sanitize_answer_sdp  # noqa: E402


def _media(kind, mid, extra=""):
    return (
        f"m={kind} 48818 UDP/TLS/RTP/SAVPF 96\r\n"
        "c=IN IP4 10.192.26.107\r\n"
        "a=candidate:1 1 udp 2130706431 10.192.26.107 48818 typ host\r\n"
        f"{extra}"
        f"a=mid:{mid}\r\n"
    )


def _answer(media_extra="", session_extra="", with_audio=True):
    body = (
        "v=0\r\no=- 1 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n"
        "a=group:BUNDLE 0 1\r\n"
        f"{session_extra}"
        + _media("video", 0, media_extra)
    )
    if with_audio:
        body += _media("audio", 1, media_extra)
    return body


class SanitizeAnswerSdpTest(unittest.TestCase):
    def test_media_level_ice_credentials_deduped(self):
        sdp = _answer(
            media_extra=(
                "a=ice-ufrag:abc\r\na=ice-pwd:def\r\n"
                "a=ice-ufrag:dup\r\na=ice-pwd:dup\r\n"
                "a=fingerprint:sha-256 AA:BB\r\na=fingerprint:sha-384 CC:DD\r\n"
                "a=setup:active\r\n"
            )
        )
        out = sanitize_answer_sdp(sdp)
        stats = describe_sdp(out)
        self.assertEqual(stats["ice_ufrag"], 2)
        self.assertEqual(stats["ice_pwd"], 2)
        self.assertEqual(stats["fingerprints"], 1)
        self.assertEqual(stats["setups"], 1)
        self.assertIn("a=ice-ufrag:abc\r\n", out)
        self.assertIn("a=ice-pwd:def\r\n", out)
        self.assertNotIn("dup", out)

    def test_session_level_credentials_move_to_media(self):
        sdp = _answer(
            session_extra="a=ice-ufrag:sess\r\na=ice-pwd:sesspwd\r\n",
            media_extra="a=fingerprint:sha-256 AA:BB\r\na=setup:active\r\n",
        )
        out = sanitize_answer_sdp(sdp)
        # Session block must not contain the credentials; each media gets a pair.
        session_part, media_part = out.split("m=", 1)
        self.assertNotIn("a=ice-ufrag:", session_part)
        self.assertNotIn("a=ice-pwd:", session_part)
        self.assertEqual(media_part.count("a=ice-ufrag:sess"), 2)
        self.assertEqual(media_part.count("a=ice-pwd:sesspwd"), 2)

    def test_glued_attribute_is_split(self):
        sdp = _answer(
            media_extra="a=ice-pwd:defa=ice-ufrag:abc\r\na=fingerprint:sha-256 AA:BB\r\na=setup:active\r\n"
        )
        out = sanitize_answer_sdp(sdp)
        self.assertIn("a=ice-pwd:def\r\n", out)
        self.assertIn("a=ice-ufrag:abc\r\n", out)
        self.assertEqual(describe_sdp(out)["ice_pwd"], 2)

    def test_tcp_and_docker_candidates_removed_when_lan_exists(self):
        sdp = _answer(
            media_extra=(
                "a=ice-ufrag:abc\r\na=ice-pwd:def\r\n"
                "a=fingerprint:sha-256 AA:BB\r\na=setup:active\r\n"
            )
        )
        sdp = sdp.replace(
            "a=mid:0\r\n",
            "a=candidate:2 1 udp 2130706431 172.20.0.1 59109 typ host\r\n"
            "a=candidate:3 1 tcp 1518280447 10.192.26.107 9 typ host tcptype active\r\n"
            "a=mid:0\r\n",
        )
        out = sanitize_answer_sdp(sdp)
        self.assertNotIn("172.20.0.1", out)
        self.assertNotIn(" tcp ", out)
        self.assertIn("10.192.26.107", out)

    def test_normal_sdp_is_stable(self):
        sdp = _answer(
            media_extra=(
                "a=ice-ufrag:abc\r\na=ice-pwd:def\r\n"
                "a=fingerprint:sha-256 AA:BB\r\na=setup:active\r\n"
            )
        )
        out = sanitize_answer_sdp(sdp)
        self.assertTrue(out.endswith("\r\n"))
        self.assertEqual(sanitize_answer_sdp(out), out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
