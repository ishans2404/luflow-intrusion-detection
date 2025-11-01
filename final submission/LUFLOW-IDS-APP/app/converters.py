from __future__ import annotations

import ipaddress
from typing import Union

PROTOCOL_MAPPING = {
    "icmp": 1,
    "tcp": 6,
    "udp": 17,
    "gre": 47,
    "esp": 50,
    "ah": 51,
    "sctp": 132,
    "other": 0,
}


def ip_to_int(value: Union[str, int, float]) -> int:
    """Convert IP address representations to an integer."""

    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)

    text = str(value).strip()
    if not text:
        return 0

    try:
        return int(ipaddress.ip_address(text))
    except ValueError:
        # Fall back to hash to keep deterministic conversion.
        return abs(hash(text)) % (2 ** 32)


def protocol_to_numeric(value: Union[str, int, float]) -> int:
    """Best-effort conversion of protocol labels to their numeric codes."""

    if value is None:
        return PROTOCOL_MAPPING["other"]
    if isinstance(value, (int, float)):
        return int(value)

    text = str(value).strip().lower()
    if text.isdigit():
        return int(text)

    return PROTOCOL_MAPPING.get(text, PROTOCOL_MAPPING["other"])
