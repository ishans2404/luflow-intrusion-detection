from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .converters import ip_to_int, protocol_to_numeric


@dataclass
class PacketInfo:
    src_ip: str
    src_port: int
    dest_ip: str
    dest_port: int
    proto: str
    length: int
    timestamp: float
    payload: bytes = b""
    direction: str = "out"


@dataclass
class FlowStats:
    key: Tuple[str, int, str, int, str]
    first_timestamp: float
    last_timestamp: float
    first_direction: str
    bytes_out: int = 0
    bytes_in: int = 0
    num_pkts_out: int = 0
    num_pkts_in: int = 0
    packet_timestamps: Deque[float] = field(default_factory=deque)
    payload_counter: Counter = field(default_factory=Counter)

    def update(self, packet: PacketInfo) -> None:
        self.last_timestamp = packet.timestamp
        self.packet_timestamps.append(packet.timestamp)
        if packet.direction == self.first_direction:
            self.num_pkts_out += 1
            self.bytes_out += packet.length
        else:
            self.num_pkts_in += 1
            self.bytes_in += packet.length

        if packet.payload:
            self.payload_counter.update(packet.payload)

    def duration(self) -> float:
        return max(0.0, self.last_timestamp - self.first_timestamp)

    def entropy(self) -> float:
        total_bytes = sum(self.payload_counter.values())
        if total_bytes == 0:
            return 0.0
        entropy = 0.0
        for count in self.payload_counter.values():
            p = count / total_bytes
            entropy -= p * math.log2(p)
        return entropy

    def total_entropy(self) -> float:
        return self.entropy() * sum(self.payload_counter.values())

    def avg_inter_packet_interval(self) -> float:
        if len(self.packet_timestamps) < 2:
            return 0.0
        diffs = [self.packet_timestamps[i + 1] - self.packet_timestamps[i] for i in range(len(self.packet_timestamps) - 1)]
        diffs = [d for d in diffs if d >= 0]
        if not diffs:
            return 0.0
        return float(sum(diffs) / len(diffs))

    def to_feature_row(self) -> Dict[str, float]:
        src_ip, src_port, dest_ip, dest_port, proto = self.key
        features = {
            "src_ip": ip_to_int(src_ip),
            "src_port": src_port,
            "dest_ip": ip_to_int(dest_ip),
            "dest_port": dest_port,
            "proto": protocol_to_numeric(proto),
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
            "num_pkts_in": self.num_pkts_in,
            "num_pkts_out": self.num_pkts_out,
            "entropy": self.entropy(),
            "total_entropy": self.total_entropy(),
            "avg_ipt": self.avg_inter_packet_interval(),
            "time_start": self.first_timestamp,
            "time_end": self.last_timestamp,
            "duration": self.duration(),
        }
        return features


class FlowAggregator:
    """Aggregate packets into bidirectional flows ready for inference."""

    def __init__(self, idle_timeout: float = 2.0) -> None:
        self.idle_timeout = idle_timeout
        self.flows: Dict[Tuple[str, int, str, int, str], FlowStats] = {}

    def _flow_key(self, packet: PacketInfo) -> Tuple[str, int, str, int, str]:
        return (packet.src_ip, packet.src_port, packet.dest_ip, packet.dest_port, packet.proto)

    def update(self, packet: PacketInfo) -> List[Dict[str, float]]:
        now = packet.timestamp
        key = self._flow_key(packet)

        if key not in self.flows:
            self.flows[key] = FlowStats(
                key=key,
                first_timestamp=packet.timestamp,
                last_timestamp=packet.timestamp,
                first_direction=packet.direction,
            )
        self.flows[key].update(packet)

        ready_flows: List[Dict[str, float]] = []
        expired_keys = []
        for flow_key, stats in self.flows.items():
            if now - stats.last_timestamp >= self.idle_timeout and stats.num_pkts_in + stats.num_pkts_out > 0:
                ready_flows.append(stats.to_feature_row())
                expired_keys.append(flow_key)

        for flow_key in expired_keys:
            del self.flows[flow_key]

        return ready_flows

    def flush(self) -> List[Dict[str, float]]:
        ready = [stats.to_feature_row() for stats in self.flows.values() if stats.num_pkts_in + stats.num_pkts_out > 0]
        self.flows.clear()
        return ready


def dataframe_from_feature_dicts(feature_dicts: Iterable[Dict[str, float]]) -> pd.DataFrame:
    if not feature_dicts:
        return pd.DataFrame()
    return pd.DataFrame(list(feature_dicts))


def estimate_direction(host_ips: Optional[Iterable[str]], src_ip: str) -> str:
    if not host_ips:
        return "out"
    return "in" if src_ip and src_ip in host_ips else "out"
