from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from PyQt5.QtCore import QObject, QThread, pyqtSignal

from .dependency_manager import ensure_tshark
from .feature_engineering import FlowAggregator, PacketInfo, dataframe_from_feature_dicts, estimate_direction
from .session_manager import SessionManager

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import pyshark
    from pyshark.capture import capture as pyshark_capture  # type: ignore

    TSharkNotFoundException = getattr(pyshark_capture, "TSharkNotFoundException", Exception)
    PYSHARK_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when PyShark/TShark missing
    pyshark = None  # type: ignore
    TSharkNotFoundException = Exception  # type: ignore
    PYSHARK_AVAILABLE = False


@dataclass
class LiveCaptureConfig:
    interface: Optional[str]
    idle_timeout: float
    host_ips: Optional[Iterable[str]] = None
    tshark_path: Optional[Path] = None


class LiveCaptureWorker(QThread):
    flow_ready = pyqtSignal(object)  # Emits pandas DataFrame with ready flows
    stats_updated = pyqtSignal(dict)
    dependency_missing = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, session: SessionManager, config: LiveCaptureConfig):
        super().__init__()
        self.session = session
        self.config = config
        self._stop_event = threading.Event()
        self.aggregator = FlowAggregator(idle_timeout=config.idle_timeout)
        self._sniffer = None

    def stop(self) -> None:
        self._stop_event.set()
        if self._sniffer is not None:
            try:
                self._sniffer.stop()
            except Exception:  # pragma: no cover
                LOGGER.exception("Failed to stop sniffer cleanly")
        self.quit()
        self.wait(2000)

    def run(self) -> None:  # pragma: no cover - complex to test
        if not PYSHARK_AVAILABLE:
            self.dependency_missing.emit(
                "PyShark/TShark is not available. Falling back to synthetic traffic generation."
            )
            self._run_synthetic_generator()
            return

        try:
            self._run_pyshark_capture()
        except TSharkNotFoundException as exc:  # pragma: no cover - depends on runtime
            LOGGER.exception("TShark not found")
            self.dependency_missing.emit(
                "TShark executable not found. Install Wireshark/TShark and ensure it is on PATH."
            )
            self._run_synthetic_generator()
        except Exception as exc:  # pragma: no cover - runtime only
            LOGGER.exception("Live capture failed")
            self.error_occurred.emit(str(exc))
            self._run_synthetic_generator()

    def _run_pyshark_capture(self) -> None:
        assert pyshark is not None
        interface = self.config.interface if self.config.interface else None
        LOGGER.info("Starting PyShark capture on %s", interface or "default interface")
        capture_kwargs = {
            "interface": interface,
            "include_raw": True,
            "use_json": True,
        }
        if self.config.tshark_path:
            capture_kwargs["tshark_path"] = str(self.config.tshark_path)

        previous_loop = None
        loop: Optional[asyncio.AbstractEventLoop] = None
        capture = None
        try:
            previous_loop = asyncio.get_event_loop()
        except RuntimeError:
            previous_loop = None

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        capture = pyshark.LiveCapture(**capture_kwargs)
        self.session.record_capture_start()

        try:
            for packet in capture.sniff_continuously():
                if self._stop_event.is_set():
                    break
                info = self._convert_pyshark_packet(packet)
                if info is None:
                    continue
                self.session.record_packet()
                ready = self.aggregator.update(info)
                self._emit_ready_flows(ready)
        finally:
            if capture is not None:
                try:
                    capture.close()
                except Exception:
                    LOGGER.debug("Failed to close capture cleanly", exc_info=True)
            ready = self.aggregator.flush()
            self._emit_ready_flows(ready)
            self.session.record_capture_stop()
            if loop is not None:
                loop.stop()
                loop.close()
            if previous_loop is not None:
                asyncio.set_event_loop(previous_loop)
            else:
                asyncio.set_event_loop(None)

    def _convert_pyshark_packet(self, packet) -> Optional[PacketInfo]:  # pragma: no cover - runtime only
        timestamp = self._safe_float(getattr(packet, "sniff_timestamp", time.time()))

        src_ip: Optional[str] = None
        dest_ip: Optional[str] = None
        proto_value: Optional[str] = None

        if hasattr(packet, "ip"):
            layer = packet.ip
            src_ip = getattr(layer, "src", None)
            dest_ip = getattr(layer, "dst", None)
            proto_value = getattr(layer, "proto", None)
        elif hasattr(packet, "ipv6"):
            layer = packet.ipv6
            src_ip = getattr(layer, "src", None)
            dest_ip = getattr(layer, "dst", None)
            proto_value = getattr(layer, "nxt", getattr(layer, "nh", None))
        else:
            return None

        if not src_ip or not dest_ip:
            return None

        src_port = 0
        dest_port = 0
        protocol_name = "other"

        if hasattr(packet, "tcp"):
            l4 = packet.tcp
            src_port = self._safe_int(getattr(l4, "srcport", 0))
            dest_port = self._safe_int(getattr(l4, "dstport", 0))
            protocol_name = "tcp"
        elif hasattr(packet, "udp"):
            l4 = packet.udp
            src_port = self._safe_int(getattr(l4, "srcport", 0))
            dest_port = self._safe_int(getattr(l4, "dstport", 0))
            protocol_name = "udp"
        elif hasattr(packet, "icmp") or hasattr(packet, "icmpv6"):
            protocol_name = "icmp"
        elif proto_value:
            protocol_name = str(proto_value)

        frame_info = getattr(packet, "frame_info", None)
        length = self._safe_int(getattr(frame_info, "len", 0)) if frame_info is not None else 0

        payload = b""
        try:
            raw_packet = packet.get_raw_packet()
            if raw_packet:
                payload = raw_packet
        except Exception:
            payload = b""

        direction = estimate_direction(self.config.host_ips, src_ip)
        return PacketInfo(
            src_ip=src_ip,
            src_port=src_port,
            dest_ip=dest_ip,
            dest_port=dest_port,
            proto=str(protocol_name),
            length=length,
            timestamp=timestamp,
            payload=payload,
            direction=direction,
        )

    @staticmethod
    def _safe_int(value: object, default: int = 0) -> int:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(str(value))
        except (TypeError, ValueError):
            return default

    def _run_synthetic_generator(self) -> None:
        self.session.record_capture_start()
        try:
            while not self._stop_event.is_set():
                packet = self._generate_synthetic_packet()
                ready = self.aggregator.update(packet)
                self._emit_ready_flows(ready)
                time.sleep(0.25)
        finally:
            ready = self.aggregator.flush()
            self._emit_ready_flows(ready)
            self.session.record_capture_stop()

    def _generate_synthetic_packet(self) -> PacketInfo:
        now = time.time()
        src_ip = f"192.168.1.{random.randint(2, 200)}"
        dest_ip = f"10.0.0.{random.randint(2, 200)}"
        src_port = random.randint(1024, 65535)
        dest_port = random.choice([22, 53, 80, 443, 8080])
        proto = random.choice(["tcp", "udp", "icmp"])
        length = random.randint(60, 800)
        payload = bytes(random.getrandbits(8) for _ in range(random.randint(0, 128)))
        direction = random.choice(["in", "out"])
        self.session.record_packet()
        return PacketInfo(
            src_ip=src_ip,
            src_port=src_port,
            dest_ip=dest_ip,
            dest_port=dest_port,
            proto=proto,
            length=length,
            timestamp=now,
            payload=payload,
            direction=direction,
        )

    def _emit_ready_flows(self, ready: list) -> None:
        if not ready:
            return
        df = dataframe_from_feature_dicts(ready)
        if not df.empty:
            self.flow_ready.emit(df)
            self.stats_updated.emit(
                {
                    "flows": len(ready),
                    "packets": self.session.stats.packets_captured,
                    "alerts": self.session.stats.alerts_triggered,
                }
            )


class LiveCaptureController(QObject):
    flow_ready = pyqtSignal(object)
    stats_updated = pyqtSignal(dict)
    dependency_missing = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, session: SessionManager, idle_timeout: float = 2.0, tshark_path: Optional[Path] = None):
        super().__init__()
        self.session = session
        self.idle_timeout = idle_timeout
        self.worker: Optional[LiveCaptureWorker] = None
        self._tshark_path = tshark_path

    def start(self, interface: Optional[str], host_ips: Optional[Iterable[str]] = None) -> None:
        if self.worker and self.worker.isRunning():
            LOGGER.info("Live capture already running")
            return

        if not self._tshark_path:
            self._tshark_path = ensure_tshark(auto_install=True)

        if not self._tshark_path:
            self.dependency_missing.emit(
                "TShark executable not found. Install Wireshark/TShark and ensure 'tshark.exe' is on PATH."
            )
            return

        config = LiveCaptureConfig(
            interface=interface,
            idle_timeout=self.idle_timeout,
            host_ips=host_ips,
            tshark_path=self._tshark_path,
        )
        self.worker = LiveCaptureWorker(self.session, config)
        self.worker.flow_ready.connect(self.flow_ready.emit)
        self.worker.stats_updated.connect(self.stats_updated.emit)
        self.worker.dependency_missing.connect(self.dependency_missing.emit)
        self.worker.error_occurred.connect(self.error_occurred.emit)
        self.worker.start()

    def stop(self) -> None:
        if self.worker:
            self.worker.stop()
            self.worker = None
