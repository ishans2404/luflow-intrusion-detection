from __future__ import annotations

import json
import logging
import os
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets

from . import config
from .csv_mode import predict_from_csv
from .dependency_manager import ensure_tshark
from .logging_utils import configure_logging
from .model_manager import ModelManager
from .session_manager import SessionManager
from .live_capture import LiveCaptureController
from .single_prediction import predict_from_manual_input

LOGGER = logging.getLogger(__name__)


def safe_open_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if platform.system().lower().startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        os.system(f"open '{path}'")
    else:
        os.system(f"xdg-open '{path}'")


def list_interfaces() -> List[str]:
    try:
        import psutil  # type: ignore

        return sorted(psutil.net_if_addrs().keys())
    except Exception:
        return []


def host_ip_addresses() -> Iterable[str]:
    try:
        hostname = socket.gethostname()
        host_ips = socket.gethostbyname_ex(hostname)[2]
        return set(host_ips)
    except Exception:
        return []


class DashboardWidget(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        self.packet_label = QtWidgets.QLabel("0")
        self.flows_label = QtWidgets.QLabel("0")
        self.alert_label = QtWidgets.QLabel("0")
        self.capture_label = QtWidgets.QLabel("Idle")

        layout.addWidget(self._create_card("Packets", self.packet_label), 0, 0)
        layout.addWidget(self._create_card("Flows", self.flows_label), 0, 1)
        layout.addWidget(self._create_card("Alerts", self.alert_label), 0, 2)
        layout.addWidget(self._create_card("Capture", self.capture_label), 1, 0, 1, 3)

    def _create_card(self, title: str, value_label: QtWidgets.QLabel) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(title)
        box_layout = QtWidgets.QVBoxLayout()

        alignment = getattr(QtCore.Qt, "AlignCenter", None)
        if alignment is None:
            alignment_flag = getattr(QtCore.Qt, "AlignmentFlag", None)
            if alignment_flag is not None:
                alignment = alignment_flag.AlignCenter
        if alignment is None:
            alignment = QtCore.Qt.AlignCenter  # type: ignore[attr-defined]

        value_label.setAlignment(alignment)  # type: ignore[arg-type]
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        value_label.setFont(font)
        box_layout.addWidget(value_label)
        box.setLayout(box_layout)
        return box

    def update_stats(self, packets: int, flows: int, alerts: int, capture_active: bool) -> None:
        self.packet_label.setText(f"{packets}")
        self.flows_label.setText(f"{flows}")
        self.alert_label.setText(f"{alerts}")
        self.capture_label.setText("Running" if capture_active else "Idle")


class LiveCaptureTab(QtWidgets.QWidget):
    start_requested = QtCore.pyqtSignal(str)
    stop_requested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        controls_layout = QtWidgets.QHBoxLayout()
        self.interface_combo = QtWidgets.QComboBox()
        self.interface_combo.addItem("Auto")
        for name in list_interfaces():
            self.interface_combo.addItem(name)
        controls_layout.addWidget(QtWidgets.QLabel("Interface:"))
        controls_layout.addWidget(self.interface_combo)

        self.start_button = QtWidgets.QPushButton("Start Capture")
        self.stop_button = QtWidgets.QPushButton("Stop Capture")
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)

        layout.addLayout(controls_layout)

        self.dashboard = DashboardWidget()
        layout.addWidget(self.dashboard)

        self.activity_table = QtWidgets.QTableWidget(0, 7)
        self.activity_table.setHorizontalHeaderLabels(
            [
                "Timestamp",
                "Mode",
                "Label",
                "Confidence",
                "Bytes In",
                "Bytes Out",
                "Duration",
            ]
        )
        header = self.activity_table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        layout.addWidget(self.activity_table)

        self.status_box = QtWidgets.QTextEdit()
        self.status_box.setReadOnly(True)
        layout.addWidget(self.status_box)

        self.start_button.clicked.connect(self._on_start_clicked)
        self.stop_button.clicked.connect(self._on_stop_clicked)

    def _on_start_clicked(self) -> None:
        interface = self.interface_combo.currentText()
        if interface.lower() == "auto":
            interface = ""
        self.start_requested.emit(interface)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def _on_stop_clicked(self) -> None:
        self.stop_requested.emit()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def append_activity(self, rows: List[Dict[str, object]]) -> None:
        for row in rows:
            current_rows = self.activity_table.rowCount()
            if current_rows >= config.MAX_TABLE_ROWS:
                self.activity_table.removeRow(0)
            self.activity_table.insertRow(self.activity_table.rowCount())
            for idx, key in enumerate(["timestamp", "mode", "label", "probability", "bytes_in", "bytes_out", "duration"]):
                value = row.get(key, "")
                if key == "probability" and isinstance(value, float):
                    value = f"{value:.2f} %"
                elif isinstance(value, float):
                    value = f"{value:.4f}"
                item = QtWidgets.QTableWidgetItem(str(value))
                if key == "label" and str(row.get("label", "")).lower() != "benign":
                    item.setForeground(QtGui.QColor("red"))
                self.activity_table.setItem(self.activity_table.rowCount() - 1, idx, item)
        self.activity_table.scrollToBottom()

    def log_message(self, message: str) -> None:
        self.status_box.append(message)

    def update_dashboard(self, packets: int, flows: int, alerts: int, capture_active: bool) -> None:
        self.dashboard.update_stats(packets, flows, alerts, capture_active)

    def reset_buttons(self) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)


class CsvPredictionTab(QtWidgets.QWidget):
    file_selected = QtCore.pyqtSignal(Path)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        instructions = QtWidgets.QLabel(
            "Select a CSV file containing the 15 required features. The columns will be aligned automatically."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        self.browse_button = QtWidgets.QPushButton("Select CSV File...")
        layout.addWidget(self.browse_button)

        self.table = QtWidgets.QTableWidget()
        layout.addWidget(self.table)

        self.summary_label = QtWidgets.QLabel()
        layout.addWidget(self.summary_label)

        self.browse_button.clicked.connect(self._on_browse)

    def _on_browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open CSV", str(Path.home()), "CSV Files (*.csv)")
        if path:
            self.file_selected.emit(Path(path))

    def display_results(self, df: pd.DataFrame, result: dict) -> None:
        probabilities = result["probabilities"]
        labels = result["predicted_labels"]
        class_names = result["class_names"]

        self.table.clear()
        self.table.setColumnCount(len(class_names) + 2)
        headers = ["Index", "Label"] + [f"P({name})" for name in class_names]
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(df))

        for idx, label in enumerate(labels):
            self.table.setItem(idx, 0, QtWidgets.QTableWidgetItem(str(idx)))
            label_item = QtWidgets.QTableWidgetItem(str(label))
            if str(label).lower() != "benign":
                label_item.setForeground(QtGui.QColor("red"))
            self.table.setItem(idx, 1, label_item)
            for col, prob in enumerate(probabilities[idx], start=2):
                self.table.setItem(idx, col, QtWidgets.QTableWidgetItem(f"{prob:.4f}"))

        self.table.resizeColumnsToContents()

        summary = {
            "samples": len(df),
            "alerts": int((pd.Series(labels).str.lower() != "benign").sum()),
        }
        self.summary_label.setText(json.dumps(summary, indent=2))


class ManualPredictionTab(QtWidgets.QWidget):
    submit_requested = QtCore.pyqtSignal(dict)

    def __init__(self, feature_names: Iterable[str], parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.feature_names = list(feature_names)
        self.inputs: Dict[str, QtWidgets.QWidget] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.form_layout = QtWidgets.QGridLayout()
        layout.addLayout(self.form_layout)

        row = 0
        for feature in self.feature_names:
            label = QtWidgets.QLabel(feature)
            if feature in {"src_ip", "dest_ip"}:
                widget = QtWidgets.QLineEdit()
                widget.setPlaceholderText("e.g. 192.168.1.10")
            elif feature == "proto":
                widget = QtWidgets.QComboBox()
                widget.addItems(["tcp", "udp", "icmp", "other"])
            else:
                widget = QtWidgets.QDoubleSpinBox()
                widget.setDecimals(6)
                widget.setRange(-1e9, 1e9)
                widget.setValue(0.0)
            self.inputs[feature] = widget
            self.form_layout.addWidget(label, row, 0)
            self.form_layout.addWidget(widget, row, 1)
            row += 1

        buttons_layout = QtWidgets.QHBoxLayout()
        self.submit_button = QtWidgets.QPushButton("Predict")
        self.reset_button = QtWidgets.QPushButton("Reset")
        buttons_layout.addWidget(self.submit_button)
        buttons_layout.addWidget(self.reset_button)
        layout.addLayout(buttons_layout)

        self.result_label = QtWidgets.QLabel("Result: -")
        font = self.result_label.font()
        font.setPointSize(12)
        font.setBold(True)
        self.result_label.setFont(font)
        layout.addWidget(self.result_label)

        self.submit_button.clicked.connect(self._on_submit)
        self.reset_button.clicked.connect(self._on_reset)

    def _on_submit(self) -> None:
        payload: Dict[str, object] = {}
        for feature, widget in self.inputs.items():
            if isinstance(widget, QtWidgets.QLineEdit):
                payload[feature] = widget.text().strip()
            elif isinstance(widget, QtWidgets.QComboBox):
                payload[feature] = widget.currentText()
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                payload[feature] = widget.value()
            else:
                payload[feature] = 0
        self.submit_requested.emit(payload)

    def _on_reset(self) -> None:
        for widget in self.inputs.values():
            if isinstance(widget, QtWidgets.QLineEdit):
                widget.clear()
            elif isinstance(widget, QtWidgets.QComboBox):
                widget.setCurrentIndex(0)
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                widget.setValue(0.0)
        self.result_label.setText("Result: -")

    def set_result(self, label: str, probability: float) -> None:
        text = f"Result: {label} (confidence {probability:.2f}%)"
        self.result_label.setText(text)
        if label.lower() != "benign":
            self.result_label.setStyleSheet("color: red;")
        else:
            self.result_label.setStyleSheet("")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        configure_logging(config.LOG_ROOT_DIR)
        self._tshark_path = ensure_tshark(auto_install=True)
        self.model_manager = ModelManager()
        self.session_manager = SessionManager()
        self.live_controller = LiveCaptureController(
            self.session_manager,
            config.DEFAULT_IDLE_TIMEOUT_SECONDS,
            tshark_path=self._tshark_path,
        )
        self._active_capture = False

        self._build_ui()
        self._connect_signals()
        self._refresh_dashboard()

    def _build_ui(self) -> None:
        self.setWindowTitle(config.APPLICATION_NAME)
        if config.ICON_PATH:
            self.setWindowIcon(QtGui.QIcon(str(config.ICON_PATH)))

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        self.live_tab = LiveCaptureTab()
        self.csv_tab = CsvPredictionTab()
        self.manual_tab = ManualPredictionTab(self.model_manager.feature_names)

        self.tabs.addTab(self.live_tab, "Live Capture")
        self.tabs.addTab(self.csv_tab, "CSV Batch")
        self.tabs.addTab(self.manual_tab, "Single Prediction")

        self._build_menus()

    def _build_menus(self) -> None:
        menu_bar = self.menuBar()
        if menu_bar is None:
            LOGGER.warning("Menu bar not available; skipping menu construction")
            return

        file_menu = menu_bar.addMenu("File")
        export_action = QtWidgets.QAction("Export Session Predictions", self)
        export_action.triggered.connect(self._export_session_predictions)
        if file_menu is not None:
            file_menu.addAction(export_action)
        exit_action = QtWidgets.QAction("Exit", self)
        exit_action.triggered.connect(self._handle_exit)
        if file_menu is not None:
            file_menu.addAction(exit_action)

        tools_menu = menu_bar.addMenu("Tools")
        view_logs_action = QtWidgets.QAction("View Logs", self)
        view_logs_action.triggered.connect(self._open_logs_folder)
        if tools_menu is not None:
            tools_menu.addAction(view_logs_action)

        open_notebook_action = QtWidgets.QAction("Developer Tools: Open Notebook", self)
        open_notebook_action.triggered.connect(self._open_notebook)
        if tools_menu is not None:
            tools_menu.addAction(open_notebook_action)

        help_menu = menu_bar.addMenu("Help")
        about_action = QtWidgets.QAction("About", self)
        about_action.triggered.connect(self._show_about)
        if help_menu is not None:
            help_menu.addAction(about_action)

    def _connect_signals(self) -> None:
        self.live_tab.start_requested.connect(self._start_live_capture)
        self.live_tab.stop_requested.connect(self._stop_live_capture)

        self.live_controller.flow_ready.connect(self._on_live_flow_ready)
        self.live_controller.stats_updated.connect(self._on_live_stats_update)
        self.live_controller.dependency_missing.connect(self._on_dependency_missing)
        self.live_controller.error_occurred.connect(self._on_live_error)

        self.csv_tab.file_selected.connect(self._on_csv_selected)
        self.manual_tab.submit_requested.connect(self._on_manual_submit)

    def _start_live_capture(self, interface: str) -> None:
        selected_interface: Optional[str] = interface if interface else None
        host_ips = host_ip_addresses()
        self.live_controller.start(interface=selected_interface, host_ips=host_ips)
        self._active_capture = True
        self.session_manager.logger.info("Live capture requested on %s", selected_interface or "auto")
        self._refresh_dashboard()

    def _stop_live_capture(self) -> None:
        self.live_controller.stop()
        self._active_capture = False
        self.session_manager.logger.info("Live capture stopped by user")
        self._refresh_dashboard()

    def _on_live_flow_ready(self, df: pd.DataFrame) -> None:
        result = self.model_manager.predict(df)
        probabilities = np.asarray(result["probabilities"])  # type: ignore[arg-type]
        labels: List[str] = list(result["predicted_labels"])  # type: ignore[list-item]
        class_names: Sequence[str] = list(result["class_names"])  # type: ignore[list-item]
        prepared: pd.DataFrame = result["prepared_features"]  # type: ignore[assignment]

        rows_for_table = []

        for idx, label in enumerate(labels):
            probability_vector = {class_names[i]: float(probabilities[idx][i]) for i in range(len(class_names))}
            best_prob = max(probability_vector.values()) * 100
            features = {col: float(prepared.iloc[idx][col]) for col in prepared.columns}
            self.session_manager.record_prediction(
                mode="Live",
                source="Capture",
                label=str(label),
                probability=best_prob,
                probability_by_class=probability_vector,
                features=features,
            )
            rows_for_table.append(
                {
                    "timestamp": datetime.utcnow().strftime("%H:%M:%S"),
                    "mode": "Live",
                    "label": label,
                    "probability": best_prob,
                    "bytes_in": features.get("bytes_in", 0.0),
                    "bytes_out": features.get("bytes_out", 0.0),
                    "duration": features.get("duration", 0.0),
                }
            )

        self.live_tab.append_activity(rows_for_table)
        self._refresh_dashboard()

    def _on_live_stats_update(self, payload: dict) -> None:
        packets = payload.get("packets", 0)
        alerts = self.session_manager.stats.alerts_triggered
        self.live_tab.update_dashboard(packets, self.session_manager.stats.flows_analyzed, alerts, self._active_capture)

    def _on_dependency_missing(self, message: str) -> None:
        self.live_tab.log_message(message)
        QtWidgets.QMessageBox.information(self, "Live Capture", message)

    def _on_live_error(self, message: str) -> None:
        self.live_tab.log_message(f"Error: {message}")
        QtWidgets.QMessageBox.warning(self, "Live Capture", message)
        self.live_tab.reset_buttons()
        self._active_capture = False
        self._refresh_dashboard()

    def _on_csv_selected(self, path: Path) -> None:
        try:
            df, result = predict_from_csv(path, self.model_manager)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "CSV Prediction", str(exc))
            return

        self.csv_tab.display_results(df, result)
        probabilities = np.asarray(result["probabilities"])  # type: ignore[arg-type]
        class_names: Sequence[str] = list(result["class_names"])  # type: ignore[list-item]
        prepared: pd.DataFrame = result["prepared_features"]  # type: ignore[assignment]
        labels: List[str] = list(result["predicted_labels"])  # type: ignore[list-item]
        for idx, label in enumerate(labels):
            probability_vector = {class_names[i]: float(probabilities[idx][i]) for i in range(len(class_names))}
            best_prob = max(probability_vector.values()) * 100
            features = {col: float(prepared.iloc[idx][col]) for col in prepared.columns}
            self.session_manager.record_prediction(
                mode="CSV",
                source=str(path.name),
                label=str(label),
                probability=best_prob,
                probability_by_class=probability_vector,
                features=features,
            )
        self._refresh_dashboard()

    def _on_manual_submit(self, payload: Dict[str, object]) -> None:
        try:
            result = predict_from_manual_input(payload, self.model_manager)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Manual Prediction", str(exc))
            return

        probabilities = np.asarray(result["probabilities"])  # type: ignore[arg-type]
        class_names: Sequence[str] = list(result["class_names"])  # type: ignore[list-item]
        labels: List[str] = list(result["predicted_labels"])  # type: ignore[list-item]
        label = labels[0]
        probability_vector = {class_names[i]: float(probabilities[0][i]) for i in range(len(class_names))}
        best_prob = max(probability_vector.values()) * 100
        self.manual_tab.set_result(str(label), best_prob)

        prepared: pd.DataFrame = result["prepared_features"]  # type: ignore[assignment]
        features = {col: float(prepared.iloc[0][col]) for col in prepared.columns}
        self.session_manager.record_prediction(
            mode="Manual",
            source="Form",
            label=str(label),
            probability=best_prob,
            probability_by_class=probability_vector,
            features=features,
        )
        self._refresh_dashboard()

    def _refresh_dashboard(self) -> None:
        stats = self.session_manager.stats
        self.live_tab.update_dashboard(stats.packets_captured, stats.flows_analyzed, stats.alerts_triggered, self._active_capture)

    def _export_session_predictions(self) -> None:
        try:
            path = self.session_manager.export_predictions_csv()
            QtWidgets.QMessageBox.information(self, "Export", f"Predictions exported to {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export", str(exc))

    def _open_logs_folder(self) -> None:
        try:
            safe_open_path(self.session_manager.session_folder)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Logs", str(exc))

    def _open_notebook(self) -> None:
        try:
            safe_open_path(config.NOTEBOOK_PATH)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Developer Tools", str(exc))

    def _show_about(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "About",
            f"{config.APPLICATION_NAME}\nVersion {config.VERSION}\nPowered by PyQt5 and XGBoost.",
        )

    def _handle_exit(self) -> None:
        self.close()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - GUI event
        self.live_controller.stop()
        self.session_manager.close()
        super().closeEvent(event)


def run() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setOrganizationName("LUFLOW")
    app.setApplicationName(config.APPLICATION_NAME)
    window = MainWindow()
    window.resize(1000, 700)
    window.show()
    sys.exit(app.exec_())
