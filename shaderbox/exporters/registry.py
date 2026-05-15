from typing import Any

from shaderbox.exporters.base import Exporter


class ExporterRegistry:
    def __init__(self) -> None:
        self._exporters: dict[str, Exporter] = {}
        self.active_id: str = ""

    def register(self, exporter: Exporter) -> None:
        self._exporters[exporter.exporter_id] = exporter
        if not self.active_id:
            self.active_id = exporter.exporter_id

    def set_active(self, exporter_id: str) -> None:
        if exporter_id in self._exporters:
            self.active_id = exporter_id

    def get_active(self) -> Exporter | None:
        return self._exporters.get(self.active_id)

    def ids(self) -> list[str]:
        return list(self._exporters.keys())

    def get(self, exporter_id: str) -> Exporter | None:
        return self._exporters.get(exporter_id)

    def rebind(self, exporter_settings: dict[str, dict[str, Any]]) -> None:
        for exporter_id, exporter in self._exporters.items():
            exporter.rebind(exporter_settings.get(exporter_id, {}))

    def release(self) -> None:
        for exporter in self._exporters.values():
            exporter.release()
