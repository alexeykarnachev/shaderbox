from typing import Any

from shaderbox.exporters.base import Exporter
from shaderbox.exporters.integrations import IntegrationsStore


class ExporterRegistry:
    def __init__(self) -> None:
        self._exporters: dict[str, Exporter] = {}
        self.active_id: str = ""

    def register(self, exporter: Exporter) -> None:
        self._exporters[exporter.exporter_id] = exporter
        # Only an available exporter may become the default active target.
        if not self.active_id and exporter.is_available:
            self.active_id = exporter.exporter_id

    def set_active(self, exporter_id: str) -> None:
        exporter = self._exporters.get(exporter_id)
        if exporter is not None and exporter.is_available:
            self.active_id = exporter_id

    def get_active(self) -> Exporter | None:
        return self._exporters.get(self.active_id)

    def ids(self) -> list[str]:
        return list(self._exporters.keys())

    def available_ids(self) -> list[str]:
        return [eid for eid, e in self._exporters.items() if e.is_available]

    def all(self) -> list[Exporter]:
        return list(self._exporters.values())

    def get(self, exporter_id: str) -> Exporter | None:
        return self._exporters.get(exporter_id)

    def set_integrations(self, store: IntegrationsStore) -> None:
        for exporter in self._exporters.values():
            exporter.set_integrations(store)

    def rebind(self, exporter_settings: dict[str, dict[str, Any]]) -> None:
        for exporter_id, exporter in self._exporters.items():
            exporter.rebind(exporter_settings.get(exporter_id, {}))

    def release(self) -> None:
        for exporter in self._exporters.values():
            exporter.release()
