from __future__ import annotations

from abc import ABC, abstractmethod


class DatasetAdapter(ABC):
    name: str
    base: str
    viewer_name: str

    @abstractmethod
    def load(self):
        ...

    @abstractmethod
    def log_static(self):
        ...

    @abstractmethod
    def frames(self):
        ...

    @abstractmethod
    def log_panels(self, panels):
        ...

    def create_blueprint(self):
        return None

    def close(self):
        return None
