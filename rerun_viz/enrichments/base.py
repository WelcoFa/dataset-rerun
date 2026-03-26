from __future__ import annotations

from abc import ABC, abstractmethod


class Enrichment(ABC):
    name: str

    @abstractmethod
    def applies_to(self, dataset_name: str) -> bool:
        ...

    @abstractmethod
    def apply(self, packet, ctx, config):
        ...

