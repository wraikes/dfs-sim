"""DFS data processors for different sports."""

from .base import BaseDataProcessor
from .mma import MMADataProcessor
from .nascar import NASCARDataProcessor
from .nfl import NFLDataProcessor

from ..models.site import SiteCode


def create_processor(sport: str, pid: str, site: SiteCode = SiteCode.DK) -> BaseDataProcessor:
    """Factory function to create sport-specific processor."""
    processors = {
        'mma': MMADataProcessor,
        'nascar': NASCARDataProcessor,
        'nfl': NFLDataProcessor,
        # 'nba': NBADataProcessor,  # Future implementation
    }

    processor_class = processors.get(sport.lower())
    if not processor_class:
        raise ValueError(f"Sport '{sport}' not yet supported")

    return processor_class(sport, pid, site)


__all__ = [
    'BaseDataProcessor',
    'MMADataProcessor',
    'NASCARDataProcessor',
    'NFLDataProcessor',
    'create_processor'
]