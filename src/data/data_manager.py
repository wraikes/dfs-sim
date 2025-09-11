"""Data manager for organizing DFS data files."""

from pathlib import Path
from typing import List
import logging

from ..models import Player
from .json_extractor import JsonExtractor

logger = logging.getLogger(__name__)


class DataManager:
    """Manage DFS data organization and loading."""
    
    # Linestar API configuration
    BASE_URL = "https://www.linestarapp.com/DesktopModules/DailyFantasyApi/API/Fantasy/GetSalariesV4"
    
    SPORT_IDS = {
        'nfl': 1,
        'nba': 2,
        'mlb': 3,
        'pga': 5,
        'nhl': 6,
        'mma': 8,
        'nascar': 9
    }
    
    SITE_IDS = {
        'dk': 1,      # DraftKings
        'fd': 2       # FanDuel
    }
    
    def __init__(self, base_dir: str = "data"):
        """
        Initialize data manager.
        
        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def get_json_path(self, sport: str, site: str, period_id: str) -> Path:
        """Get path for JSON data file."""
        sport_dir = self.base_dir / sport.lower() / period_id
        sport_dir.mkdir(parents=True, exist_ok=True)
        return sport_dir / f"{site}_{period_id}.json"
    
    def generate_download_url(self, sport: str, site: str, period_id: str) -> str:
        """
        Generate URL and create empty data files.
        
        Args:
            sport: Sport name (mma, nascar, nfl)
            site: DFS site (dk for DraftKings, fd for FanDuel)
            period_id: Period/contest ID
            
        Returns:
            URL to download JSON data
        """
        sport = sport.lower()
        site = site.lower()
        
        if sport not in self.SPORT_IDS:
            raise ValueError(f"Invalid sport: {sport}. Must be one of {list(self.SPORT_IDS.keys())}")
        if site not in self.SITE_IDS:
            raise ValueError(f"Invalid site: {site}. Must be one of {list(self.SITE_IDS.keys())}")
        
        sport_id = self.SPORT_IDS[sport]
        site_id = self.SITE_IDS[site]
        
        # Build Linestar URL
        url = f"{self.BASE_URL}?sport={sport_id}&site={site_id}&periodId={period_id}"
        
        # Create empty JSON file
        json_path = self.get_json_path(sport, site, period_id)
        json_path.write_text('{}')
        
        print(f"\n" + "="*60)
        print(f"📥 LINESTAR DATA DOWNLOAD - {sport.upper()} {site.upper()} PERIOD {period_id}")
        print("="*60)
        print(f"🌐 URL: {url}")
        print(f"📁 JSON file: {json_path}")
        print("="*60)
        print("INSTRUCTIONS:")
        print("1. Copy the URL above")
        print("2. Paste in your browser")
        print("3. Copy the entire JSON response")
        print(f"4. Paste into: {json_path}")
        print("="*60)
        
        return url
    
    def load_json_data(self, sport: str, site: str, period_id: str) -> List[Player]:
        """
        Load JSON data for a sport/site/period.
        
        Args:
            sport: Sport name
            site: Site name
            period_id: Period ID
            
        Returns:
            List of Player objects
        """
        json_path = self.get_json_path(sport, site, period_id)
        
        if not json_path.exists():
            raise FileNotFoundError(
                f"JSON file not found at {json_path}\n"
                f"Please download and save the JSON data first."
            )
        
        extractor = JsonExtractor(sport)
        players = extractor.load_json_file(str(json_path))
        
        logger.info(f"Loaded {len(players)} players from {json_path}")
        return players
    
