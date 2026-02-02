from pathlib import Path
from typing import Dict, List
import json


class ConfigManager:
    """Manages configuration persistence for lanes and folders."""
    
    def __init__(self, persist_directory: str):
        self.persist_directory = Path(persist_directory)
        self.lanes_config_path = self.persist_directory / "lanes_config.json"
        self.lane_folders_path = self.persist_directory / "lane_folders.json"
    
    def load_lanes(self) -> Dict[str, List[str]]:
        """Load lane configuration."""
        if self.lanes_config_path.exists():
            with open(self.lanes_config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def save_lanes(self, lanes: Dict[str, List[str]]) -> None:
        """Save lane configuration."""
        self.lanes_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lanes_config_path, 'w') as f:
            json.dump(lanes, f, indent=2)
    
    def load_lane_folders(self) -> Dict[str, str]:
        """Load lane folder mappings."""
        if self.lane_folders_path.exists():
            with open(self.lane_folders_path, 'r') as f:
                return json.load(f)
        return {}
    
    def save_lane_folders(self, lane_folders: Dict[str, str]) -> None:
        """Save lane folder mappings."""
        self.lane_folders_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lane_folders_path, 'w') as f:
            json.dump(lane_folders, f, indent=2)
