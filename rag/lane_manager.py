from typing import Dict, List, Optional


class LaneManager:
    """Manages lanes and their collections."""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.lanes: Dict[str, List[str]] = config_manager.load_lanes()
        self.lane_folders: Dict[str, str] = config_manager.load_lane_folders()
    
    def create_lane(self, lane_name: str) -> None:
        """Create a new lane."""
        if lane_name in self.lanes:
            print(f"⚠️ Lane '{lane_name}' already exists")
            return
        
        self.lanes[lane_name] = []
        self.config_manager.save_lanes(self.lanes)
        print(f"✅ Created lane: {lane_name}")
    
    def lane_exists(self, lane_name: str) -> bool:
        """Check if lane exists."""
        return lane_name in self.lanes
    
    def add_collection(self, lane_name: str, collection_name: str) -> None:
        """Add a collection to a lane."""
        if lane_name not in self.lanes:
            raise ValueError(f"Lane '{lane_name}' does not exist")
        
        if collection_name in self.lanes[lane_name]:
            print(f"⚠️ Collection '{collection_name}' already exists in lane '{lane_name}'")
            return
        
        self.lanes[lane_name].append(collection_name)
        self.config_manager.save_lanes(self.lanes)
        print(f"✅ Added collection '{collection_name}' to lane '{lane_name}'")
    
    def collection_exists(self, lane_name: str, collection_name: str) -> bool:
        """Check if collection exists in lane."""
        return lane_name in self.lanes and collection_name in self.lanes[lane_name]
    
    def get_collections(self, lane_name: str) -> List[str]:
        """Get all collections in a lane."""
        return self.lanes.get(lane_name, [])
    
    def set_lane_folder(self, lane_name: str, folder_path: str) -> None:
        """Set folder path for a lane."""
        if lane_name not in self.lanes:
            raise ValueError(f"Lane '{lane_name}' does not exist")
        
        self.lane_folders[lane_name] = folder_path
        self.config_manager.save_lane_folders(self.lane_folders)
        print(f"✅ Set folder for lane '{lane_name}': {folder_path}")
    
    def get_lane_folder(self, lane_name: str) -> Optional[str]:
        """Get folder path for a lane."""
        return self.lane_folders.get(lane_name)
    
    def list_all(self) -> None:
        """List all lanes and their collections."""
        if not self.lanes:
            print("⚠️ No lanes exist")
            return
        
        print("\n📋 Lanes:")
        for lane_name, collections in self.lanes.items():
            folder = self.get_lane_folder(lane_name)
            folder_info = f" (folder: {folder})" if folder else ""
            print(f"  • {lane_name}{folder_info}")
            if collections:
                for collection in collections:
                    print(f"    - {collection}")
            else:
                print(f"    (no collections)")
