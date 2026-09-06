from pathlib import Path

from arabic_legal_document_qa.configs.config import get_settings


class BaseController:
    """Base controller providing application settings and project directories.
    This class serves as the base class for application controllers.
    It initializes the application settings and resolves the project's
    root directory and data directories. 
    
    Attributes:
        app_settings: Application configuration.
        project_root: Root directory of the project.
        data_dir: Directory containing application data.
        data_processed_dir: Directory containing processed data artifacts.
    """

    def __init__(self):
        self.app_settings = get_settings()

        self.project_root = Path(__file__).resolve().parents[3]
        self.data_dir = self.project_root / "data"
        self.data_processed_dir = self.data_dir / "processed"