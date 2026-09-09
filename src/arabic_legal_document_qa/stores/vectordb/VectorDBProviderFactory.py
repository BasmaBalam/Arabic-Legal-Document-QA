from .providers import QdrantDBProvider
from .VectorDBEnums import VectorDBEnums
from arabic_legal_document_qa.controllers.BaseController import BaseController

class VectorDBProviderFactory:
    """ 
    Factory for creating configured vector database providers.
    The factory selects the appropriate vector database implementation based on the provider name.

    Attributes:
        config: Application configuration containing vector database settings. 
        base_controller: Controller used for resolving database paths. 
    """

    def __init__(self, config):
        """ 
        Initialize the vector database provider factory.
        
        Args:
            config: Application configuration object containing vector database settings, 
            such as the database path and distance method.
        """
        
        self.config = config
        self.base_controller = BaseController()

    def create(self, provider)-> Optional[VectorDBInterface]:
        """ 
        Create a vector database provider.
        
        Args: 
            provider: Name of the vector database provider to create. 
            
        Returns: 
            A configured implementation of ``VectorDBInterface`` when the provider is supported, otherwise ``None``.
            
        """

        if provider == VectorDBEnums.QDRANT.value:
            db_path = self.base_controller.get_database_path(db_name= self.config.VECTOR_DB_PATH)

            return QdrantDBProvider(
                db_path= db_path,
                distance_method= self.config.VECTOR_DB_DISTANCE_METHOD,
            )
        
        return None
    