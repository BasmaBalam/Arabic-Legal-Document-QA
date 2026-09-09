import uuid
import logging
from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
from qdrant_client import models, QdrantClient
from arabic_legal_document_qa.models.db_schemes import RetrievedDocument
from qdrant_client.http.models import Distance
from qdrant_client.models import PointStruct
from typing import List

class QdrantDBProvider(VectorDBInterface):
    def __init__(self, db_path: str, distance_method: str):

        self.client = None
        self.db_path = db_path


        distance_map = {
            "Cosine": Distance.COSINE,
            "Dot": Distance.DOT,
            "Euclid": Distance.EUCLID,
            "Manhattan": Distance.MANHATTAN
        }

        self.distance_method = distance_map.get(distance_method, Distance.COSINE)

        self.logger = logging.getLogger(__name__)

    def connect(self)-> None:
        """ Connect to the local Qdrant database. """
        self.client = QdrantClient(path = self.db_path)
    
    def disconnect(self)-> None:
        """ Disconnect from the Qdrant database. """
        self.client = None
        
    def is_collection_axisted(self, collection_name: str) -> bool:
        """ 
        Check whether a Qdrant collection exists.
        
        Args: 
            collection_name: Name of the collection to check. 
            
        Returns: 
            ``True`` if the collection exists, otherwise ``False``. 
        """
        return self.client.collection_exists(collection_name = collection_name)

    def list_all_collections(self) -> List:
        """ 
        Retrieve all collections from Qdrant.
            
        Returns:
            Qdrant collection information.
        """
        return self.client.get_collections()

    def get_collection_info(self, collection_name: str) -> dict:
        """ 
        Retrieve information about a Qdrant collection.
        
        Args:
            collection_name: Name of the collection. 
            
        Returns: 
            Qdrant collection configuration and information.
        
        """
        return self.client.get_collection(collection_name= collection_name)

    def delete_collection(self, collection_name: str)-> bool:
        """ 
        Delete a Qdrant collection if it exists.
        
        Args: 
            collection_name: Name of the collection to delete.
        
        Returns: 
            The result of the Qdrant delete operation, or ``False``
            if the collection does not exist. 
        """
        if self.is_collection_axisted(collection_name=collection_name):
            return self.client.delete_collection(collection_name=collection_name)

    def create_collection(self, collection_name: str, embedding_size: int, do_reset: bool = False)-> bool:
        """ 
        Create a Qdrant vector collection. 
        
        Args: 
            collection_name: Name of the collection to create.
            embedding_size: Dimension of the stored embedding vectors.
            do_reset: Whether to delete the existing collection first.
            
        Returns: ``True`` if the collection was created, otherwise ``False``. 
        
        """
        if do_reset:
            self.delete_collection(collection_name=collection_name)
        
        if not self.is_collection_axisted(collection_name=collection_name):

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=self.distance_method
                )
            )

            return True
        
        return False

    def insert_one(self, collection_name: str, text: str, vector: list,
                   metadata: dict = None, record_id: str = None)-> bool: 
                   
        """ 
        Insert one document and its embedding into Qdrant. 
        
        Args: 
            collection_name: Target Qdrant collection.
            text: Text associated with the embedding.
            vector: Embedding vector.
            metadata: Optional document metadata. 
            record_id: Optional unique record identifier. A UUID is generated when omitted. 
            
        Returns: 
            ``True`` if the record was inserted successfully, otherwise ``False``. 
        """

        if not self.is_collection_axisted(collection_name=collection_name):
            self.logger.error(f"Can not insert record to non existed collection {collection_name}")
            return False

        # Generate a UUID if record_id is not provided
        if record_id is None:
            record_id = str(uuid.uuid4())

        try:
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=record_id,
                        vector=vector,
                        payload={
                            "text": text,
                            "metadata": metadata
                        }
                    )
                ]
            )
        except Exception as e:
            self.logger.error(f"Error while inserting record {e}")
            return False

        return True

    def insert_many(self, collection_name: str, texts: list, vectors: list,
                    metadata: list = None, record_ids: list = None, batch_size: int = 50)-> bool:
        """ 
        Insert multiple documents into Qdrant in batches.
        
        Args: 
            collection_name: Target Qdrant collection.
            texts: List of document texts. 
            vectors: List of embedding vectors. 
            metadata: Optional metadata for each document.
            record_ids: Optional unique identifiers for each document.
            batch_size: Number of records inserted per Qdrant request.
            
        Returns: 
            ``True`` if all records were inserted successfully, otherwise ``False``. 
        """
        if metadata is None:
            metadata = [None] * len(texts)

        if record_ids is None:
            record_ids = list(range(0, len(texts)))
        
        for i in range(0, len(texts), batch_size):

            batch_end = i + batch_size

            batch_record_ids = record_ids[i: batch_end]
            batch_text = texts[i: batch_end]
            batch_vectors = vectors[i: batch_end]
            batch_metadata = metadata[i: batch_end]

            batch_records = [
                PointStruct(
                    id=batch_record_ids[x],
                    vector=batch_vectors[x],
                    payload={
                        "text": batch_text[x],
                        "metadata": batch_metadata[x]
                    }
                )
                for x in range(len(batch_text))
            ]

            try:
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch_records
                )
            except Exception as e:
                self.logger.error(f"Error while inserting batch {e}")
                return False
        
        return True

    def search_by_vector(self, collection_name: str, vector: list, limit: int = 5)-> Optional[list[RetrievedDocument]]:
        
        """ 
        Search Qdrant for documents similar to the given vector.
        
        Args: 
            collection_name: Name of the collection to search.
            vector: Query embedding vector. 
            limit: Maximum number of results to return. 
            
        Returns: 
            A list of ``RetrievedDocument`` objects containing 
            the matching documents and their similarity scores, 
            or ``None`` when no results are found. 
        """
        results = self.client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit,
            with_payload=True  
        )
        
        if not results or not results.points:
            return None

        return [
            RetrievedDocument(
                score=point.score,
                text=point.payload.get("text", "")
            )
            for point in results.points
        ]
