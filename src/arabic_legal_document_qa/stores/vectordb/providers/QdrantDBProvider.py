import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance
from qdrant_client.models import PointStruct

from arabic_legal_document_qa.models.db_schemes import RetrievedDocument

from ..VectorDBInterface import VectorDBInterface


class QdrantDBProvider(VectorDBInterface):
    def __init__(
        self,
        db_path: str,
        distance_method: str,
        host: str | None = None,
        port: int = 6333,
    ):
        self.client = None
        self.db_path = db_path
        self.host = host
        self.port = port

        distance_map = {
            "Cosine": Distance.COSINE,
            "Dot": Distance.DOT,
            "Euclid": Distance.EUCLID,
            "Manhattan": Distance.MANHATTAN,
        }

        self.distance_method = distance_map.get(distance_method, Distance.COSINE)

        self.logger = logging.getLogger(__name__)

    def connect(self) -> None:
        """Connect to Qdrant server or local storage."""
        if self.host:
            self.client = QdrantClient(
                host=self.host,
                port=self.port,
            )
        else:
            if self.db_path is None:
                raise ValueError(
                    "db_path is required when QDRANT_HOST is not configured."
                )

            self.client = QdrantClient(path=self.db_path)

    def disconnect(self) -> None:
        """Disconnect from the Qdrant database."""
        self.client = None

    def is_collection_axisted(self, collection_name: str) -> bool:
        """
        Check whether a Qdrant collection exists.

        Args:
            collection_name: Name of the collection to check.

        Returns:
            ``True`` if the collection exists, otherwise ``False``.
        """
        return self.client.collection_exists(collection_name=collection_name)

    def list_all_collections(self) -> list:
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
        return self.client.get_collection(collection_name=collection_name)

    def delete_collection(self, collection_name: str) -> bool:
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

    def create_collection(
        self, collection_name: str, embedding_size: int, do_reset: bool = False
    ) -> bool:
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
                    size=embedding_size, distance=self.distance_method
                ),
            )

            return True

        return False

    def insert_one(
        self,
        collection_name: str,
        text: str,
        vector: list,
        metadata: dict | None = None,
        record_id: str | None = None,
    ) -> bool:
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
            self.logger.error(
                f"Can not insert record to non existed collection {collection_name}"
            )
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
                        payload={"text": text, "metadata": metadata},
                    )
                ],
            )
        except Exception:
            self.logger.exception("Error while inserting record")
            return False

        return True

    def insert_many(
        self,
        collection_name: str,
        texts: list,
        vectors: list,
        metadata: list | None = None,
        record_ids: list | None = None,
        batch_size: int = 50,
    ) -> bool:
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
            record_ids = list(range(len(texts)))

        for i in range(0, len(texts), batch_size):
            batch_end = i + batch_size

            batch_record_ids = record_ids[i:batch_end]
            batch_text = texts[i:batch_end]
            batch_vectors = vectors[i:batch_end]
            batch_metadata = metadata[i:batch_end]

            batch_records = [
                PointStruct(
                    id=batch_record_ids[x],
                    vector=batch_vectors[x],
                    payload={"text": batch_text[x], "metadata": batch_metadata[x]},
                )
                for x in range(len(batch_text))
            ]

            try:
                self.client.upsert(
                    collection_name=collection_name, points=batch_records
                )
            except Exception:
                self.logger.exception("Error while inserting batch")
                return False

        return True

    def _build_metadata_filter(self, metadata_filter: dict | None) -> Any | None:
        """
        Build a Qdrant metadata filter from a dictionary.

        Args:
            metadata_filter: Optional dictionary containing metadata
            fields and their expected values.

        Returns:
            A Qdrant filter containing the specified conditions, or
            ``None`` when no valid conditions are provided.
        """
        if not metadata_filter:
            return None

        conditions = []

        for key, value in metadata_filter.items():
            if value is None:
                continue

            conditions.append(
                models.FieldCondition(
                    key=f"metadata.{key}",
                    match=models.MatchValue(value=value),
                )
            )

        if not conditions:
            return None

        return models.Filter(must=conditions)

    def search_by_vector(
        self,
        collection_name: str,
        vector: list,
        limit: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[RetrievedDocument] | None:
        """
        Search for documents using vector similarity.
        The provided vector is compared against vectors stored in the specified Qdrant collection.
        Optional metadata conditions can be applied to restrict the search results.

        Args:
            collection_name: Target Qdrant collection.
            vector: Query embedding vector.
            limit: Maximum number of documents to retrieve.
            metadata_filter: Optional metadata conditions used to filter the search results.

        Returns:
            A list of retrieved documents with their similarity scores, text, and metadata.
            Returns ``None`` when no matching documents are found.
        """
        qdrant_filter = self._build_metadata_filter(metadata_filter)

        results = self.client.query_points(
            collection_name=collection_name,
            query=vector,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
        )

        if not results or not results.points:
            return None

        return [
            RetrievedDocument(
                score=point.score,
                text=point.payload.get("text", ""),
                metadata=point.payload.get("metadata"),
            )
            for point in results.points
        ]

    def count_records(self, collection_name: str) -> int:
        """
        Count the number of records stored in a collection.

        Args:
            collection_name: Name of the collection whose records
                should be counted.

        Returns:
            The total number of records stored in the collection.
            Returns ``0`` if the collection does not exist.
        """

        if not self.is_collection_axisted(collection_name=collection_name):
            return 0
        return self.client.count(collection_name=collection_name).count
