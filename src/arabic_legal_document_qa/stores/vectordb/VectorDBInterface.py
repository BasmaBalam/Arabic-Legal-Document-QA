from abc import ABC, abstractmethod

from arabic_legal_document_qa.models.db_schemes import RetrievedDocument


class VectorDBInterface(ABC):
    """
    Abstract interface for vector database providers.

    Implementations of this interface are responsible for
    managing vector collections, inserting documents and
    their embeddings, and performing similarity searches.
    """

    @abstractmethod
    def connect(self):
        """
        Establish a connection to the vector database.
        Returns: None.
        """

    @abstractmethod
    def disconnect(self):
        """
        Close the connection to the vector database.
        Returns: None.
        """

    @abstractmethod
    def is_collection_axisted(self, collection_name: str) -> bool:
        """
        Check whether a collection exists in the vector database.

        Args:
            collection_name: Name of the collection to check.

        Returns:
            ``True`` if the collection exists, otherwise ``False``.
        """

    @abstractmethod
    def list_all_collections(self) -> list:
        """
        Retrieve information about all collections in the database.

        Returns:
            A collection list or collection information returned by the underlying vector database provider.
        """

    @abstractmethod
    def get_collection_info(self, collection_name: str) -> dict:
        """
        Retrieve information about a specific collection.
        Args:
            collection_name: Name of the collection.

        Returns:
            Collection configuration and metadata returned by the vector database provider.
        """

    @abstractmethod
    def delete_collection(self, collection_name: str):
        """
        Delete a collection from the vector database.

        Args:
            collection_name: Name of the collection to delete.

        Returns:
            ``True`` when the collection is deleted.
            If the collection does not exist, the implementation
            may return ``False``.
        """

    @abstractmethod
    def create_collection(
        self, collection_name: str, embedding_size: int, do_reset: bool = False
    ):
        """
        Create a vector collection.

        Args:
            collection_name: Name of the collection to create.
            embedding_size: Dimension of the vectors that will be stored.
            do_reset: If ``True``, delete the existing collection before creating it.

        Returns:
            ``True`` if a new collection was created, otherwise ``False`` when the collection already exists.
        """

    @abstractmethod
    def insert_one(
        self,
        collection_name: str,
        text: str,
        vector: list,
        metadata: dict | None = None,
        record_id: str | None = None,
    ):
        """
        Insert a single document and its embedding into a collection.

        Args:
            collection_name: Name of the target collection.
            text: Text associated with the embedding vector.
            vector: Embedding vector for the text.
            metadata: Optional metadata associated with the document.
            record_id: Optional unique identifier for the record. If not provided, the implementation generates an identifier.

        Returns: ``True`` if the document was successfully inserted, otherwise ``False``.

        """

    @abstractmethod
    def insert_many(
        self,
        collection_name: str,
        texts: list,
        vectors: list,
        metadata: list | None = None,
        record_ids: list | None = None,
        batch_size: int = 50,
    ):
        """
        Insert multiple documents and their embedding vectors.

        Args:
            collection_name: Name of the target collection.
            texts: List of document texts.
            vectors: List of embedding vectors corresponding to ``texts``.
            metadata: Optional list of metadata dictionaries corresponding to each document.
            record_ids: Optional list of unique record identifiers. If not provided, the implementation generates record identifiers.
            batch_size: Number of records to insert in each database batch.

        Returns:
            ``True`` if all batches were successfully inserted, otherwise ``False``.

        """

    @abstractmethod
    def search_by_vector(
        self,
        collection_name: str,
        vector: list,
        limit: int,
        metadata_filter: dict | None = None,
    ) -> list[RetrievedDocument]:
        """
        Search a collection using vector similarity.

        Args:
            collection_name: Name of the collection to search.
            vector: Query embedding vector.
            limit: Maximum number of matching documents to return.

        Returns:
            A list of ``RetrievedDocument`` objects, or ``None`` if no matching documents are found.
        """

    @abstractmethod
    def count_records(self, collection_name: str) -> int:
        """
        Count the number of records stored in a collection.

        Args:
            collection_name: Name of the collection whose records should be counted.

        Returns:
            The total number of records currently stored in the collection.
        """
