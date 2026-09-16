from langchain_core.documents import Document

from arabic_legal_document_qa.models.db_schemes import RetrievedDocument
from arabic_legal_document_qa.stores.llm.LLMEnums import DocumentTypeEnum

from .BaseController import BaseController


class VectorDBController(BaseController):
    """
    Controller for embedding, storing, and searching documents in a vector database.
    This controller handles the interaction between the application and the vector database.
    It generates embeddings for documents using the configured embedding client, stores
    the documents and their vectors in the configured collection, and performs similarity
    searches for user queries.

    Attributes:
        vectordb_client: Client used to interact with the vector database.
        embedding_client: Client used to generate text embeddings.
        COLLECTION_NAME: Name of the vector database collection used to store
        Egyptian Civil Code documents.
    """

    def __init__(self, vectordb_client, embedding_client):
        """
        Initialize the vector database controller.

        Args:
            vectordb_client: Client used to create collections, insert documents,
            search by vector, and count stored records.

            embedding_client: Client used to generate embeddings for passages and queries.
        """
        super().__init__()
        self.vectordb_client = vectordb_client
        self.embedding_client = embedding_client
        self.COLLECTION_NAME = "egyptian_civil_code"

    def add_documents(self, documents: list[Document], do_reset: bool = False) -> int:
        """
        Generate embeddings and add documents to the vector database. Each document's
        page content is embedded using the passage embedding type. The resulting vectors,
        text, and metadata are then stored in the configured vector database collection.

        Args:
            documents: List of LangChain documents to add to the collection.
            do_reset: Whether to reset the collection before inserting the documents.

        Returns:
            The number of documents added. Returns 0 if the input list is empty.
        """
        if not documents:
            return 0

        texts = [doc.page_content for doc in documents]
        metadata = [doc.metadata for doc in documents]

        vectors = [
            self.embedding_client.embed_text(
                text=text, document_type=DocumentTypeEnum.PASSAGE.value
            )
            for text in texts
        ]

        _ = self.vectordb_client.create_collection(
            collection_name=self.COLLECTION_NAME,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )

        _ = self.vectordb_client.insert_many(
            collection_name=self.COLLECTION_NAME,
            texts=texts,
            vectors=vectors,
            metadata=metadata,
        )

        return len(documents)

    def search(
        self, query: str, limit: int = 5, metadata_filter: dict | None = None
    ) -> list[RetrievedDocument] | None:
        """
        Search the vector database for documents similar to a query.
        The query is converted into an embedding using the query embedding type.
        The resulting vector is then used to perform a similarity search against
        the configured collection.

        Args:
            query: User query to search for.
            limit: Maximum number of documents to retrieve.
            metadata_filter: Optional metadata conditions used to filter the search results.

        Returns:
            A list of retrieved documents if matching results are found; otherwise, None.
        """

        vector = self.embedding_client.embed_text(
            text=query, document_type=DocumentTypeEnum.QUERY.value
        )

        if not vector or len(vector) == 0:
            return False

        result = self.vectordb_client.search_by_vector(
            collection_name=self.COLLECTION_NAME,
            vector=vector,
            limit=limit,
            metadata_filter=metadata_filter,
        )

        if not result:
            return False

        return result

    def get_document_count(self) -> int:
        """
        Return the number of documents stored in the collection.

        Returns:
            The number of records currently stored in the vector database collection.
        """
        return self.vectordb_client.count_records(collection_name=self.COLLECTION_NAME)
