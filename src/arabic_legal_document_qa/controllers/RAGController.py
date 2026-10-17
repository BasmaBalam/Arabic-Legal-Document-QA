import json
import re

from langchain_core.documents import Document

from arabic_legal_document_qa.models.db_schemes.legal_query_metadata import (
    LegalQueryMetadata,
)

from .BaseController import BaseController
from .DocumentController import DocumentController
from .VectorDBController import VectorDBController


class RAGController(BaseController):
    """
    Controller for the Retrieval-Augmented Generation (RAG) pipeline.
    This controller coordinates document processing, vector database indexing,
    query metadata extraction, document retrieval, and answer generation.
    The RAG pipeline follows these main

    steps:
        1. Extract metadata from the user's question.
        2. Build metadata filters from the extracted metadata.
        3. Retrieve relevant documents from the vector database.
        4. Fall back to vector-only search if filtered retrieval returns no
        documents.
        5. Build a prompt using the retrieved legal documents.
        6. Generate an answer using the generation client.
        7. Return the generated answer together with document citations.

    Attributes:
        generation_client: Client responsible for generating text using the
        configured language model.

        document_controller: Controller responsible for reading and processing
        legal documents.

        vectorDB_controller: Controller responsible for embedding, indexing, and retrieving
        documents from the vector database.

        template_parser: Parser responsible for loading and formatting prompt templates.
    """

    def __init__(
        self, vectordb_client, embedding_client, generation_client, template_parser
    ):
        super().__init__()
        self.generation_client = generation_client
        self.document_controller = DocumentController()
        self.vectorDB_controller = VectorDBController(vectordb_client, embedding_client)
        self.template_parser = template_parser

    def chunk_documents(self, chunk_size: int, overlap_size: int) -> list[Document]:
        """
        Read and split the legal document into chunks.
        The document content is loaded using the document controller and then divided
        into overlapping chunks according to the provided chunk size and overlap size.

        Args:
            chunk_size: Maximum size of each document chunk.
            overlap_size: Number of overlapping characters or tokens between consecutive
            chunks, depending on the document processing implementation.

        Returns:
            A list of LangChain Document objects representing the processed document chunks.
        """

        document_controller = DocumentController()
        doc_content = document_controller.get_document_content()

        chunks = document_controller.process_document_content(
            doc_content=doc_content, chunk_size=chunk_size, overlap_size=overlap_size
        )

        return chunks

    def index_documents(self, documents: list[Document], do_reset: bool = False) -> int:
        """
        Add document chunks to the vector database.

        Args:
            documents: List of document chunks to index.
            do_reset: Whether to reset the vector database
            collection before inserting the documents.

        Returns: The number of documents indexed.
        """
        return self.vectorDB_controller.add_documents(documents, do_reset)

    def get_indexed_count(self) -> int:
        """
        Return the number of indexed documents.

        Returns:
            The number of records currently stored in the vector database.
        """
        return self.vectorDB_controller.get_document_count()
    
    def extract_query_metadata(self, question: str) -> LegalQueryMetadata:
        """
        Extract legal metadata from a user question.
        The question is passed to the metadata extraction prompt, and the generated
        JSON response is parsed and validated using the LegalQueryMetadata model.
        Markdown JSON code fences returned by the language model are removed before parsing.

        If the generated response cannot be parsed or validated, an empty LegalQueryMetadata
        object is returned.

        Args:
            question: User's legal question.

        Returns:
            Extracted legal query metadata, or an empty metadata object if extraction fails.
        """

        prompt = self.template_parser.get(
            "metadata_extraction",
            "prompt",
            {"question": question},
        )

        try:
            response = self.generation_client.generate_text(prompt=prompt)

            if not isinstance(response, str) or not response.strip():
                return LegalQueryMetadata()

            cleaned_response = response.strip()

            cleaned_response = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                cleaned_response,
                flags=re.IGNORECASE,
            )

            metadata_dict = json.loads(cleaned_response)

            return LegalQueryMetadata.model_validate(metadata_dict)

        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self.logger.warning("Failed to extract legal metadata: %s", exc)
            return LegalQueryMetadata()

    def build_metadata_filter(self, metadata: LegalQueryMetadata) -> dict | None:
        """
        Build a vector database metadata filter.
        Only metadata fields that contain values are included in the filter.

        Args:
            metadata: Extracted metadata from the user's legal question.

        Returns:
            A dictionary containing the metadata filters, or None if no metadata fields are available.
        """
        metadata_filter = {}

        if metadata.book is not None:
            metadata_filter["book"] = metadata.book

        if metadata.chapter is not None:
            metadata_filter["chapter"] = metadata.chapter

        if metadata.section is not None:
            metadata_filter["section"] = metadata.section

        if metadata.topic is not None:
            metadata_filter["topic"] = metadata.topic

        if metadata.article_number is not None:
            metadata_filter["article_number"] = metadata.article_number

        return metadata_filter or None

    def answer(
        self, question: str, limit: int = 10
    ) -> tuple[str, list[str], list[str]] | None:
        """
        Retrieve legal documents and generate an answer to a question.

        The method performs the complete RAG pipeline:
            1. Extract metadata from the question.
            2. Build a metadata filter.
            3. Retrieve relevant documents using vector search and metadata.
            4. Retry vector search without metadata filtering if no documents are found.
            5. Extract the text content from the retrieved documents as contexts.
            6. Extract unique citations from the retrieved documents.
            7. Build the RAG prompt using the retrieved documents.
            8. Generate the final answer using the generation client.

        Args:
            question: User's legal question.
            limit: Maximum number of documents to retrieve.

        Returns:
            A tuple containing:
            - The generated answer.
            - A list of unique document citations.
            - A list of retrieved document texts used as contexts.
            Returns None if no relevant documents are retrieved.
        """
        # Extract metadata from the question
        query_metadata = self.extract_query_metadata(question)

        # Build metadata filter
        metadata_filter = self.build_metadata_filter(query_metadata)

        # Retrieve documents
        retrieved_documents = self.vectorDB_controller.search(
            query=question,
            limit=limit,
            metadata_filter=metadata_filter,
        )

        # If metadata-filtered search returns no documents,
        # retry using vector search without metadata filtering.
        if not retrieved_documents and metadata_filter is not None:
            retrieved_documents = self.vectorDB_controller.search(
                query=question,
                limit=limit,
                metadata_filter=None,
            )

        if not retrieved_documents:
            return None

        contexts = [doc.text for doc in retrieved_documents]
        sources = []
        for doc in retrieved_documents:
            meta = doc.metadata or {}
            citation = meta.get("citation")

            # Fallback to article number if 'citation' key is not explicitly set
            if not citation and meta.get("article_number"):
                citation = f"article {meta.get('article_number')}"

            if citation and citation not in sources:
                sources.append(str(citation))

        system_prompt = self.template_parser.get("rag", "system_prompt")

        documents_prompts = "\n".join(
            [
                self.template_parser.get(
                    "rag",
                    "document_prompt",
                    {
                        "doc_num": idx + 1,
                        "article_number": doc.metadata.get("article_number"),
                        "book": doc.metadata.get("book"),
                        "chapter": doc.metadata.get("chapter"),
                        "section": doc.metadata.get("section"),
                        "topic": doc.metadata.get("topic"),
                        "citation": doc.metadata.get("citation"),
                        "is_repealed": doc.metadata.get("is_repealed"),
                        "source_page": doc.metadata.get("source_page"),
                        "chunk_text": doc.text,
                    },
                )
                for idx, doc in enumerate(retrieved_documents)
            ]
        )

        footer_prompt = self.template_parser.get(
            "rag", "footer_prompt", vars={"question": question}
        )

        full_prompt = "\n\n".join([system_prompt, documents_prompts, footer_prompt])

        answer = self.generation_client.generate_text(
            prompt=full_prompt,
        )

        return answer, sources, contexts
