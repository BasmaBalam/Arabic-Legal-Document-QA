import logging

import cohere

from ..LLMEnums import CohereEnums, DocumentTypeEnum
from ..LLMInterface import LLMInterface


class CohereProvider(LLMInterface):
    """
    Cohere-based implementation of the LLM provider interface.
    This provider supports both text generation and text embeddings
    using the Cohere API.

    Attributes:
        api_key: API key used to authenticate with Cohere.

        api_url: Optional Cohere API URL.

        default_input_max_characters: Maximum number of input
        characters processed from a prompt or text.

        default_generation_max_output_tokens: Default maximum
        number of tokens generated per request.

        default_generation_temp: Default temperature used during
        text generation.

        generation_model_id: Identifier of the configured generation model.

        embedding_model_id: Identifier of the configured embedding model.
        embedding_size: Dimension of the generated embedding vectors.
    """

    def __init__(
        self,
        api_key: str,
        api_url: str = None,
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temp: float = 0.1,
    ):

        self.api_key = api_key
        self.api_url = api_url
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temp = default_generation_temp

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        self.client = cohere.Client(api_key=self.api_key)
        self.enums = CohereEnums

        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str) -> None:
        """
        Set the Cohere generation model.

        Args:
            model_id: Identifier of the Cohere model used for text generation.
        """
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int) -> None:
        """
        Configure the Cohere embedding model.

        Args:
            model_id: Identifier of the Cohere model used for generating
            text embeddings.

            embedding_size: Dimension of the generated embedding vectors.
        """
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def generate_text(
        self,
        prompt: str,
        chat_history: list | None = None,
        max_output_tokens: int | None = None,
        temp: float | None = None,
    ) -> str | None:
        """
        Generate text using the configured Cohere language model.

        Args:
            prompt: User prompt to send to the generation model.

            chat_history: Optional list of previous conversation messages.

            max_output_tokens: Maximum number of tokens to generate. If omitted,
            the configured default is used by the provider.

            temp: Sampling temperature used during generation.
            If omitted, the configured default temperature is used.

        Returns:
            The generated response as a string, or ``None`` if the
            Cohere client or generation model has not been configured.
        """
        if not self.client:

            self.logger.error("Cohere client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for Cohere was not set")
            return None

        response = self.client.chat(
            model=self.generation_model_id,
            chat_history=chat_history,
            message=self.process_text(prompt),
            max_tokens=max_output_tokens,
            temperature=temp,
        )

        if not response or not response.text:
            self.logger.error("Error while generation text with Cohere")
            return None

        return response.text

    def embed_text(
        self, text: str, document_type: str | None = None
    ) -> list[float] | None:
        """
        Generate an embedding vector using the configured Cohere model.

        The input type is selected based on the document type.
        Query text uses the query embedding type,
        while other text is treated as document content.

        Args:
            text: Text to encode into an embedding vector.

            document_type: Type of text being embedded.
            If it matches ``DocumentTypeEnum.QUERY.value``,
            the query-specific embedding type is used.

        Returns:
            The generated embedding vector as a list of floats,
            or ``None`` if the Cohere client, embedding model, or
            embedding response is unavailable.
        """
        if not self.client:
            self.logger.error("Cohere client was not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for Cohere was not set")
            return None

        input_type = CohereEnums.DOCUMENT.value
        if document_type == DocumentTypeEnum.QUERY.value:
            input_type = CohereEnums.QUERY.value

        response = self.client.embed(
            model=self.embedding_model_id,
            texts=[self.process_text(text)],
            input_type=input_type,
        )

        if not response or not response.embeddings:
            self.logger.error("Error while embedding text with Cohere")
            return None

        return response.embeddings[0]

    def construct_prompt(self, prompt: str, role: str) -> dict:
        """
        Construct a chat message for the Cohere chat API.

        Args:
            prompt: Message content.
            role: Role of the message, such as "system", "user", or "assistant".

        Returns:
            A dictionary containing the message role and processed text.
        """
        return {"role": role, "text": self.process_text(prompt)}

    def process_text(self, text: str) -> str:
        """Truncate and normalize input text.

        Args:
            text: Input text to process.

        Returns:
            The input text truncated to the configured maximum
            character length with leading and trailing whitespace removed.
        """
        return text[: self.default_input_max_characters].strip()
