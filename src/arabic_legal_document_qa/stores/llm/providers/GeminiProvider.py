import logging

from google import genai
from google.genai import types

from ..LLMEnums import DocumentTypeEnum, GeminiEnums
from ..LLMInterface import LLMInterface


class GeminiProvider(LLMInterface):
    """
    Gemini-based implementation of the LLM provider interface.

    This provider supports text generation and text embeddings
    using the Google Gemini API.

    Attributes:
        api_key: API key used to authenticate with Gemini.

        api_url: Optional API URL.

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
        api_url: str | None = None,
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

        self.client = genai.Client(api_key=self.api_key)

        self.enums = GeminiEnums

        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str) -> None:
        """
        Set the Gemini generation model.

        Args:
            model_id: Identifier of the Gemini model used for text generation.
        """
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int) -> None:
        """
        Configure the Gemini embedding model.

        Args:
            model_id: Identifier of the Gemini model used for
            generating text embeddings.

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
        Generate text using the configured Gemini language model.

        Args:
            prompt: User prompt to send to the generation model.

            chat_history: Optional list of previous conversation messages.

            max_output_tokens: Maximum number of tokens to generate.

            temp: Sampling temperature used during generation.

        Returns:
            The generated response as a string, or ``None`` if
            the Gemini client or generation model has not been configured.
        """

        if not self.client:
            self.logger.error("Gemini client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for Gemini was not set")
            return None

        try:
            contents = []

            if chat_history:
                contents.extend(chat_history)

            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=self.process_text(prompt))],
                )
            )

            response = self.client.models.generate_content(
                model=self.generation_model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    max_output_tokens=(
                        max_output_tokens or self.default_generation_max_output_tokens
                    ),
                    temperature=(
                        temp if temp is not None else self.default_generation_temp
                    ),
                ),
            )

            if not response or not response.text:
                self.logger.error("Error while generating text with Gemini")
                return None

            return response.text

        except Exception:
            self.logger.exception("Error while generating text with Gemini")
            return None

    def embed_text(
        self,
        text: str,
        document_type: str | None = None,
    ) -> list[float] | None:
        """
        Generate an embedding vector using the configured Gemini model.

        The input type is selected based on the document type.

        Args:
            text: Text to encode into an embedding vector.

            document_type: Type of text being embedded.

        Returns:
            The generated embedding vector as a list of floats,
            or ``None`` if the Gemini client, embedding model,
            or embedding response is unavailable.
        """

        if not self.client:
            self.logger.error("Gemini client was not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for Gemini was not set")
            return None

        try:
            task_type = GeminiEnums.RETRIEVAL_DOCUMENT.value

            if document_type == DocumentTypeEnum.QUERY.value:
                task_type = GeminiEnums.RETRIEVAL_QUERY.value

            response = self.client.models.embed_content(
                model=self.embedding_model_id,
                contents=self.process_text(text),
                config=types.EmbedContentConfig(
                    task_type=task_type,
                ),
            )

            if not response or not response.embeddings:
                self.logger.error("Error while embedding text with Gemini")
                return None

            return response.embeddings[0].values

        except Exception:
            self.logger.exception("Error while embedding text with Gemini")
            return None

    def construct_prompt(self, prompt: str, role: str) -> dict:
        """
        Construct a chat message for the Gemini API.

        Args:
            prompt: Message content.
            role: Role of the message.

        Returns:
            A dictionary containing the message role and processed text.
        """

        return {
            "role": role,
            "parts": [{"text": self.process_text(prompt)}],
        }

    def process_text(self, text: str) -> str:
        """
        Truncate and normalize input text.

        Args:
            text: Input text to process.

        Returns:
            The input text truncated to the configured maximum
            character length with leading and trailing whitespace removed.
        """

        return text[: self.default_input_max_characters].strip()
