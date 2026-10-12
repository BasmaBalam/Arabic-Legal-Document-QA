import logging

import ollama

from ..LLMInterface import LLMInterface

logger = logging.getLogger(__name__)


class OllamaProvider(LLMInterface):
    """
    Ollama-based implementation of the LLM provider interface.

    This provider supports:
        - Local chat generation through Ollama.
        - Local text embeddings through Ollama.

    Ollama must be installed and running locally.

    Attributes:
        client: Ollama client used for local model communication.
        host: Ollama server URL.
        generation_model_id: Ollama model used for generation.
        embedding_model_id: Ollama model used for embeddings.
        embedding_size: Expected embedding dimension.
        default_input_max_characters: Maximum characters processed per input.
        default_generation_max_output_tokens: Maximum generated tokens.
        default_generation_temperature: Generation temperature.
        enable_thinking: Whether model thinking mode is enabled.
        keep_alive: How long Ollama keeps models loaded.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        default_input_max_characters: int = 3000,
        default_generation_max_output_tokens: int = 1024,
        default_generation_temperature: float = 0.3,
        enable_thinking: bool = False,
        keep_alive: str | int = "10m",
    ):
        """
        Initialize the Ollama provider.

        Args:
            host:
                Ollama server URL.
            default_input_max_characters:
                Maximum number of characters accepted from input text.
            default_generation_max_output_tokens:
                Default maximum number of generated tokens.
            default_generation_temperature:
                Default generation temperature.
            enable_thinking:
                Enable thinking mode for supported models.
            keep_alive:
                How long Ollama should keep the model loaded.
                Examples: "10m", "30m", -1.
        """

        self.host = host

        self.client = ollama.Client(
            host=self.host,
        )

        # Generation model
        self.generation_model_id: str | None = None

        # Embedding model
        self.embedding_model_id: str | None = None
        self.embedding_size: int | None = None

        self.default_input_max_characters = default_input_max_characters

        self.default_generation_max_output_tokens = default_generation_max_output_tokens

        self.default_generation_temperature = default_generation_temperature

        self.enable_thinking = enable_thinking
        self.keep_alive = keep_alive

    def set_generation_model(
        self,
        model_id: str,
    ):
        """
        Configure the Ollama generation model.

        The model must already be available in Ollama.

        Example:
            qwen3:8b
        """

        self.generation_model_id = model_id

        logger.info(
            "Configured Ollama generation model: %s",
            model_id,
        )

    def set_embedding_model(
        self,
        model_id: str,
        embedding_size: int,
    ):
        """
        Configure the Ollama embedding model.

        The model must already be available in Ollama.

        Example:
            qwen3-embedding:0.6b
        """

        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

        logger.info(
            "Configured Ollama embedding model: %s (dim=%d)",
            model_id,
            embedding_size,
        )

    def process_text(
        self,
        text: str,
    ) -> str:
        """
        Truncate and normalize input text.

        Args:
            text: Input text to process.

        Returns:
            The input text truncated to the configured maximum character
            length with leading and trailing whitespace removed.
        """

        return text[: self.default_input_max_characters].strip()

    def construct_prompt(
        self,
        prompt: str,
        role: str,
    ) -> dict[str, str]:
        """
        Construct a chat message for the Ollama chat template.

        Args:
            prompt: Message content.
            role: Role of the message, such as ``"system"``, ``"user"``,
                or ``"assistant"``.

        Returns:
            A dictionary containing ``role`` and processed ``content``.
        """

        return {
            "role": role,
            "content": self.process_text(prompt),
        }

    def generate_text(
        self,
        prompt: str,
        chat_history: list | None = None,
        max_output_tokens: int | None = None,
        temp: float | None = None,
    ) -> str | None:
        """
        Generate text using the configured Ollama model.

        Args:
            prompt:
                User prompt.
            chat_history:
                Optional previous chat messages.
            max_output_tokens:
                Maximum number of newly generated tokens.
            temp:
                Sampling temperature.

        Returns:
            Generated response text, or None if the model
            has not been configured.
        """

        if self.generation_model_id is None:
            logger.error(
                "Generation model was not set. Call set_generation_model first."
            )
            return None

        max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else self.default_generation_max_output_tokens
        )

        temp = self.default_generation_temperature if temp is None else temp

        messages = list(chat_history or [])

        messages.append(
            self.construct_prompt(
                prompt=prompt,
                role="user",
            )
        )

        try:
            response = self.client.chat(
                model=self.generation_model_id,
                messages=messages,
                think=self.enable_thinking,
                keep_alive=self.keep_alive,
                options={
                    "temperature": temp,
                    "num_predict": max_output_tokens,
                },
            )

            return response.message.content.strip()

        except ollama.ResponseError as exc:
            logger.error(
                "Ollama generation error: %s",
                exc.error,
            )
            return None

        except Exception:
            logger.exception("Unexpected error during Ollama generation.")
            return None

    def embed_text(
        self,
        text: str,
        document_type: str | None = None,
    ) -> list[float] | None:
        """
        Generate an embedding vector using Ollama.

        Args:
            text:
                Text to embed.
            document_type:
                Document/query type. It is accepted to maintain
                compatibility with the LLMInterface.

        Returns:
            Embedding vector as a list of floats, or None if
            the embedding model has not been configured.
        """

        if self.embedding_model_id is None:
            logger.error("Embedding model was not set. Call set_embedding_model first.")
            return None

        text = self.process_text(text)

        try:
            response = self.client.embed(
                model=self.embedding_model_id,
                input=text,
                dimensions=self.embedding_size,
                keep_alive=self.keep_alive,
            )

            embedding = response.embeddings[0]

            return list(embedding)

        except ollama.ResponseError as exc:
            logger.error(
                "Ollama embedding error: %s",
                exc.error,
            )
            return None

        except Exception:
            logger.exception("Unexpected error during Ollama embedding.")
            return None
