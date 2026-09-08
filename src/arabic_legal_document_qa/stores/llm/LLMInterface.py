from abc import ABC, abstractmethod

class LLMInterface(ABC):

    @abstractmethod
    def set_generation_model(self, model_id: str):
        """
        Load and configure the text embedding model.

        Args:
            model_id: model name or model identifier for the embedding model.
            embedding_size: Dimension of the output embedding vector.

        Raises:
            Exception: If the embedding model cannot be loaded.
        """

        pass

    @abstractmethod
    def set_embedding_mode(self, model_id: str, embedding_size: int):
        """
        Load and configure the language generation model.

        Args:
            model_id: model name or model identifier for the generation model.

        Raises:
            Exception: If the model or tokenizer cannot be loaded.
        """
        pass

    @abstractmethod
    def generate_text(self, prompt: str, chat_history: list = [], max_output_tokens: int= None,
                      temp: float = None):
        """
        Generate a text response from the configured language model.

        Args:
            prompt: User prompt to send to the language model.

            chat_history: Optional list of previous chat messages. Each
                message should contain ``role`` and ``content`` keys.
            
            max_output_tokens: Maximum number of new tokens to generate.
                Uses the provider default when not specified.
            
            temp: Sampling temperature. A value greater than zero enables
                sampling; zero or ``None`` uses deterministic generation or
                the configured default temperature.

        Returns:
            The generated text, or ``None`` if the generation model has not
            been configured.
        """
        pass

    @abstractmethod
    def embed_text(self, text: str, document_type: str= None):
        """
        Generate an embedding vector for the provided text.

        Args:
            text: Text to convert into an embedding vector.
            document_type: Optional document type used to select the
                appropriate embedding prompt, for example ``"query"``.

        Returns:
            A normalized embedding vector as a list of floats, or ``None``
            if the embedding model has not been configured.
        """
        pass

    @abstractmethod
    def construct_prompt(self, prompt: str, role: str):
        """
        Construct a chat message compatible with the language model.

        Args:
            prompt: Message content.
            role: Message role, such as ``"system"``, ``"user"``, or
                ``"assistant"``.

        Returns:
            A dictionary containing the message ``role`` and ``content``.
        """
        pass