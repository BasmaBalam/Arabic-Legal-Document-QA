from openai import OpenAI


class VLLMProvider:
    """
    vLLM-based implementation of the LLM provider.

    This provider supports text generation using a vLLM server
    through an OpenAI-compatible API.

    Attributes:
        client: OpenAI client configured to communicate with the
            vLLM server.

        model_id: Identifier of the language model served by vLLM.

        max_tokens: Default maximum number of tokens generated
            per request.

        temperature: Default temperature used during text generation.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        api_key: str = "EMPTY",
        default_generation_max_output_tokens: int = 512,
        default_generation_temperature: float = 0.2,
    ) -> None:
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model_id = model_id
        self.max_tokens = default_generation_max_output_tokens
        self.temperature = default_generation_temperature

    def set_generation_model(self, model_id: str) -> None:
        """
        Set the vLLM generation model.

        Args:
            model_id: Identifier of the model exposed by the vLLM
                server and used for text generation.
        """
        self.model_id = model_id

    def generate_text(
        self,
        prompt: str,
        chat_history: list | None = None,
        max_output_tokens: int | None = None,
        temp: float | None = None,
    ) -> str:
        """
        Generate text using the configured vLLM language model.

        The method sends the conversation to the vLLM server
        through the OpenAI-compatible chat completions endpoint.

        Args:
            prompt: User prompt to send to the generation model.

            chat_history: Optional list of previous conversation
                messages, formatted according to the OpenAI chat
                messages structure.

            max_output_tokens: Maximum number of tokens to generate.
                If not provided, the configured default value is used.

            temp: Sampling temperature used during generation.
                If not provided, the configured default temperature
                is used.

        Returns:
            The generated response as a string. Returns an empty
            string if the response content is None.

        Raises:
            OpenAIError: If the API request fails or the vLLM server
                returns an API error.
        """
        messages = list(chat_history or [])
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=(
                max_output_tokens if max_output_tokens is not None else self.max_tokens
            ),
            temperature=temp if temp is not None else self.temperature,
        )

        return response.choices[0].message.content or ""
