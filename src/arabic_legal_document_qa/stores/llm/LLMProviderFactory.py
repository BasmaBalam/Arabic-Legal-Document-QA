from .LLMEnums import LLMEnum
from .providers import CohereProvider, GeminiProvider, OllamaProvider, QwenProvider, VLLMProvider


class LLMProviderFactory:
    """
    Factory responsible for creating configured LLM providers.

    Attributes:
        config: Configuration object containing the default settings.
    """

    def __init__(self, config: dict):
        """
        Initialize the LLM provider factory.

        Args:
            config: Configuration object containing settings.
        """
        self.config = config

    def create(self, provider: str):
        """
        Create an LLM provider based on the requested provider name.

        Args:
            provider: Provider identifier.

        Returns:
            A configured provider instance when the provider is supported;
            otherwise ``None``.
        """

        if provider == LLMEnum.QWEN.value:
            return QwenProvider(
                default_input_max_characters=self.config.DEFAULT_INPUT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.DEFAULT_GENERATION_MAX_OUTPUT_TOKENS,
                default_generation_temperature=self.config.DEFAULT_GENERATION_TEMPERATURE,
                enable_thinking=self.config.ENABLE_THINKING,
            )

        if provider == LLMEnum.COHERE.value:
            return CohereProvider(
                api_key=self.config.COHERE_API_KEY,
                default_input_max_characters=self.config.DEFAULT_INPUT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.DEFAULT_GENERATION_MAX_OUTPUT_TOKENS,
                default_generation_temp=self.config.DEFAULT_GENERATION_TEMPERATURE,
            )

        if provider == LLMEnum.GEMINI.value:
            return GeminiProvider(
                api_key=self.config.GEMINI_API_KEY,
                default_input_max_characters=(self.config.DEFAULT_INPUT_MAX_CHARACTERS),
                default_generation_max_output_tokens=(
                    self.config.DEFAULT_GENERATION_MAX_OUTPUT_TOKENS
                ),
                default_generation_temp=(self.config.DEFAULT_GENERATION_TEMPERATURE),
            )

        if provider == LLMEnum.OLLAMA.value:
            return OllamaProvider(
                host=self.config.OLLAMA_HOST,
                default_input_max_characters=(self.config.DEFAULT_INPUT_MAX_CHARACTERS),
                default_generation_max_output_tokens=(
                    self.config.DEFAULT_GENERATION_MAX_OUTPUT_TOKENS
                ),
                default_generation_temperature=(
                    self.config.DEFAULT_GENERATION_TEMPERATURE
                ),
                enable_thinking=self.config.ENABLE_THINKING,
                keep_alive=self.config.OLLAMA_KEEP_ALIVE,
            )

        if provider == LLMEnum.VLLM.value:
            return VLLMProvider(
                base_url=self.config.VLLM_BASE_URL,
                model_id=self.config.VLLM_MODEL_ID,
                api_key=self.config.VLLM_API_KEY,
                default_generation_max_output_tokens=(
                    self.config.DEFAULT_GENERATION_MAX_OUTPUT_TOKENS
                ),
                default_generation_temperature=(
                    self.config.DEFAULT_GENERATION_TEMPERATURE
                ),
            )

        return None
