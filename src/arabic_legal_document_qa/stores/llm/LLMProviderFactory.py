from .LLMEnums import LLMEnum
from .providers import QwenProvider


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
                default_input_max_characters = self.config.DEFAULT_INPUT_MAX_CHARACTERS,
                default_generation_max_output_tokens= self.config.DEFAULT_GENERATION_MAX_OUTPUT_TOKENS,
                default_generation_temperature = self.config.DEFAULT_GENERATION_TEMPERATURE,
                enable_thinking= self.config.ENABLE_THINKING
            )

        return None