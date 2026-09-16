import os


class TemplateParser:
    """Loads and formats language-specific prompt templates."""

    def __init__(self, language: str | None = None, default_language: str = "en"):
        """
        Initialize the template parser.

        Args:
            language: Language to use for loading templates.
            default_language: Fallback language when the requested
            language is not available.
        """

        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.defult_language = default_language
        self.language = None

        self.set_language(language)

    def set_language(self, language: str):
        """
        Set the current template language.

        If the requested language is not available, the default language
        is used instead.

        Args:
            language: Language code to use for templates.
        """
        language_path = os.path.join(self.current_path, "locales", language)
        if language and os.path.exists(language_path):
            self.language = language
        else:
            self.language = self.defult_language

    def get(self, group: str, key: str, vars: dict | None = None):
        """
        Retrieve and format a template.
        The template is loaded from the requested language.
        If the template group is not available in the current language,
        the default language is used.

        Args:
            group: Name of the template group.
            key: Name of the template within the group.
            vars: Optional variables used to substitute template placeholders.

        Returns:
            The formatted template string, or None if the group or key cannot be found.
        """
        if not group or not key:
            return None
        group_path = os.path.join(
            self.current_path, "locales", self.language, f"{group}.py"
        )
        if not os.path.exists(group_path):
            group_path = os.path.join(
                self.current_path, "locales", self.defult_language, f"{group}.py"
            )

        if not os.path.exists(group_path):
            return None

        # import group module
        module = __import__(
            f"src.arabic_legal_document_qa.stores.llm.templates.locales.{self.language}.{group}",
            fromlist=[group],
        )

        if not module:
            return None

        key_attribute = getattr(module, key)
        return key_attribute.substitute(vars)
