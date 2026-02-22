"""
Anthropic LLM Client Implementation
Encapsulates all Anthropic Claude AI interactions via the anthropic Python SDK.
"""

import os
import sys
from typing import Optional, List, Dict, cast, Iterable
from anthropic import Anthropic
from dotenv import load_dotenv
from cli import console
from .anthropic_validation import AnthropicConfig


class AnthropicResponse:
    """
    Response object that mimics the Gemini response interface.
    Provides a .text attribute containing the response text.
    """

    def __init__(self, text: str, input_tokens: int = 0, output_tokens: int = 0):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class AnthropicChatSession:
    """
    Wrapper class that provides a chat session interface compatible with Gemini's interface.
    Manages conversation history and communicates with the Anthropic Messages API.
    """

    def __init__(self, anthropic_client: Anthropic, model_name: str,
                 system_instruction: str, max_tokens: int = 4096,
                 history: Optional[List[Dict]] = None):
        """
        Initialize the Anthropic chat session.

        Args:
            anthropic_client: Initialized Anthropic SDK client instance
            model_name: Name of the model to use (e.g., 'claude-sonnet-4-20250514')
            system_instruction: System prompt for the assistant
            max_tokens: Maximum tokens in the response
            history: Previous conversation history in universal dict format
        """
        self.anthropic_client = anthropic_client
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.max_tokens = max_tokens
        self._history = history or []
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def send_message(self, text: str) -> AnthropicResponse:
        """
        Sends a message to the Anthropic model and returns a response object.

        Args:
            text: User's message

        Returns:
            AnthropicResponse object with .text attribute containing the response
        """
        # Add user message to history
        user_message = {"role": "user", "parts": [{"text": text}]}
        self._history.append(user_message)

        # Build messages list in Anthropic's expected format
        messages = self._build_messages_from_history()

        try:
            # Call Anthropic Messages API
            # System prompt is a top-level parameter, NOT inside messages[]
            response = self.anthropic_client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                system=self.system_instruction,
                messages=cast(Iterable, messages),
            )

            # Extract text from the response content blocks
            response_text = ""
            for block in response.content:
                if block.type == "text":
                    response_text += block.text

            response_text = response_text.strip()

            # Track token usage from response metadata
            input_tokens = response.usage.input_tokens if response.usage else 0
            output_tokens = response.usage.output_tokens if response.usage else 0
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens

            # Add assistant response to history
            assistant_message = {"role": "model", "parts": [{"text": response_text}]}
            self._history.append(assistant_message)

            return AnthropicResponse(response_text, input_tokens, output_tokens)

        except Exception as e:
            console.print_error(f"Błąd podczas generowania odpowiedzi Anthropic: {e}")
            error_text = "Przepraszam, wystąpił błąd podczas generowania odpowiedzi."
            assistant_message = {"role": "model", "parts": [{"text": error_text}]}
            self._history.append(assistant_message)
            return AnthropicResponse(error_text)

    def get_history(self) -> List[Dict]:
        """Returns the current conversation history in universal dict format."""
        return self._history

    def _build_messages_from_history(self) -> List[Dict[str, str]]:
        """
        Converts universal history format to Anthropic's message format.

        The universal format uses:
            {"role": "user"|"model", "parts": [{"text": "..."}]}

        Anthropic expects:
            {"role": "user"|"assistant", "content": "..."}

        Note: System prompt is NOT included in messages[] — it is passed
        as a separate top-level 'system' parameter to messages.create().

        Returns:
            List of message dicts in Anthropic's format
        """
        messages = []

        for entry in self._history:
            role = entry["role"]
            text = entry["parts"][0]["text"] if entry.get("parts") else ""

            if not text:
                continue

            # Map universal role names to Anthropic role names
            # "model" -> "assistant" (Anthropic's convention)
            if role == "user":
                anthropic_role = "user"
            elif role == "model":
                anthropic_role = "assistant"
            else:
                anthropic_role = role

            messages.append({
                "role": anthropic_role,
                "content": text
            })

        return messages

    @property
    def total_tokens(self) -> int:
        """Returns total tokens consumed across all messages in this session."""
        return self._total_input_tokens + self._total_output_tokens


class AnthropicClient:
    """
    Encapsulates all Anthropic Claude AI interactions.
    Provides a clean interface compatible with GeminiLLMClient and LlamaClient.
    """

    def __init__(self, model_name: str, api_key: str, max_tokens: int = 4096):
        """
        Initialize the Anthropic client with explicit parameters.

        Args:
            model_name: Model to use (e.g., 'claude-sonnet-4-20250514')
            api_key: Anthropic API key
            max_tokens: Maximum tokens per response (default: 4096)

        Raises:
            ValueError: If api_key is empty or None
        """
        if not api_key:
            raise ValueError("API key cannot be empty or None")

        self.model_name = model_name
        self.api_key = api_key
        self.max_tokens = max_tokens

        # Initialize the client during construction
        self._client = self._initialize_client()

    @staticmethod
    def preparing_for_use_message() -> str:
        """
        Returns a message indicating that Anthropic client is being prepared.

        Returns:
            Formatted preparation message string
        """
        return "🧠 Przygotowywanie klienta Anthropic..."

    @classmethod
    def from_environment(cls) -> 'AnthropicClient':
        """
        Factory method that creates an AnthropicClient instance from environment variables.

        Environment variables:
            ANTHROPIC_MODEL_NAME: Model name (default: 'claude-sonnet-4-20250514')
            ANTHROPIC_API_KEY: API key (required)
            ANTHROPIC_MAX_TOKENS: Max response tokens (default: 4096)

        Returns:
            AnthropicClient instance initialized with environment variables

        Raises:
            ValueError: If required environment variables are not set
        """
        load_dotenv()

        # Validate with Pydantic
        config = AnthropicConfig(
            model_name=os.getenv('ANTHROPIC_MODEL_NAME', 'claude-haiku-4-5-20251001'),
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY', ''),
            max_tokens=int(os.getenv('ANTHROPIC_MAX_TOKENS', '4096'))
        )

        return cls(
            model_name=config.model_name,
            api_key=config.anthropic_api_key,
            max_tokens=config.max_tokens
        )

    def _initialize_client(self) -> Anthropic:
        """
        Initializes the Anthropic SDK client.

        Returns:
            Initialized Anthropic client

        Raises:
            SystemExit: If client initialization fails
        """
        try:
            return Anthropic(api_key=self.api_key)
        except Exception as e:
            console.print_error(f"Błąd inicjalizacji klienta Anthropic: {e}")
            sys.exit(1)

    def create_chat_session(self,
                            system_instruction: str,
                            history: Optional[List[Dict]] = None,
                            thinking_budget: int = 0) -> AnthropicChatSession:
        """
        Creates a new chat session with the specified configuration.

        Args:
            system_instruction: System role/prompt for the assistant
            history: Previous conversation history (optional, in universal dict format)
            thinking_budget: Ignored for Anthropic (compatibility parameter)

        Returns:
            AnthropicChatSession object
        """
        if not self._client:
            raise RuntimeError("Anthropic client not initialized")

        return AnthropicChatSession(
            anthropic_client=self._client,
            model_name=self.model_name,
            system_instruction=system_instruction,
            max_tokens=self.max_tokens,
            history=history or []
        )

    def count_history_tokens(self, history: List[Dict]) -> int:
        """
        Estimates token count for the given conversation history.

        Uses Anthropic's Messages API count_tokens endpoint when available,
        with a word-based heuristic as fallback.

        Args:
            history: Conversation history in universal dict format

        Returns:
            Estimated token count
        """
        if not history:
            return 0

        try:
            # Build messages in Anthropic format for accurate counting
            messages = []
            for entry in history:
                if isinstance(entry, dict) and 'role' in entry and 'parts' in entry:
                    text = entry['parts'][0].get('text', '') if entry['parts'] else ''
                    if text:
                        role = "assistant" if entry['role'] == "model" else entry['role']
                        messages.append({"role": role, "content": text})

            if not messages:
                return 0

            # Try using the count_tokens API
            response = self._client.messages.count_tokens(
                model=self.model_name,
                messages=messages,
            )
            return response.input_tokens

        except Exception as e:
            # Fallback: heuristic estimation (~0.75 tokens per word)
            try:
                text_parts = []
                for message in history:
                    if "parts" in message and message["parts"]:
                        text_parts.append(message["parts"][0]["text"])
                full_text = " ".join(text_parts)
                word_count = len(full_text.split())
                return int(word_count * 0.75)
            except Exception:
                return 0

    def get_model_name(self) -> str:
        """Returns the currently configured model name."""
        return self.model_name

    def is_available(self) -> bool:
        """
        Checks if the Anthropic service is available and properly configured.

        Returns:
            True if client is properly initialized and has API key
        """
        return self._client is not None and bool(self.api_key)

    def ready_for_use_message(self) -> str:
        """
        Returns a ready-to-use message with model info and masked API key.

        Returns:
            Formatted message string for display
        """
        # Mask API key - show first 4 and last 4 characters
        if len(self.api_key) <= 8:
            masked_key = "****"
        else:
            masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}"

        return f"✅ Klient Anthropic gotowy do użycia (Model: {self.model_name}, Key: {masked_key})"

    @property
    def client(self):
        """
        Provides access to the underlying Anthropic client for backwards compatibility.
        """
        return self._client




