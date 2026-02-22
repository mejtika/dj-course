"""
Ollama LLM Client Implementation
Encapsulates all Ollama model interactions via the ollama Python SDK.
Communicates with a locally running Ollama server over HTTP.
"""

import os
from typing import Optional, List, Dict
from ollama import Client as OllamaSDKClient
from dotenv import load_dotenv
from cli import console
from .ollama_validation import OllamaConfig


class OllamaResponse:
    """
    Response object that mimics the Gemini response interface.
    Provides a .text attribute containing the response text.
    """

    def __init__(self, text: str):
        self.text = text


class OllamaChatSession:
    """
    Wrapper class that provides a chat session interface compatible with Gemini's interface.
    Manages conversation history and communicates with the Ollama server via chat API.
    """

    def __init__(self, ollama_client: OllamaSDKClient, model_name: str,
                 system_instruction: str, history: Optional[List[Dict]] = None):
        """
        Initialize the Ollama chat session.

        Args:
            ollama_client: Initialized Ollama SDK client instance
            model_name: Name of the model to use (e.g., 'llama3.1', 'mistral')
            system_instruction: System prompt for the assistant
            history: Previous conversation history in universal dict format
        """
        self.ollama_client = ollama_client
        self.model_name = model_name
        self.system_instruction = system_instruction
        self._history = history or []

    def send_message(self, text: str) -> OllamaResponse:
        """
        Sends a message to the Ollama model and returns a response object.

        Args:
            text: User's message

        Returns:
            OllamaResponse object with .text attribute containing the response
        """
        # Add user message to history
        user_message = {"role": "user", "parts": [{"text": text}]}
        self._history.append(user_message)

        # Build messages list in Ollama's expected format
        messages = self._build_messages_from_history()

        try:
            # Call Ollama chat API (non-streaming for compatibility)
            response = self.ollama_client.chat(
                model=self.model_name,
                messages=messages,
                stream=False,
            )

            response_text = response['message']['content'].strip()

            # Add assistant response to history
            assistant_message = {"role": "model", "parts": [{"text": response_text}]}
            self._history.append(assistant_message)

            return OllamaResponse(response_text)

        except Exception as e:
            console.print_error(f"Błąd podczas generowania odpowiedzi Ollama: {e}")
            error_text = "Przepraszam, wystąpił błąd podczas generowania odpowiedzi."
            assistant_message = {"role": "model", "parts": [{"text": error_text}]}
            self._history.append(assistant_message)
            return OllamaResponse(error_text)

    def get_history(self) -> List[Dict]:
        """Returns the current conversation history in universal dict format."""
        return self._history

    def _build_messages_from_history(self) -> List[Dict[str, str]]:
        """
        Converts universal history format to Ollama's message format.

        The universal format uses:
            {"role": "user"|"model", "parts": [{"text": "..."}]}

        Ollama expects:
            {"role": "system"|"user"|"assistant", "content": "..."}

        Returns:
            List of message dicts in Ollama's format
        """
        messages = []

        # Add system instruction as the first message
        if self.system_instruction:
            messages.append({
                "role": "system",
                "content": self.system_instruction
            })

        # Convert each history entry
        for entry in self._history:
            role = entry["role"]
            text = entry["parts"][0]["text"] if entry.get("parts") else ""

            if not text:
                continue

            # Map universal role names to Ollama role names
            if role == "user":
                ollama_role = "user"
            elif role == "model":
                ollama_role = "assistant"
            else:
                ollama_role = role

            messages.append({
                "role": ollama_role,
                "content": text
            })

        return messages


class OllamaClient:
    """
    Encapsulates all Ollama model interactions.
    Provides a clean interface compatible with GeminiLLMClient and LlamaClient.
    Communicates with a locally running Ollama server over HTTP.
    """

    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        """
        Initialize the Ollama client with explicit parameters.

        Args:
            model_name: Name of the Ollama model to use (e.g., 'llama3.1', 'mistral')
            host: URL of the Ollama server (default: http://localhost:11434)

        Raises:
            ValueError: If model_name is empty
        """
        if not model_name:
            raise ValueError("Model name cannot be empty")

        self.model_name = model_name
        self.host = host

        # Initialize the Ollama SDK client during construction
        self._client = self._initialize_client()

    @staticmethod
    def preparing_for_use_message() -> str:
        """
        Returns a message indicating that Ollama client is being prepared.

        Returns:
            Formatted preparation message string
        """
        return "🦙 Przygotowywanie klienta Ollama..."

    @classmethod
    def from_environment(cls) -> 'OllamaClient':
        """
        Factory method that creates an OllamaClient instance from environment variables.

        Environment variables:
            OLLAMA_MODEL_NAME: Name of the model (default: 'llama3.1')
            OLLAMA_HOST: Ollama server URL (default: 'http://localhost:11434')

        Returns:
            OllamaClient instance initialized with environment variables

        Raises:
            ValueError: If configuration is invalid
        """
        load_dotenv()

        # Validate with Pydantic
        config = OllamaConfig(
            model_name=os.getenv('OLLAMA_MODEL_NAME', 'qwen2.5:7b-instruct'),
            ollama_host=os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        )

        console.print_info(f"Łączenie z serwerem Ollama: {config.ollama_host}")

        return cls(
            model_name=config.model_name,
            host=config.ollama_host
        )

    def _initialize_client(self) -> OllamaSDKClient:
        """
        Initializes the Ollama SDK client.

        Returns:
            Initialized Ollama SDK client

        Raises:
            RuntimeError: If client initialization fails
        """
        try:
            console.print_info(f"Inicjalizacja klienta Ollama: model={self.model_name}, host={self.host}")
            return OllamaSDKClient(host=self.host)
        except Exception as e:
            console.print_error(f"Błąd inicjalizacji klienta Ollama: {e}")
            raise RuntimeError(f"Failed to initialize Ollama client: {e}")

    def create_chat_session(self,
                            system_instruction: str,
                            history: Optional[List[Dict]] = None,
                            thinking_budget: int = 0) -> OllamaChatSession:
        """
        Creates a new chat session with the specified configuration.

        Args:
            system_instruction: System role/prompt for the assistant
            history: Previous conversation history (optional, in universal dict format)
            thinking_budget: Ignored for Ollama (compatibility parameter)

        Returns:
            OllamaChatSession object
        """
        if not self._client:
            raise RuntimeError("Ollama client not initialized")

        return OllamaChatSession(
            ollama_client=self._client,
            model_name=self.model_name,
            system_instruction=system_instruction,
            history=history or []
        )

    def count_history_tokens(self, history: List[Dict]) -> int:
        """
        Estimates token count for the given conversation history.

        Ollama does not provide a standalone token-counting API, so we use
        a heuristic: approximately 0.75 tokens per word (English average).

        Args:
            history: Conversation history in universal dict format

        Returns:
            Estimated token count
        """
        if not history:
            return 0

        try:
            text_parts = []
            for message in history:
                if "parts" in message and message["parts"]:
                    text_parts.append(message["parts"][0]["text"])

            full_text = " ".join(text_parts)

            # Heuristic: split by whitespace, multiply by 0.75 tokens/word
            word_count = len(full_text.split())
            return int(word_count * 0.75)

        except Exception as e:
            console.print_error(f"Błąd podczas szacowania tokenów: {e}")
            total_chars = sum(
                len(msg["parts"][0]["text"])
                for msg in history if "parts" in msg and msg["parts"]
            )
            return total_chars // 4

    def get_model_name(self) -> str:
        """Returns the currently configured model name."""
        return self.model_name

    def is_available(self) -> bool:
        """
        Checks if the Ollama server is available and the model is accessible.

        Returns:
            True if client is initialized and server responds
        """
        if not self._client:
            return False

        try:
            self._client.list()
            return True
        except Exception:
            return False

    def ready_for_use_message(self) -> str:
        """
        Returns a ready-to-use message with model info and server host.

        Returns:
            Formatted message string for display
        """
        return f"✅ Klient Ollama gotowy do użycia (Model: {self.model_name}, Host: {self.host})"

    @property
    def client(self):
        """
        Provides access to the underlying Ollama SDK client for backwards compatibility.
        """
        return self._client


