# Gemini LLM Client — Deep Dive

## Overview

The **Gemini client** (`src/llm/gemini_client.py`) connects this application to **Google's Gemini API** — a cloud-based large language model service. It is the original LLM backend of the Azor chatdog application and serves as the reference implementation that all other clients follow.

This document explains every architectural decision, every class, and every method so you can deeply understand how a cloud-based LLM client integrates into a Python chat application.

---

## Architecture at a Glance

```
┌──────────────────────────────────────────────────┐
│                  ChatSession                     │
│         (src/session/chat_session.py)            │
│                                                  │
│  Calls: create_chat_session()                    │
│         count_history_tokens()                   │
│         get_model_name()                         │
└──────────────┬───────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────┐
│              GeminiLLMClient                     │
│         (src/llm/gemini_client.py)               │
│                                                  │
│  Wraps: google.genai.Client                      │
│  Hides: google.genai.types (Content, Part, etc.) │
└──────────────┬───────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────┐
│          GeminiChatSessionWrapper                │
│                                                  │
│  Wraps: genai chat session object                │
│  Converts: Gemini Content → universal dict       │
└──────────────┬───────────────────────────────────┘
               │  HTTP/REST
               ▼
┌──────────────────────────────────────────────────┐
│         Google Gemini API (Cloud)                │
│         models: gemini-2.5-flash, etc.           │
└──────────────────────────────────────────────────┘
```

---

## Key Concepts

### 1. The Universal History Format

The application uses a **universal dictionary format** for conversation history that all clients must produce and consume:

```python
{
    "role": "user" | "model",
    "parts": [{"text": "message content here"}]
}
```

This was inspired by Google Gemini's native `Content` objects but simplified to plain Python dicts. The Gemini client is unique because it must **convert between** this universal format and Gemini's proprietary `types.Content` objects.

### 2. Why a Wrapper?

Google's Gemini SDK returns history as `types.Content` objects (rich, SDK-specific classes). Other clients (Llama, Ollama, Anthropic) work with plain dicts. To keep the rest of the application SDK-agnostic, `GeminiChatSessionWrapper` translates:

- **Outgoing**: Universal `dict` history → `types.Content` objects (when creating a session)
- **Incoming**: `types.Content` history → Universal `dict` format (when reading history)

---

## Class-by-Class Breakdown

### `GeminiChatSessionWrapper`

**Purpose**: Sits between `ChatSession` and the actual Gemini SDK chat session. Ensures that `get_history()` always returns universal dicts, not Gemini-specific objects.

#### `__init__(self, gemini_session)`
Stores the raw Gemini chat session. No conversion happens here — it's lazy.

#### `send_message(self, text: str)`
Simply delegates to `gemini_session.send_message(text)`. The Gemini SDK handles adding the message to its internal history, sending the HTTP request, and returning a response object with a `.text` attribute.

#### `get_history(self) -> List[Dict]`
This is where the real work happens:

```python
gemini_history = self.gemini_session.get_history()  # Returns List[types.Content]
universal_history = []

for content in gemini_history:
    # Extract text from the Content object's parts
    text_part = ""
    if hasattr(content, 'parts') and content.parts:
        for part in content.parts:
            if hasattr(part, 'text') and part.text:
                text_part = part.text
                break  # Only take the first text part

    if text_part:
        universal_content = {
            "role": content.role,       # "user" or "model"
            "parts": [{"text": text_part}]
        }
        universal_history.append(universal_content)
```

**Key learning**: Gemini `Content` objects can have multiple `Part` objects (text, images, function calls, etc.). This app only uses text, so we extract the first text part and ignore everything else.

---

### `GeminiLLMClient`

**Purpose**: The main client class. Manages the Google GenAI SDK client, creates chat sessions, and counts tokens.

#### Constructor: `__init__(self, model_name: str, api_key: str)`

Takes **explicit** parameters (not environment variables). This is a deliberate design choice:

```python
def __init__(self, model_name: str, api_key: str):
    if not api_key:
        raise ValueError("API key cannot be empty or None")

    self.model_name = model_name
    self.api_key = api_key
    self._client = self._initialize_client()  # Eagerly initializes
```

**Why explicit params?** It makes the class testable. You can pass mock values without touching environment variables. The `from_environment()` factory method handles env var loading separately.

#### Factory: `from_environment(cls) -> GeminiLLMClient`

This is a `@classmethod` — a factory that reads `.env` and creates an instance:

```python
@classmethod
def from_environment(cls) -> 'GeminiLLMClient':
    load_dotenv()
    config = GeminiConfig(
        model_name=os.getenv('MODEL_NAME', 'gemini-2.5-flash'),
        gemini_api_key=os.getenv('GEMINI_API_KEY', '')
    )
    return cls(model_name=config.model_name, api_key=config.gemini_api_key)
```

**How it's called**: `ChatSession._initialize_llm_session()` calls `GeminiLLMClient.from_environment()` when `ENGINE=GEMINI`.

**Pydantic validation**: `GeminiConfig` validates that the API key is non-empty and strips whitespace. If validation fails, Pydantic raises `ValidationError` before the client is ever created.

#### `_initialize_client(self) -> genai.Client`

Creates the Google GenAI SDK client:

```python
def _initialize_client(self) -> genai.Client:
    try:
        return genai.Client()
    except Exception as e:
        console.print_error(f"Błąd inicjalizacji klienta Gemini: {e}")
        sys.exit(1)
```

**Note**: `genai.Client()` reads the API key from the `GEMINI_API_KEY` environment variable automatically (set by `load_dotenv()` earlier). The `self.api_key` is stored for display/validation purposes, but the SDK uses the env var directly.

#### `create_chat_session(self, system_instruction, history, thinking_budget)`

This is the heart of the Gemini integration. It:

1. **Converts** universal dict history → Gemini `types.Content` objects
2. **Creates** a Gemini chat session with system instruction and thinking config
3. **Wraps** it in `GeminiChatSessionWrapper`

```python
# Step 1: Convert history format
gemini_history = []
if history:
    for entry in history:
        if isinstance(entry, dict) and 'role' in entry and 'parts' in entry:
            text = entry['parts'][0].get('text', '') if entry['parts'] else ''
            if text:
                content = types.Content(
                    role=entry['role'],
                    parts=[types.Part.from_text(text=text)]
                )
                gemini_history.append(content)

# Step 2: Create Gemini session
gemini_session = self._client.chats.create(
    model=self.model_name,
    history=gemini_history,
    config=types.GenerateContentConfig(
        system_instruction=system_instruction,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
    )
)

# Step 3: Wrap it
return GeminiChatSessionWrapper(gemini_session)
```

**`thinking_budget`**: Gemini supports "thinking" — the model can reason internally before responding. `thinking_budget=0` disables this (faster, cheaper). Higher values allow more reasoning tokens.

#### `count_history_tokens(self, history: List[Dict]) -> int`

Uses the **Gemini-specific** `models.count_tokens` API for accurate token counting:

```python
response = self._client.models.count_tokens(
    model=self.model_name,
    contents=gemini_history  # Converted to types.Content objects
)
return response.total_tokens
```

**Why this matters**: Unlike other clients that estimate tokens, Gemini provides an **exact** count via its API. This means the token display in the chat is perfectly accurate when using the Gemini engine.

#### `ready_for_use_message(self) -> str`

Displays the model name and a **masked** API key:

```python
if len(self.api_key) <= 8:
    masked_key = "****"
else:
    masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}"
```

This shows enough of the key to identify which one is in use, without exposing the full secret.

---

## Pydantic Validation (`gemini_validation.py`)

```python
class GeminiConfig(BaseModel):
    engine: Literal["GEMINI"] = Field(default="GEMINI")
    model_name: str = Field(..., description="Nazwa modelu Gemini")
    gemini_api_key: str = Field(..., min_length=1, description="Klucz API Google Gemini")

    @validator('gemini_api_key')
    def validate_api_key(cls, v):
        if not v or v.strip() == "":
            raise ValueError("GEMINI_API_KEY nie może być pusty")
        return v.strip()
```

**What you learn here**:
- `Field(...)` means the field is **required** (no default)
- `min_length=1` catches empty strings at the schema level
- The `@validator` adds custom logic: strip whitespace, reject blank strings
- `Literal["GEMINI"]` ensures the engine field can only be "GEMINI"

---

## How It's Wired Into the Application

### Engine Selection (`chat_session.py`)

```python
ENGINE_MAPPING = {
    'GEMINI': GeminiLLMClient,
    'LLAMA_CPP': LlamaClient,
    'OLLAMA': OllamaClient,
    'ANTHROPIC': AnthropicClient,
}
```

When `ENGINE=GEMINI` in `.env`:
1. `ChatSession._initialize_llm_session()` reads `ENGINE` from env
2. Looks up `GeminiLLMClient` from `ENGINE_MAPPING`
3. Calls `GeminiLLMClient.preparing_for_use_message()` → displays "🤖 Przygotowywanie klienta Gemini..."
4. Calls `GeminiLLMClient.from_environment()` → validates config, creates client
5. Calls `client.ready_for_use_message()` → displays "✅ Klient Gemini gotowy do użycia..."
6. Calls `client.create_chat_session(...)` → creates the wrapped Gemini session

### The Interface Contract

Every LLM client must provide these methods (duck-typing):

| Method | Purpose |
|--------|---------|
| `preparing_for_use_message()` | Static method, returns loading message |
| `from_environment()` | Class method, creates instance from env vars |
| `ready_for_use_message()` | Returns ready confirmation message |
| `create_chat_session(system_instruction, history, thinking_budget)` | Returns a session with `send_message()` and `get_history()` |
| `count_history_tokens(history)` | Returns token count for history |
| `get_model_name()` | Returns model name string |
| `is_available()` | Returns True if client is usable |

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ENGINE` | Yes | `GEMINI` | Must be set to `GEMINI` |
| `GEMINI_API_KEY` | Yes | — | Your Google AI API key |
| `MODEL_NAME` | No | `gemini-2.5-flash` | Gemini model to use |

---

## Key Takeaways

1. **Wrapper pattern**: `GeminiChatSessionWrapper` isolates SDK-specific types from the rest of the app
2. **Factory pattern**: `from_environment()` separates configuration loading from object construction
3. **Exact token counting**: Gemini is the only client with API-based token counting (others estimate)
4. **Eager initialization**: The client connects to the API during construction, failing fast if credentials are wrong
5. **Thinking budget**: A Gemini-specific feature that controls how much the model "thinks" before responding

