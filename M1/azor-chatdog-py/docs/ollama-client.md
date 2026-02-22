# Ollama Client — Deep Dive

## Overview

The Ollama client (`src/llm/ollama_client.py`) connects the Azor application to models served by an **Ollama server** running on your local machine (or network). Ollama is a model-serving platform that downloads, manages, and serves LLM models through a simple HTTP API.

This client is architecturally similar to the LLaMA client (both use local models) but differs fundamentally in **how** it communicates: via HTTP requests to a server process, rather than loading the model directly into the Python process.

---

## Architecture

    ChatSession  ─>  OllamaClient  ─>  OllamaChatSession  ─>  ollama SDK  ─>  HTTP  ─>  Ollama Server  ─>  Model

**Key insight**: The Ollama server is a separate process that:
1. Downloads and manages model files for you (no manual `.gguf` downloads)
2. Loads models into memory and serves them via REST API
3. Handles prompt formatting, tokenization, and inference internally

This means:
- You need the Ollama server running (`ollama serve`)
- You need a model pulled (`ollama pull qwen2.5:7b-instruct`)
- No API key needed (local server)
- The Python SDK (`ollama` package) is a thin HTTP client

---

## Key Concepts

### 1. Ollama Server Architecture

Unlike `llama-cpp-python` which loads models **into your Python process**, Ollama runs as a **separate server process**:

    [Your Python App] --HTTP--> [Ollama Server :11434] --loads--> [Model in Server Memory]

Benefits:
- **Model management**: `ollama pull`, `ollama list`, `ollama rm` manage models
- **Shared access**: Multiple applications can use the same server
- **Memory isolation**: If the model crashes, your app survives
- **Hot-swapping**: Switch models without restarting your app

### 2. Message Format Conversion

Ollama's chat API uses a message format similar to OpenAI:

    {"role": "system"|"user"|"assistant", "content": "..."}

The app's universal format uses:

    {"role": "user"|"model", "parts": [{"text": "..."}]}

`OllamaChatSession._build_messages_from_history()` converts between them:
- `"model"` → `"assistant"` (role name mapping)
- `"parts": [{"text": "..."}]` → `"content": "..."` (structure flattening)
- System instruction is prepended as a `{"role": "system", ...}` message

### 3. Non-Streaming Mode

The Ollama SDK supports `stream=True` for token-by-token streaming responses. This implementation uses `stream=False` for **compatibility with the existing `response.text` pattern** used by all other clients. The response is received as a complete string.

Future enhancement: streaming could be added for a more interactive feel, but would require changes to the chat loop in `chat.py`.

---

## Class-by-Class Breakdown

### `OllamaResponse`

Compatibility shim, identical in purpose to `LlamaResponse`:

    class OllamaResponse:
        def __init__(self, text: str):
            self.text = text

Ensures Ollama responses match the `response.text` interface expected by `chat.py`.

### `OllamaChatSession`

**Purpose**: Manages conversation history and communicates with the Ollama server via the chat API. Provides the same `send_message()` / `get_history()` interface as all other session wrappers.

#### `__init__(self, ollama_client, model_name, system_instruction, history)`

Stores the Ollama SDK client, model name, system instruction, and initial history. Unlike `LlamaChatSession`, this also stores `model_name` because Ollama needs it for each API call (the server can serve multiple models).

#### `send_message(self, text) -> OllamaResponse`

Step-by-step:

1. Appends user message to history in universal format
2. Calls `_build_messages_from_history()` to convert to Ollama format
3. Calls `self.ollama_client.chat(model=..., messages=..., stream=False)`
4. Extracts response text from `response['message']['content']`
5. Appends assistant response to history in universal format
6. Returns `OllamaResponse(response_text)`

**Note**: The Ollama SDK client is **stateless** — every `chat()` call sends the full conversation history. There's no server-side session. The `OllamaChatSession` manages state locally.

#### `_build_messages_from_history(self) -> List[Dict[str, str]]`

Converts universal history to Ollama's expected format:

1. **System instruction first**: Adds `{"role": "system", "content": system_instruction}` as the first message
2. **Role mapping**: Converts `"model"` → `"assistant"` for each history entry
3. **Structure flattening**: Converts `"parts": [{"text": "..."}]` → `"content": "..."`

This method is called on every `send_message()` because the full history must be sent each time (stateless API).

### `OllamaClient`

The main client class. Manages the Ollama SDK client and session creation.

#### Constructor: `__init__(self, model_name, host)`

- `model_name`: Name of the Ollama model (e.g., "qwen2.5:7b-instruct", "llama3.1", "mistral")
- `host`: URL of the Ollama server (default: "http://localhost:11434")

Creates an `ollama.Client(host=...)` instance during construction.

#### `from_environment(cls) -> OllamaClient`

Factory method reading from env vars via Pydantic `OllamaConfig`:
- `OLLAMA_MODEL_NAME` (default: "qwen2.5:7b-instruct")
- `OLLAMA_HOST` (default: "http://localhost:11434")

Validated: host must start with `http://` or `https://`, model name must be non-empty.

#### `create_chat_session(self, system_instruction, history, thinking_budget)`

Creates an `OllamaChatSession`. The `thinking_budget` parameter is accepted but **ignored** (Gemini-specific, kept for interface compatibility).

#### `count_history_tokens(self, history) -> int`

**Heuristic-based estimation** — Ollama does not provide a standalone token-counting API:

    word_count = len(full_text.split())
    return int(word_count * 0.75)

Uses ~0.75 tokens per word as an approximation (English average). Falls back to 4 chars per token on error.

**Note**: Ollama does return `eval_count` and `prompt_eval_count` in chat responses, which could be accumulated for more accurate counting. This is a potential future improvement.

#### `is_available(self) -> bool`

Uniquely among all clients, this method **actually checks server connectivity**:

    try:
        self._client.list()  # HTTP call to Ollama server
        return True
    except Exception:
        return False

This is useful because the Ollama server might not be running, unlike Gemini (which validates the API key) or LLaMA (which validates the file path).

---

## Pydantic Validation (`ollama_validation.py`)

    class OllamaConfig(BaseModel):
        engine: Literal["OLLAMA"] = Field(default="OLLAMA")
        model_name: str = Field(...)
        ollama_host: str = Field(default="http://localhost:11434")

Key validations:
- Model name must be non-empty
- Host must start with `http://` or `https://`
- Trailing slashes are stripped from the host URL

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ENGINE` | Yes | `GEMINI` | Must be set to `OLLAMA` |
| `OLLAMA_MODEL_NAME` | No | `qwen2.5:7b-instruct` | Name of the Ollama model |
| `OLLAMA_HOST` | No | `http://localhost:11434` | Ollama server URL |

---

## Comparison: Ollama vs. LLaMA (llama-cpp-python)

Both use local models, but the approach is fundamentally different:

| Aspect | LLaMA (llama-cpp-python) | Ollama |
|--------|--------------------------|--------|
| Model loading | In-process (Python loads .gguf) | Separate server process |
| Model management | Manual .gguf downloads | `ollama pull` / `ollama rm` |
| Communication | Direct function call | HTTP REST API |
| Prompt format | Raw text string | Structured messages (like OpenAI) |
| Token counting | Exact (local tokenizer) | Heuristic estimation |
| Multi-app access | Single process only | Shared server |
| Memory isolation | Model crashes = app crashes | Model crashes = server crashes, app survives |
| Setup complexity | Download .gguf, set path | Install Ollama, `ollama pull model` |
| GPU control | `n_gpu_layers` parameter | Ollama handles automatically |

---

## Setup Guide

1. **Install Ollama**: https://ollama.ai
2. **Start the server**: `ollama serve` (runs on port 11434 by default)
3. **Pull a model**: `ollama pull qwen2.5:7b-instruct` (or `llama3.1`, `mistral`, `codellama`, etc.)
4. **Configure `.env`**:

        ENGINE=OLLAMA
        OLLAMA_MODEL_NAME=qwen2.5:7b-instruct
        OLLAMA_HOST=http://localhost:11434

5. **Run the app**: `python src/run.py`

---

## Key Takeaways

1. **HTTP-based**: Ollama communicates via HTTP, making it a thin client — the heavy lifting is in the Ollama server
2. **Stateless API**: Every `chat()` call sends the full conversation history (no server-side sessions)
3. **Message format**: Uses OpenAI-compatible `{role, content}` format, requiring `"model"` → `"assistant"` role mapping
4. **System instruction as message**: Unlike Anthropic (top-level param), Ollama includes the system prompt as the first message in the array
5. **Heuristic token counting**: No dedicated token-counting API; uses word-count approximation
6. **Server health check**: `is_available()` actually pings the server, making it robust for detecting connectivity issues
7. **Model management built-in**: Ollama handles model downloads, updates, and cleanup — much simpler than manual `.gguf` management
