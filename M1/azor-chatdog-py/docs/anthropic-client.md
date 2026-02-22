# Anthropic Client (Claude) — Deep Dive

## Overview

The Anthropic client (`src/llm/anthropic_client.py`) connects the Azor application to **Anthropic's Claude** language models via the official `anthropic` Python SDK. Like the Gemini client, this is a cloud-based integration that requires an API key and internet connectivity.

This client is architecturally the most similar to the Gemini client (both are cloud API clients with API keys), but Anthropic's Messages API has several unique design choices that require specific handling.

---

## Architecture

    ChatSession  ─>  AnthropicClient  ─>  AnthropicChatSession  ─>  anthropic SDK  ─>  HTTPS  ─>  Anthropic API

**Key similarities with Gemini**: Cloud-based, API key required, structured message format.
**Key differences from Gemini**: System prompt handling, role naming, response structure, token counting approach.

---

## Key Concepts

### 1. System Prompt as a Top-Level Parameter

This is the most important architectural difference from other clients.

**Gemini**: System instruction is part of `GenerateContentConfig`, embedded in the session configuration.
**Ollama**: System instruction is a `{"role": "system", ...}` message in the messages array.
**Anthropic**: System instruction is a **separate top-level parameter** in the API call:

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_instruction,    # <-- TOP-LEVEL, not in messages[]
        messages=messages,            # <-- Only user/assistant messages here
    )

This means `_build_messages_from_history()` does **NOT** include the system prompt in the messages list. It is passed separately.

**Why does Anthropic do this?** Their API treats the system prompt as configuration, not conversation. This prevents prompt injection attacks where a user might try to override the system prompt by injecting a `{"role": "system", ...}` message.

### 2. Role Mapping: "model" vs. "assistant"

The universal history format uses `"role": "model"` (Gemini's convention). Anthropic requires `"role": "assistant"`. The conversion happens in `_build_messages_from_history()`:

    if role == "model":
        anthropic_role = "assistant"

This mapping is also needed in `count_history_tokens()` when building messages for the token counting API.

### 3. Response Content Blocks

Anthropic responses use a **content block** structure instead of a simple text field:

    response.content = [
        TextBlock(type="text", text="Hello! How can I help?"),
        # Could also contain tool_use blocks, etc.
    ]

The session extracts text by iterating over content blocks:

    response_text = ""
    for block in response.content:
        if block.type == "text":
            response_text += block.text

This is more complex than Gemini's `response.text` but more flexible (supports mixed content types).

### 4. Exact Token Counting via API

Anthropic provides two ways to know token usage:

**Per-response usage metadata** (tracked in `AnthropicChatSession`):

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

**Dedicated count_tokens API** (used in `AnthropicClient.count_history_tokens()`):

    response = client.messages.count_tokens(
        model=self.model_name,
        messages=messages,
    )
    return response.input_tokens

The client tries the dedicated API first and falls back to a word-count heuristic (~0.75 tokens/word) if the API call fails.

---

## Class-by-Class Breakdown

### `AnthropicResponse`

Compatibility shim with **extra metadata**:

    class AnthropicResponse:
        def __init__(self, text, input_tokens=0, output_tokens=0):
            self.text = text
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

Unlike `LlamaResponse` and `OllamaResponse` which only carry `.text`, `AnthropicResponse` also tracks token usage per response. The chat loop in `chat.py` only uses `.text`, but the extra fields are available for future use.

### `AnthropicChatSession`

**Purpose**: Manages conversation history and communicates with the Anthropic Messages API. Provides the same `send_message()` / `get_history()` interface.

#### `__init__(self, anthropic_client, model_name, system_instruction, max_tokens, history)`

Stores the client, model name, system instruction, max tokens per response, and initial history. Also initializes token tracking counters: `_total_input_tokens` and `_total_output_tokens`.

#### `send_message(self, text) -> AnthropicResponse`

Step-by-step:

1. Appends user message to history in universal format
2. Calls `_build_messages_from_history()` to convert to Anthropic format
3. Calls `self.anthropic_client.messages.create(model=..., max_tokens=..., system=..., messages=...)`
4. Iterates over `response.content` blocks, extracting text from `TextBlock` types
5. Records `response.usage.input_tokens` and `response.usage.output_tokens`
6. Appends assistant response to history in universal format
7. Returns `AnthropicResponse(response_text, input_tokens, output_tokens)`

**Key detail**: The `system` parameter is passed as a **top-level argument**, not inside `messages[]`.

#### `_build_messages_from_history(self) -> List[Dict[str, str]]`

Converts universal history to Anthropic's format:
- Maps `"model"` → `"assistant"` role
- Flattens `"parts": [{"text": "..."}]` → `"content": "..."`
- **Does NOT include system instruction** (it is a separate API parameter)

#### `total_tokens` property

Returns cumulative `_total_input_tokens + _total_output_tokens` across all messages in the session. Useful for cost tracking.

### `AnthropicClient`

The main client class. Manages the Anthropic SDK client and session creation.

#### Constructor: `__init__(self, model_name, api_key, max_tokens)`

Takes explicit parameters:
- `model_name`: Claude model name (e.g., "claude-sonnet-4-20250514")
- `api_key`: Anthropic API key (validated non-empty)
- `max_tokens`: Maximum response tokens (default: 4096)

Creates an `Anthropic(api_key=...)` instance during construction. Unlike Gemini which reads the API key from env vars automatically, the Anthropic SDK requires the key to be passed explicitly.

#### `from_environment(cls) -> AnthropicClient`

Factory method reading from env vars via Pydantic `AnthropicConfig`:
- `ANTHROPIC_MODEL_NAME` (default: "claude-sonnet-4-20250514")
- `ANTHROPIC_API_KEY` (required, validated non-empty)
- `ANTHROPIC_MAX_TOKENS` (default: 4096)

#### `_initialize_client(self) -> Anthropic`

    return Anthropic(api_key=self.api_key)

Simple and direct. The API key is passed explicitly (unlike Gemini which uses env vars internally). Calls `sys.exit(1)` on failure, matching Gemini's behavior.

#### `create_chat_session(self, system_instruction, history, thinking_budget)`

Creates an `AnthropicChatSession`. The `thinking_budget` parameter is accepted but **ignored** (Gemini-specific, kept for interface compatibility).

**Note**: `max_tokens` is passed through to the session, which uses it in every `messages.create()` call. This is Anthropic-specific — you must specify `max_tokens` for every request (unlike Gemini which has defaults).

#### `count_history_tokens(self, history) -> int`

Two-tier approach:

1. **Primary**: Uses `client.messages.count_tokens(model=..., messages=...)` for exact counting
2. **Fallback**: Word-count heuristic (~0.75 tokens/word) if the API call fails

The primary method converts universal history to Anthropic message format first (same role mapping as `_build_messages_from_history()`).

#### `ready_for_use_message(self) -> str`

Displays the model name and **masked** API key, identical pattern to Gemini:

    "sk-a...xyz1"  (first 4 + last 4 characters)

---

## Pydantic Validation (`anthropic_validation.py`)

    class AnthropicConfig(BaseModel):
        engine: Literal["ANTHROPIC"] = Field(default="ANTHROPIC")
        model_name: str = Field(...)
        anthropic_api_key: str = Field(..., min_length=1)
        max_tokens: int = Field(default=4096, ge=1)

Key validations:
- API key must be non-empty (and stripped of whitespace)
- Model name must be non-empty
- Max tokens must be >= 1

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ENGINE` | Yes | `GEMINI` | Must be set to `ANTHROPIC` |
| `ANTHROPIC_API_KEY` | Yes | — | Your Anthropic API key |
| `ANTHROPIC_MODEL_NAME` | No | `claude-sonnet-4-20250514` | Claude model to use |
| `ANTHROPIC_MAX_TOKENS` | No | `4096` | Max tokens per response |

---

## Comparison: Anthropic vs. Gemini (Both Cloud APIs)

| Aspect | Gemini | Anthropic |
|--------|--------|-----------|
| SDK | `google-genai` | `anthropic` |
| API key handling | SDK reads env var internally | Passed explicitly to `Anthropic(api_key=...)` |
| System prompt | In `GenerateContentConfig` | Top-level `system=` parameter |
| Assistant role name | `"model"` | `"assistant"` (requires mapping) |
| Response format | `response.text` (direct) | `response.content[0].text` (content blocks) |
| Token counting | `models.count_tokens` API | `messages.count_tokens` API + `usage` metadata |
| Thinking budget | Supported via `ThinkingConfig` | Ignored (compatibility param) |
| `max_tokens` | Optional (has defaults) | **Required** per request |
| Session wrapper | Converts Content objects | Stateless (full history per call) |
| History conversion | Both directions (dict ↔ Content) | One direction (dict → Anthropic format) |

---

## Key Takeaways

1. **System prompt is NOT a message**: Anthropic treats it as configuration, passed as a top-level `system=` parameter. This is a deliberate security design choice.
2. **Role mapping is critical**: The universal `"model"` role must be converted to `"assistant"` for every API call and token count.
3. **Content blocks, not text**: Responses contain typed blocks (`TextBlock`, `ToolUseBlock`, etc.). Must iterate and filter for `type == "text"`.
4. **`max_tokens` is mandatory**: Unlike Gemini, Anthropic requires you to specify the maximum response length on every request.
5. **Two-tier token counting**: Primary API-based counting with heuristic fallback provides both accuracy and resilience.
6. **Token usage tracking**: `AnthropicResponse` carries `input_tokens` and `output_tokens` metadata, enabling precise cost tracking.
7. **Explicit API key**: The Anthropic SDK does not auto-read env vars — the key must be passed to the constructor, giving you more control over configuration.
