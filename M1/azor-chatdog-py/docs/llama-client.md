# LLaMA Client (llama-cpp-python) — Deep Dive

## Overview

The LLaMA client (`src/llm/llama_client.py`) connects the Azor application to a **locally running LLaMA model** via the `llama-cpp-python` library. Unlike the Gemini client which calls a cloud API, this client loads a model file directly into memory and runs inference on your own hardware (CPU or GPU).

---

## Architecture

    ChatSession  ─>  LlamaClient  ─>  LlamaChatSession  ─>  llama_cpp.Llama  ─>  .gguf file on disk

**Key difference from Gemini**: No network calls. The model runs **in-process**:
- No API key needed — authentication is replaced by a file path
- No internet required — fully offline capable
- Inference speed depends on your hardware (CPU cores, GPU VRAM)
- Model must be downloaded as a `.gguf` file beforehand

---

## Key Concepts

### 1. GGUF Model Files

`llama-cpp-python` is a Python binding for `llama.cpp`, a C/C++ library that runs quantized LLaMA-family models. Models use the **GGUF** (GPT-Generated Unified Format) file format.

You download a `.gguf` file (often several GB) and point `LLAMA_MODEL_PATH` to it:

    LLAMA_MODEL_PATH=/path/to/llama-3.1-8b-instruct.Q4_K_M.gguf

The filename encodes: model name, parameter count, and quantization level (e.g., `Q4_K_M` = 4-bit quantization, K-quant, medium quality). Lower quantization = smaller file + less RAM, but slightly lower quality.

### 2. Raw Text Prompt Building

Cloud APIs (Gemini, Anthropic, Ollama) accept **structured messages** — arrays of `{role, content}` objects. The API handles prompt formatting internally.

`llama-cpp-python`'s basic completion API accepts a **raw text prompt** — a single string. `LlamaChatSession._build_prompt_from_history()` manually constructs this:

    System: You are a helpful assistant named Azor...

    User: Hello!

    Assistant: Hi there! How can I help you?

    User: What is Python?

    Assistant:

The model generates text starting from where `Assistant:` ends. This is the fundamental difference — the LLaMA client must do prompt engineering that cloud APIs handle automatically.

### 3. GPU Layer Offloading

The `n_gpu_layers` parameter controls how many transformer layers run on GPU vs. CPU:

| Value | Meaning |
|-------|---------|
| `0` | Pure CPU inference (slowest, works everywhere) |
| `1` | One layer on GPU (minimal GPU usage) |
| `-1` | All layers on GPU (fastest, needs enough VRAM) |

Configured via `LLAMA_GPU_LAYERS` environment variable.

---

## Class-by-Class Breakdown

### `LlamaResponse`

The simplest class — just a compatibility shim (Adapter pattern):

    class LlamaResponse:
        def __init__(self, text: str):
            self.text = text

**Why does this exist?** The Gemini SDK returns response objects with a `.text` attribute. The chat loop in `chat.py` does `response.text`. `LlamaResponse` ensures LLaMA responses have the same interface. Without it, you would need `if/else` checks everywhere in the calling code.

### `LlamaChatSession`

**Purpose**: Provides the same `send_message()` / `get_history()` interface as `GeminiChatSessionWrapper`, but builds raw text prompts for the local model.

#### `__init__(self, llama_model, system_instruction, history)`

Stores the Llama model instance, system instruction, and initial history. History is stored as universal dicts (same `{"role", "parts"}` format as Gemini).

#### `send_message(self, text) -> LlamaResponse`

Step-by-step:

1. Appends user message to history: `{"role": "user", "parts": [{"text": text}]}`
2. Calls `_build_prompt_from_history()` to construct the raw prompt string
3. Calls `self.llama_model(prompt, ...)` — this is the actual inference call
4. Extracts response text from `output["choices"][0]["text"]`
5. Appends assistant response to history: `{"role": "model", "parts": [{"text": response_text}]}`
6. Returns `LlamaResponse(response_text)`

The inference call parameters:
- **`max_tokens=512`**: Maximum response length
- **`stop=["User:", "Assistant:", ...]`**: Stop sequences prevent the model from generating the next turn
- **`echo=False`**: Do not include the prompt in the output

#### `_build_prompt_from_history(self) -> str`

This is the most critical method. It manually formats the conversation:

1. Adds `"System: {system_instruction}"` at the top
2. Iterates through all history entries **except the last one**
3. Maps `"user"` role to `"User: {text}"` and `"model"` role to `"Assistant: {text}"`
4. Adds the current (last) user message as `"User: {text}"`
5. Ends with `"Assistant:"` (no text — the model generates from here)
6. Joins everything with double newlines

**Why exclude the last message in step 2?** Because step 4 adds it separately. This prevents duplicate entries.

#### `get_history(self) -> List[Dict]`

Simply returns `self._history`. Unlike `GeminiChatSessionWrapper` which must convert Content objects, `LlamaChatSession` already stores history in universal dict format natively.

### `LlamaClient`

The main client class. Manages model loading and session creation.

#### Constructor: `__init__(self, model_name, model_path, n_gpu_layers, n_ctx)`

Takes **explicit** parameters (not env vars — testability!):
- `model_name`: Display name (e.g., "llama-3.1-8b-instruct")
- `model_path`: Path to .gguf file (validated to exist)
- `n_gpu_layers`: GPU layer count (default: 1)
- `n_ctx`: Context window size in tokens (default: 2048)

Validates that `model_path` is non-empty and the file exists. Then **eagerly initializes** the model (loading into memory immediately).

#### `from_environment(cls) -> LlamaClient`

Factory method reading from env vars via Pydantic `LlamaConfig`:
- `LLAMA_MODEL_NAME` (default: "llama-3.1-8b-instruct")
- `LLAMA_MODEL_PATH` (required — validated to exist and end with `.gguf`)
- `LLAMA_GPU_LAYERS` (default: 1)
- `LLAMA_CONTEXT_SIZE` (default: 2048)

#### `_initialize_model(self) -> Llama`

Creates the actual `llama_cpp.Llama` instance:

    Llama(
        model_path=self.model_path,
        n_gpu_layers=self.n_gpu_layers,
        n_ctx=self.n_ctx,
        verbose=False
    )

This is the **heaviest** operation — loading a multi-GB model into memory. `verbose=False` suppresses `llama.cpp` diagnostic output. The model is created once and reused for all sessions.

#### `create_chat_session(self, system_instruction, history, thinking_budget)`

Creates a `LlamaChatSession`. The `thinking_budget` parameter is accepted but **ignored** (Gemini-specific feature kept for interface compatibility).

#### `count_history_tokens(self, history) -> int`

Uses the model's **built-in tokenizer** for accurate counting:

    tokens = self._llama_model.tokenize(full_text.encode("utf-8"))
    return len(tokens)

This is more accurate than heuristics because it uses the exact same tokenizer the model uses for inference. Falls back to rough estimation (4 chars per token) on error.

---

## Pydantic Validation (`llama_validation.py`)

    class LlamaConfig(BaseModel):
        engine: Literal["LLAMA"] = Field(default="LLAMA")
        model_name: str = Field(...)
        llama_model_path: str = Field(...)
        llama_gpu_layers: int = Field(default=1, ge=0)
        llama_context_size: int = Field(default=2048, ge=1)

Key validations:
- Model path must point to an **existing file**
- File must have `.gguf` extension
- GPU layers must be >= 0
- Context size must be >= 1

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ENGINE` | Yes | `GEMINI` | Must be set to `LLAMA_CPP` |
| `LLAMA_MODEL_NAME` | No | `llama-3.1-8b-instruct` | Display name for the model |
| `LLAMA_MODEL_PATH` | Yes | — | Absolute path to .gguf file |
| `LLAMA_GPU_LAYERS` | No | `1` | Number of GPU layers to offload |
| `LLAMA_CONTEXT_SIZE` | No | `2048` | Context window size (tokens) |

---

## Comparison with Gemini Client

| Aspect | Gemini | LLaMA |
|--------|--------|-------|
| Location | Cloud API | Local machine |
| Authentication | API key | File path |
| Prompt format | Structured messages (Content objects) | Raw text string |
| History storage | Converts to/from Content objects | Native universal dicts |
| Token counting | Exact via API (`models.count_tokens`) | Exact via local tokenizer |
| Session wrapper | `GeminiChatSessionWrapper` | `LlamaChatSession` |
| Response wrapper | Native SDK response (has `.text`) | `LlamaResponse` shim |
| Thinking budget | Supported | Ignored (compatibility param) |
| Cost | Per-token API pricing | Hardware electricity only |

---

## Setup Guide

1. **Install llama-cpp-python**:

        python3 -m pip install llama-cpp-python

   For GPU acceleration (Metal on macOS, CUDA on Linux/Windows), install with build flags:

        # macOS (Metal)
        CMAKE_ARGS="-DGGML_METAL=on" python3 -m pip install llama-cpp-python

        # Linux/Windows (CUDA)
        CMAKE_ARGS="-DGGML_CUDA=on" python3 -m pip install llama-cpp-python

2. **Download a GGUF model file**: Browse [Hugging Face](https://huggingface.co/models?search=gguf) for quantized models. Example:

        # Install the Hugging Face Hub library (if not already installed)
        curl -LsSf https://hf.co/cli/install.sh | bash

        # Download the model (saves into ./models in the current directory)
        hf download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir "$HOME/.models"

   If you downloaded the model via the Hugging Face website or another tool and don't know where it is:

        # Search your home directory for .gguf files
        find ~ -name "*.gguf" 2>/dev/null

   Choose a quantization level based on your hardware:

   | Quantization | File Size (7B) | RAM Needed | Quality |
   |-------------|----------------|------------|---------|
   | `Q2_K` | ~2.7 GB | ~5 GB | Lowest |
   | `Q4_K_M` | ~4.1 GB | ~7 GB | Good balance |
   | `Q5_K_M` | ~4.8 GB | ~8 GB | Higher quality |
   | `Q8_0` | ~7.2 GB | ~10 GB | Near-original |

3. **Configure `.env`**:

        ENGINE=LLAMA_CPP
        LLAMA_MODEL_NAME=llama-3.1-8b-instruct
        LLAMA_MODEL_PATH=/absolute/path/to/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
        LLAMA_GPU_LAYERS=1
        LLAMA_CONTEXT_SIZE=2048

   Adjust `LLAMA_GPU_LAYERS` based on your GPU VRAM:
   - `0` — CPU only (no GPU required)
   - `1` to `N` — offload N layers to GPU (more = faster, needs more VRAM)
   - `-1` — offload all layers to GPU (fastest, needs enough VRAM for the entire model)

4. **Run the app**: `python src/run.py`

**Note**: First launch will be slower as the model loads into memory. Subsequent messages within the same session are fast since the model stays loaded.

---

## Key Takeaways

1. **Prompt engineering is manual**: The LLaMA client formats conversation history into a raw text prompt with role prefixes and stop sequences
2. **Adapter pattern**: `LlamaResponse` wraps a plain string to match Gemini's `response.text` interface
3. **Tokenizer-based counting**: Uses the actual model tokenizer for exact token counts (not heuristics)
4. **GPU layer control**: `n_gpu_layers` lets you balance inference speed vs. VRAM usage
5. **Context window**: `n_ctx` limits total conversation length — smaller values use less RAM but limit conversation depth
6. **Heavy initialization**: Loading a GGUF model takes seconds and several GB of RAM — the client is created once and reused across all sessions
