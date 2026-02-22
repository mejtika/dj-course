# Zadanie 13 — Temperature, Top P, Top K w klientach LLM

## 🎯 Cel zadania

Umożliwić użytkownikowi konfigurację trzech kluczowych parametrów **samplowania** (sampling) we wszystkich klientach LLM działających w projekcie AZOR:

- **Temperature**
- **Top P** (nucleus sampling)
- **Top K**

Parametry te kontrolują **jak model wybiera kolejny token** podczas generowania tekstu. Są fundamentalne dla dostrajania zachowania modelu — od deterministycznego i precyzyjnego, po kreatywny i zaskakujący.

---

## 📖 Czym są te parametry? Głębokie wyjaśnienie

### Jak model generuje tekst — krok po kroku

Duży model językowy (LLM) generuje tekst **token po tokenie**. Na każdym kroku model oblicza **rozkład prawdopodobieństwa** nad całym słownikiem (vocabulary) — np. 32 000 tokenów. Każdy token dostaje pewne prawdopodobieństwo.

Przykład — model generuje następne słowo po "Pies jest":

| Token        | Raw logit | Prawdopodobieństwo (po softmax) |
|-------------|-----------|-------------------------------|
| wierny      | 5.2       | 0.35                          |
| duży        | 4.8       | 0.25                          |
| głodny      | 4.1       | 0.12                          |
| mądry       | 3.9       | 0.09                          |
| szybki      | 3.5       | 0.06                          |
| ... (32000) | ...       | ...                           |

**Bez żadnego samplowania** (greedy decoding) model zawsze wybierze "wierny" (najwyższe prawdopodobieństwo). Jest to deterministyczne, ale nudne i powtarzalne.

Parametry samplowania modyfikują ten rozkład **przed** wyborem tokena.

---

### 🌡️ Temperature

**Co robi**: Skaluje logity (raw scores) przed zastosowaniem softmax. Dzieli każdy logit przez wartość temperature.

**Wzór matematyczny**:

```
P(token_i) = exp(logit_i / T) / Σ exp(logit_j / T)
```

gdzie `T` = temperature.

**Efekt**:

| Temperature | Efekt na rozkład | Zachowanie modelu |
|------------|-----------------|-------------------|
| `0.0`      | Degeneruje do argmax (greedy) | Zawsze wybiera najwyżej oceniony token. Deterministyczne, powtarzalne. |
| `0.1–0.3`  | Bardzo spłaszczone szczyty | Prawie deterministyczne, minimalna wariacja |
| `0.5–0.7`  | Umiarkowane wygładzenie | Dobry balans: spójne, ale z lekką kreatywnością |
| `1.0`      | Brak modyfikacji (oryginał) | Domyślny rozkład prawdopodobieństw modelu |
| `1.2–1.5`  | Wyrównanie rozkładu | Więcej "zaskakujących" wyborów, mniej przewidywalne |
| `2.0`      | Prawie równomierny rozkład | Chaotyczne, często niespójne odpowiedzi |

**Analogia**: Temperature to jak "rozgrzewanie" kostki do gry. Niska temperatura = kostka prawie zawsze ląduje na jednej stronie. Wysoka temperatura = wszystkie strony równie prawdopodobne.

**Kiedy używać**:
- **Niska (0.0–0.3)**: Tłumaczenia, odpowiedzi na pytania faktograficzne, kod
- **Średnia (0.5–0.8)**: Dialog, asystent ogólnego przeznaczenia
- **Wysoka (1.0–1.5)**: Kreatywne pisanie, brainstorming, generowanie wielu wariantów

---

### 🎯 Top P (Nucleus Sampling)

**Co robi**: Zamiast rozważać wszystkie tokeny, Top P **obcina** rozkład do najmniejszego zestawu tokenów, których skumulowane prawdopodobieństwo przekracza wartość `p`.

**Algorytm**:
1. Posortuj tokeny malejąco wg prawdopodobieństwa
2. Dodawaj tokeny po kolei, sumując prawdopodobieństwa
3. Zatrzymaj się, gdy suma ≥ `top_p`
4. Wylosuj token z tego podzbioru (z renormalizacją)

**Przykład** (top_p = 0.7):

| Token   | P     | Skumulowane P | W zestawie? |
|---------|-------|---------------|-------------|
| wierny  | 0.35  | 0.35          | ✅          |
| duży    | 0.25  | 0.60          | ✅          |
| głodny  | 0.12  | 0.72          | ✅ (≥ 0.7)  |
| mądry   | 0.09  | 0.81          | ❌          |
| szybki  | 0.06  | 0.87          | ❌          |

Model losuje tylko z {wierny, duży, głodny} — 3 tokeny zamiast 32 000.

**Efekt wartości**:

| Top P | Efekt |
|-------|-------|
| `0.1` | Bardzo wąski zestaw (1-2 tokeny) — prawie greedy |
| `0.5` | Umiarkowany — kilka najlepszych tokenów |
| `0.9` | Szeroki zestaw — wiele opcji, ale wyklucza "ogon" rozkładu |
| `1.0` | Brak filtrowania — wszystkie tokeny uwzględnione |

**Dlaczego Top P jest lepszy niż stała temperatura?** Ponieważ adaptuje się do kontekstu. Gdy model jest "pewny" (jeden token dominuje), Top P wybiera mało tokenów. Gdy model jest "niepewny" (płaski rozkład), Top P dopuszcza więcej kandydatów.

---

### 🔢 Top K

**Co robi**: Ogranicza wybór do `K` tokenów o najwyższym prawdopodobieństwie. Reszta jest odrzucana.

**Algorytm**:
1. Posortuj tokeny malejąco wg prawdopodobieństwa
2. Weź pierwszych `K` tokenów
3. Wylosuj token z tego podzbioru (z renormalizacją)

**Przykład** (top_k = 3):

Model rozważa tylko: {wierny, duży, głodny} — dokładnie 3 tokeny, niezależnie od ich prawdopodobieństw.

**Efekt wartości**:

| Top K | Efekt |
|-------|-------|
| `1`   | Greedy decoding — zawsze najlepszy token |
| `10`  | Bardzo ograniczony wybór |
| `40`  | Typowa wartość — dobry balans |
| `100` | Szeroki wybór |
| `∞`   | Brak filtrowania (domyślne zachowanie) |

**Wada Top K vs. Top P**: Top K jest "ślepy" na rozkład. Jeśli jeden token ma P=0.95, a następne 39 mają po P=0.001 — Top K=40 nadal rozważa te 39 bezużytecznych tokenów. Top P=0.95 wybrałby tylko 1 token. Dlatego Top P jest generalnie preferowany, ale Top K jest prostszy obliczeniowo.

---

### 🔗 Jak parametry współdziałają?

Parametry są stosowane **sekwencyjnie** (kolejność zależy od implementacji, ale typowo):

1. **Temperature** modyfikuje logity → zmienia rozkład prawdopodobieństwa
2. **Top K** obcina do K najlepszych tokenów
3. **Top P** dalej zawęża do jądra (nucleus) prawdopodobieństwa
4. Model losuje z wynikowego zestawu

**Rekomendowane kombinacje**:

| Scenariusz | Temperature | Top P | Top K |
|-----------|-------------|-------|-------|
| Precyzyjne odpowiedzi (FAQ, kod) | 0.0–0.2 | 1.0 | — |
| Asystent ogólny (domyślne Azor) | 0.7 | 0.9 | 40 |
| Kreatywne pisanie | 1.0–1.2 | 0.95 | 50–100 |
| Brainstorming (max kreatywność) | 1.5 | 1.0 | 100 |

**Uwaga**: Najlepiej modyfikować **jeden** parametr naraz i obserwować efekt. Zmienianie wszystkich trzech jednocześnie utrudnia zrozumienie wpływu każdego z nich.

---

## 🔧 Różnice konfiguracji między klientami LLM

### Tabela porównawcza

| Parametr | Gemini (google-genai) | LLaMA (llama-cpp-python) | Ollama | Anthropic |
|----------|----------------------|-------------------------|--------|-----------|
| **Temperature** | ✅ `temperature` w `GenerateContentConfig` | ✅ `temperature` w `Llama.__call__()` | ✅ `temperature` w `options` dict | ✅ `temperature` w `messages.create()` |
| **Top P** | ✅ `top_p` w `GenerateContentConfig` | ✅ `top_p` w `Llama.__call__()` | ✅ `top_p` w `options` dict | ✅ `top_p` w `messages.create()` |
| **Top K** | ✅ `top_k` w `GenerateContentConfig` | ✅ `top_k` w `Llama.__call__()` | ✅ `top_k` w `options` dict | ✅ `top_k` w `messages.create()` |
| **Zakres Temperature** | 0.0 – 2.0 | 0.0 – ∞ (praktycznie 0.0–2.0) | 0.0 – 2.0 | 0.0 – 1.0 |
| **Zakres Top P** | 0.0 – 1.0 | 0.0 – 1.0 | 0.0 – 1.0 | 0.0 – 1.0 |
| **Zakres Top K** | ≥ 1 (int) | ≥ 1 (int) | ≥ 1 (int) | ≥ 1 (int) |
| **Domyślna Temperature** | Zależy od modelu (~1.0) | 0.8 | 0.8 | 1.0 |
| **Domyślne Top P** | Zależy od modelu (~0.95) | 0.95 | 0.9 | 0.999 |
| **Domyślne Top K** | Zależy od modelu (~40) | 40 | 40 | Brak domyślnego (wyłączone, gdy nie ustawione) |

---

### Gemini (google-genai)

**Dokumentacja**: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/content-generation-parameters

Parametry przekazywane w obiekcie `GenerateContentConfig`:

```python
from google.genai import types

config = types.GenerateContentConfig(
    system_instruction=system_instruction,
    temperature=0.7,        # float, 0.0–2.0
    top_p=0.9,              # float, 0.0–1.0
    top_k=40,               # int, >= 1
    thinking_config=types.ThinkingConfig(thinking_budget=0)
)

session = client.chats.create(model=model_name, config=config)
```

**Uwagi Gemini**:
- Parametry ustawia się na poziomie konfiguracji sesji czatu
- `GenerateContentConfig` przyjmuje wszystkie trzy parametry bezpośrednio
- Gdy ustawiasz `thinking_budget > 0` (tryb myślenia), temperature jest ignorowane

---

### LLaMA (llama-cpp-python)

**Dokumentacja**: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/

Parametry przekazywane bezpośrednio w wywołaniu `Llama.__call__()`:

```python
output = llama_model(
    prompt,
    max_tokens=512,
    temperature=0.7,    # float
    top_p=0.9,          # float, 0.0–1.0
    top_k=40,           # int
    stop=["User:", "Assistant:"],
    echo=False,
)
```

**Uwagi LLaMA**:
- Parametry ustawia się **per wywołanie** (nie per sesja)
- `temperature=0.0` oznacza greedy decoding
- Dodatkowe parametry niedostępne w innych klientach: `repeat_penalty`, `frequency_penalty`, `presence_penalty`, `mirostat_mode`, `mirostat_tau`, `mirostat_eta`
- llama-cpp-python nie nakłada sztywnego górnego limitu na temperature, ale wartości > 2.0 dają chaotyczne wyniki

---

### Ollama

**Dokumentacja**: https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values

Parametry przekazywane w dict `options` w wywołaniu `chat()`:

```python
response = ollama_client.chat(
    model=model_name,
    messages=messages,
    options={
        "temperature": 0.7,   # float, 0.0–2.0
        "top_p": 0.9,         # float, 0.0–1.0
        "top_k": 40,          # int, >= 1
    },
    stream=False,
)
```

**Uwagi Ollama**:
- Parametry przekazywane jako dict `options` — nie jako named kwargs
- Ollama przekazuje te parametry do underlying modelu (llama.cpp, itp.)
- Wspiera dodatkowe parametry: `num_predict`, `repeat_penalty`, `seed`, `num_ctx`, `mirostat`
- Serwer Ollama może nadpisać parametry zdefiniowane w `Modelfile` modelu

---

### Anthropic

**Dokumentacja**: https://docs.anthropic.com/en/api/messages

Parametry przekazywane bezpośrednio w `messages.create()`:

```python
response = anthropic_client.messages.create(
    model=model_name,
    max_tokens=4096,
    system=system_instruction,
    messages=messages,
    temperature=0.7,    # float, 0.0–1.0
    top_p=0.9,          # float, 0.0–1.0
    top_k=40,           # int, ≥ 1
)
```

**Uwagi Anthropic**:
- Zakres `temperature` jest ograniczony do **0.0–1.0** (mniejszy niż u Gemini/Ollama)
- `temperature=0.0` jest w pełni deterministyczne
- `top_k` jest wspierane — gdy nie ustawione, Anthropic nie stosuje filtrowania Top-K (efektywnie top_k=∞)
- Anthropic zaleca: "Jeśli używasz `top_p`, ustaw `temperature` na 1.0" — i odwrotnie

---

## 🔐 Zmienne środowiskowe

Każdy klient ma swój prefiks, a parametry samplowania mają suffix `_TEMPERATURE`, `_TOP_P`, `_TOP_K`:

| Zmienna | Typ | Domyślnie | Klient |
|---------|-----|-----------|--------|
| `GEMINI_TEMPERATURE` | float | `None` (domyślne API) | Gemini |
| `GEMINI_TOP_P` | float | `None` | Gemini |
| `GEMINI_TOP_K` | int | `None` | Gemini |
| `LLAMA_TEMPERATURE` | float | `None` | LLaMA |
| `LLAMA_TOP_P` | float | `None` | LLaMA |
| `LLAMA_TOP_K` | int | `None` | LLaMA |
| `OLLAMA_TEMPERATURE` | float | `None` | Ollama |
| `OLLAMA_TOP_P` | float | `None` | Ollama |
| `OLLAMA_TOP_K` | int | `None` | Ollama |
| `ANTHROPIC_TEMPERATURE` | float | `None` (domyślne API) | Anthropic |
| `ANTHROPIC_TOP_P` | float | `None` | Anthropic |
| `ANTHROPIC_TOP_K` | int | `None` | Anthropic |

**Wartość `None`** oznacza: "użyj domyślnej wartości biblioteki/API". Dzięki temu bez ustawiania zmiennych nic się nie zmienia — zachowanie jest identyczne jak przed zadaniem 13.

### Przykład `.env`

```env
# Gemini z niską temperaturą i nucleus sampling
ENGINE=GEMINI
GEMINI_API_KEY=your-key-here
MODEL_NAME=gemini-2.5-flash
GEMINI_TEMPERATURE=0.3
GEMINI_TOP_P=0.85

# LLaMA z wyższą kreatywnością
ENGINE=LLAMA_CPP
LLAMA_MODEL_PATH=/path/to/model.gguf
LLAMA_TEMPERATURE=1.0
LLAMA_TOP_P=0.95
LLAMA_TOP_K=50

# Ollama
ENGINE=OLLAMA
OLLAMA_MODEL_NAME=qwen2.5:7b-instruct
OLLAMA_TEMPERATURE=0.7
OLLAMA_TOP_K=40

# Anthropic
ENGINE=ANTHROPIC
ANTHROPIC_API_KEY=your-key-here
ANTHROPIC_TEMPERATURE=0.5
ANTHROPIC_TOP_P=0.9
ANTHROPIC_TOP_K=40
```

---

## 📐 Przebieg implementacji

### Krok 1: Rozszerzenie walidacji Pydantic

W każdym pliku `*_validation.py` dodano pola `temperature`, `top_p`, `top_k` (jako `Optional`) z odpowiednimi zakresami wartości i walidacją. Wartość `None` = domyślna biblioteki.

### Krok 2: Rozszerzenie konstruktorów klientów

W `__init__()` każdego klienta dodano parametry `temperature`, `top_p`, `top_k`. W `from_environment()` odczytuje się je z odpowiednich zmiennych środowiskowych.

### Krok 3: Propagacja do sesji czatu

Parametry z klienta przekazywane są do obiektów sesji (`LlamaChatSession`, `OllamaChatSession`, `AnthropicChatSession`) — a w przypadku Gemini wstawiane do `GenerateContentConfig`.

### Krok 4: Użycie w wywołaniach API

Każda sesja używa parametrów samplowania w swoich wywołaniach API:
- **LLaMA**: `self.llama_model(prompt, temperature=..., top_p=..., top_k=...)`
- **Ollama**: `self.ollama_client.chat(..., options={...})`
- **Gemini**: `types.GenerateContentConfig(temperature=..., top_p=..., top_k=...)`
- **Anthropic**: `self.anthropic_client.messages.create(temperature=..., top_p=..., top_k=...)`

### Krok 5: Informacja w wiadomości powitalnej

Metoda `ready_for_use_message()` wyświetla ustawione parametry samplowania, np.:
```
✅ Klient Ollama gotowy do użycia (Model: qwen2.5:7b-instruct, Host: ..., T=0.7, TopP=0.9, TopK=40)
```

---

## 🧠 Kluczowe wnioski

1. **Temperature, Top P i Top K to filtry na rozkład prawdopodobieństwa** — każdy działa inaczej, ale wszystkie ograniczają "losowość" modelu
2. **Top P jest adaptacyjny** (automatycznie dostosowuje liczbę kandydatów), Top K jest stały
3. **Wszystkie cztery klienty wspierają temperature, top_p i top_k** — pełna parność funkcji
4. **Anthropic ma węższy zakres temperature (0.0–1.0)** — inne klienty pozwalają na 0.0–2.0
5. **Domyślne `None`** gwarantuje wsteczną kompatybilność — bez ustawiania zmiennych zachowanie się nie zmienia
6. **Najlepsza praktyka**: zmieniaj jeden parametr naraz, obserwuj efekt, potem dostrajaj kolejny

