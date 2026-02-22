# 🐶 Azor ChatDog — Pełna dokumentacja przepływu i architektury

> Dokument opisuje szczegółowy przepływ sterowania przez wszystkie klasy i moduły aplikacji **Azor ChatDog** — interaktywnego chatbota CLI z obsługą wielu silników LLM. Celem jest dogłębne zrozumienie zarówno generycznego flow, jak i szczegółów implementacyjnych każdego z obsługiwanych modeli.

---

## Spis treści

1. [Przegląd architektury — widok z lotu ptaka](#1-przegląd-architektury--widok-z-lotu-ptaka)
2. [Punkt wejścia — `run.py` → `chat.py`](#2-punkt-wejścia--runpy--chatpy)
3. [Warstwa CLI — `cli/`](#3-warstwa-cli--cli)
4. [Moduł asystenta — `assistant/`](#4-moduł-asystenta--assistant)
5. [Zarządzanie sesjami — `session/`](#5-zarządzanie-sesjami--session)
6. [Warstwa klientów LLM — `llm/` (generyczny flow)](#6-warstwa-klientów-llm--llm-generyczny-flow)
7. [Szczegóły implementacji: Gemini](#7-szczegóły-implementacji-gemini)
8. [Szczegóły implementacji: LLaMA (llama.cpp)](#8-szczegóły-implementacji-llama-llamacpp)
9. [Szczegóły implementacji: Ollama](#9-szczegóły-implementacji-ollama)
10. [Szczegóły implementacji: Anthropic (Claude)](#10-szczegóły-implementacji-anthropic-claude)
11. [Walidacja konfiguracji — Pydantic](#11-walidacja-konfiguracji--pydantic)
12. [Warstwa plików — `files/`](#12-warstwa-plików--files)
13. [System komend — `commands/` + `command_handler.py`](#13-system-komend--commands--command_handlerpy)
14. [Pełny flow wysłania wiadomości — krok po kroku](#14-pełny-flow-wysłania-wiadomości--krok-po-kroku)
15. [Uniwersalny format historii](#15-uniwersalny-format-historii)
16. [Porównanie klientów LLM — tabela zbiorcza](#16-porównanie-klientów-llm--tabela-zbiorcza)
17. [Diagram zależności między modułami](#17-diagram-zależności-między-modułami)

---

## 1. Przegląd architektury — widok z lotu ptaka

Aplikacja Azor ChatDog jest konsolowym chatbotem napisanym w Pythonie, który obsługuje **cztery różne silniki LLM**:

| Silnik | Klasa klienta | Typ połączenia | SDK / Biblioteka |
|--------|---------------|----------------|-------------------|
| **Gemini** | `GeminiLLMClient` | API chmurowe (Google) | `google-genai` |
| **LLaMA (llama.cpp)** | `LlamaClient` | Model lokalny (plik .gguf) | `llama-cpp-python` |
| **Ollama** | `OllamaClient` | Serwer lokalny (HTTP) | `ollama` (SDK) |
| **Anthropic (Claude)** | `AnthropicClient` | API chmurowe (Anthropic) | `anthropic` |

### Schemat warstw

```
┌─────────────────────────────────────────────────┐
│                   run.py                        │  ← Punkt wejścia
├─────────────────────────────────────────────────┤
│                   chat.py                       │  ← Główna pętla
├────────────────┬────────────────────────────────┤
│  cli/          │  command_handler.py            │  ← Interfejs użytkownika
│  ├─ args.py    │  commands/                     │     i obsługa komend
│  ├─ console.py │  ├─ welcome.py                 │
│  └─ prompt.py  │  ├─ session_list.py            │
│                │  ├─ session_display.py          │
│                │  ├─ session_summary.py          │
│                │  ├─ session_remove.py           │
│                │  └─ session_to_pdf.py           │
├────────────────┴────────────────────────────────┤
│                 session/                         │  ← Zarządzanie sesjami
│  ├─ __init__.py (singleton SessionManager)       │
│  ├─ session_manager.py                           │
│  └─ chat_session.py                              │
├──────────────────────────────────────────────────┤
│                 assistant/                       │  ← Definicja asystenta
│  ├─ assistent.py (klasa Assistant)               │
│  └─ azor.py (fabryka create_azor_assistant)      │
├──────────────────────────────────────────────────┤
│                 llm/                             │  ← Klienci LLM
│  ├─ gemini_client.py      + gemini_validation.py │
│  ├─ llama_client.py       + llama_validation.py  │
│  ├─ ollama_client.py      + ollama_validation.py │
│  └─ anthropic_client.py   + anthropic_validation │
├──────────────────────────────────────────────────┤
│                 files/                           │  ← Persystencja
│  ├─ config.py (ścieżki katalogów)                │
│  ├─ session_files.py (zapis/odczyt sesji JSON)   │
│  ├─ wal.py (Write-Ahead Log)                     │
│  └─ pdf/ (generowanie PDF)                       │
└──────────────────────────────────────────────────┘
```

### Zależności zewnętrzne (requirements.txt)

| Pakiet | Wersja | Cel |
|--------|--------|-----|
| `google-genai` | latest | SDK Google Gemini |
| `python-dotenv` | latest | Ładowanie zmiennych z .env |
| `llama-cpp-python` | latest | Lokalne modele LLaMA (GGUF) |
| `ollama` | latest | SDK do serwera Ollama |
| `anthropic` | latest | SDK Anthropic Claude |
| `prompt_toolkit` | latest | Zaawansowany prompt CLI |
| `colorama` | latest | Kolorowe wyjście terminala |
| `pydantic` | >=2.0.0 | Walidacja konfiguracji |
| `fpdf2` | latest | Generowanie plików PDF |
| `markdown` | latest | Konwersja Markdown → HTML (do PDF) |

---

## 2. Punkt wejścia — `run.py` → `chat.py`

### `run.py` (6 linii)

```python
import chat

if __name__ == "__main__":
    chat.init_chat()
    chat.main_loop()
```

To jest absolutnie minimalny punkt wejścia. Cała logika delegowana jest do modułu `chat.py`.

### `chat.py` — inicjalizacja (`init_chat`)

Funkcja `init_chat()` wykonuje sekwencję startową:

```
init_chat()
  │
  ├─ 1. print_welcome()              → Wyświetla ASCII art psa Azora ("Woof Woof!")
  │
  ├─ 2. get_session_manager()        → Tworzy/pobiera singleton SessionManager
  │
  ├─ 3. cli.args.get_session_id_from_cli()
  │     → Parsuje argumenty CLI: --session-id=<ID>
  │     → Zwraca str | None
  │
  ├─ 4. manager.initialize_from_cli(cli_session_id)
  │     ├─ Jeśli podano --session-id:
  │     │   ├─ Tworzy Assistant (create_azor_assistant)
  │     │   ├─ ChatSession.load_from_file(assistant, session_id)
  │     │   │   ├─ session_files.load_session_history(session_id)
  │     │   │   └─ Tworzy ChatSession z załadowaną historią
  │     │   │       └─ _initialize_llm_session() ← TU POWSTAJE KLIENT LLM!
  │     │   └─ Wyświetla pomoc + podsumowanie historii (jeśli niepusta)
  │     └─ Jeśli NIE podano:
  │         ├─ Tworzy nowy Assistant
  │         ├─ Tworzy nowy ChatSession (nowy UUID)
  │         │   └─ _initialize_llm_session() ← TU POWSTAJE KLIENT LLM!
  │         └─ Wyświetla pomoc
  │
  └─ 5. atexit.register(manager.cleanup_and_save)
        → Rejestruje handler zapisu przy wyjściu z programu
```

**Kluczowy moment**: Klient LLM jest tworzony wewnątrz `ChatSession._initialize_llm_session()`. To jest jedyne miejsce w kodzie, gdzie następuje wybór silnika na podstawie zmiennej środowiskowej `ENGINE`.

### `chat.py` — główna pętla (`main_loop`)

```
main_loop()
  │
  └─ while True:
      │
      ├─ get_user_input()              → Prompt z auto-uzupełnianiem (prompt_toolkit)
      │
      ├─ Jeśli input pusty → continue
      │
      ├─ Jeśli input zaczyna się od '/':
      │   └─ command_handler.handle_command(user_input)
      │       └─ Zwraca True → break (wyjście z pętli)
      │
      └─ W przeciwnym razie (rozmowa z modelem):
          ├─ session = manager.get_current_session()
          ├─ response = session.send_message(user_input)
          │   ├─ _llm_chat_session.send_message(text)
          │   ├─ Synchronizacja historii
          │   ├─ Zapis do WAL (Write-Ahead Log)
          │   └─ return response
          ├─ (total_tokens, remaining, max) = session.get_token_info()
          ├─ console.print_assistant(f"AZOR: {response.text}")
          ├─ console.print_info(f"Tokens: {total} (Pozostało: {remaining} / {max})")
          └─ session.save_to_file()
              └─ Jeśli błąd → console.print_error(...)
```

**Obsługa wyjątków w pętli:**

| Wyjątek | Reakcja |
|---------|---------|
| `KeyboardInterrupt` (Ctrl+C) | Komunikat + `break` → atexit wywoła `cleanup_and_save()` |
| `EOFError` (Ctrl+D) | Komunikat + `break` |
| `Exception` (ogólny) | Komunikat + traceback + `break` |

Niezależnie od sposobu wyjścia z pętli, handler `atexit` (zarejestrowany w `init_chat()`) zapewnia próbę finalnego zapisu sesji.

---

## 3. Warstwa CLI — `cli/`

### `cli/args.py` — Parsowanie argumentów

Używa standardowego `argparse`. Definiuje jeden opcjonalny argument:

| Argument | Typ | Domyślnie | Opis |
|----------|-----|-----------|------|
| `--session-id` | `str` | `None` | ID sesji do wznowienia (np. `a1b2c3d4`) |

Opis programu (w `--help`): *"Interaktywny pies asystent! 🐶"*

**Przykład użycia:**
```bash
python run.py --session-id=a1b2c3d4
```

### `cli/console.py` — Kolorowe wyjście terminala

Centralizuje wyświetlanie tekstu w terminalu z użyciem biblioteki `colorama`:

| Funkcja | Kolor | Użycie |
|---------|-------|--------|
| `print_error(msg)` | 🔴 Czerwony (`Fore.RED`) | Komunikaty o błędach |
| `print_assistant(msg)` | 🟦 Cyan (`Fore.CYAN`) | Odpowiedzi asystenta |
| `print_user(msg)` | 🔵 Niebieski (`Fore.BLUE`) | Wiadomości użytkownika (w wyświetlanej historii) |
| `print_info(msg)` | ⚪ Domyślny (brak koloru) | Informacje systemowe |
| `print_help(msg)` | 🟡 Żółty (`Fore.YELLOW`) | Komunikaty pomocy i komendy |

**Złożone funkcje wyświetlania:**

- `display_help(session_id)` — wyświetla:
  - Aktualne ID sesji
  - Ścieżkę katalogu logów (`~/.azor`)
  - Pełną listę dostępnych komend slash z opisami

- `display_final_instructions(session_id)` — wyświetla po wyjściu:
  - Instrukcję kontynuacji sesji:
  ```
  python run.py --session-id=<ID>
  ```
  - Sformatowaną jasnym białym boldem (`Fore.WHITE + Style.BRIGHT`)

Inicjalizacja: `init(autoreset=True)` — automatyczny reset stylu po każdym princie.

### `cli/prompt.py` — Zaawansowany prompt wejściowy

Wykorzystuje bibliotekę `prompt_toolkit` do zaawansowanego interfejsu wejściowego. To najbardziej rozbudowany moduł CLI:

**1. Kolorowanie składni — klasa `SlashCommandLexer(Lexer)`:**

Metoda `lex_document()` zwraca funkcję `get_line_tokens(lineno)`, która:
- Sprawdza czy linia zaczyna się od jednej z komend: `/exit`, `/quit`, `/switch`, `/help`, `/session`
- Jeśli tak → koloruje komendę na `#ff0066 bold` (różowo-czerwony, klasa `slash-command`)
- Dla `/session` — dodatkowo rozpoznaje podkomendy (`list`, `display`, `pop`, `clear`, `new`, `remove`) i koloruje je na `#00ff00 bold` (zielony, klasa `subcommand`)
- Reszta tekstu → `#aaaaaa` (szary, klasa `normal-text`)

**2. Auto-uzupełnianie — `NestedCompleter`:**

```python
_commands_completer = NestedCompleter({
    '/exit': None,
    '/quit': None,
    '/help': None,
    '/switch': None,
    '/session': WordCompleter(['list', 'display', 'pop', 'clear', 'new', 'remove'])
})
```

Dzięki `NestedCompleter`, po wpisaniu `/session ` i Tab, użytkownik zobaczy podkomendy. Dla pozostałych komend brak podsugestii.

**3. Inteligentne zachowanie klawisza Enter — `KeyBindings`:**

```python
@kb.add('enter', filter=completion_is_selected)
def _(event):
    event.app.current_buffer.complete_state = None
```

Logika:
- Gdy dropdown auto-uzupełniania jest otwarty i pozycja zaznaczona → Enter **akceptuje sugestię** (zamyka menu, NIE wysyła)
- Gdy dropdown jest zamknięty → domyślne zachowanie Entera → **wysłanie promptu**

**Funkcja `get_user_input(prompt_text="TY: ")`** — łączy wszystko razem:

```python
return prompt(
    prompt_text,                          # "TY: "
    completer=_commands_completer,        # auto-uzupełnianie
    lexer=SlashCommandLexer(),            # kolorowanie składni
    style=_prompt_style,                  # definicje kolorów
    complete_while_typing=True,           # sugestie podczas pisania
    key_bindings=_key_bindings            # niestandardowe Enter
).strip()
```

---

## 4. Moduł asystenta — `assistant/`

### `assistant/assistent.py` — Klasa `Assistant`

Prosta klasa enkapsulująca konfigurację asystenta — jego tożsamość i zachowanie:

```python
class Assistant:
    _system_prompt: str   # Prompt systemowy definiujący zachowanie i osobowość
    _name: str            # Nazwa wyświetlana w czacie (np. "AZOR")
```

**Właściwości (property):**
- `system_prompt` → zwraca `_system_prompt` (read-only)
- `name` → zwraca `_name` (read-only)

**Kluczowa cecha**: Klasa `Assistant` jest **całkowicie niezależna od modelu LLM**. Definiuje jedynie *tożsamość* i *zachowanie* asystenta, nie implementację techniczną. Ten sam asystent może być użyty z dowolnym z czterech silników. To jest separacja **"co asystent robi"** od **"jak technicznie to robi"**.

### `assistant/azor.py` — Fabryka `create_azor_assistant()`

Tworzy konkretną instancję asystenta "Azor":

| Parametr | Wartość |
|----------|---------|
| `name` | `"AZOR"` |
| `system_prompt` | *"Jesteś pomocnym asystentem, Nazywasz się Azor i jesteś psem o wielkich możliwościach. Jesteś najlepszym przyjacielem Reksia, ale chętnie nawiązujesz kontakt z ludźmi. Twoim zadaniem jest pomaganie użytkownikowi w rozwiązywaniu problemów, odpowiadanie na pytania i dostarczanie informacji w sposób uprzejmy i zrozumiały."* |

Fabryka `create_azor_assistant()` jest wywoływana w `SessionManager` za każdym razem, gdy tworzona jest nowa sesja lub ładowana istniejąca. Wzorzec fabryki pozwala łatwo dodać w przyszłości nowych asystentów z innymi promptami systemowymi (np. inny pies, inna osobowość).

---

## 5. Zarządzanie sesjami — `session/`

### `session/__init__.py` — Singleton `SessionManager`

Moduł implementuje wzorzec **Singleton** (na poziomie modułu) dla `SessionManager`:

```python
_session_manager: SessionManager | None = None

def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
```

To gwarantuje, że w całej aplikacji istnieje **dokładnie jedna instancja** managera sesji. Każdy moduł (chat.py, command_handler.py, komendy) wywołuje `get_session_manager()` i dostaje tę samą instancję.

**Eksportowane symbole:** `ChatSession`, `SessionManager`, `get_session_manager`.

### `session/session_manager.py` — Klasa `SessionManager`

Orkiestruje cykl życia sesji i zarządza aktywną sesją. To warstwa wyższego poziomu nad `ChatSession` — odpowiada za przełączanie między sesjami, zapis przy przełączaniu, cleanup przy wyjściu, etc.

**Stan wewnętrzny:**
```python
_current_session: ChatSession | None = None
```

#### Metoda `initialize_from_cli(cli_session_id)`:

```
initialize_from_cli(cli_session_id)
│
├─ cli_session_id != None (podano --session-id):
│   ├─ assistant = create_azor_assistant()
│   ├─ session, error = ChatSession.load_from_file(assistant, session_id)
│   │
│   ├─ Jeśli error:
│   │   ├─ console.print_error(error)
│   │   ├─ Fallback: session = ChatSession(assistant=assistant)  ← nowa sesja
│   │   └─ "Rozpoczęto nową sesję z ID: ..."
│   │
│   ├─ self._current_session = session
│   ├─ display_help(session.session_id)
│   ├─ Jeśli sesja ma historię:
│   │   └─ display_history_summary(history, assistant_name)
│   └─ return session
│
└─ cli_session_id == None (nowa sesja):
    ├─ print("Rozpoczynanie nowej sesji.")
    ├─ assistant = create_azor_assistant()
    ├─ session = ChatSession(assistant=assistant)
    │   └─ __init__ → _initialize_llm_session()  ← TU POWSTAJE KLIENT LLM
    ├─ self._current_session = session
    ├─ display_help(session.session_id)
    └─ return session
```

#### Metoda `create_new_session(save_current=True)`:

```
create_new_session(save_current=True)
│
├─ Jeśli save_current i jest bieżąca sesja:
│   ├─ save_attempted = True
│   ├─ previous_session_id = current.session_id
│   └─ success, error = current.save_to_file()
│       └─ save_error = error jeśli nie powiodło się
│
├─ assistant = create_azor_assistant()
├─ new_session = ChatSession(assistant=assistant)  ← nowy UUID, nowy klient LLM
├─ self._current_session = new_session
│
└─ return (new_session, save_attempted, previous_session_id, save_error)
```

#### Metoda `switch_to_session(session_id)`:

```
switch_to_session(session_id)
│
├─ Jeśli jest bieżąca sesja:
│   ├─ save_attempted = True
│   ├─ previous_session_id = current.session_id
│   └─ current.save_to_file()  ← zapis bieżącej przed przełączeniem
│
├─ assistant = create_azor_assistant()
├─ new_session, error = ChatSession.load_from_file(assistant, session_id)
│
├─ Jeśli error:
│   └─ return (None, save_attempted, previous_id, False, error, False)
│      ← _current_session NIE zmieniona! Użytkownik zostaje w bieżącej sesji.
│
└─ Jeśli success:
    ├─ self._current_session = new_session
    ├─ has_history = not new_session.is_empty()
    └─ return (new_session, save_attempted, previous_id, True, None, has_history)
```

#### Metoda `remove_current_session_and_create_new()`:

```
remove_current_session_and_create_new()
│
├─ Brak aktywnej sesji → raise RuntimeError
│
├─ removed_session_id = current.session_id
├─ session_files.remove_session_file(removed_session_id)
│
├─ ZAWSZE tworzy nową sesję (nawet jeśli usuwanie się nie powiodło):
│   ├─ assistant = create_azor_assistant()
│   ├─ new_session = ChatSession(assistant=assistant)
│   └─ self._current_session = new_session
│
└─ return (new_session, removed_session_id, remove_success, remove_error)
```

#### Metoda `cleanup_and_save()`:

```
cleanup_and_save()  ← Wywoływane przez atexit
│
├─ Brak aktywnej sesji → return
│
├─ session.is_empty() (< 2 wpisy):
│   └─ "Sesja jest pusta/niekompletna. Pominięto finalny zapis."
│
└─ session NIE jest pusta:
    ├─ "Finalny zapis historii sesji: {session_id}"
    ├─ session.save_to_file()
    └─ display_final_instructions(session_id)
```

### `session/chat_session.py` — Klasa `ChatSession`

**Najważniejsza klasa w aplikacji.** Enkapsuluje wszystko co dotyczy pojedynczej sesji czatu.

#### Stan wewnętrzny:

```python
assistant: Assistant                                # Konfiguracja asystenta
session_id: str                                     # UUID sesji
_history: List[Any]                                 # Historia rozmowy (format uniwersalny)
_llm_client: Union[GeminiLLMClient, LlamaClient,
                   OllamaClient, AnthropicClient,
                   None] = None                     # Klient LLM (lazy init, singleton)
_llm_chat_session: Any | None = None                # Wrapper sesji czatu
_max_context_tokens: int = 32768                    # Stały limit kontekstu
```

#### Mapowanie silników (`ENGINE_MAPPING`):

```python
ENGINE_MAPPING = {
    'LLAMA_CPP':  LlamaClient,
    'GEMINI':     GeminiLLMClient,
    'OLLAMA':     OllamaClient,
    'ANTHROPIC':  AnthropicClient,
}
```

#### Metoda `_initialize_llm_session()` — serce wyboru silnika:

```
_initialize_llm_session()
│
├─ 1. engine = os.getenv('ENGINE', 'GEMINI').upper()
│
├─ 2. Walidacja: engine in ENGINE_MAPPING?
│     Nie → raise ValueError("ENGINE musi być: LLAMA_CPP, GEMINI, OLLAMA, ANTHROPIC")
│
├─ 3. Jeśli _llm_client == None (PIERWSZA inicjalizacja):
│     ├─ SelectedClientClass = ENGINE_MAPPING[engine]
│     ├─ console: "🤖 Przygotowywanie klienta..."
│     ├─ self._llm_client = SelectedClientClass.from_environment()
│     └─ console: "✅ Klient gotowy do użycia (...)"
│
└─ 4. self._llm_chat_session = self._llm_client.create_chat_session(
          system_instruction=self.assistant.system_prompt,
          history=self._history,
          thinking_budget=0
      )
```

**Ważny niuans**: `_llm_client` tworzony **raz** (lazy init). Przy `clear_history()` i `pop_last_exchange()` tworzony jest tylko nowy `_llm_chat_session`.

#### Metoda `send_message(text)`:

```
send_message(text)
│
├─ Walidacja: _llm_chat_session?
├─ response = _llm_chat_session.send_message(text)
├─ self._history = _llm_chat_session.get_history()
├─ total_tokens = self.count_tokens()
├─ append_to_wal(session_id, text, response.text, total_tokens, model_name)
│     → WAL failure jest ignorowane (nie blokuje flow)
└─ return response
```

#### Metoda `save_to_file()`:

```
save_to_file()
│
├─ Synchronizacja: _history = _llm_chat_session.get_history()
└─ session_files.save_session_history(session_id, _history, system_prompt, model_name)
```

#### Metoda `load_from_file()` (classmethod):

```
ChatSession.load_from_file(assistant, session_id)
│
├─ history, error = session_files.load_session_history(session_id)
├─ error → return (None, error)
└─ cls(assistant, session_id, history) → _initialize_llm_session() z historią
```

#### Pozostałe metody:

| Metoda | Opis |
|--------|------|
| `get_history()` | Synchronizuje z `_llm_chat_session` i zwraca `_history` |
| `clear_history()` | `_history = []` → `_initialize_llm_session()` → `save_to_file()` |
| `pop_last_exchange()` | Sprawdza `len >= 2`, ucina `[-2:]`, reinicjalizuje, zapisuje |
| `count_tokens()` | Deleguje do `_llm_client.count_history_tokens(_history)` |
| `is_empty()` | `len(_history) < 2` |
| `get_remaining_tokens()` | `32768 - count_tokens()` |
| `get_token_info()` | `(total, remaining, max=32768)` |
| `assistant_name` (property) | `self.assistant.name` → "AZOR" |

---

## 6. Warstwa klientów LLM — `llm/` (generyczny flow)

### Wspólny kontrakt (implicit interface — duck typing)

Wszystkie cztery klasy klientów LLM implementują **identyczny zestaw metod**. Nie ma formalnej klasy bazowej (ABC) — kontrakt jest wymuszony przez **duck typing** w `ChatSession`:

```python
# Pseudo-interfejs — NIE istnieje w kodzie, ale jest niejawnie wymuszony
class LLMClient:
    @staticmethod
    def preparing_for_use_message() -> str: ...

    @classmethod
    def from_environment(cls) -> 'Self': ...

    def create_chat_session(self,
                            system_instruction: str,
                            history: Optional[List[Dict]] = None,
                            thinking_budget: int = 0) -> ChatSessionWrapper: ...

    def count_history_tokens(self, history: List[Dict]) -> int: ...

    def get_model_name(self) -> str: ...

    def is_available(self) -> bool: ...

    def ready_for_use_message(self) -> str: ...

    @property
    def client(self): ...  # Backwards compatibility
```

### Wspólny kontrakt sesji czatu

```python
class ChatSessionWrapper:  # Pseudo-interfejs
    def send_message(self, text: str) -> Response: ...
    def get_history(self) -> List[Dict]: ...
```

### Wspólny kontrakt odpowiedzi

Każdy silnik ma swoją klasę odpowiedzi, ale wszystkie mają atrybut `.text: str`:
- **Gemini**: natywny obiekt Response z Google GenAI (ma `.text` natywnie)
- **LLaMA**: `LlamaResponse(text)` — prosta klasa z atrybutem `text`
- **Ollama**: `OllamaResponse(text)` — prosta klasa z atrybutem `text`
- **Anthropic**: `AnthropicResponse(text, input_tokens, output_tokens)` — rozszerzona o info o tokenach

### Generyczny flow tworzenia klienta (`from_environment()`)

```
SelectedClientClass.from_environment()
│
├─ load_dotenv()                         → Ładowanie zmiennych z pliku .env
│
├─ Tworzenie Pydantic config:            → Walidacja parametrów
│   XxxConfig(
│       model_name=os.getenv(..., default),
│       api_key=os.getenv(...),
│       ...dodatkowe parametry...
│   )
│   ├─ Walidacja typów (Pydantic BaseModel)
│   ├─ Walidacja reguł biznesowych (@validator)
│   └─ raise ValueError jeśli niepoprawne
│
└─ cls(model_name=..., api_key=..., ...)
    └─ __init__():
        ├─ Walidacja parametrów wejściowych
        ├─ Zapisanie konfiguracji jako atrybuty instancji
        └─ _initialize_client() / _initialize_model()
            → SDK client / załadowany model
```

### Generyczny flow wysyłania wiadomości

```
chat_session_wrapper.send_message(text)
│
├─ Dodanie user message do _history:
│   {"role": "user", "parts": [{"text": text}]}
│
├─ Przygotowanie danych do wywołania:
│   ├─ Gemini:    forward do natywnej sesji (auto-zarządzanie historią)
│   ├─ LLaMA:     _build_prompt_from_history() → jeden string
│   ├─ Ollama:    _build_messages_from_history() → lista messages
│   └─ Anthropic: _build_messages_from_history() → lista messages
│
├─ Wywołanie API / modelu
│   ├─ success → ekstrakcja tekstu
│   └─ error → "Przepraszam, wystąpił błąd..."
│
├─ Dodanie model message do _history:
│   {"role": "model", "parts": [{"text": response_text}]}
│
└─ return Response(text=response_text)
```

---

## 7. Szczegóły implementacji: Gemini

### Klasy

| Klasa | Plik | Rola |
|-------|------|------|
| `GeminiLLMClient` | `gemini_client.py` | Klient API Google Gemini |
| `GeminiChatSessionWrapper` | `gemini_client.py` | Wrapper sesji — konwersja format historii |
| `GeminiConfig` | `gemini_validation.py` | Walidacja konfiguracji (Pydantic) |

### Konfiguracja

| Env Var | Domyślna wartość | Opis |
|---------|-----------------|------|
| `ENGINE` | `GEMINI` | Musi być `GEMINI` |
| `MODEL_NAME` | `gemini-2.5-flash` | Nazwa modelu |
| `GEMINI_API_KEY` | (wymagany) | Klucz API Google |

### Inicjalizacja klienta

```
GeminiLLMClient.from_environment()
│
├─ load_dotenv()
├─ GeminiConfig(model_name=..., gemini_api_key=...)
│   └─ Walidacja: API key nie pusty (min_length=1 + @validator)
└─ GeminiLLMClient(model_name, api_key)
    └─ __init__():
        ├─ if not api_key → raise ValueError
        └─ _initialize_client() → genai.Client()
            ├─ success → return
            └─ exception → sys.exit(1)
```

**Uwaga**: `genai.Client()` bez argumentów — SDK szuka klucza w `GOOGLE_API_KEY` env var.

### Tworzenie sesji czatu

```
create_chat_session(system_instruction, history, thinking_budget=0)
│
├─ Konwersja historii: Dict → types.Content
│   entry → types.Content(role=role, parts=[types.Part.from_text(text)])
│
├─ gemini_session = self._client.chats.create(
│     model=self.model_name,
│     history=gemini_history,
│     config=types.GenerateContentConfig(
│         system_instruction=system_instruction,
│         thinking_config=types.ThinkingConfig(thinking_budget=0)
│     )
│   )
│
└─ return GeminiChatSessionWrapper(gemini_session)
```

**Cechy specyficzne Gemini:**
- System prompt jest **parametrem konfiguracyjnym** sesji, NIE wiadomością w historii
- Obsługuje `thinking_budget` (extended reasoning) — tutaj wyłączony (0)
- Natywne **stateful sessions** — Gemini SDK zarządza historią wewnętrznie
- Wymaga **konwersji formatu** w obie strony (Dict ↔ Content objects)

### `GeminiChatSessionWrapper`

**`send_message(text)`** — prosty forward:
```python
return self.gemini_session.send_message(text)
```
Gemini SDK sam dodaje user/model messages do wewnętrznej historii. Response ma natywny `.text`.

**`get_history()`** — konwersja z powrotem:
```
Dla każdego Content w gemini_session.get_history():
  ├─ Szuka pierwszego part z text (nie pusty)
  └─ → {"role": content.role, "parts": [{"text": text_part}]}
```

### Liczenie tokenów

```python
response = self._client.models.count_tokens(model=self.model_name, contents=gemini_history)
return response.total_tokens
```

Gemini ma **natywne API do liczenia tokenów** (`models.count_tokens()`), ale z fallbackiem na heurystykę w razie błędów.

### Emoji i komunikaty

- Preparing: `"🤖 Przygotowywanie klienta Gemini..."`
- Ready: `"✅ Klient Gemini gotowy do użycia (Model: gemini-2.5-flash, Key: AIza...xY9z)"`

---

## 8. Szczegóły implementacji: LLaMA (llama.cpp)

### Klasy

| Klasa | Plik | Rola |
|-------|------|------|
| `LlamaClient` | `llama_client.py` | Klient modelu lokalnego |
| `LlamaChatSession` | `llama_client.py` | Wrapper sesji — budowanie promptu tekstowego |
| `LlamaResponse` | `llama_client.py` | Prosta klasa response (`.text`) |
| `LlamaConfig` | `llama_validation.py` | Walidacja konfiguracji (Pydantic) |

### Konfiguracja

| Env Var | Domyślna wartość | Opis |
|---------|-----------------|------|
| `ENGINE` | - | Musi być `LLAMA_CPP` |
| `LLAMA_MODEL_NAME` | `llama-3.1-8b-instruct` | Nazwa wyświetlana |
| `LLAMA_MODEL_PATH` | (wymagany) | Ścieżka do pliku `.gguf` |
| `LLAMA_GPU_LAYERS` | `1` | Ile warstw na GPU (offloading) |
| `LLAMA_CONTEXT_SIZE` | `2048` | Rozmiar okna kontekstu |

### Inicjalizacja klienta

```
LlamaClient.from_environment()
│
├─ load_dotenv()
├─ LlamaConfig(model_name, llama_model_path, llama_gpu_layers, llama_context_size)
│   └─ Walidacja @validator('llama_model_path'):
│       ├─ os.path.exists(v) → ValueError jeśli nie istnieje
│       └─ v.endswith('.gguf') → ValueError jeśli złe rozszerzenie
│
├─ console: "Ładowanie modelu LLaMA z: {path}"
│
└─ LlamaClient(model_name, model_path, n_gpu_layers, n_ctx)
    └─ __init__():
        ├─ if not model_path → raise ValueError
        ├─ if not os.path.exists → raise ValueError
        └─ _initialize_model():
            ├─ console: "Inicjalizacja modelu LLaMA: ..."
            └─ Llama(model_path, n_gpu_layers, n_ctx, verbose=False)
                ├─ success → Llama instance
                └─ exception → raise RuntimeError
```

**Kluczowa różnica**: LLaMA ładuje **cały model do pamięci RAM/VRAM**. To może trwać kilka-kilkanaście sekund. `n_gpu_layers` kontroluje ile warstw offloadować na GPU.

### Tworzenie sesji czatu

```
create_chat_session(system_instruction, history, thinking_budget=0)
│
├─ thinking_budget jest IGNOROWANY (parametr kompatybilności)
│
└─ return LlamaChatSession(
      llama_model=self._llama_model,
      system_instruction=system_instruction,
      history=history or []
   )
```

**Brak konwersji historii** — LLaMA pracuje bezpośrednio na formacie uniwersalnym (Dict). Konwersja następuje dopiero przy budowaniu promptu tekstowego.

### LlamaChatSession — szczegóły

**`send_message(text)`:**

```
send_message(text)
│
├─ Dodanie do _history:
│   {"role": "user", "parts": [{"text": text}]}
│
├─ prompt = _build_prompt_from_history()
│
├─ output = self.llama_model(
│     prompt,
│     max_tokens=512,                    ← stałe 512
│     stop=["User:", "Assistant:",
│           "\n\nUser:", "\n\nAssistant:"],
│     echo=False
│   )
│
├─ response_text = output["choices"][0]["text"].strip()
│
├─ Dodanie do _history:
│   {"role": "model", "parts": [{"text": response_text}]}
│
└─ return LlamaResponse(response_text)
```

**`_build_prompt_from_history()` — kluczowa metoda:**

LLaMA nie ma natywnego API czatu — cała rozmowa jest formatowana jako **jeden ciąg tekstowy**:

```
System: {system_instruction}

User: {wiadomość_1}
```

---

## 9. Szczegóły implementacji: Ollama

### Klasy

| Klasa | Plik | Rola |
|-------|------|------|
| `OllamaClient` | `ollama_client.py` | Klient serwera Ollama |
| `OllamaChatSession` | `ollama_client.py` | Wrapper sesji — konwersja na format messages Ollama |
| `OllamaResponse` | `ollama_client.py` | Prosta klasa response (`.text`) |
| `OllamaConfig` | `ollama_validation.py` | Walidacja konfiguracji (Pydantic) |

### Architektura Ollama vs LLaMA

**Ollama ≠ LLaMA**. Choć Ollama może uruchamiać modele LLaMA, to:
- **LLaMA (llama.cpp)**: ładuje model bezpośrednio do pamięci procesu Python (in-process)
- **Ollama**: komunikuje się z **osobnym serwerem** przez HTTP API (out-of-process)

Ollama wymaga uruchomionego serwera (`ollama serve`) w tle.

### Konfiguracja

| Env Var | Domyślna wartość | Opis |
|---------|-----------------|------|
| `ENGINE` | - | Musi być `OLLAMA` |
| `OLLAMA_MODEL_NAME` | `qwen2.5:7b-instruct` | Nazwa modelu w rejestrze Ollama |
| `OLLAMA_HOST` | `http://localhost:11434` | Adres serwera Ollama |

### Inicjalizacja klienta

```
OllamaClient.from_environment()
│
├─ load_dotenv()
├─ OllamaConfig(model_name=..., ollama_host=...)
│   └─ Walidacja:
│       ├─ model_name: nie pusty (@validator)
│       └─ ollama_host: nie pusty, http:// lub https://, strip + rstrip('/')
│
├─ console: "Łączenie z serwerem Ollama: {host}"
│
└─ OllamaClient(model_name, host)
    └─ __init__():
        ├─ if not model_name → raise ValueError
        └─ _initialize_client():
            └─ OllamaSDKClient(host=self.host)
                ├─ success → SDK client instance
                └─ exception → raise RuntimeError
```

### Tworzenie sesji czatu

```
create_chat_session(system_instruction, history, thinking_budget=0)
│
├─ thinking_budget IGNOROWANY (kompatybilność)
│
└─ return OllamaChatSession(
      ollama_client=self._client,
      model_name=self.model_name,
      system_instruction=system_instruction,
      history=history or []
   )
```

### `OllamaChatSession.send_message()` — API messages

```
send_message(text)
│
├─ Dodanie do _history:
│   {"role": "user", "parts": [{"text": text}]}
│
├─ messages = _build_messages_from_history()
│   │
│   │  Konwersja na format Ollama:
│   │  [
│   │    {"role": "system",    "content": "{system_instruction}"},  ← pierwsza msg
│   │    {"role": "user",      "content": "{wiadomość_1}"},
│   │    {"role": "assistant", "content": "{odpowiedź_1}"},
│   │    {"role": "user",      "content": "{wiadomość_2}"},
│   │  ]
│   │
│   └─ Mapowanie ról: "user"→"user", "model"→"assistant"
│      System prompt → osobna wiadomość {"role": "system"} na początku
│
├─ response = self.ollama_client.chat(
│     model=self.model_name,
│     messages=messages,
│     stream=False                       ← bez streamingu
│   )
│
├─ response_text = response['message']['content'].strip()
│
├─ Dodanie do _history:
│   {"role": "model", "parts": [{"text": response_text}]}
│
└─ return OllamaResponse(response_text)
```

**Kluczowe różnice vs LLaMA:**
1. **Prawdziwe API czatu** (chat API) — nie budowanie promptu tekstowego
2. **System prompt jako osobna wiadomość** `{"role": "system", ...}` w tablicy messages
3. **Mapowanie ról**: `"model"` (wewnętrzny) → `"assistant"` (Ollama API)
4. **Bezstanowe** — pełna historia wysyłana przy każdym requestcie
5. **`stream=False`** — odpowiedź w jednym kawałku (nie streaming)

### Liczenie tokenów — heurystyka

Ollama **nie ma natywnego API do zliczania tokenów**, więc stosowana jest heurystyka:

```python
word_count = len(full_text.split())
return int(word_count * 0.75)   # ~0.75 tokena/słowo (średnia angielska)
```

Fallback: `total_chars // 4`.

### `is_available()` — sprawdzenie serwera

```python
def is_available(self):
    try:
        self._client.list()  # Wywołuje API listy modeli
        return True
    except Exception:
        return False
```

Ollama jest **jedynym klientem z aktywnym sprawdzaniem dostępności serwera** (wysyła faktyczny request HTTP).

### Emoji i komunikaty

- Preparing: `"🦙 Przygotowywanie klienta Ollama..."`
- Ready: `"✅ Klient Ollama gotowy do użycia (Model: qwen2.5:7b-instruct, Host: http://localhost:11434)"`

---

## 10. Szczegóły implementacji: Anthropic (Claude)

### Klasy

| Klasa | Plik | Rola |
|-------|------|------|
| `AnthropicClient` | `anthropic_client.py` | Klient API Anthropic Claude |
| `AnthropicChatSession` | `anthropic_client.py` | Wrapper sesji — konwersja + tracking tokenów per request |
| `AnthropicResponse` | `anthropic_client.py` | Response z `.text` + `.input_tokens` + `.output_tokens` |
| `AnthropicConfig` | `anthropic_validation.py` | Walidacja konfiguracji (Pydantic) |

### Konfiguracja

| Env Var | Domyślna wartość | Opis |
|---------|-----------------|------|
| `ENGINE` | - | Musi być `ANTHROPIC` |
| `ANTHROPIC_MODEL_NAME` | `claude-haiku-4-5-20251001` | Nazwa modelu Claude |
| `ANTHROPIC_API_KEY` | (wymagany) | Klucz API Anthropic |
| `ANTHROPIC_MAX_TOKENS` | `4096` | Maks. tokenów w odpowiedzi |

### Inicjalizacja klienta

```
AnthropicClient.from_environment()
│
├─ load_dotenv()
├─ AnthropicConfig(model_name=..., anthropic_api_key=..., max_tokens=...)
│   └─ Walidacja:
│       ├─ API key: nie pusty (min_length=1 + @validator strip)
│       ├─ model_name: nie pusty (@validator strip)
│       └─ max_tokens: >= 1
│
└─ AnthropicClient(model_name, api_key, max_tokens)
    └─ __init__():
        ├─ if not api_key → raise ValueError
        └─ _initialize_client():
            └─ Anthropic(api_key=self.api_key)  ← klucz jawnie!
                ├─ success → Anthropic SDK client
                └─ exception → sys.exit(1)
```

**W przeciwieństwie do Gemini**, Anthropic client dostaje klucz API **jawnie** w konstruktorze `Anthropic(api_key=...)`.

### Tworzenie sesji czatu

```
create_chat_session(system_instruction, history, thinking_budget=0)
│
├─ thinking_budget IGNOROWANY (kompatybilność)
│
└─ return AnthropicChatSession(
      anthropic_client=self._client,
      model_name=self.model_name,
      system_instruction=system_instruction,
      max_tokens=self.max_tokens,          ← 4096 domyślnie
      history=history or []
   )
```

### `AnthropicChatSession` — dodatkowy stan

```python
_total_input_tokens: int = 0
_total_output_tokens: int = 0
```

Anthropic jest **jedynym silnikiem z kumulatywnym trackingiem tokenów per sesja**.

### `AnthropicChatSession.send_message()` — API Messages

```
send_message(text)
│
├─ Dodanie do _history:
│   {"role": "user", "parts": [{"text": text}]}
│
├─ messages = _build_messages_from_history()
│   │
│   │  Konwersja na format Anthropic:
│   │  [
│   │    {"role": "user",      "content": "{wiadomość_1}"},
│   │    {"role": "assistant", "content": "{odpowiedź_1}"},
│   │    {"role": "user",      "content": "{wiadomość_2}"},
│   │  ]
│   │
│   │  UWAGA: System prompt NIE jest w messages[]!
│   │  Mapowanie ról: "user"→"user", "model"→"assistant"
│
├─ response = self.anthropic_client.messages.create(
│     model=self.model_name,
│     max_tokens=self.max_tokens,         ← konfigurowalny (4096)
│     system=self.system_instruction,     ← OSOBNY parametr top-level!
│     messages=cast(Iterable, messages)
│   )
│
├─ Ekstrakcja tekstu z content blocks:
│   for block in response.content:
│     if block.type == "text":
│       response_text += block.text
│
├─ Tracking tokenów (unikalne dla Anthropic):
│   input_tokens = response.usage.input_tokens
│   output_tokens = response.usage.output_tokens
│   self._total_input_tokens += input_tokens
│   self._total_output_tokens += output_tokens
│
├─ Dodanie do _history:
│   {"role": "model", "parts": [{"text": response_text}]}
│
└─ return AnthropicResponse(response_text, input_tokens, output_tokens)
```

**Kluczowe cechy Anthropic:**

1. **System prompt jako parametr `system=`** — NIE jako wiadomość w `messages[]`. To ważna różnica vs Ollama, które wstawia system prompt jako pierwszą wiadomość w tablicy.

2. **`max_tokens` jest wymagany** — Anthropic API wymaga jawnego podania limitu tokenów odpowiedzi (domyślnie 4096).

3. **Content blocks** — Odpowiedź Anthropic to lista bloków (`"text"`, `"tool_use"`, etc.). Iterujemy po blokach typu `"text"`.

4. **Tracking tokenów** — `response.usage.input_tokens` i `output_tokens` dają precyzyjne dane per request. Kumulowane w sesji.

5. **`cast(Iterable, messages)`** — type hint dla Pythona (API oczekuje `Iterable`).

### Liczenie tokenów — natywne API z fallbackiem

```
count_history_tokens(history)
│
├─ TRY: self._client.messages.count_tokens(
│     model=self.model_name,
│     messages=messages  ← skonwertowane na format Anthropic
│   )
│   └─ return response.input_tokens
│
└─ EXCEPT (fallback heurystyka):
    word_count * 0.75
```

Anthropic ma **natywne API do liczenia tokenów** (`messages.count_tokens()`), z automatycznym fallbackiem na heurystykę.

### `AnthropicResponse` — rozszerzony response

```python
class AnthropicResponse:
    text: str                 # Tekst odpowiedzi
    input_tokens: int = 0     # Tokeny wejściowe tego requestu
    output_tokens: int = 0    # Tokeny wyjściowe tego requestu
```

Jedyny response z per-request informacjami o zużyciu tokenów.

### Emoji i komunikaty

- Preparing: `"🧠 Przygotowywanie klienta Anthropic..."`
- Ready: `"✅ Klient Anthropic gotowy do użycia (Model: claude-haiku-4-5-20251001, Key: sk-a...Yz9z)"`

---

## 11. Walidacja konfiguracji — Pydantic

Każdy silnik ma dedykowany model Pydantic (`BaseModel`) do walidacji konfiguracji z zmiennych środowiskowych. Walidacja odbywa się **przed** tworzeniem klienta — zasada fail-fast.

### `GeminiConfig`

```python
class GeminiConfig(BaseModel):
    engine: Literal["GEMINI"] = "GEMINI"
    model_name: str                              # np. "gemini-2.5-flash"
    gemini_api_key: str  # min_length=1

    @validator('gemini_api_key')
    def validate_api_key(cls, v):
        if not v or v.strip() == "":
            raise ValueError("GEMINI_API_KEY nie może być pusty")
        return v.strip()
```

### `LlamaConfig`

```python
class LlamaConfig(BaseModel):
    engine: Literal["LLAMA"] = "LLAMA"
    model_name: str
    llama_model_path: str                        # ścieżka do .gguf
    llama_gpu_layers: int = 1     # ge=0
    llama_context_size: int = 2048  # ge=1

    @validator('llama_model_path')
    def validate_model_path(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"Plik modelu nie istnieje: {v}")
        if not v.endswith('.gguf'):
            raise ValueError("Plik modelu musi mieć rozszerzenie .gguf")
        return v
```

**Najsurowsza walidacja** — sprawdza istnienie pliku na dysku i rozszerzenie.

### `OllamaConfig`

```python
class OllamaConfig(BaseModel):
    engine: Literal["OLLAMA"] = "OLLAMA"
    model_name: str
    ollama_host: str = "http://localhost:11434"

    @validator('model_name')  # nie pusty, strip
    @validator('ollama_host')  # nie pusty, http(s)://, strip + rstrip('/')
```

### `AnthropicConfig`

```python
class AnthropicConfig(BaseModel):
    engine: Literal["ANTHROPIC"] = "ANTHROPIC"
    model_name: str
    anthropic_api_key: str  # min_length=1
    max_tokens: int = 4096  # ge=1

    @validator('anthropic_api_key')   # nie pusty, strip
    @validator('model_name')          # nie pusty, strip
```

### Porównanie walidacji per silnik

| Aspekt | Gemini | LLaMA | Ollama | Anthropic |
|--------|--------|-------|--------|-----------|
| API Key | ✅ wymagany | ❌ brak | ❌ brak | ✅ wymagany |
| Model name | ✅ (z default) | ✅ (z default) | ✅ (z default) | ✅ (z default) |
| Ścieżka pliku | ❌ | ✅ istnienie + .gguf | ❌ | ❌ |
| Host URL | ❌ | ❌ | ✅ http(s):// | ❌ |
| Max tokens | ❌ | ❌ | ❌ | ✅ ge=1 |
| GPU layers | ❌ | ✅ ge=0 | ❌ | ❌ |
| Context size | ❌ | ✅ ge=1 | ❌ | ❌ |

---

## 12. Warstwa plików — `files/`

### `files/config.py` — Konfiguracja ścieżek

```python
LOG_DIR    = ~/.azor/                    # Katalog sesji i WAL
OUTPUT_DIR = ~/.azor/output/             # Katalog wyjściowy (PDF)
WAL_FILE   = ~/.azor/azor-wal.json       # Plik Write-Ahead Log
```

Katalogi tworzone automatycznie przy importcie (`os.makedirs(exist_ok=True)`). `load_dotenv()` wywoływane tutaj również.

### `files/session_files.py` — Operacje na plikach sesji

#### `load_session_history(session_id)` → `(List[Dict], str | None)`

```
load_session_history(session_id)
│
├─ log_filename = ~/.azor/{session_id}-log.json
├─ Nie istnieje → ([], "Session log file does not exist...")
├─ JSONDecodeError → ([], "Cannot decode log file...")
└─ Konwersja z JSON na format uniwersalny:
    {"role": ..., "timestamp": ..., "text": ...}
                        ↓
    {"role": role, "parts": [{"text": text}]}
```

#### `save_session_history(session_id, history, system_prompt, model_name)` → `(bool, str | None)`

```
save_session_history(...)
│
├─ len(history) < 2 → (True, None) — nie zapisuj pustych sesji
├─ Konwersja: Dict → JSON z timestamp
└─ json.dump() → ~/.azor/{session_id}-log.json
```

**Format pliku JSON na dysku:**
```json
{
    "session_id": "a1b2c3d4-...",
    "model": "gemini-2.5-flash",
    "system_role": "Jesteś pomocnym asystentem...",
    "history": [
        {"role": "user",  "timestamp": "2026-02-22T10:00:00", "text": "Cześć!"},
        {"role": "model", "timestamp": "2026-02-22T10:00:01", "text": "Hau hau!"}
    ]
}
```

`ensure_ascii=False` pozwala na zapis polskich znaków.

#### `list_sessions()` → `List[Dict]`

Skanuje `~/.azor/` po plikach `*-log.json`. Zwraca metadane: ID, liczba wiadomości, data ostatniej aktywności.

#### `remove_session_file(session_id)` → `(bool, str | None)`

Proste `os.remove()` z obsługą błędów (plik nie istnieje, OSError).

### `files/wal.py` — Write-Ahead Log

WAL to dodatkowa warstwa bezpieczeństwa — zapisuje **każdą interakcję natychmiast**, niezależnie od głównego zapisu sesji.

```
append_to_wal(session_id, prompt, response_text, total_tokens, model_name)
│
├─ wal_entry = {
│     "timestamp": now().isoformat(),
│     "session_id": session_id,
│     "model": model_name,
│     "prompt": prompt,
│     "response": response_text,
│     "tokens_used": total_tokens
│   }
│
├─ Odczyt istniejącego WAL → JSONDecodeError → reset
├─ data.append(wal_entry)
└─ json.dump(data) → ~/.azor/azor-wal.json
```

**Cel WAL-a**: Nawet jeśli główny plik sesji nie zostanie zapisany (crash, kill), WAL zawiera historię wszystkich interakcji. Jest to **append-only log** w jednym pliku globalnym.

### `files/pdf/pdf.py` — Generowanie PDF

```
generate_pdf_from_markdown(markdown_content, output_filename)
│
├─ FPDF() — tworzenie dokumentu
├─ Ładowanie czcionek Lato (Regular, Bold, Italic, BoldItalic)
├─ Markdown → HTML (biblioteka markdown)
├─ Owinięcie HTML w <font face="Lato">...</font>
├─ pdf.write_html(html_template)
└─ pdf.output(~/.azor/output/{filename})
```

---

## 13. System komend — `commands/` + `command_handler.py`

### `command_handler.py` — Router komend

Centralne miejsce obsługi komend slash:

```python
VALID_SLASH_COMMANDS = ['/exit', '/quit', '/switch', '/help', '/session', '/pdf']
```

**`handle_command(user_input) → bool`** (True = wyjście z pętli):

| Komenda | Działanie |
|---------|-----------|
| `/help` | `display_help(session_id)` |
| `/exit`, `/quit` | Komunikat + `return True` |
| `/switch <ID>` | `manager.switch_to_session(ID)` |
| `/session <sub>` | Delegacja do `handle_session_subcommand()` |
| `/pdf` | `export_session_to_pdf()` |
| Inna | Komunikat o błędzie + help |

**`handle_session_subcommand(subcommand, manager)`:**

| Podkomenda | Działanie |
|------------|-----------|
| `list` | `list_sessions_command()` — lista sesji z metadanymi |
| `display` | `display_full_session()` — numerowana pełna historia |
| `pop` | `current.pop_last_exchange()` — usunięcie ostatniej pary user+model |
| `clear` | `current.clear_history()` — wyczyszczenie historii |
| `new` | `manager.create_new_session()` — zapisanie bieżącej i nowa sesja |
| `remove` | `remove_session_command()` — usunięcie pliku sesji + nowa sesja |

### Pliki komend (`commands/`)

| Plik | Funkcja | Opis |
|------|---------|------|
| `welcome.py` | `print_welcome()` | ASCII art psa z dymkiem "Woof Woof!" |
| `session_list.py` | `list_sessions_command()` | Lista sesji z `session_files.list_sessions()` |
| `session_display.py` | `display_full_session()` | Numerowana historia: `[1] TY: ...`, `[2] AZOR: ...` |
| `session_summary.py` | `display_history_summary()` | Ostatnie 2 wiadomości + info o pominiętych |
| `session_remove.py` | `remove_session_command()` | Usunięcie pliku + nowa sesja |
| `session_to_pdf.py` | `export_session_to_pdf()` | Historia → Markdown → PDF |

---

## 14. Pełny flow wysłania wiadomości — krok po kroku

Poniżej pełny, liniowy flow od momentu wpisania tekstu do wyświetlenia odpowiedzi — z zaznaczeniem, gdzie każdy silnik się różni:

```
 1. Użytkownik wpisuje tekst w prompt_toolkit
    prompt("TY: ", completer=..., lexer=...) → "Cześć, Azor!"

 2. chat.main_loop() otrzymuje user_input = "Cześć, Azor!"

 3. Tekst NIE zaczyna się od '/' → gałąź rozmowy

 4. session = manager.get_current_session() → aktywna ChatSession

 5. response = session.send_message("Cześć, Azor!")
    │
    │  5a. _llm_chat_session.send_message("Cześć, Azor!")
    │      │
    │      │  [GEMINI]:
    │      │  └─ gemini_session.send_message(text) → Google API call
    │      │     → SDK automatycznie zarządza historią
    │      │     → return natywny Response
    │      │
    │      │  [LLAMA]:
    │      │  ├─ Dodaje user msg do _history
    │      │  ├─ _build_prompt_from_history()
    │      │  │   → "System: ...\n\nUser: Cześć, Azor!\n\nAssistant:"
    │      │  ├─ llama_model(prompt, max_tokens=512, stop=[...])
    │      │  ├─ Dodaje model msg do _history
    │      │  └─ return LlamaResponse(text)
    │      │
    │      │  [OLLAMA]:
    │      │  ├─ Dodaje user msg do _history
    │      │  ├─ _build_messages_from_history()
    │      │  │   → [{"role":"system",...}, {"role":"user","content":"Cześć, Azor!"}]
    │      │  ├─ ollama_client.chat(model=..., messages=..., stream=False)
    │      │  ├─ Dodaje model msg do _history
    │      │  └─ return OllamaResponse(text)
    │      │
    │      │  [ANTHROPIC]:
    │      │  ├─ Dodaje user msg do _history
    │      │  ├─ _build_messages_from_history()
    │      │  │   → [{"role":"user","content":"Cześć, Azor!"}]
    │      │  ├─ client.messages.create(model=..., max_tokens=4096,
    │      │  │     system=system_instruction, messages=messages)
    │      │  ├─ Ekstrakcja z content blocks + tracking tokenów
    │      │  ├─ Dodaje model msg do _history
    │      │  └─ return AnthropicResponse(text, in_tokens, out_tokens)
    │
    │  5b. self._history = _llm_chat_session.get_history()
    │      [GEMINI]: Konwersja Content → Dict
    │      [Inne]:   Zwrot bezpośrednio _history (już Dict)
    │
    │  5c. total_tokens = count_tokens()
    │      [GEMINI]:    client.models.count_tokens() ← precyzyjne API
    │      [LLAMA]:     llama_model.tokenize() ← wbudowany tokenizer
    │      [OLLAMA]:    word_count * 0.75 ← heurystyka
    │      [ANTHROPIC]: TRY count_tokens() EXCEPT heurystyka
    │
    │  5d. append_to_wal(...) → ~/.azor/azor-wal.json

 6. (total, remaining, max) = session.get_token_info()

 7. console.print_assistant("AZOR: Hau hau! Jestem Azor!")  → CYAN

 8. console.print_info("Tokens: 150 (Pozostało: 32618 / 32768)")

 9. session.save_to_file()
    → session_files.save_session_history(...)
    → ~/.azor/{session_id}-log.json

10. Powrót do pętli → czekanie na kolejny input
```

---

## 15. Uniwersalny format historii

Jednym z kluczowych rozwiązań architektonicznych jest **uniwersalny format historii** — jednolita struktura danych niezależna od silnika LLM.

### Format w pamięci (Python Dict)

```python
{"role": "user" | "model", "parts": [{"text": "treść wiadomości"}]}
```

### Konwersje per silnik

```
                    Uniwersalny format (Dict)
                    {"role": "user|model", "parts": [{"text": "..."}]}
                           │
          ┌────────────────┼────────────────────────────────┐
          │                │                                │
     ┌────▼─────┐   ┌─────▼──────┐   ┌──────▼──────┐  ┌───▼────┐
     │  Gemini   │   │   LLaMA    │   │   Ollama    │  │Anthropic│
     │ Content   │   │  Prompt    │   │  Messages   │  │Messages │
     │ objects   │   │  string    │   │   list      │  │  list   │
     └──────────┘   └────────────┘   └─────────────┘  └─────────┘

Gemini:    types.Content(role=..., parts=[types.Part.from_text(...)])
LLaMA:     "System: ...\n\nUser: ...\n\nAssistant: ...\n\nAssistant:"
Ollama:    [{"role": "system|user|assistant", "content": "..."}]
Anthropic: [{"role": "user|assistant", "content": "..."}] + system= osobno
```

### Format na dysku (JSON)

```json
{"role": "user", "timestamp": "2026-02-22T10:00:00", "text": "treść"}
```

Konwersja przy zapisie: `parts[0]["text"]` → `text` + dodanie `timestamp`.
Konwersja przy odczycie: `text` → `parts: [{"text": text}]`.

---

## 16. Porównanie klientów LLM — tabela zbiorcza

| Cecha | Gemini | LLaMA (llama.cpp) | Ollama | Anthropic |
|-------|--------|-------------------|--------|-----------|
| **ENGINE env** | `GEMINI` | `LLAMA_CPP` | `OLLAMA` | `ANTHROPIC` |
| **Typ połączenia** | API chmurowe | Model lokalny (in-process) | Serwer lokalny (HTTP) | API chmurowe |
| **SDK** | `google-genai` | `llama-cpp-python` | `ollama` | `anthropic` |
| **Wymaga klucza API** | ✅ `GEMINI_API_KEY` | ❌ | ❌ | ✅ `ANTHROPIC_API_KEY` |
| **Wymaga serwera** | ❌ (chmura) | ❌ (in-process) | ✅ `ollama serve` | ❌ (chmura) |
| **Wymaga pliku modelu** | ❌ | ✅ `.gguf` | ❌ | ❌ |
| **System prompt** | Parametr konfiguracji | Część promptu tekstowego | Wiadomość `system` | Osobny parametr `system=` |
| **Zarządzanie historią** | Natywne (stateful SDK) | Manualne (wrapper) | Manualne (wrapper) | Manualne (wrapper) |
| **Format do API** | `Content` objects | Jeden string prompt | Lista messages | Lista messages |
| **max_tokens** | Dynamiczny | 512 (hardcoded) | Domyślny Ollama | Konfigurowalny (4096) |
| **Liczenie tokenów** | ✅ Natywne API | ✅ Tokenizer modelu | ❌ Heurystyka | ✅/❌ API + fallback |
| **Tracking per request** | ❌ | ❌ | ❌ | ✅ (input + output) |
| **thinking_budget** | ✅ Obsługiwany | ❌ Ignorowany | ❌ Ignorowany | ❌ Ignorowany |
| **Emoji** | 🤖 | 🦙 | 🦙 | 🧠 |
| **Domyślny model** | `gemini-2.5-flash` | `llama-3.1-8b-instruct` | `qwen2.5:7b-instruct` | `claude-haiku-4-5-20251001` |
| **Obsługa błędów init** | `sys.exit(1)` | `raise RuntimeError` | `raise RuntimeError` | `sys.exit(1)` |

---

## 17. Diagram zależności między modułami

```
run.py
  └─ chat.py
       ├─ cli/args.py                    (parsowanie argumentów)
       ├─ cli/prompt.py                  (input użytkownika)
       ├─ cli/console.py                 (kolorowe output)
       ├─ commands/welcome.py            (ASCII art)
       ├─ command_handler.py
       │    ├─ commands/session_list.py
       │    │    └─ files/session_files.py
       │    ├─ commands/session_display.py
       │    ├─ commands/session_summary.py
       │    ├─ commands/session_remove.py
       │    │    └─ session/session_manager.py
       │    └─ commands/session_to_pdf.py
       │         └─ files/pdf/pdf.py
       │              └─ files/config.py
       │
       └─ session/ (singleton get_session_manager)
            ├─ session_manager.py
            │    └─ assistant/azor.py
            │         └─ assistant/assistent.py
            │
            └─ chat_session.py
                 ├─ assistant/assistent.py       (system prompt + nazwa)
                 ├─ files/session_files.py       (zapis/odczyt JSON)
                 ├─ files/wal.py                 (Write-Ahead Log)
                 │    └─ files/config.py
                 │
                 └─ llm/ (dynamiczny wybór na podstawie ENGINE)
                      ├─ gemini_client.py
                      │    └─ gemini_validation.py    (Pydantic)
                      ├─ llama_client.py
                      │    └─ llama_validation.py     (Pydantic)
                      ├─ ollama_client.py
                      │    └─ ollama_validation.py    (Pydantic)
                      └─ anthropic_client.py
                           └─ anthropic_validation.py (Pydantic)
```

### Przepływ danych — podsumowanie

```
Użytkownik → prompt_toolkit → chat.main_loop()
                                    │
                              SessionManager (singleton)
                                    │
                              ChatSession
                               ├─ Assistant (tożsamość: "AZOR")
                               ├─ LLM Client (silnik) ← wybór: ENGINE env var
                               │   └─ Chat Session Wrapper
                               │       └─ SDK / Model API
                               ├─ WAL (bezpieczeństwo: append-only log)
                               └─ Session Files (persystencja: JSON)
                                    │
                              ~/.azor/{id}-log.json
                              ~/.azor/azor-wal.json
                              ~/.azor/output/{id}.pdf
```

---

*Dokument wygenerowany na podstawie analizy pełnego kodu źródłowego aplikacji Azor ChatDog (Python). Ostatnia aktualizacja: luty 2026.*



