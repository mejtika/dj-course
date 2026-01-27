# 📖 Podsumowanie Zadania 3: Tokenizery BPE

## Spis treści
1. [Podstawowe pojęcia](#-podstawowe-pojęcia)
2. [Co to jest tokenizer?](#-co-to-jest-tokenizer)
3. [Algorytm BPE](#-algorytm-bpe-byte-pair-encoding)
4. [Omówienie plików projektu](#-omówienie-plików-projektu)
5. [Składnia Pythona - najważniejsze elementy](#-składnia-pythona---najważniejsze-elementy)
6. [Wnioski z zadania](#-wnioski-z-zadania)

---

## 🧠 Podstawowe pojęcia

### Token
**Token** to najmniejsza jednostka tekstu, którą model językowy (LLM) przetwarza. Może to być:
- Całe słowo: `"hello"` → `["hello"]`
- Część słowa: `"unhappiness"` → `["un", "happiness"]`
- Pojedynczy znak: `"🎉"` → `["🎉"]`

### Tokenizacja
**Tokenizacja** to proces dzielenia tekstu na tokeny. Jest to pierwszy krok w przetwarzaniu tekstu przez LLM.

```
"Litwo! Ojczyzno moja!" → ["Li", "two", "!", "Ojczy", "zno", "moja", "!"]
```

### Słownik (Vocabulary)
**Słownik** to zbiór wszystkich znanych tokenów. Rozmiar słownika (np. 32 000) określa ile różnych tokenów może rozpoznać tokenizer.

### Korpus
**Korpus** to zbiór tekstów używanych do trenowania tokenizera. Im większy i bardziej różnorodny korpus, tym lepszy tokenizer.

---

## 🔤 Co to jest tokenizer?

Tokenizer to narzędzie, które:
1. **Dzieli tekst** na mniejsze jednostki (tokeny)
2. **Mapuje tokeny na liczby** (ID) - bo komputery rozumieją tylko liczby
3. **Odwraca proces** - zamienia ID z powrotem na tekst

### Przykład tokenizacji

```python
tekst = "Litwo! Ojczyzno moja!"

# Po tokenizacji:
tokeny = ["Li", "two", "!", "Ojczy", "zno", "moja", "!"]
ids = [496, 521, 5, 1272, 482, 1850, 5]
```

### Dlaczego to ważne?

| Aspekt | Wpływ |
|--------|-------|
| **Koszty API** | Płacisz za tokeny, nie znaki! Lepszy tokenizer = mniej tokenów = niższe koszty |
| **Jakość modelu** | Tokenizer wpływa na to, jak model "rozumie" tekst |
| **Język** | Tokenizer wytrenowany na polskim tekście lepiej radzi sobie z polszczyzną |

---

## 🔧 Algorytm BPE (Byte Pair Encoding)

BPE to najpopularniejszy algorytm tokenizacji używany przez GPT, Bielik, Mistral i inne modele.

### Jak działa BPE?

1. **Start**: Zacznij od pojedynczych znaków jako tokenów
2. **Zlicz pary**: Znajdź najczęściej występującą parę znaków
3. **Połącz**: Zamień tę parę na nowy token
4. **Powtórz**: Wróć do kroku 2 aż osiągniesz żądany rozmiar słownika

### Przykład krok po kroku

```
Korpus: "low lower lowest"

Krok 1: Początkowe tokeny
  l o w _ l o w e r _ l o w e s t
  
Krok 2: Najczęstsza para = "lo" (występuje 3x)
  Nowy token: "lo"
  lo w _ lo w e r _ lo w e s t

Krok 3: Najczęstsza para = "low" (występuje 3x)
  Nowy token: "low"
  low _ low e r _ low e s t

... i tak dalej
```

### BPE vs WordPiece

| BPE | WordPiece |
|-----|-----------|
| Używany przez: GPT, Bielik, Mistral | Używany przez: BERT, Herbert |
| Łączy najczęstsze pary | Maksymalizuje likelihood |
| Prosty algorytm | Bardziej złożony |

---

## 📁 Omówienie plików projektu

### 1. `corpora.py` - Zarządzanie korpusami

Ten plik definiuje **skąd brać teksty do trenowania** tokenizera.

```python
# IMPORT: Ładowanie bibliotek
import glob                    # Do wyszukiwania plików po wzorcu (*.txt)
from pathlib import Path       # Nowoczesna obsługa ścieżek plików

# SŁOWNIKI: Definiowanie ścieżek do korpusów
CORPORA_DIRS = {
    "NKJP": Path("../korpus-nkjp/output"),      # Narodowy Korpus Języka Polskiego
    "WOLNELEKTURY": Path("../korpus-wolnelektury"),  # Wolne Lektury
}

# SŁOWNIKI: Listy plików dla każdego korpusu
CORPORA_FILES = {
    "NKJP": list(CORPORA_DIRS["NKJP"].glob("*.txt")),           # Wszystkie .txt z NKJP
    "WOLNELEKTURY": list(CORPORA_DIRS["WOLNELEKTURY"].glob("*.txt")),  # Wszystkie z Wolnych Lektur
    "PAN_TADEUSZ": list(CORPORA_DIRS["WOLNELEKTURY"].glob("pan-tadeusz-ksiega-*.txt")),  # Tylko Pan Tadeusz
}

# LIST COMPREHENSION: Tworzenie listy ALL bez duplikatów
KEYS_WITHOUT_PAN_TADEUSZ = [key for key in CORPORA_FILES.keys() if key != "PAN_TADEUSZ"]
CORPORA_FILES["ALL"] = [
    FILE for key in KEYS_WITHOUT_PAN_TADEUSZ for FILE in CORPORA_FILES[key]
]

# FUNKCJA: Pobieranie plików z korpusu po wzorcu
def get_corpus_file(corpus_name: str, glob_pattern: str) -> Path:
    if corpus_name not in CORPORA_FILES:
        raise ValueError(f"Corpus {corpus_name} not found")  # Rzuć błąd jeśli nie znaleziono
    return list(CORPORA_DIRS[corpus_name].glob(glob_pattern))

# ENTRY POINT: Kod uruchamiany tylko gdy plik jest wykonywany bezpośrednio
if __name__ == "__main__":    
    print("\ncorpora (total files):")
    for corpus_name, corpus_files in CORPORA_FILES.items():
        print(f"{corpus_name}: {len(corpus_files)}")  # f-string do formatowania
```

**Kluczowe elementy składni:**
- `Path()` - obiekt reprezentujący ścieżkę do pliku/folderu
- `.glob("*.txt")` - wyszukuje pliki pasujące do wzorca
- `list()` - konwertuje generator na listę
- `[x for x in lista]` - list comprehension (skrócona pętla tworząca listę)
- `if __name__ == "__main__"` - kod uruchamiany tylko gdy plik jest głównym skryptem

---

### 2. `tokenizer-build.py` - Budowanie tokenizera

Ten plik **trenuje nowy tokenizer BPE** na podstawie wybranego korpusu.

```python
# IMPORTY
import argparse                              # Parsowanie argumentów z linii komend
from tokenizers import Tokenizer             # Główna klasa tokenizera
from tokenizers.models import BPE            # Model BPE
from tokenizers.trainers import BpeTrainer   # Trener dla BPE
from tokenizers.pre_tokenizers import Whitespace  # Wstępny podział po spacjach
from corpora import CORPORA_FILES            # Import z naszego pliku corpora.py

# FUNKCJA z type hints (podpowiedziami typów)
def build_tokenizer(files: list[str], output_path: str, vocab_size: int = 32000):
    """
    Docstring - opis funkcji (pojawia się w dokumentacji).
    Buduje tokenizer BPE na podstawie podanych plików i zapisuje go do output_path.
    """
    # Krok 1: Inicjalizacja tokenizera z modelem BPE
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))  # [UNK] = nieznany token

    # Krok 2: Pre-tokenizer dzieli tekst po spacjach PRZED główną tokenizacją
    tokenizer.pre_tokenizer = Whitespace()

    # Krok 3: Konfiguracja trenera
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],  # Tokeny specjalne
        vocab_size=vocab_size,     # Rozmiar słownika (domyślnie 32000)
        min_frequency=2            # Token musi wystąpić min. 2 razy
    )

    # Krok 4: Trening tokenizera na plikach
    tokenizer.train(files, trainer=trainer)

    # Krok 5: Zapis do pliku JSON
    tokenizer.save(output_path)
    
    return tokenizer  # Zwróć wytrenowany tokenizer


def main():
    # ARGPARSE: Definiowanie argumentów CLI
    parser = argparse.ArgumentParser(
        description="Budowanie tokenizera BPE na podstawie korpusu tekstowego"
    )
    
    # Argument --corpus (wymagany)
    parser.add_argument(
        "--corpus",                           # Nazwa argumentu
        type=str,                             # Typ: string
        required=True,                        # Wymagany
        choices=list(CORPORA_FILES.keys()),   # Dozwolone wartości
        help=f"Nazwa korpusu: {list(CORPORA_FILES.keys())}"  # Pomoc
    )
    
    # Argument --output (wymagany)
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Ścieżka wyjściowa dla tokenizera"
    )
    
    # Argument --vocab-size (opcjonalny, domyślnie 32000)
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,                        # Wartość domyślna
        help="Rozmiar słownika (domyślnie 32000)"
    )

    # Parsuj argumenty z linii komend
    args = parser.parse_args()

    # Pobierz pliki dla wybranego korpusu
    files = [str(f) for f in CORPORA_FILES[args.corpus]]  # Konwersja Path → str

    # Buduj tokenizer
    tokenizer = build_tokenizer(files, args.output, args.vocab_size)

    # Test tokenizera
    test_texts = ["Litwo! Ojczyzno moja!"]
    for txt in test_texts:
        encoded = tokenizer.encode(txt)       # Tokenizacja
        print(f"Tokeny: {encoded.tokens}")    # Lista tokenów
        print(f"IDs: {encoded.ids}")          # Lista ID


# ENTRY POINT
if __name__ == "__main__":
    main()
```

**Wywołanie z CLI:**
```bash
python tokenizer-build.py --corpus PAN_TADEUSZ --output tokenizers/tokenizer-pan-tadeusz.json --vocab-size 32000
```

---

### 3. `tokenizer-compare.py` - Porównanie tokenizerów

Ten plik **porównuje efektywność** różnych tokenizerów i **wizualizuje wyniki**.

```python
# IMPORTY
from pathlib import Path
from tokenizers import Tokenizer              # Dla tokenizerów BPE
from transformers import AutoTokenizer        # Dla tokenizerów z HuggingFace (Herbert)
from rich.console import Console              # Kolorowy output w terminalu
from rich.panel import Panel                  # Ramki w terminalu
from typing import Optional                   # Typy opcjonalne (może być None)

# Obiekt do wypisywania kolorowego tekstu
console = Console()

# STAŁE: Ścieżki do tokenizerów
TOKENIZERS_DIR = Path("tokenizers")

# SŁOWNIK: Mapowanie nazwa → ścieżka dla tokenizerów BPE
BPE_TOKENIZERS = {
    "bielik-v1": TOKENIZERS_DIR / "bielik-v1-tokenizer.json",   # / łączy ścieżki
    "bielik-v2": TOKENIZERS_DIR / "bielik-v2-tokenizer.json",
    "bielik-v3": TOKENIZERS_DIR / "bielik-v3-tokenizer.json",
    "pan-tadeusz": TOKENIZERS_DIR / "tokenizer-pan-tadeusz.json",
    "wolnelektury": TOKENIZERS_DIR / "tokenizer-wolnelektury.json",
    "nkjp": TOKENIZERS_DIR / "tokenizer-nkjp.json",
    "all-corpora": TOKENIZERS_DIR / "tokenizer-all-corpora.json",
}

# SŁOWNIK: Teksty testowe
TEST_TEXTS = {
    "Pan Tadeusz Księga 1": Path("../korpus-wolnelektury/pan-tadeusz-ksiega-1.txt"),
    "Pickwick Papers": Path("../korpus-mini/the-pickwick-papers-gutenberg.txt"),
    "Fryderyk Chopin": Path("../korpus-mini/fryderyk-chopin-wikipedia.txt"),
}


# FUNKCJA z Optional (może zwrócić None)
def load_bpe_tokenizer(path: Path) -> Optional[Tokenizer]:
    """Ładuje tokenizer BPE z pliku JSON."""
    if not path.exists():         # Sprawdź czy plik istnieje
        return None               # Zwróć None jeśli nie
    try:
        return Tokenizer.from_file(str(path))  # Załaduj tokenizer
    except Exception as e:        # Złap dowolny błąd
        console.print(f"[red]Błąd: {e}[/red]")  # Wyświetl na czerwono
        return None


def count_tokens_bpe(tokenizer: Tokenizer, text: str) -> int:
    """Zlicza tokeny dla tokenizera BPE."""
    encoded = tokenizer.encode(text)  # Tokenizuj tekst
    return len(encoded.ids)           # Zwróć liczbę tokenów


def load_text(path: Path) -> str:
    """Wczytuje tekst z pliku."""
    with open(path, 'r', encoding='utf-8') as f:  # Context manager - auto-zamykanie
        return f.read()                            # Wczytaj całą zawartość


def visualize_results(text_name: str, results: dict[str, int]):
    """Wizualizuje wyniki jako poziome słupki."""
    
    # Sortuj wyniki od najmniejszej liczby tokenów (najlepszy = najmniej)
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    
    # Znajdź maksimum do skalowania słupków
    max_tokens = max(results.values())
    
    # Przygotuj linie do wyświetlenia
    lines = []
    for i, (name, tokens) in enumerate(sorted_results):  # enumerate daje indeks i wartość
        # Medale dla top 3
        if i == 0:
            medal = "🥇"
        elif i == 1:
            medal = "🥈"
        elif i == 2:
            medal = "🥉"
        else:
            medal = "  "
        
        # Oblicz długość słupka proporcjonalnie
        bar_length = int((tokens / max_tokens) * 50)
        bar = "▓" * bar_length + "░" * (50 - bar_length)
        
        # Formatuj liczbę z separatorem tysięcy
        tokens_formatted = f"{tokens:,}"  # :, dodaje separator tysięcy
        
        lines.append(f"{medal} {name.ljust(15)} {bar} {tokens_formatted:>10} tokenów")
    
    # Wyświetl w ramce
    panel = Panel("\n".join(lines), title=f"📊 {text_name}")
    console.print(panel)


def main():
    # Załaduj wszystkie tokenizery
    tokenizers = {}
    for name, path in BPE_TOKENIZERS.items():  # Iteruj po słowniku
        tokenizer = load_bpe_tokenizer(path)
        if tokenizer:                          # Jeśli udało się załadować
            tokenizers[name] = ("bpe", tokenizer)  # Dodaj do słownika
    
    # Dla każdego tekstu testowego
    for text_name, text_path in TEST_TEXTS.items():
        text = load_text(text_path)            # Wczytaj tekst
        
        # Zlicz tokeny dla każdego tokenizera
        results = {}
        for name, (tokenizer_type, tokenizer) in tokenizers.items():
            results[name] = count_tokens_bpe(tokenizer, text)
        
        # Wizualizuj wyniki
        visualize_results(text_name, results)


if __name__ == "__main__":
    main()
```

---

### 4. `download-herbert.py` - Pobieranie Herbert

```python
from transformers import AutoTokenizer  # Biblioteka HuggingFace
import os

OUTPUT_DIR = "tokenizers/herbert"

def main():
    # Pobierz tokenizer z HuggingFace (wymaga internetu)
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    
    # Utwórz folder jeśli nie istnieje
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # exist_ok=True nie rzuca błędu jeśli istnieje
    
    # Zapisz tokenizer lokalnie
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Test
    encoded = tokenizer.encode("Test tekstu")
    tokens = tokenizer.convert_ids_to_tokens(encoded)  # ID → tokeny
    print(f"Tokeny: {tokens}")


if __name__ == "__main__":
    main()
```

---

### 5. `download-bielik.py` - Pobieranie Bielik

```python
import os
import urllib.request  # Biblioteka do pobierania plików z internetu

# Słownik z URLami do tokenizerów
TOKENIZERS = {
    "bielik-v1": "https://huggingface.co/.../tokenizer.json",
    "bielik-v2": "https://huggingface.co/.../tokenizer.json",
    "bielik-v3": "https://huggingface.co/.../tokenizer.json",
}

def main():
    os.makedirs("tokenizers", exist_ok=True)
    
    for name, url in TOKENIZERS.items():  # Iteruj po słowniku
        output_path = os.path.join("tokenizers", f"{name}-tokenizer.json")  # Złącz ścieżkę
        
        if os.path.exists(output_path):   # Sprawdź czy plik już istnieje
            print(f"✓ {name} już istnieje")
            continue                       # Pomiń do następnej iteracji
        
        try:
            urllib.request.urlretrieve(url, output_path)  # Pobierz plik
            print(f"✓ Zapisano: {output_path}")
        except Exception as e:
            print(f"✗ Błąd: {e}")


if __name__ == "__main__":
    main()
```

---

## 🐍 Składnia Pythona - najważniejsze elementy

### 1. Importy

```python
import os                          # Import całego modułu
from pathlib import Path           # Import konkretnej klasy z modułu
from tokenizers import Tokenizer   # Import z zewnętrznej biblioteki
```

### 2. Zmienne i typy

```python
# Python nie wymaga deklaracji typów, ale można je dodać (type hints)
name: str = "tekst"                # String (napis)
count: int = 42                    # Integer (liczba całkowita)
price: float = 3.14                # Float (liczba zmiennoprzecinkowa)
is_active: bool = True             # Boolean (prawda/fałsz)
items: list = [1, 2, 3]            # Lista (tablica)
config: dict = {"key": "value"}    # Słownik (mapa klucz-wartość)
```

### 3. Funkcje

```python
# Definicja funkcji
def nazwa_funkcji(argument1: str, argument2: int = 10) -> str:
    """Docstring - opis funkcji."""
    wynik = argument1 + str(argument2)
    return wynik

# Wywołanie
rezultat = nazwa_funkcji("test", 5)  # "test5"
```

### 4. Pętle

```python
# Pętla for po liście
for item in [1, 2, 3]:
    print(item)

# Pętla for ze słownikiem
for key, value in {"a": 1, "b": 2}.items():
    print(f"{key}: {value}")

# enumerate - indeks + wartość
for i, item in enumerate(["a", "b", "c"]):
    print(f"{i}: {item}")  # 0: a, 1: b, 2: c
```

### 5. List Comprehension (skrócone pętle)

```python
# Zamiast:
wynik = []
for x in [1, 2, 3]:
    wynik.append(x * 2)

# Możesz napisać:
wynik = [x * 2 for x in [1, 2, 3]]  # [2, 4, 6]

# Z warunkiem:
parzyste = [x for x in [1, 2, 3, 4] if x % 2 == 0]  # [2, 4]
```

### 6. F-stringi (formatowanie tekstu)

```python
name = "Jan"
age = 25
print(f"Mam na imię {name} i mam {age} lat")

# Formatowanie liczb
tokens = 12345
print(f"{tokens:,}")      # "12,345" (separator tysięcy)
print(f"{tokens:>10}")    # "     12345" (wyrównanie do prawej, 10 znaków)
```

### 7. Context Manager (with)

```python
# Automatycznie zamyka plik po zakończeniu bloku
with open("plik.txt", "r", encoding="utf-8") as f:
    content = f.read()
# Tutaj plik jest już zamknięty
```

### 8. Try/Except (obsługa błędów)

```python
try:
    wynik = 10 / 0  # Spowoduje błąd
except ZeroDivisionError:
    print("Nie można dzielić przez zero!")
except Exception as e:  # Złap dowolny błąd
    print(f"Błąd: {e}")
```

### 9. Entry Point

```python
if __name__ == "__main__":
    main()
```
Ten blok uruchamia się **tylko gdy plik jest wykonywany bezpośrednio** (`python plik.py`), a nie gdy jest importowany.

---

## 📊 Wnioski z zadania

### 1. Efektywność tokenizacji zależy od korpusu treningowego

| Tekst | Najlepszy tokenizer | Dlaczego? |
|-------|---------------------|-----------|
| Pan Tadeusz | `pan-tadeusz` | Treninowany na tym samym tekście |
| Pickwick Papers (ang.) | `bielik-v1/v2` | Oparte na Mistral, mają angielski |
| Fryderyk Chopin (pl) | `nkjp` lub `herbert` | Treninowane na polskim |

### 2. Bielik v3 > v1/v2 dla polskiego

Bielik v3 jest nowszy i zoptymalizowany pod kątem polszczyzny. Produkuje mniej tokenów dla polskich tekstów.

### 3. Rozmiar słownika (vocab_size) ma znaczenie

| vocab_size | Efekt |
|------------|-------|
| 16 000 | Więcej tokenów, mniejszy plik |
| 32 000 | Domyślny, dobry kompromis |
| 64 000 | Mniej tokenów, większy plik |

### 4. Własny tokenizer = najlepsza efektywność

Jeśli wiesz, z jakim tekstem będziesz pracować, warto wytrenować dedykowany tokenizer. Oszczędza to tokeny (= pieniądze przy API).

---

## 🚀 Polecenia do uruchomienia

```bash
# 1. Przejdź do folderu
cd M1/tokenizer

# 2. Aktywuj środowisko wirtualne
source venv/bin/activate

# 3. Zainstaluj zależności
pip install -r requirements.txt

# 4. Zbuduj wszystkie tokenizery
./build-all-tokenizers.sh

# 5. Pobierz Herbert
python download-herbert.py

# 6. Pobierz Bielik (jeśli brakuje)
python download-bielik.py

# 7. Uruchom porównanie z wizualizacją
python tokenizer-compare.py
```

---

## 🔗 Przydatne zasoby

- [Dokumentacja tokenizers (HuggingFace)](https://huggingface.co/docs/tokenizers/)
- [Badanie o tokenizacji](https://arxiv.org/pdf/2503.01996) - cytowane w zadaniu
- [Python Tutorial (W3Schools)](https://www.w3schools.com/python/)
- [Real Python - Type Hints](https://realpython.com/python-type-checking/)
