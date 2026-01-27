# 📚 Plan pracy domowej: Zadanie 3 - Budowa własnych tokenizerów BPE

## Cel zadania
Stworzenie własnych tokenizerów BPE na podstawie różnych korpusów polskich tekstów, pobranie tokenizera z HuggingFace (`allegro/herbert-base-cased`) i przeprowadzenie analizy porównawczej efektywności tokenizacji.

---

## 🔧 Krok 0: Konfiguracja środowiska

### Przejdź do folderu projektu
```bash
cd M1/tokenizer
```

### Utwórz wirtualne środowisko Python
```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
```

### Zainstaluj wymagane pakiety
```bash
pip install -r requirements.txt
```

Plik `requirements.txt` zawiera:
- `tokenizers` - biblioteka HuggingFace do budowania tokenizerów BPE
- `transformers` - do pobierania tokenizerów z HuggingFace (Herbert)
- `rich` - do wizualizacji wyników w terminalu

### Sprawdź dostępność korpusów
```bash
python corpora.py
```
Upewnij się, że foldery `korpus-wolnelektury/*.txt` i `korpus-nkjp/output/*.txt` zawierają pliki.

---

## 📝 Krok 1: Zbuduj 4 własne tokenizery

Skrypt `tokenizer-build.py` został zdynamizowany i przyjmuje argumenty CLI:
- `--corpus` - nazwa korpusu (`PAN_TADEUSZ`, `WOLNELEKTURY`, `NKJP`, `ALL`)
- `--output` - ścieżka wyjściowa dla tokenizera
- `--vocab-size` - rozmiar słownika (domyślnie 32000)

### Metoda 1: Użyj skryptu automatycznego
```bash
chmod +x build-all-tokenizers.sh
./build-all-tokenizers.sh
```

### Metoda 2: Uruchom ręcznie każdy tokenizer
```bash
# 1. Pan Tadeusz (12 ksiąg)
python tokenizer-build.py --corpus PAN_TADEUSZ --output tokenizers/tokenizer-pan-tadeusz.json

# 2. Wolne Lektury (cały korpus)
python tokenizer-build.py --corpus WOLNELEKTURY --output tokenizers/tokenizer-wolnelektury.json

# 3. NKJP (Narodowy Korpus Języka Polskiego)
python tokenizer-build.py --corpus NKJP --output tokenizers/tokenizer-nkjp.json

# 4. Wszystkie korpusy razem
python tokenizer-build.py --corpus ALL --output tokenizers/tokenizer-all-corpora.json
```

### Na co zwrócić uwagę:
- **Większy korpus** = dłuższy czas treningu, ale lepsze pokrycie słownictwa
- **min_frequency=2** - tokeny występujące tylko raz są ignorowane
- Obserwuj liczbę plików i czas treningu

---

## 🌐 Krok 2: Pobierz tokenizer Herbert z HuggingFace

Herbert (`allegro/herbert-base-cased`) to polski model językowy używający WordPiece (nie BPE).

```bash
python download-herbert.py
```

Tokenizer zostanie zapisany do `tokenizers/herbert/`.

**Uwaga:** Herbert wymaga innej metody ładowania niż tokenizery BPE:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("tokenizers/herbert")
```

---

## 🔽 Krok 3: Pobierz tokenizery Bielik (jeśli brakuje)

Jeśli pliki `bielik-v1/v2/v3-tokenizer.json` nie istnieją:

```bash
python download-bielik.py
```

**Uwaga:** Może być wymagana akceptacja [terms of use](https://bielik.ai/terms/) na HuggingFace.

---

## 📊 Krok 4: Uruchom porównanie z wizualizacją

```bash
python tokenizer-compare.py
```

### Co robi skrypt:
1. Ładuje wszystkie dostępne tokenizery (8 łącznie):
   - 3 bielikowe: `bielik-v1`, `bielik-v2`, `bielik-v3`
   - 1 herbert: `tokenizers/herbert/`
   - 4 własne: `pan-tadeusz`, `wolnelektury`, `nkjp`, `all-corpora`

2. Tokenizuje 3 teksty testowe:
   - `Pan Tadeusz Księga 1` (polski, z wolnelektury)
   - `Pickwick Papers` (angielski, Gutenberg)
   - `Fryderyk Chopin` (polski, Wikipedia)

3. Wyświetla wizualizację w terminalu z:
   - Poziomymi słupkami posortowanymi od najlepszego
   - Medalami 🥇🥈🥉 dla top 3
   - Liczbą tokenów po prawej stronie

---

## 🧪 Krok 5: Eksperymentuj z vocab_size

Sprawdź wpływ rozmiaru słownika na efektywność tokenizacji:

```bash
# Mniejszy słownik (16k)
python tokenizer-build.py --corpus ALL --output tokenizers/tokenizer-all-16k.json --vocab-size 16000

# Większy słownik (64k)
python tokenizer-build.py --corpus ALL --output tokenizers/tokenizer-all-64k.json --vocab-size 64000
```

### Dodaj nowe tokenizery do porównania:
Edytuj `tokenizer-compare.py` i dodaj nowe wpisy do słownika `BPE_TOKENIZERS`.

---

## 📈 Krok 6: Wnioski do wyciągnięcia

Po uruchomieniu porównania odpowiedz na pytania:

### 1. Który tokenizer był najefektywniejszy dla każdego tekstu?

| Tekst | Spodziewany najlepszy tokenizer |
|-------|--------------------------------|
| Pan Tadeusz Księga 1 | `pan-tadeusz` (treninowany na tym korpusie) |
| Pickwick Papers (ang.) | `bielik-v1/v2` (mają angielski w słowniku) |
| Fryderyk Chopin (pl) | `nkjp` lub `all-corpora` |

### 2. Jak wypada Bielik v3 vs v1/v2?
- Bielik v3 powinien być bardziej efektywny dla polskiego tekstu
- Bielik v1/v2 lepszy dla angielskiego (oparty na Mistral)

### 3. Jak vocab_size wpływa na liczbę tokenów?
- **Większy słownik** → mniej tokenów (dłuższe segmenty)
- **Mniejszy słownik** → więcej tokenów (krótsze segmenty)
- Istnieje punkt nasycenia - po pewnym rozmiarze przyrosty maleją

### 4. Dlaczego własny tokenizer jest najefektywniejszy dla "swojego" tekstu?
- Tokenizer uczy się częstych wzorców z korpusu treningowego
- Tekst z tego samego źródła zawiera te same wzorce
- To potwierdza, że tokenizery powinny być dostosowane do domeny

---

## 📁 Struktura plików po wykonaniu zadania

```
M1/tokenizer/
├── tokenizer-build.py          # Zdynamizowany skrypt budowania
├── tokenizer-compare.py        # Porównanie z wizualizacją
├── download-herbert.py         # Pobieranie Herbert z HF
├── download-bielik.py          # Pobieranie Bielik z HF
├── build-all-tokenizers.sh     # Skrypt automatyczny
├── corpora.py                  # Utils do korpusów
├── requirements.txt            # Zależności
├── venv/                       # Środowisko wirtualne
└── tokenizers/
    ├── bielik-v1-tokenizer.json
    ├── bielik-v2-tokenizer.json
    ├── bielik-v3-tokenizer.json
    ├── tokenizer-pan-tadeusz.json
    ├── tokenizer-wolnelektury.json
    ├── tokenizer-nkjp.json
    ├── tokenizer-all-corpora.json
    └── herbert/
        ├── tokenizer.json
        └── tokenizer_config.json
```

---

## 🎯 Podsumowanie

1. **Najefektywniejszy tokenizer** = najmniej tokenów dla danego tekstu
2. **Własne tokenizery** są najlepsze dla tekstów z ich korpusu treningowego
3. **Bielik v3** > v1/v2 dla polskiego (nowszy, lepiej zoptymalizowany)
4. **Herbert** (WordPiece) dobrze radzi sobie z polskim
5. **vocab_size** ma znaczący wpływ - eksperymentuj!

---

## 🔗 Przydatne linki

- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/)
- [Bielik Terms of Use](https://bielik.ai/terms/)
- [Herbert na HuggingFace](https://huggingface.co/allegro/herbert-base-cased)
- [Badanie o tokenizacji (arXiv)](https://arxiv.org/pdf/2503.01996)
