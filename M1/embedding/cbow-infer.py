"""
CBOW Inference Script - Wnioskowanie z wytrenowanego modelu Word2Vec (CBOW)
Rozdzielony skrypt do testowania podobieństw słów.

Użycie:
    python cbow-infer.py --model models/cbow_all_bielik-v3.model --tokenizer bielik-v3
    python cbow-infer.py --model models/cbow_all_bielik-v3.model --tokenizer bielik-v3 --words król,książę,wojsko
    python cbow-infer.py --model models/cbow_all_bielik-v3.model --tokenizer bielik-v3 --analogy dziecko,kobieta
"""

import argparse
import numpy as np
from gensim.models import Word2Vec
from tokenizers import Tokenizer
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Dostępne tokenizery (muszą być zsynchronizowane z cbow-train.py)
TOKENIZERS = {
    "bielik-v1": "../tokenizer/tokenizers/bielik-v1-tokenizer.json",
    "bielik-v2": "../tokenizer/tokenizers/bielik-v2-tokenizer.json",
    "bielik-v3": "../tokenizer/tokenizers/bielik-v3-tokenizer.json",
    "pan-tadeusz": "../tokenizer/tokenizers/tokenizer-pan-tadeusz.json",
    "wolnelektury": "../tokenizer/tokenizers/tokenizer-wolnelektury.json",
    "nkjp": "../tokenizer/tokenizers/tokenizer-nkjp.json",
    "all-corpora": "../tokenizer/tokenizers/tokenizer-all-corpora.json",
    "custom": "../tokenizer/tokenizers/custom_bpe_tokenizer.json",
}

# Domyślne słowa do testowania
DEFAULT_TEST_WORDS = ['wojsko', 'szlachta', 'choroba', 'król', 'kobieta', 'dziecko', 'książę', 'rycerz']

# Domyślne analogie do testowania
DEFAULT_ANALOGIES = [
    ('dziecko', 'kobieta'),
    ('król', 'mężczyzna'),
]


def load_tokenizer(tokenizer_name: str) -> Tokenizer:
    """Ładuje tokenizer na podstawie nazwy."""
    if tokenizer_name not in TOKENIZERS:
        raise ValueError(f"Nieznany tokenizer: {tokenizer_name}. Dostępne: {list(TOKENIZERS.keys())}")

    path = TOKENIZERS[tokenizer_name]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono pliku tokenizera: {path}")

    return Tokenizer.from_file(path)


def load_model(model_path: str) -> Word2Vec:
    """Ładuje wytrenowany model Word2Vec."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Nie znaleziono modelu: {model_path}")

    return Word2Vec.load(model_path)


def get_word_vector(word: str, tokenizer: Tokenizer, model: Word2Vec):
    """
    Oblicza wektor dla słowa poprzez uśrednienie wektorów jego tokenów.
    Zwraca (wektor, lista_tokenów, lista_brakujących_tokenów)
    """
    # Tokenizacja słowa (dodajemy spacje, aby tokenizer traktował to jako osobne słowo)
    encoding = tokenizer.encode(" " + word + " ")
    word_tokens = [t.strip() for t in encoding.tokens if t.strip()]

    # Usuń specjalne tokeny (w tym tokeny Bielik)
    special_tokens = ['[CLS]', '[SEP]', '<s>', '</s>', '[PAD]', '[MASK]', '[UNK]', '▁']
    # Filtruj również tokeny hex typu <0x...>
    word_tokens = [t for t in word_tokens if t not in special_tokens and not t.startswith('<0x')]

    valid_vectors = []
    missing_tokens = []

    for token in word_tokens:
        if token in model.wv:
            valid_vectors.append(model.wv[token])
        else:
            missing_tokens.append(token)

    if not valid_vectors:
        return None, word_tokens, missing_tokens

    # Uśrednij wektory
    word_vector = np.mean(valid_vectors, axis=0)
    return word_vector, word_tokens, missing_tokens


def find_similar_words(word: str, tokenizer: Tokenizer, model: Word2Vec, topn: int = 10):
    """Znajduje najbardziej podobne słowa/tokeny do zadanego słowa."""
    word_vector, tokens, missing = get_word_vector(word, tokenizer, model)

    if word_vector is None:
        return None, tokens, missing

    similar = model.wv.most_similar(positive=[word_vector], topn=topn)
    return similar, tokens, missing


def find_analogy(words: list[str], tokenizer: Tokenizer, model: Word2Vec, topn: int = 10):
    """
    Znajduje słowa podobne do kombinacji wektorów.
    Np. dla ['dziecko', 'kobieta'] szuka słów podobnych do obu.
    """
    vectors = []
    all_tokens = []

    for word in words:
        vector, tokens, missing = get_word_vector(word, tokenizer, model)
        if vector is not None:
            vectors.append(vector)
            all_tokens.extend(tokens)

    if not vectors:
        return None, all_tokens

    # Sprawdź czy tokeny są w słowniku modelu
    valid_tokens = [t for t in all_tokens if t in model.wv]

    if len(valid_tokens) < len(words):
        # Fallback: użyj uśrednionych wektorów
        combined = np.mean(vectors, axis=0)
        similar = model.wv.most_similar(positive=[combined], topn=topn)
    else:
        # Użyj tokenów bezpośrednio
        similar = model.wv.most_similar(positive=valid_tokens, topn=topn)

    return similar, all_tokens


def display_similar_words(word: str, similar: list, tokens: list, missing: list):
    """Wyświetla wyniki podobieństwa w ładnym formacie."""
    if similar is None:
        console.print(f"[red]❌ Nie można obliczyć wektora dla '{word}'[/red]")
        if missing:
            console.print(f"   Brakujące tokeny: {missing}")
        return

    table = Table(title=f"🔍 Podobne do '{word}' (tokeny: {tokens})")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Token", style="cyan")
    table.add_column("Podobieństwo", justify="right", style="green")

    for i, (token, similarity) in enumerate(similar, 1):
        # Koloruj wysokie podobieństwa
        if similarity >= 0.7:
            sim_style = "bold green"
        elif similarity >= 0.5:
            sim_style = "yellow"
        else:
            sim_style = "dim"

        table.add_row(
            str(i),
            token,
            f"[{sim_style}]{similarity:.4f}[/{sim_style}]"
        )

    console.print(table)
    console.print()


def display_analogy(words: list[str], similar: list, tokens: list):
    """Wyświetla wyniki analogii."""
    if similar is None:
        console.print(f"[red]❌ Nie można obliczyć analogii dla {words}[/red]")
        return

    table = Table(title=f"🧮 Analogia: {' + '.join(words)} → ? (tokeny: {tokens})")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Token", style="cyan")
    table.add_column("Podobieństwo", justify="right", style="green")

    for i, (token, similarity) in enumerate(similar, 1):
        if similarity >= 0.7:
            sim_style = "bold green"
        elif similarity >= 0.5:
            sim_style = "yellow"
        else:
            sim_style = "dim"

        table.add_row(
            str(i),
            token,
            f"[{sim_style}]{similarity:.4f}[/{sim_style}]"
        )

    console.print(table)
    console.print()


def interactive_mode(tokenizer: Tokenizer, model: Word2Vec):
    """Tryb interaktywny - użytkownik może wpisywać słowa."""
    console.print("\n[bold cyan]🎮 TRYB INTERAKTYWNY[/bold cyan]")
    console.print("Wpisz słowo, aby znaleźć podobne. Wpisz 'q' aby zakończyć.\n")

    while True:
        try:
            word = input("Słowo: ").strip()
            if word.lower() == 'q':
                break
            if not word:
                continue

            similar, tokens, missing = find_similar_words(word, tokenizer, model, topn=10)
            display_similar_words(word, similar, tokens, missing)

        except KeyboardInterrupt:
            break

    console.print("\n[dim]Koniec trybu interaktywnego.[/dim]")


def main():
    parser = argparse.ArgumentParser(
        description="Wnioskowanie z modelu CBOW - szukanie podobnych słów"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Ścieżka do pliku modelu (.model)"
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        choices=list(TOKENIZERS.keys()),
        help=f"Tokenizer użyty podczas treningu: {list(TOKENIZERS.keys())}"
    )

    parser.add_argument(
        "--words",
        type=str,
        default=None,
        help="Lista słów do testowania, oddzielonych przecinkami (np. 'król,książę,wojsko')"
    )

    parser.add_argument(
        "--analogy",
        type=str,
        default=None,
        help="Para słów do analogii, oddzielonych przecinkami (np. 'dziecko,kobieta')"
    )

    parser.add_argument(
        "--topn",
        type=int,
        default=10,
        help="Liczba najbardziej podobnych słów do wyświetlenia (domyślnie 10)"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Uruchom tryb interaktywny"
    )

    parser.add_argument(
        "--all-tests",
        action="store_true",
        help="Uruchom wszystkie domyślne testy (słowa + analogie)"
    )

    args = parser.parse_args()

    # Załaduj model i tokenizer
    console.print(f"\n[bold]📚 Ładowanie modelu:[/bold] {args.model}")
    model = load_model(args.model)
    console.print(f"[bold]🔤 Ładowanie tokenizera:[/bold] {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    console.print(f"\n[dim]Rozmiar słownika modelu: {len(model.wv)} tokenów[/dim]")
    console.print(f"[dim]Wymiar wektora: {model.wv.vector_size}[/dim]\n")

    # Tryb interaktywny
    if args.interactive:
        interactive_mode(tokenizer, model)
        return

    # Określ słowa do testowania
    if args.words:
        test_words = [w.strip() for w in args.words.split(',')]
    elif args.all_tests:
        test_words = DEFAULT_TEST_WORDS
    else:
        test_words = DEFAULT_TEST_WORDS[:4]  # Domyślnie 4 słowa

    # Testuj podobieństwo słów
    console.print(Panel("🔍 TESTOWANIE PODOBIEŃSTWA SŁÓW", style="bold cyan"))

    for word in test_words:
        similar, tokens, missing = find_similar_words(word, tokenizer, model, topn=args.topn)
        display_similar_words(word, similar, tokens, missing)

    # Testuj analogie
    if args.analogy:
        analogy_words = [w.strip() for w in args.analogy.split(',')]
        console.print(Panel("🧮 TESTOWANIE ANALOGII", style="bold cyan"))
        similar, tokens = find_analogy(analogy_words, tokenizer, model, topn=args.topn)
        display_analogy(analogy_words, similar, tokens)
    elif args.all_tests:
        console.print(Panel("🧮 TESTOWANIE ANALOGII", style="bold cyan"))
        for analogy in DEFAULT_ANALOGIES:
            similar, tokens = find_analogy(list(analogy), tokenizer, model, topn=args.topn)
            display_analogy(list(analogy), similar, tokens)


if __name__ == "__main__":
    main()
