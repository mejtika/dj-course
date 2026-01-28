"""
CBOW Comparison Script - Porównanie różnych konfiguracji modeli CBOW
Automatycznie trenuje modele z różnymi ustawieniami i porównuje wyniki.

Użycie:
    python cbow-compare.py                    # Uruchom pełne porównanie
    python cbow-compare.py --quick            # Szybki test (mniej kombinacji)
    python cbow-compare.py --tokenizers-only  # Porównaj tylko tokenizery
"""

import argparse
import numpy as np
import json
import os
import time
from gensim.models import Word2Vec
from tokenizers import Tokenizer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from corpora import CORPORA_FILES

console = Console()

# Dostępne tokenizery
TOKENIZERS = {
    "bielik-v1": "../tokenizer/tokenizers/bielik-v1-tokenizer.json",
    "bielik-v3": "../tokenizer/tokenizers/bielik-v3-tokenizer.json",
    "all-corpora": "../tokenizer/tokenizers/tokenizer-all-corpora.json",
    "wolnelektury": "../tokenizer/tokenizers/tokenizer-wolnelektury.json",
    "nkjp": "../tokenizer/tokenizers/tokenizer-nkjp.json",
}

# Słowa referencyjne do testowania jakości embeddingów
REFERENCE_PAIRS = [
    ("król", "książę"),      # Powinno być ~0.7+
    ("kobieta", "dziecko"),  # Powinno być ~0.6+
    ("wojsko", "armia"),     # Powinno być ~0.6+
    ("szlachta", "rycerz"),  # Powinno być ~0.5+
]


def load_tokenizer(tokenizer_name: str) -> Tokenizer:
    """Ładuje tokenizer."""
    path = TOKENIZERS.get(tokenizer_name)
    if not path or not os.path.exists(path):
        return None
    return Tokenizer.from_file(path)


def aggregate_raw_sentences(files: list) -> list[str]:
    """Wczytuje zdania z plików."""
    raw_sentences = []
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                raw_sentences.extend(lines)
        except FileNotFoundError:
            continue
    return raw_sentences


def tokenize_sentences(tokenizer: Tokenizer, sentences: list[str]) -> list[list[str]]:
    """Tokenizuje zdania."""
    encodings = tokenizer.encode_batch(sentences)
    return [encoding.tokens for encoding in encodings]


def train_model(tokenized_sentences, vector_size, window, min_count, epochs, sample, sg=0):
    """Trenuje model Word2Vec."""
    return Word2Vec(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=sg,
        epochs=epochs,
        sample=sample,
    )


def get_word_vector(word: str, tokenizer: Tokenizer, model: Word2Vec):
    """Oblicza wektor dla słowa."""
    encoding = tokenizer.encode(" " + word + " ")
    word_tokens = [t.strip() for t in encoding.tokens if t.strip()]
    special_tokens = ['[CLS]', '[SEP]', '<s>', '</s>', '[PAD]', '[MASK]', '[UNK]', '▁']
    # Filtruj również tokeny hex typu <0x...>
    word_tokens = [t for t in word_tokens if t not in special_tokens and not t.startswith('<0x')]

    valid_vectors = []
    for token in word_tokens:
        if token in model.wv:
            valid_vectors.append(model.wv[token])

    if not valid_vectors:
        return None

    return np.mean(valid_vectors, axis=0)


def compute_similarity(word1: str, word2: str, tokenizer: Tokenizer, model: Word2Vec) -> float:
    """Oblicza podobieństwo między dwoma słowami."""
    vec1 = get_word_vector(word1, tokenizer, model)
    vec2 = get_word_vector(word2, tokenizer, model)

    if vec1 is None or vec2 is None:
        return -1.0  # Nie można obliczyć

    # Cosine similarity
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return float(similarity)


def evaluate_model(tokenizer: Tokenizer, model: Word2Vec) -> dict:
    """Ewaluuje model na parach referencyjnych."""
    results = {}
    total_score = 0.0
    valid_pairs = 0

    for word1, word2 in REFERENCE_PAIRS:
        sim = compute_similarity(word1, word2, tokenizer, model)
        results[f"{word1}-{word2}"] = sim
        if sim >= 0:
            total_score += sim
            valid_pairs += 1

    results["avg_score"] = total_score / valid_pairs if valid_pairs > 0 else 0.0
    return results


def run_experiment(config: dict, corpus_sentences: dict) -> dict:
    """Uruchamia pojedynczy eksperyment z daną konfiguracją."""
    tokenizer = load_tokenizer(config["tokenizer"])
    if tokenizer is None:
        return {"error": f"Tokenizer {config['tokenizer']} nie znaleziony"}

    # Pobierz zdania dla korpusu
    sentences = corpus_sentences.get(config["corpus"], [])
    if not sentences:
        return {"error": f"Korpus {config['corpus']} pusty"}

    # Tokenizuj
    tokenized = tokenize_sentences(tokenizer, sentences)

    # Trenuj
    start_time = time.time()
    model = train_model(
        tokenized,
        vector_size=config["vector_size"],
        window=config["window"],
        min_count=config["min_count"],
        epochs=config["epochs"],
        sample=config["sample"],
    )
    train_time = time.time() - start_time

    # Ewaluuj
    eval_results = evaluate_model(tokenizer, model)

    return {
        "config": config,
        "train_time": train_time,
        "vocab_size": len(model.wv),
        **eval_results
    }


def display_results(results: list[dict]):
    """Wyświetla wyniki w formie tabeli."""
    # Sortuj po średnim wyniku (malejąco)
    results = sorted(results, key=lambda x: x.get("avg_score", 0), reverse=True)

    table = Table(title="📊 PORÓWNANIE MODELI CBOW")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Tokenizer", style="cyan")
    table.add_column("Korpus", style="blue")
    table.add_column("Vec", justify="right")
    table.add_column("Win", justify="right")
    table.add_column("Epochs", justify="right")
    table.add_column("król-książę", justify="right")
    table.add_column("Avg Score", justify="right", style="bold")
    table.add_column("Time", justify="right", style="dim")

    for i, result in enumerate(results, 1):
        if "error" in result:
            table.add_row(
                str(i), result["config"]["tokenizer"], result["config"]["corpus"],
                "-", "-", "-", "-", f"[red]{result['error'][:20]}[/red]", "-"
            )
            continue

        config = result["config"]
        krol_ksiaze = result.get("król-książę", -1)
        avg_score = result.get("avg_score", 0)

        # Kolorowanie wyników
        if krol_ksiaze >= 0.7:
            kk_style = "[bold green]"
        elif krol_ksiaze >= 0.5:
            kk_style = "[yellow]"
        elif krol_ksiaze >= 0:
            kk_style = "[dim]"
        else:
            kk_style = "[red]"

        if avg_score >= 0.6:
            avg_style = "[bold green]"
        elif avg_score >= 0.4:
            avg_style = "[yellow]"
        else:
            avg_style = "[dim]"

        table.add_row(
            str(i),
            config["tokenizer"],
            config["corpus"],
            str(config["vector_size"]),
            str(config["window"]),
            str(config["epochs"]),
            f"{kk_style}{krol_ksiaze:.4f}[/]" if krol_ksiaze >= 0 else "[red]N/A[/red]",
            f"{avg_style}{avg_score:.4f}[/]",
            f"{result['train_time']:.1f}s"
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Porównanie konfiguracji CBOW")
    parser.add_argument("--quick", action="store_true", help="Szybki test")
    parser.add_argument("--tokenizers-only", action="store_true", help="Porównaj tylko tokenizery")
    parser.add_argument("--save-results", type=str, help="Zapisz wyniki do pliku JSON")
    args = parser.parse_args()

    console.print(Panel("🚀 CBOW COMPARISON - Zadanie 4.1", style="bold cyan"))

    # Wczytaj korpusy raz
    console.print("\n[dim]Wczytywanie korpusów...[/dim]")
    corpus_sentences = {}
    for corpus_name in ["ALL", "WOLNELEKTURY"]:
        files = [str(f) for f in CORPORA_FILES.get(corpus_name, [])]
        corpus_sentences[corpus_name] = aggregate_raw_sentences(files)
        console.print(f"  {corpus_name}: {len(corpus_sentences[corpus_name])} zdań")

    # Definiuj konfiguracje do testowania
    if args.quick:
        configs = [
            {"tokenizer": "bielik-v3", "corpus": "ALL", "vector_size": 100, "window": 5, "min_count": 2, "epochs": 20, "sample": 1e-3},
            {"tokenizer": "all-corpora", "corpus": "ALL", "vector_size": 100, "window": 5, "min_count": 2, "epochs": 20, "sample": 1e-3},
        ]
    elif args.tokenizers_only:
        base_config = {"corpus": "ALL", "vector_size": 100, "window": 5, "min_count": 2, "epochs": 20, "sample": 1e-3}
        configs = [
            {**base_config, "tokenizer": "bielik-v1"},
            {**base_config, "tokenizer": "bielik-v3"},
            {**base_config, "tokenizer": "all-corpora"},
            {**base_config, "tokenizer": "wolnelektury"},
            {**base_config, "tokenizer": "nkjp"},
        ]
    else:
        # Pełne porównanie
        configs = []

        # Różne tokenizery
        for tokenizer in ["bielik-v1", "bielik-v3", "all-corpora"]:
            configs.append({
                "tokenizer": tokenizer, "corpus": "ALL",
                "vector_size": 100, "window": 5, "min_count": 2, "epochs": 20, "sample": 1e-3
            })

        # Różne vector_size
        for vs in [50, 100, 200]:
            configs.append({
                "tokenizer": "bielik-v3", "corpus": "ALL",
                "vector_size": vs, "window": 5, "min_count": 2, "epochs": 20, "sample": 1e-3
            })

        # Różne window
        for win in [3, 5, 8]:
            configs.append({
                "tokenizer": "bielik-v3", "corpus": "ALL",
                "vector_size": 100, "window": win, "min_count": 2, "epochs": 20, "sample": 1e-3
            })

        # Różne epochs
        for ep in [10, 30, 50]:
            configs.append({
                "tokenizer": "bielik-v3", "corpus": "ALL",
                "vector_size": 100, "window": 5, "min_count": 2, "epochs": ep, "sample": 1e-3
            })

    # Usuń duplikaty
    unique_configs = []
    seen = set()
    for c in configs:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)
    configs = unique_configs

    console.print(f"\n[bold]Liczba konfiguracji do przetestowania: {len(configs)}[/bold]\n")

    # Uruchom eksperymenty
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Trenowanie modeli...", total=len(configs))

        for config in configs:
            progress.update(task, description=f"[cyan]{config['tokenizer']}[/] + [blue]{config['corpus']}[/]")
            result = run_experiment(config, corpus_sentences)
            results.append(result)
            progress.advance(task)

    # Wyświetl wyniki
    console.print("\n")
    display_results(results)

    # Zapisz wyniki
    if args.save_results:
        with open(args.save_results, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        console.print(f"\n[dim]Wyniki zapisane do: {args.save_results}[/dim]")

    # Podsumowanie
    best = max(results, key=lambda x: x.get("avg_score", 0))
    if "error" not in best:
        console.print(f"\n[bold green]🏆 Najlepszy model:[/bold green]")
        console.print(f"   Tokenizer: {best['config']['tokenizer']}")
        console.print(f"   Korpus: {best['config']['corpus']}")
        console.print(f"   Vector size: {best['config']['vector_size']}")
        console.print(f"   Window: {best['config']['window']}")
        console.print(f"   Epochs: {best['config']['epochs']}")
        console.print(f"   Średni wynik: {best['avg_score']:.4f}")
        console.print(f"   król-książę: {best.get('król-książę', 'N/A')}")


if __name__ == "__main__":
    main()
