"""
Skrypt porównawczy tokenizerów z wizualizacją w terminalu.
Porównuje efektywność tokenizacji różnych tekstów przez różne tokenizery.
"""

from pathlib import Path
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from typing import Optional

console = Console()

# Ścieżki do tokenizerów
TOKENIZERS_DIR = Path("tokenizers")

# Tokenizery BPE (Bielik + własne)
BPE_TOKENIZERS = {
    "bielik-v1": TOKENIZERS_DIR / "bielik-v1-tokenizer.json",
    "bielik-v2": TOKENIZERS_DIR / "bielik-v2-tokenizer.json",
    "bielik-v3": TOKENIZERS_DIR / "bielik-v3-tokenizer.json",
    "pan-tadeusz": TOKENIZERS_DIR / "tokenizer-pan-tadeusz.json",
    "wolnelektury": TOKENIZERS_DIR / "tokenizer-wolnelektury.json",
    "nkjp": TOKENIZERS_DIR / "tokenizer-nkjp.json",
    "all-corpora": TOKENIZERS_DIR / "tokenizer-all-corpora.json",
}

# Tokenizer Herbert (WordPiece) - wymaga innego ładowania
HERBERT_DIR = TOKENIZERS_DIR / "herbert"

# Teksty testowe
TEST_TEXTS = {
    "Pan Tadeusz Księga 1": Path("../korpus-wolnelektury/pan-tadeusz-ksiega-1.txt"),
    "Pickwick Papers": Path("../korpus-mini/the-pickwick-papers-gutenberg.txt"),
    "Fryderyk Chopin": Path("../korpus-mini/fryderyk-chopin-wikipedia.txt"),
}


def load_bpe_tokenizer(path: Path) -> Optional[Tokenizer]:
    """Ładuje tokenizer BPE z pliku JSON."""
    if not path.exists():
        return None
    try:
        return Tokenizer.from_file(str(path))
    except Exception as e:
        console.print(f"[red]Błąd ładowania {path}: {e}[/red]")
        return None


def load_herbert_tokenizer(path: Path):
    """Ładuje tokenizer Herbert (WordPiece) z folderu."""
    if not path.exists():
        return None
    try:
        return AutoTokenizer.from_pretrained(str(path))
    except Exception as e:
        console.print(f"[red]Błąd ładowania Herbert: {e}[/red]")
        return None


def count_tokens_bpe(tokenizer: Tokenizer, text: str) -> int:
    """Zlicza tokeny dla tokenizera BPE."""
    encoded = tokenizer.encode(text)
    return len(encoded.ids)


def count_tokens_herbert(tokenizer, text: str) -> int:
    """Zlicza tokeny dla tokenizera Herbert."""
    encoded = tokenizer.encode(text)
    return len(encoded)


def load_text(path: Path) -> str:
    """Wczytuje tekst z pliku."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def visualize_results(text_name: str, results: dict[str, int]):
    """
    Wizualizuje wyniki tokenizacji jako poziome słupki w terminalu.
    Sortuje od najmniejszej do największej liczby tokenów.
    """
    # Sortuj wyniki od najmniejszej do największej liczby tokenów
    sorted_results = sorted(results.items(), key=lambda x: x[1])

    # Znajdź maksymalną wartość dla skalowania słupków
    max_tokens = max(results.values())
    max_name_len = max(len(name) for name in results.keys())

    # Szerokość słupka (dostosuj do szerokości terminalu)
    bar_width = 50

    # Przygotuj zawartość panelu
    lines = []

    for i, (name, tokens) in enumerate(sorted_results):
        # Ikona medalu dla top 3
        if i == 0:
            medal = "🥇"
        elif i == 1:
            medal = "🥈"
        elif i == 2:
            medal = "🥉"
        else:
            medal = "  "

        # Oblicz długość słupka
        bar_length = int((tokens / max_tokens) * bar_width)
        bar = "▓" * bar_length + "░" * (bar_width - bar_length)

        # Formatuj liczbę tokenów z separatorem tysięcy
        tokens_formatted = f"{tokens:,}".replace(",", ",")

        # Wyrównaj nazwę tokenizera
        name_padded = name.ljust(max_name_len)

        lines.append(f"{medal} {name_padded} {bar} {tokens_formatted:>10} tokenów")

    content = "\n".join(lines)

    # Wyświetl panel
    panel = Panel(
        content,
        title=f"📊 {text_name}",
        border_style="cyan"
    )
    console.print(panel)
    console.print()


def main():
    console.print("\n[bold cyan]═══ PORÓWNANIE TOKENIZERÓW ═══[/bold cyan]\n")

    # Załaduj wszystkie tokenizery BPE
    tokenizers = {}
    for name, path in BPE_TOKENIZERS.items():
        tokenizer = load_bpe_tokenizer(path)
        if tokenizer:
            tokenizers[name] = ("bpe", tokenizer)
            console.print(f"[green]✓[/green] Załadowano: {name}")
        else:
            console.print(f"[yellow]⚠[/yellow] Brak tokenizera: {name}")

    # Załaduj tokenizer Herbert
    herbert = load_herbert_tokenizer(HERBERT_DIR)
    if herbert:
        tokenizers["herbert"] = ("herbert", herbert)
        console.print(f"[green]✓[/green] Załadowano: herbert")
    else:
        console.print(f"[yellow]⚠[/yellow] Brak tokenizera: herbert (uruchom download-herbert.py)")

    console.print()

    # Dla każdego tekstu testowego
    for text_name, text_path in TEST_TEXTS.items():
        if not text_path.exists():
            console.print(f"[red]✗[/red] Brak pliku: {text_path}")
            continue

        text = load_text(text_path)
        console.print(f"[dim]Przetwarzanie: {text_name} ({len(text):,} znaków)[/dim]")

        # Zlicz tokeny dla każdego tokenizera
        results = {}
        for name, (tokenizer_type, tokenizer) in tokenizers.items():
            if tokenizer_type == "bpe":
                token_count = count_tokens_bpe(tokenizer, text)
            else:  # herbert
                token_count = count_tokens_herbert(tokenizer, text)
            results[name] = token_count

        # Wizualizuj wyniki
        visualize_results(text_name, results)

    # Podsumowanie
    console.print("[bold cyan]═══ PODSUMOWANIE ═══[/bold cyan]\n")
    console.print("🎯 [bold]Najefektywniejszy tokenizer[/bold] to ten, który produkuje [green]najmniej tokenów[/green].")
    console.print("   Własne tokenizery powinny być najlepsze dla tekstów z ich korpusu treningowego.")
    console.print("   Bielik v3 powinien być lepszy od v1/v2 dla polskiego tekstu.\n")


if __name__ == "__main__":
    main()
