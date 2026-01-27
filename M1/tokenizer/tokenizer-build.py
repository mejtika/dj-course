import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from corpora import CORPORA_FILES

def build_tokenizer(files: list[str], output_path: str, vocab_size: int = 32000):
    """
    Buduje tokenizer BPE na podstawie podanych plików i zapisuje go do output_path.
    """
    # 1. Initialize the Tokenizer (BPE model)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # 2. Set the pre-tokenizer (e.g., split on spaces)
    tokenizer.pre_tokenizer = Whitespace()

    # 3. Set the Trainer
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        vocab_size=vocab_size,
        min_frequency=2
    )

    print(f"Trening tokenizera na {len(files)} plikach...")
    print(f"Rozmiar słownika: {vocab_size}")

    # 4. Train the Tokenizer
    tokenizer.train(files, trainer=trainer)

    # 5. Save the vocabulary and tokenization rules
    tokenizer.save(output_path)
    print(f"Tokenizer zapisany do: {output_path}")

    return tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Budowanie tokenizera BPE na podstawie korpusu tekstowego"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        choices=list(CORPORA_FILES.keys()),
        help=f"Nazwa korpusu: {list(CORPORA_FILES.keys())}"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Ścieżka wyjściowa dla tokenizera (np. tokenizers/tokenizer-pan-tadeusz.json)"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Rozmiar słownika (domyślnie 32000)"
    )

    args = parser.parse_args()

    # Pobierz listę plików dla wybranego korpusu
    files = [str(f) for f in CORPORA_FILES[args.corpus]]

    if not files:
        print(f"Błąd: Korpus '{args.corpus}' nie zawiera plików!")
        return

    print(f"Korpus: {args.corpus}")
    print(f"Liczba plików: {len(files)}")

    # Buduj tokenizer
    tokenizer = build_tokenizer(files, args.output, args.vocab_size)

    # Test tokenizera
    test_texts = [
        "Litwo! Ojczyzno moja! ty jesteś jak zdrowie.",
        "Jakże mi wesoło!",
    ]

    print("\n--- Test tokenizera ---")
    for txt in test_texts:
        encoded = tokenizer.encode(txt)
        print(f"Tekst: {txt}")
        print(f"Tokeny: {encoded.tokens}")
        print(f"IDs: {encoded.ids}")
        print()


if __name__ == "__main__":
    main()

