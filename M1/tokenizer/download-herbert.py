"""
Skrypt do pobierania tokenizera allegro/herbert-base-cased z HuggingFace.
Herbert używa WordPiece (nie BPE), ale porównanie efektywności tokenizacji jest sensowne.
"""

from transformers import AutoTokenizer
import os

OUTPUT_DIR = "tokenizers/herbert"

def main():
    print("Pobieranie tokenizera allegro/herbert-base-cased...")

    # Pobierz tokenizer z HuggingFace
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")

    # Utwórz folder jeśli nie istnieje
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Zapisz tokenizer
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Herbert tokenizer zapisany do: {OUTPUT_DIR}/")

    # Test tokenizera
    test_texts = [
        "Litwo! Ojczyzno moja! ty jesteś jak zdrowie.",
        "Jakże mi wesoło!",
    ]

    print("\n--- Test tokenizera Herbert ---")
    for txt in test_texts:
        encoded = tokenizer.encode(txt)
        tokens = tokenizer.convert_ids_to_tokens(encoded)
        print(f"Tekst: {txt}")
        print(f"Tokeny: {tokens}")
        print(f"IDs: {encoded}")
        print()


if __name__ == "__main__":
    main()
