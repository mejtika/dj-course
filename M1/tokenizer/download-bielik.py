"""
Skrypt do pobierania tokenizerów Bielik v1, v2 i v3 z HuggingFace.
Wymaga zaakceptowania terms of use na https://bielik.ai/terms/
"""

import os
import urllib.request

# URLs tokenizerów Bielik
TOKENIZERS = {
    "bielik-v1": "https://huggingface.co/speakleash/Bielik-7B-Instruct-v0.1/raw/main/tokenizer.json",
    "bielik-v2": "https://huggingface.co/speakleash/Bielik-11B-v2.5-Instruct/raw/main/tokenizer.json",
    "bielik-v3": "https://huggingface.co/speakleash/Bielik-4.5B-v3.0-Instruct/raw/main/tokenizer.json",
}

OUTPUT_DIR = "tokenizers"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, url in TOKENIZERS.items():
        output_path = os.path.join(OUTPUT_DIR, f"{name}-tokenizer.json")

        if os.path.exists(output_path):
            print(f"✓ {name} już istnieje: {output_path}")
            continue

        print(f"Pobieranie {name} z {url}...")
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"✓ Zapisano: {output_path}")
        except Exception as e:
            print(f"✗ Błąd pobierania {name}: {e}")
            print(f"  Może być wymagana akceptacja terms of use: https://bielik.ai/terms/")


if __name__ == "__main__":
    main()
