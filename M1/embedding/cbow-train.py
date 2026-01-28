"""
CBOW Training Script - Trening modelu embeddingowego Word2Vec (CBOW)
Rozdzielony skrypt do osobnego trenowania modelu.

Użycie:
    python cbow-train.py --corpus ALL --tokenizer bielik-v3 --epochs 20 --vector-size 100
"""

import argparse
import numpy as np
import json
import logging
from gensim.models import Word2Vec
from tokenizers import Tokenizer
import os
import time
from corpora import CORPORA_FILES

# Ustawienie logowania
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Dostępne tokenizery
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


def load_tokenizer(tokenizer_name: str) -> Tokenizer:
    """Ładuje tokenizer na podstawie nazwy."""
    if tokenizer_name not in TOKENIZERS:
        raise ValueError(f"Nieznany tokenizer: {tokenizer_name}. Dostępne: {list(TOKENIZERS.keys())}")

    path = TOKENIZERS[tokenizer_name]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono pliku tokenizera: {path}")

    print(f"📚 Ładowanie tokenizera: {tokenizer_name} ({path})")
    return Tokenizer.from_file(path)


def aggregate_raw_sentences(files: list) -> list[str]:
    """Wczytuje i agreguje zdania z plików."""
    raw_sentences = []
    print(f"📂 Wczytywanie tekstu z {len(files)} plików...")

    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                raw_sentences.extend(lines)
        except FileNotFoundError:
            print(f"⚠️  Nie znaleziono pliku: {file}")
            continue

    if not raw_sentences:
        raise ValueError("Pliki wejściowe są puste lub nie zostały wczytane.")

    print(f"✅ Wczytano {len(raw_sentences)} zdań")
    return raw_sentences


def tokenize_sentences(tokenizer: Tokenizer, sentences: list[str]) -> list[list[str]]:
    """Tokenizuje zdania używając tokenizera BPE."""
    print(f"🔤 Tokenizacja {len(sentences)} zdań...")
    encodings = tokenizer.encode_batch(sentences)
    tokenized = [encoding.tokens for encoding in encodings]
    print(f"✅ Przygotowano {len(tokenized)} sekwencji")
    return tokenized


def train_cbow(
    tokenized_sentences: list[list[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    workers: int = 4,
    epochs: int = 20,
    sample: float = 1e-3,
    sg: int = 0  # 0 = CBOW, 1 = Skip-gram
) -> Word2Vec:
    """Trenuje model Word2Vec (CBOW)."""

    print(f"\n{'='*60}")
    print("🚀 TRENING WORD2VEC (CBOW)")
    print(f"{'='*60}")
    print(f"  Parametry:")
    print(f"    - vector_size: {vector_size}")
    print(f"    - window: {window}")
    print(f"    - min_count: {min_count}")
    print(f"    - epochs: {epochs}")
    print(f"    - sample: {sample}")
    print(f"    - mode: {'CBOW' if sg == 0 else 'Skip-gram'}")
    print(f"{'='*60}\n")

    start_time = time.time()

    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs,
        sample=sample,
    )

    elapsed = time.time() - start_time
    print(f"\n✅ Trening zakończony w {elapsed:.2f}s")
    print(f"📊 Rozmiar słownika: {len(model.wv)} tokenów")

    return model


def save_model(model: Word2Vec, output_dir: str, model_name: str, tokenizer_name: str):
    """Zapisuje model i powiązane pliki."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Zapisz model gensim
    model_path = os.path.join(output_dir, f"{model_name}.model")
    model.save(model_path)
    print(f"💾 Model zapisany: {model_path}")

    # 2. Zapisz tensor NumPy
    tensor_path = os.path.join(output_dir, f"{model_name}_tensor.npy")
    np.save(tensor_path, model.wv.vectors)
    print(f"💾 Tensor zapisany: {tensor_path}")

    # 3. Zapisz mapowanie tokenów
    map_path = os.path.join(output_dir, f"{model_name}_token_map.json")
    token_to_index = {token: model.wv.get_index(token) for token in model.wv.index_to_key}
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(token_to_index, f, ensure_ascii=False, indent=2)
    print(f"💾 Mapa tokenów zapisana: {map_path}")

    # 4. Zapisz konfigurację
    config_path = os.path.join(output_dir, f"{model_name}_config.json")
    config = {
        "tokenizer": tokenizer_name,
        "vector_size": model.wv.vector_size,
        "window": model.window,
        "min_count": model.min_count,
        "epochs": model.epochs,
        "vocab_size": len(model.wv),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"💾 Konfiguracja zapisana: {config_path}")

    return model_path


def main():
    parser = argparse.ArgumentParser(
        description="Trening modelu CBOW (Word2Vec) na polskich korpusach"
    )

    parser.add_argument(
        "--corpus",
        type=str,
        default="ALL",
        choices=list(CORPORA_FILES.keys()),
        help=f"Korpus treningowy: {list(CORPORA_FILES.keys())}"
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bielik-v3",
        choices=list(TOKENIZERS.keys()),
        help=f"Tokenizer do użycia: {list(TOKENIZERS.keys())}"
    )

    parser.add_argument(
        "--vector-size",
        type=int,
        default=100,
        help="Wymiar wektora embeddingu (domyślnie 100)"
    )

    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Rozmiar okna kontekstowego (domyślnie 5)"
    )

    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimalna liczba wystąpień tokenu (domyślnie 2)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Liczba epok treningu (domyślnie 20)"
    )

    parser.add_argument(
        "--sample",
        type=float,
        default=1e-3,
        help="Próg subsamplingu częstych słów (domyślnie 1e-3)"
    )

    parser.add_argument(
        "--sg",
        type=int,
        default=0,
        choices=[0, 1],
        help="Tryb: 0=CBOW, 1=Skip-gram (domyślnie 0)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Katalog wyjściowy dla modeli (domyślnie 'models')"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Nazwa modelu (domyślnie: cbow_{corpus}_{tokenizer})"
    )

    args = parser.parse_args()

    # Nazwa modelu
    if args.model_name is None:
        mode = "cbow" if args.sg == 0 else "skipgram"
        args.model_name = f"{mode}_{args.corpus.lower()}_{args.tokenizer}"

    print(f"\n{'='*60}")
    print("🎯 CBOW TRAINING - Zadanie 4.1")
    print(f"{'='*60}")
    print(f"  Korpus: {args.corpus}")
    print(f"  Tokenizer: {args.tokenizer}")
    print(f"  Model: {args.model_name}")
    print(f"{'='*60}\n")

    # 1. Załaduj tokenizer
    tokenizer = load_tokenizer(args.tokenizer)

    # 2. Wczytaj korpus
    files = [str(f) for f in CORPORA_FILES[args.corpus]]
    raw_sentences = aggregate_raw_sentences(files)

    # 3. Tokenizuj
    tokenized_sentences = tokenize_sentences(tokenizer, raw_sentences)

    # 4. Trenuj model
    model = train_cbow(
        tokenized_sentences,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        sample=args.sample,
        sg=args.sg,
    )

    # 5. Zapisz model
    model_path = save_model(model, args.output_dir, args.model_name, args.tokenizer)

    print(f"\n{'='*60}")
    print("✅ TRENING ZAKOŃCZONY")
    print(f"{'='*60}")
    print(f"  Model zapisany w: {model_path}")
    print(f"  Użyj: python cbow-infer.py --model {model_path} --tokenizer {args.tokenizer}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
