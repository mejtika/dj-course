#!/bin/bash
# Skrypt do budowania wszystkich własnych tokenizerów

echo "═══════════════════════════════════════════════════════"
echo "  BUDOWANIE WŁASNYCH TOKENIZERÓW BPE"
echo "═══════════════════════════════════════════════════════"
echo ""

# 1. Pan Tadeusz
echo "📚 [1/4] Budowanie tokenizera: Pan Tadeusz..."
python tokenizer-build.py \
    --corpus PAN_TADEUSZ \
    --output tokenizers/tokenizer-pan-tadeusz.json \
    --vocab-size 32000

echo ""

# 2. Wolne Lektury
echo "📚 [2/4] Budowanie tokenizera: Wolne Lektury..."
python tokenizer-build.py \
    --corpus WOLNELEKTURY \
    --output tokenizers/tokenizer-wolnelektury.json \
    --vocab-size 32000

echo ""

# 3. NKJP
echo "📚 [3/4] Budowanie tokenizera: NKJP..."
python tokenizer-build.py \
    --corpus NKJP \
    --output tokenizers/tokenizer-nkjp.json \
    --vocab-size 32000

echo ""

# 4. Wszystkie korpusy
echo "📚 [4/4] Budowanie tokenizera: Wszystkie korpusy..."
python tokenizer-build.py \
    --corpus ALL \
    --output tokenizers/tokenizer-all-corpora.json \
    --vocab-size 32000

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  GOTOWE! Wszystkie tokenizery zostały zbudowane."
echo "═══════════════════════════════════════════════════════"
