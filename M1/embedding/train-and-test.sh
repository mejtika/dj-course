#!/bin/bash
# Skrypt do automatycznego trenowania i testowania modeli CBOW

echo "═══════════════════════════════════════════════════════════════"
echo "  CBOW TRAINING & TESTING - Zadanie 4.1"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Utwórz folder na modele
mkdir -p models

# 1. Trenuj model z bielik-v3 (domyślny)
echo "📚 [1/3] Trening modelu: bielik-v3 + ALL corpus"
python cbow-train.py \
    --corpus ALL \
    --tokenizer bielik-v3 \
    --vector-size 100 \
    --window 5 \
    --epochs 30 \
    --output-dir models

echo ""

# 2. Trenuj model z własnym tokenizerem (all-corpora)
echo "📚 [2/3] Trening modelu: all-corpora tokenizer + ALL corpus"
python cbow-train.py \
    --corpus ALL \
    --tokenizer all-corpora \
    --vector-size 100 \
    --window 5 \
    --epochs 30 \
    --output-dir models

echo ""

# 3. Trenuj model z bielik-v1 (dla porównania)
echo "📚 [3/3] Trening modelu: bielik-v1 + ALL corpus"
python cbow-train.py \
    --corpus ALL \
    --tokenizer bielik-v1 \
    --vector-size 100 \
    --window 5 \
    --epochs 30 \
    --output-dir models

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  TESTOWANIE MODELI"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Testuj każdy model
for model in models/*.model; do
    if [ -f "$model" ]; then
        # Wyciągnij nazwę tokenizera z nazwy pliku
        filename=$(basename "$model" .model)
        tokenizer=$(echo "$filename" | sed 's/cbow_all_//')

        echo "🔍 Testowanie: $filename"
        python cbow-infer.py \
            --model "$model" \
            --tokenizer "$tokenizer" \
            --all-tests
        echo ""
    fi
done

echo "═══════════════════════════════════════════════════════════════"
echo "  GOTOWE!"
echo "═══════════════════════════════════════════════════════════════"
