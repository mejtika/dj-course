# Zadanie 10 — Dotrenowanie sieci circle-in-square do 100% accuracy

## Opis problemu

Sieć ma rozwiązać problem **klasyfikacji binarnej**: czy punkt $(x, y)$ leży wewnątrz okręgu o promieniu $0.5$ (etykieta **1**) czy poza nim (etykieta **0**), gdy punkt znajduje się w kwadracie $[-1, 1] \times [-1, 1]$.

Kod w `circle-in-square-network.py` jest praktycznie gotowy, ale ma **celowo zaniżone** 4 parametry oznaczone 🔥. Naszym celem jest je poprawić, aby sieć osiągnęła **100% accuracy przy małym rozmiarze**.

---

## Słownik pojęć

| Pojęcie | Wyjaśnienie |
|---|---|
| **Sieć neuronowa (Neural Network)** | Program inspirowany działaniem mózgu — dane wejściowe przechodzą przez warstwy "neuronów" (mnożenie przez wagi + dodanie biasu), a na końcu wychodzi wynik. Sieć "uczy się" dobierając wagi tak, by wynik był poprawny. |
| **Klasyfikacja binarna** | Zadanie "odpowiedz 0 albo 1". Tutaj: punkt leży wewnątrz okręgu (1) lub nie (0). |
| **Warstwa (Layer)** | Grupa neuronów na jednym "piętrze" sieci. `nn.Linear(2, 8)` = warstwa z 2 wejściami i 8 neuronami. |
| **Warstwa ukryta (Hidden Layer)** | Warstwa między wejściem a wyjściem — tutaj sieć "myśli". Im więcej neuronów, tym bardziej złożone wzorce może wykryć. |
| **Neuron** | Pojedynczy "węzeł" — liczy sumę ważoną wejść, dodaje bias, przepuszcza przez funkcję aktywacji. |
| **Funkcja aktywacji (ReLU)** | Funkcja nieliniowa: $ReLU(x) = \max(0, x)$. Bez niej sieć byłaby zwykłym mnożeniem macierzy — nie umiałaby uczyć się krzywych/okręgów. |
| **Sigmoid** | Funkcja ściskająca wynik do zakresu $(0, 1)$: $\sigma(x) = \frac{1}{1+e^{-x}}$. Interpretujemy wynik jako prawdopodobieństwo. |
| **Logit** | Surowy wynik sieci PRZED zastosowaniem Sigmoida. `BCEWithLogitsLoss` łączy Sigmoid + loss w jednym kroku (numerycznie stabilniej). |
| **Loss (Strata)** | Miara "jak bardzo sieć się myli". Im mniejszy loss → tym lepsza sieć. `BCEWithLogitsLoss` = Binary Cross-Entropy. |
| **Epoka (Epoch)** | Jedno przejście przez CAŁY zbiór danych treningowych. 2000 epok = sieć zobaczyła dane 2000 razy. |
| **Learning Rate** | Jak duży krok robi optymalizator przy aktualizacji wag. Za mały → sieć uczy się wieki. Za duży → "przeskakuje" optimum. |
| **Optymalizator Adam** | Zaawansowany algorytm aktualizacji wag — adaptuje learning rate per parametr. Lepszy od prostego SGD dla większości zastosowań. |
| **Backward Pass (Backpropagation)** | Algorytm obliczający gradienty (pochodne) — mówi sieci "w którą stronę i o ile zmienić każdą wagę, by zmniejszyć loss". |
| **Gradient** | Kierunek i wielkość zmiany wagi — wskazuje jak szybko loss rośnie/maleje przy zmianie danej wagi. |
| **Próbki (Samples)** | Punkty treningowe. 10 punktów to ZA MAŁO, by sieć "zobaczyła" okrąg. Potrzeba setek/tysięcy. |
| **TensorBoard** | Narzędzie wizualizacyjne — wykresy loss, histogramy wag/gradientów. Pozwala zobaczyć CZY i JAK sieć się uczy. |
| **Accuracy** | Procent poprawnych odpowiedzi: $\frac{\text{poprawne predykcje}}{\text{wszystkie próbki}} \times 100\%$ |

---

## Co jest źle w obecnym kodzie i dlaczego

W `circle-in-square-network.py` są **4 celowo zaniżone wartości**:

### 1. Struktura sieci (linie 23–24)

```python
# OBECNIE:
self.fc1 = nn.Linear(2, 2)
self.fc2 = nn.Linear(2, 1)
```

Sieć ma tylko **2 neurony w jednej warstwie ukrytej**. Aby odwzorować okrąg (granicę nieliniową), potrzeba więcej neuronów. Wyobraź sobie, że każdy neuron z ReLU to jedna linia prosta — aby "narysować" okrąg, potrzebujesz wielu odcinków prostych.

### 2. Liczba próbek (linia 49)

```python
# OBECNIE:
NUM_SAMPLES = 10
```

Z 10 punktów sieć nie jest w stanie "zobaczyć" kształtu okręgu. To jakbyś miał 10 punktów na mapie i próbował odgadnąć kształt Polski.

### 3. Learning rate (linia 56)

```python
# OBECNIE:
LEARNING_RATE = 0.00001
```

Absurdalnie mały. Przy Adamie typowa wartość to `0.001`–`0.01`. Z tak małym LR sieć potrzebowałaby milionów epok.

### 4. Liczba epok (linia 57)

```python
# OBECNIE:
EPOCHS = 500
```

Za mało, biorąc pod uwagę resztę problemów. Przy poprawionych parametrach 1000–3000 powinno wystarczyć.

---

## Kroki do wykonania

### Krok 0: Przygotowanie środowiska

```bash
cd M1/neural-networks
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Uwaga**: TensorBoard zainstaluje się automatycznie z `requirements.txt` — nie wymaga osobnej instalacji.

### Krok 1: Powiększ strukturę sieci

Zmień klasę `CircleInSquareNet` (linie 21–29):

```python
# BYŁO:
class CircleInSquareNet(nn.Module):
    def __init__(self):
        super(CircleInSquareNet, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x
```

```python
# MA BYĆ:
class CircleInSquareNet(nn.Module):
    def __init__(self):
        super(CircleInSquareNet, self).__init__()
        self.fc1 = nn.Linear(2, 16)   # 2 wejścia → 16 neuronów
        self.fc2 = nn.Linear(16, 8)   # 16 → 8 neuronów
        self.fc3 = nn.Linear(8, 1)    # 8 → 1 wyjście

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x
```

**Dlaczego 16 → 8 → 1?**
- Granica decyzyjna to okrąg — krzywa nieliniowa
- ReLU tworzy odcinki proste, więc potrzebujesz wystarczająco wielu neuronów, by z odcinków "złożyć" okrąg
- 16 neuronów w pierwszej warstwie to wystarczająca ilość do aproksymacji
- Druga warstwa (8) pomaga lepiej kombinować wyuczone cechy
- Cel zadania mówi "przy małym rozmiarze" — 16+8 to dobry kompromis

**Alternatywy**:
| Struktura | Komentarz |
|---|---|
| 2 → 8 → 1 | Może zadziałać, ale wymaga więcej epok |
| 2 → 10 → 1 | Rozsądna opcja, minimalnie mniejsza |
| 2 → 32 → 16 → 1 | Overkill — za duża sieć jak na tak proste zadanie |

### Krok 2: Zwiększ liczbę próbek

```python
# BYŁO:
NUM_SAMPLES = 10

# MA BYĆ:
NUM_SAMPLES = 1000
```

**Dlaczego 1000?** Aby sieć "zobaczyła" okrąg — potrzebuje punktów gęsto rozmieszczonych w kwadracie $[-1, 1] \times [-1, 1]$. 1000 punktów daje wystarczającą gęstość.

### Krok 3: Zwiększ learning rate

```python
# BYŁO:
LEARNING_RATE = 0.00001

# MA BYĆ:
LEARNING_RATE = 0.01
```

**Dlaczego 0.01?** Optymalizator Adam adaptuje LR sam, ale startowa wartość powinna być rozsądna. Obecna wartość `0.00001` jest **1000× za mała**.

### Krok 4: Zwiększ liczbę epok

```python
# BYŁO:
EPOCHS = 500

# MA BYĆ:
EPOCHS = 2000
```

### Krok 5: Dostosuj logowanie TensorBoard

Po dodaniu warstwy `fc3` — dodaj logowanie gradientów i wag dla nowej warstwy.

### Krok 6: Uruchom i zweryfikuj

```bash
python circle-in-square-network.py
```

**Oczekiwany wynik**: `Dokładność na zbiorze treningowym: 100.00%` i `Good job! 🎉`

### Krok 7 (opcjonalny): Wizualizacja w TensorBoard

```bash
tensorboard --logdir=runs
```

Otwórz http://localhost:6006/

---

## Podsumowanie zmian

| Parametr | Było | Ma być |
|---|---|---|
| Struktura sieci | 2 → 2 → 1 | 2 → 16 → 8 → 1 |
| NUM_SAMPLES | 10 | 1000 |
| LEARNING_RATE | 0.00001 | 0.01 |
| EPOCHS | 500 | 2000 |

---

## Pomocne narzędzia do budowania intuicji

### TensorFlow Playground
https://playground.tensorflow.org

- Ustaw problem na **"Circle"** (ikona okręgu)
- Dodawaj/usuwaj neurony i warstwy
- Obserwuj jak sieć "rysuje" granicę decyzyjną w czasie rzeczywistym

### TensorBoard
- Zainstaluje się razem z `requirements.txt`
- Uruchom: `tensorboard --logdir=runs`
- Otwórz: http://localhost:6006/
- Sekcje: SCALARS (loss), HISTOGRAMS (gradienty i wagi)

---

## TensorBoard — czym jest, jak działa, jak czytać wykresy

### Czym jest TensorBoard?

TensorBoard to **narzędzie wizualizacyjne** stworzone przez Google do monitorowania procesu treningu sieci neuronowych. Choć powstało jako część TensorFlow, działa doskonale z PyTorch dzięki modułowi `torch.utils.tensorboard.SummaryWriter`.

Można o nim myśleć jak o **"dashboardzie" dla treningu** — zamiast wpatrywać się w liczby lecące przez terminal, dostajesz interaktywne wykresy, które natychmiast pokazują czy sieć uczy się dobrze, źle, czy wcale.

### Jak działa?

1. **Zapis danych** — w kodzie treningowym `SummaryWriter` zapisuje metryki (loss, wagi, gradienty) do plików logów w folderze `runs/`:
   ```python
   writer = SummaryWriter('runs/circle_in_square')
   writer.add_scalar('Loss', loss.item(), epoch)          # wartość skalarna
   writer.add_histogram('Weights/FC1', model.fc1.weight.data, epoch)  # rozkład
   ```
2. **Serwer HTTP** — komenda `tensorboard --logdir=runs` uruchamia lokalny serwer, który czyta logi i generuje interaktywne wykresy.
3. **Przeglądarka** — otwierasz http://localhost:6006/ i widzisz wyniki.

Dane zapisywane są **przyrostowo** — możesz uruchomić TensorBoard w trakcie treningu i obserwować postępy na żywo (odśwież stronę lub kliknij ikonę odświeżania).

### Zakładki TensorBoard

#### 1. SCALARS — "Czy sieć się uczy?"

To najważniejsza zakładka. Pokazuje **wykres wartości skalarnych w czasie** (oś X = epoki, oś Y = wartość).

**Wykres Loss (Strata)**:
- **Dobry trening**: loss **monotonnie spada** od dużej wartości do bliskiej zeru. W naszym przypadku: od ~0.17 do ~0.004.
- **Zły trening (za mały LR)**: loss praktycznie się nie zmienia — linia jest płaska.
- **Zły trening (za duży LR)**: loss skacze w górę i w dół chaotycznie (oscylacje).
- **Overfitting**: loss treningowy spada, ale loss walidacyjny (gdyby był) rośnie — sieć "uczy się na pamięć".

**Jak czytać**: Jeśli krzywa loss:
- Spada szybko na początku, potem zwalnia → **OK** (typowe)
- Jest płaska → learning rate za mały lub sieć za mała
- Skacze w górę/dół → learning rate za duży
- Spada do 0 → ideał

**Suwak "Smoothing"** (po prawej, np. 0.6): wygładza krzywą — przydatny gdy loss oscyluje. Wartość 0 = surowe dane, 1 = maksymalne wygładzenie.

#### 2. HISTOGRAMS — "Co dzieje się wewnątrz sieci?"

Pokazuje **rozkłady wartości w czasie** w formie 3D "gór". Oś X = wartości (np. wagi), oś Y = epoki (czas), oś Z (głębokość) = gęstość.

**Histogramy gradientów (Gradients/Layer_FCx_Weights)**:
- Pokazują rozkład gradientów (pochodnych) dla wag każdej warstwy.
- **Zdrowe gradienty**: rozkład skupiony wokół zera, ale NIE dokładnie na zerze — sieć aktywnie się uczy.
- **Vanishing gradients** (zanikające gradienty): gradienty skupione na dokładnie 0 → sieć przestaje się uczyć (problem głębokich sieci).
- **Exploding gradients** (eksplodujące gradienty): wartości gradientów rosną do ogromnych liczb → trening staje się niestabilny.
- **Stabilizacja w czasie**: w miarę jak sieć się uczy, gradienty powinny maleć (bo loss maleje, więc korekty są coraz mniejsze).

**Histogramy wag (Weights/Layer_FCx_Weights)**:
- Pokazują jak zmieniają się wagi (parametry) sieci w czasie treningu.
- **Na początku**: wagi są losowe (inicjalizacja), rozkład jest mały i chaotyczny.
- **W trakcie treningu**: wagi "rozchodzą się" — sieć kształtuje swoją wewnętrzną reprezentację.
- **Na końcu treningu**: rozkład powinien się ustabilizować (nie zmienia się drastycznie).
- Jeśli wagi rosną do ogromnych wartości → problem z treningiem.

**Tryby wyświetlania** (Settings → Histograms → Mode):
- **Offset** (domyślny): kolejne epoki narysowane jedna za drugą w 3D — widać "ewolucję" rozkładu w czasie.
- **Overlay**: wszystkie epoki nałożone na siebie na jednym wykresie 2D.

#### 3. DISTRIBUTIONS — alternatywny widok histogramów

Ta sama informacja co HISTOGRAMS, ale w postaci **wstęg percentylowych** (jak "wąsy" na wykresie). Pokazuje medianę, 1-szy i 3-ci kwartyl, min i max wartości w czasie. Łatwiejsze do odczytu gdy interesuje Cię ogólny trend, a nie dokładny kształt rozkładu.

#### 4. TIME SERIES — dane czasowe

Alternatywny widok danych skalarnych z bardziej zaawansowanymi opcjami filtrowania i porównywania.

### Interpretacja naszego treningu (circle_in_square)

Na podstawie wyników z TensorBoard (widocznych na screenie):

1. **Gradients/Layer_FC1_Weights**: Rozkład gradientów pierwszej warstwy — widać że na początku (epoka ~100-400) gradienty są większe i bardziej "rozstrzelone", z czasem się zwężają. To normalne — na początku sieć dużo koryguje, potem coraz mniej, bo jest coraz bliżej optimum.

2. **Gradients/Layer_FC2_Weights**: Podobny wzorzec jak FC1 — gradienty maleją w czasie. Zakres wartości jest inny (od -0.003 do 0.007) => każda warstwa uczy się w swoim "tempie".

3. **Gradients/Layer_FC3_Weights**: Warstwa wyjściowa — gradienty są dodatnie i maleją (od ~0.015 do mniejszych wartości). To logiczne — warstwa wyjściowa bezpośrednio odpowiada za predykcję, więc na początku dostaje najsilniejszy sygnał "popraw się".

4. **Loss** (widoczny poniżej na screenie): Powinien pokazywać gładki spadek od ~0.17 do ~0.004 — potwierdzenie że sieć uczyła się stabilnie i skutecznie.

### Porady praktyczne

- **Porównywanie przebiegów**: Uruchom trening z różnymi parametrami — TensorBoard pokaże WSZYSTKIE przebiegi z folderu `runs/` jednocześnie. Możesz filtrować po nazwie (panel po lewej).
- **Usuwanie starych logów**: Jeśli chcesz zacząć "czysto", usuń folder `runs/` przed treningiem.
- **Logi z wielu sieci**: W tym projekcie XOR, binary-classification i circle-in-square zapisują do osobnych podfolderów `runs/` — możesz porównywać je jednocześnie.
- **Odświeżanie**: TensorBoard nie odświeża się automatycznie — kliknij ikonę odświeżania (strzałka w kółku, prawy górny róg) lub odśwież stronę w przeglądarce.
