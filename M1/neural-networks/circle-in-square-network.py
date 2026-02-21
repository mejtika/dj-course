import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from safetensors.torch import save_file
from torch.utils.tensorboard import SummaryWriter
import os

np.set_printoptions(precision=4, suppress=True)

# Config / settings
LOG_DIR = 'runs/circle_in_square'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
writer = SummaryWriter(LOG_DIR)

# 🔥🔥🔥 PRACUJESZ TUTAJ (kształt/struktura sieci)

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

np.random.seed(42)

def generate_circle_data(num_samples, radius=0.5):
    # Generowanie losowych punktów (x, y) w zakresie [-1, 1]
    X_data = 2 * np.random.rand(num_samples, 2) - 1

    # Obliczanie kwadratu odległości od środka (0, 0)
    # distance^2 = x^2 + y^2
    distance_sq = X_data[:, 0]**2 + X_data[:, 1]**2

    # Przypisanie etykiet: 1 jeśli wewnątrz okręgu (distance < radius)
    Y_data = (distance_sq < radius**2).astype(np.float32).reshape(-1, 1)

    # Konwersja do tensorów
    X = torch.tensor(X_data, dtype=torch.float32)
    Y = torch.tensor(Y_data, dtype=torch.float32)
    print(f"wygenerowano {num_samples} punktów treningowych")
    return X, Y

# Generowanie danych
NUM_SAMPLES = 1000 # 🔥🔥🔥 PRACUJESZ TUTAJ
X_circle, Y_circle = generate_circle_data(NUM_SAMPLES)

# Inicjalizacja:
model2 = CircleInSquareNet()

# X_circle, Y_circle z sekcji 2
# model z sekcji 1

# Krok 1: Definicja Parametrów
LEARNING_RATE = 0.01 # 🔥🔥🔥 PRACUJESZ TUTAJ
EPOCHS = 2000 # 🔥🔥🔥 PRACUJESZ TUTAJ
STEP = 100

# Krok 2: Zmiana Optymalizatora i Funkcji Straty
# BCEWithLogitsLoss = Logistyka (Sigmoid) + BCE. Działa na surowych logitach.
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model2.parameters(), lr=LEARNING_RATE) # Używamy Adam

# --- Pętla Treningowa ---
for epoch in range(1, EPOCHS + 1):
    # Zawsze zeruj gradienty na początku
    optimizer.zero_grad()

    # 1. Forward Pass: Zwraca logity
    outputs = model2(X_circle)

    # 2. Obliczenie Straty: Używa logitów i etykiet
    loss = criterion(outputs, Y_circle)

    # 3. Backward Pass: Obliczenie gradientów
    loss.backward()

    # 4. Aktualizacja Wag
    optimizer.step()

    if epoch % STEP == 0:
        writer.add_scalar('Loss', loss.item(), epoch)
        writer.add_histogram('Outputs', outputs.data, epoch)
        writer.add_histogram('Gradients/Layer_FC1_Weights', model2.fc1.weight.grad, epoch)
        writer.add_histogram('Gradients/Layer_FC2_Weights', model2.fc2.weight.grad, epoch)
        writer.add_histogram('Weights/Layer_FC1_Weights', model2.fc1.weight.data, epoch)
        writer.add_histogram('Weights/Layer_FC2_Weights', model2.fc2.weight.data, epoch)
        print(f'Epoka [{epoch}/{EPOCHS}], Strata (Loss): {loss.item():.6f}')

print("--- Trening Zakończony ---")

# Ocena modelu:
with torch.no_grad():
    # Stosujemy Sigmoid tylko do predykcji, aby uzyskać 0/1
    predicted_probs = torch.sigmoid(model2(X_circle))

    # Konwersja prawdopodobieństw na 0 lub 1
    predicted_classes = (predicted_probs >= 0.5).float()

    accuracy = (predicted_classes == Y_circle).sum().item() / NUM_SAMPLES * 100

    print(f"Dokładność na zbiorze treningowym: {accuracy:.2f}%")
    
    if accuracy == 100:
        print("Good job! 🎉")
        MODEL_PATH = "circle_in_square_model_weights.pth"
        torch.save(model2.state_dict(), MODEL_PATH)
        print(f"Model zapisany jako: {MODEL_PATH}")
    else:
        print("FIX ME PLEASE! 😟")
    
print(f"(run tensorboard/venv): tensorboard --logdir={LOG_DIR}")
print(f"(run tensorboard/venv): tensorboard --logdir=runs")
print("\nopen http://localhost:6006/; SCALARS - how loss changed over time; HISTOGRAMS - how gradients distributed over epochs")
