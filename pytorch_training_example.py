"""
Esempio di addestramento PyTorch con 3 input e 2 output.

Input: 3 valori double nel range [1, 100]
Output 1: input1 + input2
Output 2: input2 + input3
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CustomDataset(Dataset):
    """Dataset personalizzato con 3 input e 2 output."""

    def __init__(self, num_samples=1000):
        """
        Genera dati sintetici.

        Args:
            num_samples: Numero di campioni da generare
        """
        # Genera input casuali nel range [1, 100]
        self.inputs = np.random.uniform(1, 100, size=(num_samples, 3))

        # Calcola gli output
        # Output 1 = input1 + input2
        # Output 2 = input2 + input3
        output1 = self.inputs[:, 0] + self.inputs[:, 1]
        output2 = self.inputs[:, 1] + self.inputs[:, 2]
        self.outputs = np.stack([output1, output2], axis=1)

        # Converti in tensori float
        self.inputs = torch.FloatTensor(self.inputs)
        self.outputs = torch.FloatTensor(self.outputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


class NeuralNetwork(nn.Module):
    """Rete neurale con 3 input e 2 output."""

    def __init__(self, hidden_size=64):
        """
        Inizializza la rete neurale.

        Args:
            hidden_size: Dimensione degli strati nascosti
        """
        super(NeuralNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor di forma (batch_size, 3)

        Returns:
            Output tensor di forma (batch_size, 2)
        """
        return self.layers(x)


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """
    Addestra il modello.

    Args:
        model: Il modello da addestrare
        train_loader: DataLoader per i dati di training
        val_loader: DataLoader per i dati di validazione
        num_epochs: Numero di epoche di training
        learning_rate: Learning rate per l'ottimizzatore
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"Training su device: {device}")
    print(f"Numero di parametri: {sum(p.numel() for p in model.parameters())}")
    print("-" * 60)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass e ottimizzazione
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validazione
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Stampa progresso ogni 10 epoche
        if (epoch + 1) % 10 == 0:
            print(f"Epoca [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f}")

    print("-" * 60)
    print("Training completato!")


def test_model(model, test_samples=5):
    """
    Testa il modello con alcuni campioni casuali.

    Args:
        model: Il modello addestrato
        test_samples: Numero di campioni da testare
    """
    model.eval()
    device = next(model.parameters()).device

    print("\nTest del modello:")
    print("-" * 60)

    with torch.no_grad():
        for i in range(test_samples):
            # Genera input casuale
            inputs = np.random.uniform(1, 100, size=3)

            # Calcola output attesi
            expected_output1 = inputs[0] + inputs[1]
            expected_output2 = inputs[1] + inputs[2]

            # Predizione del modello
            input_tensor = torch.FloatTensor(inputs).unsqueeze(0).to(device)
            predicted = model(input_tensor).cpu().numpy()[0]

            print(f"\nCampione {i+1}:")
            print(f"  Input: [{inputs[0]:.2f}, {inputs[1]:.2f}, {inputs[2]:.2f}]")
            print(f"  Output atteso: [{expected_output1:.2f}, {expected_output2:.2f}]")
            print(f"  Output predetto: [{predicted[0]:.2f}, {predicted[1]:.2f}]")
            print(f"  Errore: [{abs(expected_output1 - predicted[0]):.2f}, "
                  f"{abs(expected_output2 - predicted[1]):.2f}]")


def main():
    """Funzione principale."""

    # Parametri
    NUM_TRAIN_SAMPLES = 10000
    NUM_VAL_SAMPLES = 2000
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    HIDDEN_SIZE = 64

    print("=" * 60)
    print("Esempio di Training PyTorch")
    print("=" * 60)
    print(f"Numero campioni di training: {NUM_TRAIN_SAMPLES}")
    print(f"Numero campioni di validazione: {NUM_VAL_SAMPLES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Numero epoche: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("=" * 60)

    # Crea i dataset
    train_dataset = CustomDataset(num_samples=NUM_TRAIN_SAMPLES)
    val_dataset = CustomDataset(num_samples=NUM_VAL_SAMPLES)

    # Crea i DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Crea il modello
    model = NeuralNetwork(hidden_size=HIDDEN_SIZE)

    # Addestra il modello
    train_model(model, train_loader, val_loader,
                num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)

    # Testa il modello
    test_model(model, test_samples=5)

    # Salva il modello
    torch.save(model.state_dict(), 'model.pth')
    print("\nModello salvato in 'model.pth'")


if __name__ == "__main__":
    main()
