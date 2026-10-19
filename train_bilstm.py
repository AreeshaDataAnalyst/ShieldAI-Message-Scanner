
from pathlib import Path
import json
import re
import collections

import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

SEED = 42
MAX_VOCAB = 20000
MAX_LEN = 120
BATCH_SIZE = 64
EPOCHS = 6

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    for path in [
        DATA_DIR / "SpamSmsDataset.xlsx",
        DATA_DIR / "SpamSmsDataset.csv",
        ROOT / "SpamSmsDataset.xlsx",
    ]:
        if not path.exists():
            continue
        try:
            df = pd.read_excel(path)
        except Exception:
            df = pd.read_csv(path, encoding="latin-1")

        df.columns = [str(c).strip().lower() for c in df.columns]
        rename = {}
        for c in df.columns:
            if c in ("text", "sms", "v2"):
                rename[c] = "message"
            if c in ("category", "target", "class", "v1"):
                rename[c] = "label"
        df = df.rename(columns=rename)

        if {"message", "label"}.issubset(df.columns):
            df = df[["message", "label"]].dropna()
            df["message"] = df["message"].astype(str).str.strip()
            df["label"] = df["label"].astype(str).str.lower().str.strip()
            df = df[
                (df["message"] != "") &
                (df["label"].isin(["ham", "spam"]))
            ].drop_duplicates("message")
            return df.reset_index(drop=True)

    raise FileNotFoundError("Put the dataset in the data folder.")

def tokenize(text):
    return re.findall(r"\b\w+\b|[!?$%]+", text.lower())

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"]
    )

    counts = collections.Counter(
        token for message in X_train for token in tokenize(message)
    )
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counts.most_common(MAX_VOCAB - 2):
        vocab[word] = len(vocab)

    def encode(text):
        ids = [vocab.get(t, 1) for t in tokenize(text)][:MAX_LEN]
        return ids + [0] * (MAX_LEN - len(ids))

    class Messages(Dataset):
        def __init__(self, texts, labels):
            self.x = torch.tensor(
                [encode(t) for t in texts], dtype=torch.long
            )
            self.y = torch.tensor(
                [1 if y == "spam" else 0 for y in labels],
                dtype=torch.float32
            )

        def __len__(self):
            return len(self.y)

        def __getitem__(self, index):
            return self.x[index], self.y[index]

    class BiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(
                len(vocab), 96, padding_idx=0
            )
            self.lstm = nn.LSTM(
                input_size=96,
                hidden_size=64,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(128, 1)

        def forward(self, x):
            embedded = self.embedding(x)
            _, (hidden, _) = self.lstm(embedded)
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            combined = torch.cat(
                (forward_hidden, backward_hidden), dim=1
            )
            return self.fc(self.dropout(combined)).squeeze(1)

    train_loader = DataLoader(
        Messages(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_loader = DataLoader(
        Messages(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = BiLSTM().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(texts)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {total_loss / max(len(train_loader), 1):.4f}"
        )

    model.eval()
    all_predictions, all_labels = [], []

    with torch.no_grad():
        for texts, labels in test_loader:
            logits = model(texts.to(device))
            predictions = (torch.sigmoid(logits) >= 0.5).long()
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.long().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    spam_f1 = f1_score(
        all_labels, all_predictions, zero_division=0
    )

    report = classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1],
        target_names=["ham", "spam"],
        output_dict=True,
        zero_division=0
    )

    torch.save({
        "state_dict": model.cpu().state_dict(),
        "vocab": vocab,
        "max_len": MAX_LEN,
        "embedding_dim": 96,
        "hidden_size": 64
    }, MODEL_DIR / "bilstm_model.pt")

    results = {
        "architecture": "Bidirectional LSTM",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": float(accuracy),
        "spam_f1": float(spam_f1),
        "classification_report": report
    }

    with open(
        MODEL_DIR / "bilstm_metrics.json", "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2)

    print("\nBiLSTM test accuracy:", round(accuracy, 4))
    print("BiLSTM spam F1:", round(spam_f1, 4))
    print("Saved models/bilstm_model.pt")
    print("Saved models/bilstm_metrics.json")

if __name__ == "__main__":
    main()
