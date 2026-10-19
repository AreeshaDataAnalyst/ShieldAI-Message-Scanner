
from pathlib import Path
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def load_dataset():
    files = [
        DATA_DIR / "SpamSmsDataset.xlsx",
        DATA_DIR / "SpamSmsDataset.csv",
        ROOT / "SpamSmsDataset.xlsx",
        ROOT / "SpamSmsDataset.csv",
    ]

    for path in files:
        if not path.exists():
            continue

        try:
            df = pd.read_excel(path)
        except Exception:
            try:
                df = pd.read_csv(path, encoding="utf-8")
            except Exception:
                df = pd.read_csv(path, encoding="latin-1")

        df.columns = [str(c).strip().lower() for c in df.columns]

        # Support common alternative column names.
        rename = {}
        for c in df.columns:
            if c in ("text", "sms", "message_text", "v2"):
                rename[c] = "message"
            elif c in ("category", "target", "class", "v1"):
                rename[c] = "label"
        df = df.rename(columns=rename)

        if {"message", "label"}.issubset(df.columns):
            print(f"Dataset loaded: {path}")
            return df[["message", "label"]].copy()

    raise FileNotFoundError(
        "Dataset not found or columns missing. "
        "Put SpamSmsDataset.xlsx in the data folder. "
        "Required columns: message and label."
    )

def main():
    df = load_dataset()

    df["message"] = df["message"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].fillna("").astype(str).str.strip().str.lower()
    df = df[df["message"] != ""]
    df = df[df["label"].isin(["spam", "ham"])]

    # Avoid the same message appearing in both train and test sets.
    df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)

    if df["label"].nunique() != 2:
        raise ValueError("Dataset must contain both spam and ham labels.")

    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    candidates = {
        "logistic_regression": Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_features=100000,
                sublinear_tf=True
            )),
            ("classifier", LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                random_state=42
            ))
        ]),
        "naive_bayes": Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=100000
            )),
            ("classifier", MultinomialNB(alpha=0.5))
        ])
    }

    results = {}
    trained = {}

    for name, model in candidates.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "spam_precision": float(precision_score(
                y_test, predictions, pos_label="spam",
                zero_division=0
            )),
            "spam_recall": float(recall_score(
                y_test, predictions, pos_label="spam",
                zero_division=0
            )),
            "spam_f1": float(f1_score(
                y_test, predictions, pos_label="spam",
                zero_division=0
            )),
            "classification_report": classification_report(
                y_test, predictions, output_dict=True,
                zero_division=0
            ),
            "confusion_matrix_labels": ["ham", "spam"],
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=["ham", "spam"]
            ).tolist()
        }

        trained[name] = model

        print(f"Accuracy: {results[name]['accuracy']:.4f}")
        print(f"Spam F1:  {results[name]['spam_f1']:.4f}")

    best_name = max(
        results,
        key=lambda name: results[name]["spam_f1"]
    )
    best_model = trained[best_name]

    joblib.dump(best_model, MODEL_DIR / "spam_model.joblib")

    metadata = {
        "best_model": best_name,
        "total_unique_messages": int(len(df)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "split": "80/20",
        "random_state": 42,
        "classes": ["ham", "spam"],
        "results": results
    }

    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nBest model:", best_name)
    print("Saved:", MODEL_DIR / "spam_model.joblib")
    print("Saved:", MODEL_DIR / "metrics.json")
    print("\nTraining and testing completed successfully.")

if __name__ == "__main__":
    main()
