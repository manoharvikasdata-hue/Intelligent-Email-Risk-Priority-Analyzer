import argparse
import pickle
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    text_candidates = [
        "text",
        "email",
        "email_text",
        "email text",
        "message",
        "body",
        "content",
    ]
    label_candidates = [
        "label",
        "spam",
        "target",
        "class",
        "is_phishing",
        "phishing",
        "email_type",
        "email type",
    ]

    def canon(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")

    col_lookup = {canon(c): c for c in df.columns}

    text_col = next((col_lookup.get(canon(c)) for c in text_candidates if canon(c) in col_lookup), None)
    label_col = next(
        (col_lookup.get(canon(c)) for c in label_candidates if canon(c) in col_lookup), None
    )

    if label_col is None:
        raise ValueError(
            f"Could not find label column. Expected one of: {label_candidates}. "
            f"Found columns: {list(df.columns)}"
        )

    if text_col is None:
        feature_cols = [c for c in df.columns if c != label_col]
        if not feature_cols:
            raise ValueError("No feature columns available to build text input.")
        df = df.copy()
        # Build pseudo text for structured datasets (e.g., spam feature tables).
        df["text"] = df[feature_cols].astype(str).agg(" ".join, axis=1)
        text_col = "text"

    out = df[[text_col, label_col]].copy()
    out.columns = ["text", "label"]
    # Normalize common string labels into binary values.
    label_map = {
        "phishing": 1,
        "phishing email": 1,
        "spam": 1,
        "malicious": 1,
        "safe email": 0,
        "safe": 0,
        "ham": 0,
        "legitimate": 0,
        "not phishing": 0,
    }
    out["label"] = (
        out["label"].astype(str).str.strip().str.lower().map(label_map).fillna(out["label"])
    )
    return out


def main(dataset_path: str) -> None:
    dataset_file = Path(dataset_path).expanduser()
    if not dataset_file.is_absolute():
        dataset_file = (Path.cwd() / dataset_file).resolve()
    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_file}. "
            "Pass a real CSV path with --data."
        )

    raw_df = pd.read_csv(dataset_file)
    df = normalize_training_dataframe(raw_df).dropna().copy()
    df["text"] = df["text"].apply(preprocess_text)
    original_labels = df["label"].copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    if df["label"].isna().any():
        bad_values = original_labels.loc[df["label"].isna()].astype(str).unique().tolist()
        raise ValueError(
            "Found unsupported label values after normalization. "
            f"Please map them to 0/1. Samples: {bad_values[:5]}"
        )
    df["label"] = df["label"].astype(int)

    x = df["text"]
    y = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    if pos_count == 0:
        raise ValueError("No positive (phishing) samples found in training split.")
    scale_pos_weight = neg_count / pos_count

    vectorizer = TfidfVectorizer(max_features=5000)
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(x_train_tfidf, y_train)

    y_pred = model.predict(x_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Training negatives: {neg_count}")
    print(f"Training positives: {pos_count}")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1 Score:        {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print(f"Unique predicted values: {sorted(pd.Series(y_pred).unique().tolist())}")

    with open(PROJECT_ROOT / "model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    with open(PROJECT_ROOT / "vectorizer.pkl", "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)

    print("Saved model to model.pkl")
    print("Saved vectorizer to vectorizer.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train phishing email detection model")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV dataset with columns: text, label",
    )
    args = parser.parse_args()
    main(args.data)
