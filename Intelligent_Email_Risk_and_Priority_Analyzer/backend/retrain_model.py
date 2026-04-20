import argparse
import pickle
import re
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DB_PATH = PROJECT_ROOT / "analysis_logs.db"


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_classifier():
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        print("Using classifier: XGBoost")
        return model
    except Exception as exc:
        print(f"XGBoost unavailable ({exc}). Falling back to RandomForestClassifier.")
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        print("Using classifier: RandomForest")
        return model


def load_feedback_dataframe(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame(columns=["text", "label"])

    conn = sqlite3.connect(db_path)
    try:
        query = """
            SELECT email_text AS text, correct_label AS label
            FROM feedback_logs
            WHERE email_text IS NOT NULL
              AND correct_label IS NOT NULL
        """
        feedback_df = pd.read_sql_query(query, conn)
    except Exception:
        feedback_df = pd.DataFrame(columns=["text", "label"])
    finally:
        conn.close()

    return feedback_df


def normalize_label(value):
    if pd.isna(value):
        return None

    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "phishing", "spam", "malicious"}:
        return 1
    if raw in {"0", "false", "no", "legitimate", "ham", "safe", "not phishing"}:
        return 0

    try:
        numeric = int(float(raw))
        if numeric in (0, 1):
            return numeric
    except Exception:
        pass
    return None


def main(dataset_path: str) -> None:
    original_df = pd.read_csv(dataset_path)
    required_cols = {"text", "label"}
    if not required_cols.issubset(original_df.columns):
        raise ValueError("Original dataset must contain columns: text, label")

    original_df = original_df[["text", "label"]].copy()
    feedback_df = load_feedback_dataframe(DB_PATH)

    combined_df = pd.concat([original_df, feedback_df], ignore_index=True)
    combined_df = combined_df.dropna(subset=["text", "label"])
    combined_df["label"] = combined_df["label"].apply(normalize_label)
    combined_df = combined_df.dropna(subset=["label"]).copy()
    combined_df["label"] = combined_df["label"].astype(int)
    combined_df["text"] = combined_df["text"].apply(preprocess_text)

    if combined_df.empty:
        raise ValueError("No valid training rows after merging original and feedback data.")

    if combined_df["label"].nunique() < 2:
        raise ValueError("Training requires at least two classes in merged data.")

    x = combined_df["text"]
    y = combined_df["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    model = build_classifier()
    model.fit(x_train_tfidf, y_train)

    y_pred = model.predict(x_test_tfidf)
    print(f"Rows in original dataset: {len(original_df)}")
    print(f"Rows in feedback dataset: {len(feedback_df)}")
    print(f"Rows used for retraining: {len(combined_df)}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")

    with open(PROJECT_ROOT / "model_v2.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    with open(PROJECT_ROOT / "vectorizer_v2.pkl", "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)

    print("Saved retrained model to model_v2.pkl")
    print("Saved retrained vectorizer to vectorizer_v2.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain phishing model with feedback data")
    parser.add_argument(
        "--data",
        required=True,
        type=str,
        help="Path to original CSV dataset with columns: text,label",
    )
    args = parser.parse_args()
    main(args.data)
