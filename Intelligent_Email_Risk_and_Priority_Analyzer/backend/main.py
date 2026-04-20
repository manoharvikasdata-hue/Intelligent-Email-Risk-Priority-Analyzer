import pickle
import re
import sqlite3
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = PROJECT_ROOT / "model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "vectorizer.pkl"
MODEL_V2_PATH = PROJECT_ROOT / "model_v2.pkl"
VECTORIZER_V2_PATH = PROJECT_ROOT / "vectorizer_v2.pkl"
LOG_DB_PATH = PROJECT_ROOT / "analysis_logs.db"
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "email_risk_db")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "classified_items")

try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def to_risk_level(risk_score: float) -> str:
    if risk_score < 34:
        return "Low"
    if risk_score < 67:
        return "Medium"
    return "High"


def extract_urls(text: str) -> List[str]:
    url_pattern = r"(https?://[^\s]+|www\.[^\s]+)"
    return re.findall(url_pattern, text)


def count_urgent_words(text: str) -> int:
    words = ["urgent", "verify", "click"]
    lower_text = text.lower()
    return sum(len(re.findall(rf"\b{re.escape(word)}\b", lower_text)) for word in words)


def load_model_artifacts():
    artifact_pairs = [
        (MODEL_PATH, VECTORIZER_PATH),
        (MODEL_V2_PATH, VECTORIZER_V2_PATH),
    ]
    for model_path, vectorizer_path in artifact_pairs:
        if model_path.exists() and vectorizer_path.exists():
            with open(model_path, "rb") as model_file:
                loaded_model = pickle.load(model_file)
            with open(vectorizer_path, "rb") as vec_file:
                loaded_vectorizer = pickle.load(vec_file)
            return loaded_model, loaded_vectorizer, model_path.name, vectorizer_path.name
    return None, None, None, None


model, vectorizer, loaded_model_name, loaded_vectorizer_name = load_model_artifacts()


def init_db() -> None:
    conn = sqlite3.connect(LOG_DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_text TEXT NOT NULL,
                prediction TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_text TEXT NOT NULL,
                correct_label TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_log(email_text: str, prediction: str) -> None:
    conn = sqlite3.connect(LOG_DB_PATH)
    try:
        ts = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO analysis_logs (email_text, prediction, timestamp) VALUES (?, ?, ?)",
            (email_text, prediction, ts),
        )
        conn.commit()
    finally:
        conn.close()


init_db()


class AnalyzeRequest(BaseModel):
    email: str = Field(..., min_length=1, description="Raw email text")


class AnalyzeResponse(BaseModel):
    risk_score: float
    risk_level: str
    reasons: List[str]
    summary: str


class MongoIngestRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Unstructured text/email content")
    source: Optional[str] = Field(default="unknown", description="Source system name")
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MongoIngestResponse(BaseModel):
    document_id: str
    risk_score: float
    risk_level: str
    reasons: List[str]
    summary: str


class FeedbackRequest(BaseModel):
    email: str = Field(..., min_length=1, description="Raw email text")
    correct_label: str = Field(..., min_length=1, description="Correct label")


class FeedbackResponse(BaseModel):
    message: str


app = FastAPI(title="Phishing Email Analysis API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def summarize_email_text(email_text: str, max_sentences: int = 3) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", (email_text or "").strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return ""
    return "\n".join(sentences[:max_sentences])


def run_risk_analysis(raw_text: str) -> AnalyzeResponse:
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model artifacts not found. Add model/vectorizer files in project root: "
                "model.pkl + vectorizer.pkl (or model_v2.pkl + vectorizer_v2.pkl)."
            ),
        )

    cleaned_text = preprocess_text(raw_text)
    features = vectorizer.transform([cleaned_text])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        if len(probabilities) < 2:
            raise ValueError("Model probability output is not binary.")
        ml_score = float(probabilities[1])
    else:
        prediction = model.predict(features)[0]
        ml_score = float(prediction)
    ml_score = max(0.0, min(1.0, ml_score))

    urls = extract_urls(raw_text)
    url_count = len(urls)
    urgent_count = count_urgent_words(raw_text)

    url_score = min(url_count / 5, 1.0)
    urgent_score = min(urgent_count / 6, 1.0)

    final_risk_0_1 = (ml_score * 0.5) + (url_score * 0.3) + (urgent_score * 0.2)
    final_risk_score = round(final_risk_0_1 * 100, 2)

    reasons: List[str] = []
    reasons.append(f"ML model phishing probability: {ml_score:.2f}")
    if url_count > 0:
        reasons.append(f"Detected {url_count} URL(s) in email content.")
    if urgent_count > 0:
        reasons.append(
            f"Detected {urgent_count} urgent keyword occurrence(s): urgent/verify/click."
        )
    if not urls and urgent_count == 0:
        reasons.append("No URLs or urgent trigger words detected.")
    reasons.append(
        f"Weighted score contributions -> ML: {ml_score * 50:.2f}, "
        f"URLs: {url_score * 30:.2f}, Urgent words: {urgent_score * 20:.2f}"
    )

    return AnalyzeResponse(
        risk_score=final_risk_score,
        risk_level=to_risk_level(final_risk_score),
        reasons=reasons,
        summary=summarize_email_text(raw_text, 3),
    )


def get_mongo_collection():
    if MongoClient is None:
        raise HTTPException(
            status_code=503,
            detail="pymongo is not installed. Install it with: pip install pymongo",
        )
    client = MongoClient(MONGODB_URI)
    db = client[MONGODB_DB]
    return client, db[MONGODB_COLLECTION]


@app.get("/")
def root():
    return {
        "message": "Phishing Email Analysis API is running.",
        "endpoints": ["/analyze", "/feedback", "/ingest/mongodb", "/health", "/docs"],
        "model_loaded": model is not None and vectorizer is not None,
        "model_file": loaded_model_name,
        "vectorizer_file": loaded_vectorizer_name,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None and vectorizer is not None,
    }


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_email(payload: AnalyzeRequest) -> AnalyzeResponse:
    try:
        raw_text = payload.email
        analysis = run_risk_analysis(raw_text)

        prediction_text = (
            f"risk_score={analysis.risk_score:.2f}, "
            f"risk_level={analysis.risk_level}, "
            f"model={loaded_model_name}, vectorizer={loaded_vectorizer_name}"
        )
        save_log(raw_text, prediction_text)
        return analysis
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    try:
        conn = sqlite3.connect(LOG_DB_PATH)
        try:
            ts = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO feedback_logs (email_text, correct_label, timestamp) VALUES (?, ?, ?)",
                (payload.email, payload.correct_label, ts),
            )
            conn.commit()
        finally:
            conn.close()

        return FeedbackResponse(message="Feedback saved successfully.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Feedback save failed: {exc}") from exc


@app.post("/ingest/mongodb", response_model=MongoIngestResponse)
def ingest_and_classify_mongodb(payload: MongoIngestRequest) -> MongoIngestResponse:
    try:
        analysis = run_risk_analysis(payload.text)
        now_iso = datetime.now(timezone.utc).isoformat()
        cleaned_text = preprocess_text(payload.text)

        document = {
            "raw_text": payload.text,
            "normalized_text": cleaned_text,
            "source": payload.source,
            "file": {
                "name": payload.file_name,
                "type": payload.file_type,
            },
            "metadata": payload.metadata or {},
            "classification": {
                "risk_score": analysis.risk_score,
                "risk_level": analysis.risk_level,
                "reasons": analysis.reasons,
                "summary": analysis.summary,
                "model_file": loaded_model_name,
                "vectorizer_file": loaded_vectorizer_name,
            },
            "created_at": now_iso,
        }

        client, collection = get_mongo_collection()
        try:
            insert_result = collection.insert_one(document)
        finally:
            client.close()

        return MongoIngestResponse(
            document_id=str(insert_result.inserted_id),
            risk_score=analysis.risk_score,
            risk_level=analysis.risk_level,
            reasons=analysis.reasons,
            summary=analysis.summary,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MongoDB ingest failed: {exc}") from exc
