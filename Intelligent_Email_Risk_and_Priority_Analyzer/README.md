# Intelligent Email Risk & Priority Analyzer

## Summary
Intelligent Email Risk & Priority Analyzer is an AI-based system that helps users detect suspicious emails quickly and understand threat severity with clear outputs. The backend analyzes email text using TF-IDF and XGBoost, then combines model confidence with URL and urgency signals to produce a final risk score from 0 to 100. It returns risk level, explainable reasons, and a short summary to reduce reading effort. The project also includes a simple web interface and a Chrome extension that works with Gmail content, enabling near real-time analysis. It supports logging and feedback collection, so the model can be retrained and improved over time.

## Tech Stack
- Python
- FastAPI
- Scikit-learn (TF-IDF)
- XGBoost
- HTML, CSS, JavaScript
- Chrome Extension (Manifest V3)
- SQLite

## Output
- Accuracy: `0.9530`
- Precision: `0.9101`
- Recall: `0.9767`
- F1 Score: `0.9423`

## Chrome Extension
Reads opened Gmail email text, sends it to backend `/analyze`, and shows risk score + risk level.

## Steps
1. Activate environment:
   `source .venv/bin/activate`
2. Train model:
   `python3 backend/train_phishing_model.py --data /Users/balivadamanoharvikas/Downloads/Phishing_Email.csv`
3. Start backend:
   `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`
4. Load extension in Chrome:
   open `chrome://extensions` -> enable Developer Mode -> Load unpacked -> select `extension/`
5. Open Gmail email and click Analyze to view risk result.
