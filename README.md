# Fake News Detector

This project uses NLP (BERT-like) models to detect fake news. It provides a web interface to check if a news item is 'True', 'False', 'Mostly False', or 'Misleading'.

## Setup

### Backend
1. Go to the backend directory:
   ```
   cd backend
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the API server:
   ```
   python app.py
   ```

### Frontend
Just open `frontend/index.html` in your browser.

## Usage
- Paste news text in the web page and click 'Check'.
- The backend will return a label prediction.
