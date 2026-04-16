# Customer Chunk Prediction API

A FastAPI-based REST API for predicting customer message chunks (likely sentiment/review classification) using a pre-trained sklearn LogisticRegression model.

## Features
- Predict text classification from input message
- Preprocessing pipeline: HTML tag removal, tokenization, TF-IDF vectorization
- Deployable with uvicorn/ASGI server
- JSON input/output

## Files
- `api.py`: Main FastAPI application
- `model.pkl`: Trained LogisticRegression model
- `pre1.pkl`, `pre2.pkl`, `pre3.pkl`: Custom preprocessing transformers (e.g., remove_tags, etc.)
- `tf.pkl`: TF-IDF vectorizer
- `IMDB Dataset.csv`: Original training data (assumed)
- `hello.ipynb`: Model training notebook

## Requirements
- Python 3.8+
- FastAPI, Uvicorn
- scikit-learn (note: version mismatch warning possible if !=1.6.1)
- pandas, numpy

Install:
```bash
pip install fastapi uvicorn scikit-learn pandas numpy
```

## Quickstart
1. Ensure all `.pkl` files are in the same directory as `api.py`.
2. Run the server:
   
```bash
   uvicorn api:app --reload
   
```
   Server runs on http://127.0.0.1:8000

3. Test endpoints:
   - GET `/`: `{"message": "hello this is api of customer chunk prediction"}`
   - POST `/predict`: See below

## API Endpoints

### POST /predict
Predict class for input message.

**Request:**
```json
{
  "message": "this is a test review"
}
```

**Response:**
```json
{
  "prediction": "Postive" 
}
```

Example with curl:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"this is a test review\"}"
```

## Interactive Docs
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## Troubleshooting
- **Pickle unpickling error** (e.g., 'remove_tags'): Ensure custom functions like `remove_tags` are defined in `api.py`.
- **Sklearn version warning**: Model trained with 1.6.1, runtime 1.8.0 - may work but retrain if issues.
- Windows/PowerShell curl: Use `curl.exe` or PowerShell `Invoke-RestMethod`.

## Training
See `hello.ipynb` for model training on IMDB Dataset.csv.
