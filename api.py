
from pathlib import Path
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Reuse the existing SmartLib implementation.
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app import LibraryRecommender  # noqa: E402

DATA_DIR = BASE_DIR / "data"

def load_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        raise RuntimeError(f"Required dataset not found: {path}")
    return pd.read_csv(path)

books_df = load_csv("books.csv")
users_df = load_csv("users.csv")
borrowing_df = load_csv("borrowing_history.csv")

recommender = LibraryRecommender(
    books_df,
    users_df,
    borrowing_df,
)

api = FastAPI(title="SmartLib API", version="1.0.0")

# Local development only. Restrict this before public deployment.
api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["GET"],
    allow_headers=["*"],
)

def clean_records(df):
    # Convert pandas/NumPy values into JSON-friendly Python values.
    return (
        df.astype(object)
        .where(pd.notna(df), None)
        .to_dict(orient="records")
    )

@api.get("/api/health")
def health():
    return {
        "status": "ok",
        "books_loaded": len(books_df),
        "users_loaded": len(users_df),
        "borrowing_records_loaded": len(borrowing_df),
    }

@api.get("/api/books")
def get_books(
    q: str = Query(default="", max_length=200),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    result = books_df.copy()

    if q.strip():
        query = q.strip()
        columns = [
            col for col in ["title", "author", "genre", "isbn", "description"]
            if col in result.columns
        ]
        if columns:
            mask = pd.Series(False, index=result.index)
            for col in columns:
                mask |= result[col].astype(str).str.contains(
                    query, case=False, na=False, regex=False
                )
            result = result[mask]

    total = len(result)
    page = result.iloc[offset:offset + limit]

    return {"total": total, "items": clean_records(page)}

@api.get("/api/recommendations/{user_id}")
def get_recommendations(
    user_id: int,
    limit: int = Query(default=6, ge=1, le=20),
):
    if not (users_df["user_id"] == user_id).any():
        raise HTTPException(status_code=404, detail="User not found")

    try:
        results = recommender.recommend_books_for_user(
            user_id, top_n=limit
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Recommendation generation failed",
        ) from exc

    return {"user_id": user_id, "items": clean_records(results)}
