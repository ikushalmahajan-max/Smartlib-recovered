"""
Dataset Audit & Validation Module for Library Recommendation System.
Detects missing foreign keys, duplicate records, missing values, and type anomalies.
Generates safe, clean DataFrames without silently modifying underlying raw CSV files unless requested.
"""

import os
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import USERS_CSV, BOOKS_CSV, BORROWING_CSV, DATA_DIR


def audit_datasets(users_path=USERS_CSV, books_path=BOOKS_CSV, borrow_path=BORROWING_CSV):
    """
    Performs a thorough audit on users, books, and borrowing history datasets.
    Returns a dictionary summarizing all detected issues and statistics.
    """
    issues = {
        'users_missing_in_users_table': [],
        'books_missing_in_books_table': [],
        'duplicate_user_ids': [],
        'duplicate_book_ids': [],
        'duplicate_transaction_ids': [],
        'duplicate_borrow_records': [],
        'users_null_fields': [],
        'books_null_fields': [],
        'borrow_null_fields': [],
        'invalid_ratings': [],
        'summary': {}
    }

    users_df = pd.read_csv(users_path)
    books_df = pd.read_csv(books_path)
    borrow_df = pd.read_csv(borrow_path)

    # 1. Primary Key Duplicates
    dup_u = users_df[users_df.duplicated(subset=['user_id'], keep=False)]
    if not dup_u.empty:
        issues['duplicate_user_ids'] = dup_u['user_id'].tolist()

    dup_b = books_df[books_df.duplicated(subset=['book_id'], keep=False)]
    if not dup_b.empty:
        issues['duplicate_book_ids'] = dup_b['book_id'].tolist()

    dup_t = borrow_df[borrow_df.duplicated(subset=['transaction_id'], keep=False)]
    if not dup_t.empty:
        issues['duplicate_transaction_ids'] = dup_t['transaction_id'].tolist()

    dup_rec = borrow_df[borrow_df.duplicated(subset=['user_id', 'book_id', 'borrow_date'], keep=False)]
    if not dup_rec.empty:
        issues['duplicate_borrow_records'] = dup_rec.to_dict(orient='records')

    # 2. Foreign Key Integrity
    valid_user_ids = set(users_df['user_id'].dropna().astype(int))
    valid_book_ids = set(books_df['book_id'].dropna().astype(int))

    borrow_users = set(borrow_df['user_id'].dropna().astype(int))
    borrow_books = set(borrow_df['book_id'].dropna().astype(int))

    missing_users = list(borrow_users - valid_user_ids)
    missing_books = list(borrow_books - valid_book_ids)

    issues['users_missing_in_users_table'] = missing_users
    issues['books_missing_in_books_table'] = missing_books

    # 3. Null Values & Rating Anomalies
    issues['users_null_fields'] = users_df.isnull().sum().to_dict()
    issues['books_null_fields'] = books_df.isnull().sum().to_dict()
    issues['borrow_null_fields'] = borrow_df[['transaction_id', 'user_id', 'book_id', 'borrow_date']].isnull().sum().to_dict()

    invalid_r = books_df[(books_df['rating'] < 1.0) | (books_df['rating'] > 5.0)]
    if not invalid_r.empty:
        issues['invalid_ratings'] = invalid_r['book_id'].tolist()

    # Summary Stats
    issues['summary'] = {
        'total_users_records': len(users_df),
        'unique_user_ids': len(valid_user_ids),
        'total_books_records': len(books_df),
        'unique_book_ids': len(valid_book_ids),
        'total_borrow_records': len(borrow_df),
        'has_critical_fk_errors': len(missing_users) > 0 or len(missing_books) > 0
    }

    return issues


def get_cleaned_datasets(users_path=USERS_CSV, books_path=BOOKS_CSV, borrow_path=BORROWING_CSV, save_cleaned=False):
    """
    Generates cleaned and verified DataFrames for users, books, and borrowing history.
    Fixes foreign key integrity, deduplicates transactions, and auto-registers missing profiles.
    """
    users_df = pd.read_csv(users_path)
    books_df = pd.read_csv(books_path)
    borrow_df = pd.read_csv(borrow_path)

    # 1. Clean Books (Deduplicate book_id, clip ratings)
    books_df = books_df.drop_duplicates(subset=['book_id'], keep='first').copy()
    books_df['rating'] = books_df['rating'].clip(1.0, 5.0)
    books_df['book_id'] = books_df['book_id'].astype(int)

    # 2. Clean Borrowing History (Deduplicate transaction_id and identical loans)
    borrow_df = borrow_df.drop_duplicates(subset=['transaction_id'], keep='first').copy()
    borrow_df = borrow_df.drop_duplicates(subset=['user_id', 'book_id', 'borrow_date'], keep='first').copy()
    borrow_df['transaction_id'] = borrow_df['transaction_id'].astype(int)
    borrow_df['user_id'] = borrow_df['user_id'].astype(int)
    borrow_df['book_id'] = borrow_df['book_id'].astype(int)

    # Filter out any borrow records referencing non-existent book IDs
    valid_book_ids = set(books_df['book_id'])
    borrow_df = borrow_df[borrow_df['book_id'].isin(valid_book_ids)].copy()

    # 3. Clean & Synchronize Users (Auto-register missing user_ids from borrowing history)
    users_df = users_df.drop_duplicates(subset=['user_id'], keep='first').copy()
    users_df['user_id'] = users_df['user_id'].astype(int)

    existing_user_ids = set(users_df['user_id'])
    required_user_ids = set(borrow_df['user_id'])
    missing_user_ids = sorted(list(required_user_ids - existing_user_ids))

    if missing_user_ids:
        # Default user profiles for missing IDs in borrowing history
        new_records = []
        for uid in missing_user_ids:
            new_records.append({
                'user_id': uid,
                'name': f"Student_{uid}",
                'department': 'E&TC' if uid % 2 == 1 else 'Computer Science',
                'year': 'SE' if uid % 3 == 0 else 'TE',
                'email': f"student_{uid}@nmiet.edu.in"
            })
        new_users_df = pd.DataFrame(new_records)
        users_df = pd.concat([users_df, new_users_df], ignore_index=True)

    users_df = users_df.sort_values(by='user_id').reset_index(drop=True)
    books_df = books_df.sort_values(by='book_id').reset_index(drop=True)
    borrow_df = borrow_df.sort_values(by='transaction_id').reset_index(drop=True)

    if save_cleaned:
        os.makedirs(DATA_DIR, exist_ok=True)
        users_df.to_csv(USERS_CSV, index=False)
        books_df.to_csv(BOOKS_CSV, index=False)
        borrow_df.to_csv(BORROWING_CSV, index=False)

    return users_df, books_df, borrow_df


if __name__ == '__main__':
    print("--- AUDITING DATASETS ---")
    report = audit_datasets()
    import json
    print(json.dumps(report, indent=2))
    print("\n--- GENERATING CLEANED DATASETS ---")
    u_df, b_df, br_df = get_cleaned_datasets()
    print(f"Clean Users: {len(u_df)} records")
    print(f"Clean Books: {len(b_df)} records")
    print(f"Clean Borrow History: {len(br_df)} records")
