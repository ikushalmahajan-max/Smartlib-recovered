"""
Synthetic Academic Library Dataset Generator for Research Publication.
Generates reproducible synthetic datasets with realistic academic library distributions
(Power-law book popularity, Pareto user borrowing activity, and department domain preferences).

IMPORTANT NOTICE:
This dataset is strictly SYNTHETIC and generated using probabilistic distributions for research evaluation.
It does NOT represent real historical student borrowing logs from any institution.
"""

import os
import sys
import random
import datetime
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    RANDOM_SEED, SYNTHETIC_BOOKS_COUNT, SYNTHETIC_USERS_COUNT,
    SYNTHETIC_TRANSACTIONS_COUNT, DATA_DIR,
    SYNTHETIC_BOOKS_CSV, SYNTHETIC_USERS_CSV, SYNTHETIC_BORROWING_CSV,
    BOOKS_CSV, USERS_CSV, BORROWING_CSV
)


DEPARTMENTS = [
    'E&TC', 'Computer Science', 'AI & Data Science',
    'Software Engineering', 'Electronics', 'Information Technology',
    'Databases', 'Networking'
]

GENRES_BY_DEPT = {
    'E&TC': ['Electronics', 'Signal Processing', 'Microcontrollers', 'Communication Systems'],
    'Computer Science': ['Algorithms', 'Operating Systems', 'Computer Architecture', 'Theory of Computation'],
    'AI & Data Science': ['AI/ML', 'Deep Learning', 'Pattern Recognition', 'Data Mining'],
    'Software Engineering': ['Software Design', 'Agile Architecture', 'Clean Code', 'Testing'],
    'Electronics': ['Semiconductors', 'VLSI Design', 'Embedded Systems', 'Robotics'],
    'Information Technology': ['Web Technologies', 'Cloud Computing', 'Cyber Security', 'DevOps'],
    'Databases': ['Relational DB', 'Distributed Databases', 'NoSQL', 'Query Optimization'],
    'Networking': ['Computer Networks', 'Wireless Protocols', 'Network Security', 'IoT']
}

TITLE_PREFIXES = ['Introduction to', 'Advanced', 'Principles of', 'Fundamentals of', 'Handbook of', 'Applied', 'Modern']
TITLE_SUBJECTS = ['Algorithms', 'Deep Learning', 'Embedded Systems', 'Computer Networks', 'Database Systems',
                  'VLSI Architecture', 'Machine Learning', 'Signal Processing', 'Clean Code', 'Pattern Recognition',
                  'Data Structures', 'Operating Systems', 'Cloud Computing', 'Cyber Security', 'Robotics']

FIRST_NAMES = ['Aarav', 'Priya', 'Rohan', 'Ananya', 'Aditya', 'Isha', 'Siddharth', 'Sneha', 'Vikram', 'Neha',
               'Dev', 'Kavya', 'Rahul', 'Pooja', 'Amit', 'Shruti', 'Yash', 'Tanvi', 'Manish', 'Riya']
LAST_NAMES = ['Sharma', 'Patel', 'Verma', 'Rao', 'Deshmukh', 'Kulkarni', 'Gupta', 'Malhotra', 'Joshi', 'Mehta',
              'Chavan', 'Nair', 'Bhat', 'Pawar', 'Singh']


def generate_synthetic_dataset(
    num_books=SYNTHETIC_BOOKS_COUNT,
    num_users=SYNTHETIC_USERS_COUNT,
    num_transactions=SYNTHETIC_TRANSACTIONS_COUNT,
    seed=RANDOM_SEED,
    activate_as_main=True
):
    """
    Generates realistic synthetic academic library datasets and writes them to data/.
    """
    random.seed(seed)
    np.random.seed(seed)

    print(f"Generating synthetic research dataset: {num_books} books, {num_users} users, {num_transactions} transactions (Seed: {seed})...")

    # 1. Generate Books
    books = []
    for i in range(1, num_books + 1):
        b_id = 100 + i
        dept = random.choice(DEPARTMENTS)
        genre = random.choice(GENRES_BY_DEPT[dept])
        title = f"{random.choice(TITLE_PREFIXES)} {random.choice(TITLE_SUBJECTS)} (Vol. {random.randint(1, 4)})"
        author = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        isbn = f"978-013{i:06d}"
        rating = round(float(np.clip(np.random.normal(4.5, 0.4), 3.0, 5.0)), 1)
        pub_year = random.randint(2000, 2024)
        desc = f"Academic engineering reference textbook on {genre.lower()} and {dept.lower()} principles."

        books.append({
            'book_id': b_id,
            'title': title,
            'author': author,
            'genre': genre,
            'isbn': isbn,
            'rating': rating,
            'publication_year': pub_year,
            'description': desc
        })

    books_df = pd.DataFrame(books)

    # 2. Generate Users (Include cold-start, sparse, and normal readers)
    users = []
    years = ['FE', 'SE', 'TE', 'BE']
    for i in range(1, num_users + 1):
        u_id = 200 + i
        fn = random.choice(FIRST_NAMES)
        ln = random.choice(LAST_NAMES)
        name = f"{fn} {ln}"
        dept = random.choice(DEPARTMENTS)
        year = random.choice(years)
        email = f"{fn.lower()}.{ln.lower()}{u_id}@nmiet.edu.in"

        users.append({
            'user_id': u_id,
            'name': name,
            'department': dept,
            'year': year,
            'email': email
        })

    users_df = pd.DataFrame(users)

    # 3. Generate Transactions using Power-Law Popularity & User Borrowing Probabilities
    # Book selection probability: Power-law (Pareto distribution)
    book_weights = np.random.pareto(a=1.5, size=num_books) + 0.1
    book_probs = book_weights / np.sum(book_weights)
    book_id_pool = books_df['book_id'].values

    # User activity distribution: Power-law (some heavy readers, many average, some 0/1 cold-start)
    user_weights = np.random.pareto(a=1.2, size=num_users)
    
    # Intentionally zero out ~15% of users to evaluate true Cold-Start performance
    num_cold = int(0.15 * num_users)
    cold_indices = np.random.choice(num_users, size=num_cold, replace=False)
    user_weights[cold_indices] = 0.0

    if np.sum(user_weights) > 0:
        user_probs = user_weights / np.sum(user_weights)
    else:
        user_probs = np.ones(num_users) / num_users

    user_id_pool = users_df['user_id'].values

    transactions = []
    start_date = datetime.date(2025, 9, 1)
    end_date = datetime.date(2026, 6, 30)
    total_days = (end_date - start_date).days

    seen_pairs = set()
    tx_counter = 3001

    while len(transactions) < num_transactions:
        u_id = int(np.random.choice(user_id_pool, p=user_probs))
        b_id = int(np.random.choice(book_id_pool, p=book_probs))

        pair = (u_id, b_id)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        # Generate realistic borrow date
        rand_day = random.randint(0, total_days)
        b_date = start_date + datetime.timedelta(days=rand_day)
        r_date = b_date + datetime.timedelta(days=random.randint(7, 21))

        status = 'Returned' if r_date <= end_date else 'Borrowed'
        r_date_str = r_date.strftime('%Y-%m-%d') if status == 'Returned' else ''

        transactions.append({
            'transaction_id': tx_counter,
            'user_id': u_id,
            'book_id': b_id,
            'borrow_date': b_date.strftime('%Y-%m-%d'),
            'return_date': r_date_str,
            'status': status
        })
        tx_counter += 1

    borrow_df = pd.DataFrame(transactions)
    borrow_df = borrow_df.sort_values(by=['borrow_date', 'transaction_id']).reset_index(drop=True)
    borrow_df['transaction_id'] = range(3001, 3001 + len(borrow_df))

    # 4. Write CSV files
    os.makedirs(DATA_DIR, exist_ok=True)
    books_df.to_csv(SYNTHETIC_BOOKS_CSV, index=False)
    users_df.to_csv(SYNTHETIC_USERS_CSV, index=False)
    borrow_df.to_csv(SYNTHETIC_BORROWING_CSV, index=False)

    if activate_as_main:
        books_df.to_csv(BOOKS_CSV, index=False)
        users_df.to_csv(USERS_CSV, index=False)
        borrow_df.to_csv(BORROWING_CSV, index=False)

    # 5. Create Dataset Documentation
    sparsity_val = 100.0 * (1.0 - len(borrow_df) / (num_books * num_users))
    doc_content = f"""# Synthetic Academic Library Research Dataset Documentation

> **IMPORTANT DECLARATION**: This dataset is **100% SYNTHETIC** and was generated using probabilistic distributions for offline recommendation algorithm evaluation and computational complexity benchmarking. It does **NOT** contain real personal data or actual historical borrowing logs from any academic institution.

## Dataset Summary Statistics
- **Total Books**: {num_books:,}
- **Total Users**: {num_users:,}
- **Total Borrowing Transactions**: {len(borrow_df):,}
- **Random Seed**: {seed}
- **Date Range**: 2025-09-01 to 2026-06-30
- **Interaction Matrix Sparsity**: {sparsity_val:.2f}%

## Generative Probability Distributions
1. **Book Popularity**: Pareto Distribution (alpha = 1.5), modeling power-law preference for core engineering textbooks.
2. **User Borrowing Activity**: Pareto Distribution (alpha = 1.2), featuring ~15% cold-start users (0 loans), sparse readers (1-5 loans), and active scholars (>5 loans).
3. **Referential Integrity**: 100% verified. Every `user_id` in transactions exists in `users.csv`, and every `book_id` exists in `books.csv`.

## File Manifest
- `synthetic_books.csv`: Book catalog metadata
- `synthetic_users.csv`: Anonymized student profiles
- `synthetic_borrowing_history.csv`: Transaction logs with borrow/return dates
"""
    with open(os.path.join(DATA_DIR, 'DATASET_DOCUMENTATION.md'), 'w', encoding='utf-8') as f:
        f.write(doc_content)

    print(f"[OK] Synthetic dataset successfully generated!")
    print(f"   - Books: {len(books_df):,}")
    print(f"   - Users: {len(users_df):,} (with {len(users_df) - len(borrow_df['user_id'].unique())} cold-start users)")
    print(f"   - Transactions: {len(borrow_df):,}")
    return books_df, users_df, borrow_df


if __name__ == '__main__':
    generate_synthetic_dataset()
