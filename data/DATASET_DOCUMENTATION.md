# Synthetic Academic Library Research Dataset Documentation

> **IMPORTANT DECLARATION**: This dataset is **100% SYNTHETIC** and was generated using probabilistic distributions for offline recommendation algorithm evaluation and computational complexity benchmarking. It does **NOT** contain real personal data or actual historical borrowing logs from any academic institution.

## Dataset Summary Statistics
- **Total Books**: 750
- **Total Users**: 250
- **Total Borrowing Transactions**: 4,000
- **Random Seed**: 42
- **Date Range**: 2025-09-01 to 2026-06-30
- **Interaction Matrix Sparsity**: 97.87%

## Generative Probability Distributions
1. **Book Popularity**: Pareto Distribution (alpha = 1.5), modeling power-law preference for core engineering textbooks.
2. **User Borrowing Activity**: Pareto Distribution (alpha = 1.2), featuring ~15% cold-start users (0 loans), sparse readers (1-5 loans), and active scholars (>5 loans).
3. **Referential Integrity**: 100% verified. Every `user_id` in transactions exists in `users.csv`, and every `book_id` exists in `books.csv`.

## File Manifest
- `synthetic_books.csv`: Book catalog metadata
- `synthetic_users.csv`: Anonymized student profiles
- `synthetic_borrowing_history.csv`: Transaction logs with borrow/return dates
