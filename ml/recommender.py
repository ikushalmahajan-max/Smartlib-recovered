"""
AI/ML Recommendation Engine.
Provides Content-Based Filtering (TF-IDF + Cosine Similarity), Collaborative Filtering (User-User Cosine Similarity),
Rating-Based Baseline, and Hybrid Recommender with full explainability (scores & reasons) and strict user_id tracking.
Optimized with vectorized NumPy matrix operations for high-speed offline evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import DEFAULT_HYBRID_WEIGHTS


class LibraryRecommender:
    def __init__(self, books_df, users_df, borrowing_df):
        self.books_df = books_df.copy()
        self.users_df = users_df.copy()
        self.borrowing_df = borrowing_df.copy()

        # Ensure numeric primary & foreign key types
        self.books_df['book_id'] = self.books_df['book_id'].astype(int)
        self.users_df['user_id'] = self.users_df['user_id'].astype(int)
        self.borrowing_df['user_id'] = self.borrowing_df['user_id'].astype(int)
        self.borrowing_df['book_id'] = self.borrowing_df['book_id'].astype(int)

        self.book_id_to_idx = {b_id: idx for idx, b_id in enumerate(self.books_df['book_id'])}
        self.idx_to_book_id = {idx: b_id for idx, b_id in enumerate(self.books_df['book_id'])}

        self._build_content_model()
        self._build_collaborative_model()

    def _build_content_model(self):
        """Build TF-IDF Vectorizer and Cosine Similarity Matrix for book metadata."""
        self.books_df['content_features'] = (
            self.books_df['genre'].fillna('') + " " +
            self.books_df['author'].fillna('') + " " +
            self.books_df['description'].fillna('')
        )
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.books_df['content_features'])
        self.content_sim_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def _build_collaborative_model(self):
        """Build User-Item Borrowing Matrix and calculate User-User Cosine Similarity."""
        if not self.borrowing_df.empty:
            user_item = self.borrowing_df.groupby(['user_id', 'book_id']).size().unstack(fill_value=0)
            user_item = (user_item > 0).astype(int)
            self.user_item_matrix = user_item

            if user_item.shape[0] > 1:
                user_sim = cosine_similarity(user_item)
                self.user_sim_df = pd.DataFrame(
                    user_sim,
                    index=user_item.index,
                    columns=user_item.index
                )
            else:
                self.user_sim_df = pd.DataFrame()
        else:
            self.user_item_matrix = pd.DataFrame()
            self.user_sim_df = pd.DataFrame()

    def get_borrowed_books_for_user(self, user_id):
        """Get list of book IDs borrowed by a given user_id."""
        user_history = self.borrowing_df[self.borrowing_df['user_id'] == int(user_id)]
        return list(user_history['book_id'].unique())

    def recommend_popularity(self, user_id, top_n=5):
        """Popularity-based baseline recommender."""
        user_id = int(user_id)
        borrowed_ids = self.get_borrowed_books_for_user(user_id)
        candidate_books = self.books_df[~self.books_df['book_id'].isin(borrowed_ids)].copy()

        if candidate_books.empty:
            candidate_books = self.books_df.copy()

        candidate_books['content_score'] = 0.0
        candidate_books['collab_score'] = 0.0
        candidate_books['hybrid_score'] = (candidate_books['rating'] / 5.0).round(3)
        candidate_books['recommendation_score'] = candidate_books['hybrid_score']
        candidate_books['reason'] = "Highest overall reader rating (Rating-Based Baseline)"

        top_df = candidate_books.sort_values(by=['rating', 'publication_year'], ascending=False).head(top_n)
        return top_df[['book_id', 'title', 'author', 'genre', 'rating', 'content_score', 'collab_score', 'hybrid_score', 'recommendation_score', 'reason']]

    def recommend_content_only(self, user_id, top_n=5):
        """Content-only TF-IDF recommender (weights 1.0 content, 0.0 collab)."""
        return self.recommend_books_for_user(user_id, top_n=top_n, content_weight=1.0, collab_weight=0.0)

    def recommend_collaborative_only(self, user_id, top_n=5):
        """Collaborative-only user-user recommender (weights 0.0 content, 1.0 collab)."""
        return self.recommend_books_for_user(user_id, top_n=top_n, content_weight=0.0, collab_weight=1.0)

    def recommend_books_for_user(self, user_id, top_n=5, content_weight=0.6, collab_weight=0.4):
        """
        Flexible Hybrid Recommendation Function with Explainability.
        Evaluates candidate books based on user borrowing history, content similarity, and user-user similarity.
        Optimized with NumPy vectorization.
        """
        user_id = int(user_id)
        borrowed_ids = self.get_borrowed_books_for_user(user_id)
        all_book_ids = list(self.books_df['book_id'].unique())
        candidate_ids = [b_id for b_id in all_book_ids if b_id not in borrowed_ids]

        if not candidate_ids:
            candidate_ids = all_book_ids

        # Cold start case: no borrow history
        if not borrowed_ids:
            top_books = self.books_df[self.books_df['book_id'].isin(candidate_ids)].sort_values(
                by=['rating', 'publication_year'], ascending=False
            ).head(top_n).copy()
            top_books['content_score'] = 0.0
            top_books['collab_score'] = 0.0
            top_books['hybrid_score'] = (top_books['rating'] / 5.0).round(3)
            top_books['recommendation_score'] = top_books['hybrid_score']
            top_books['reason'] = "Popular & High Rated Book (Cold-Start Default)"
            return top_books[['book_id', 'title', 'author', 'genre', 'rating', 'content_score', 'collab_score', 'hybrid_score', 'recommendation_score', 'reason']]

        # 1. Collaborative Filtering Scores (Vectorized)
        collab_scores = np.zeros(len(candidate_ids), dtype=float)
        if user_id in self.user_sim_df.index and not self.user_item_matrix.empty:
            user_sim_series = self.user_sim_df[user_id]
            # Get similar users with positive similarity
            sim_users = user_sim_series[user_sim_series > 0].drop(labels=[user_id], errors='ignore')

            if not sim_users.empty:
                # Vectorized dot product of user similarity vector and user-item matrix
                valid_sim_users = [u for u in sim_users.index if u in self.user_item_matrix.index]
                if valid_sim_users:
                    sub_item_matrix = self.user_item_matrix.loc[valid_sim_users]
                    sim_weights = sim_users.loc[valid_sim_users].values
                    item_sums = np.dot(sim_weights, sub_item_matrix.values) # (num_items,)
                    item_id_map = {b_id: idx for idx, b_id in enumerate(self.user_item_matrix.columns)}

                    for i, cand_id in enumerate(candidate_ids):
                        if cand_id in item_id_map:
                            collab_scores[i] = item_sums[item_id_map[cand_id]]

        max_collab = np.max(collab_scores) if len(collab_scores) > 0 and np.max(collab_scores) > 0 else 1.0
        norm_collab_scores = collab_scores / max_collab

        # 2. Content-Based Filtering Scores (Vectorized)
        content_scores = np.zeros(len(candidate_ids), dtype=float)
        borrowed_indices = [self.book_id_to_idx[b_id] for b_id in borrowed_ids if b_id in self.book_id_to_idx]
        candidate_indices = [self.book_id_to_idx[b_id] for b_id in candidate_ids if b_id in self.book_id_to_idx]

        if borrowed_indices and candidate_indices:
            sub_sim = self.content_sim_matrix[np.ix_(borrowed_indices, candidate_indices)]
            content_scores = np.max(sub_sim, axis=0) # max over borrowed books

        # 3. Hybrid Weight Combination
        total_weight = content_weight + collab_weight
        if total_weight > 0:
            w_c = content_weight / total_weight
            w_col = collab_weight / total_weight
        else:
            w_c, w_col = 0.5, 0.5

        final_hybrid_scores = w_c * content_scores + w_col * norm_collab_scores

        records = []
        for i, b_id in enumerate(candidate_ids):
            records.append({
                'book_id': b_id,
                'content_score': round(float(content_scores[i]), 4),
                'collab_score': round(float(norm_collab_scores[i]), 4),
                'hybrid_score': round(float(final_hybrid_scores[i]), 4)
            })

        scores_df = pd.DataFrame(records)
        scores_df = scores_df.sort_values(by='hybrid_score', ascending=False).head(top_n)

        result_df = pd.merge(scores_df, self.books_df, on='book_id')
        result_df['recommendation_score'] = result_df['hybrid_score']

        def get_reason(row):
            if row['collab_score'] > 0.3 and w_col > 0:
                return "Borrowed by readers with similar borrowing history"
            elif row['content_score'] > 0.3 and w_c > 0:
                return "Similar content/genre to books in your reading profile"
            elif row['rating'] >= 4.7:
                return "High rating across departmental readers"
            else:
                return "Matches your reading domain interests"

        result_df['reason'] = result_df.apply(get_reason, axis=1)
        return result_df[['book_id', 'title', 'author', 'genre', 'rating', 'content_score', 'collab_score', 'hybrid_score', 'recommendation_score', 'reason']]

    def get_similar_books(self, book_id, top_n=4):
        """Get books similar to a specific book using content similarity."""
        if book_id not in self.book_id_to_idx:
            return pd.DataFrame()

        target_idx = self.book_id_to_idx[book_id]
        sim_scores = list(enumerate(self.content_sim_matrix[target_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [item for item in sim_scores if item[0] != target_idx][:top_n]

        similar_indices = [item[0] for item in sim_scores]
        scores = [item[1] for item in sim_scores]

        result = self.books_df.iloc[similar_indices].copy()
        result['similarity_score'] = np.round(scores, 3)
        return result[['book_id', 'title', 'author', 'genre', 'rating', 'similarity_score']]
