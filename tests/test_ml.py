"""
Unit tests for AI/ML Recommendation Engine and Student Performance Scorer.
Compatible with standard library unittest and pytest.
"""

import unittest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml import LibraryRecommender, StudentPerformanceAnalyzer
from ml.validate_dataset import get_cleaned_datasets


class TestMLRecommender(unittest.TestCase):

    def setUp(self):
        self.users_df, self.books_df, self.borrow_df = get_cleaned_datasets()
        self.recommender = LibraryRecommender(self.books_df, self.users_df, self.borrow_df)
        self.scorer = StudentPerformanceAnalyzer(self.books_df, self.users_df, self.borrow_df)

    def test_recommendation_output(self):
        # Test for existing user 201
        recs = self.recommender.recommend_books_for_user(201, top_n=5)
        self.assertEqual(len(recs), 5)
        self.assertIn('book_id', recs.columns)
        self.assertIn('content_score', recs.columns)
        self.assertIn('collab_score', recs.columns)
        self.assertIn('hybrid_score', recs.columns)
        self.assertIn('recommendation_score', recs.columns)
        self.assertIn('reason', recs.columns)

        # Ensure already borrowed books are excluded
        borrowed = self.recommender.get_borrowed_books_for_user(201)
        rec_ids = list(recs['book_id'])
        for b_id in borrowed:
            self.assertNotIn(b_id, rec_ids)

    def test_cold_start_recommendation(self):
        # User 999 does not exist in borrowing history
        recs = self.recommender.recommend_books_for_user(999, top_n=3)
        self.assertEqual(len(recs), 3)
        reason_first = recs.iloc[0]['reason']
        self.assertTrue("Popular" in reason_first or "Cold-Start" in reason_first or "High Rated" in reason_first)

    def test_similar_books(self):
        sims = self.recommender.get_similar_books(103, top_n=3)
        self.assertEqual(len(sims), 3)
        self.assertIn('similarity_score', sims.columns)

    def test_student_performance_scoring(self):
        # Test score calculation for user 201 (active reader)
        score_res = self.scorer.calculate_student_score(201)
        self.assertIn('total_score', score_res)
        self.assertIn('tier', score_res)
        self.assertIn('selection_score', score_res)
        self.assertIn('timing_score', score_res)
        self.assertTrue(0.0 <= score_res['total_score'] <= 100.0)

        # Leaderboard test
        leaderboard = self.scorer.get_all_students_leaderboard()
        self.assertGreater(len(leaderboard), 0)
        self.assertIn('total_score', leaderboard.columns)


if __name__ == '__main__':
    unittest.main()
