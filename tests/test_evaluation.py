"""
Unit tests for Recommendation Evaluation Metrics, Train/Test Split, and Model Comparisons.
"""

import unittest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.evaluate import (
    calculate_user_metrics,
    split_borrowing_data_3way,
    verify_no_data_leakage_3way,
    run_validation_weight_selection,
    run_final_test_evaluation
)
from ml.validate_dataset import get_cleaned_datasets


class TestEvaluationMetrics(unittest.TestCase):

    def test_calculate_user_metrics_perfect_hit(self):
        # Recommended: [101, 102, 103, 104, 105], Actual: [101, 102]
        recommended = [101, 102, 103, 104, 105]
        actual = [101, 102]
        p, r, f1, ndcg = calculate_user_metrics(recommended, actual, k=5)

        self.assertEqual(p, 0.4)  # 2 hits / 5 recs
        self.assertEqual(r, 1.0)  # 2 hits / 2 actuals
        self.assertAlmostEqual(f1, (2 * 0.4 * 1.0) / 1.4, places=4)
        self.assertEqual(ndcg, 1.0)  # Perfect rank at top 2 positions

    def test_calculate_user_metrics_no_hits(self):
        recommended = [101, 102, 103, 104, 105]
        actual = [999, 888]
        p, r, f1, ndcg = calculate_user_metrics(recommended, actual, k=5)

        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)
        self.assertEqual(f1, 0.0)
        self.assertEqual(ndcg, 0.0)

    def test_3way_temporal_train_val_test_split(self):
        _, _, borrow_df = get_cleaned_datasets()
        train_df, val_df, test_df = split_borrowing_data_3way(borrow_df, val_ratio=0.15, test_ratio=0.15)

        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(val_df), 0)
        self.assertGreater(len(test_df), 0)
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(borrow_df))

        # Check temporal order: train <= val <= test per user
        for u_id in test_df['user_id'].unique():
            user_train = train_df[train_df['user_id'] == u_id]
            user_val = val_df[val_df['user_id'] == u_id]
            user_test = test_df[test_df['user_id'] == u_id]
            
            if not user_train.empty and not user_val.empty:
                self.assertGreaterEqual(user_val['borrow_date'].min(), user_train['borrow_date'].max())
            if not user_val.empty and not user_test.empty:
                self.assertGreaterEqual(user_test['borrow_date'].min(), user_val['borrow_date'].max())

    def test_verify_no_data_leakage_3way(self):
        _, _, borrow_df = get_cleaned_datasets()
        train_df, val_df, test_df = split_borrowing_data_3way(borrow_df, val_ratio=0.15, test_ratio=0.15)
        self.assertTrue(verify_no_data_leakage_3way(train_df, val_df, test_df))

        # Test transaction overlap detection
        leaky_val = pd.concat([val_df, train_df.iloc[:1]], ignore_index=True)
        with self.assertRaises(ValueError):
            verify_no_data_leakage_3way(train_df, leaky_val, test_df)

    def test_full_evaluation_pipeline(self):
        val_df, best_cw, best_colw = run_validation_weight_selection(save_results=False)
        self.assertGreater(len(val_df), 0)
        self.assertIn('Validation_Eligible_Users', val_df.columns)

        eval_df = run_final_test_evaluation(best_cw, best_colw, save_results=False)
        self.assertEqual(len(eval_df), 4)
        self.assertIn('Precision@5', eval_df.columns)
        self.assertIn('Recall@5', eval_df.columns)
        self.assertIn('F1@5', eval_df.columns)
        self.assertIn('NDCG@5', eval_df.columns)


if __name__ == '__main__':
    unittest.main()
