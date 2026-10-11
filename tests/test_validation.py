"""
Unit tests for Dataset Validation & Data Integrity Audit.
"""

import unittest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.validate_dataset import audit_datasets, get_cleaned_datasets


class TestDatasetValidation(unittest.TestCase):

    def test_dataset_audit(self):
        report = audit_datasets()
        self.assertIn('users_missing_in_users_table', report)
        self.assertIn('duplicate_borrow_records', report)
        self.assertIn('summary', report)

    def test_cleaned_datasets_foreign_key_integrity(self):
        users_df, books_df, borrow_df = get_cleaned_datasets()

        valid_user_ids = set(users_df['user_id'])
        valid_book_ids = set(books_df['book_id'])

        # Check every user_id in borrowing_history exists in users_df
        for u_id in borrow_df['user_id']:
            self.assertIn(u_id, valid_user_ids)

        # Check every book_id in borrowing_history exists in books_df
        for b_id in borrow_df['book_id']:
            self.assertIn(b_id, valid_book_ids)

        # Ensure no duplicate transaction IDs
        self.assertEqual(len(borrow_df['transaction_id']), len(borrow_df['transaction_id'].unique()))


if __name__ == '__main__':
    unittest.main()
