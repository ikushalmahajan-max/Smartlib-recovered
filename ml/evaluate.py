"""
Offline Recommendation System Evaluation, 3-Way Temporal Split, Weight Selection, and Cold-Start Module.

Evaluates Models via a Research-Grade 3-Way Temporal Split:
  TRAIN -> VALIDATION -> FINAL TEST

1. Train set builds the recommender model.
2. Validation set is used ONLY to select optimal hybrid weights based on Validation NDCG@5.
3. Final Test set is evaluated EXACTLY ONCE using the frozen selected model and baselines.

Prevents data leakage via strict temporal/user split protocols.
Generates evaluation CSVs and figures in results/ and results/figures/.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TOP_K, DEFAULT_HYBRID_WEIGHTS, WEIGHT_ABLATION_WEIGHTS, RESULTS_DIR, FIGURES_DIR
)
from ml.validate_dataset import get_cleaned_datasets
from ml.recommender import LibraryRecommender


def split_borrowing_data_3way(borrow_df, val_ratio=0.15, test_ratio=0.15):
    """
    Splits borrowing history into Train, Validation, and Final Test sets using a 3-way temporal timeline split.
    For users with multiple transactions:
      - Earliest transactions -> Train set
      - Middle transactions  -> Validation set (for hyperparameter tuning)
      - Latest transactions  -> Final Test set (evaluated ONCE)
    """
    df_sorted = borrow_df.sort_values(by=['borrow_date', 'transaction_id']).copy()

    train_records = []
    val_records = []
    test_records = []

    for user_id, group in df_sorted.groupby('user_id'):
        n = len(group)
        if n == 1:
            train_records.append(group)
        elif n == 2:
            train_records.append(group.iloc[:1])
            val_records.append(group.iloc[1:])
        else:
            n_val = max(1, int(np.round(n * val_ratio)))
            n_test = max(1, int(np.round(n * test_ratio)))

            # Prevent val + test taking entire group
            if n_val + n_test >= n:
                n_val = 1
                n_test = 1

            train_g = group.iloc[:-n_val - n_test]
            val_g = group.iloc[-n_val - n_test : -n_test]
            test_g = group.iloc[-n_test:]

            train_records.append(train_g)
            val_records.append(val_g)
            test_records.append(test_g)

    train_df = pd.concat(train_records, ignore_index=True) if train_records else pd.DataFrame()
    val_df = pd.concat(val_records, ignore_index=True) if val_records else pd.DataFrame()
    test_df = pd.concat(test_records, ignore_index=True) if test_records else pd.DataFrame()

    return train_df, val_df, test_df


def verify_no_data_leakage_3way(train_df, val_df, test_df):
    """
    Verifies 3-way temporal data leakage invariants:
    1. No transaction_id overlap between train, val, and test sets.
    2. Chronological order invariant per user:
       - train interaction date <= val interaction date
       - val interaction date <= test interaction date
       - train interaction date <= test interaction date
    """
    train_tx = set(train_df['transaction_id'])
    val_tx = set(val_df['transaction_id'])
    test_tx = set(test_df['transaction_id'])

    if train_tx.intersection(val_tx):
        raise ValueError("DATA LEAKAGE DETECTED: Overlap between Train and Validation transaction IDs!")

    if train_tx.intersection(test_tx):
        raise ValueError("DATA LEAKAGE DETECTED: Overlap between Train and Test transaction IDs!")

    if val_tx.intersection(test_tx):
        raise ValueError("DATA LEAKAGE DETECTED: Overlap between Validation and Test transaction IDs!")

    # Per-user chronological order verification
    all_users = set(train_df['user_id']).union(val_df['user_id']).union(test_df['user_id'])
    for u_id in all_users:
        u_train = train_df[train_df['user_id'] == u_id]
        u_val = val_df[val_df['user_id'] == u_id]
        u_test = test_df[test_df['user_id'] == u_id]

        max_train = u_train['borrow_date'].max() if not u_train.empty else None
        min_val = u_val['borrow_date'].min() if not u_val.empty else None
        max_val = u_val['borrow_date'].max() if not u_val.empty else None
        min_test = u_test['borrow_date'].min() if not u_test.empty else None

        if max_train is not None and min_val is not None and max_train > min_val:
            raise ValueError(f"DATA LEAKAGE DETECTED: User {u_id} has train date ({max_train}) > validation date ({min_val})!")

        if max_val is not None and min_test is not None and max_val > min_test:
            raise ValueError(f"DATA LEAKAGE DETECTED: User {u_id} has validation date ({max_val}) > test date ({min_test})!")

        if max_train is not None and min_test is not None and max_val is None and max_train > min_test:
            raise ValueError(f"DATA LEAKAGE DETECTED: User {u_id} has train date ({max_train}) > test date ({min_test})!")

    return True


def calculate_user_metrics(recommended_ids, actual_ids, k=TOP_K):
    """
    Calculates Precision@K, Recall@K, F1@K, and NDCG@K for a single user recommendation.
    """
    rec_k = recommended_ids[:k]
    actual_set = set(actual_ids)

    if not actual_set:
        return 0.0, 0.0, 0.0, 0.0

    hits = sum([1 for b_id in rec_k if b_id in actual_set])
    precision = hits / k
    recall = hits / len(actual_set)

    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    # DCG@K
    dcg = 0.0
    for i, b_id in enumerate(rec_k):
        if b_id in actual_set:
            dcg += 1.0 / np.log2(i + 2)

    # IDCG@K
    idcg = 0.0
    for i in range(min(len(actual_set), k)):
        idcg += 1.0 / np.log2(i + 2)

    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return precision, recall, f1, ndcg


def evaluate_recommender_on_eval_set(recommender, eval_target_df, target_user_ids, k=TOP_K, model_type='hybrid', content_w=0.6, collab_w=0.4):
    """
    Evaluates recommender over a target set of users having target items in eval_target_df.
    Reports Total Users, Eligible Users, Evaluated Users, and Metric Averages.
    """
    # Filter target users to eligible users with actual items in target evaluation set
    eval_user_targets = eval_target_df.groupby('user_id')['book_id'].apply(list).to_dict()
    eligible_users = [u for u in target_user_ids if u in eval_user_targets and len(eval_user_targets[u]) > 0]

    precisions, recalls, f1s, ndcgs = [], [], [], []

    for u_id in eligible_users:
        actual_ids = eval_user_targets[u_id]

        if model_type == 'popularity':
            rec_df = recommender.recommend_popularity(u_id, top_n=k)
        elif model_type == 'content_only':
            rec_df = recommender.recommend_content_only(u_id, top_n=k)
        elif model_type == 'collaborative_only':
            rec_df = recommender.recommend_collaborative_only(u_id, top_n=k)
        else:
            rec_df = recommender.recommend_books_for_user(u_id, top_n=k, content_weight=content_w, collab_weight=collab_w)

        rec_ids = rec_df['book_id'].tolist() if not rec_df.empty else []
        p, r, f1, n = calculate_user_metrics(rec_ids, actual_ids, k=k)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        ndcgs.append(n)

    metrics = {
        'total_target_users': len(target_user_ids),
        'eligible_users': len(eligible_users),
        'evaluated_users': len(precisions),
        f'Precision@{k}': round(float(np.mean(precisions)) if precisions else 0.0, 4),
        f'Recall@{k}': round(float(np.mean(recalls)) if recalls else 0.0, 4),
        f'F1@{k}': round(float(np.mean(f1s)) if f1s else 0.0, 4),
        f'NDCG@{k}': round(float(np.mean(ndcgs)) if ndcgs else 0.0, 4)
    }

    return metrics


def run_validation_weight_selection(k=TOP_K, save_results=True):
    """
    Evaluates content/collaborative weights on the VALIDATION set to select the optimal weight ratio.
    Saves validation selection results to results/final_weight_selection.csv.
    """
    users_df, books_df, borrow_df = get_cleaned_datasets()
    train_df, val_df, test_df = split_borrowing_data_3way(borrow_df)
    verify_no_data_leakage_3way(train_df, val_df, test_df)

    recommender = LibraryRecommender(books_df, users_df, train_df)
    val_users = val_df['user_id'].unique()

    ablation_results = []
    for content_w, collab_w in WEIGHT_ABLATION_WEIGHTS:
        res = evaluate_recommender_on_eval_set(
            recommender, val_df, val_users, k=k, model_type='hybrid', content_w=content_w, collab_w=collab_w
        )
        row = {
            'Content_Weight': content_w,
            'Collaborative_Weight': collab_w,
            'Weight_Ratio': f"{content_w:.1f}/{collab_w:.1f}",
            'Validation_Eligible_Users': res['eligible_users'],
            f'Precision@{k}': res[f'Precision@{k}'],
            f'Recall@{k}': res[f'Recall@{k}'],
            f'F1@{k}': res[f'F1@{k}'],
            f'NDCG@{k}': res[f'NDCG@{k}']
        }
        ablation_results.append(row)

    val_df_results = pd.DataFrame(ablation_results)

    # Select best weights based on Validation NDCG@K
    best_row = val_df_results.sort_values(by=[f'NDCG@{k}', f'F1@{k}'], ascending=False).iloc[0]
    best_content_w = float(best_row['Content_Weight'])
    best_collab_w = float(best_row['Collaborative_Weight'])

    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        val_df_results.to_csv(os.path.join(RESULTS_DIR, 'final_weight_selection.csv'), index=False)
        val_df_results.to_csv(os.path.join(RESULTS_DIR, 'weight_ablation.csv'), index=False)

        # Plot Weight Ablation Figure
        plt.figure(figsize=(8, 4.5))
        ratios = val_df_results['Weight_Ratio']
        plt.plot(ratios, val_df_results[f'Precision@{k}'], marker='o', label=f'Validation Precision@{k}', color='#3B82F6', linewidth=2)
        plt.plot(ratios, val_df_results[f'Recall@{k}'], marker='s', label=f'Validation Recall@{k}', color='#10B981', linewidth=2)
        plt.plot(ratios, val_df_results[f'F1@{k}'], marker='^', label=f'Validation F1@{k}', color='#F59E0B', linewidth=2)
        plt.plot(ratios, val_df_results[f'NDCG@{k}'], marker='d', label=f'Validation NDCG@{k}', color='#8B5CF6', linewidth=2)

        plt.axvline(x=best_row['Weight_Ratio'], color='#DC2626', linestyle='--', label=f'Selected Best ({best_row["Weight_Ratio"]})')
        plt.xlabel('Content / Collaborative Weight Ratio (Validation Set)')
        plt.ylabel('Metric Score')
        plt.title('Validation Set Hybrid Weight Selection Curve')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'weight_ablation.png'), dpi=300)
        plt.close()

    return val_df_results, best_content_w, best_collab_w


def run_final_test_evaluation(best_content_w, best_collab_w, k=TOP_K, save_results=True):
    """
    Evaluates the frozen selected hybrid model and baselines EXACTLY ONCE on the FINAL TEST set.
    Saves final comparison to results/final_model_comparison.csv.
    """
    users_df, books_df, borrow_df = get_cleaned_datasets()
    train_df, val_df, test_df = split_borrowing_data_3way(borrow_df)

    recommender = LibraryRecommender(books_df, users_df, train_df)
    test_users = test_df['user_id'].unique()

    models_to_eval = [
        ('Rating-Based Baseline', 'popularity', 0.0, 0.0),
        ('Content-Only (TF-IDF)', 'content_only', 1.0, 0.0),
        ('Collaborative-Only (User-User)', 'collaborative_only', 0.0, 1.0),
        (f'Selected Hybrid ({best_content_w:.1f}/{best_collab_w:.1f})', 'hybrid', best_content_w, best_collab_w)
    ]

    results = []
    for name, m_type, cw, colw in models_to_eval:
        res = evaluate_recommender_on_eval_set(recommender, test_df, test_users, k=k, model_type=m_type, content_w=cw, collab_w=colw)
        row = {
            'Model': name,
            'Total_Test_Users': res['total_target_users'],
            'Eligible_Test_Users': res['eligible_users'],
            'Evaluated_Users': res['evaluated_users'],
            f'Precision@{k}': res[f'Precision@{k}'],
            f'Recall@{k}': res[f'Recall@{k}'],
            f'F1@{k}': res[f'F1@{k}'],
            f'NDCG@{k}': res[f'NDCG@{k}']
        }
        results.append(row)

    final_df = pd.DataFrame(results)

    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        final_df.to_csv(os.path.join(RESULTS_DIR, 'final_model_comparison.csv'), index=False)
        final_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)
        final_df.to_csv(os.path.join(RESULTS_DIR, 'recommendation_evaluation.csv'), index=False)

        # Plot Final Model Comparison
        plt.figure(figsize=(9, 5))
        models = final_df['Model']
        x = np.arange(len(models))
        width = 0.2

        plt.bar(x - 1.5*width, final_df[f'Precision@{k}'], width, label=f'Precision@{k}', color='#3B82F6')
        plt.bar(x - 0.5*width, final_df[f'Recall@{k}'], width, label=f'Recall@{k}', color='#10B981')
        plt.bar(x + 0.5*width, final_df[f'F1@{k}'], width, label=f'F1@{k}', color='#F59E0B')
        plt.bar(x + 1.5*width, final_df[f'NDCG@{k}'], width, label=f'NDCG@{k}', color='#8B5CF6')

        plt.xlabel('Models Evaluated on Final Test Set')
        plt.ylabel('Metric Score')
        plt.title(f'Final Test Set Recommendation Model Comparison (K={k})')
        plt.xticks(x, models, rotation=15)
        plt.ylim(0.0, max(0.25, float(final_df[f'NDCG@{k}'].max()) * 1.25))
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'model_comparison.png'), dpi=300)
        plt.close()

    return final_df


def run_final_cold_start_evaluation(best_content_w, best_collab_w, k=TOP_K, save_results=True):
    """
    Evaluates the selected hybrid model across Cold-Start, Sparse, and Normal user tiers on the Final Test set.
    Explicitly reports Total Users, Eligible Users, Evaluated Users, and Metrics.
    Saves to results/final_cold_start_results.csv.
    """
    users_df, books_df, borrow_df = get_cleaned_datasets()
    train_df, val_df, test_df = split_borrowing_data_3way(borrow_df)

    recommender = LibraryRecommender(books_df, users_df, train_df)

    # User train loan count
    train_user_counts = train_df.groupby('user_id').size().to_dict()
    all_users = users_df['user_id'].unique()

    cold_users = [u for u in all_users if train_user_counts.get(u, 0) <= 1]
    sparse_users = [u for u in all_users if 2 <= train_user_counts.get(u, 0) <= 5]
    normal_users = [u for u in all_users if train_user_counts.get(u, 0) > 5]

    user_groups = [
        ('Cold-Start Users (<=1 loan)', cold_users),
        ('Sparse-History Users (2-5 loans)', sparse_users),
        ('Normal-History Users (>5 loans)', normal_users)
    ]

    results = []
    for g_name, u_list in user_groups:
        res = evaluate_recommender_on_eval_set(
            recommender, test_df, u_list, k=k, model_type='hybrid', content_w=best_content_w, collab_w=best_collab_w
        )
        row = {
            'User_Group': g_name,
            'Total_Group_Users': len(u_list),
            'Eligible_Test_Users': res['eligible_users'],
            'Evaluated_Users': res['evaluated_users'],
            f'Precision@{k}': res[f'Precision@{k}'],
            f'Recall@{k}': res[f'Recall@{k}'],
            f'F1@{k}': res[f'F1@{k}'],
            f'NDCG@{k}': res[f'NDCG@{k}']
        }
        results.append(row)

    cs_df = pd.DataFrame(results)

    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        cs_df.to_csv(os.path.join(RESULTS_DIR, 'final_cold_start_results.csv'), index=False)
        cs_df.to_csv(os.path.join(RESULTS_DIR, 'cold_start_results.csv'), index=False)

        # Plot Cold Start Figure
        plt.figure(figsize=(8, 4.5))
        groups = cs_df['User_Group']
        x = np.arange(len(groups))
        width = 0.2

        plt.bar(x - 1.5*width, cs_df[f'Precision@{k}'], width, label=f'Precision@{k}', color='#3B82F6')
        plt.bar(x - 0.5*width, cs_df[f'Recall@{k}'], width, label=f'Recall@{k}', color='#10B981')
        plt.bar(x + 0.5*width, cs_df[f'F1@{k}'], width, label=f'F1@{k}', color='#F59E0B')
        plt.bar(x + 1.5*width, cs_df[f'NDCG@{k}'], width, label=f'NDCG@{k}', color='#8B5CF6')

        plt.xlabel('User Activity Categories (Final Test Set)')
        plt.ylabel('Metric Score')
        plt.title('Selected Hybrid Recommender Performance Across Activity Tiers')
        plt.xticks(x, groups, rotation=10)
        plt.ylim(0.0, max(0.25, float(cs_df[f'NDCG@{k}'].max()) * 1.25))
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'cold_start.png'), dpi=300)
        plt.close()

    return cs_df


if __name__ == '__main__':
    print("=== STEP 1: VALIDATION SET WEIGHT SELECTION ===")
    val_results, best_cw, best_colw = run_validation_weight_selection()
    print(val_results.to_string(index=False))
    print(f"\n[SELECTED WEIGHT CONFIGURATION]: Content={best_cw:.1f}, Collaborative={best_colw:.1f}")

    print("\n=== STEP 2: FROZEN FINAL TEST SET MODEL EVALUATION ===")
    final_df = run_final_test_evaluation(best_cw, best_colw)
    print(final_df.to_string(index=False))

    print("\n=== STEP 3: FINAL TEST SET COLD-START ANALYSIS ===")
    cs_df = run_final_cold_start_evaluation(best_cw, best_colw)
    print(cs_df.to_string(index=False))
