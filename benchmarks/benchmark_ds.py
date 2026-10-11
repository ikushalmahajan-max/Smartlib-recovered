"""
Data Structure Empirical Performance Benchmark Module with Multi-Repetition Batch Timing.
Compares Linear Search, Custom Binary Search Tree (BST), and Custom Hash Table (Separate Chaining).
Measures query lookup latency across dataset sizes 100 to 50,000 items.
Executes 5 independent repetitions of 2,000 batch operations (10,000 total operations) per size with initial warm-ups.
Saves results to results/final_ds_benchmark.csv and figure to results/figures/ds_benchmark.png.
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BENCHMARK_SIZES, BENCHMARK_ITERATIONS, RANDOM_SEED, RESULTS_DIR, FIGURES_DIR
from ds.bst import BinarySearchTree
from ds.hash_table import HashTable


def generate_synthetic_books(n, seed=RANDOM_SEED):
    """Generates n synthetic book records with unique IDs and ISBN strings."""
    random.seed(seed)
    books = []
    for i in range(1, n + 1):
        b_id = i + 100
        isbn = f"978-013{i:06d}"
        books.append({
            'book_id': b_id,
            'title': f"Synthetic Engineering Book {i}",
            'author': f"Author {i % 100}",
            'genre': 'Computer Science' if i % 2 == 0 else 'AI/ML',
            'isbn': isbn,
            'rating': round(3.5 + (i % 15) / 10.0, 1)
        })
    return books


def linear_search_by_id(books_list, target_id):
    """Linear search over Python list in O(N) time."""
    for b in books_list:
        if b['book_id'] == target_id:
            return b
    return None


def run_ds_benchmark(sizes=BENCHMARK_SIZES, batch_queries=2000, repetitions=5, seed=RANDOM_SEED, save_results=True):
    """
    Executes multi-repetition batch timing benchmarks for Linear Search, BST Search, and Hash Table Lookup.
    Performs 2,000 queries per batch across 5 measured repetitions (10,000 total operations) with warm-up runs.
    Saves full summary statistics to results/final_ds_benchmark.csv.
    """
    random.seed(seed)
    np.random.seed(seed)

    benchmark_rows = []

    for size in sizes:
        print(f"Benchmarking dataset size N = {size:,} (Batch size: {batch_queries:,} queries x {repetitions} reps)...")
        books = generate_synthetic_books(size, seed=seed)

        # Build Custom BST (with shuffled insertion for average O(log N) tree height) and Hash Table
        bst = BinarySearchTree()
        hash_table = HashTable(capacity=max(64, int(size * 1.5)))

        shuffled_books = books.copy()
        random.shuffle(shuffled_books)

        for b in shuffled_books:
            bst.insert(b['book_id'], b)

        for b in books:
            hash_table.put(b['isbn'], b)

        # Generate query workload
        query_books = random.choices(books, k=batch_queries)
        target_ids = [b['book_id'] for b in query_books]
        target_isbns = [b['isbn'] for b in query_books]

        # Warm-up runs before timing
        for _ in range(100):
            _ = linear_search_by_id(books[:10], target_ids[0])
            _ = bst.search(target_ids[0])
            _ = hash_table.get(target_isbns[0])

        # 1. Benchmark Linear Search across repetitions
        linear_rep_means = []
        for r in range(repetitions):
            t0 = time.perf_counter()
            for t_id in target_ids:
                _ = linear_search_by_id(books, t_id)
            t1 = time.perf_counter()
            batch_ms = (t1 - t0) * 1000.0
            linear_rep_means.append(batch_ms / batch_queries)

        # 2. Benchmark Custom BST Search across repetitions
        bst_rep_means = []
        for r in range(repetitions):
            t0 = time.perf_counter()
            for t_id in target_ids:
                _ = bst.search(t_id)
            t1 = time.perf_counter()
            batch_ms = (t1 - t0) * 1000.0
            bst_rep_means.append(batch_ms / batch_queries)

        # 3. Benchmark Custom Hash Table Lookup across repetitions
        hash_rep_means = []
        for r in range(repetitions):
            t0 = time.perf_counter()
            for t_isbn in target_isbns:
                _ = hash_table.get(t_isbn)
            t1 = time.perf_counter()
            batch_ms = (t1 - t0) * 1000.0
            hash_rep_means.append(batch_ms / batch_queries)

        # Gather summary statistics across repetitions
        algos = [
            ('Linear Search (List)', linear_rep_means),
            ('Custom BST Search', bst_rep_means),
            ('Custom Hash Table Lookup', hash_rep_means)
        ]

        for algo_name, rep_means in algos:
            arr = np.array(rep_means)
            benchmark_rows.append({
                'Dataset_Size': size,
                'Algorithm': algo_name,
                'Batch_Queries': batch_queries,
                'Repetitions': repetitions,
                'Mean_Time_ms': round(float(np.mean(arr)), 6),
                'Median_Time_ms': round(float(np.median(arr)), 6),
                'Std_Time_ms': round(float(np.std(arr)), 6),
                'Min_Time_ms': round(float(np.min(arr)), 6),
                'Max_Time_ms': round(float(np.max(arr)), 6)
            })

    results_df = pd.DataFrame(benchmark_rows)

    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        results_df.to_csv(os.path.join(RESULTS_DIR, 'final_ds_benchmark.csv'), index=False)
        results_df.to_csv(os.path.join(RESULTS_DIR, 'ds_benchmark.csv'), index=False)

        # Plot DS Benchmark figure
        plt.figure(figsize=(9, 5))
        for algo in results_df['Algorithm'].unique():
            sub = results_df[results_df['Algorithm'] == algo]
            plt.plot(sub['Dataset_Size'], sub['Mean_Time_ms'], marker='o', label=algo, linewidth=2)

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Dataset Size (N items)')
        plt.ylabel('Mean Lookup Latency (ms per op, log scale)')
        plt.title('Data Structure Search Latency Comparison Across Scaled Datasets')
        plt.legend()
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'ds_benchmark.png'), dpi=300)
        plt.close()

    return results_df


if __name__ == '__main__':
    print("=== RUNNING MULTI-REPETITION DATA STRUCTURE PERFORMANCE BENCHMARK ===")
    df = run_ds_benchmark()
    print("\n--- BENCHMARK RESULTS SUMMARY ---")
    print(df.to_string(index=False))
