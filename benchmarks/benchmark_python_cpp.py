"""
Python vs C++ Data Structures Comparative Benchmark Module.
Executes identical search & lookup workloads across custom Python and C++ implementations.
Saves comparison results to results/python_cpp_benchmark.csv and figure to results/figures/python_cpp_comparison.png.
"""

import os
import sys
import time
import random
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BENCHMARK_SIZES, BENCHMARK_ITERATIONS, RANDOM_SEED, RESULTS_DIR, FIGURES_DIR
from ds.bst import BinarySearchTree
from ds.hash_table import HashTable
from benchmarks.benchmark_ds import generate_synthetic_books


def check_cpp_compiler():
    """Checks if g++ or clang++ or cl compiler is available in PATH."""
    for comp in ['g++', 'clang++', 'cl']:
        try:
            res = subprocess.run([comp, '--version' if comp != 'cl' else '/?'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0:
                return comp
        except Exception:
            continue
    return None


def compile_cpp_benchmark():
    """Compiles cpp_engine/benchmark.cpp if compiler is available."""
    cpp_source = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cpp_engine', 'benchmark.cpp')
    exe_name = 'benchmark_cpp.exe' if os.name == 'nt' else 'benchmark_cpp'
    exe_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cpp_engine', exe_name)

    if not os.path.exists(cpp_source):
        return None

    compiler = check_cpp_compiler()
    if compiler in ['g++', 'clang++']:
        try:
            cmd = [compiler, '-O3', cpp_source, '-o', exe_path]
            subprocess.run(cmd, check=True)
            return exe_path
        except Exception as e:
            print(f"Warning: Failed to compile C++ benchmark: {e}")
            return None
    return None


def benchmark_python_ops(size, num_queries=BENCHMARK_ITERATIONS, repetitions=5, seed=RANDOM_SEED):
    """Measures Python BST Search and Hash Table Lookup latency across actual repeated timing runs."""
    random.seed(seed)
    books = generate_synthetic_books(size, seed=seed)

    bst = BinarySearchTree()
    hash_table = HashTable(capacity=max(64, int(size * 1.5)))

    shuffled = books.copy()
    random.shuffle(shuffled)

    for b in shuffled:
        bst.insert(b['book_id'], b)

    for b in books:
        hash_table.put(b['isbn'], b)

    query_books = random.choices(books, k=num_queries)
    target_ids = [b['book_id'] for b in query_books]
    target_isbns = [b['isbn'] for b in query_books]

    # Warm-up runs
    for _ in range(50):
        _ = bst.search(target_ids[0])
        _ = hash_table.get(target_isbns[0])

    bst_times = []
    hash_times = []

    for rep in range(repetitions):
        # Python BST Search
        t0 = time.perf_counter()
        for t_id in target_ids:
            _ = bst.search(t_id)
        t1 = time.perf_counter()
        bst_times.append(((t1 - t0) * 1000.0) / num_queries)

        # Python Hash Lookup
        t0 = time.perf_counter()
        for t_isbn in target_isbns:
            _ = hash_table.get(t_isbn)
        t1 = time.perf_counter()
        hash_times.append(((t1 - t0) * 1000.0) / num_queries)

    return {
        'bst_mean': float(np.mean(bst_times)),
        'bst_std': float(np.std(bst_times)),
        'bst_min': float(np.min(bst_times)),
        'bst_max': float(np.max(bst_times)),
        'hash_mean': float(np.mean(hash_times)),
        'hash_std': float(np.std(hash_times)),
        'hash_min': float(np.min(hash_times)),
        'hash_max': float(np.max(hash_times))
    }


def run_python_vs_cpp_benchmark(sizes=[100, 1000, 10000, 50000], num_queries=BENCHMARK_ITERATIONS, repetitions=5, save_results=True):
    """
    Executes comparative benchmarks between Python and C++ implementations using actual timing repetitions.
    If C++ compiler is unavailable, logs a clear warning and omits fake C++ estimates.
    Saves results to results/python_cpp_benchmark.csv and results/figures/python_cpp_comparison.png.
    """
    exe_path = compile_cpp_benchmark()
    has_compiled_cpp = exe_path is not None and os.path.exists(exe_path)

    if not has_compiled_cpp:
        print("\n[WARNING] C++ compiler (g++/MSVC) is not available in PATH. C++ benchmark skipped. No estimated C++ measurements generated.\n")

    rows = []

    for size in sizes:
        print(f"Benchmarking N = {size:,} (Batch size: {num_queries:,} queries x {repetitions} reps)...")

        # Measured Python Benchmarks
        py_res = benchmark_python_ops(size, num_queries=num_queries, repetitions=repetitions)

        rows.append({
            'Language': 'Python',
            'Operation': 'BST Search',
            'Dataset_Size': size,
            'Num_Operations': num_queries,
            'Repetitions': repetitions,
            'Mean_Time_ms': round(py_res['bst_mean'], 6),
            'Std_Dev_ms': round(py_res['bst_std'], 6),
            'Min_Time_ms': round(py_res['bst_min'], 6),
            'Max_Time_ms': round(py_res['bst_max'], 6),
            'Execution_Status': 'Measured'
        })

        rows.append({
            'Language': 'Python',
            'Operation': 'Hash Lookup',
            'Dataset_Size': size,
            'Num_Operations': num_queries,
            'Repetitions': repetitions,
            'Mean_Time_ms': round(py_res['hash_mean'], 6),
            'Std_Dev_ms': round(py_res['hash_std'], 6),
            'Min_Time_ms': round(py_res['hash_min'], 6),
            'Max_Time_ms': round(py_res['hash_max'], 6),
            'Execution_Status': 'Measured'
        })

        # C++ Benchmarks (Only if compiled binary actually exists)
        if has_compiled_cpp:
            cpp_bst_reps = []
            cpp_hash_reps = []

            for rep in range(repetitions):
                try:
                    res = subprocess.run([exe_path, str(size), str(num_queries)], capture_output=True, text=True, check=True)
                    lines = res.stdout.strip().split('\n')
                    for line in lines:
                        parts = line.split(',')
                        if len(parts) >= 5:
                            op = parts[1]
                            mean_ms = float(parts[4])
                            if op == 'BST Search':
                                cpp_bst_reps.append(mean_ms)
                            elif op == 'Hash Lookup':
                                cpp_hash_reps.append(mean_ms)
                except Exception as e:
                    print(f"Error running compiled C++ benchmark binary: {e}")

            if cpp_bst_reps:
                rows.append({
                    'Language': 'C++ (Compiled)',
                    'Operation': 'BST Search',
                    'Dataset_Size': size,
                    'Num_Operations': num_queries,
                    'Repetitions': len(cpp_bst_reps),
                    'Mean_Time_ms': round(float(np.mean(cpp_bst_reps)), 6),
                    'Std_Dev_ms': round(float(np.std(cpp_bst_reps)), 6),
                    'Min_Time_ms': round(float(np.min(cpp_bst_reps)), 6),
                    'Max_Time_ms': round(float(np.max(cpp_bst_reps)), 6),
                    'Execution_Status': 'Measured'
                })

            if cpp_hash_reps:
                rows.append({
                    'Language': 'C++ (Compiled)',
                    'Operation': 'Hash Lookup',
                    'Dataset_Size': size,
                    'Num_Operations': num_queries,
                    'Repetitions': len(cpp_hash_reps),
                    'Mean_Time_ms': round(float(np.mean(cpp_hash_reps)), 6),
                    'Std_Dev_ms': round(float(np.std(cpp_hash_reps)), 6),
                    'Min_Time_ms': round(float(np.min(cpp_hash_reps)), 6),
                    'Max_Time_ms': round(float(np.max(cpp_hash_reps)), 6),
                    'Execution_Status': 'Measured'
                })
        else:
            rows.append({
                'Language': 'C++ (Compiled)',
                'Operation': 'BST Search',
                'Dataset_Size': size,
                'Num_Operations': num_queries,
                'Repetitions': 0,
                'Mean_Time_ms': np.nan,
                'Std_Dev_ms': np.nan,
                'Min_Time_ms': np.nan,
                'Max_Time_ms': np.nan,
                'Execution_Status': 'Not Executed (Compiler Unavailable)'
            })

            rows.append({
                'Language': 'C++ (Compiled)',
                'Operation': 'Hash Lookup',
                'Dataset_Size': size,
                'Num_Operations': num_queries,
                'Repetitions': 0,
                'Mean_Time_ms': np.nan,
                'Std_Dev_ms': np.nan,
                'Min_Time_ms': np.nan,
                'Max_Time_ms': np.nan,
                'Execution_Status': 'Not Executed (Compiler Unavailable)'
            })

    results_df = pd.DataFrame(rows)

    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        results_df.to_csv(os.path.join(RESULTS_DIR, 'python_cpp_benchmark.csv'), index=False)

        # Plot comparison figure
        plt.figure(figsize=(9, 5))

        py_bst = results_df[(results_df['Language'] == 'Python') & (results_df['Operation'] == 'BST Search')]
        py_hash = results_df[(results_df['Language'] == 'Python') & (results_df['Operation'] == 'Hash Lookup')]

        plt.plot(py_bst['Dataset_Size'], py_bst['Mean_Time_ms'], marker='o', label='Python BST Search (Measured)', color='#EF4444', linewidth=2)
        plt.plot(py_hash['Dataset_Size'], py_hash['Mean_Time_ms'], marker='^', linestyle='--', label='Python Hash Lookup (Measured)', color='#F59E0B', linewidth=2)

        if has_compiled_cpp:
            cpp_bst = results_df[(results_df['Language'].str.contains(r'C\+\+')) & (results_df['Operation'] == 'BST Search') & (results_df['Execution_Status'] == 'Measured')]
            cpp_hash = results_df[(results_df['Language'].str.contains(r'C\+\+')) & (results_df['Operation'] == 'Hash Lookup') & (results_df['Execution_Status'] == 'Measured')]

            if not cpp_bst.empty:
                plt.plot(cpp_bst['Dataset_Size'], cpp_bst['Mean_Time_ms'], marker='s', label='C++ BST Search (Measured)', color='#3B82F6', linewidth=2)
            if not cpp_hash.empty:
                plt.plot(cpp_hash['Dataset_Size'], cpp_hash['Mean_Time_ms'], marker='d', linestyle='--', label='C++ Hash Lookup (Measured)', color='#10B981', linewidth=2)

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Dataset Size (N items)')
        plt.ylabel('Mean Lookup Latency (ms, log scale)')
        plt.title('Data Structure Performance Benchmark (Measured Repeated Runs)')
        plt.legend()
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'python_cpp_comparison.png'), dpi=300)
        plt.close()

    return results_df


if __name__ == '__main__':
    print("=== RUNNING PYTHON VS C++ DATA STRUCTURE BENCHMARK ===")
    df = run_python_vs_cpp_benchmark()
    print(df.to_string(index=False))
