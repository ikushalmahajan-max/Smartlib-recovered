"""
Centralized Configuration for Library Recommendation System Experiments & Backend.
Loads reproducibility settings from config.yaml if available.
"""

import os
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')

# Default Fallback Values
RANDOM_SEED = 42
TOP_K = 5
DEFAULT_HYBRID_WEIGHTS = {'content': 0.6, 'collaborative': 0.4}
EVALUATION_TEST_RATIO = 0.2
BENCHMARK_ITERATIONS = 500
BENCHMARK_SIZES = [100, 500, 1000, 5000, 10000, 50000]

# Synthetic Dataset Defaults
SYNTHETIC_BOOKS_COUNT = 750
SYNTHETIC_USERS_COUNT = 250
SYNTHETIC_TRANSACTIONS_COUNT = 4000

# Load settings from config.yaml if file exists
if os.path.exists(YAML_CONFIG_PATH):
    try:
        with open(YAML_CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            RANDOM_SEED = cfg.get('random_seed', RANDOM_SEED)
            TOP_K = cfg.get('top_k', TOP_K)
            if 'hybrid_weights' in cfg:
                DEFAULT_HYBRID_WEIGHTS = cfg['hybrid_weights']
            if 'synthetic_dataset' in cfg:
                s_cfg = cfg['synthetic_dataset']
                SYNTHETIC_BOOKS_COUNT = s_cfg.get('num_books', SYNTHETIC_BOOKS_COUNT)
                SYNTHETIC_USERS_COUNT = s_cfg.get('num_users', SYNTHETIC_USERS_COUNT)
                SYNTHETIC_TRANSACTIONS_COUNT = s_cfg.get('num_transactions', SYNTHETIC_TRANSACTIONS_COUNT)
            if 'benchmarks' in cfg:
                b_cfg = cfg['benchmarks']
                BENCHMARK_SIZES = b_cfg.get('dataset_sizes', BENCHMARK_SIZES)
                BENCHMARK_ITERATIONS = b_cfg.get('num_queries', BENCHMARK_ITERATIONS)
            if 'evaluation' in cfg:
                e_cfg = cfg['evaluation']
                EVALUATION_TEST_RATIO = e_cfg.get('test_ratio', EVALUATION_TEST_RATIO)
    except Exception as e:
        pass

# Directory Paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')

# CSV Paths (Original small dataset and synthetic dataset)
USERS_CSV = os.path.join(DATA_DIR, 'users.csv')
BOOKS_CSV = os.path.join(DATA_DIR, 'books.csv')
BORROWING_CSV = os.path.join(DATA_DIR, 'borrowing_history.csv')

SYNTHETIC_USERS_CSV = os.path.join(DATA_DIR, 'synthetic_users.csv')
SYNTHETIC_BOOKS_CSV = os.path.join(DATA_DIR, 'synthetic_books.csv')
SYNTHETIC_BORROWING_CSV = os.path.join(DATA_DIR, 'synthetic_borrowing_history.csv')

# Weight Ablation Configurations (content_weight, collaborative_weight)
WEIGHT_ABLATION_WEIGHTS = [
    (1.0, 0.0),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.3, 0.7),
    (0.2, 0.8),
    (0.0, 1.0)
]
