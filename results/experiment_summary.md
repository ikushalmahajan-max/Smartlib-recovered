# Final Research Experiment Summary Report

> **REPRODUCIBILITY STATEMENT**: All numerical metrics reported in this document were generated from automated execution of `run_experiments.py` using fixed random seed `42` and a strict 3-way temporal evaluation split protocol (Train -> Validation -> Final Test). No values were manually edited, fabricated, or assumed.

---

## 1. Dataset Overview
- **Synthetic Books**: 750 items
- **Synthetic Users**: 250 profiles (including cold-start and sparse readers)
- **Borrowing Transactions**: 4,000 records
- **Date Range**: 2025-09-01 to 2026-06-30

---

## 2. Validation Set Hybrid Weight Selection ($K=5$)

```
 Content_Weight  Collaborative_Weight Weight_Ratio  Validation_Eligible_Users  Precision@5  Recall@5   F1@5  NDCG@5
            1.0                   0.0      1.0/0.0                        167       0.0096    0.0149 0.0074  0.0172
            0.8                   0.2      0.8/0.2                        167       0.0407    0.0500 0.0329  0.0693
            0.7                   0.3      0.7/0.3                        167       0.0551    0.0712 0.0447  0.0888
            0.6                   0.4      0.6/0.4                        167       0.0623    0.0799 0.0508  0.0962
            0.5                   0.5      0.5/0.5                        167       0.0731    0.1180 0.0673  0.1216
            0.4                   0.6      0.4/0.6                        167       0.0862    0.1609 0.0846  0.1594
            0.3                   0.7      0.3/0.7                        167       0.0934    0.1772 0.0939  0.1737
            0.2                   0.8      0.2/0.8                        167       0.0970    0.1847 0.0986  0.1803
            0.0                   1.0      0.0/1.0                        167       0.0994    0.1874 0.1004  0.1850
```

### Selection Result:
- Optimal Validation Weight Ratio: **0.0/1.0** (Content Weight = 0.0, Collaborative Weight = 1.0) with Validation NDCG@5 = `0.1850`.
- This hyperparameter configuration is frozen and evaluated ONCE on the Final Test set.

---

## 3. Frozen Final Test Set Model Evaluation ($K=5$)

```
                         Model  Total_Test_Users  Eligible_Test_Users  Evaluated_Users  Precision@5  Recall@5   F1@5  NDCG@5
         Rating-Based Baseline               150                  150              150       0.0040    0.0072 0.0030  0.0101
         Content-Only (TF-IDF)               150                  150              150       0.0040    0.0036 0.0023  0.0060
Collaborative-Only (User-User)               150                  150              150       0.0867    0.1581 0.0875  0.1600
     Selected Hybrid (0.0/1.0)               150                  150              150       0.0867    0.1581 0.0875  0.1600
```

### Key Findings:
- Collaborative Filtering (User-User Cosine Similarity) achieves Precision@5 = `0.0867` and NDCG@5 = `0.1600`.
- The Selected Hybrid Recommender achieves Precision@5 = `0.0867` and NDCG@5 = `0.1600` on the frozen test set.

---

## 4. Cold-Start Performance Analysis (Final Test Set)

```
                      User_Group  Total_Group_Users  Eligible_Test_Users  Evaluated_Users  Precision@5  Recall@5   F1@5  NDCG@5
     Cold-Start Users (<=1 loan)                111                   11               11       0.0364    0.1818 0.0606  0.1301
Sparse-History Users (2-5 loans)                 42                   42               42       0.0429    0.2143 0.0714  0.1608
 Normal-History Users (>5 loans)                 97                   97               97       0.1113    0.1310 0.0975  0.1631
```

### Key Observation:
- Users with normal borrowing history (>5 loans) achieve the highest recommendation precision (`0.1113`), whereas cold-start users rely on popularity fallback until borrowing history accumulates.

---

## 5. Empirical Data Structure Performance Benchmarks ($N=100$ to $50,000$)

```
 Dataset_Size                Algorithm  Batch_Queries  Repetitions  Mean_Time_ms  Median_Time_ms  Std_Time_ms  Min_Time_ms  Max_Time_ms
        50000     Linear Search (List)           2000            5      1.322816        1.390740     0.271433     0.793394     1.527143
        50000        Custom BST Search           2000            5      0.001375        0.001406     0.000090     0.001214     0.001468
        50000 Custom Hash Table Lookup           2000            5      0.002169        0.002111     0.000184     0.001968     0.002517
```

---

## 6. Python vs C++ Comparative Latency Benchmark ($N=50,000$)

```
      Language   Operation  Dataset_Size  Num_Operations  Repetitions  Mean_Time_ms  Std_Dev_ms  Min_Time_ms  Max_Time_ms                    Execution_Status
        Python  BST Search         50000            1000            5      0.001302    0.000197     0.001039     0.001578                            Measured
        Python Hash Lookup         50000            1000            5      0.002158    0.000285     0.001931     0.002710                            Measured
C++ (Compiled)  BST Search         50000            1000            0           NaN         NaN          NaN          NaN Not Executed (Compiler Unavailable)
C++ (Compiled) Hash Lookup         50000            1000            0           NaN         NaN          NaN          NaN Not Executed (Compiler Unavailable)
```
