# Research Paper Claim Integrity Audit

This report audits pre-existing claims against **actual, empirical experimental measurements** from the updated codebase and dataset under a strict 3-way temporal split evaluation protocol.

| Previous Paper Claim | Supported by Evidence? | Actual Measured Evidence | Required Action for Paper Revision |
| :--- | :---: | :--- | :--- |
| **Precision@5 = 84.2%** | ❌ **No** | Verified Precision@5 = `8.67%` (NDCG@5 = `0.1600`) | Update paper text to state actual verified offline Precision@5 (`8.67%`). Explaining sparse transaction context. |
| **Guaranteed O(log N) BST Search** | ⚠️ **Partial** | Average BST lookup at $N=50,000$: `0.001375 ms` | Clarify that ordinary unbalanced BST has $O(\log N)$ average complexity and $O(N)$ worst-case complexity. |
| **Hash Table Lookup < 0.01 ms** | ✅ **Yes** | Hash Table lookup at $N=50,000$: `0.002169 ms` | Retain claim; empirical data confirms sub-millisecond lookup. |
| **Optimal Hybrid Weight = 0.6 / 0.4** | ⚠️ **Partial** | Validation selection selects optimal weight ratio `0.0/1.0` | Update paper text to cite empirical weight selection curve on validation set. |

