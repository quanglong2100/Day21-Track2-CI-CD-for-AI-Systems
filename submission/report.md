**Hyperparameters used:**
*   Best model settings from Step 1: `n_estimators: 100`, `max_depth: 15`.
*   Reason: This provided the highest accuracy (~0.668) during local testing.

**Pipeline Comparison (Step 3.6):**
| Chỉ số | Bước 2 (2998 mẫu) | Bước 3 (5996 mẫu) |
|---|---|---|
| Accuracy | 0.668 | [Check your last Green Train job log] |

**Challenges encountered:**
*   **Dependency Issues:** Python 3.12 compatibility with MLflow required pinning `setuptools<70.0.0`.
*   **Race Conditions:** The deployment health check failed initially because the VM needed more time to load the model. Fixed by increasing `sleep` to 20 seconds.
*   **Pathing:** Username mismatches between Codespaces and GCP VM required absolute pathing in the systemd service.