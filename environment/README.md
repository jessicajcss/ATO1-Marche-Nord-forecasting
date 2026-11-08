# Computational environments

The reference outputs in `output_*/` were produced on Windows with Anaconda and VS Code, using two Python environments. The Python version of each notebook is recorded in its metadata (`language_info.version`).

| Notebook | Python | Environment | Snapshot |
|---|---|---|---|
| `ATO1_minimal_pipeline.ipynb` | 3.11.15 | conda env `neural311` | `requirements-lock-py311.txt` |
| `ATO1_extended_pipeline.ipynb` | 3.13.5 | Anaconda `base` | `requirements-lock-py313.txt` |
| `ATO1_parsimonious_pipeline.ipynb` | 3.13.5 | Anaconda `base` | `requirements-lock-py313.txt` |

The snapshots are `pip freeze` exports of each environment:

```bash
conda activate neural311 && python -m pip freeze > environment/requirements-lock-py311.txt
conda activate base      && python -m pip freeze > environment/requirements-lock-py313.txt
```
