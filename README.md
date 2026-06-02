## Salience bias in natural visual search

Public API for evaluating predictive maps against scanpaths, and preprocessing Eyelink sessions into scanpath JSON.

### Install

```bash
pip install -e .
```

Eyetracking preprocess also needs `pyedfread` (install via conda-forge or your environment; not pinned on PyPI here).

### Evaluate (tutorial)

```python
from salience_bias.analyses.evaluate.api import evaluate_single, evaluate_convex_mixture
from salience_bias.analyses.evaluate.datasets.steer_search import zoom_to_hw
```

See `examples/tutorial_evaluation.ipynb` (or Colab notebook under `private/notebooks/` locally).

### Preprocess

```bash
python -m salience_bias.analyses.preprocess.workflow.preprocess \
  --subject SUBJECT --datacode YYYYMMDD_HHMMSS \
  --dataset-name steer-search
```

Raw layout: `private/data/raw/<dataset-name>/` with `meta.yaml`, `/<subject>/behav/*.json`, `/<subject>/eye/*.edf`.

Image naming is defined in `salience_bias/analyses/preprocess/config/steer-search.yaml`.

Colab walkthrough: `private/notebooks/tutorial_preprocess.ipynb` (gitignored; copy to Drive or duplicate under `examples/` if you want it on GitHub).
