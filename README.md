# Energy-Based Learning and JEPA Experiments
Backend and pipeline for the **LeWorldModel** Godot Sim in the companion repo:
👉 [https://github.com/InfiniteInbox/JEPAimplementation](https://github.com/InfiniteInbox/i-jepa-sim)
This repository is a personal implementation study of energy-based learning, world models, and Joint-Embedding Predictive Architectures (JEPAs). It follows ideas from the papers below and turns them into small, inspectable experiments using PyTorch and synthetic physics data.

The project is intentionally experimental. The implementations are learning tools and research prototypes rather than production-ready reproductions of the papers.

## Reading and Implementations

- **A Tutorial on Energy-Based Learning (2006)**
  - Implemented a toy energy-based model in `toyEBM.py` and `toyEBMNotebook.ipynb`.
  - The experiment assigns low energy to points sampled from two circular regions and higher energy to synthetic negative samples, then visualizes the learned energy surface.
- **World Models (Ha et al., 2018)**
  - Used as background for learning compact latent representations of an environment and predicting dynamics in latent space.
- **Revisiting Feature Prediction for Learning Visual Representations from Video (LeCun et al., 2024)**
  - Motivated the video feature-prediction experiments in `V-JEPA/`.
- **A Path Towards Autonomous Machine Intelligence (LeCun, 2022)**
  - Provides the broader architectural and representation-learning motivation for this work.
- **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (LeCun et al., 2023)**
  - Informs the student/teacher and embedding-prediction structure used in the JEPA experiments.
- **V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning (LeCun et al., 2025)**
  - Motivated the temporal video experiments, including prediction in representation space and exponential moving average teacher updates.
- **LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics**
  - Implemented the SigReg regularizer in `LeWorldModel/SigReg.py` and used it to encourage well-behaved latent representations.
- **LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels (LeCun et al., 2026)**
  - Motivated the current pixel-to-latent world-model experiment in `LeWorldModel/`, including action-conditioned prediction and latent-space diagnostics.

## Repository Layout

```text
.
├── toyEBM.py                 # Standalone toy energy-based model
├── toyEBMNotebook.ipynb      # Notebook version of the toy EBM experiment
├── Godot/
│   ├── gd_pipeline.py        # Training, evaluation, PCA, and visualization pipeline
│   ├── test_pca.py           # PCA/checkpoint inspection
│   ├── testTCP.py            # TCP experiment utilities
│   ├── testUDP.py            # UDP experiment utilities
│   └── notes.txt
├── V-JEPA/
│   ├── VJEPA.py              # EMA teacher utility and JEPA experiment entry point
│   ├── models.py             # Convolutional encoder and latent predictor
│   ├── data.py               # Video/episode data utilities
│   ├── metrics.py            # Representation and prediction metrics
│   ├── train.ipynb           # Training notebook
│   └── episodes/              # Synthetic video episodes
└── LeWorldModel/
    ├── models.py             # Pixel encoder and action-conditioned predictor
    ├── data.py               # Bouncing-object physics/data generator
    ├── SigReg.py             # SigReg latent regularizer
    ├── metrics.py            # Latent quality and probing metrics
    ├── train.ipynb           # Training notebook
    ├── episodes/              # Generated training episodes
    └── checkpoint/             # Saved model, PCA basis, and related artifacts
```

## Main Experiments

### Toy energy-based model

`toyEBM.py` creates a simple two-region 2D dataset, generates positive and negative samples, trains an MLP with a contrastive margin loss, and plots the resulting energy heatmap. The notebook provides the same experiment split into independently runnable sections.

### V-JEPA-style video prediction

`V-JEPA/` uses convolutional encoders to map frames to 128-dimensional embeddings and predicts future embeddings rather than pixels. The model structure is kept small so the role of representation-space prediction, student/teacher asymmetry, and EMA updates can be studied on synthetic episodes.

### LeWorldModel

`LeWorldModel/` generates 64x64 bouncing-object sequences with positions, velocities, and discrete actions. The encoder maps pixels to a latent vector, while the action-conditioned predictor rolls the latent state forward. Training combines prediction loss with SigReg, and the metrics include effective rank, embedding statistics, action sensitivity, and probes for recovering physical state.

The `Godot/gd_pipeline.py` script connects training and analysis, including checkpointing, PCA-based visualization, and evaluation reports.

## Setup

The experiments use Python, PyTorch, NumPy, Matplotlib, Seaborn, pandas, and scikit-learn. From the repository root, create or activate a virtual environment and install the dependencies available in your environment. For example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch numpy matplotlib seaborn pandas scikit-learn jupyter
```

Some notebooks may require additional notebook support in VS Code or Jupyter. The code currently assumes that commands are run from the repository root or from the relevant experiment directory, depending on the import paths in the notebook.

## Running the Experiments

Run the toy EBM script from the repository root:

```powershell
python toyEBM.py
```

For the JEPA and world-model experiments, open the corresponding `train.ipynb` notebook in `V-JEPA/` or `LeWorldModel/`. The generated episodes and checkpoint artifacts are kept beside each experiment. The Godot pipeline can be used for the more complete LeWorldModel training and visualization workflow:

```powershell
python Godot/gd_pipeline.py
```

Exact training settings, dataset splits, and checkpoint configuration are kept in the notebooks and pipeline code because they are still being iterated on.

## Areas for Expansion

- I am currently working primarily with latent representations and latent-space prediction.
- Add a frozen decoder that maps learned latents back to pixels for qualitative inspection without allowing reconstruction loss to reshape the representation during JEPA training.
- Compare decoded predictions with ground-truth frames while keeping the predictive objective in embedding space.
- Improve negative-sample generation and sampling strategies for the toy EBM, including adversarial or MCMC-based approaches.
- Extend the world model to longer-horizon rollouts, richer actions, and more varied environments.
- Investigate additional representation regularizers and stronger evaluation protocols.

## Status

This is an active research notebook and implementation log. APIs, hyperparameters, data formats, and experiment organization may change as the ideas are tested.
