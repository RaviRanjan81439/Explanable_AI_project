# FreeMCG: Explainable AI Experiments

A reproducibility-focused project exploring **FreeMCG (Free-form Multi-Concept Guidance)** for explainable image classification. The repository contains notebooks and saved outputs for feature attribution, ROAD-based explanation evaluation, counterfactual generation, and a Phase 2 experiment studying the runtime–quality trade-off of using fewer reverse-diffusion steps.

> **Important:** This repository is an experiment workspace, not a fully self-contained application. The FreeMCG implementation, pretrained diffusion checkpoint, and ImageNet validation images are external inputs and are not included in this repository. A fresh end-to-end run requires those assets and a compatible GPU environment. Saved reports and results are included for review without rerunning the experiments.

## Repository structure

```text
EXP_AI/
├── README.md
├── requirements.txt
├── .gitignore
├── Original research paper FREEMCG.pdf
├── experimental results and comparison.pdf
├── ONLY PHASE 2 REPORT.pdf
├── Phase_1(Re-run_Experiment)/
│   └── Phase_1(Re-run_Experiment)/
│       ├── Experiment_1_FreeMCG_Feature_Attribution.ipynb
│       ├── Experiment_2_ROAD_Evaluation.ipynb
│       ├── Experiment_3_Counterfactual_Generation.ipynb
│       ├── FreeMCG_Checks_Reference_Based.ipynb
│       ├── images/
│       └── results/
└── phase_2_project_with_README/
    ├── Jupyter notebook code for phase 2/
    │   └── notebookd2e54cab17 (2).ipynb
    ├── implementation results/
    └── Reproduction_Guide/README.md
```

## Experiments

### Phase 1 — Re-running and evaluating FreeMCG

| Notebook | Purpose |
|---|---|
| `Experiment_1_FreeMCG_Feature_Attribution.ipynb` | Computes FreeMCG feature-attribution maps and compares them with Vanilla Gradient, Input × Gradient, and Integrated Gradients. SHAP is an optional comparison. |
| `Experiment_2_ROAD_Evaluation.ipynb` | Evaluates explanations using a ROAD-style feature-removal experiment. |
| `Experiment_3_Counterfactual_Generation.ipynb` | Generates counterfactual image particles using FreeMCG. |
| `FreeMCG_Checks_Reference_Based.ipynb` | Performs reference-based checks of the FreeMCG setup and intermediate calculations. |

### Phase 2 — Reverse-diffusion step-count experiment

Compares **100, 50, and 25 reverse-diffusion steps** for three selected ImageNet source images (cheetah, rhinoceros beetle, and tabby cat), using ladybug as the target class. The experiment records runtime, target-class flip rate, target probability, and L2 distance.

There are nine image/step conditions (3 images × 3 step counts). There is one selected image per source class, so the class-level aggregate is descriptive and should not be interpreted as statistically representative of the full class.

The Phase 2-specific execution guide is available at [`phase_2_project_with_README/Reproduction_Guide/README.md`](phase_2_project_with_README/Reproduction_Guide/README.md).

## Requirements

- Python **3.10.x** is recommended (the Phase 2 notebook was recorded with Python 3.10.21).
- Jupyter Notebook / JupyterLab, or Kaggle Notebooks.
- NVIDIA GPU with CUDA support strongly recommended. The diffusion experiments can be very slow or impractical on CPU.
- External FreeMCG source repository and its bundled guided-diffusion code.
- OpenAI 256×256 unconditional diffusion checkpoint used by the FreeMCG implementation.
- ImageNet validation images, with legitimate access. The notebook uses the ImageNet validation directory structure.

Install the common Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Install **PyTorch and torchvision separately** using the official selector for your operating system and CUDA version: [PyTorch installation instructions](https://pytorch.org/get-started/locally/). In Kaggle, the preinstalled PyTorch stack is usually the easiest option; avoid replacing it unless necessary.

## Quick start — Phase 2 (recommended entry point)

The Phase 2 notebook has a more detailed reproduction guide and saves completed results condition-by-condition.

### Option A: Kaggle Notebooks

1. Download or clone this repository and locate the notebook under `phase_2_project_with_README/Jupyter notebook code for phase 2/`.
2. Create a Kaggle Notebook and enable a GPU accelerator.
3. Add the dataset containing the complete FreeMCG project, diffusion checkpoint, and ImageNet validation data as a Kaggle input. The supplied notebook currently expects the dataset path:

   ```text
   /kaggle/input/datasets/raviranjanbharti/freemcg/XAI_FREEMCG/XAI_FREEMCG/
   ```

4. Import the Phase 2 notebook into Kaggle.
5. Run the notebook from the first cell to the last. Check the **Project paths** section and update paths if your Kaggle dataset has a different directory structure.
6. Confirm that the FreeMCG source, ImageNet validation directory, and diffusion checkpoint are found. Confirm that the notebook reports `Classifier equivalence check: PASSED` before launching the full experiment.
7. Run the three-class step-count comparison cell, followed by the aggregate plotting cell.
8. Results are written to:

   ```text
   /kaggle/working/Phase2_Results/three_class_step_sweep/
   ```

9. Download the output directory from the Kaggle notebook's output panel.

### Option B: Local Jupyter

1. Create and activate a Python 3.10 environment (commands for Windows PowerShell):

   ```powershell
   py -3.10 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   On macOS/Linux, use `python3.10 -m venv .venv` and `source .venv/bin/activate`.

2. Install a compatible PyTorch/torchvision build as described above.
3. Place the external FreeMCG repository, diffusion checkpoint, and ImageNet validation data on your machine. The expected FreeMCG directory contains at least `freemcg.py`, `utils.py`, `blended_diffusion_custom/`, `diffusion_ckpts/`, and `ILSVRC2012/val/`.
4. Open the Phase 2 notebook and edit its **2. Project paths** cell. Set `PROJECT_ROOT`, `FREE_MCG_ROOT`, `VAL_DIR`, and the output directory to locations that exist on your machine. Replace Kaggle-only `/kaggle/working/...` output paths with a writable local folder.
5. Run all cells in order. The model checkpoint must be present at the path expected by the notebook; check the exact filename in the checkpoint-loading cell.

## Running Phase 1 notebooks

1. Open the required notebook from `Phase_1(Re-run_Experiment)/Phase_1(Re-run_Experiment)/` in Jupyter.
2. Before running, edit the path configuration cells. The supplied Phase 1 notebooks contain Windows-specific absolute paths such as `C:/Users/.../FreeMCG` and `C:/Users/.../Phase_1(Re-run_Experiment)`. Replace these with your own paths:
   - `FREE_MCG_ROOT`: location of the external FreeMCG source directory.
   - `IMAGE_PATH`: path to the ImageNet image used by the experiment.
   - `PHASE1_ROOT`: location where Phase 1 results should be saved.
3. Ensure the diffusion checkpoint and ImageNet image exist at the configured paths.
4. Run notebook cells sequentially. Each experiment writes its outputs under the configured `PHASE1_ROOT/results/` subfolder.

Do not assume that the notebooks can run immediately after cloning: the external source code, model checkpoint, and dataset must first be configured.

## Included results

The repository includes previously generated CSV/JSON metrics, plots, image panels, and PDF reports. These allow readers to inspect the experiment outputs without running the computationally expensive diffusion pipeline. Results may be regenerated by running the notebooks with the required external assets.

## Reproducibility notes

- Keep the same FreeMCG source version, checkpoint, input images, package versions, and random seeds when comparing runs.
- Phase 2 uses a fixed target class (`ladybug`), 100 particles, and step counts of 100/50/25. See the Phase 2 reproduction guide for the full parameter table.
- GPU type, CUDA, PyTorch version, and nondeterministic GPU kernels can affect runtime and may cause small numerical differences.
- A high target-class flip rate alone does not establish that a counterfactual is visually plausible; inspect the saved image panels alongside the quantitative metrics.
- The Phase 2 notebook's saved graph filenames include the legacy phrase `5_images`; the actual aggregate in this project uses three selected images.

## Troubleshooting

| Problem | What to check |
|---|---|
| `ModuleNotFoundError: freemcg` | Confirm `FREE_MCG_ROOT` points to the directory containing `freemcg.py` and that it is added to `sys.path`. |
| Guided-diffusion import error | Confirm the complete FreeMCG repository, including `blended_diffusion_custom/`, is present. |
| Diffusion checkpoint not found | Check the checkpoint directory and exact filename used in the notebook. |
| ImageNet directory not found | Confirm that `VAL_DIR` points to the validation image folder and that the expected synset subfolders are present. |
| CUDA out of memory | Use a GPU with more memory or reduce execution micro-batch sizes where the notebook allows it. Do not change the particle count if you intend to reproduce the reported Phase 2 configuration. |
| Very long execution time | Run Phase 2 on a GPU. The full experiment performs nine diffusion conditions and can take substantial time. |
| Windows path errors | Replace all machine-specific absolute paths with paths on your computer; use raw strings (`r"..."`) or `pathlib.Path`. |

## Citation and acknowledgement

This project reproduces and evaluates methods from the FreeMCG research work. Please consult `Original research paper FREEMCG.pdf` for the source paper's bibliographic details and cite the original authors when reusing the method. Also cite ImageNet and the pretrained model/checkpoint sources where applicable. This repository's notebooks and experimental results should be cited separately if reused.


No license has been specified for this repository. Unless a license is added, do not assume that the project code, reports, or included assets are available for unrestricted reuse. External datasets, pretrained weights, and the original FreeMCG implementation remain subject to their respective licenses and terms.
