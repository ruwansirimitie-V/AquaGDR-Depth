# AquaGDR-Depth

AquaGDR-Depth estimates metric depth from a single underwater equirectangular
(ERP) panorama. Its model combines projection-specific input adaptation,
five-scale gated ERP/Cubemap fusion, and local-detail residual refinement.

> **Release status:** this is a clean source-code package prepared from the
> research project. The dataset, split lists, and checkpoints are not bundled.
> Before public release, fill the author/citation fields and choose a license.
> The top-level `LICENSE` is intentionally a placeholder and is not a grant of
> rights. Do not publish until it is replaced with the license you choose.

## Repository layout

```text
AquaGDR-Depth/
├── README.md
├── LICENSE
├── CITATION.cff
├── environment.yml
├── configs/                 # Stage 1 and Stage 2 example configurations
├── aquagdr/                 # AquaGDR model, training, losses, metrics
├── tools/                   # Train/evaluate shortcuts and dataset check
├── docs/                    # Reproduction notes and figure guidance
├── tests/                   # Lightweight repository checks
├── data/                    # User-provided data (ignored by Git)
└── third_party/unifuse/     # Adapted UniFuse components and upstream license
```

## Installation

The example environment targets Python 3.10, PyTorch 2.2, and CUDA 12.1. Adjust
the PyTorch/CUDA entries to match your machine if needed.

```bash
conda env create -f environment.yml
conda activate aquagdr-depth
```

## Dataset preparation

The expected layout is:

```text
data/underwater_erp/
├── rgb/       # ERP RGB panoramas
└── depth/     # metric depth maps, in the scale expected by the dataset loader
data/splits/
├── train.txt
├── val.txt
└── test.txt
```

Each split line has an RGB path and a depth path relative to
`data/underwater_erp`, separated by whitespace. See
[`data/splits/README.md`](data/splits/README.md). The source data and split
files are not redistributed here. Confirm the dataset license and preserve the
official attribution. The paper's reported scores require the same data
generation, split, preprocessing, and evaluation protocol described in the
paper; merely using similarly named files does not reproduce those results.

## Checkpoints

Training starts from a UniFuse-derived underwater backbone checkpoint. Obtain
or create a checkpoint that you are authorized to use and place it at
`checkpoints/backbone_init.pth`, or update `pano_backbone.weight_file` in the
configuration. Checkpoints are excluded from Git. Stage 2 resumes from the
Stage 1 checkpoint by default; update `train.resume_ckpt` if your output path
differs.

## Train and evaluate

Run commands from the repository root. First review every data and checkpoint
path in the YAML file.

```bash
python -m aquagdr.train_fusion --config configs/stage1.yaml
python -m aquagdr.train_fusion --config configs/stage2.yaml
python -m aquagdr.eval_fusion --config configs/stage2.yaml --split test
```

Use `--smoke_test` with the training command to run one forward/backward batch
before a full run. Evaluation writes JSON metrics under the configured output
directory. A GPU is strongly recommended for 512 × 1024 inputs.

## Tests

```bash
python -m pytest -q
```

The included tests check the release layout and prevent accidental reintroduction
of machine-specific paths. They do not replace a full end-to-end training run.

Check a split before training:

```bash
python tools/check_dataset.py --split data/splits/train.txt
python tools/check_dataset.py --split data/splits/val.txt
```

## Method and reproduction notes

The backbone predicts an initial depth map `D0`. The local-detail branch and
residual refinement head then produce a bounded, spatially modulated correction
and the final metric depth. The release contains the final architecture, not
the archived global-context or coarse-refinement variants. The two-stage protocol is configured in
`configs/stage1.yaml` and `configs/stage2.yaml`.

See [`docs/`](docs/) for publication/reproduction notes. For scientific
reproduction, report the exact data version, split hashes, checkpoint source,
software versions, and metric protocol. Do not commit private raw data, model
weights, evaluation outputs, or credentials.

## Third-party software

The UniFuse-derived source is under `third_party/unifuse/`; its MIT license is
retained there. Please cite the UniFuse paper in addition to AquaGDR-Depth when
appropriate. See `third_party/unifuse/UPSTREAM.md` for the provenance fields to
complete before release.

## Citation

The `CITATION.cff` file is a template. Replace the author placeholder and add
the final paper DOI or URL once available.
