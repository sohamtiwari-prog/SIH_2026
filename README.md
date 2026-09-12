# lunareg — Sun-angle and scale invariant lunar image registration

**ISRO SIH 2026 · Problem Statement 26166** — *Multi-modal, sun angle and
scale invariant image correspondence using Chandrayaan-2 optical images
(OHRC, TMC-2, IIRS)*

`lunareg` finds sub-pixel correspondences between a Chandrayaan-2 source
image (OHRC / TMC-2 / IIRS) and a reference lunar map (LRO NAC, SELENE, or
another Chandrayaan-2 product), and returns the registered image, the
matched tie points, and evaluation metrics — even when the two images were
taken under a completely different sun angle and at a 10:1+ scale ratio.

## Why this is hard, and what actually works

On an airless body, the only thing that changes between two views of the
same terrain is the sun. A crater lit from the east has a bright west wall
and a shadowed east wall; lit from the west, the pattern inverts. SIFT-style
descriptors are gradient-orientation histograms, so a 180° sun-azimuth
change flips every gradient and collapses the match — measured on our
physically-rendered validation harness:

| Method (sparse features) | Success rate, 0°→180° sun-azimuth sweep |
|---|---|
| SIFT | 100% at 0°, **0% by 60°** |
| RIFT (phase-congruency + MIM) | 100% at 0°/180°, **0% from 30°–150°** |
| **Dense phase-congruency area correlation** | **100% across the entire sweep**, sub-pixel median error (0.06–0.17 px true RMSE) |

The fix isn't a better keypoint descriptor — it's changing *what* survives
illumination change (Fourier **phase**, not amplitude/gradient — see
`lunareg/photometric.py`) and *how* correspondences are found (dense area
correlation on the phase-congruency map, not sparse keypoint matching — see
`lunareg/dense.py`). The default `hybrid` pipeline therefore treats sparse
RIFT/feature matching as a bootstrap for the coarse prior and does the real
work with dense correlation, exactly the way operational planetary
pipelines lean on a SPICE/PDS geometric prior first and use feature matching
only when that prior is absent or unreliable.

The metadata scale prior matters just as much as illumination: on the same
pairs, disabling Stage 0 (pre-resampling the source to the reference GSD
using the two PDS-label GSDs) degrades the median true error from
**0.06–0.15 px to 184–415 px** (`E3_ablation`, `variant=no_scale_prior` in
`examples/outputs/`) — the matcher can still find points, but a homography
fit with an unconstrained scale is not well-posed enough to land near the
truth. Skipping that resampling step is the most common design error.

## Pipeline

```
Stage 0  Metadata scale prior     resample source to the reference GSD (PDS labels give it exactly)
Stage 1  Illumination transform   phase congruency (Kovesi) — invariant to sun azimuth/elevation
Stage 2  Detect + describe        RIFT (phase-congruency keypoints + MIM descriptor) or dense grid
Stage 3  Match + robust geometry  mutual-NN + Lowe ratio, then MAGSAC++ on a similarity→affine→homography ladder
Stage 4  Sub-pixel refinement     upsampled-DFT phase correlation per tie point (Guizar-Sicairos)
Stage 5  Uniformity enforcement   block-cap + adaptive non-maximal suppression, then refit
Stage 6  Metrics                  reprojection RMSE (+ bootstrap CI), inlier ratio, uniformity, residual structure
```

Every stage is a flag on `PipelineConfig` (`lunareg/pipeline.py`) so the
ablation study can isolate exactly which stage is responsible for a given
result. Two independent matching strategies are available when metadata is
missing or untrustworthy: `lunareg/craters.py` matches images by the
*geometric arrangement* of crater rims (scale/rotation-invariant by
construction, no appearance matching at all — the way a star tracker
matches star patterns, not star brightness), and `lunareg/dense.py`'s
Fourier-Mellin coarse aligner bootstraps a prior with no metadata at all.

## Validating sub-pixel accuracy without real ground truth

Hand-digitised tie points on real imagery carry ~0.5–2 px of operator
noise — the same order as the sub-pixel accuracy the problem statement asks
for, so they cannot certify it. `lunareg/synth.py` instead renders a
physically-motivated virtual Moon (crater DEM with realistic size-frequency
distribution, cast shadows by horizon ray-marching, Lunar-Lambert
photometry, per-sensor MTF/noise/GSD) twice, under different sun angles and
sensor models, and derives the *exact* homography relating the two renders
analytically. Every number in `examples/outputs/` and this README is
measured against that exact ground truth. `lunareg/validate.py` also
implements two truth-free consistency checks — forward/backward cycle
composition and split-half reprojection — for use on real data where no
ground truth exists; `lunareg/stats.py` fits a failure-probability model
(AUC 0.95–0.99 across CV schemes) from self-consistency signals alone, so a
real-data run can flag a likely-bad registration without ever seeing the
truth.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[geo,stats,dev]"
```

### Single-file version

`scripts/lunareg_all_in_one.py` is the whole package — every module under
`lunareg/`, in dependency order — concatenated into one runnable script, for
environments where a package install is inconvenient:

```bash
pip install -r requirements.txt
python scripts/lunareg_all_in_one.py register --src A.xml --ref B.tif --method hybrid
```

It takes the same subcommands as the `lunareg` CLI below (`register`,
`experiments`, `analyse`, `dashboard`) and is verified to produce bit-identical
output to the packaged version (see `tests/`, all of which pass against
either). It's generated, not hand-maintained: after changing anything under
`lunareg/`, regenerate it with `python scripts/build_single_file.py` rather
than editing it directly.

`geo` pulls in `rasterio` for reading GeoTIFF reference products with true
ground-sample-distance recovery; `stats` pulls in `scikit-learn` for the
failure-prediction study; both are optional — the core pipeline only needs
numpy/scipy/opencv/scikit-image/pandas.

## Usage

### Register two real images

```bash
lunareg register \
    --src ohrc_strip.xml --ref lro_nac_mosaic.tif \
    --method hybrid --out-dir outputs/register
```

`--src`/`--ref` accept PDS4 (`.xml` label + `.img`/`.qub` array — the format
Chandrayaan-2 OHRC/TMC-2/IIRS products are archived in on ISSDC/PRADAN),
PDS3 (`.img`/`.lbl` — LRO NAC EDR/CDR), GeoTIFF (already map-projected
reference mosaics, e.g. from LROC QuickMap), or a plain PNG/TIFF with
`--src-gsd`/`--ref-gsd` supplied manually. See `lunareg/io.py` for the
per-format reader; whatever geometry keywords it finds (GSD, solar
azimuth/elevation) feed Stage 0 and the RIFT MIM-shift prior automatically,
and it tells you on stderr when something you might want (a scale prior, an
azimuth prior) isn't available in the label.

Output in `outputs/register/`:
- `product.json` — the fitted homography, per-stage match counts, and the
  evaluation metrics (reprojection RMSE + 95% CI, inlier ratio, uniformity).
- `warped_source.png` — the source resampled into the reference frame.
- `checkerboard.png` — QC mosaic; misalignment shows as broken edges at the
  tile seams.

### Reproduce the validation study and dashboard

```bash
lunareg experiments --out-dir outputs   # E1 illumination, E2 scale, E3 ablation, E4 noise
lunareg analyse     --out-dir outputs   # aggregate CSVs -> dashboard_data.json
lunareg dashboard   --out-dir outputs   # inject payload -> lunar_registration_dashboard.html
```

A pre-built run is checked in at `examples/outputs/` —
`lunar_registration_dashboard.html` is the interactive results dashboard and
`product.json` is a sample deliverable for one worked example pair.

### Run the tests

```bash
pytest
```

43 tests cover the sub-pixel shift estimator against a known Fourier-exact
shift, homography estimation/RANSAC behaviour, the uniformity metrics, the
illumination-invariance claim (phase congruency vs. raw intensity on a real
rendered sun-flip pair), the full pipeline against synthetic ground truth,
the real-product loaders (PDS3/PDS4, built from synthetic-but-spec-accurate
fixtures), and the CLI end to end.

## Repository layout

```
lunareg/
  photometric.py   phase congruency, MIM, classical Lommel-Seeliger correction
  features.py      RIFT keypoints/descriptor, sun-azimuth -> MIM-bin-shift prediction
  matching.py      descriptor matching, MAGSAC++ model ladder, sub-pixel refinement
  dense.py         Fourier-Mellin bootstrap + coarse-to-fine dense area correlation
  craters.py       scale/rotation-invariant crater-constellation matching (no appearance, no metadata)
  distribution.py  enforcing and measuring uniform tie-point coverage
  pipeline.py      the end-to-end registration cascade (PipelineConfig / register)
  metrics.py       ground-truth + self-consistency evaluation metrics
  validate.py      cycle-consistency and split-half checks for real data (no ground truth needed)
  synth.py         physically-motivated synthetic lunar scene generator (the validation harness)
  io.py            real product loaders: PDS3, PDS4, GeoTIFF
  cli.py           `lunareg register|experiments|analyse|dashboard`
  experiments.py   the E1-E4 statistical study
  analyse.py       aggregates experiment CSVs + renders the dashboard imagery
  stats.py         failure-prediction model, dose-response curves, variance decomposition
  build_dashboard.py  injects dashboard_data.json into the HTML template
tests/             pytest suite (43 tests)
scripts/
  lunareg_all_in_one.py  generated single-file build of the whole package (see above)
  build_single_file.py   regenerates it from lunareg/
examples/outputs/  a pre-built dashboard + sample product.json deliverable
.github/workflows/ CI: pytest across Python 3.10-3.12, non-blocking ruff lint
```

## Known limitations / next steps

- `io.py`'s PDS4 reader is schema-tolerant (it locates data by PDS4 core
  class, not a fixed namespace path) but has only been exercised against
  synthetic, spec-accurate fixtures — it has not yet been run against a real
  ISSDC-archived OHRC/TMC-2/IIRS product, since none is bundled with the
  problem statement (dataset link is TBD). Point-in-time fixes to instrument-
  specific label quirks should be expected once real products are available.
- IIRS is hyperspectral; `io.py` currently band-averages the cube to a single
  registration image. A band-selection or band-weighted strategy tuned to
  which IIRS bands actually carry spatial structure would likely help.
- The crater-constellation matcher's bottleneck is detector recall (measured
  ~0.6 with the matched-filter detector), not the matching/voting stage —
  see the discussion in `lunareg/craters.py`.
