# Wan2.1 DiffSynth Training Utilities

This directory contains project-local utilities for preparing and launching
Wan2.1 training with DiffSynth-Studio. The official Wan2.1 code and the
DiffSynth-Studio source tree are kept unchanged.

## Prepare WISA-80K

The WISA-80K video shards are zip files whose internal paths contain historical
absolute-looking prefixes such as `home/jovyan/.../128_split/0/<video>.mp4`.
Use the preparation script to flatten each shard into one directory:

```bash
python wan21_train/prepare_wisa80k_for_diffsynth.py \
  --extract \
  --build-metadata \
  --num-workers 8
```

Default paths:

- Zip shards: `/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/videos`
- Extracted videos: `/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data`
- Metadata JSON: `/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/snapshots/66e0fd0d6963a76999d0653b5d2d0e3b5c1442f5/data/wisa-80k.json`
- DiffSynth metadata: `/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/metadata.csv`
- Extraction manifest: `/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/extraction_manifest.csv`

For a smoke test that writes nothing:

```bash
python wan21_train/prepare_wisa80k_for_diffsynth.py \
  --inspect-meta \
  --extract \
  --limit-zips 1 \
  --dry-run
```

To build separated physics-category metadata files after extraction, use
`--metadata-mode categories` and pass an output directory to `--metadata-out`:

```bash
python wan21_train/prepare_wisa80k_for_diffsynth.py \
  --build-metadata \
  --metadata-mode categories \
  --metadata-out /datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/physics_metadata \
  --min-duration 2.0 \
  --video-check on \
  --video-check-workers 8
```

During category metadata generation, the script first applies cheap metadata
filters, then optionally checks candidate videos by decoding the first frame.
Progress bars are shown for metadata scanning and video checking when `tqdm` is
available. Use `--video-check off` to skip first-frame decoding for faster
metadata generation. When `--video-check on`, use `--video-check-workers` to
control the parallel video-checking workers; the default is
`min(8, os.cpu_count())`.

This writes three DiffSynth-compatible CSV files:

- `/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/physics_metadata/metadata_dynamics.csv`
- `/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/physics_metadata/metadata_thermodynamics.csv`
- `/datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/physics_metadata/metadata_optics.csv`

All three category files require a non-empty `duration` field,
`duration > --min-duration`, `duration * fps >= 81`, and an extracted video file
whose first frame can be decoded when `--video-check on`. The default
`--min-duration` is `2.0` seconds. Category assignment uses the WISA `label`
field:

- Dynamics: `collision`, `deformation`, `elastic motion`, `explosion`,
  `gas motion`, `liquid motion`, and `rigid body motion`.
- Thermodynamics: `combustion`, `liquefaction`, `melting`, `solidification`,
  and `vaporization`.
- Optics: `interference and diffraction`, `reflection`, `refraction`,
  `scattering`, and `unnatural light source`.

The dynamics CSV applies additional filters: `q0` must not be
`no obvious dynamic phenomenon`; `magnet` and `magetic` must not appear in the
prompt or `n0`; `metallic` must not appear in `n0`; and numeric filters remove
rows with `motion_score <= 0.10`, `motion_score_v2 <= 0.01`, or
`motion_score_v2 >= 6.50`.

The default `--metadata-mode all` writes one metadata file for the whole dataset.
The older dynamics-only single-file command remains available but now uses the
same strict dynamics filters:

```bash
python wan21_train/prepare_wisa80k_for_diffsynth.py \
  --build-metadata \
  --only-dynamics \
  --metadata-out /datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/metadata_dynamics.csv
```


To merge the three category CSVs into one training metadata file while
downsampling the overrepresented `reflection` subtype to 4000 rows:

```bash
python wan21_train/prepare_wisa80k_for_diffsynth.py \
  --merge-category-metadata /datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/physics_metadata \
  --merge-out /datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/physics_metadata/metadata_physics_merged_reflection4000.csv \
  --reflection-sample-n 4000 \
  --merge-seed 0
```

After generating a metadata CSV, check its label and `q0` distribution:

```bash
python wan21_train/prepare_wisa80k_for_diffsynth.py \
  --summarize-csv /datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/physics_metadata/metadata_dynamics.csv
```

To inspect a CSV without opening the full file, export a small JSONL sample:

```bash
python wan21_train/prepare_wisa80k_for_diffsynth.py \
  --sample-csv /datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/physics_metadata/metadata_dynamics.csv \
  --sample-n 20 \
  --sample-out /datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/metadata_sample.jsonl
```

Use `--sample-random --sample-seed 0` to draw random rows instead of taking the
first rows.

If you also want the sampled videos copied next to the JSONL file:

```bash
python wan21_train/prepare_wisa80k_for_diffsynth.py \
  --sample-csv /datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/physics_metadata/metadata_dynamics.csv \
  --sample-n 20 \
  --sample-copy-dir /tmp/wisa_sample_videos \
  --sample-base-dir /datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data \
  --sample-out /tmp/wisa_sample.jsonl
```

DiffSynth Wan training should then use:

```bash
--dataset_base_path /datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data
--dataset_metadata_path /datacache/huggingface/hub/datasets--qihoo360--WISA-80K/data/video_data/physics_metadata/metadata_dynamics.csv
--data_file_keys "video"
```
