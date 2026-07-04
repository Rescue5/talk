# Ore analysis backend

Local FastAPI service that runs talc segmentation/CV first, then invokes the
ordinary/difficult sulfide classifier only when the refined talc percentage is
less than or equal to the configured threshold. The same processing pass also
builds sulfide inclusion masks with CV and, when configured, MobileSAM.

## Run

```bash
# CPU example; use the matching CUDA index on a GPU host.
python -m pip install torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.lock
export TALC_CHECKPOINT_PATH=/models/talc-best.pt
export SULFIDE_CHECKPOINT_PATH=/models/sulfide-best.pt
export SULFIDE_SAM_CHECKPOINT_PATH=/models/mobile_sam.pt  # optional
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Run this command from `backend/`. Both inference source packages are vendored
under `vendor/`, so a deployed backend does not depend on sibling repositories.
Model weights are deliberately not vendored.
PyTorch is installed separately so Docker/host deployments can deliberately
select CPU or the appropriate CUDA wheel without `requirements.lock` replacing
it.

Environment:

- `TALC_CHECKPOINT_PATH` — required talc DeepLabV3/SegFormer checkpoint.
- `SULFIDE_CHECKPOINT_PATH` — required ordinary/difficult classifier checkpoint.
- `JOBS_DATA_DIR` — persistent uploads, state and artifacts; defaults to
  `backend/data`.
- `TALC_CONFIG_PATH` — optional talc CV/runtime YAML.
- `SULFIDE_CONFIG_PATH` — optional classifier YAML; defaults to the vendored
  config.
- `SULFIDE_SEGMENTATION_CONFIG_PATH` — optional CV sulfide segmentation YAML;
  defaults to the vendored config.
- `SULFIDE_SAM_CHECKPOINT_PATH` — optional MobileSAM checkpoint for sulfide
  mask refinement. If absent, the API still completes with CV-only sulfide
  masks and records the SAM warning in `result.json`.
- `SULFIDE_SAM_DEVICE=auto|cpu|cuda` — MobileSAM device; defaults to
  `MODEL_DEVICE`.
- `TALC_SOURCE_PATH`, `SULFIDE_SOURCE_PATH` — optional source overrides.
- `MODEL_DEVICE=auto|cpu|cuda` — sulfide model device; defaults to `auto`.
- `MAX_UPLOAD_BYTES` — maximum bytes per uploaded file; defaults to 100 MiB.
- `CORS_ORIGINS` — comma-separated origins. Defaults to common localhost
  frontend ports.
- `DEMO_MODE=true` — explicit visual demo without checkpoints. Demo results are
  always marked `demo: true` and must not be treated as model output. It is off
  by default.

If a required checkpoint is absent, the API returns job status
`model_unavailable` and does not invent a classification.

## API

Create a job with multipart field `files` (repeat for multiple images) and a
JSON string in `settings`:

```bash
curl -X POST http://localhost:8000/api/jobs \
  -F 'files=@sample.jpg' \
  -F 'settings={"mode":"overlap","segmentation_threshold":0.5,"cv_threshold":0.55,"talc_threshold_percent":10,"sulfide_threshold":0.5}'
```

Thresholds are job-local and do not mutate the process-wide loaded models:

- `segmentation_threshold` (`0..1`, default `0.5`) controls tile probability
  voting.
- `cv_threshold` (`>0..1`, default `0.55`) overrides the CV hysteresis seed;
  the service keeps its grow threshold strictly lower and leaves the
  strong-response threshold intact.
- `talc_threshold_percent` (`0..100`, default `10`) selects talc-bearing ore
  using a strict `>` comparison.
- `sulfide_threshold` (`0..1`, default `0.5`) selects difficult versus ordinary
  ore.

Applied values, including the effective CV grow and strong-response thresholds,
are written to `result.json` under `processing.thresholds`.

Endpoints:

- `GET /api/health`
- `POST /api/jobs`
- `GET /api/jobs?limit=50`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/results`
- `POST /api/jobs/{job_id}/images` — append files without replacing results.
- `PATCH /api/jobs/{job_id}/settings` — patch all or the `image_ids` supplied
  in the JSON body.
- `PATCH /api/jobs/{job_id}/images/{image_id}/settings` — patch one image;
  accepts either a full settings object or a partial object.
- `GET /api/history?limit=50` — newest completed image records with job/image
  IDs, preview artifacts, classification, settings, and timestamps.
- `GET /api/jobs/{job_id}/artifacts/{image_id}/{artifact_name}`

`GET /results` includes every image immediately after upload, including
`queued`, `running`, and `reprocessing` records. Each record persists its own
settings and progress, while the original job-level fields remain for older
clients.

Each completed image exposes `original`, `segmentation_mask`,
`refined_talc_mask`, `sulfide_cv_mask`, optional `sulfide_sam_mask`,
`coarse_overlay`, `talc_overlay`, `sulfide_cv_overlay`, optional
`sulfide_sam_overlay`, `overlay`, `confidence_maps`, and `result` URLs in its
`artifacts` object. `original` is normalized to browser-compatible PNG.
Overlay files are transparent RGBA layers intended for independent viewer
toggles and opacity controls. `overlay` remains a combined talc preview.

Sulfide masks never overlap refined talc pixels: the talc mask takes precedence
and replaces sulfide candidates in the shared area. The ordinary/difficult
classifier payload remains under `sulfide`; mask statistics live under
`sulfide_segmentation`.

Job progress stages are `upload`, `talc_segmentation`, `cv_refinement`,
`sulfide_segmentation`, `sulfide_classification`, `export`, then `completed`
(or an error status).
Inference jobs are serialized in a one-worker queue to keep model memory usage
bounded.

Settings changes use the earliest affected cached stage:

- segmentation threshold: re-threshold the cached mean probability map, then
  CV and classification (no neural inference);
- CV threshold: cached coarse segmentation, then CV and classification;
- talc/sulfide classification thresholds: cached statistics and sulfide
  probabilities only.

Segmentation mode is immutable after upload because switching overlap strategy
requires new tile inference; attempting to patch it returns HTTP 422.

Sulfide probabilities are cached for every successfully analyzed image,
including talc-bearing images. The final talc-bearing classification has no
probabilistic confidence; ordinary/difficult confidence comes from the sulfide
classifier. If sulfide is unavailable, a valid talc-bearing result remains
completed with a sulfide cache error, while a non-talc result is
`model_unavailable`.

History retention is bounded to the latest 50 terminal image records globally.
Cleanup removes the corresponding upload and artifacts for older images, and
removes a job directory only after none of its image records remain.

Final classes are:

- refined talc `> talc_threshold_percent`: `talc_bearing`;
- otherwise the sulfide model returns `ordinary` or `difficult`.

## Tests

```bash
pytest -q
```
