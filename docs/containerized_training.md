# Containerized training

`Dockerfile` + `scripts/train-entrypoint.sh` let you train on managed
training services (pay-per-job, higher specs than a long-running ECS)
or plain `docker run`. Data is **mounted**, never baked into the image.

## Build

```bash
docker build -t quant2026-train .
```

The build context excludes `data/`, `out/`, and model artifacts
(see `.dockerignore`), so it stays small and fast.

## Run locally (data on disk)

```bash
docker run --rm \
  -v "$(pwd)/data:/mnt/data:ro" \
  -v "$(pwd)/out:/mnt/output" \
  -e TRAIN_ARGS="--partitions 2 --n-folds 3 --save-model" \
  quant2026-train
```

Model + CV checkpoints land in `./out`. Default run (no `TRAIN_ARGS`,
no args) does a full `--save-model` over all partitions.

## Managed platforms

Mount input data at `/mnt/data` (with `manifest.json`), write artifacts to
`/mnt/output`. Pass flags via `TRAIN_ARGS` or as container args.

### GCP Vertex AI (Custom Container Training)
- Push image to Artifact Registry: `docker tag ... REGION-docker.pkg.dev/PROJECT/REPO/quant2026-train` then `docker push`.
- Put `data/` in a GCS bucket, mount it (or set `INPUT_URI=gs://bucket/quant2026/data`).
- Use a `n1-highmem-16` (16 vCPU / 128 GB) or `n1-highmem-8` (64 GB) worker pool. CPU only — no GPU needed for LightGBM.

### Aliyun PAI-DLC
- Push image to ACR; store data on OSS.
- `INPUT_URI=oss://bucket/quant2026/data` (the entrypoint auto-selects `ossutil`).
- Pick a CPU instance with ≥64 GB memory; PAI mounts the OSS/CPFS data path to `/mnt/data`.

### AWS SageMaker Training
- ECR image, S3 data; set `INPUT_URI=s3://bucket/quant2026/data`.
- `ml.m5.4xlarge` (16 vCPU / 64 GB) or `ml.m5.8xlarge` (128 GB).

## Pulling the model back

After the job, the trained model is in the output path
(`strategy/model.txt`, `model_meta.json`, `cv/`). Download locally and run
the inference validator + generate a submission:

```bash
python timeseries_api/run_timeseries_api.py \
  --data-root data --strategy-dir strategy --output out/sub.csv
```

## Notes
- **Spot/抢占式 interruption safety**: `train.py` checkpoints each fold to
  `<out>/cv/`; re-running resumes from completed folds. Pair this with the
  platform's "auto-retry on preemption" for cheap, resilient runs.
- **Inference constraint**: the final eval env is 4 vCPU / 12 GB / no GPU.
  Train only models that fit there — the current LightGBM booster (~500 KB) does.
- **GPU**: skip for LightGBM (negligible gain). Add a GPU image + PyTorch
  later only if you move to neural models.
