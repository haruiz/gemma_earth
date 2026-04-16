# Gemma Earth

## Project Overview

The GemmaEarth project is a domain-focused Tunix post-training and benchmarking effort that fine-tunes Google’s Gemma 3 4B IT model for Earth Observation (EO) understanding, beginning with satellite scene classification on the EarthDial dataset, specifically the BigEarthNet subset—a large-scale Sentinel benchmark for multi-label land-use and land-cover classification.

The project showcases an end-to-end JAX-based workflow, using Tunix for LoRA fine-tuning and model loading, Grain for efficient data pipelines, Optax for optimization and learning-rate scheduling, and Orbax for checkpointing. Within this pipeline, Qwix is used to inject LoRA adapters into the base model and support parameter-efficient adaptation in a sharded training setup on TPUs.

Training and evaluation are conducted on Google Cloud TPU v5litepod-8, with an emphasis on scalable benchmarking and a longer-term goal of expanding toward multitask EO reasoning across the full EarthDial dataset.

## Impact

This project explores how a general-purpose multimodal foundation model can be adapted to the unique challenges of Earth Observation data using the [JAX ecosystem](https://jaxstack.ai/). Compared with traditional GPU-based workflows and implementations, this TPU-oriented approach offers potential gains in scalability and efficiency, enabling faster iteration and experimentation at scale.

The implementation is intentionally modular and task-ready, making it easier to expand from scene classification to multiple EO tasks such as captioning, visual question answering, and multimodal reasoning on EarthDial with minimal pipeline changes.

## Experimental Features

* EarthDial dataset preparation pipeline
* LoRA-based fine-tuning (JAX/Flax + Tunix + Qwix)
* Checkpoint compatibility management
* Evaluation pipeline with JSON outputs
* FastAPI + Chainlit inference serving workflow

## Libraries

| Library               | Role                                         |
| --------------------- | -------------------------------------------- |
| JAX + Flax (NNX) + XLA| High-performance tensor computation and model definition |
| Tunix                 | LoRA fine-tuning, model loading, and generation helpers |
| Qwix                  | Parameter-efficient LoRA injection utilities |
| Optax                 | Optimization and learning-rate scheduling    |
| Orbax                 | Checkpoint save/restore                      |
| Grain                 | Data pipeline construction and iteration     |
| Hugging Face Datasets | Dataset loading                              |
| Hugging Face Hub      | Dataset and checkpoint access                |
| FastAPI + Uvicorn     | Inference API serving                        |
| Chainlit              | Lightweight chat UI for inference            |
| TensorBoard           | Training visualization                       |

## Requirements

* Python 3.11+
* Linux environment
* Access to Gemma checkpoints
* Hugging Face token

## Project Layout

```bash
main.py                         # entrypoint example
src/gemma_earth/__init__.py    # Public package exports
src/gemma_earth/trainers/      # Base + source-specific trainer implementations
src/gemma_earth/dataset.py     # Dataset pipeline
src/gemma_earth/config.py      # Config via .env
.env.*                         # Environment profiles
```


## Setup

```bash
git clone https://github.com/haruiz/gemma_earth
uv sync
cp .env.pro .env
```

Update required variables:

* `HF_TOKEN`
* `OUTPUT_DIR`
* `DATASET_DOWNLOAD_DIR`

### TPU VM And Disk Setup

The scripts in [scripts/create-vm.sh](scripts/create-vm.sh), [scripts/create-disk.sh](scripts/create-disk.sh), and [scripts/resize-disk.sh](scripts/resize-disk.sh) automate infrastructure setup.

1. Create the TPU VM (default accelerator: `v5litepod-8`):

```bash
chmod +x scripts/*.sh
PROJECT_ID=<your-gcp-project> \
ZONE=<TPU_ZONE> \
TPU_NAME=<TPU_VM_NAME> \
ACCELERATOR_TYPE=v5litepod-8 \
./scripts/create-vm.sh
```

2. Create, attach, format, and mount a persistent disk:

```bash
PROJECT_ID=<your-gcp-project> \
ZONE=<TPU_ZONE> \
TPU_NAME=<TPU_VM_NAME> \
DISK_NAME=<DATA_DISK_NAME> \
DISK_SIZE=<DISK_SIZE_GB> \
MOUNT_POINT=<MOUNT_POINT> \
./scripts/create-disk.sh
```

3. Resize the disk later if you need more space:

```bash
PROJECT_ID=<your-gcp-project> \
ZONE=<TPU_ZONE> \
TPU_NAME=<TPU_VM_NAME> \
DISK_NAME=<DATA_DISK_NAME> \
NEW_SIZE_GB=<NEW_DISK_SIZE_GB> \
MOUNT_POINT=<MOUNT_POINT> \
./scripts/resize-disk.sh
```

Notes:

* Use the same `TPU_NAME` and `ZONE` across all three scripts.
* Run `./scripts/<script>.sh --help` to see all supported environment variables.

### Run Inference UI Remotely From A TPU VM

Use this flow when the model runs on the TPU VM and you want to access the UI from your local machine.

1. Add an SSH host entry on your local machine (in `~/.ssh/config`):

```sshconfig
Host <SSH_HOST_ALIAS>
  HostName <TPU_VM_EXTERNAL_IP>
  User <SSH_USER>
  IdentityFile ~/.ssh/google_compute_engine
    IdentitiesOnly yes
    CheckHostIP no
    StrictHostKeyChecking no
  UserKnownHostsFile ~/.ssh/google_compute_known_hosts
    ServerAliveInterval 60
```

2. Add the TPU host key to your known hosts file (run locally):

```bash
ssh-keyscan -H <TPU_VM_EXTERNAL_IP> >> ~/.ssh/google_compute_known_hosts
```

3. SSH into the TPU VM and start the API server:

```bash
ssh <SSH_HOST_ALIAS>
cd ~/workspace/<REPO_NAME>

uv run python scripts/serve_fastapi.py \
  --model-checkpoint-source tunix \
  --model-dir <MODEL_CHECKPOINT_DIR> \
  --host 127.0.0.1 \
  --port 8000
```

4. In another shell on the same TPU VM, start the Chainlit UI:

```bash
cd ~/workspace/<REPO_NAME>
uv run chainlit run scripts/serve_ui.py --host 127.0.0.1 --port 8501
```

5. From your local machine, open an SSH tunnel to the UI:

```bash
ssh -N -L 8501:127.0.0.1:8501 <SSH_HOST_ALIAS>
```

6. Open the UI locally:

```text
http://localhost:8501
```

Optional: tunnel both API and UI if you want local access to both endpoints:

```bash
ssh -N -L 8000:127.0.0.1:8000 -L 8501:127.0.0.1:8501 <SSH_HOST_ALIAS>
```

## Experiment Parameters And Infrastructure

### Fine-Tuning Parameters (.env)

| Parameter | Value | Description |
| --- | --- | --- |
| Base model checkpoint | `gs://gemma-data/checkpoints/gemma3-4b-it` | Pretrained Gemma 3 4B IT checkpoint used as the starting point for LoRA fine-tuning. |
| Base model checkpoint source | `tunix` | Selects checkpoint loader implementation (`tunix` or `huggingface`). |
| HF model ID | `google/gemma-3-4b-it` | Hugging Face model repo used when checkpoint source is `huggingface`. |
| HF ignore patterns | `*.pth` | Comma-separated patterns ignored during HF snapshot download. |
| Tokenizer | `gs://gemma-data/tokenizers/tokenizer_gemma3.model` | Tokenizer model used to convert prompts and responses into token IDs. |
| LoRA rank | `32` | Adapter rank; higher rank increases trainable LoRA capacity and memory usage. |
| LoRA alpha | `64.0` | LoRA scaling factor controlling effective update strength of adapters. |
| Max sequence length | `768` | Maximum token length per sample after truncation and padding. |
| Learning rate | `3e-5` | Step size used by the optimizer for parameter updates. |
| Weight decay | `1e-4` | L2-style regularization applied during optimization to reduce overfitting. |
| Warmup ratio | `0.1` | Fraction of training steps used for learning-rate warmup before decay. |
| Batch size | `4` | Number of samples processed per training step. |
| Num epochs | `4` | Number of full passes over the sampled training data. |
| Eval every n steps | `100` | Frequency (in steps) for running validation/evaluation during training. |
| Checkpoint save interval | `200` | Frequency (in steps) for saving model checkpoints. |
| Max checkpoints kept | `5` | Maximum number of recent checkpoints retained on disk. |
| Num samples | `40000` | Cap on total dataset examples used from EarthDial for this run. |
| Validation split ratio | `0.1` | Proportion of sampled data allocated to validation split. |
| Min validation samples | `500` | Lower bound for validation split size after ratio-based calculation. |
| Max validation samples | `1500` | Upper bound for validation split size after ratio-based calculation. |
| Shuffle seed | `42` | Random seed for deterministic dataset shuffling and split reproducibility. |
| Preserve multi-turn | `true` | Keeps full multi-turn conversations instead of reducing to a single QA pair. |
| Dataset repo | `akshaydudhane/EarthDial-Dataset` | Hugging Face dataset repository used as training source. |
| Dataset allow pattern | `training_set/**` | File pattern that filters which dataset files are downloaded/used. |

### TPU And VM Features (scripts)

| Component | Configuration |
| --- | --- |
| TPU model | `v5litepod-8` |
| TPU VM name | `<TPU_VM_NAME>` |
| TPU runtime image | `tpu-ubuntu2204-base` |
| Zone strategy | Primary `<PRIMARY_TPU_ZONE>` with fallback zones (`<FALLBACK_ZONE_1>`, `<FALLBACK_ZONE_2>`, `<FALLBACK_ZONE_3>`, `<FALLBACK_ZONE_4>`) |
| Service account | `tpu-vm-sa` with TPU admin, Storage admin, Logging writer, Monitoring metric writer roles |
| VM bootstrap | Installs `python3`, `python3-venv`, `curl`, `ca-certificates`, and `uv` at `/opt/uv` |
| Data disk type | `pd-ssd` |
| Disk size flow | Created at `1500GB` disk |
| Mount point | `/mnt/disks/data` |
| Disk attachment mode | Read-write attached to TPU VM |


## Training & Evaluation

Train:

```bash
python main.py train
```

Train with explicit checkpoint source:

```bash
python main.py train --model-checkpoint-source tunix
python main.py train --model-checkpoint-source huggingface
```

Evaluate:

```bash
python main.py eval
```

Custom evaluation:

```bash
python main.py eval --start-index 200 --num-examples 30 --model-checkpoint-source tunix
```

One-example evaluation (single image):

```bash
python scripts/one_example_eval.py \
  --model-checkpoint-source huggingface \
  --model-dir /path/to/hf_checkpoint_dir \
  --image-path /path/to/image.jpg

python scripts/one_example_eval.py \
  --model-checkpoint-source tunix \
  --model-dir gs://gemma-data/checkpoints/gemma3-4b-it \
  --image-path /path/to/image.jpg
```

Outputs:

```bash
results.json
```

## Checkpoints

* `--train-restore-policy` supports `strict` and `permissive` (default: `permissive`).
* `--eval-restore-policy` supports `strict` and `permissive` (default: `permissive`).
* `permissive` restore works for both checkpoint sources (`tunix` and `huggingface`).
* `scripts/one_example_eval.py` supports both `tunix` and `huggingface`.

Use consistent `.env` profiles to avoid mismatches.

## Recommended Workflow

**Standard workflow**

1. Train
2. Evaluate

**Quick testing**

1. Use `.env.smoke`
2. Run short training + evaluation

## Future Work

Planned next steps for this project include:

* Scaling fine-tuning from the current subset to the full EarthDial dataset.
* Attempting multitask fine-tuning across classification, captioning, visual question answering, and reasoning tasks.
* Evaluating cross-task transfer to measure whether multitask training improves generalization on unseen EO scenarios.
* Studying trade-offs between task balance, training stability, and compute cost on larger TPU runs.

## Contributions

This experiment demonstrates that a Gemma 3 4B IT model can be adapted to remote-sensing scene classification with a lightweight LoRA pipeline on top of EarthDial-derived data.

Main contributions:

* Built an end-to-end EarthDial training/evaluation workflow with reproducible `.env` profiles.
* Implemented an `EarthDialDataset` class with a Grain-based, JAX-compatible data pipeline for efficient shuffling, mapping, batching, and iteration.
* Added checkpoint-source flexibility (`tunix` and `huggingface`) for base model loading.
* Added compatibility-aware checkpoint organization to reduce silent mismatch failures across runs.
* Added single-image inference tooling for quick qualitative validation (`scripts/one_example_eval.py`).
* Produced a practical baseline focused on EO classification that can be extended to other EarthDial tasks.

## Extending To Other EarthDial Tasks

Current training in this repo is focused on classification-style examples. To further fine-tune on the rest of EarthDial tasks (captioning, VQA, reasoning, etc.), use the dataset pipeline in [dataset.py](src/gemma_earth/dataset.py):

1. Define task selection in `EarthDialDataset`:
Add task filtering before `_build_pipeline(...)` in `build(...)`. Use metadata fields available in each row (for example task/category fields in conversations or sample metadata) and create per-task or mixed-task splits.

2. Adjust prompt/response formatting:
Update `_format_prompt_and_response(...)` to preserve task-specific instruction style. For generative tasks, keep descriptive answers unchanged; for short-answer tasks, enforce concise target formats.

3. Add task-aware preprocessing:
Extend `_to_training_example(...)` to emit optional task identifiers (for example prefixes like `Task: Captioning` in prompt text) so a single model can learn multiple behaviors.

4. Tune sampling and balance:
In `build(...)`, rebalance tasks before split (or oversample underrepresented tasks) to avoid classification dominating the objective.

5. Validate with task-specific eval:
Reuse `eval(...)` in the trainer as a template, but add task-specific metrics and output normalization per task (for example exact-match for VQA, text quality metrics for captioning).

6. Start with controlled multitask runs:
Begin with a small subset of 2-3 tasks, verify convergence, then scale to full multitask training once prompt formatting and balancing are stable.


## Credits

This project builds upon:

* EarthDial Dataset
* Gemma 3 4B IT by Google
* [BigEarthNet](https://bigearth.net/) — A Large-Scale Sentinel Benchmark Archive for remote sensing scene classification
* The open-source JAX/Flax ecosystem

## Citation

If you use this project, please cite:

```bibtex
@misc{gemma_earth_2026,
  title={Gemma Earth: Fine-tuning Gemma for Remote Sensing Scene Classification},
  author={<AUTHOR_NAME>},
  year={2026},
  howpublished={GitHub repository},
  url={https://github.com/<GITHUB_USER_OR_ORG>/<REPO_NAME>}
}
```

### Related Work

```bibtex
@misc{soni2024earthdial,
  title={EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues},
  author={Soni, Sagar and Dudhane, Akshay and Debary, Hiyam and Fiaz, Mustansar and Munir, Muhammad Akhtar and Danish, Muhammad Sohail and Fraccaro, Paolo and Watson, Campbell D and others},
  year={2024},
  eprint={2412.15190},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  doi={10.48550/arXiv.2412.15190},
  url={https://arxiv.org/abs/2412.15190}
}


@misc{sumbul2019bigearthnet,
  title={BigEarthNet: A Large-Scale Benchmark Archive For Remote Sensing Image Understanding},
  author={Sumbul, Gencer and Charfuelan, Marcela and Demir, Beg{\"u}m and Markl, Volker},
  year={2019},
  eprint={1902.06148},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  doi={10.48550/arXiv.1902.06148},
  url={https://arxiv.org/abs/1902.06148}
}

@misc{tunix2025,
  title={Tunix (Tune-in-JAX)},
  author={Bao, Tianshu and Carpenter, Jeff and Chai, Lin and Gao, Haoyu and Jiang, Yangmu and Noghabi, Shadi and Sharma, Abheesht and Tan, Sizhi and Wang, Lance and Yan, Ann and Yu, Weiren and others},
  year={2025},
  howpublished={\url{https://github.com/google/tunix}},
}

```

## Acknowledgements

Google Cloud credits are provided for this project #TPUSprint

## Tips

### Save TPU Costs When Idle

To reduce cost, stop the TPU VM when you are not using it.

When you start it again, the external IP can change. Refresh the SSH connection info by running:

```bash
gcloud alpha compute tpus tpu-vm ssh <SSH_USER>@<TPU_VM_NAME> --zone=<TPU_ZONE> --tunnel-through-iap --dry-run
```

Then:

1. Copy the updated external IP from the dry-run output.
2. Update the `HostName` value in your `~/.ssh/config` entry for `<SSH_HOST_ALIAS>`.
3. Reconnect with:

```bash
ssh <SSH_HOST_ALIAS>
```
