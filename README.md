# Gemma Earth

Scaling automated analysis of Earth Observation (EO) data is critical for environmental monitoring, disaster response, and resource management. While general-purpose Vision-Language Models (VLMs) perform well on natural imagery, they often struggle with remote sensing data due to multi-sensor complexity, varying resolutions, and fine-grained spatial patterns.

The EarthDial Dataset bridges this gap with **11M+ multimodal instruction-tuning pairs** across RGB, SAR, and multispectral imagery, enabling conversational EO systems for classification, detection, captioning, and reasoning.

This project fine-tunes Gemma 3 4B IT for **scene classification**, using an EarthDial subset derived from the FAIR1M benchmark. FAIR1M is a large-scale, high-resolution remote sensing dataset designed for fine-grained object recognition, featuring diverse scenes with complex spatial patterns, scale variation, and object orientations.

Using **LoRA fine-tuning** with JAX/Flax and Tunix, the model learns to accurately interpret land-cover patterns and classify satellite imagery across diverse EO scenarios.

## Experimental Features

* EarthDial dataset preparation pipeline
* LoRA-based fine-tuning (JAX/Flax + Tunix)
* Checkpoint compatibility management
* Evaluation pipeline with JSON outputs

## Libraries

| Library               | Role                                         |
| --------------------- | -------------------------------------------- |
| JAX + Flax (NNX)      | Model definition and accelerated computation |
| Tunix                 | Gemma loading, LoRA training, preprocessing  |
| Optax                 | Optimization and scheduling                  |
| Orbax                 | Checkpointing                                |
| Grain                 | Data pipelines                               |
| Hugging Face Datasets | Dataset loading                              |
| Hugging Face Hub      | Dataset access                               |
| Pillow                | Image processing                             |
| NumPy                 | Tensor manipulation                          |
| Pydantic Settings     | Config management                            |
| Qwix                  | Distributed/TPU setup                        |
| TensorBoard           | Training visualization                       |

## Requirements

* Python 3.11+
* Linux environment
* Access to Gemma checkpoints
* Hugging Face token

## Project Layout

```bash
main.py                         # entrypoint example
src/gemma_earth/trainer.py     # Training + evaluation logic
src/gemma_earth/dataset.py     # Dataset pipeline
src/gemma_earth/config.py      # Config via .env
.env.*                         # Environment profiles
```


## Setup

```bash
uv sync
cp .env.example .env
```

Update required variables:

* `HF_TOKEN`
* `OUTPUT_DIR`
* `DATASET_DOWNLOAD_DIR`

### TPU VM And Disk Setup (Optional, GCP)

The scripts in [scripts/create-vm.sh](scripts/create-vm.sh), [scripts/create-disk.sh](scripts/create-disk.sh), and [scripts/resize-disk.sh](scripts/resize-disk.sh) automate infrastructure setup.

1. Create the TPU VM (default accelerator: `v5litepod-8`):

```bash
chmod +x scripts/*.sh
PROJECT_ID=<your-gcp-project> \
ZONE=us-west1-c \
TPU_NAME=tpu-sprint-machine \
ACCELERATOR_TYPE=v5litepod-8 \
./scripts/create-vm.sh
```

2. Create, attach, format, and mount a persistent disk:

```bash
PROJECT_ID=<your-gcp-project> \
ZONE=us-west1-c \
TPU_NAME=tpu-sprint-machine \
DISK_NAME=data-disk \
DISK_SIZE=1500GB \
MOUNT_POINT=/mnt/disks/data \
./scripts/create-disk.sh
```

3. Resize the disk later if you need more space:

```bash
PROJECT_ID=<your-gcp-project> \
ZONE=us-west1-c \
TPU_NAME=tpu-sprint-machine \
DISK_NAME=data-disk \
NEW_SIZE_GB=2000 \
MOUNT_POINT=/mnt/disks/data \
./scripts/resize-disk.sh
```

Notes:

* Use the same `TPU_NAME` and `ZONE` across all three scripts.
* Run `./scripts/<script>.sh --help` to see all supported environment variables.

## Experiment Parameters And Infrastructure

### Fine-Tuning Parameters (.env)

| Parameter | Value | Description |
| --- | --- | --- |
| Base model checkpoint | `gs://gemma-data/checkpoints/gemma3-4b-it` | Pretrained Gemma 3 4B IT checkpoint used as the starting point for LoRA fine-tuning. |
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
| TPU VM name | `tpu-sprint-machine` |
| TPU runtime image | `tpu-ubuntu2204-base` |
| Zone strategy | Primary `us-central1-a` with fallback zones (`us-south1-a`, `us-west1-c`, `us-west4-a`, `europe-west4-b`) |
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

Evaluate:

```bash
python main.py eval
```

Custom evaluation:

```bash
python main.py eval --start-index 200 --num-examples 30
```

Outputs:

```bash
results.json
```

## Checkpoints

* `train` → strict restore
* `eval` → permissive restore

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


## Credits

This project builds upon:

* EarthDial Dataset
* Gemma 3 4B IT by Google
* FAIR1M dataset for high-resolution remote sensing benchmarks
* The open-source JAX/Flax ecosystem

## Citation

If you use this project, please cite:

```bibtex
@misc{gemma_earth_2026,
  title={Gemma Earth: Fine-tuning Gemma for Remote Sensing Scene Classification},
  author={Henry Ruiz Guzman},
  year={2026},
  howpublished={GitHub repository},
  url={https://github.com/<your-username>/gemma-earth}
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

@misc{sun2021fair1m,
  title={FAIR1M: A Benchmark Dataset for Fine-grained Object Recognition in High-Resolution Remote Sensing Imagery},
  author={Sun, Xian and Wang, Peijin and Yan, Zhiyuan and Xu, Feng and Wang, Ruiping and Diao, Wenhui and Chen, Jin and Li, Jihao and Feng, Yingchao and Xu, Tao and others},
  year={2021},
  eprint={2103.05569},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  doi={10.48550/arXiv.2103.05569},
  url={https://arxiv.org/abs/2103.05569}
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
