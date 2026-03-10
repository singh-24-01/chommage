# PAL: Persona-Augmented Emotional Support — Reproduction

Clean, self-contained reproduction of **PAL: Persona-Augmented Emotional Support
Conversation Generation** (Cheng et al., ACL 2023 Findings).

This repository integrates code from two sources into a single runnable project:
- [PAL](https://github.com/chengjl19/PAL) — the persona-augmented model
- [Emotional-Support-Conversation](https://github.com/thu-coai/Emotional-Support-Conversation)
  (`codes_zcj/`) — the ESConv baseline framework (Liu et al., ACL 2021)

All changes from the originals are documented in [CHANGES.md](CHANGES.md).
All external downloads are listed in [DOWNLOADS.md](DOWNLOADS.md).

> **Platform note**: These instructions are written for **macOS on Apple Silicon (arm64)**
> using [Miniforge](https://github.com/conda-forge/miniforge). The original `env.yml` targets
> Linux + CUDA and does **not** work on Mac Silicon. Follow the steps below instead.

---

## Quick Start

### 1. Create the conda environment

The `env.yml` is not compatible with Mac Silicon (it requires CUDA and old PyTorch).
Create the environment manually:

```bash
conda create -n pal python=3.8 -y
conda activate pal

# Install numpy, pandas, scikit-learn, scipy, statsmodels via Apple + conda-forge
# (Apple channel provides arm64-optimized builds)
conda install -c apple -c conda-forge \
    numpy=1.21.2 pandas scikit-learn=0.24.2 scipy=1.5.4 statsmodels=0.13.5 -y

# Install PyTorch (1.13.1 is the latest version with arm64 wheel support)
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1

# Install remaining Python packages
pip install \
    transformers==4.28.1 \
    pytorch-lightning==1.9.0 \
    gensim==4.3.2 \
    nltk \
    tqdm \
    psutil \
    tokenizers==0.13.3
```

> **Note on CUDA**: Mac Silicon has no NVIDIA GPU. Training runs on CPU (or MPS if
> supported). The `CUDA_VISIBLE_DEVICES=0` in the shell scripts is harmless — it is
> ignored on Mac and the code falls back to CPU automatically.

> **Note on package versions**: These differ from `env.yml` (which targets Linux).
> The versions above are what actually work on arm64 and have been tested for this project.

### 2. Install Java 11 (required for METEOR evaluation)

On Mac, install Java 11 via conda (not `apt`):

```bash
conda activate pal
conda install -c conda-forge openjdk=11 -y
```

Verify:
```bash
java -version   # should print openjdk version "11.x.x"
```

> **Important**: Run this inside the `pal` env. Java installed this way is only available
> when the `pal` env is active.

### 3. Download NLTK data

```bash
python -c "import nltk; nltk.download('punkt')"
```

### 4. Download model weights

On Mac, use `curl` instead of `wget` (or install wget via `brew install wget`).

**BlenderBot-small-90M** (required — PAL base model):
```bash
# From the repository root:
curl -L -o codes/Blenderbot_small-90M/pytorch_model.bin \
    https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/pytorch_model.bin
```

**BART-large-CNN** (required — persona extractor base model):
```bash
mkdir -p persona_extractor/bart-large-cnn
cd persona_extractor/bart-large-cnn
curl -L -o pytorch_model.bin  https://huggingface.co/facebook/bart-large-cnn/resolve/main/pytorch_model.bin
curl -L -o config.json        https://huggingface.co/facebook/bart-large-cnn/resolve/main/config.json
curl -L -o tokenizer.json     https://huggingface.co/facebook/bart-large-cnn/resolve/main/tokenizer.json
curl -L -o vocab.json         https://huggingface.co/facebook/bart-large-cnn/resolve/main/vocab.json
curl -L -o merges.txt         https://huggingface.co/facebook/bart-large-cnn/resolve/main/merges.txt
cd ../..
```
If downloaded manually, update `hparams.model_dir_or_name` in
`persona_extractor/train_bart.py` from `"facebook/bart-large-cnn"` to
`"./bart-large-cnn"`.

**GloVe embeddings** (required for full evaluation):
```bash
# Download and extract
curl -L -o glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.300d.txt
cp glove.6B.300d.txt codes/metric/word2vec/

# Convert to gensim binary format
cd codes/metric/word2vec
python generate_w2v_files.py
cd ../../..

# Clean up source files
rm glove.6B.zip glove.6B.300d.txt
```

**METEOR paraphrase data** (required for METEOR evaluation):
```bash
cd /tmp
curl -L -o meteor-1.5.tar.gz https://github.com/cmu-mtlab/meteor/releases/download/v1.5/meteor-1.5.tar.gz
tar xzf meteor-1.5.tar.gz
cp -r meteor-1.5/data /Users/$USER/Desktop/PAL-Reproduction-master/codes/metric/pycocoevalcap/meteor/
rm -rf meteor-1.5 meteor-1.5.tar.gz
cd -
```

See [DOWNLOADS.md](DOWNLOADS.md) for the complete list of external resources
including optional ones (SimCSE, PersonaChat dataset).

### 5. Prepare data

The persona-augmented dataset (PESConv.json) is already included. To generate
the train/valid/test splits:

```bash
cd codes/_reformat
python process.py --add_persona True
cd ..
```

This produces `train.txt`, `valid.txt`, `test.txt` in `codes/_reformat/`.

Then prepare the tokenized features:

```bash
# From codes/ directory:
bash RUN/prepare_strat.sh
```

### 6. Train the PAL model

```bash
# From codes/ directory:
bash RUN/train_strat.sh
```

Checkpoints are saved under `codes/DATA/strat.strat_persona_attention_final_rebuttal/`.
Training creates a timestamped directory (e.g., `2026-0210174049.1.5e-05.4.1gpu`).

> **Training time on CPU**: Training on Mac Silicon CPU is significantly slower than
> on a GPU. Expect several hours per epoch.

### 7. Run inference

The inference script auto-detects the latest training run directory and selects
the best epoch (lowest validation loss from `eval_log.csv`):

```bash
# From codes/ directory:
bash RUN/infer_strat.sh                # auto-detect latest run + best epoch
bash RUN/infer_strat.sh 4              # auto-detect latest run, use epoch 4
bash RUN/infer_strat.sh 4 MY_RUN_DIR   # use specific run dir + epoch
```

This produces `gen.json` and `gen.txt` under a `res_...` subdirectory of your
run directory, with generated responses and automatic metrics (BLEU, ROUGE-L,
Distinct, etc.).

### 8. Additional evaluation

These scripts must be run from the `codes/` directory:

**EAD score** (Expectancy-Adjusted Distinct):
```bash
python get_EAD_score.py --input_file ./DATA/strat.strat_persona_attention_final_rebuttal/<run_dir>/<res_dir>/gen.json
```

**Cosine similarity** (persona-response alignment via SimCSE):

Install `git-lfs` via Homebrew first (needed to download large model files):
```bash
brew install git-lfs
git lfs install
```

Then clone the SimCSE model into `codes/`:
```bash
cd codes
git clone https://huggingface.co/princeton-nlp/sup-simcse-bert-base-uncased simcse-bert-base-uncased
cd simcse-bert-base-uncased && git lfs pull && cd ..
```

Then run:
```bash
python get_cos_similarity.py \
    --input_file ./DATA/strat.strat_persona_attention_final_rebuttal/<run_dir>/<res_dir>/gen.json \
    --simcse_model ./simcse-bert-base-uncased
```

---

## Reproduction Results

Results from our reproduction (Mac Silicon, CPU, seed=13, lr=1.5e-5, warmup=0, epoch 4)
compared to the paper (Table 3):

| Metric | Paper (PAL) | Reproduction | Notes |
|--------|------------|--------------|-------|
| ACC    | 34.51      | 32.98        | Strategy classification accuracy |
| PPL    | 15.92      | 15.55        | Perplexity (lower is better) |
| B-2    | 8.75       | 8.63         | BLEU-2 |
| B-4    | 2.66       | 2.59         | BLEU-4 |
| D-1    | 5.00       | 3.72         | Distinct-1 |
| D-2    | 30.27      | 19.84        | Distinct-2 |
| E-1    | 6.73       | 4.78         | EAD-1 |
| E-2    | 41.82      | 26.63        | EAD-2 |
| R-L    | 18.06      | 17.73        | ROUGE-L |
| Cos-Sim| 0.244      | 0.235        | SimCSE cosine similarity |

**Analysis**: Most metrics are close to paper values. The main gap is in
diversity metrics (D-1, D-2, E-1, E-2), which are consistently lower. This may
be due to differences in training environment (the paper used multi-GPU
training) or unreported settings. PPL is actually slightly better than the paper.

**Hyperparameter note**: The paper reports lr=2.5e-5 and warmup=100, but the
code defaults to lr=1.5e-5 and warmup=0. We tested both configurations and
found the code defaults produce better overall results. See
[CHANGES.md](CHANGES.md) for details.

---

## Repository Structure

```
PAL-Reproduction/
├── env.yml                          # Conda environment (Linux/CUDA only — do not use on Mac)
├── README.md                        # This file
├── CHANGES.md                       # All changes from original repos
├── DOWNLOADS.md                     # External downloads with URLs
│
├── persona_extractor/               # BART persona extractor (trained on PersonaChat)
│   ├── process_bart_df.py           #   PersonaChat data preprocessing
│   ├── train_bart.py                #   Training & inference script
│   └── data/                        #   Place PersonaChat .txt files here
│
└── codes/                           # Main PAL model
    ├── _reformat/                   #   Data preprocessing
    │   ├── process.py               #     ESConv/PESConv → train/valid/test splits
    │   ├── strategy.json            #     8 ESC strategy definitions
    │   └── PESConv.json             #     Persona-augmented ESConv dataset
    ├── Blenderbot_small-90M/        #   Tokenizer files (download weights separately)
    ├── CONFIG/                      #   Model configurations
    │   ├── strat.json               #     PAL config (with persona tokens)
    │   └── strat_no_persona.json    #     Ablation config (no persona attention)
    ├── RUN/                         #   Shell scripts (all with relative paths)
    ├── DATA/                        #   Output dir for processed data & checkpoints
    ├── inputters/                   #   Data loading & feature extraction
    ├── models/                      #   Model definitions
    │   ├── strat_blenderbot_small.py          # PAL model (core contribution)
    │   ├── strat_blenderbot_small_no_persona.py  # Ablation (no persona attention)
    │   └── vanilla_blenderbot_small.py        # Vanilla baseline
    ├── metric/                      #   Evaluation metrics (BLEU, ROUGE, Distinct, etc.)
    ├── utils/                       #   Training infrastructure
    ├── apex/                        #   NVIDIA Apex (optional, for FP16 — not used on Mac)
    ├── prepare.py                   #   Tokenize data → DATA/
    ├── train.py                     #   Training loop
    ├── infer.py                     #   Batch inference + automatic evaluation
    ├── interact.py                  #   Interactive chat demo
    ├── get_EAD_score.py             #   Expectancy-Adjusted Distinct score
    └── get_cos_similarity.py        #   SimCSE persona-response similarity
```

---

## Training the Persona Extractor (optional)

The PESConv.json dataset already contains persona annotations, so this step is
only needed if you want to reproduce the persona extraction process or use the
extractor for interactive mode.

1. Download PersonaChat data (see [DOWNLOADS.md](DOWNLOADS.md)) into
   `persona_extractor/data/`.

2. Preprocess:
   ```bash
   cd persona_extractor
   python process_bart_df.py
   ```

3. Train:
   ```bash
   python train_bart.py
   ```
   Checkpoints are saved under `persona_extractor/pl_root/`.

---

## Citations

If you use this code, please cite both papers:

```bibtex
@inproceedings{cheng2023pal,
  title={PAL: Persona-Augmented Emotional Support Conversation Generation},
  author={Cheng, Jiale and Sabour, Sahand and Sun, Hao and Chen, Zhuang and Huang, Minlie},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2023},
  year={2023}
}

@inproceedings{liu2021towards,
  title={Towards Emotional Support Dialog Systems},
  author={Liu, Siyang and Zheng, Chujie and Demasi, Orianna and Sabour, Sahand and Li, Yu and Yu, Zhou and Jiang, Yong and Huang, Minlie},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics},
  year={2021}
}
```
