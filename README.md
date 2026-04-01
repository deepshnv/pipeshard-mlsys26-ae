# Introduction

This is the README for the artifact evaluation process for our **MLSys26 Paper:** [EFFICIENT, VRAM-CONSTRAINED XLM INFERENCE ON CLIENTS](link arriving soon).

We introduce a technique called pipelined sharding to enable running large LLMs (as well as language decoders of VLMs) at user-specified VRAM budgets through a combination of benchmark profile driven CPU-GPU hybrid scheduling and carefully orchestrated pipelined PCIe copies to avoid exposing copy costs. 
VLMOpt provides complementary VRAM-reduction optimizations for the vision encoder to enable VLMs to handle high-resolution image inference at user-specified VRAM budgets. 

We have implemented both pipelined sharding and VLMOpt in [llama.cpp tag b6097](https://github.com/deepshnv/pipeshard-mlsys26-ae/blob/main/llama-cpp-README.md


## Requirements

1. **Hardware**: An x86_64 machine with an NVIDIA RTX (ideally, an RTX 5090 or 5070 TI) or any A100 or newer compute class GPU.

2. **NVIDIA Driver & CUDA Toolkit** (for NVIDIA GPUs):
   - **Windows**:
     - Install the latest **Game Ready Driver** (GRD) from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx):
       1. Select your GPU product, OS, and "Game Ready Driver" download type.
       2. Download and run the installer; a reboot may be required.
       3. Verify with `nvidia-smi` in a terminal.
     - Install **CUDA Toolkit 12.8+** from [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads):
       1. Select your OS, architecture, and installer type (recommended: network installer).
       2. Run the installer and follow the on-screen prompts.
       3. Ensure `nvcc` is on your PATH. Verify with `nvcc --version`.

   - **Ubuntu 24.04 + NVIDIA A100**:
     1. Install build tools, CMake, Git, and the NVIDIA driver:
        ```bash
        sudo apt update
        sudo apt install -y build-essential dkms linux-headers-$(uname -r) git cmake ninja-build pkg-config ubuntu-drivers-common
        sudo ubuntu-drivers install --gpgpu
        sudo apt install -y \
          linux-modules-nvidia-570-server-open-$(uname -r) \
          linux-modules-nvidia-570-server-open-generic-hwe-24.04 \
          nvidia-driver-570-server-open \
          nvidia-utils-570-server
        sudo reboot
        ```
     2. Verify the driver:
        ```bash
        nvidia-smi
        ```
     3. Install **CUDA Toolkit 12.8**:
        ```bash
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt update
        sudo apt install -y cuda-toolkit-12-8
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
        source ~/.bashrc
        ```
     4. Verify CUDA / toolchain:
        ```bash
        nvcc --version
        cmake --version
        git --version
        ```

3. *(Optional, Windows)* **Visual Studio 2022+** — needed for the MSVC compiler and CMake generator on Windows. Install from [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/) and select the **"Desktop development with C++"** workload during setup.

## Step 1: Build

You can **build from source**, **download pre-built release binaries**, or **use the Docker image**.

#### Option A: Build from Source

```bash
git clone https://github.com/deepshnv/pipeshard-mlsys26-ae.git
cd pipeshard-mlsys26-ae
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j16

On Windows, the CMake configure step also generates a Visual Studio solution (`build/llama.cpp.sln`) that can be opened and built directly in Visual Studio. Binaries are placed in `build/bin/Release/` (Windows) or `build/bin/` (Linux/macOS).

> All development and testing for our paper was done on Windows; we recommend Windows for the smoothest reproduction experience. That said, llama.cpp and our algorithmic changes are platform-independent and should build and run on Linux/macOS as well.
```

> **Linux shared library note:** If you get `error while loading shared libraries: libllama.so: cannot open shared object file` when running any binary, set `LD_LIBRARY_PATH` to include the build output directory:
> ```bash
> export LD_LIBRARY_PATH=$(pwd)/build/src:$LD_LIBRARY_PATH
> ```

### Option B: Pre-built Release Binaries (Windows x86_64)

If you do not wish to build from source, pre-built binaries are available on the [Releases](https://github.com/deepshnv/pipeshard-mlsys26-ae/releases) page.

**Build environment for this Release was:** Windows x86_64 (Intel), MSVC, CUDA Toolkit 12.9, NVIDIA RTX 5090.

```powershell
# Download and extract the release archive
Invoke-WebRequest -Uri "https://github.com/deepshnv/pipeshard-mlsys26-ae/releases/download/v1.0.0-mlsys26/pipeshard-mlsys26-ae-win-x64-cuda12.9.zip" -OutFile release.zip
Expand-Archive release.zip -DestinationPath build\bin\Release
```

**Note:** The pre-built binaries are linked against CUDA 12.9. You still need a compatible NVIDIA driver installed (see [Requirements](#requirements) above). If your CUDA version differs significantly, build from source using Option A.

### Option C: Docker

A pre-built Docker image with all dependencies and compiled binaries is available. This is the easiest way to get started -- no local build or dependency setup required.

<details>
<summary><strong>One-time setup: Docker + NVIDIA GPU support</strong></summary>

1. Install
 
	a. On windows: [Docker Desktop](https://www.docker.com/products/docker-desktop/) (enable WSL 2 backend on Windows, restart after install).
	
	b. On linux: run:-  chmod +x ./install_docker.sh   followed by running ./install_docker.sh script
	
2. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) inside your Host machine i.e. WSL2 or Linux:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo service docker restart
```

3. Restart Docker Desktop, then verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
```

</details>

**Pre-built Docker image (hosted on GitHub Container Registry, GHCR):**

- Where it lives: https://ghcr.io/deepshnv/pipeshard-mlsys26-ae

**To pull and run all reproduction scripts automatically:**

```bash
docker pull ghcr.io/deepshnv/pipeshard-mlsys26-ae:v1.0.0

# Mount your model weights directory; the container runs all 5 repro scripts (Table 4, 8, 9, Figures 2, 7)
docker run --gpus all -v /path/to/your/gguf_models:/workspace/gguf_models ghcr.io/deepshnv/pipeshard-mlsys26-ae:v1.0.0
```

**Or run interactively** (to download models, run scripts individually, inspect results):

```bash
docker run --gpus all -it -v /path/to/your/gguf_models:/workspace/gguf_models ghcr.io/deepshnv/pipeshard-mlsys26-ae:v1.0.0 bash

# Inside the container:
./download_models.sh                              # download models (if not mounted)
./paper_results/repro_table4.sh                   # similarly run other scripts like repro_table8.sh, repro_table9.sh, repro_figure2.sh, repro_figure7.sh
```

> The Dockerfile is in the repository root for transparency. To rebuild locally: `docker build -t pipeshard-mlsys26-ae .`

## Step 2: Set Environment Variables

```bash
# Linux/macOS
export GGML_CUDA_PIPELINE_SHARDING=1
export GGML_CUDA_REGISTER_HOST=1

# Windows (cmd)
set GGML_CUDA_PIPELINE_SHARDING=1
set GGML_CUDA_REGISTER_HOST=1

# Windows (PowerShell)
$env:GGML_CUDA_PIPELINE_SHARDING = "1"
$env:GGML_CUDA_REGISTER_HOST = "1"
```

## Step 3: Run Profilers

The profilers generate benchmark data (`concurrent_results.txt` and `gpu_results.txt`) used by the pipeline executor at runtime.

**Fast profiles (reduced configs, quick turnaround):**

```bash
./concurrent_profiler --cold --fast
./gpu_profiler --cold --fast
```

**Full profiles (all configs, best accuracy):**

```bash
./concurrent_profiler --cold
./gpu_profiler --cold
```

### Step 4: Run Text-Only Inference

```bash
./llama-cli -m <model.gguf> -c <context> -n <max_tokens> -ub <ubatch> \
    -mva <vram_mb> -pipe-shard --temp 0.0 -no-cnv --no-display-prompt
```

**Example (Qwen3-4B, text-only):**

```bash
./llama-cli -m Qwen3-4B-Q4_0.gguf -c 4096 --file prompt-1k.txt \
    --temp 0.0 -no-cnv -n 256 --no-display-prompt \
    -ub 1024 -mva 1500 -pipe-shard
```

## Step 5: Run Multimodal / VLM Inference

For vision-language models, use `llama-mtmd-cli` with the `--mmproj` projector file and VLMOpt flags to keep VRAM usage low during CLIP encoding.

**Set VLMOpt environment variable** (recommended for high-resolution images):

The `MTMD_CLIP_FLASH_ATTN` variable controls Flash Attention inside the CLIP vision encoder. Mode `2` (tiled FA) is strongly recommended for large image resolutions (`-cis` above ~2000).

| Variable | Values | Description |
|----------|--------|-------------|
| `MTMD_CLIP_FLASH_ATTN` | `0` (default), `1`, `2` | `0` = disabled, `1` = full FA (bypasses tiled-Q, one pass), `2` = tiled FA (FA within each tiled-Q chunk). |

```bash
# Linux/macOS
export MTMD_CLIP_FLASH_ATTN=2

# Windows (cmd)
set MTMD_CLIP_FLASH_ATTN=2

# Windows (PowerShell)
$env:MTMD_CLIP_FLASH_ATTN = "2"
```

```bash
./llama-mtmd-cli \
    -m <llm.gguf> --mmproj <projector.gguf> \
    -p "<prompt>" --image <image_path> \
    -c <context> -n <max_tokens> -b <batch> -ub <ubatch> \
    -mva <vram_mb> -pipe-shard \
    -vto-offload-cpu -vto-tiled-attention -clip-tiled-mb <budget_mb> \
    [--chat-template <template>] [-cis <max_dim>] [-fes <strategy>]
```

**Example (Cosmos-Reason1-7B, image captioning):**

```bash
./llama-mtmd-cli \
    -m Cosmos_Reason1_7B.gguf \
    --mmproj mmproj_Cosmos_Reason1_7B.gguf \
    -p "Describe what is happening in this image in under 100 words." \
    --image photo.jpg \
    -c 12000 -n 100 -b 2048 -ub 2048 -mva 3500 -pipe-shard \
    -vto-offload-cpu -vto-tiled-attention -clip-tiled-mb 12000 -cis 3840
```

**Example (Qwen2.5-VL-7B, image captioning):**

```bash
./llama-mtmd-cli \
    -m Qwen2.5-VL-7B-Instruct-q8_0.gguf \
    --mmproj Qwen2.5-VL-7B-Instruct-mmproj-bf16.gguf \
    -p "Describe what is happening in this image in under 100 words." \
    --image photo.jpg \
    -c 12000 -n 100 -b 2048 -ub 2048 -mva 3500 -pipe-shard \
    -vto-offload-cpu -vto-tiled-attention -clip-tiled-mb 12000 -cis 3840
```

## PipeShard CLI Flags

| Flag | Description |
|------|-------------|
| `-pipe-shard` | Enable pipeline sharding |
| `-mva N` | Max VRAM allocation budget in MB |
| `-psa N` | Pinned system memory allocation in GB (0 = auto) |
| `-fes N` | Force execution strategy (`-1` = auto, `0` = FULL_GPU, `1` = FULL_GPU_NO_OUTPUT, `2` = STATIC_ATTN_PRIO, `3` = SHARD_ATTN_PRIO, `4` = FULL_SHARD) |
| `--cpu-profile PATH` | Path to CPU benchmark profile (default: `concurrent_results.txt`) |
| `--gpu-profile PATH` | Path to GPU benchmark profile (default: `gpu_results.txt`) |

## VLMOpt CLI Flags

These flags control vision encoder (CLIP) VRAM optimizations in `llama-mtmd-cli`. They enable high-resolution VLM inference within tight VRAM budgets by offloading and tiling the vision encoder independently of the LLM pipeline.

| Flag | Description |
|------|-------------|
| `-vto-offload-cpu` | Offload all CLIP vision encoder weights to CPU. Weights are streamed to GPU on demand during encoding, freeing VRAM for the LLM. Recommended when VRAM is constrained. |
| `-vto-tiled-attention` | Reduce peak VRAM of the O(N^2) QK and QKV attention tensors inside the vision encoder by processing the Q matrix in smaller tiles. Prevents large transient allocations that can exceed VRAM during high-resolution image encoding. |
| `-clip-tiled-mb N` | VRAM budget (in MiB) for CLIP tiled-Q attention. Controls the tile size: larger values use fewer, bigger tiles (faster); smaller values reduce peak VRAM further but increase encoding passes. Typical range: 2000-12000 MiB. |
| `-cis N` / `-clip-img-size N` | Override the maximum image dimension fed to the vision encoder. Models like Qwen2.5-VL benefit from native-resolution input; set this to the image's longer side (e.g., `3840` for 4K). Default: model-defined (typically 336 or 384). |

> **Note:** When using large image resolutions (e.g., `-cis 3840`), the standard (non-FA) attention path builds a full N x N attention mask that can exceed the 2 GB internal tensor size limit, causing an assertion failure during the CUDA copy operation. To avoid this, set `MTMD_CLIP_FLASH_ATTN=2` (tiled Flash Attention), which eliminates the explicit mask tensor entirely and processes attention within each tiled-Q chunk using FA kernels instead.

# Reproducing MLSys'26 Paper Results

> **Note on reproducibility:** Absolute performance numbers will vary across hardware; the relative speedups and directional trends should remain consistent. For best results, close other GPU-intensive applications (browsers, game launchers, other ML workloads, etc.) before running benchmarks -- any process consuming VRAM reduces the budget available to pipeline sharding and can degrade performance.

**One-command run all:** First, download all required models, then run all 5 reproduction scripts sequentially, then compare against paper:
```powershell
.\download_models.ps1        # Windows – download/verify all models
.\run_all_repro.ps1          # Windows
python compare_all_results.py   # compare reproduced results against paper
```
```bash
# Linux/macOS – make scripts executable first
chmod +x download_models.sh run_all_repro.sh paper_results/*.sh
./download_models.sh
./run_all_repro.sh
python3 compare_all_results.py   # compare reproduced results against paper
```

**Common flags** available on all repro scripts:

| Flag (PowerShell) | Flag (Bash) | Description |
|---|---|---|
| `-FilterModel <name>` | `--filter-model <name>` | Run only this model (Tables 4, Figure 2; download script) |
| `-TerminateOnFailure` | `--terminate-on-failure` | Stop on first error instead of logging and continuing (default: log and continue) |
| `-SkipProfiling` | `--skip-profiling` | Skip hardware profiler runs |
| `-CompareAbsMetricsToo` | `--compare-abs-metrics-too` | Also compare absolute TPS/TTFT values (Tables 4, 9) after reproduction. By default only speedup comparisons (Figures 2, 7, Table 8) are run. |

---

## Step 1: Download Required Models

All models must be placed in the `gguf_models/` directory before running any experiments.

### Download manually

| Model | URL | Download Instructions |
|-------|-----|----------------------|
| `mistral-nemo-minitron-4b-128k-instruct-f16` | [NVIDIA ACE (4B)](https://developer.nvidia.com/downloads/assets/ace/model_zip/mistral-nemo-minitron-4b-128k-instruct_v1.0.0.7z) | Click the URL, extract the `.7z` archive into `gguf_models/` |
| `mistral-nemo-minitron-8b-128k-instruct-f16` | [NVIDIA ACE (8B)](https://developer.nvidia.com/downloads/assets/ace/model_zip/mistral-nemo-minitron-8b-128k-instruct_v1.0.0.7z) | Click the URL, extract the `.7z` archive into `gguf_models/` |
| `Qwen3-30B-A3B-Instruct-2507-q4` | [Hugging Face (Q4_0)](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/main/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf?download=true) | Click the URL to download the single GGUF file, place it in gguf_models/ |
| `Qwen3-235B-A22B-Instruct-2507-q2_k` | [Hugging Face (Q2_K)](https://huggingface.co/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF/tree/main/Q2_K) | Download the two split shards into `gguf_models/Qwen3-235B-A22B/`; the download script auto-creates a symlink so llama.cpp loads them natively (no merge needed). See note below. |
| `Cosmos-Reason1` | [Hugging Face (7B-GGUF)](https://huggingface.co/deepshekhar03/Cosmos-Reason1-7B-GGUF/tree/main) | `huggingface-cli download deepshekhar03/Cosmos-Reason1-7B-GGUF --local-dir gguf_models/cosmos_reason1` |

> **Note on Qwen3-235B (~85 GB):** This model is distributed as two split GGUF shards. The download script downloads both shards and creates a symlink (`Qwen3-235B-A22B-Instruct-2507-Q2_K.gguf` -> shard 1) so that all repro scripts work without merging. llama.cpp automatically discovers and loads all shards from the same directory. If downloading manually, use `wget` instead of `huggingface-cli` to avoid the HF cache doubling disk usage:
> ```bash
> cd gguf_models/Qwen3-235B-A22B
> wget -O Qwen3-235B-A22B-Instruct-2507-Q2_K-00001-of-00002.gguf "https://huggingface.co/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF/resolve/main/Q2_K/Qwen3-235B-A22B-Instruct-2507-Q2_K-00001-of-00002.gguf"
> wget -O Qwen3-235B-A22B-Instruct-2507-Q2_K-00002-of-00002.gguf "https://huggingface.co/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF/resolve/main/Q2_K/Qwen3-235B-A22B-Instruct-2507-Q2_K-00002-of-00002.gguf"
> ln -s Qwen3-235B-A22B-Instruct-2507-Q2_K-00001-of-00002.gguf Qwen3-235B-A22B-Instruct-2507-Q2_K.gguf
> ```

### Setting Up the Hugging Face CLI

The download commands above require Python 3.12+ and `huggingface_hub[cli]<1.0`. Create a venv and install:

**Windows (PowerShell):**
```powershell
winget install Python.Python.3.12          # skip if already installed
python -m venv hf_venv; .\hf_venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
sudo apt install -y python3.12 python3.12-venv   # Ubuntu/Debian (macOS: brew install python@3.12)
python3.12 -m venv hf_venv && source hf_venv/bin/activate
```
> For 7-Zip extraction (`nemo-4b`, `nemo-8b` models), install p7zip first: `sudo apt install -y p7zip-full` (Ubuntu/Debian) or `brew install p7zip` (macOS), then re-run the download script.

**Then (all platforms):**
```bash
pip install "huggingface_hub[cli]<1.0"
huggingface-cli login   # requires a token from https://huggingface.co/settings/tokens
```

After installing the CLI, you can download all models at once (or just one):
```powershell
.\download_models.ps1                     # all models
.\download_models.ps1 -Model qwen-30b     # only qwen-30b
```
```bash
chmod +x download_models.sh
./download_models.sh                          # all models
./download_models.sh --filter-model qwen-30b  # only qwen-30b
```

> Valid model names: `nemo-4b`, `nemo-8b`, `qwen-30b`, `qwen-235b`, `cosmos-reason1`. The NVIDIA ACE models (`nemo-4b`, `nemo-8b`) are downloaded as `.7z` archives and auto-extracted if 7-Zip is installed. If not, install it first: `winget install 7zip.7zip` (Windows) or `sudo apt install p7zip-full` (Linux).


---

## Step 2: Reproduce Table 4 — TPS and TTFT under Pipelined Sharding

> **Important — VRAM budget vs. physical VRAM:** The sweep includes budgets up to 32G. If your GPU has less physical VRAM (e.g., 8 GB, 12 GB), the higher budget columns will naturally show the same performance as your GPU's maximum, even on a 32 GB GPU, setting `-mva 32768` may over-subscribe usable VRAM (the OS and driver generally reserve ~1–2 GB), which can cause degraded performance this is expected and not a bug.

Table 4 measures **tokens per second (TPS)** and **time to first token (TTFT)** for four text-only LLMs under pipeline sharding, sweeping across 4 context sizes (1K, 4K, 16K, 64K tokens) and 7 VRAM budgets (2G, 4G, 6G, 8G, 12G, 24G, 32G) for the models `minitron 4B fp16`, `minitron 8B fp16`, `Qwen3-30B-A3B Q4`, `Qwen3-235B-A22B Q2_K`

The reproduction script automates the entire sweep: enables pipeline-sharding environment variables, runs the hardware profilers, executes all model/context/VRAM combinations, parses TPS and TTFT from the output logs, and writes a CSV summary to `paper_results/table4_results.csv`.

**Windows (PowerShell):**
```powershell
cd pipeshard-mlsys26-ae
.\paper_results\repro_table4.ps1
```

**Linux / macOS:**
```bash
cd pipeshard-mlsys26-ae
chmod +x paper_results/repro_table4.sh
./paper_results/repro_table4.sh
```

**Options:**

| Flag | Description |
|------|-------------|
| `-BinDir` / `--bin-dir` | Path to directory containing `llama-cli` and profiler executables (default: `./build/bin/Release` on Windows, `./build/bin` on Linux) |
| `-ModelsDir` / `--models-dir` | Path to `gguf_models/` directory (default: `./gguf_models`) |
| `-SkipProfiling` / `--skip-profiling` | Skip profiler runs and reuse existing `concurrent_results.txt` / `gpu_results.txt` |

> **Note:** Models not found in `gguf_models/` are automatically skipped with a warning.

---

## Step 3: Reproduce Table 8 — E2EL Speedups for VLM with PipeShard + VLMOpt

Table 8 measures **end-to-end latency (E2EL) speedups** for the Cosmos-Reason1 VLM at 4 image resolutions (480p, 720p, 1080p, 1440p) using pipeline sharding combined with VLMOpt. For each resolution, a **baseline** run (no sharding, no VLMOpt) is compared against **VLMOpt** runs at 3 VRAM budgets (4G, 8G, 14.5G).

The script:
- Runs `llama-mtmd-cli` with the Cosmos-Reason1 model and a test image (`paper_results/dummy_image/165_4k.jpg`)
- Varies the input resolution via `-cis` (640, 1280, 1920, 2560)
- For each resolution: runs baseline first, then VLMOpt at each VRAM budget
- Parses **image encode time**, **image decode time**, **TTFT**, **TPS** from the output
- Computes **E2EL** = encode + decode + TTFT + (100 / TPS) for each run
- Computes the **speedup**

> **Tip:** To monitor peak VRAM usage during runs, use Task Manager (Windows) or `watch -n 0.5 nvidia-smi` (Linux) in a separate terminal.

**Windows (PowerShell):**
```powershell
cd pipeshard-mlsys26-ae
.\paper_results\repro_table8.ps1
```

**Linux / macOS:**
```bash
cd pipeshard-mlsys26-ae
chmod +x paper_results/repro_table8.sh
./paper_results/repro_table8.sh
```

**Options:**

| Flag | Description |
|------|-------------|
| `-VramBudgets` / `--vram-budgets` | Comma-separated VRAM budgets in MB (default: `4096,8192,14848` = 4G, 8G, 14.5G) |
| `-ImagePath` / `--image-path` | Path to the test image (default: `paper_results/dummy_image/165_4k.jpg`) |
| `-BinDir` / `--bin-dir` | Path to directory containing `llama-mtmd-cli` |
| `-SkipProfiling` / `--skip-profiling` | Skip profiler runs |

The output CSV (`paper_results/table8_results.csv`) contains columns: `Resolution, RunType, VramBudget, Encode(msec), Decode(msec), TTFT(msec), TPS, E2EL(msec), Speedup`. Compare the speedup values against the reference in `paper_results/table8.png`.

---

## Step 4: Reproduce Figure 2 -- TTFT/TPS/E2EL Speedups from Pipelined Sharding

In the paper, each bar in Figure 2 represents the **best speedup** across ubatch sizes (1024, 2048) for a given (model, context, VRAM budget) triple.

The baseline caps GPU layer offloading via `-ngl` (pre-profiled values from `benchmark_summary_5090_base.csv`, which depend only on ubatch and VRAM budget, not hardware). Pipeline sharding replaces this with a single `-mva` flag that automatically schedules layers across CPU and GPU.

**Windows (PowerShell):**
```powershell
cd pipeshard-mlsys26-ae
.\paper_results\repro_figure2.ps1 -MaxVramGB 31
```

**Linux / macOS:**
```bash
cd pipeshard-mlsys26-ae
chmod +x paper_results/repro_figure2.sh
./paper_results/repro_figure2.sh --max-vram-gb 31
```

**Options:**

| Flag | Description |
|------|-------------|
| `-MaxVramGB` / `--max-vram-gb` | Max available VRAM in GB on this machine; budgets exceeding this are skipped (default: `31`) |
| `-BinDir` / `--bin-dir` | Path to directory containing `llama-cli` and profiler executables |
| `-ModelsDir` / `--models-dir` | Path to `gguf_models/` directory |
| `-SkipProfiling` / `--skip-profiling` | Skip profiler runs |

> **Note:** This is the longest-running reproduction script (4 models x 4 contexts x 8 budgets x 2 ubatches = up to 256 paired runs). Use `-SkipProfiling` on subsequent runs to skip the hardware profiling step.

The output CSV (`paper_results/figure2_results.csv`) contains per-run baseline and pipeline-sharded metrics (all times in msec) plus speedup columns: `TTFTSpeedup, TPSSpeedup, E2ELSpeedup`. The TTFT/TPS/E2EL speedup numbers should be in the same ballpark as Figure 2 in the paper.

**(Optional) Generate the Figure 2 bar chart:**
```bash
pip install pandas matplotlib
python paper_results/plot_figure2.py --csv paper_results/figure2_results.csv --out paper_results/figure2_repro.png
```

---

## Step 5: Reproduce Table 9 — TPS vs Multi-Request Batch Size across VRAM Budgets

Table 9 measures **tokens per second (TPS)** for `Qwen3-30B-A3B Q4` with multi-request batching across 3 VRAM budgets (2G, 8G, 16G). "Batch size" here means the number of **parallel 1K-context requests** processed simultaneously (not larger context files). The script uses `-np <N>` for N parallel sequences and `-kvu` for unified KV cache, which the paper finds works best for pipelined sharding due to contiguous KV cache updates reducing copy overhead.

**Windows (PowerShell):**
```powershell
cd pipeshard-mlsys26-ae
.\paper_results\repro_table9.ps1
```

**Linux / macOS:**
```bash
cd pipeshard-mlsys26-ae
chmod +x paper_results/repro_table9.sh
./paper_results/repro_table9.sh
```

**Options:**

| Flag | Description |
|------|-------------|
| `-BinDir` / `--bin-dir` | Path to directory containing `llama-batched-bench` and profiler executables |
| `-ModelsDir` / `--models-dir` | Path to `gguf_models/` directory |
| `-SkipProfiling` / `--skip-profiling` | Skip profiler runs |

The output CSV (`paper_results/table9_results.csv`) contains columns: `Model, VramBudget, VramMB, BatchSize, TPS`. TPS should generally increase with batch size at each VRAM budget. At 2G with 64 requests, both baseline and pipelined sharding struggle to fit tensors in VRAM, so performance may plateau or regress. Compare against the reference in `paper_results/table9.png`.

---

## Step 6: Reproduce Figure 7 — TPS Speedups across Batch Sizes

Figure 7 shows how **TPS speedups** from pipelined sharding hold and even scale across different batch sizes for `Qwen3-30B-A3B Q4` at 1K context per request with 3 VRAM budgets (2G, 8G, 16G). For a fair comparison, speedups are calculated for pipelined sharding using **unified KV** (`-kvu`) against the baseline using **non-unified KV** with NGL-capped GPU offloading.

The script uses `llama-batched-bench` for both the baseline and pipelined sharding runs at batch sizes 1, 4, 16, and 64 (parallel 1K-context requests). High speedups at larger batch sizes (especially 64 requests at 8G and 16G) are due to the increased tensor sizes that push baseline to offload more to CPU, while pipeline sharding schedules the work more efficiently.

**Windows (PowerShell):**
```powershell
cd pipeshard-mlsys26-ae
.\paper_results\repro_figure7.ps1
```

**Linux / macOS:**
```bash
cd pipeshard-mlsys26-ae
chmod +x paper_results/repro_figure7.sh
./paper_results/repro_figure7.sh
```

**Options:**

| Flag | Description |
|------|-------------|
| `-BinDir` / `--bin-dir` | Path to directory containing `llama-batched-bench` and profiler executables |
| `-ModelsDir` / `--models-dir` | Path to `gguf_models/` directory |
| `-SkipProfiling` / `--skip-profiling` | Skip profiler runs |

The output CSV (`paper_results/figure7_results.csv`) contains columns: `Model, VramBudget, VramMB, BatchSize, BaseNGL, BaseTPS, PipeshardTPS, TPSSpeedup`. Speedups should generally increase with batch size at higher VRAM budgets. Compare against the reference in `paper_results/figure7.png`.

---

## Comparing Reproduced Results Against Paper

After running any or all reproduction scripts, compare the output CSVs against the paper's reference values:

```bash
python compare_all_results.py
```

By default this runs **speedup comparisons** only (Figure 2, Table 8, Figure 7) — these compare hardware-independent speedup ratios and should match across different GPUs. It prints **PASS** for each metric within 90% of the paper's value, or the achieved ratio (e.g., `0.76x of paper`) for those below. Comparisons are automatically skipped for any results CSV that hasn't been generated yet.

To also compare **absolute TPS/TTFT values** (Table 4, Table 9), which are hardware-dependent and expected to vary across machines:

```bash
python compare_all_results.py --compare-abs-metrics-too
```

Individual comparisons can also be run directly:

```bash
python paper_results/compare_figure2.py     # TTFT/TPS/E2EL speedups
python paper_results/compare_table8.py      # E2EL speedups (VLM)
python paper_results/compare_figure7.py     # TPS speedups (batching)
python paper_results/compare_table4.py      # absolute TPS/TTFT
python paper_results/compare_table9.py      # absolute TPS
```

> **Note:** Speedup comparisons (Figures 2, 7 and Table 8) are hardware-independent and should match closely. Absolute metric comparisons (Tables 4, 9) will vary across GPUs — the directional trends should remain consistent.

# Citation

If you find our work useful, please consider citing it as follows:

```bibtex
@inproceedings{ukarande:26:mlsys,
   title = "Efficient, VRAM-Constrained xLM Inference on Clients",
   author = "Aditya Ukarande and Deep Shekhar and Marc Blackstein and Ram Rangan",
   booktitle = "Proceedings of the 9th Annual Conference on Machine Learning and Systems (MLSys)",
   month = "May",
   year = "2026" }
