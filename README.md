# llama.cpp

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)

[Manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md)

LLM inference in C/C++

## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Hot topics

- Support for the `gpt-oss` model with native MXFP4 format has been added | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [Collaboration with NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [Comment](https://github.com/ggml-org/llama.cpp/discussions/15095)
- Hot PRs: [All](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr+label%3Ahot+) | [Open](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr+label%3Ahot+is%3Aopen)
- Multimodal support arrived in `llama-server`: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [documentation](./docs/multimodal.md)
- VS Code extension for FIM completions: https://github.com/ggml-org/llama.vscode
- Vim/Neovim plugin for FIM completions: https://github.com/ggml-org/llama.vim
- Introducing GGUF-my-LoRA https://github.com/ggml-org/llama.cpp/discussions/10123
- Hugging Face Inference Endpoints now support GGUF out of the box! https://github.com/ggml-org/llama.cpp/discussions/9669
- Hugging Face GGUF editor: [discussion](https://github.com/ggml-org/llama.cpp/discussions/9268) | [tool](https://huggingface.co/spaces/CISCai/gguf-editor)

----

## Quick start

Getting started with llama.cpp is straightforward. Here are several ways to install it on your machine:

- Install `llama.cpp` using [brew, nix or winget](docs/install.md)
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed, you'll need a model to work with. Head to the [Obtaining and quantizing models](#obtaining-and-quantizing-models) section to learn more.

Example command:

```sh
# Use a local model file
llama-cli -m my_model.gguf

# Or download and run a model directly from Hugging Face
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

# Launch OpenAI-compatible API server
llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

## Description

The main goal of `llama.cpp` is to enable LLM inference with minimal setup and state-of-the-art performance on a wide
range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is the main playground for developing new features for the [ggml](https://github.com/ggml-org/ggml) library.

<details>
<summary>Models</summary>

Typically finetunes of the base models below are supported as well.

Instructions for adding support for new models: [HOWTO-add-model.md](docs/development/HOWTO-add-model.md)

#### Text-only

- [X] LLaMA 🦙
- [x] LLaMA 2 🦙🦙
- [x] LLaMA 3 🦙🦙🦙
- [X] [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [x] [DBRX](https://huggingface.co/databricks/dbrx-instruct)
- [X] [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [X] [Chinese LLaMA / Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) and [Chinese LLaMA-2 / Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)
- [X] [Vigogne (French)](https://github.com/bofenghuang/vigogne)
- [X] [BERT](https://github.com/ggml-org/llama.cpp/pull/5423)
- [X] [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- [X] [Baichuan 1 & 2](https://huggingface.co/models?search=baichuan-inc/Baichuan) + [derivations](https://huggingface.co/hiyouga/baichuan-7b-sft)
- [X] [Aquila 1 & 2](https://huggingface.co/models?search=BAAI/Aquila)
- [X] [Starcoder models](https://github.com/ggml-org/llama.cpp/pull/3187)
- [X] [Refact](https://huggingface.co/smallcloudai/Refact-1_6B-fim)
- [X] [MPT](https://github.com/ggml-org/llama.cpp/pull/3417)
- [X] [Bloom](https://github.com/ggml-org/llama.cpp/pull/3553)
- [x] [Yi models](https://huggingface.co/models?search=01-ai/Yi)
- [X] [StableLM models](https://huggingface.co/stabilityai)
- [x] [Deepseek models](https://huggingface.co/models?search=deepseek-ai/deepseek)
- [x] [Qwen models](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [PLaMo-13B](https://github.com/ggml-org/llama.cpp/pull/3557)
- [x] [Phi models](https://huggingface.co/models?search=microsoft/phi)
- [x] [PhiMoE](https://github.com/ggml-org/llama.cpp/pull/11003)
- [x] [GPT-2](https://huggingface.co/gpt2)
- [x] [Orion 14B](https://github.com/ggml-org/llama.cpp/pull/5118)
- [x] [InternLM2](https://huggingface.co/models?search=internlm2)
- [x] [CodeShell](https://github.com/WisdomShell/codeshell)
- [x] [Gemma](https://ai.google.dev/gemma)
- [x] [Mamba](https://github.com/state-spaces/mamba)
- [x] [Grok-1](https://huggingface.co/keyfan/grok-1-hf)
- [x] [Xverse](https://huggingface.co/models?search=xverse)
- [x] [Command-R models](https://huggingface.co/models?search=CohereForAI/c4ai-command-r)
- [x] [SEA-LION](https://huggingface.co/models?search=sea-lion)
- [x] [GritLM-7B](https://huggingface.co/GritLM/GritLM-7B) + [GritLM-8x7B](https://huggingface.co/GritLM/GritLM-8x7B)
- [x] [OLMo](https://allenai.org/olmo)
- [x] [OLMo 2](https://allenai.org/olmo)
- [x] [OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
- [x] [Granite models](https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330)
- [x] [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) + [Pythia](https://github.com/EleutherAI/pythia)
- [x] [Snowflake-Arctic MoE](https://huggingface.co/collections/Snowflake/arctic-66290090abe542894a5ac520)
- [x] [Smaug](https://huggingface.co/models?search=Smaug)
- [x] [Poro 34B](https://huggingface.co/LumiOpen/Poro-34B)
- [x] [Bitnet b1.58 models](https://huggingface.co/1bitLLM)
- [x] [Flan T5](https://huggingface.co/models?search=flan-t5)
- [x] [Open Elm models](https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca)
- [x] [ChatGLM3-6b](https://huggingface.co/THUDM/chatglm3-6b) + [ChatGLM4-9b](https://huggingface.co/THUDM/glm-4-9b) + [GLMEdge-1.5b](https://huggingface.co/THUDM/glm-edge-1.5b-chat) + [GLMEdge-4b](https://huggingface.co/THUDM/glm-edge-4b-chat)
- [x] [GLM-4-0414](https://huggingface.co/collections/THUDM/glm-4-0414-67f3cbcb34dd9d252707cb2e)
- [x] [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)
- [x] [EXAONE-3.0-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)
- [x] [FalconMamba Models](https://huggingface.co/collections/tiiuae/falconmamba-7b-66b9a580324dd1598b0f6d4a)
- [x] [Jais](https://huggingface.co/inceptionai/jais-13b-chat)
- [x] [Bielik-11B-v2.3](https://huggingface.co/collections/speakleash/bielik-11b-v23-66ee813238d9b526a072408a)
- [x] [RWKV-6](https://github.com/BlinkDL/RWKV-LM)
- [x] [QRWKV-6](https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1)
- [x] [GigaChat-20B-A3B](https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct)
- [X] [Trillion-7B-preview](https://huggingface.co/trillionlabs/Trillion-7B-preview)
- [x] [Ling models](https://huggingface.co/collections/inclusionAI/ling-67c51c85b34a7ea0aba94c32)
- [x] [LFM2 models](https://huggingface.co/collections/LiquidAI/lfm2-686d721927015b2ad73eaa38)

#### Multimodal

- [x] [LLaVA 1.5 models](https://huggingface.co/collections/liuhaotian/llava-15-653aac15d994e992e2677a7e), [LLaVA 1.6 models](https://huggingface.co/collections/liuhaotian/llava-16-65b9e40155f60fd046a5ccf2)
- [x] [BakLLaVA](https://huggingface.co/models?search=SkunkworksAI/Bakllava)
- [x] [Obsidian](https://huggingface.co/NousResearch/Obsidian-3B-V0.5)
- [x] [ShareGPT4V](https://huggingface.co/models?search=Lin-Chen/ShareGPT4V)
- [x] [MobileVLM 1.7B/3B models](https://huggingface.co/models?search=mobileVLM)
- [x] [Yi-VL](https://huggingface.co/models?search=Yi-VL)
- [x] [Mini CPM](https://huggingface.co/models?search=MiniCPM)
- [x] [Moondream](https://huggingface.co/vikhyatk/moondream2)
- [x] [Bunny](https://github.com/BAAI-DCAI/Bunny)
- [x] [GLM-EDGE](https://huggingface.co/models?search=glm-edge)
- [x] [Qwen2-VL](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)

</details>

<details>
<summary>Bindings</summary>

- Python: [ddh0/easy-llama](https://github.com/ddh0/easy-llama)
- Python: [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Go: [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp)
- Node.js: [withcatai/node-llama-cpp](https://github.com/withcatai/node-llama-cpp)
- JS/TS (llama.cpp server client): [lgrammel/modelfusion](https://modelfusion.dev/integration/model-provider/llamacpp)
- JS/TS (Programmable Prompt Engine CLI): [offline-ai/cli](https://github.com/offline-ai/cli)
- JavaScript/Wasm (works in browser): [tangledgroup/llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm)
- Typescript/Wasm (nicer API, available on npm): [ngxson/wllama](https://github.com/ngxson/wllama)
- Ruby: [yoshoku/llama_cpp.rb](https://github.com/yoshoku/llama_cpp.rb)
- Rust (more features): [edgenai/llama_cpp-rs](https://github.com/edgenai/llama_cpp-rs)
- Rust (nicer API): [mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp)
- Rust (more direct bindings): [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)
- Rust (automated build from crates.io): [ShelbyJenkins/llm_client](https://github.com/ShelbyJenkins/llm_client)
- C#/.NET: [SciSharp/LLamaSharp](https://github.com/SciSharp/LLamaSharp)
- C#/VB.NET (more features - community license): [LM-Kit.NET](https://docs.lm-kit.com/lm-kit-net/index.html)
- Scala 3: [donderom/llm4s](https://github.com/donderom/llm4s)
- Clojure: [phronmophobic/llama.clj](https://github.com/phronmophobic/llama.clj)
- React Native: [mybigday/llama.rn](https://github.com/mybigday/llama.rn)
- Java: [kherud/java-llama.cpp](https://github.com/kherud/java-llama.cpp)
- Zig: [deins/llama.cpp.zig](https://github.com/Deins/llama.cpp.zig)
- Flutter/Dart: [netdur/llama_cpp_dart](https://github.com/netdur/llama_cpp_dart)
- Flutter: [xuegao-tzx/Fllama](https://github.com/xuegao-tzx/Fllama)
- PHP (API bindings and features built on top of llama.cpp): [distantmagic/resonance](https://github.com/distantmagic/resonance) [(more info)](https://github.com/ggml-org/llama.cpp/pull/6326)
- Guile Scheme: [guile_llama_cpp](https://savannah.nongnu.org/projects/guile-llama-cpp)
- Swift [srgtuszy/llama-cpp-swift](https://github.com/srgtuszy/llama-cpp-swift)
- Swift [ShenghaiWang/SwiftLlama](https://github.com/ShenghaiWang/SwiftLlama)
- Delphi [Embarcadero/llama-cpp-delphi](https://github.com/Embarcadero/llama-cpp-delphi)

</details>

<details>
<summary>UIs</summary>

*(to have a project listed here, it should clearly state that it depends on `llama.cpp`)*

- [AI Sublime Text plugin](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) (MIT)
- [cztomsik/ava](https://github.com/cztomsik/ava) (MIT)
- [Dot](https://github.com/alexpinel/Dot) (GPL)
- [eva](https://github.com/ylsdamxssjxxdd/eva) (MIT)
- [iohub/collama](https://github.com/iohub/coLLaMA) (Apache-2.0)
- [janhq/jan](https://github.com/janhq/jan) (AGPL)
- [johnbean393/Sidekick](https://github.com/johnbean393/Sidekick) (MIT)
- [KanTV](https://github.com/zhouwg/kantv?tab=readme-ov-file) (Apache-2.0)
- [KodiBot](https://github.com/firatkiral/kodibot) (GPL)
- [llama.vim](https://github.com/ggml-org/llama.vim) (MIT)
- [LARS](https://github.com/abgulati/LARS) (AGPL)
- [Llama Assistant](https://github.com/vietanhdev/llama-assistant) (GPL)
- [LLMFarm](https://github.com/guinmoon/LLMFarm?tab=readme-ov-file) (MIT)
- [LLMUnity](https://github.com/undreamai/LLMUnity) (MIT)
- [LMStudio](https://lmstudio.ai/) (proprietary)
- [LocalAI](https://github.com/mudler/LocalAI) (MIT)
- [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp) (AGPL)
- [MindMac](https://mindmac.app) (proprietary)
- [MindWorkAI/AI-Studio](https://github.com/MindWorkAI/AI-Studio) (FSL-1.1-MIT)
- [Mobile-Artificial-Intelligence/maid](https://github.com/Mobile-Artificial-Intelligence/maid) (MIT)
- [Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile) (Apache-2.0)
- [nat/openplayground](https://github.com/nat/openplayground) (MIT)
- [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all) (MIT)
- [ollama/ollama](https://github.com/ollama/ollama) (MIT)
- [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) (AGPL)
- [PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai) (MIT)
- [psugihara/FreeChat](https://github.com/psugihara/FreeChat) (MIT)
- [ptsochantaris/emeltal](https://github.com/ptsochantaris/emeltal) (MIT)
- [pythops/tenere](https://github.com/pythops/tenere) (AGPL)
- [ramalama](https://github.com/containers/ramalama) (MIT)
- [semperai/amica](https://github.com/semperai/amica) (MIT)
- [withcatai/catai](https://github.com/withcatai/catai) (MIT)
- [Autopen](https://github.com/blackhole89/autopen) (GPL)

</details>

<details>
<summary>Tools</summary>

- [akx/ggify](https://github.com/akx/ggify) – download PyTorch models from HuggingFace Hub and convert them to GGML
- [akx/ollama-dl](https://github.com/akx/ollama-dl) – download models from the Ollama library to be used directly with llama.cpp
- [crashr/gppm](https://github.com/crashr/gppm) – launch llama.cpp instances utilizing NVIDIA Tesla P40 or P100 GPUs with reduced idle power consumption
- [gpustack/gguf-parser](https://github.com/gpustack/gguf-parser-go/tree/main/cmd/gguf-parser) - review/check the GGUF file and estimate the memory usage
- [Styled Lines](https://marketplace.unity.com/packages/tools/generative-ai/styled-lines-llama-cpp-model-292902) (proprietary licensed, async wrapper of inference part for game development in Unity3d with pre-built Mobile and Web platform wrappers and a model example)

</details>

<details>
<summary>Infrastructure</summary>

- [Paddler](https://github.com/distantmagic/paddler) - Stateful load balancer custom-tailored for llama.cpp
- [GPUStack](https://github.com/gpustack/gpustack) - Manage GPU clusters for running LLMs
- [llama_cpp_canister](https://github.com/onicai/llama_cpp_canister) - llama.cpp as a smart contract on the Internet Computer, using WebAssembly
- [llama-swap](https://github.com/mostlygeek/llama-swap) - transparent proxy that adds automatic model switching with llama-server
- [Kalavai](https://github.com/kalavai-net/kalavai-client) - Crowdsource end to end LLM deployment at any scale
- [llmaz](https://github.com/InftyAI/llmaz) - ☸️ Easy, advanced inference platform for large language models on Kubernetes.
</details>

<details>
<summary>Games</summary>

- [Lucy's Labyrinth](https://github.com/MorganRO8/Lucys_Labyrinth) - A simple maze game where agents controlled by an AI model will try to trick you.

</details>


## Supported backends

| Backend | Target devices |
| --- | --- |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [SYCL](docs/backend/SYCL.md) | Intel and Nvidia GPU |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [WebGPU [In Progress]](docs/build.md#webgpu) | All |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |

## Obtaining and quantizing models

The [Hugging Face](https://huggingface.co) platform hosts a [number of LLMs](https://huggingface.co/models?library=gguf&sort=trending) compatible with `llama.cpp`:

- [Trending](https://huggingface.co/models?library=gguf&sort=trending)
- [LLaMA](https://huggingface.co/models?sort=trending&search=llama+gguf)

You can either manually download the GGUF file or directly use any `llama.cpp`-compatible models from [Hugging Face](https://huggingface.co/) or other model hosting sites, such as [ModelScope](https://modelscope.cn/), by using this CLI argument: `-hf <user>/<model>[:quant]`. For example:

```sh
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```

By default, the CLI would download from Hugging Face, you can switch to other options with the environment variable `MODEL_ENDPOINT`. For example, you may opt to downloading model checkpoints from ModelScope or other model sharing communities by setting the environment variable, e.g. `MODEL_ENDPOINT=https://www.modelscope.cn/`.

After downloading a model, use the CLI tools to run it locally - see below.

`llama.cpp` requires the model to be stored in the [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) file format. Models in other data formats can be converted to GGUF using the `convert_*.py` Python scripts in this repo.

The Hugging Face platform provides a variety of online tools for converting, quantizing and hosting models with `llama.cpp`:

- Use the [GGUF-my-repo space](https://huggingface.co/spaces/ggml-org/gguf-my-repo) to convert to GGUF format and quantize model weights to smaller sizes
- Use the [GGUF-my-LoRA space](https://huggingface.co/spaces/ggml-org/gguf-my-lora) to convert LoRA adapters to GGUF format (more info: https://github.com/ggml-org/llama.cpp/discussions/10123)
- Use the [GGUF-editor space](https://huggingface.co/spaces/CISCai/gguf-editor) to edit GGUF meta data in the browser (more info: https://github.com/ggml-org/llama.cpp/discussions/9268)
- Use the [Inference Endpoints](https://ui.endpoints.huggingface.co/) to directly host `llama.cpp` in the cloud (more info: https://github.com/ggml-org/llama.cpp/discussions/9669)

To learn more about model quantization, [read this documentation](tools/quantize/README.md)

## [`llama-cli`](tools/main)

#### A CLI tool for accessing and experimenting with most of `llama.cpp`'s functionality.

- <details open>
    <summary>Run in conversation mode</summary>

    Models with a built-in chat template will automatically activate conversation mode. If this doesn't occur, you can manually enable it by adding `-cnv` and specifying a suitable chat template with `--chat-template NAME`

    ```bash
    llama-cli -m model.gguf

    # > hi, who are you?
    # Hi there! I'm your helpful assistant! I'm an AI-powered chatbot designed to assist and provide information to users like you. I'm here to help answer your questions, provide guidance, and offer support on a wide range of topics. I'm a friendly and knowledgeable AI, and I'm always happy to help with anything you need. What's on your mind, and how can I assist you today?
    #
    # > what is 1+1?
    # Easy peasy! The answer to 1+1 is... 2!
    ```

    </details>

- <details>
    <summary>Run in conversation mode with custom chat template</summary>

    ```bash
    # use the "chatml" template (use -h to see the list of supported templates)
    llama-cli -m model.gguf -cnv --chat-template chatml

    # use a custom template
    llama-cli -m model.gguf -cnv --in-prefix 'User: ' --reverse-prompt 'User:'
    ```

    </details>

- <details>
    <summary>Run simple text completion</summary>

    To disable conversation mode explicitly, use `-no-cnv`

    ```bash
    llama-cli -m model.gguf -p "I believe the meaning of life is" -n 128 -no-cnv

    # I believe the meaning of life is to find your own truth and to live in accordance with it. For me, this means being true to myself and following my passions, even if they don't align with societal expectations. I think that's what I love about yoga – it's not just a physical practice, but a spiritual one too. It's about connecting with yourself, listening to your inner voice, and honoring your own unique journey.
    ```

    </details>

- <details>
    <summary>Constrain the output with a custom grammar</summary>

    ```bash
    llama-cli -m model.gguf -n 256 --grammar-file grammars/json.gbnf -p 'Request: schedule a call at 8pm; Command:'

    # {"appointmentTime": "8pm", "appointmentDetails": "schedule a a call"}
    ```

    The [grammars/](grammars/) folder contains a handful of sample grammars. To write your own, check out the [GBNF Guide](grammars/README.md).

    For authoring more complex JSON grammars, check out https://grammar.intrinsiclabs.ai/

    </details>


## [`llama-server`](tools/server)

#### A lightweight, [OpenAI API](https://github.com/openai/openai-openapi) compatible, HTTP server for serving LLMs.

- <details open>
    <summary>Start a local HTTP server with default configuration on port 8080</summary>

    ```bash
    llama-server -m model.gguf --port 8080

    # Basic web UI can be accessed via browser: http://localhost:8080
    # Chat completion endpoint: http://localhost:8080/v1/chat/completions
    ```

    </details>

- <details>
    <summary>Support multiple-users and parallel decoding</summary>

    ```bash
    # up to 4 concurrent requests, each with 4096 max context
    llama-server -m model.gguf -c 16384 -np 4
    ```

    </details>

- <details>
    <summary>Enable speculative decoding</summary>

    ```bash
    # the draft.gguf model should be a small variant of the target model.gguf
    llama-server -m model.gguf -md draft.gguf
    ```

    </details>

- <details>
    <summary>Serve an embedding model</summary>

    ```bash
    # use the /embedding endpoint
    llama-server -m model.gguf --embedding --pooling cls -ub 8192
    ```

    </details>

- <details>
    <summary>Serve a reranking model</summary>

    ```bash
    # use the /reranking endpoint
    llama-server -m model.gguf --reranking
    ```

    </details>

- <details>
    <summary>Constrain all outputs with a grammar</summary>

    ```bash
    # custom grammar
    llama-server -m model.gguf --grammar-file grammar.gbnf

    # JSON
    llama-server -m model.gguf --grammar-file grammars/json.gbnf
    ```

    </details>


## [`llama-perplexity`](tools/perplexity)

#### A tool for measuring the [perplexity](tools/perplexity/README.md) [^1] (and other quality metrics) of a model over a given text.

- <details open>
    <summary>Measure the perplexity over a text file</summary>

    ```bash
    llama-perplexity -m model.gguf -f file.txt

    # [1]15.2701,[2]5.4007,[3]5.3073,[4]6.2965,[5]5.8940,[6]5.6096,[7]5.7942,[8]4.9297, ...
    # Final estimate: PPL = 5.4007 +/- 0.67339
    ```

    </details>

- <details>
    <summary>Measure KL divergence</summary>

    ```bash
    # TODO
    ```

    </details>

[^1]: [https://huggingface.co/docs/transformers/perplexity](https://huggingface.co/docs/transformers/perplexity)

## [`llama-bench`](tools/llama-bench)

#### Benchmark the performance of the inference for various parameters.

- <details open>
    <summary>Run default benchmark</summary>

    ```bash
    llama-bench -m model.gguf

    # Output:
    # | model               |       size |     params | backend    | threads |          test |                  t/s |
    # | ------------------- | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         pp512 |      5765.41 ± 20.55 |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         tg128 |        197.71 ± 0.81 |
    #
    # build: 3e0ba0e60 (4229)
    ```

    </details>

## [`llama-run`](tools/run)

#### A comprehensive example for running `llama.cpp` models. Useful for inferencing. Used with RamaLama [^3].

- <details>
    <summary>Run a model with a specific prompt (by default it's pulled from Ollama registry)</summary>

    ```bash
    llama-run granite-code
    ```

    </details>

[^3]: [RamaLama](https://github.com/containers/ramalama)

## [`llama-simple`](examples/simple)

#### A minimal example for implementing apps with `llama.cpp`. Useful for developers.

- <details>
    <summary>Basic text completion</summary>

    ```bash
    llama-simple -m model.gguf

    # Hello my name is Kaitlyn and I am a 16 year old girl. I am a junior in high school and I am currently taking a class called "The Art of
    ```

    </details>


## Contributing

- Contributors can open PRs
- Collaborators can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Collaborators will be invited based on contributions
- Any help with managing issues, PRs and projects is very appreciated!
- See [good first issues](https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for tasks suitable for first contributions
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information
- Make sure to read this: [Inference at the edge](https://github.com/ggml-org/llama.cpp/discussions/205)
- A bit of backstory for those who are interested: [Changelog podcast](https://changelog.com/podcast/532)

## Other documentation

- [main (cli)](tools/main/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development documentation

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

#### Seminal papers and background on the models

If your issue is with model generation quality, then please at least scan the following links and papers to understand the limitations of LLaMA models. This is especially important when choosing an appropriate model size and appreciating both the significant and subtle differences between LLaMA models and ChatGPT:
- LLaMA:
    - [Introducing LLaMA: A foundational, 65-billion-parameter large language model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
    - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- GPT-3
    - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- GPT-3.5 / InstructGPT / ChatGPT:
    - [Aligning language models to follow instructions](https://openai.com/research/instruction-following)
    - [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

## XCFramework
The XCFramework is a precompiled version of the library for iOS, visionOS, tvOS,
and macOS. It can be used in Swift projects without the need to compile the
library from source. For example:
```swift
// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MyLlamaPackage",
    targets: [
        .executableTarget(
            name: "MyLlamaPackage",
            dependencies: [
                "LlamaFramework"
            ]),
        .binaryTarget(
            name: "LlamaFramework",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b5046/llama-b5046-xcframework.zip",
            checksum: "c19be78b5f00d8d29a25da41042cb7afa094cbf6280a225abe614b03b20029ab"
        )
    ]
)
```
The above example is using an intermediate build `b5046` of the library. This can be modified
to use a different version by changing the URL and checksum.

## Completions
Command-line completion is available for some environments.

#### Bash Completion
```bash
$ build/bin/llama-cli --completion-bash > ~/.llama-completion.bash
$ source ~/.llama-completion.bash
```
Optionally this can be added to your `.bashrc` or `.bash_profile` to load it
automatically. For example:
```console
$ echo "source ~/.llama-completion.bash" >> ~/.bashrc
```

## Pipeline Sharding (PipeShard)

Pipeline sharding enables running large models that exceed VRAM by scheduling layers across GPU and CPU with concurrent PCIe transfers.
VLMOpt provides complementary VRAM-reduction optimizations for the vision encoder so that high-resolution VLM inference.
**MLSys26 Paper:** [EFFICIENT, VRAM-CONSTRAINED XLM INFERENCE ON CLIENTS](link arriving soon)


### Requirements

1. **Hardware**: An x86_64 machine with a discrete GPU (preferably an NVIDIA RTX series GPU).

2. **NVIDIA Driver & CUDA Toolkit** (for NVIDIA GPUs):
   - Install the latest **Game Ready Driver** (GRD) from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx):
     1. Select your GPU product, OS, and "Game Ready Driver" download type.
     2. Download and run the installer; a reboot may be required.
     3. Verify with `nvidia-smi` in a terminal.
   - Install **CUDA Toolkit 12.8+** from [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads):
     1. Select your OS, architecture, and installer type (recommended: network installer).
     2. Run the installer and follow the on-screen prompts (the default options are fine).
     3. Ensure `nvcc` is on your PATH. Verify with `nvcc --version`.

3. *(Optional, Windows)* **Visual Studio 2022+** — needed for the MSVC compiler and CMake generator on Windows. Install from [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/) and select the **"Desktop development with C++"** workload during setup.

### Step 1: Build

You can **build from source**, **download pre-built release binaries**, or **use the Docker image**.

#### Option A: Build from Source

```bash
git clone https://github.com/deepshnv/pipeshard-mlsys26-ae.git
cd pipeshard-mlsys26-ae
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j16
```

On Windows, the CMake configure step also generates a Visual Studio solution (`build/llama.cpp.sln`) that can be opened and built directly in Visual Studio. Binaries are placed in `build/bin/Release/` (Windows) or `build/bin/` (Linux/macOS).

> All development and testing for our paper was done on Windows; we recommend Windows for the smoothest reproduction experience. That said, llama.cpp and our algorithmic changes are platform-independent and should build and run on Linux/macOS as well.

#### Option B: Pre-built Release Binaries (Windows x86_64)

If you do not wish to build from source, pre-built binaries are available on the [Releases](https://github.com/deepshnv/pipeshard-mlsys26-ae/releases) page.

**Build environment for this Release was:** Windows x86_64 (Intel), MSVC, CUDA Toolkit 12.9, NVIDIA RTX 5090.

```powershell
# Download and extract the release archive
Invoke-WebRequest -Uri "https://github.com/deepshnv/pipeshard-mlsys26-ae/releases/download/v1.0.0-mlsys26/pipeshard-mlsys26-ae-win-x64-cuda12.9.zip" -OutFile release.zip
Expand-Archive release.zip -DestinationPath build\bin\Release
```

**Note:** The pre-built binaries are linked against CUDA 12.9. You still need a compatible NVIDIA driver installed (see [Requirements](#requirements) above). If your CUDA version differs significantly, build from source using Option A.

#### Option C: Docker

A pre-built Docker image with all dependencies and compiled binaries is available. This is the easiest way to get started -- no local build or dependency setup required.

<details>
<summary><strong>One-time setup: Docker + NVIDIA GPU support</strong></summary>

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (enable WSL 2 backend on Windows, restart after install).
2. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) inside WSL2:

```bash
wsl
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
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

# Mount your model weights directory; the container runs all 5 repro scripts (Table 4, 5, 8, 9, Figure 2)
docker run --gpus all -v /path/to/your/gguf_models:/workspace/gguf_models ghcr.io/deepshnv/pipeshard-mlsys26-ae:v1.0.0
```

**Or run interactively** (to download models, run scripts individually, inspect results):

```bash
docker run --gpus all -it -v /path/to/your/gguf_models:/workspace/gguf_models ghcr.io/deepshnv/pipeshard-mlsys26-ae:v1.0.0 bash

# Inside the container:
./download_models.sh                              # download models (if not mounted)
./paper_results/repro_table4.sh                   # similarly run other scripts like repro_table5.sh, repro_table8.sh, repro_table9.sh, repro_figure2.sh
```

> The Dockerfile is in the repository root for transparency. To rebuild locally: `docker build -t pipeshard-mlsys26-ae .`

### Step 2: Set Environment Variables

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

### Step 3: Run Profilers

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

### Step 5: Run Multimodal / VLM Inference

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

### PipeShard CLI Flags

| Flag | Description |
|------|-------------|
| `-pipe-shard` | Enable pipeline sharding |
| `-mva N` | Max VRAM allocation budget in MB |
| `-psa N` | Pinned system memory allocation in GB (0 = auto) |
| `-fes N` | Force execution strategy (`-1` = auto, `0` = FULL_GPU, `1` = FULL_GPU_NO_OUTPUT, `2` = STATIC_ATTN_PRIO, `3` = SHARD_ATTN_PRIO, `4` = FULL_SHARD) |
| `--cpu-profile PATH` | Path to CPU benchmark profile (default: `concurrent_results.txt`) |
| `--gpu-profile PATH` | Path to GPU benchmark profile (default: `gpu_results.txt`) |

### VLMOpt CLI Flags

These flags control vision encoder (CLIP) VRAM optimizations in `llama-mtmd-cli`. They enable high-resolution VLM inference within tight VRAM budgets by offloading and tiling the vision encoder independently of the LLM pipeline.

| Flag | Description |
|------|-------------|
| `-vto-offload-cpu` | Offload all CLIP vision encoder weights to CPU. Weights are streamed to GPU on demand during encoding, freeing VRAM for the LLM. Recommended when VRAM is constrained. |
| `-vto-tiled-attention` | Reduce peak VRAM of the O(N^2) QK and QKV attention tensors inside the vision encoder by processing the Q matrix in smaller tiles. Prevents large transient allocations that can exceed VRAM during high-resolution image encoding. |
| `-clip-tiled-mb N` | VRAM budget (in MiB) for CLIP tiled-Q attention. Controls the tile size: larger values use fewer, bigger tiles (faster); smaller values reduce peak VRAM further but increase encoding passes. Typical range: 2000-12000 MiB. |
| `-cis N` / `-clip-img-size N` | Override the maximum image dimension fed to the vision encoder. Models like Qwen2.5-VL benefit from native-resolution input; set this to the image's longer side (e.g., `3840` for 4K). Default: model-defined (typically 336 or 384). |

> **Note:** When using large image resolutions (e.g., `-cis 3840`), the standard (non-FA) attention path builds a full N x N attention mask that can exceed the 2 GB internal tensor size limit, causing an assertion failure during the CUDA copy operation. To avoid this, set `MTMD_CLIP_FLASH_ATTN=2` (tiled Flash Attention), which eliminates the explicit mask tensor entirely and processes attention within each tiled-Q chunk using FA kernels instead.

## Reproducing MLSys'26 Paper Results

> This section provides step-by-step instructions for reproducing the main results presented in the paper.
>
> **Note on reproducibility:** Absolute performance numbers will vary across hardware; the relative speedups and directional trends should remain consistent.

---

### Step 1: Download Required Models

All models must be placed in the `gguf_models/` directory before running any experiments.

| Model | URL | Download Instructions |
|-------|-----|----------------------|
| `mistral-nemo-minitron-4b-128k-instruct-f16` | [NVIDIA ACE (4B)](https://developer.nvidia.com/downloads/assets/ace/model_zip/mistral-nemo-minitron-4b-128k-instruct_v1.0.0.7z) | Click the URL, extract the `.7z` archive into `gguf_models/` |
| `mistral-nemo-minitron-8b-128k-instruct-f16` | [NVIDIA ACE (8B)](https://developer.nvidia.com/downloads/assets/ace/model_zip/mistral-nemo-minitron-8b-128k-instruct_v1.0.0.7z) | Click the URL, extract the `.7z` archive into `gguf_models/` |
| `Qwen3-30B-A3B-Instruct-2507-q4` | [Hugging Face (Q4_0)](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/main/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf?download=true) | Click the URL to download the single GGUF file, place it in gguf_models/ |
| `Qwen3-235B-A22B-Instruct-2507-q2_k` | [Hugging Face (Q2_K)](https://huggingface.co/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF/tree/main/Q2_K) | `hf download unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF --include "Q2_K/*" --local-dir gguf_models/Qwen3-235B-A22B` |
| `Cosmos-Reason1` | [Hugging Face (7B-GGUF)](https://huggingface.co/deepshekhar03/Cosmos-Reason1-7B-GGUF/tree/main) | `hf download deepshekhar03/Cosmos-Reason1-7B-GGUF --local-dir gguf_models/cosmos_reason1` |

#### Setting Up the Hugging Face CLI

The `hf download` commands above require Python 3.12+ and the HF Hub CLI. Create a venv and install:

**Windows (PowerShell):**
```powershell
winget install Python.Python.3.12          # skip if already installed
python -m venv hf_venv && .\hf_venv\Scripts\Activate.ps1
```

After installing the CLI (see below), you can download all models at once:
```powershell
.\download_models.ps1
```
> The script auto-creates `gguf_models/` subdirectories, downloads HF-hosted models via `hf download`, and prints instructions for the NVIDIA ACE models that require manual browser download.

**Linux / macOS:**
```bash
# Ubuntu/Debian: sudo apt install -y python3.12 python3.12-venv
# macOS:         brew install python@3.12
python3.12 -m venv hf_venv && source hf_venv/bin/activate
```

After installing the CLI (see below), you can download all models at once:
```bash
chmod +x download_models.sh
./download_models.sh
```

**Then (all platforms):**
```bash
pip install huggingface_hub[cli]
huggingface-cli login   # requires a token from https://huggingface.co/settings/tokens
```

---

### Step 2: Reproduce Table 4 — TPS and TTFT under Pipelined Sharding

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

### Step 3: Reproduce Table 5 — TPS and TTFT at Peak VRAM Capacity

Table 5 measures **TPS** and **TTFT** for the same four LLMs but at a single VRAM budget: the GPU's **peak usable capacity**. The paper reports results on two machines (cli2 at 16G and cli1 at 12G); the reproduction script lets you specify your GPU's peak VRAM so results are comparable on any hardware.

**Windows (PowerShell):**
```powershell
cd pipeshard-mlsys26-ae
.\paper_results\repro_table5.ps1 -PeakVramMB 30720
```

**Linux / macOS:**
```bash
cd pipeshard-mlsys26-ae
chmod +x paper_results/repro_table5.sh
./paper_results/repro_table5.sh --peak-vram-mb 12288
```

**Options:**

| Flag | Description |
|------|-------------|
| `-PeakVramMB` / `--peak-vram-mb` | Peak VRAM budget in MB (default: `30720` = 30G) |
| `-BinDir` / `--bin-dir` | Path to directory containing `llama-cli` and profiler executables |
| `-ModelsDir` / `--models-dir` | Path to `gguf_models/` directory |
| `-SkipProfiling` / `--skip-profiling` | Skip profiler runs and reuse existing profiles |

The output CSV (`paper_results/table5_results.csv`) contains columns: `Model, CtxSize, PeakVramMB, PeakVram, TPS, TTFT(msec)`. Compare against the reference in `paper_results/table5.png`.

---

### Step 4: Reproduce Table 8 — E2EL Speedups for VLM with PipeShard + VLMOpt

Table 8 measures **end-to-end latency (E2EL) speedups** for the Cosmos-Reason1 VLM at 4 image resolutions (480p, 720p, 1080p, 1440p) using pipeline sharding combined with VLMOpt. For each resolution, a **baseline** run (no sharding, no VLMOpt) is compared against **VLMOpt** runs at 3 VRAM budgets (4G, 8G, 14.5G).

The script:
- Runs `llama-mtmd-cli` with the Cosmos-Reason1 model and a test image (`paper_results/dummy_image/165_4k.jpg`)
- Varies the input resolution via `-cis` (640, 1280, 1920, 2560)
- For each resolution: runs baseline first, then VLMOpt at each VRAM budget
- Parses **image encode time**, **image decode time**, **TTFT**, **TPS** from the output
- Computes **E2EL** = encode + decode + TTFT + (100 / TPS) for each run
- Monitors **peak VRAM usage** via `nvidia-smi` in the background
- Computes the **speedup**

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

The output CSV (`paper_results/table8_results.csv`) contains columns: `Resolution, RunType, VramBudget, Encode(msec), Decode(msec), TTFT(msec), TPS, E2EL(msec), PeakVramMB, Speedup`. Compare the speedup values against the reference in `paper_results/table8.png`.

---

### Step 5: Reproduce Figure 2 -- TTFT/TPS/E2EL Speedups from Pipelined Sharding

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

### Step 6: Reproduce Table 9 — TPS vs Multi-Request Batch Size across VRAM Budgets

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

## Dependencies

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [minja](https://github.com/google/minja) - Minimal Jinja parser in C++, used by various tools/examples - MIT License
- [linenoise.cpp](./tools/run/linenoise.cpp/linenoise.cpp) - C++ library that provides readline-like line editing capabilities, used by `llama-run` - BSD 2-Clause License
- [curl](https://curl.se/) - Client-side URL transfer library, used by various tools/examples - [CURL License](https://curl.se/docs/copyright.html)
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
