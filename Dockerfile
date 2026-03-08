FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    bc \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir huggingface_hub[cli] pandas matplotlib

WORKDIR /workspace

RUN git clone https://github.com/deepshnv/pipeshard-mlsys26-ae.git .

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

ENV LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}"

RUN cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF \
    -DCMAKE_CXX_FLAGS="-fpermissive" -DCMAKE_C_FLAGS="-fpermissive" \
    -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs" \
    && cmake --build build --config Release -j$(nproc) \
    && rm /usr/local/cuda/lib64/stubs/libcuda.so.1

ENV PATH="/workspace/build/bin:${PATH}"

RUN chmod +x download_models.sh \
    paper_results/repro_table4.sh \
    paper_results/repro_table8.sh \
    paper_results/repro_table9.sh \
    paper_results/repro_figure2.sh \
    paper_results/repro_figure7.sh

VOLUME ["/workspace/gguf_models"]

CMD ["bash", "-c", "\
    if [ -z \"$(find gguf_models -name '*.gguf' 2>/dev/null)\" ]; then \
        echo '=== No models found in gguf_models/ -- downloading ===' && \
        ./download_models.sh; \
    else \
        echo '=== Models found in gguf_models/ -- skipping download ==='; \
    fi && \
    echo '' && \
    echo '=== Running all reproduction scripts ===' && \
    echo '' && \
    echo '--- Step 1/5: Table 4 ---' && \
    ./paper_results/repro_table4.sh && \
    echo '' && \
    echo '--- Step 2/5: Table 8 ---' && \
    ./paper_results/repro_table8.sh --skip-profiling && \
    echo '' && \
    echo '--- Step 3/5: Table 9 ---' && \
    ./paper_results/repro_table9.sh --skip-profiling && \
    echo '' && \
    echo '--- Step 4/5: Figure 2 ---' && \
    ./paper_results/repro_figure2.sh --skip-profiling && \
    echo '' && \
    echo '--- Step 5/5: Figure 7 ---' && \
    ./paper_results/repro_figure7.sh --skip-profiling && \
    echo '' && \
    echo '=== All reproduction scripts complete ===' \
"]
