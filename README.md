# LLM info collection
The repo to store useful info about LLM

- [LLMs](#llms)
- [Train Finetune LLM](#train-finetune-llm)
- [Inference for LLM](#inference-for-llm)
- [Wrapper for LLM](#wrapper-for-llm)
- [Web UI Inference](#web-ui-inference)
---
## LLMs
* [ruGPTs](https://github.com/ai-forever/ru-gpts) - This repository contains bunch of autoregressive transformer language models trained on a huge dataset of russian language. Russian GPT-3 models (ruGPT3XL, ruGPT3Large, ruGPT3Medium, ruGPT3Small) trained with 2048 sequence length with sparse and dense attention blocks. We also provide Russian GPT-2 large model (ruGPT2Large) trained with 1024 sequence length.
      
## Train Finetune LLM
* [xTuring](https://github.com/stochasticai/xTuring) - xTuring provides fast, efficient and simple fine-tuning of LLMs, such as LLaMA, GPT-J, Galactica, and more. By providing an easy-to-use interface for fine-tuning LLMs to your own data and application, xTuring makes it simple to build, customize and control LLMs. The entire process can be done inside your computer or in your private cloud, ensuring data privacy and security.
* [GPT4All](https://github.com/nomic-ai/gpt4all) - is an ecosystem to train and deploy powerful and customized large language models that run locally on consumer grade CPUs. A GPT4All model is a 3GB - 8GB file that you can download and plug into the GPT4All open-source ecosystem software. Nomic AI supports and maintains this software ecosystem to enforce quality and security alongside spearheading the effort to allow any person or enterprise to easily train and deploy their own on-edge large language models.
* [FastChat](https://github.com/lm-sys/FastChat) - FastChat is an open platform for training, serving, and evaluating large language model based chatbots. The core features include:
    - The weights, training code, and evaluation code for state-of-the-art models (e.g., Vicuna, FastChat-T5).
    - A distributed multi-model serving system with Web UI and OpenAI-compatible RESTful APIs.

## Inference for LLM
* [llama.cpp](https://github.com/ggerganov/llama.cpp) - The main goal of llama.cpp is to run the LLaMA model using 4-bit integer quantization on a MacBook. Features: 1) Plain C/C++ implementation without dependencies 2) Apple silicon first-class citizen - optimized via ARM NEON and Accelerate framework 3) AVX, AVX2 and AVX512 support for x86 architectures 4) Mixed F16 / F32 precision 5) 4-bit, 5-bit and 8-bit integer quantization support 6) Runs on the CPU 7) Supports OpenBLAS/Apple BLAS/ARM Performance Lib/ATLAS/BLIS/Intel MKL/NVHPC/ACML/SCSL/SGIMATH and more in BLAS
cuBLAS and CLBlast support. The original implementation of llama.cpp was hacked in an evening. Since then, the project has improved significantly thanks to many contributions. This project is for educational purposes and serves as the main playground for developing new features for the ggml library.

## Wrapper for LLM
* [langchain](https://github.com/hwchase17/langchain) This library aims to assist in the development of LLM based applications. Common examples of these applications include:
    * Question Answering over specific documents
    * Chatbots
    * Agents


## Web UI Inference
* [oobabooga](https://github.com/oobabooga/text-generation-webui) - A gradio web UI for running Large Language Models like LLaMA, llama.cpp, GPT-J, Pythia, OPT, and GALACTICA. Its goal is to become the AUTOMATIC1111/stable-diffusion-webui of text generation.
