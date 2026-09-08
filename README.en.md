# 2× RTX 2080 Ti — Local LLM + Speech Stack

Self-contained inference services for **large language models** and **speech recognition with speaker diarization**, running entirely on a dual-GPU workstation. No cloud round-trips, no SaaS — your hardware, your weights, your data.

> **22 GB + 22 GB = 44 GB of VRAM** is the whole trick: it lets a 27B-parameter language model and a large acoustic model fit on consumer cards that are otherwise a decade old.

*Russian version: [README.md](README.md)*

## Contents

- [Hardware](#hardware)
- [Repository layout](#repository-layout)
- [Helper files](#helper-files)
- [What's built and how](#whats-built-and-how)
- [Quick start](#quick-start)
- [API reference](#api-reference)
- [Configuration reference](#configuration-reference)
- [Models & licensing](#models--licensing)
- [Version pins](#version-pins)
- [Known caveats](#known-caveats)
- [Credits](#credits)

## Hardware

| Component | Detail |
| --- | --- |
| GPUs | 2× NVIDIA GeForce **RTX 2080 Ti** |
| VRAM | 22 GB per card → **44 GB** total |
| Compute capability | **sm75** (Turing) |
| CUDA | Required (driver 450+) |

Both cards are exposed to the containers; tensor-parallel work is split across `device_ids [0, 1]`.

## Repository layout

```
.
├── README.md                                ← Russian version
├── README.en.md                             ← you are here
├── kvm.xml                                  ← local libvirt VM profile with GPU passthrough
├── proxy.py                                 ← local vLLM debug proxy
├── qwen/                                    ← LLM inference service
│   ├── .env.example                         ← HF_TOKEN and MODEL presets
│   ├── Dockerfile                           ← custom sm75 + CUDA 13 vLLM build
│   ├── docker-compose.yml                   ← container, GPUs, volumes
│   └── vllm-2080ti-sm75-cu130-wheels.tar.gz ← pinned wheels (vLLM + FlashQLA)
└── whisper/                                 ← speech → dialogue service
    ├── .env.example                         ← HF_TOKEN, WHISPER_MODEL, DIARIZE_MODEL
    ├── Dockerfile                           ← WhisperX + faster-whisper + pyannote
    ├── app.py                               ← FastAPI + transcription pipeline
    └── docker-compose.yml                   ← container, GPUs, volumes
```

Two sibling services, each a self-contained Docker build. They share the same two GPUs and the same Hugging Face cache.

## Helper files

- **`kvm.xml`** — a local libvirt profile for the `ubuntu26.04` VM: Ubuntu 26.04, 24 GB RAM, 6 statically pinned vCPUs, passthrough of both RTX 2080 Ti GPUs (8 PCI functions) and the model disk `MODEL.qcow2`. This is a host VM profile, not a Docker service.
- **`proxy.py`** — a minimal HTTP proxy on `0.0.0.0:8080` that forwards `GET`/`POST` to `http://ubuntu26:8000` (vLLM inside the VM) and prints request/response bodies on `400 Bad Request`. Useful for debugging OpenCode-like clients; requires `requests`.

## What's built and how

### 1. LLM inference — `qwen/`

The language side is a **vLLM** server exposing a 27B chat model behind the standard OpenAI-compatible interface. Everything is served locally; `GET /v1/models` reports it as `Qwen3.8`.

Key build decisions:

- **Custom vLLM wheels.** Upstream vLLM does not ship a precompiled build for the combination our hardware actually uses — Turing (sm75) paired with CUDA 13.0. Rather than rebuild on every deploy, we pin a prebuilt wheel archive (`vllm-2080ti-sm75-cu130-wheels.tar.gz`, 38.4 MiB / ≈40.3 MB) and install it together with a small `FlashQLA` companion, both with `--no-deps` so the pinned dependency set stays in control.
- **Model choice.** The current `MODEL` in `qwen/.env.example` is `shawnw3i/Qwen3.8-27B-AWQ-MTP`. The name encodes two properties: **AWQ** (weight-4-activation-16 quantization, ~14 GB — comfortable in 44 GB VRAM with a 128K context) and **MTP** (a multi-token prediction head used by vLLM for speculative decoding). The same `.env.example` also has commented presets: `QuantTrio/Qwen3.6-27B-AWQ` (fast), `twolven/Qwen3.8-27B-abliterated-AWQ-MTP` (abliterated/uncensored), and `lued/Qwen3.8-27B-INT8-W8A16-MTP` (hard INT8).
- **Serving configuration.** Tensor parallelism of 2, a **131 072**-token context window, **94 %** of VRAM reserved for the KV cache, **prefix caching enabled**, and speculative decoding that drafts **3 tokens** per step via the MTP head before verification. There is no explicit `--dtype`; vLLM chooses the precision automatically. Attention — FlashInfer, AWQ dequantization — Marlin, reasoning parser — `qwen3`.

The result is an HTTP endpoint that any OpenAI client can talk to without changing its config — point it at the local host and it behaves as if it were talking to the cloud.

### 2. Speech recognition — `whisper/`

The audio side is a **WhisperX**-based service that turns an uploaded recording into a **turn-by-turn transcript with each line attributed to a speaker**. It is exposed as a small FastAPI app with a single endpoint.

Pipeline, in order:

1. **Decode** the upload to a canonical format with FFmpeg.
2. **Transcribe** with **faster-whisper** (the CTranslate2 port of the Whisper family) in `int8` precision on **GPU 0**. The deployed model is the Russian-tuned turbo variant `dvislobokov/faster-whisper-large-v3-turbo-russian`; it is overridable via the `WHISPER_MODEL` environment variable (the code default is `Systran/faster-whisper-large-v3`).
3. **Force-align** the produced words onto the audio timeline to obtain word-level start/end timestamps.
4. **Diarize** the recording with the **pyannote** `speaker-diarization-3.1` pipeline on **GPU 1** (a gated Hugging Face model, so an `HF_TOKEN` is required at deploy time).
5. **Attribute** each word to a speaker by overlapping its timestamp against the diarization segments.
6. **Group** consecutive words sharing a speaker into utterance turns, and emit the dialogue.

The response shape is deliberately simple so downstream consumers do not need to model the internals:

```json
{ "dialogue": [ { "speaker": "SPEAKER_00", "start": 0.36, "end": 2.14, "text": "..." } ] }
```

Because the two heavy stages run on separate GPUs (ASR on one, diarization on the other), they can overlap on a busy machine — a deliberate split given by the two-card topology.

## Quick start

Prerequisites: a Linux host with the NVIDIA driver and `nvidia-container-toolkit`, Docker + Compose v2, and Hugging Face access (a token for the gated models).

### Language service

```sh
cd qwen

# Copy the variable template: HF_TOKEN + MODEL (current MODEL is shawnw3i/Qwen3.8-27B-AWQ-MTP)
cp .env.example .env
# edit .env and set HF_TOKEN

docker compose up -d --build
```

Wait for the model to finish loading onto both GPUs (the first boot downloads and quantizes; subsequent boots are much faster). Then probe it:

```sh
curl -s http://localhost:8000/v1/models
```

### Speech service

```sh
cd whisper

# Copy the variable template: HF_TOKEN + WHISPER_MODEL + DIARIZE_MODEL
cp .env.example .env
# edit .env and set HF_TOKEN (required for the gated pyannote model)

docker compose up -d --build
```

The speech service is published on **host port 8001**; inside the container it still listens on 8000.

Model artifacts (faster-whisper weights, pyannote, the WhisperX runtime) land in the shared Hugging Face cache volume, so the two services do not re-download each other's payloads.

## API reference

### Language — OpenAI-compatible

Base URL `http://<host>:8000/v1`, model `Qwen3.8`.

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8",
    "messages": [
      { "role": "system", "content": "You are a concise assistant." },
      { "role": "user",   "content": "Explain tensor parallelism in one sentence." }
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

Tool/function calling uses the `qwen3_xml` scheme — the model emits a tagged block describing the tool and its arguments, and the client parses and dispatches it.

### Speech — transcription

```bash
curl -s -X POST http://localhost:8001/v1/audio/transcriptions \
  -F 'file=@meeting.wav' \
  -F 'num_speakers=2'     # optional hint; omit to let it estimate
```

Returns the dialogue described in the service section above.

## Configuration reference

Most knobs are set in the respective `docker-compose.yml`; a few are runtime environment variables.

### Language (`qwen/`)

| Setting | Value | Meaning |
| --- | --- | --- |
| Model | `shawnw3i/Qwen3.8-27B-AWQ-MTP` (current `MODEL` in `qwen/.env.example`) | 27B, AWQ, MTP; `.env.example` also lists QuantTrio / twolven / lued presets |
| Served name | `Qwen3.8` | What clients address |
| Tensor parallel | 2 | Split across both GPUs |
| Precision | automatic (no explicit `--dtype`) | vLLM selects the model precision |
| Context | 131 072 | Max tokens per sequence |
| Concurrent seqs | 1 | One in-flight sequence |
| GPU memory util | 0.94 | VRAM reserved for the KV cache |
| Prefix caching | enabled | Reuse repeated prefixes |
| Speculative decoding | MTP, 3 drafts | Draft then verify |
| Reasoning parser | `qwen3` | Parse reasoning blocks |
| Attention | FlashInfer | Kernel backend |
| Quant kernel | Marlin | AWQ dequantization |

`qwen/docker-compose.yml` passes `MODEL_PATH=${MODEL}` into the container and mounts `~/models:/models`, so `MODEL` can be either a Hugging Face repository or a local path inside `~/models`.

### Speech (`whisper/`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `HF_TOKEN` | — | Hugging Face token (gated models) |
| `WHISPER_MODEL` | code: `Systran/faster-whisper-large-v3`; `.env.example`: `dvislobokov/faster-whisper-large-v3-turbo-russian` | faster-whisper checkpoint |
| `DIARIZE_MODEL` | `pyannote/speaker-diarization-3.1` | Diarization checkpoint |
| `BATCH_SIZE` | 16 in `app.py` | Alignment batch size; not wired through `docker-compose.yml` |

## Models & licensing

| Service | Model | Notes |
| --- | --- | --- |
| Language | `shawnw3i/Qwen3.8-27B-AWQ-MTP` (current `.env.example`) | 27B, AWQ W4A16 (~14 GB), MTP head; optional presets include twolven abliterated and lued INT8 |
| Speech (ASR) | `dvislobokov/faster-whisper-large-v3-turbo-russian` | faster-whisper (CTranslate2), int8, Russian-tuned turbo |
| Speech (diarization) | `pyannote/speaker-diarization-3.1` | Gated Hugging Face model; token required |

Check each model card on Hugging Face for its specific license; several of the checkpoints used here are non-commercial or attribution-bearing. The `abliterated` preset in particular is a community derivative, so treat its license as whatever the base model and the abliteration transform declare.

## Version pins

Reproducibility matters on a stack where the only reason things work is that the pieces agree about the GPU. The exact versions that define a working build:

**Language:**

- Base image `nvidia/cuda:12.6.0-devel-ubuntu24.04`
- Python 3.12 (virtualenv; the system Python is PEP 668-protected)
- Torch `2.13.0` (+cu130)
- FlashInfer `0.6.16.post3`
- Transformers ≥ `5.5.3`
- vLLM + FlashQLA from the pinned wheels archive

**Speech:**

- Base image `nvidia/cuda:12.2.2-runtime-ubuntu22.04`
- Python 3.10
- Torch `+cu121`
- WhisperX (installed from source — the maintained fork)
- fastapi, uvicorn, python-multipart, ffmpeg

Pin these when you fork; drift is how a working stack quietly stops being one.

## Known caveats

- **Single in-flight sequence.** The language server is configured for one concurrent sequence (`max_num_seqs = 1`). Ideal for a personal endpoint and for correctness; it is not tuned for many simultaneous users. Raise it (and lower the context) if you are serving traffic.
- **Prefix caching is enabled.** `qwen/Dockerfile` passes `--enable-prefix-caching`, so repeated system prefixes should reuse already-computed state. At 128K context and TP=2, keep an eye on VRAM headroom.
- **Service ports are split.** `qwen` is published on host port **8000**, while `whisper` is published on **8001** (the speech service still listens on 8000 inside its container). This lets both Docker services run on the same host at the same time.
- **Generation defaults come from the checkpoint.** The model ships its own `generation_config.json`; sampling temperature, top-p, top-k, and repetition penalty are whatever that file says, unless a request overrides them. If output feels off, look at that file before suspecting the server.
- **The first request pays warm-up.** The very first inference after boot does extra work (kernel/JIT warming). Expect a slow start, then steady state.
- **Startup noise is partly normal.** Some warnings during the vLLM boot are expected and harmless — the interesting ones concern GPU capability mismatches and the attention backend.
- **Two different CUDA stacks.** The language service runs a CUDA-13 build; the speech service runs CUDA 12.1. That is intentional (each is pinned to what its dependency set supports) and it is why the two `Dockerfile`s do not look alike.

## Credits

- **NVIDIA** — the two cards that make all of this possible.
- **vLLM**, **faster-whisper / CTranslate2**, **WhisperX**, and the **pyannote** diarization pipeline — the inference engines behind each service.
- **Qwen** (and the community that AWQ-quantized and shipped the MTP builds we serve) — for the model itself.
- **FlashInfer**, **Marlin**, and **FFmpeg** — the kernels and codecs that move the actual work.

---

*This repository documents a real, running local stack — build it, run it, and replace the cloud you were paying for.*
