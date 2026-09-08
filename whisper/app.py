import os
import tempfile
import json
import torch
import ctranslate2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import whisperx
from whisperx.diarize import DiarizationPipeline

app = FastAPI(title="Distributed Whisper + Diarization API")

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_NAME = os.getenv("WHISPER_MODEL", "Systran/faster-whisper-large-v3")
DIARIZE_MODEL_NAME = os.getenv("DIARIZE_MODEL", "pyannote/speaker-diarization-3.1")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))

# Вспомогательная функция загрузки ASR (с проверкой на CTranslate2)
def load_whisper_model(model_name: str):
    if os.path.exists(model_name) or "faster-whisper" in model_name or "turbo" in model_name:
        return whisperx.load_model(model_name, device="cuda", device_index=0, compute_type="int8", language="ru")

    # Конвертация HF PyTorch-моделей (если передана не-CTranslate2 модель)
    ct2_dir = f"/tmp/ct2_{model_name.replace('/', '_')}"
    if not os.path.exists(os.path.join(ct2_dir, "model.bin")):
        print(f"Конвертация {model_name} в CTranslate2 format...")
        converter = ctranslate2.converters.TransformersConverter(model_name, copy_files=["tokenizer.json", "preprocessor_config.json"])
        converter.convert(output_dir=ct2_dir, quantization="int8", force=True)
    return whisperx.load_model(ct2_dir, device="cuda", device_index=0, compute_type="int8", language="ru")

print(f"Загрузка Whisper ({MODEL_NAME}) на cuda:0...")
model = load_whisper_model(MODEL_NAME)

print(f"Загрузка PyAnnote Diarization ({DIARIZE_MODEL_NAME}) на cuda:1...")
diarize_model = DiarizationPipeline(
    model_name=DIARIZE_MODEL_NAME,
    token=HF_TOKEN,
    device="cuda:1"
)

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    num_speakers: int = Form(None)
):
    suffix = os.path.splitext(file.filename)[-1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        audio = whisperx.load_audio(tmp_path)
        result = model.transcribe(audio, batch_size=BATCH_SIZE)

        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device="cuda:0"
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            "cuda:0",
            return_char_alignments=False
        )

        diarize_segments = diarize_model(
            audio,
            min_speakers=num_speakers,
            max_speakers=num_speakers
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)

        formatted_dialogue = []
        for segment in result["segments"]:
            formatted_dialogue.append({
                "speaker": segment.get("speaker", "SPEAKER_UNKNOWN"),
                "start": round(segment["start"], 2),
                "end": round(segment["end"], 2),
                "text": segment.get("text", "").strip()
            })

        return JSONResponse(content={"dialogue": formatted_dialogue})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
