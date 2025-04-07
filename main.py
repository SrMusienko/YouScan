import json
import asyncio
import logging
from typing import Dict, List

from fastapi import (
    FastAPI,
    Request,
    WebSocket,
    WebSocketDisconnect,
    Form,
    Body,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from services import (
    SpeechRecognizer,
    YSAssistant,
    NoteVectorStore,
)

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Константы и глобальные переменные
MODEL_ASSISTANT = "../models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
speech_recognizer = None
ys_assistant = None
model_sp_rec_lang = "en"

# Инициализация FastAPI
app = FastAPI(title="YouScan")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# Модели запросов

class LangRequest(BaseModel):
    lang: str


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Главная страница"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws/recognize")
async def websocket_endpoint(websocket: WebSocket):
    global speech_recognizer, model_sp_rec_lang

    await websocket.accept()

    if speech_recognizer is None:
        try:
            model_map = {
                "en": "../models/vosk-model-small-en-us-zamia-0.5",
                "ru": "../models/vosk-model-small-ru-0.22",
                "uk": "../models/vosk-model-small-uk-v3-small",
            }
            model_path = model_map.get(model_sp_rec_lang)
            if model_path:
                speech_recognizer = SpeechRecognizer(model_path)
            else:
                raise ValueError("Unsupported language model")
        except Exception as e:
            await websocket.send_json({"type": "error", "text": f"Initialization error: {str(e)}"})
            await websocket.close()
            return

    try:
        if speech_recognizer.is_running:
            speech_recognizer.stop()

        recognition_task = asyncio.create_task(speech_recognizer.recognize_stream(websocket))

        while True:
            try:
                msg = await websocket.receive_text()
                data = json.loads(msg)
                if data.get("action") == "stop":
                    speech_recognizer.stop()
                    break
            except WebSocketDisconnect:
                speech_recognizer.stop()
                break
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                break

        await recognition_task

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        if speech_recognizer:
            speech_recognizer.stop()
    except Exception as e:
        logger.error(f"Error in websocket endpoint: {e}")
        if speech_recognizer:
            speech_recognizer.stop()


def parse_model_output(text: str) -> List[Dict[str, str]]:
    blocks = text.strip().split("---")
    parsed_blocks = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        entry = {}
        lines = block.split("\n")
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                entry[key.strip().lower()] = value.strip()

        if "type" in entry and "content" in entry:
            entry.setdefault("date", "missing")
            parsed_blocks.append(entry)

    return parsed_blocks


@app.post("/get_anal")
async def get_anal(text: str = Form(...)):
    global ys_assistant

    if ys_assistant is None:
        ys_assistant = YSAssistant(MODEL_ASSISTANT)

    message = ys_assistant.process_request(text)
    store = NoteVectorStore("./faiss_data")

    if "SEARCH" in message:
        data = store.semantic_search(text)
        if len(data) < 1:
            return JSONResponse(content={"message": "no found information"})
        final_prompt = f"DATA:\n{json.dumps(data, indent=4, ensure_ascii=False)}\nCOMMAND:\n{text}"
        message = ys_assistant.process_request(final_prompt)
        return parse_model_output(message)

    parsed_notes = parse_model_output(message)

    if parsed_notes:
        for note in parsed_notes:
            store.add_note(
                content=note.get("content"),
                note_type=note.get("type"),
                event_date=note.get("date") if note.get("date") != "missing" else None,
            )
        store.persist()
        return JSONResponse(content={"message": f"note(s) were added.\n{message}"})

    return JSONResponse(status_code=400, content={"message": "Could not parse model output."})


@app.post("/get_lang")
async def get_lang(data: LangRequest):
    global model_sp_rec_lang, speech_recognizer
    speech_recognizer = None
    model_sp_rec_lang = data.lang
    return {"message": f"Language changed to {data.lang}"}


@app.post("/delete_note")
async def delete_note_endpoint(request: Request, note_id: Dict = Body(...)):
    note_to_delete_id = note_id.get("note_id")
    return JSONResponse(content={"message": f"Note with ID {note_to_delete_id} deleted."})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)