import asyncio
import json
import queue
import uuid
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional

import faiss
import numpy as np
import pyaudio
import sounddevice as sd
import torch
from llama_cpp import Llama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from vosk import KaldiRecognizer, Model

from logger_config import logger

class SpeechRecognizer:
    """ !!!!!!!Class for converting speech to text!!!!!!!!!!!!!!"""
    def __init__(self, model_path, sample_rate=16000, frame_size=4000):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.model = Model(str(model_path))
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False

    def open_stream(self):
        if self.stream is None or not self.stream.is_active():
            self.stream = self.audio.open(format=pyaudio.paInt16,
                                        channels=1,
                                        rate=self.sample_rate,
                                        input=True,
                                        frames_per_buffer=self.frame_size)
            self.stream.start_stream()
            logger.info('Audio stream opened.')

    def close_stream(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            logger.info('Audio stream closed.')
        self.audio.terminate()  
        self.audio = pyaudio.PyAudio()  
        logger.info('Audio stream closed and reset.')

    async def recognize_stream(self, websocket):
        self.stop()
        self.open_stream()
        rec = KaldiRecognizer(self.model, self.sample_rate)
        rec.Reset() 
        self.is_running = True
        last_final_text = ""
        last_partial_text = ""

        try:
            while self.is_running:
                try:
                    data = self.stream.read(self.frame_size, exception_on_overflow=False)
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        if "text" in result and result["text"].strip():
                            current_text = result["text"]
                            if current_text != last_final_text:
                                await websocket.send_json({
                                    "type": "final",
                                    "text": current_text
                                })
                                last_final_text = current_text
                                last_partial_text = ""
                    else:
                        partial = json.loads(rec.PartialResult())
                        if "partial" in partial and partial["partial"].strip():
                            current_partial = partial["partial"]
                            if (current_partial != last_partial_text and 
                                current_partial != last_final_text):
                                await websocket.send_json({
                                    "type": "partial",
                                    "text": current_partial
                                })
                                last_partial_text = current_partial
                    
                    # Give other tasks a chance to run
                    await asyncio.sleep(0.01)
                except Exception as e:
                    logger.error(f"Error during recognition: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "text": f"Ошибка распознавания: {str(e)}"
                    })
                    break
        finally:
            self.close_stream()
            self.is_running = False

    def stop(self):
        self.is_running = False

class TextToSpeechPlayer:
    """ Text to speech class """
    def __init__(self, model_path=r'lang_models/v3_1_ru.pt', sample_rate=48000):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        self.device = torch.device('cpu')
        torch.set_num_threads(4)
        self.model = torch.package.PackageImporter(str(self.model_path)).load_pickle("tts_models", "model")
        self.model.to(self.device)
        
        # Queue for texts
        self.text_queue = queue.Queue()
        self.sample_rate = sample_rate
        self.is_running = False
        
        # Start playback thread
        self.playing_thread = Thread(target=self._play_thread, daemon=True)
        self.is_running = True
        self.playing_thread.start()
        
    def _play_thread(self):
        while self.is_running:
            try:
                text = self.text_queue.get()
                if text is None:
                    break
                # Generate audio
                audio = self.model.apply_tts(text=text,
                                          speaker='kseniya',
                                          sample_rate=self.sample_rate)
               
                # Play through speakers
                sd.play(audio, self.sample_rate)
                sd.wait()  # Wait for playback to finish
            except Exception as e:
                print(f"Playback error: {e}")
                
    def say(self, text):
        """Add text to the playback queue"""
        self.text_queue.put(text)
        
    def stop(self):
        """Stop playback and close thread"""
        self.is_running = False
        self.text_queue.put(None)
        self.playing_thread.join()

# LLM model for translations and explanations
class YSAssistant:
    """ Class for assistant """
    def __init__(self, model_path):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=16384,
            n_gpu_layers=50
        )
        self.question = ""
        
    def process_request(self, text):

        system_part1 = [
            f"You are helping process user messages for a personal note-taking assistant. "
            f"Each message is either: "
            f"1. A command to save a note. "
            f"2. A request to find previously saved notes. "
            f"Your task is to detect which type it is and act accordingly."
        ]
        system_part2 = [
            f"If it's a command to SAVE a note: "
            f"Extract and return the following 3 fields: "

            f"1. Type — choose one from this list only: "
            f"- meetings "
            f"- tasks "
            f"- recipes "
            f"- vocabulary "
            f"- other "

            f"2. Content — the full useful note, as-is. "
            f"3. Date — if the message includes a date or time "
            f"(e.g. 'Monday', 'tomorrow', 'April 8 at 10am'), convert it to ISO 8601 format. "
            f"Return the result in this format: "
            f"Type: [type] "
            f"Content: [note text] "
            f"Date: [ISO 8601 date or 'missing'] "
        ]

        system_part3 = [
            f"Examples: "

            f"Input: "
            f"'Dentist appointment on Monday at 10:00' "
            f"Output: "
            f"Type: meetings "
            f"Content: Dentist appointment on Monday at 10:00 "
            f"Date: 2025-04-07T10:00:00 "

            f"Input: "
            f"'#portuguese sorrir - to smile' "
            f"Output: "
            f"Type: vocabulary "
            f"Content: #portuguese sorrir - to smile "
            f"Date: missing "

            f"Input: "
            f"'Recipe for Sunday dinner: 8 cups pasta, 12 cups water...' "
            f"Output: "
            f"Type: recipes "
            f"Content: Recipe for Sunday dinner: 8 cups pasta, 12 cups water... "
            f"Date: 2025-04-06 "
        ]

        system_part4 = [
            f"If it's a request to find previously saved notes: "
            f"You must form a command to retrieve data. To do this, return one word: SEARCH"
        ]

        system_part5 = [
            f"If it's a request with two parts: "
            f"1. DATA: information from data base about notices "
            f"2. COMMAND: it's a request to find previously saved notes "
            f"You must form the result in this format: "
            f"Type: [type] "
            f"Content: [note text] "
            f"Date: [ISO 8601 date or 'missing'] "
            f"--- "
            f"Do not use data from previous messages under any circumstances,"
            f"  only what comes after DATA"
        ]

        system_part6 = [
            f"Example:",
            f"Input:",
            f"DATA:",
            f"[",
            f"    {{",
            f"        metadata: {{",
            f"            type: vocabulary,",
            f"            date: missing",
            f"            content: #portuguese sorrir - to smile,",
            f"        }}",
            f"    }},",
            f"    {{",
            f"        metadata: {{",
            f"            type: vocabulary,",
            f"            date: missing",
            f"            content: #spanish correr - to run,",
            f"        }}",
            f"    }},",
            f"    {{",
            f"        metadata: {{",
            f"            type: tasks,",
            f"            date: 2025-04-05",
            f"            content: Finish the LLM summary,",
            f"        }}",
            f"    }}",
            f"]",
            f"COMMAND:",
            f"Portuguese words to learn",
            f"Output:",
            f"Type: vocabulary",
            f"Content: #portuguese sorrir - to smile",
            f"Date: missing",
            f"---"
        ]

        # Get response from LLM
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_part1},
                {"role": "system", "content": system_part2},
                {"role": "system", "content": system_part3},
                {"role": "user", "content": "visit the exhibition on tuesday at 3pm"},
                {"role": "assistant", "content":
                    "Type: meetings\n"
                    "Content: visit the exhibition on tuesday at 3pm\n"
                    "Date: 2025-04-08T15:00:00"
                    },
                {"role": "user", "content": "perfect, answer in this style"},
                {"role": "user", "content": "visit the exhibition on tuesday at 3pm"},

                {"role": "system", "content": system_part4},
                {"role": "system", "content": system_part5},
                {"role": "system", "content": system_part6},
                {"role": "user", "content": "What plans do I have for this weekend?"},
                {"role": "assistant", "content": "SEARCH"},
                {"role": "user", "content": "perfect, it is correct"},

                {"role": "user", "content":
                    "DATA:\n"
                    "[\n"
                    "    {\n"
                    "        metadata: {\n"
                    "            type: meetings,\n"
                    "            date: 2024-04-07T10:00:00\n"
                    "            content: Dentist appointment on Monday at 10:00,\n"
                    "        }\n"
                    "    },\n"
                    "    {\n"
                    "        metadata: {\n"
                    "            type: meetings,\n"
                    "            date: 2024-04-11T15:00:00\n"
                    "            content: Call with John on Friday at 15:00,\n"
                    "        }\n"
                    "    }\n"
                    "]\n"
                    "COMMAND:\n"
                    "What are my plans for this week?"
                },
                {"role": "assistant", "content":
                    "Type: meetings\n"
                    "Content: Dentist appointment on Monday at 10:00\n"
                    "Date: 2024-04-07T10:00:00\n"
                    "---\n"
                    "Type: meetings\n"
                    "Content: Call with John on Friday at 15:00\n"
                    "Date: 2024-04-11T15:00:00"
                    "---\n"
                },
                {"role": "user", "content": "perfect, it is correct"},

                {"role": "user", "content": text},
            ]
        )
        self.llm.reset()
        return response["choices"][0]["message"]["content"].strip()
    


class NoteVectorStore:
    def __init__(self, persist_dir: str = "./faiss_data"):
        self.persist_dir = Path(persist_dir)
        self.embed_model = HuggingFaceEmbedding(model_name="../models/all-MiniLM-L6-v2")
        #self.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")
        self.notes: List[Dict] = []
        self.embeddings: List[np.ndarray] = []
        self.index = None
        self.dimension = None

        index_path = self.persist_dir / "index.faiss"
        notes_path = self.persist_dir / "notes.json"

        if notes_path.exists():
            with notes_path.open("r", encoding="utf-8") as f:
                self.notes = json.load(f)
            self._rebuild_index()
        else:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            print("Инициализация нового векторного хранилища")

    def _rebuild_index(self):
        if not self.notes:
            self.dimension = None
            self.index = None
            return
        self.embeddings = [self.embed_model.get_text_embedding(note["content"]) for note in self.notes]
        embedding_array = np.ascontiguousarray(np.array(self.embeddings).astype("float32"))
        self.dimension = embedding_array.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embedding_array)

    def add_note(self, content: str, note_type: str, event_date: Optional[str] = None):
        note_id = str(uuid.uuid4())
        embedding = self.embed_model.get_text_embedding(content)
        embedding_array = np.array(embedding).reshape(1, -1).astype("float32")

        if self.index is None:
            self.dimension = embedding_array.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
        elif embedding_array.shape[1] != self.dimension:
            raise ValueError("Размерность эмбеддинга не совпадает с индексом")

        self.index.add(embedding_array)
        self.embeddings.append(embedding)

        metadata = {
            "note_id": note_id,
            "type": note_type,
            "event_date": event_date if event_date else "missing",
            "created_at": datetime.now().isoformat(),
            "content": content
        }
        self.notes.append(metadata)

    def semantic_search(self, query: str, threshold: float = 10.8) -> List[Dict]:
        if self.index is None or not self.notes:
            return []

        query_embedding = self.embed_model.get_text_embedding(query)
        embedding_array = np.array(query_embedding).reshape(1, -1).astype("float32")

        if embedding_array.shape[1] != self.dimension:
            raise ValueError("Размерность запроса не совпадает с размерностью индекса")

        distances, indices = self.index.search(embedding_array, len(self.notes))
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if distance < threshold:
                metadata = self.notes[idx]
                results.append({
                    "metadata": metadata,
                })
        return results

    def delete_note(self, note_id: str):
        idx_to_delete = next((i for i, note in enumerate(self.notes) if note["note_id"] == note_id), None)
        if idx_to_delete is not None:
            del self.notes[idx_to_delete]
            del self.embeddings[idx_to_delete]
            self._rebuild_index()

    def persist(self):
        index_path = self.persist_dir / "index.faiss"
        notes_path = self.persist_dir / "notes.json"
        if self.index is not None:
            faiss.write_index(self.index, str(index_path))
        with notes_path.open("w", encoding="utf-8") as f:
            json.dump(self.notes, f, ensure_ascii=False, indent=2)

    def load(self):
        index_path = self.persist_dir / "index.faiss"
        notes_path = self.persist_dir / "notes.json"
        if notes_path.exists():
            with notes_path.open("r", encoding="utf-8") as f:
                self.notes = json.load(f)
            self._rebuild_index()