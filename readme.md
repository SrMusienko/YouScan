# Personal Note-Taking Application MVP

## Overview

This is a Minimal Viable Product (MVP) for a personal note-taking application powered by Large Language Models (LLMs). Inspired by gadgets like Humane AI Pin and Omi, this application aims to provide a seamless and intuitive way to capture, access, and interact with your personal notes using natural language.

The core functionalities include flexible note storage, quick access to recent notes, and intuitive natural language search powered by an LLM.


## Tools and Architecture

### Choice of Tools

* **FastAPI:** Chosen as the backend framework for its simplicity, high performance, and built-in support for asynchronous operations (async/await), which is crucial for handling LLM interactions and WebSocket connections efficiently.
* **Jinja2:** Used for templating the simple web interface for demonstration purposes.
* **Vosk:** Selected as the speech recognition library for its open-source nature and support for local models, enabling offline speech-to-text capabilities. It also supports multiple languages.
* **FAISS (via `NoteVectorStore`):** Utilized for efficient similarity search over the notes, enabling the natural language search functionality. Notes are vectorized and stored in a FAISS index for quick retrieval.
* **Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf (Local LLM):** A quantized version of the Llama 3 model is used for processing natural language queries and extracting information from notes. Local models offer benefits like cost-effectiveness and improved privacy.
* **Python Standard Libraries:** Used for basic operations like JSON handling and logging.

### Overall Architecture

The application follows a client-server architecture:

1.  **Frontend (Simple Web Interface):** A basic HTML page with JavaScript for:
    * Recording audio and sending it to the backend via WebSocket for speech-to-text.
    * Submitting text notes and search queries to the backend via HTTP POST requests.
    * Displaying recent notes and search results.

2.  **Backend (FastAPI Application):**
    * **`/` (GET):** Serves the main HTML page.
    * **`/ws/recognize` (WebSocket):** Handles real-time audio input, performs speech recognition using the local Vosk model, and sends the transcribed text back to the frontend.
    * **`/get_anal` (POST):** Receives text notes or search queries.
        * If it's a note, it's processed by the local LLM to potentially extract structured information (currently basic parsing is implemented). The note content (and extracted information) is then stored in the `NoteVectorStore`.
        * If it's a search query, the `NoteVectorStore` performs a semantic search using FAISS. The top results are then fed back to the LLM to generate a concise answer based on the retrieved notes.
    * **`/get_lang` (POST):** Allows changing the speech recognition language.
    * **`/delete_note` (POST):** (Currently a placeholder) Intended for future functionality to delete specific notes.
    * **`NoteVectorStore`:** A custom class that handles storing notes, vectorizing them using a simple approach (could be enhanced with more sophisticated embeddings in the future), and performing similarity searches using FAISS.
    * **`YSAssistant`:** A simple wrapper around the local LLM, responsible for generating responses based on the input prompts.
    * **`SpeechRecognizer`:** A class that encapsulates the Vosk speech recognition functionality.

### Why Local Models?

The decision to use local LLMs (like the quantized Llama 3 model) and a local speech recognition model (Vosk) was driven by several key considerations:

* **Cost-Effectiveness:** Eliminates the recurring costs associated with cloud-based LLM and speech-to-text APIs.
* **Reliability and Stability:** Reduces dependency on external services, which can be subject to API changes, outages, or rate limits.
* **Privacy:** Keeps user data and processing on the local machine, enhancing privacy.
* **Prompt Engineering Efficiency (on weaker hardware):** If a relatively "dumb" quantized model can understand and process the prompts effectively, it suggests that the prompt design is robust and likely to work well even with more powerful cloud-based models. This is particularly relevant when developing on resource-constrained hardware.
* **Offline Functionality:** Enables basic note-taking and search capabilities even without an internet connection (for speech recognition and LLM processing).


### Demo Video

A short demo video showcasing the key functionalities of the application can be found here:

[https://www.youtube.com/watch?v=rcef62aBh8c](https://www.youtube.com/watch?v=rcef62aBh8c)

*(Replace `https://youtu.be/your-youtube-video-id` with the actual link to your demo video once you record and upload it.)*

## Essential Functionalities

The MVP includes the following essential functionalities:

* **Flexible Note Storage:** Users can input notes in various formats, as demonstrated in the examples provided in the problem description. These notes are stored as text and associated with vector embeddings for semantic search.
* **Quick Access to Recent Notes:** While not explicitly implemented in the current UI, the backend (`NoteVectorStore`) could be easily extended to keep track of and retrieve the last N added notes. This could be displayed on the frontend.
* **Intuitive Natural Language Search:** Users can ask questions in natural language (e.g., "What plans do I have for this weekend?", "Show recent notes about LLM", "Portuguese words to learn"). The backend performs a semantic search over the stored notes and uses the LLM to provide a relevant answer based on the search results.

## Optional Advanced Functionalities (Initial Considerations)

The current MVP lays the groundwork for several advanced functionalities:

* **Automatically Extract Structured Data:** The `/get_anal` endpoint currently performs basic parsing based on the LLM's output. This could be significantly enhanced to reliably extract structured data like appointments (date, time), tasks (description), recipes (ingredients, steps), and vocabulary (word, translation). This would involve more sophisticated prompt engineering or fine-tuning of the LLM.
* **Automatically Classify or Tag Notes:** The LLM could be used to analyze the content of each note and automatically assign relevant categories or tags. This would improve organization and make it easier to filter and retrieve notes based on topics.
* **Improved User Experience:**
    * A more user-friendly interface for viewing, editing, and deleting notes.
    * The ability to organize notes into notebooks or folders.
    * Context-aware suggestions based on recent notes or current tasks.
    * Integration with other personal information management tools (calendars, task managers).

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-github-username/llm-powered-note-taker-mvp.git](https://www.google.com/search?q=https://github.com/your-github-username/llm-powered-note-taker-mvp.git)
    cd llm-powered-note-taker-mvp
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download the necessary models:**
    * Download the `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` model and place it in the `models` directory. You might need to agree to licensing terms from the model provider.
    * Download the Vosk language models (e.g., `vosk-model-small-en-us-zamia-0.5`, `vosk-model-small-ru-0.22`, `vosk-model-small-uk-v3-small`) and place them in the `models` directory.
4.  **Run the FastAPI application:**
    ```bash
    uvicorn main:app --reload
    ```
5.  **Open your web browser and navigate to `http://localhost:5000`.**
