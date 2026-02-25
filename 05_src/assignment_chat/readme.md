# Assignment 2 Chat Client

This implementation provides a conversational AI assistant called **North Star Guide**.
The assistant uses a chat interface (Gradio), keeps short-term conversation memory through the `history` argument, and routes requests to three services via OpenAI function calling.

## Services

### Service 1: API Calls

- Implemented by `get_weather_summary_data`.
- Backend API: Open-Meteo geocoding + forecast endpoints.
- The tool returns structured data, and the model rewrites it into natural user-facing responses (not verbatim API output).

### Service 2: Semantic Query

- Implemented by `search_course_knowledge`.
- Uses `chromadb.PersistentClient` with local persistence at `./05_src/assignment_chat/chroma_store`.
- Embeddings are generated with `text-embedding-3-small` through the configured OpenAI client.
- Retrieval runs with `collection.query(query_embeddings=[...])`.

### Service 3: Your Choice (Function Calling)

- Implemented by `build_weekly_study_plan`.
- Uses function calling to build a structured weekly learning plan from topic, time budget, and level.

## Guardrails and Limitations

Guardrails are enforced in two layers:

1. **Pre-check guardrails** (`guardrail_response` in `main.py`):
- Blocks attempts to reveal or modify the system prompt.
- Refuses restricted topics:
  - cats or dogs
  - horoscopes or zodiac signs
  - Taylor Swift

2. **System instructions guardrails** (`SYSTEM_INSTRUCTIONS`):
- Reiterates prompt-protection and restricted-topic rules inside model instructions.

## Memory and UI

- Chat interface is implemented in `app.py` using `gr.ChatInterface(type="messages")`.
- The chat function uses the Gradio `history` input and keeps a rolling window (`MAX_HISTORY_MESSAGES`) for short-term memory.

## Files

- `main.py`: model client setup, tools/services, guardrails, and chat orchestration.
- `app.py`: Gradio app entrypoint.
- `readme.md`: implementation notes.

## Run

From the repository root:

```bash
PYTHONPATH=05_src python -m assignment_chat.app
```

Or from `05_src`:

```bash
cd 05_src
python -m assignment_chat.app
```

Environment:

- Uses existing course setup only (no new libraries required).
- Requires either:
  - `API_GATEWAY_KEY` (course gateway mode), or
  - valid `OPENAI_API_KEY` (direct OpenAI mode).
