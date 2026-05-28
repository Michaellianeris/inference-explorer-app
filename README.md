# Inference Parameters Explorer

A hands-on lab for understanding how LLM inference parameters affect output quality, creativity, and consistency. Includes a **standalone CLI script**, a **FastAPI backend**, and a **dark-theme web UI** — all containerised with Docker.

---

## Project structure

```
inference-explorer/
├── inference_parameters.py   # Standalone CLI — no server needed
├── docker-compose.yml
├── .env.example
├── backend/
│   ├── main.py               # FastAPI app (5 endpoints)
│   ├── requirements.txt
│   └── Dockerfile
└── frontend/
    ├── index.html            # Single-file UI
    ├── nginx.conf
    └── Dockerfile
```

---

## Quick start

### Option A — CLI only (no Docker)

```bash
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
python inference_parameters.py
```

All five experiments run sequentially and print formatted results to the terminal.

---

### Option B — Full stack with Docker Compose

**1. Copy and fill in your API key**

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY
```

**2. Build and start**

```bash
docker compose up --build
```

**3. Open the UI**

```
http://localhost:3000
```

The backend API is available at `http://localhost:8000`.

---

## The five experiments

| # | Parameter | What it controls | Tested values |
|---|---|---|---|
| 1 | `temperature` | Randomness / creativity | 0.0, 0.5, 1.0 |
| 2 | `top_p` | Vocabulary breadth (nucleus sampling) | 0.2, 0.6, 1.0 |
| 3 | `max_tokens` | Hard response length cap | 30, 100, 300 |
| 4 | `frequency_penalty` | Penalises repeated tokens | 0.0, 0.5, 1.0 |
| 5 | `logprobs` | Per-token probability scores | top 5 alternatives |

### Key concepts

**Temperature vs Top P**
- `temperature` scales the entire probability distribution — lower = sharper, higher = flatter.
- `top_p` cuts off the tail of low-probability tokens entirely. They are independent axes; in practice, adjust one at a time.

**Anthropic-specific notes**
- Claude accepts `temperature` (0.0–1.0) and `max_tokens` natively.
- `top_p` and `frequency_penalty` are not exposed as numeric knobs. This project simulates their effects via calibrated system instructions, which produces equivalent behavioural output.
- `logprobs` (raw token probabilities) are not returned by the Anthropic API. Experiment 5 asks the model to introspect and return structured JSON with confidence estimates — a useful approximation for learning purposes.

---

## API endpoints

| Method | Path | Body |
|---|---|---|
| GET | `/health` | — |
| POST | `/temperature` | `{prompt, temperature}` |
| POST | `/temperature/compare` | `{prompt, temperatures:[]}` |
| POST | `/top-p` | `{prompt, top_p, temperature?}` |
| POST | `/max-tokens` | `{prompt, max_tokens}` |
| POST | `/frequency-penalty` | `{prompt, frequency_penalty}` |
| POST | `/logprobs` | `{prompt, top_logprobs?}` |

All endpoints return `{text, model, input_tokens, output_tokens}`.

Example with curl:

```bash
curl -X POST http://localhost:8000/temperature \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Tell me a joke","temperature":0.9}'
```

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | — | Your Anthropic API key |
| `ANTHROPIC_MODEL` | ❌ | `claude-opus-4-5` | Model to use |

---

## Development (without Docker)

**Backend**

```bash
cd backend
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
uvicorn main:app --reload --port 8000
```

**Frontend**

The frontend is a single `index.html` — open it directly in a browser or serve it with any static server:

```bash
cd frontend
python -m http.server 3000
```

The UI connects to `http://localhost:8000` by default.

---

## Requirements

- Docker ≥ 24 and Docker Compose v2, **or** Python 3.11+ for the CLI
- An [Anthropic API key](https://console.anthropic.com/)
