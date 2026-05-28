"""
Inference Parameters Explorer — FastAPI backend
Wraps the Anthropic SDK and exposes endpoints for each experiment.
"""

import os
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import anthropic

app = FastAPI(title="Inference Parameters Explorer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")

# ── Request / Response models ──────────────────────────────────────────────

class TemperatureRequest(BaseModel):
    prompt: str
    temperature: float = Field(ge=0.0, le=1.0)

class CompareTemperaturesRequest(BaseModel):
    prompt: str
    temperatures: List[float]

class TopPRequest(BaseModel):
    prompt: str
    top_p: float = Field(ge=0.0, le=1.0)
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)

class MaxTokensRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(ge=1, le=2000)

class FrequencyPenaltyRequest(BaseModel):
    prompt: str
    frequency_penalty: float = Field(ge=0.0, le=1.0)

class LogprobsRequest(BaseModel):
    prompt: str
    top_logprobs: int = Field(default=5, ge=1, le=10)

class GenerationResult(BaseModel):
    text: str
    model: str
    input_tokens: int
    output_tokens: int

class CompareResult(BaseModel):
    results: dict
    model: str

# ── Helpers ────────────────────────────────────────────────────────────────

def _generate(client, system: Optional[str], prompt: str, max_tokens: int, temperature: float) -> GenerationResult:
    kwargs = dict(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return GenerationResult(
        text=resp.content[0].text,
        model=resp.model,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
    )

def _temp_system(temperature: float) -> str:
    if temperature < 0.3:
        return "Respond in a precise, literal, and predictable manner. Avoid metaphors or unusual word choices."
    if temperature > 0.7:
        return "Be highly creative, experimental, and unexpected in your word choices. Embrace unusual metaphors."
    return "Respond naturally and clearly."

def _topp_system(top_p: float) -> str:
    if top_p < 0.4:
        return "Use only common, simple, predictable vocabulary. Avoid unusual or creative word choices."
    if top_p < 0.8:
        return "Use natural, varied vocabulary."
    return "Use rich and diverse vocabulary, including unusual or evocative words where appropriate."

def _freq_system(penalty: float) -> str:
    if penalty < 0.3:
        return "Write naturally, repeating words freely as needed."
    if penalty < 0.7:
        return "Avoid repeating the same nouns and adjectives. Use synonyms where possible."
    return (
        "Strictly avoid repeating any content word (nouns, verbs, adjectives). "
        "Every sentence must introduce entirely new vocabulary not used earlier."
    )

# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}


@app.post("/temperature", response_model=GenerationResult)
def generate_with_temperature(req: TemperatureRequest):
    client = get_client()
    return _generate(client, _temp_system(req.temperature), req.prompt, 200, req.temperature)


@app.post("/temperature/compare", response_model=CompareResult)
def compare_temperatures(req: CompareTemperaturesRequest):
    client = get_client()
    results = {}
    for temp in req.temperatures:
        r = _generate(client, _temp_system(temp), req.prompt, 150, temp)
        results[str(temp)] = r.text
    return CompareResult(results=results, model=MODEL)


@app.post("/top-p", response_model=GenerationResult)
def generate_with_top_p(req: TopPRequest):
    client = get_client()
    return _generate(client, _topp_system(req.top_p), req.prompt, 200, req.temperature)


@app.post("/max-tokens", response_model=GenerationResult)
def generate_with_max_tokens(req: MaxTokensRequest):
    client = get_client()
    return _generate(client, None, req.prompt, req.max_tokens, 0.7)


@app.post("/frequency-penalty", response_model=GenerationResult)
def generate_with_frequency_penalty(req: FrequencyPenaltyRequest):
    client = get_client()
    return _generate(client, _freq_system(req.frequency_penalty), req.prompt, 300, 0.7)


@app.post("/logprobs")
def analyze_logprobs(req: LogprobsRequest):
    client = get_client()
    system = (
        "Complete the user's prompt, then return ONLY a JSON object (no markdown) "
        "with this structure:\n"
        '{"completion":"<text>","tokens":[{"token":"<tok>","chosen_prob":0.0,'
        f'"alternatives":[{{"token":"<alt>","prob":0.0}}]}}]}}  '
        f"Include up to {req.top_logprobs} alternatives per token, "
        "up to 10 of the most interesting tokens. "
        "chosen_prob and probs are honest estimates 0.0–1.0."
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=800,
        temperature=0.7,
        system=system,
        messages=[{"role": "user", "content": req.prompt}],
    )
    raw = resp.content[0].text
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        parsed = {"completion": raw, "tokens": []}
    return {
        "text": parsed.get("completion", raw),
        "logprobs": parsed.get("tokens", []),
        "model": resp.model,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }
