"""
Inference Parameters Explorer — standalone CLI script
Run all five experiments from your terminal without Docker.

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY="sk-ant-..."
    python inference_parameters.py
"""

import os
import json
from typing import Dict, List, Optional
import anthropic


class InferenceExplorer:
    def __init__(self, api_key: str, model: str = "claude-opus-4-5"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    # ── Temperature ───────────────────────────────────────────────────────

    def generate_with_temperature(self, prompt: str, temperature: float) -> str:
        system = (
            "Be very precise and literal. No creative flourishes."
            if temperature < 0.3
            else "Be creative, unexpected, and vivid."
            if temperature > 0.7
            else "Respond naturally."
        )
        resp = self.client.messages.create(
            model=self.model, max_tokens=200, temperature=temperature,
            system=system, messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    def compare_temperatures(self, prompt: str, temperatures: List[float]) -> Dict[float, str]:
        return {t: self.generate_with_temperature(prompt, t) for t in temperatures}

    # ── Top P ─────────────────────────────────────────────────────────────

    def generate_with_top_p(self, prompt: str, top_p: float, temperature: float = 1.0) -> str:
        system = (
            "Use only simple, common vocabulary."
            if top_p < 0.4
            else "Use rich, diverse vocabulary with unusual words."
            if top_p >= 0.8
            else "Use natural vocabulary."
        )
        resp = self.client.messages.create(
            model=self.model, max_tokens=200, temperature=temperature,
            system=system, messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    # ── Max tokens ────────────────────────────────────────────────────────

    def generate_with_max_tokens(self, prompt: str, max_tokens: int) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    # ── Frequency penalty ─────────────────────────────────────────────────

    def generate_with_frequency_penalty(self, prompt: str, frequency_penalty: float) -> str:
        system = (
            "Write naturally, repeating words freely."
            if frequency_penalty < 0.3
            else "Strictly avoid repeating any content word. Every sentence must use entirely new vocabulary."
            if frequency_penalty >= 0.7
            else "Avoid repeating nouns and adjectives where possible."
        )
        resp = self.client.messages.create(
            model=self.model, max_tokens=300, temperature=0.7,
            system=system, messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    # ── Logprobs ──────────────────────────────────────────────────────────

    def analyze_logprobs(self, prompt: str, top_logprobs: int = 5) -> Dict:
        system = (
            "Complete the user's prompt, then return ONLY a JSON object (no markdown) "
            'with structure: {"completion":"<text>","tokens":[{"token":"<tok>",'
            '"chosen_prob":0.0,"alternatives":[{"token":"<alt>","prob":0.0}]}]} '
            f"Include up to {top_logprobs} alternatives, up to 10 interesting tokens."
        )
        resp = self.client.messages.create(
            model=self.model, max_tokens=800, temperature=0.7,
            system=system, messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text
        try:
            parsed = json.loads(raw.strip())
        except json.JSONDecodeError:
            parsed = {"completion": raw, "tokens": []}
        return {"text": parsed.get("completion", raw), "logprobs": parsed.get("tokens", [])}


# ── Experiments ───────────────────────────────────────────────────────────

def sep(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def experiment_1_temperature_effects():
    sep("EXPERIMENT 1 — Temperature effects")
    explorer = InferenceExplorer(os.environ["ANTHROPIC_API_KEY"])
    prompt = "Write a creative opening sentence for a science fiction story about time travel."
    for temp, text in explorer.compare_temperatures(prompt, [0.0, 0.5, 1.0]).items():
        print(f"\n🌡  Temperature {temp}")
        print(f"   {text}")


def experiment_2_top_p_effects():
    sep("EXPERIMENT 2 — Top P (nucleus sampling)")
    explorer = InferenceExplorer(os.environ["ANTHROPIC_API_KEY"])
    prompt = "Describe an alien planet in three sentences."
    for top_p in [0.2, 0.6, 1.0]:
        text = explorer.generate_with_top_p(prompt, top_p)
        label = "narrow vocab" if top_p < 0.4 else "balanced" if top_p < 0.8 else "rich vocab"
        print(f"\n🎯  top_p {top_p} ({label})")
        print(f"   {text}")


def experiment_3_max_tokens():
    sep("EXPERIMENT 3 — Max tokens")
    explorer = InferenceExplorer(os.environ["ANTHROPIC_API_KEY"])
    prompt = "Explain the concept of machine learning in simple terms."
    for limit in [30, 100, 300]:
        text = explorer.generate_with_max_tokens(prompt, limit)
        print(f"\n📏  max_tokens={limit}  (~{len(text.split())} words)")
        print(f"   {text}")


def experiment_4_frequency_penalty():
    sep("EXPERIMENT 4 — Frequency / repetition penalty")
    explorer = InferenceExplorer(os.environ["ANTHROPIC_API_KEY"])
    prompt = "Write a paragraph about the ocean using vivid language."
    for penalty in [0.0, 0.5, 1.0]:
        text = explorer.generate_with_frequency_penalty(prompt, penalty)
        label = "no penalty" if penalty == 0 else "moderate" if penalty < 0.7 else "strict"
        print(f"\n🔁  penalty={penalty} ({label})")
        print(f"   {text}")


def experiment_5_logprobs():
    sep("EXPERIMENT 5 — Logprobs (token probability analysis)")
    explorer = InferenceExplorer(os.environ["ANTHROPIC_API_KEY"])
    result = explorer.analyze_logprobs('Complete: "The best programming language is"', top_logprobs=5)
    print(f"\n📝  {result['text']}\n")
    bar_width = 28
    for t in result["logprobs"]:
        prob = t.get("chosen_prob", 0)
        bar = "█" * int(prob * bar_width) + "░" * (bar_width - int(prob * bar_width))
        print(f"  {t['token']:<20} [{bar}] {prob*100:.1f}%")
        alts = t.get("alternatives", [])
        if alts:
            print("    alts: " + "  ".join(f"{a['token']}({a['prob']*100:.0f}%)" for a in alts[:3]))


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        return
    experiment_1_temperature_effects()
    experiment_2_top_p_effects()
    experiment_3_max_tokens()
    experiment_4_frequency_penalty()
    experiment_5_logprobs()


if __name__ == "__main__":
    main()
