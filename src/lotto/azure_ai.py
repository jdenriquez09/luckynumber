"""
Azure AI Foundry (OpenAI) client for natural-language lotto statistics interpretation.

Reads credentials from environment variables (or a .env file in the project root):
    AZURE_OPENAI_ENDPOINT    https://YOUR-RESOURCE.openai.azure.com/
    AZURE_OPENAI_API_KEY     your Azure API key
    AZURE_OPENAI_DEPLOYMENT  your model deployment name (e.g. "sol", "gpt-4o")
    AZURE_OPENAI_API_VERSION (optional, defaults to 2024-02-01)
"""
from __future__ import annotations

import os
from pathlib import Path

# Load .env from project root (C:\luckynumber\.env) if present
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[2] / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

# Instructions prepended to every user message (avoids 'system' role rejection on o1/o3)
_INSTRUCTIONS = (
    "You are a data analyst interpreting historical lottery statistics.\n"
    "RULES: (1) Always state that past draws CANNOT predict future draws. "
    "(2) Lotto draws are independent random events with no memory. "
    "(3) Explain patterns in plain language for a non-technical audience. "
    "(4) Do NOT call any number 'lucky' or 'due'. "
    "(5) Frame all findings as historical observations only — not gambling advice.\n\n"
)


class AzureAIClient:
    """Synchronous wrapper around the openai SDK pointing at Azure AI Foundry.
    Uses the sync client to avoid asyncio event-loop conflicts inside Streamlit."""

    def __init__(self) -> None:
        self._endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        self._api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        self._deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
        self._api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")

    def is_configured(self) -> bool:
        return bool(self._endpoint and self._api_key and self._deployment)

    def setup_instructions(self) -> str:
        missing = [
            v for v, val in [
                ("AZURE_OPENAI_ENDPOINT", self._endpoint),
                ("AZURE_OPENAI_API_KEY", self._api_key),
                ("AZURE_OPENAI_DEPLOYMENT", self._deployment),
            ] if not val
        ]
        return (
            f"Azure AI is not configured. Missing env vars: {', '.join(missing)}.\n"
            "Create a `.env` file in C:\\luckynumber\\ with:\n"
            "  AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/\n"
            "  AZURE_OPENAI_API_KEY=your-key-here\n"
            "  AZURE_OPENAI_DEPLOYMENT=sol\n"
            "Then restart the dashboard."
        )

    def _client(self):
        from openai import AzureOpenAI
        return AzureOpenAI(
            azure_endpoint=self._endpoint,
            api_key=self._api_key,
            api_version=self._api_version,
        )

    def test_connection(self) -> str:
        """Send a minimal prompt to verify credentials and deployment are working."""
        if not self.is_configured():
            raise RuntimeError(self.setup_instructions())
        response = self._client().chat.completions.create(
            model=self._deployment,
            messages=[{"role": "user", "content": "Reply with exactly the word: OK"}],
            max_completion_tokens=10,
        )
        return response.choices[0].message.content or "(empty response)"

    def interpret_lotto_stats(
        self,
        game_name: str,
        slots: list[dict],   # [{"label": "2:00 PM", "heat_map_rows": [...], "total_draws": N}, ...]
        chi_sq_result: str,
        gap_result: str,
        picks: int,
    ) -> dict[str, dict]:
        """
        Call the Azure OpenAI deployment with per-draw-time lotto statistics.

        For single-draw games, pass one slot; the return dict has one key.
        For multi-draw games (2D/3D), pass three slots; the return dict has three keys.

        Returns a dict keyed by slot label, each value is:
            {combination: list[int], reasoning: str, disclaimer: str}
        """
        if not self.is_configured():
            raise RuntimeError(self.setup_instructions())

        slot_blocks = []
        for slot in slots:
            label = slot["label"]
            rows = slot["heat_map_rows"]
            n_draws = slot["total_draws"]
            score_lines = "\n".join(
                f"  {r['number']:>3d} | heat={r['heat_score']:.3f} | "
                f"gap_score={r['gap_score']:.3f} | composite={r['composite_score']:.3f} | "
                f"total_draws={r['total_draws']} | current_gap={r['current_gap']}"
                for r in sorted(rows, key=lambda x: x["composite_score"], reverse=True)
            )
            slot_blocks.append(
                f"--- DRAW TIME: {label} ({n_draws} draws) ---\n"
                "num | heat_score | gap_score | composite | total_draws | current_gap\n"
                + score_lines
            )

        multi = len(slots) > 1
        slot_instructions = (
            "For EACH draw time above, respond with one block in this EXACT format:\n"
            "SLOT: <draw time label>\n"
            "COMBINATION: <comma-separated numbers, sorted ascending>\n"
            "REASONING: <1-2 sentences on why these numbers score well for this draw time>\n"
            "DISCLAIMER: <one sentence — historical data only, not a prediction>\n\n"
            "Output all blocks back-to-back, one per draw time."
        ) if multi else (
            "Respond in this EXACT format (no extra text before or after):\n"
            "COMBINATION: <comma-separated numbers, sorted ascending>\n"
            "REASONING: <2-4 sentences explaining why these numbers score well statistically>\n"
            "DISCLAIMER: <one sentence reminding that this is historical data only, not a prediction>"
        )

        user_content = (
            _INSTRUCTIONS
            + f"Game: {game_name}  |  Pick {picks} numbers per combination\n\n"
            "=== Per-Number Scores (composite = 0.5×heat + 0.5×gap, normalised 0-1) ===\n\n"
            + "\n\n".join(slot_blocks) + "\n\n"
            f"=== Chi-Square Test (overall) ===\n{chi_sq_result}\n\n"
            f"=== Gap Analysis (overall) ===\n{gap_result}\n\n"
            f"TASK: For each draw time, select the {picks} numbers that best balance "
            "high recent frequency (heat_score) and being statistically overdue (gap_score). "
            "Each draw time should have a DIFFERENT combination since the historical patterns differ.\n\n"
            + slot_instructions
        )

        max_tokens = 400 * len(slots)
        response = self._client().chat.completions.create(
            model=self._deployment,
            messages=[{"role": "user", "content": user_content}],
            max_completion_tokens=max_tokens,
        )
        raw = response.choices[0].message.content or ""

        if multi:
            return _parse_multi_slot_response(raw, picks, [s["label"] for s in slots])
        else:
            label = slots[0]["label"]
            return {label: _parse_ai_response(raw, picks)}


    def explain_chart(
        self,
        game_name: str,
        chart_name: str,
        key_facts: str,
    ) -> str:
        """
        Generate a plain-language 3-4 sentence explanation of one chart.
        Written for a non-technical audience — no jargon, no formulas.
        """
        if not self.is_configured():
            raise RuntimeError(self.setup_instructions())

        user_content = (
            _INSTRUCTIONS
            + f"Game: {game_name}\n\n"
            f"Chart being explained: {chart_name}\n\n"
            f"Statistical facts extracted from this chart:\n{key_facts}\n\n"
            "Write 3-4 sentences explaining what this chart means in plain everyday language "
            "that anyone — not just a statistician — can understand.\n"
            "Rules:\n"
            "- Use simple words: instead of 'chi-square test' say 'a fairness test', "
            "instead of 'p-value' say 'the chance this happened by luck', "
            "instead of 'z-score' say 'how unusual compared to normal', "
            "instead of 'statistically significant' say 'more than just luck'.\n"
            "- Name specific numbers from the facts to make it concrete.\n"
            "- Focus on the single most interesting or surprising finding.\n"
            "- End with one sentence reminding the reader this is historical data only "
            "and cannot tell us what will happen in future draws.\n"
            "Write as one flowing paragraph. No bullet points. No headers. No markdown.\n\n"
            "EXPLANATION:"
        )

        import re as _re
        response = self._client().chat.completions.create(
            model=self._deployment,
            messages=[{"role": "user", "content": user_content}],
            max_completion_tokens=300,
        )
        raw = response.choices[0].message.content or ""
        raw = _re.sub(r"(?i)^explanation\s*:\s*", "", raw.strip())
        return raw.strip()


def _parse_multi_slot_response(raw: str, picks: int, labels: list[str]) -> dict[str, dict]:
    """Split the model response on SLOT: headers and parse each block."""
    import re

    result: dict[str, dict] = {}
    # Split on "SLOT:" lines (case-insensitive)
    parts = re.split(r"(?im)^SLOT\s*:\s*", raw)
    for part in parts[1:]:  # skip text before first SLOT:
        lines = part.strip().splitlines()
        if not lines:
            continue
        slot_label = lines[0].strip()
        block = "\n".join(lines[1:])
        # Match label to known slots (fuzzy: check if any label substring matches)
        matched_label = slot_label
        for lbl in labels:
            if lbl.lower() in slot_label.lower() or slot_label.lower() in lbl.lower():
                matched_label = lbl
                break
        result[matched_label] = _parse_ai_response(block, picks)

    # Fill any missing labels with an empty result
    for lbl in labels:
        if lbl not in result:
            result[lbl] = {"combination": [], "reasoning": raw.strip(), "disclaimer": ""}

    return result


def _parse_ai_response(raw: str, picks: int) -> dict:
    """Extract COMBINATION / REASONING / DISCLAIMER from the model's response."""
    import re

    combination: list[int] = []
    reasoning = ""
    disclaimer = ""

    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("COMBINATION:"):
            nums_txt = line.split(":", 1)[1]
            found = [int(m) for m in re.findall(r"\d+", nums_txt)]
            combination = sorted(found)[:picks]
        elif line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
        elif line.upper().startswith("DISCLAIMER:"):
            disclaimer = line.split(":", 1)[1].strip()

    # Fallback: if model didn't use the expected format, return raw text
    if not combination:
        return {"combination": [], "reasoning": raw.strip(), "disclaimer": ""}
    return {"combination": combination, "reasoning": reasoning, "disclaimer": disclaimer}
