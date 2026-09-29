"""
DealBrief AI — Memory-powered sales intelligence agent.
Built for Hack With Hyderabad 3.0 (AI Agents That Learn Using Hindsight).

Architecture:
  User → FastAPI → MemoryService (Hindsight) → Groq LLM → DealBrief response

The MemoryService is a clean abstraction over Hindsight.  All Hindsight-specific
code lives inside that class so it is easy to explain to judges.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("dealbrief")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Environment variables — never hardcoded
HINDSIGHT_BASE_URL: str = os.getenv("HINDSIGHT_BASE_URL", "https://api.hindsight.vectorize.io")
HINDSIGHT_API_KEY: str = os.getenv("HINDSIGHT_API_KEY", "")
HINDSIGHT_BANK_ID: str = os.getenv("HINDSIGHT_BANK_ID", "dealbrief-ai")

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
# Default to a reliable Groq production model.  Override in .env if needed.
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class InteractionRequest(BaseModel):
    """Structured sales interaction to store in Hindsight memory."""
    prospect: str = Field(min_length=2, description="Prospect's first/full name")
    company: str = Field(min_length=2, description="Prospect's company name")
    # Optional rich fields — all folded into a natural-language memory string
    summary: str = Field(default="", description="High-level meeting summary")
    requirements: str = Field(default="", description="What the prospect needs")
    objections: str = Field(default="", description="Concerns or blockers raised")
    budget: str = Field(default="", description="Budget information")
    competitors: str = Field(default="", description="Competing products mentioned")
    stakeholders: str = Field(default="", description="Other people involved")
    commitments: str = Field(default="", description="What was promised / next steps")
    meeting_type: str = Field(default="", description="Discovery / Demo / Follow-up / etc.")
    date: str = Field(default="", description="Meeting date (free text, e.g. 2026-09-27)")


class BriefRequest(BaseModel):
    """Generate a personalized DealBrief for a prospect."""
    prospect: str = Field(min_length=2)
    company: str = Field(default="", description="Company name (helps scope recall)")
    question: str = Field(
        default="What should I know before my next call?",
        description="What the salesperson wants to know",
    )


# ---------------------------------------------------------------------------
# MemoryService — thin, clean abstraction over Hindsight
# ---------------------------------------------------------------------------

class MemoryService:
    """
    Isolates all Hindsight-specific code in one place.

    Public methods:
        retain_interaction(interaction)  → stores a sales interaction
        recall_prospect(prospect, company, question, limit) → retrieves memories
    """

    def __init__(self) -> None:
        self._client = None
        self._bank_ensured = False
        self._available = False

        if not HINDSIGHT_API_KEY:
            log.warning("HINDSIGHT_API_KEY not set — running without Hindsight memory.")
            return

        try:
            from hindsight_client import Hindsight  # type: ignore
            self._client = Hindsight(
                base_url=HINDSIGHT_BASE_URL,
                api_key=HINDSIGHT_API_KEY,
            )
            self._available = True
            log.info("Hindsight client initialised (bank: %s)", HINDSIGHT_BANK_ID)
        except Exception as exc:
            log.error("Failed to initialise Hindsight client: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_bank(self) -> None:
        """Create the memory bank if it does not exist yet (idempotent)."""
        if self._bank_ensured:
            return
        try:
            self._client.create_bank(
                bank_id=HINDSIGHT_BANK_ID,
                name="DealBrief AI — Sales Intelligence",
                mission=(
                    "Retain detailed sales interaction history for B2B prospects. "
                    "Capture budget, objections, requirements, competitors, stakeholders, "
                    "commitments and recommended next actions so sales reps can recall "
                    "everything relevant before their next call."
                ),
            )
            log.info("Memory bank created: %s", HINDSIGHT_BANK_ID)
        except Exception as exc:
            # Bank already existing is expected — that's fine.
            already_msg = str(exc).lower()
            if "already" in already_msg or "exist" in already_msg or "409" in already_msg:
                log.debug("Memory bank already exists: %s", HINDSIGHT_BANK_ID)
            else:
                log.warning("create_bank warning (non-fatal): %s", exc)
        self._bank_ensured = True

    @staticmethod
    def _build_memory_text(interaction: InteractionRequest) -> str:
        """
        Convert a structured InteractionRequest into a rich natural-language
        string for Hindsight to process and remember.
        """
        date_str = interaction.date or datetime.now().strftime("%Y-%m-%d")
        parts = [
            f"Sales interaction — {date_str}",
            f"Prospect: {interaction.prospect} | Company: {interaction.company}",
        ]
        if interaction.meeting_type:
            parts.append(f"Meeting type: {interaction.meeting_type}")
        if interaction.summary:
            parts.append(f"Summary: {interaction.summary}")
        if interaction.requirements:
            parts.append(f"Requirements: {interaction.requirements}")
        if interaction.objections:
            parts.append(f"Objections / concerns: {interaction.objections}")
        if interaction.budget:
            parts.append(f"Budget: {interaction.budget}")
        if interaction.competitors:
            parts.append(f"Competitors mentioned: {interaction.competitors}")
        if interaction.stakeholders:
            parts.append(f"Stakeholders: {interaction.stakeholders}")
        if interaction.commitments:
            parts.append(f"Commitments / next steps: {interaction.commitments}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._available

    def retain_interaction(self, interaction: InteractionRequest) -> dict:
        """
        Persist a sales interaction to Hindsight.

        Returns a dict with keys: source, message, stored_text
        """
        text = self._build_memory_text(interaction)

        if not self._available:
            # Fall back to in-memory demo store (never claim Hindsight was used)
            _DEMO_STORE.append({
                "prospect": interaction.prospect.lower(),
                "company": interaction.company.lower(),
                "text": text,
                "ts": datetime.now().isoformat(timespec="seconds"),
            })
            return {
                "source": "demo",
                "message": "Demo mode — Hindsight not connected. Interaction stored locally only.",
                "stored_text": text,
            }

        try:
            self._ensure_bank()
            tags = [
                f"prospect:{interaction.prospect.lower().replace(' ', '-')}",
                f"company:{interaction.company.lower().replace(' ', '-')}",
            ]
            if interaction.meeting_type:
                tags.append(f"meeting:{interaction.meeting_type.lower()}")

            self._client.retain(
                bank_id=HINDSIGHT_BANK_ID,
                content=text,
                context=(
                    "B2B sales interaction. Extract and remember: prospect name, company, "
                    "budget figures, product requirements, objections/blockers, competitor names, "
                    "stakeholder names, commitments made, and recommended next steps."
                ),
                tags=tags,
            )
            log.info("Retained interaction for %s / %s in Hindsight", interaction.prospect, interaction.company)
            return {
                "source": "hindsight",
                "message": f"Interaction retained in Hindsight memory (bank: {HINDSIGHT_BANK_ID}).",
                "stored_text": text,
            }
        except Exception as exc:
            err_msg = str(exc)
            log.warning("Hindsight retain error: %s", err_msg)
            # Store locally so user can still test LLM generation, but explicitly mark source as demo/fallback
            _DEMO_STORE.append({
                "prospect": interaction.prospect.lower(),
                "company": interaction.company.lower(),
                "text": text,
                "ts": datetime.now().isoformat(timespec="seconds"),
            })
            if "402" in err_msg or "credit" in err_msg.lower():
                return {
                    "source": "demo",
                    "message": "Hindsight API: Insufficient credits (402 Payment Required). Stored in local fallback so Groq DealBrief generation can proceed.",
                    "stored_text": text,
                }
            return {
                "source": "demo",
                "message": f"Hindsight error ({err_msg}). Stored in local fallback.",
                "stored_text": text,
            }

    def recall_prospect(
        self,
        prospect: str,
        company: str = "",
        question: str = "Summarise all relevant sales history.",
        limit: int = 8,
    ) -> tuple[list[dict], str]:
        """
        Retrieve relevant memories for a prospect from Hindsight.

        Returns (memories_list, source) where source is 'hindsight' or 'demo'.
        Each memory dict has: text, type, score, context, tags
        """
        company_fragment = f" from {company}" if company else ""
        query = (
            f"Prospect: {prospect}{company_fragment}. "
            f"{question} "
            "Include budget, requirements, objections, competitors, stakeholders, "
            "commitments, follow-up dates, and recommended next actions."
        )

        if not self._available:
            # Demo fallback — search in-memory store
            results = []
            for item in _DEMO_STORE:
                if prospect.lower() in item["prospect"]:
                    results.append({
                        "text": item["text"],
                        "type": "experience",
                        "score": 1.0,
                        "context": "Locally stored demo memory",
                        "tags": [],
                    })
            return results, "demo"

        try:
            self._ensure_bank()
            prospect_tag = f"prospect:{prospect.lower().replace(' ', '-')}"
            recall_result = self._client.recall(
                bank_id=HINDSIGHT_BANK_ID,
                query=query,
                max_tokens=4096,
                budget="mid",
                tags=[prospect_tag] if prospect_tag else None,
            )

            memories = []
            for item in recall_result.results[:limit]:
                score_val = None
                if item.scores:
                    if hasattr(item.scores, "final") and item.scores.final is not None:
                        score_val = item.scores.final
                    else:
                        scores_dict = item.scores.to_dict() if hasattr(item.scores, "to_dict") else {}
                        score_val = max((v for v in scores_dict.values() if isinstance(v, (int, float))), default=None)

                memories.append({
                    "text": item.text,
                    "type": item.type or "memory",
                    "score": round(score_val, 3) if score_val is not None else None,
                    "context": item.context or "",
                    "tags": item.tags or [],
                })

            log.info(
                "Recalled %d memories for %s from Hindsight", len(memories), prospect
            )
            return memories, "hindsight"
        except Exception as exc:
            log.warning("Hindsight recall error (%s), using local fallback store", exc)
            results = []
            for item in _DEMO_STORE:
                if prospect.lower() in item["prospect"]:
                    results.append({
                        "text": item["text"],
                        "type": "experience",
                        "score": 1.0,
                        "context": f"Local memory fallback (Hindsight credit limit reached)",
                        "tags": [],
                    })
            return results, "demo"

    def recall_context(
        self,
        query: str,
        limit: int = 8,
    ) -> tuple[list[dict], str]:
        """
        Recall memories matching a general context query from Hindsight.

        Returns (memories_list, source).
        """
        if not self._available:
            results = []
            for item in _DEMO_STORE:
                if any(w in item["text"].lower() for w in query.lower().split()):
                    results.append({
                        "text": item["text"],
                        "type": "experience",
                        "score": 1.0,
                        "context": "Locally stored demo memory",
                        "tags": [],
                    })
            return results[:limit], "demo"

        self._ensure_bank()
        recall_result = self._client.recall(
            bank_id=HINDSIGHT_BANK_ID,
            query=query,
            max_tokens=6000,
            budget="mid",
        )
        memories = []
        for item in recall_result.results[:limit]:
            score_val = None
            if item.scores:
                scores_dict = item.scores.to_dict() if hasattr(item.scores, "to_dict") else {}
                score_val = max((v for v in scores_dict.values() if isinstance(v, (int, float))), default=None)
            memories.append({
                "text": item.text,
                "type": item.type or "memory",
                "score": round(score_val, 3) if score_val is not None else None,
                "context": item.context or "",
                "tags": item.tags or [],
            })
        return memories, "hindsight"


# ---------------------------------------------------------------------------
# Demo memory fallback (only used when Hindsight is not configured)
# ---------------------------------------------------------------------------
_DEMO_STORE: list[dict] = []


# ---------------------------------------------------------------------------
# LLM — Groq
# ---------------------------------------------------------------------------

def generate_brief(prospect: str, company: str, question: str, memories: list[dict]) -> tuple[str, str]:
    """
    Call Groq with recalled Hindsight memories as context to generate a DealBrief.

    Returns (brief_text, llm_source).
    """
    if memories:
        memory_context = "\n\n".join(
            f"[Memory {i+1} — type: {m.get('type','?')}]\n{m['text']}"
            for i, m in enumerate(memories)
        )
    else:
        memory_context = "No prior memory found for this prospect."

    company_fragment = f" at {company}" if company else ""
    system_prompt = (
        "You are DealBrief AI, a sales intelligence assistant with persistent memory. "
        "Your superpower is recalling previous interactions with prospects. "
        "Use ONLY the retrieved memory below for factual claims about this prospect. "
        "Do NOT invent budget figures, objections, requirements, or history. "
        "If a section lacks information, say so explicitly."
    )

    user_prompt = f"""You have been asked about:
PROSPECT: {prospect}{company_fragment}
QUESTION: {question}

RETRIEVED HINDSIGHT MEMORIES:
{memory_context}

Generate a concise, actionable DealBrief with exactly these sections:

## Executive Summary
One paragraph overview of where the deal stands based on memory.

## What They Care About
Key requirements, priorities, and motivations recalled from memory.

## Previous Objections
Blockers and concerns the prospect has raised (from memory).

## Buying Signals
Positive indicators and expressed interest (from memory).

## Risks
Deal risks identified from history.

## Recommended Approach
How to handle the next interaction, grounded in recalled history.

## Suggested Talking Points
3-5 specific, personalized talking points referencing what the prospect said before.

## Next Best Action
One clear, specific action to advance this deal.

Base every claim on the recalled memories. Flag anything you are inferring rather than recalling."""

    if not GROQ_API_KEY:
        # Demo fallback — deterministic brief so the UI is still usable
        if memories:
            first = memories[0]["text"]
            return (
                "## Executive Summary\n"
                f"{first[:400]}...\n\n"
                "## What They Care About\n"
                "See recalled memories above — requirements are captured there.\n\n"
                "## Previous Objections\n"
                "Review the objections section of the recalled memories.\n\n"
                "## Buying Signals\n"
                "Prospect has engaged and provided detailed context — a strong signal.\n\n"
                "## Risks\n"
                "Review blockers captured in recalled memories.\n\n"
                "## Recommended Approach\n"
                "Address the recalled objections directly before advancing.\n\n"
                "## Suggested Talking Points\n"
                "- Reference the specific requirement the prospect mentioned\n"
                "- Acknowledge the budget ceiling and show ROI within it\n"
                "- Propose a concrete timeline for the next step\n\n"
                "## Next Best Action\n"
                "Send a follow-up referencing the specific concerns and commitments from "
                "the last interaction.\n\n"
                "---\n_[Demo mode — add GROQ_API_KEY for AI-generated brief]_"
            ), "demo"
        return (
            "No memory found for this prospect yet.\n\n"
            "Log an interaction first using the panel on the left, then generate a brief.\n\n"
            "---\n_[Demo mode — Hindsight and/or Groq not configured]_"
        ), "demo"

    try:
        from groq import Groq  # type: ignore
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.2,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        brief_text = response.choices[0].message.content
        log.info("Generated DealBrief via Groq (%s)", GROQ_MODEL)
        return brief_text, "groq"
    except Exception as exc:
        log.error("Groq error: %s", exc)
        raise RuntimeError(f"Groq API error: {exc}") from exc


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DealBrief AI",
    description="Memory-powered sales intelligence agent (Hindsight + Groq)",
    version="1.0.0",
)

# Serve static files at /static/*
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Singleton memory service
_memory = MemoryService()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def home():
    """Serve the single-page application."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    """
    Health / configuration status.

    Never exposes API keys. Safe to show to judges.
    """
    return {
        "status": "ok",
        "hindsight_configured": bool(HINDSIGHT_API_KEY),
        "hindsight_base_url": HINDSIGHT_BASE_URL if HINDSIGHT_API_KEY else None,
        "hindsight_bank_id": HINDSIGHT_BANK_ID,
        "hindsight_connected": _memory.available,
        "groq_configured": bool(GROQ_API_KEY),
        "groq_model": GROQ_MODEL if GROQ_API_KEY else None,
        "mode": "live" if (_memory.available and GROQ_API_KEY) else "demo",
    }


@app.post("/api/interactions")
def add_interaction(interaction: InteractionRequest):
    """
    Store a structured sales interaction in Hindsight memory.

    Step 1 of the Hindsight demo: RETAIN.
    """
    try:
        result = _memory.retain_interaction(interaction)
        return {
            "ok": True,
            **result,
            "prospect": interaction.prospect,
            "company": interaction.company,
        }
    except Exception as exc:
        log.error("retain error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Memory retain error: {exc}")


@app.post("/api/brief")
def generate_deal_brief(request: BriefRequest):
    """
    Recall prospect memory from Hindsight, then generate a personalised DealBrief via Groq.

    Step 2 of the Hindsight demo: RECALL → LLM → Brief.
    """
    try:
        memories, memory_source = _memory.recall_prospect(
            prospect=request.prospect,
            company=request.company,
            question=request.question,
        )
        brief, llm_source = generate_brief(
            prospect=request.prospect,
            company=request.company,
            question=request.question,
            memories=memories,
        )
        return {
            "prospect": request.prospect,
            "company": request.company,
            "question": request.question,
            "brief": brief,
            "memories": memories,
            "memory_count": len(memories),
            "memory_source": memory_source,
            "llm_source": llm_source,
            "llm_model": GROQ_MODEL if llm_source == "groq" else None,
            "hindsight_bank": HINDSIGHT_BANK_ID,
        }
    except Exception as exc:
        log.error("brief error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Agent error: {exc}")


@app.get("/api/memories/{prospect}")
def get_memories(prospect: str, company: str = ""):
    """
    Retrieve all recalled memories for a prospect.

    Useful for the memory panel and for debugging.
    """
    try:
        memories, source = _memory.recall_prospect(
            prospect=prospect,
            company=company,
            question="Recall everything about this prospect: full sales history, all interactions, objections, requirements, budget, commitments, stakeholders, and recommended next steps.",
            limit=12,
        )
        return {
            "prospect": prospect,
            "company": company,
            "source": source,
            "memory_count": len(memories),
            "memories": memories,
            "hindsight_bank": HINDSIGHT_BANK_ID,
        }
    except Exception as exc:
        log.error("memory recall error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Memory error: {exc}")


@app.post("/api/demo/seed")
def seed_demo():
    """
    Seed 3 realistic demo interactions for Rahul Sharma / Acme Technologies.

    These are stored through Hindsight retain() — not hardcoded as the answer.
    Demonstrates that memory actually accumulates across multiple interactions.
    """
    interactions = [
        InteractionRequest(
            prospect="Rahul Sharma",
            company="Acme Technologies",
            date="2026-09-15",
            meeting_type="Discovery",
            summary=(
                "Initial discovery call with Rahul, Head of Operations at Acme Technologies. "
                "He is evaluating automation platforms to streamline their customer support workflow."
            ),
            requirements=(
                "Needs WhatsApp integration as a primary channel. "
                "Must support multi-language conversations — specifically Hindi and Telugu. "
                "Requires easy onboarding with no heavy migration effort."
            ),
            objections=(
                "Main concern is implementation complexity — worried the rollout will take too long "
                "and disrupt existing support processes."
            ),
            budget="Annual budget approximately ₹50,000. No flexibility beyond this ceiling.",
            competitors="Comparing with Freshdesk and a local vendor he declined to name.",
            stakeholders=(
                "Rahul is the champion. His manager, the VP of Operations (name not shared), "
                "has final sign-off authority."
            ),
            commitments=(
                "Sales rep promised to send a detailed implementation plan and timeline within 3 days. "
                "Follow-up call scheduled for next Friday."
            ),
        ),
        InteractionRequest(
            prospect="Rahul Sharma",
            company="Acme Technologies",
            date="2026-09-22",
            meeting_type="Demo",
            summary=(
                "Product demo call. Rahul brought in two colleagues from the tech team. "
                "Good engagement — they asked detailed questions about the WhatsApp connector "
                "and the language model's accuracy on regional languages."
            ),
            requirements=(
                "Confirmed WhatsApp is non-negotiable. "
                "They want the AI to handle conversations in Hindi, Telugu, and English — "
                "sometimes switching mid-conversation."
            ),
            objections=(
                "Tech team raised concern about data residency — where is customer conversation data stored? "
                "Rahul asked whether the system can be trialled without full migration first."
            ),
            budget=(
                "Budget confirmed at ₹50,000/year. Rahul mentioned he might be able to push for ₹60,000 "
                "if the ROI case is strong — but this is not confirmed."
            ),
            competitors=(
                "Freshdesk's WhatsApp integration was directly compared during the demo. "
                "Rahul noted our multi-language support is better, but Freshdesk has a simpler pricing model."
            ),
            stakeholders=(
                "Two tech team members present: Arun (backend) and Priya (integrations). "
                "Priya seems to be the technical decision-influencer."
            ),
            commitments=(
                "Sales rep committed to send a data residency FAQ and a pilot proposal by Wednesday. "
                "Rahul said he will share the proposal internally before next call."
            ),
        ),
        InteractionRequest(
            prospect="Rahul Sharma",
            company="Acme Technologies",
            date="2026-09-27",
            meeting_type="Follow-up",
            summary=(
                "Short follow-up call. Rahul was positive but said internal approval is taking longer "
                "than expected. He personally wants to move forward."
            ),
            requirements=(
                "No new requirements raised. Confirmed the same: WhatsApp + multilingual + simple migration."
            ),
            objections=(
                "Internal budget approval is blocked — VP of Operations is travelling until mid-October. "
                "Rahul cannot commit without that sign-off."
            ),
            budget="No change — ₹50,000 ceiling, possible ₹60,000 if strong ROI case.",
            competitors=(
                "Rahul mentioned the local vendor has submitted a lower quote. "
                "He did not share the figure but implied it is meaningfully cheaper."
            ),
            stakeholders=(
                "VP of Operations is the gatekeeper — returning mid-October. "
                "Rahul suggested we address the ROI email directly to the VP."
            ),
            commitments=(
                "Agreed to follow up the week of October 14th when VP returns. "
                "Sales rep to prepare a ROI summary email addressed to the VP. "
                "Rahul will introduce via email once we send the draft."
            ),
        ),
    ]

    results = []
    errors = []
    for interaction in interactions:
        try:
            result = _memory.retain_interaction(interaction)
            results.append({
                "prospect": interaction.prospect,
                "date": interaction.date,
                "meeting_type": interaction.meeting_type,
                "source": result["source"],
            })
        except Exception as exc:
            errors.append(str(exc))

    return {
        "ok": len(errors) == 0,
        "retained": len(results),
        "interactions": results,
        "errors": errors,
        "message": (
            f"Seeded {len(results)} interactions for Rahul Sharma / Acme Technologies. "
            "Now click 'Recall Memory' or generate a DealBrief to see Hindsight in action."
        ),
    }
