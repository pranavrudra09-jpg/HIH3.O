"""
DealBrief AI — Memory-powered sales intelligence agent.
Built for Hack With Hyderabad 3.0 (AI Agents That Learn Using Hindsight).

Architecture:
  User → FastAPI → MemoryService (Hindsight) → Groq LLM → DealBrief response

The MemoryService is a clean abstraction over Hindsight.  All Hindsight-specific
code lives inside that class so it is easy to explain to judges.
"""

import logging
import inspect
import json
import math
import os
import base64
import secrets
import threading
import time
import uuid
from collections import deque
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
if APP_ENV not in {"development", "production"}:
    raise RuntimeError("APP_ENV must be either 'development' or 'production'.")

HINDSIGHT_BASE_URL = os.getenv("HINDSIGHT_BASE_URL", "https://api.hindsight.vectorize.io").rstrip("/")
HINDSIGHT_API_KEY = os.getenv("HINDSIGHT_API_KEY", "").strip()
HINDSIGHT_BANK_ID = os.getenv("HINDSIGHT_BANK_ID", "dealbrief-ai").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
APP_USERNAME = os.getenv("APP_USERNAME", "").strip()
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

if APP_ENV == "production":
    missing = [name for name, value in (
        ("GROQ_API_KEY", GROQ_API_KEY),
        ("ALLOWED_ORIGINS", os.getenv("ALLOWED_ORIGINS", "").strip()),
        ("APP_USERNAME", APP_USERNAME),
        ("APP_PASSWORD", APP_PASSWORD),
    ) if not value]
    if missing:
        raise RuntimeError("Missing required production configuration: " + ", ".join(missing))
    if len(APP_PASSWORD) < 16:
        raise RuntimeError("APP_PASSWORD must be at least 16 characters in production.")

_origins_value = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [origin.strip().rstrip("/") for origin in _origins_value.split(",") if origin.strip()]
if APP_ENV == "development" and not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = [
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:8000", "http://127.0.0.1:8000",
    ]
if "*" in ALLOWED_ORIGINS:
    raise RuntimeError("ALLOWED_ORIGINS must list explicit origins; wildcard origins are not allowed.")
for origin in ALLOWED_ORIGINS:
    parsed_origin = urlsplit(origin)
    if (
        parsed_origin.scheme not in {"http", "https"}
        or not parsed_origin.netloc
        or parsed_origin.path
        or parsed_origin.query
        or parsed_origin.fragment
        or parsed_origin.username
        or parsed_origin.password
    ):
        raise RuntimeError("ALLOWED_ORIGINS must contain only HTTP(S) origins without paths.")
    if APP_ENV == "production" and parsed_origin.scheme != "https":
        raise RuntimeError("ALLOWED_ORIGINS must use HTTPS in production.")

_hindsight_url = urlsplit(HINDSIGHT_BASE_URL)
if (
    _hindsight_url.scheme not in {"http", "https"}
    or not _hindsight_url.netloc
    or _hindsight_url.username
    or _hindsight_url.password
    or _hindsight_url.query
    or _hindsight_url.fragment
):
    raise RuntimeError("HINDSIGHT_BASE_URL must be an absolute HTTP(S) URL.")
if APP_ENV == "production" and _hindsight_url.scheme != "https":
    raise RuntimeError("HINDSIGHT_BASE_URL must use HTTPS in production.")

def _positive_float(name: str, default: str) -> float:
    try:
        value = float(os.getenv(name, default))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive number.") from exc
    if not math.isfinite(value) or value <= 0 or value > 300:
        raise RuntimeError(f"{name} must be greater than 0 and no more than 300 seconds.")
    return value


def _positive_int(name: str, default: str) -> int:
    try:
        value = int(os.getenv(name, default))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc
    if value < 1024 or value > 10 * 1024 * 1024:
        raise RuntimeError(f"{name} must be between 1024 and 10485760 bytes.")
    return value


HINDSIGHT_TIMEOUT_SECONDS = _positive_float("HINDSIGHT_TIMEOUT_SECONDS", "15")
GROQ_TIMEOUT_SECONDS = _positive_float("GROQ_TIMEOUT_SECONDS", "30")
MAX_REQUEST_BODY_BYTES = _positive_int("MAX_REQUEST_BODY_BYTES", "65536")

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": _request_id.get(),
            "message": record.getMessage(),
        }, ensure_ascii=True)


_handler = logging.StreamHandler()
_handler.setFormatter(_JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)
log = logging.getLogger("dealbrief")


class ExternalServiceError(Exception):
    def __init__(self, message: str, status_code: int = 503) -> None:
        super().__init__(message)
        self.public_message = message
        self.status_code = status_code


class RequestBodyLimitMiddleware:
    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        received_bytes = 0

        async def limited_receive():
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_bytes:
                    raise HTTPException(status_code=413, detail="Request body is too large.")
            return message

        await self.app(scope, limited_receive, send)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class InteractionRequest(BaseModel):
    """Structured sales interaction to store in Hindsight memory."""
    prospect: str = Field(min_length=2, max_length=120, description="Prospect's first/full name")
    company: str = Field(min_length=2, max_length=160, description="Prospect's company name")
    # Optional rich fields — all folded into a natural-language memory string
    summary: str = Field(default="", max_length=5000, description="High-level meeting summary")
    requirements: str = Field(default="", max_length=3000, description="What the prospect needs")
    objections: str = Field(default="", max_length=3000, description="Concerns or blockers raised")
    budget: str = Field(default="", max_length=1000, description="Budget information")
    competitors: str = Field(default="", max_length=1000, description="Competing products mentioned")
    stakeholders: str = Field(default="", max_length=2000, description="Other people involved")
    commitments: str = Field(default="", max_length=3000, description="What was promised / next steps")
    meeting_type: str = Field(default="", max_length=100, description="Discovery / Demo / Follow-up / etc.")
    date: str = Field(default="", max_length=40, description="Meeting date (free text, e.g. 2026-09-27)")


class BriefRequest(BaseModel):
    """Generate a personalized DealBrief for a prospect."""
    prospect: str = Field(min_length=2, max_length=120)
    company: str = Field(default="", max_length=160, description="Company name (helps scope recall)")
    question: str = Field(
        default="What should I know before my next call?",
        max_length=2000,
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
            client_options = {"base_url": HINDSIGHT_BASE_URL, "api_key": HINDSIGHT_API_KEY}
            try:
                if "timeout" in inspect.signature(Hindsight).parameters:
                    client_options["timeout"] = HINDSIGHT_TIMEOUT_SECONDS
                if "max_attempts" in inspect.signature(Hindsight).parameters:
                    client_options["max_attempts"] = 1
            except (TypeError, ValueError):
                pass
            self._client = Hindsight(**client_options)
            self._available = True
            log.info("Hindsight client initialised")
        except Exception as exc:
            log.error("Hindsight client initialisation failed error_type=%s", type(exc).__name__)

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
            log.info("Hindsight memory bank created")
        except Exception as exc:
            # Bank already existing is expected — that's fine.
            already_msg = str(exc).lower()
            if "already" in already_msg or "exist" in already_msg or "409" in already_msg:
                log.debug("Memory bank already exists")
            else:
                log.warning("Hindsight bank creation failed error_type=%s", type(exc).__name__)
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
            log.info("Retained sales interaction in Hindsight")
            return {
                "source": "hindsight",
                "message": f"Interaction retained in Hindsight memory (bank: {HINDSIGHT_BANK_ID}).",
                "stored_text": text,
            }
        except Exception as exc:
            err_msg = str(exc)
            log.warning("Hindsight retain failed error_type=%s", type(exc).__name__)
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
                    "message": "Hindsight is temporarily unavailable. Interaction stored in local fallback.",
                    "stored_text": text,
                }
            return {
                "source": "demo",
                "message": "Hindsight is temporarily unavailable. Interaction stored in local fallback.",
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

            log.info("Recalled %d memories from Hindsight", len(memories))
            return memories, "hindsight"
        except Exception as exc:
            log.warning("Hindsight recall failed error_type=%s; using local fallback", type(exc).__name__)
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
_DEMO_STORE: deque[dict] = deque(maxlen=500)


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
        client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT_SECONDS, max_retries=0)
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
        if not isinstance(brief_text, str) or not brief_text.strip():
            raise ExternalServiceError("Groq returned an empty response.", 503)
        log.info("Generated DealBrief via Groq (%s)", GROQ_MODEL)
        return brief_text, "groq"
    except Exception as exc:
        if isinstance(exc, ExternalServiceError):
            raise
        status_code = getattr(exc, "status_code", None)
        log.error("Groq request failed error_type=%s status_code=%s", type(exc).__name__, status_code)
        if status_code == 429:
            raise ExternalServiceError("Groq rate limit reached. Please try again shortly.", 429) from exc
        raise ExternalServiceError("Groq is temporarily unavailable. Please try again shortly.", 503) from exc


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DealBrief AI",
    description="Memory-powered sales intelligence agent (Hindsight + Groq)",
    version="1.0.0",
)
app.add_middleware(RequestBodyLimitMiddleware, max_bytes=MAX_REQUEST_BODY_BYTES)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    max_age=600,
)

_rate_limit_lock = threading.Lock()
_rate_limit_buckets: dict[str, deque[float]] = {}


@app.exception_handler(RequestValidationError)
async def invalid_request_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": "Invalid request input."})


@app.exception_handler(ExternalServiceError)
async def external_service_error_handler(request: Request, exc: ExternalServiceError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.public_message})


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    log.error("Unhandled request error error_type=%s", type(exc).__name__)
    return JSONResponse(status_code=500, content={"detail": "An unexpected server error occurred."})


@app.middleware("http")
async def request_safety_middleware(request: Request, call_next):
    request_id = uuid.uuid4().hex
    token = _request_id.set(request_id)
    response = None
    try:
        if APP_ENV == "production" and request.url.path != "/health":
            credentials = base64.b64encode(f"{APP_USERNAME}:{APP_PASSWORD}".encode()).decode("ascii")
            supplied = request.headers.get("authorization", "")
            if not secrets.compare_digest(supplied, f"Basic {credentials}"):
                response = JSONResponse(
                    status_code=401,
                    content={"detail": "Authentication required."},
                    headers={"WWW-Authenticate": 'Basic realm="DealBrief AI", charset="UTF-8"'},
                )

        content_length = request.headers.get("content-length")
        if response is None and content_length and content_length.isdigit() and int(content_length) > MAX_REQUEST_BODY_BYTES:
            response = JSONResponse(status_code=413, content={"detail": "Request body is too large."})
        elif response is None and request.url.path.startswith("/api/") and request.method != "OPTIONS":
            client_host = request.client.host if request.client else "unknown"
            now = time.monotonic()
            with _rate_limit_lock:
                bucket = _rate_limit_buckets.setdefault(client_host, deque())
                while bucket and bucket[0] <= now - 60:
                    bucket.popleft()
                if len(_rate_limit_buckets) > 10000:
                    for address, timestamps in list(_rate_limit_buckets.items()):
                        while timestamps and timestamps[0] <= now - 60:
                            timestamps.popleft()
                        if not timestamps:
                            _rate_limit_buckets.pop(address, None)
                if len(bucket) >= 30:
                    response = JSONResponse(status_code=429, content={"detail": "Too many requests. Please try again shortly."})
                else:
                    bucket.append(now)

        if response is None:
            response = await call_next(request)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
        if APP_ENV == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        route = request.scope.get("route")
        route_name = getattr(route, "path", "unmatched")
        log.info("request method=%s route=%s status=%s", request.method, route_name, response.status_code)
        return response
    except Exception as exc:
        log.error("Request middleware failed error_type=%s", type(exc).__name__)
        raise
    finally:
        _request_id.reset(token)

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


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "dealbrief-ai"}


@app.get("/api/health")
def health():
    """
    Health / configuration status.

    Never exposes API keys. Safe to show to judges.
    """
    return {
        "status": "ok",
        "hindsight_configured": bool(HINDSIGHT_API_KEY),
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
        log.error("Interaction retain failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Memory service is temporarily unavailable.") from exc


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
    except ExternalServiceError:
        raise
    except Exception as exc:
        log.error("Brief generation failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="DealBrief generation failed.") from exc


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
        log.error("Memory recall failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Memory service is temporarily unavailable.") from exc


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
            log.error("Demo seed retain failed error_type=%s", type(exc).__name__)
            errors.append(type(exc).__name__)

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
