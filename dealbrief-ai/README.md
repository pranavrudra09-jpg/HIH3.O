# DealBrief AI

> **AI Agents That Learn Using Hindsight** — Hack With Hyderabad 3.0

A memory-powered sales intelligence agent that uses **Hindsight by Vectorize** as its persistent memory layer to recall historical prospect interactions and generate personalised sales briefs.

---

## What DealBrief AI Does

A salesperson logs interactions with prospects (meeting notes, objections, budget, requirements, competitors, stakeholders, commitments). Later, when they ask "What should I know before my next call with Rahul from Acme?" — the agent:

1. **Recalls** relevant historical information from Hindsight memory
2. **Constructs** a rich prompt including those memories
3. **Generates** a personalised DealBrief via Groq LLM
4. **Shows** the recalled memories visually so judges can see Hindsight at work

### The Hackathon Demo Contrast

| Without Memory | With Hindsight Memory |
|---|---|
| Generic sales advice | Personalized advice grounded in real history |
| "Address common objections" | "Rahul raised implementation complexity — show the pilot path" |
| No budget awareness | "Budget ceiling is ₹50,000 — stay within it" |
| Generic follow-up | "VP of Operations returns mid-October — follow up then" |

---

## Why Hindsight Is Central

Hindsight is NOT a simple vector store. It is an **agent memory system** that:
- Extracts and structures knowledge from natural-language inputs (`retain()`)
- Uses multi-strategy retrieval (semantic + entity graph + temporal) to find what's relevant (`recall()`)
- Persists memory across conversations and sessions

In DealBrief AI:
- Every logged interaction is stored via `Hindsight.retain()` with structured context and tags
- Every DealBrief request queries `Hindsight.recall()` to retrieve relevant history
- The LLM receives recalled memories as context — **not session history**

This is exactly the hackathon theme: **AI Agents That Learn Using Hindsight**.

---

## Architecture

```
Salesperson (browser)
        │
        ▼
  FastAPI (app.main:app; existing implementation in app.py)
        │
   ┌────┴────┐
   │         │
   ▼         ▼
Hindsight  Groq LLM
  retain()  (llama-3.3-70b)
  recall()
        │
        ▼
   DealBrief + Recalled Memories
        │
        ▼
  Browser (rendered visually)
```

### Memory abstraction

All Hindsight code is isolated in the `MemoryService` class:

```python
class MemoryService:
    def retain_interaction(self, interaction) → dict
    def recall_prospect(self, prospect, company, question, limit) → (list, source)
```

---

## Setup

### 1. Prerequisites

- Python 3.12+ (or 3.14 — tested and working)
- A **Hindsight** account at [hindsight.vectorize.io](https://hindsight.vectorize.io)
- A **Groq** account at [console.groq.com](https://console.groq.com)

### 2. Create virtual environment

```powershell
# Windows — if py launcher works normally:
py -m venv .venv

# Windows — if Python 3.14 installed at C:\Python314 (broken stdlib):
$env:PYTHONHOME = "C:\Users\prana\AppData\Local\Programs\Python\Python314"
C:\Python314\python.exe -m venv .venv
```

### 3. Install dependencies

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Configure environment

```powershell
copy .env.example .env
```

Edit `.env`:

```env
HINDSIGHT_BASE_URL=https://api.hindsight.vectorize.io
HINDSIGHT_API_KEY=your_hindsight_api_key_here
HINDSIGHT_BANK_ID=dealbrief-ai

GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

### 5. Run in development

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --no-access-log
```

Open **http://127.0.0.1:8000**

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `HINDSIGHT_BASE_URL` | Hindsight API base URL | Yes (default provided) |
| `HINDSIGHT_API_KEY` | Your Hindsight API key | Yes (for live mode) |
| `HINDSIGHT_BANK_ID` | Memory bank namespace | No (default: `dealbrief-ai`) |
| `GROQ_API_KEY` | Your Groq API key | Yes (for AI brief) |
| `GROQ_MODEL` | Groq model ID | No (default: `llama-3.3-70b-versatile`) |
| `APP_ENV` | `development` or `production` | No (default: `development`) |
| `ALLOWED_ORIGINS` | Comma-separated frontend origins; no paths or wildcards | Required in production |
| `APP_USERNAME` | HTTP Basic username for the production UI and API | Required in production |
| `APP_PASSWORD` | HTTP Basic password (minimum 16 characters in production) | Required in production |
| `GROQ_TIMEOUT_SECONDS` | Maximum Groq request duration | No (default: 30) |
| `HINDSIGHT_TIMEOUT_SECONDS` | Hindsight client timeout when supported by its installed SDK | No (default: 15) |
| `MAX_REQUEST_BODY_BYTES` | Maximum accepted request body size | No (default: 65536) |

Hindsight remains optional, matching the existing fallback behavior. Without it, interactions are kept in a bounded, in-memory local fallback and are lost when the process restarts. Without Groq, development mode uses the existing deterministic demo brief. Production mode requires `GROQ_API_KEY`, explicit origins, and HTTP Basic credentials. Never use the development defaults for an internet-facing deployment.

---

## API Endpoints

### `GET /api/health`
Returns UI configuration status. It does not expose credentials, provider URLs, or bank identifiers.

```json
{
  "status": "ok",
  "hindsight_configured": true,
  "hindsight_connected": true,
  "hindsight_bank_id": "dealbrief-ai",
  "groq_configured": true,
  "groq_model": "llama-3.3-70b-versatile",
  "mode": "live"
}
```

### `GET /health`
Minimal public liveness endpoint:

```json
{"status":"healthy","service":"dealbrief-ai"}
```

### `POST /api/interactions`
Store a structured sales interaction in Hindsight.

```json
{
  "prospect": "Rahul Sharma",
  "company": "Acme Technologies",
  "date": "2026-09-27",
  "meeting_type": "Follow-up",
  "summary": "Rahul said internal approval is delayed...",
  "requirements": "WhatsApp integration, multilingual support",
  "objections": "Implementation complexity",
  "budget": "₹50,000/year",
  "competitors": "Freshdesk",
  "stakeholders": "VP Operations (final sign-off)",
  "commitments": "Follow up week of October 14th"
}
```

### `POST /api/brief`
Recall memories from Hindsight → generate DealBrief via Groq.

```json
{
  "prospect": "Rahul Sharma",
  "company": "Acme Technologies",
  "question": "What should I know before my next call?"
}
```

Returns: `brief`, `memories`, `memory_count`, `memory_source`, `llm_source`

### `GET /api/memories/{prospect}?company=...`
Retrieve all recalled memories for a prospect (for the memory panel).

### `POST /api/demo/seed`
Seeds 3 realistic interactions for Rahul Sharma / Acme Technologies through Hindsight `retain()`.

---

## How Memory Is Stored

Each interaction is converted to a rich natural-language string:

```
Sales interaction — 2026-09-27
Prospect: Rahul Sharma | Company: Acme Technologies
Meeting type: Follow-up
Summary: Rahul said internal approval is delayed...
Requirements: WhatsApp integration, multilingual support
Objections / concerns: Implementation complexity
Budget: ₹50,000/year
Competitors mentioned: Freshdesk
Stakeholders: VP Operations (final sign-off)
Commitments / next steps: Follow up week of October 14th
```

This is passed to `Hindsight.retain()` with:
- `context`: "B2B sales interaction. Extract and remember: prospect name, company, budget..."
- `tags`: `["prospect:rahul-sharma", "company:acme-technologies", "meeting:follow-up"]`

Hindsight processes this, extracts entities, and stores it in the `dealbrief-ai` memory bank.

## How Memory Is Recalled

When generating a brief, the app calls `Hindsight.recall()` with:
- `query`: "Prospect: Rahul Sharma from Acme Technologies. What should I know before my next call? Include budget, requirements, objections..."
- `tags`: `["prospect:rahul-sharma"]` (scopes recall to this prospect)
- `budget`: `"mid"` (controls recall depth/cost)

Hindsight's multi-strategy retrieval finds all relevant memories and returns ranked results.

## How the LLM Uses Recalled Context

The recalled memories are injected directly into the LLM prompt:

```
RETRIEVED HINDSIGHT MEMORIES:
[Memory 1 — type: experience]
Sales interaction — 2026-09-15
Prospect: Rahul Sharma | Company: Acme Technologies
...

[Memory 2 — type: experience]
...

Generate a concise, actionable DealBrief...
```

The LLM is instructed: **"Use ONLY the retrieved memory below for factual claims. Do NOT invent budget figures, objections, requirements, or history."**

---

## Demo Script (Hackathon)

### Step 1 — Show the app loading in demo mode
Point out the status badge shows "Hindsight not connected · Groq not configured" initially.

### Step 2 — Configure credentials
Show `.env` being edited (don't reveal keys). Restart server. Status badge turns green: "Hindsight ● Groq"

### Step 3 — Seed realistic demo data
Click **⚡ Seed Demo Data** — this stores 3 interactions for Rahul Sharma through Hindsight `retain()`.

### Step 4 — Show memory panel
Click **Recall Memory** — 3 memory cards appear. Point to: "These are not session messages stored in the UI — these are memories retrieved from Hindsight's persistent store."

### Step 5 — Generate DealBrief
Click **Generate DealBrief**. The brief appears with:
- `✦ HINDSIGHT MEMORY` badge
- `◈ GROQ` badge
- Section headers: Executive Summary, What They Care About, Previous Objections, etc.

Point out: **"Every fact in this brief — the ₹50,000 budget, the WhatsApp requirement, the October 14 follow-up date — came from Hindsight memory, not from the LLM's training data."**

### Step 6 — Add a new interaction
Log a new interaction (e.g., Rahul agreed to a pilot). Click Save to Memory. Click Recall Memory again — the new memory card appears. Regenerate the brief — it now includes the new information.

### Key Talking Point
> "This is not conversation history stored in session state. The memories persist across conversations, across days, even if the server restarts. That's the power of Hindsight."

---

## Project Structure

```
dealbrief-ai/
├── app.py              # Existing FastAPI app + MemoryService + Groq logic
├── app/
│   ├── __init__.py
│   └── main.py          # Stable app.main:app ASGI entrypoint
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Ignores .env and .venv
├── Dockerfile
├── .dockerignore
├── README.md           # This file
└── static/
    ├── index.html      # Single-page app
    ├── style.css       # Dark SaaS aesthetic
    └── app.js          # Frontend logic

  ---

  ## Production Deployment

  Use Python 3.12 or newer, install `requirements.txt`, and provide production configuration through the deployment environment or a local `.env` file that is never committed. Set `APP_ENV=production`, `GROQ_API_KEY`, `APP_USERNAME`, `APP_PASSWORD` (at least 16 characters), and `ALLOWED_ORIGINS` to the exact HTTPS frontend origin. Hindsight credentials are optional; set `HINDSIGHT_API_KEY` to enable persistent memory.

  Run the production server without reload:

  ```powershell
  python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --no-access-log
  ```

  Build and run the Docker image from this directory:

  ```powershell
```
  docker run --rm -p 8000:8000 --env-file .env dealbrief-ai:latest
  ```

  The container runs as a non-root user. Terminate TLS at a reverse proxy or managed ingress and keep that proxy in front of the app; HTTP Basic credentials must only travel over HTTPS. `/health` is public and reveals only service status. The app has no user/tenant database or persistent local store, so use an identity-aware gateway and managed persistent storage before exposing it to multiple customers. CORS is an origin policy, not authentication.

  Check the service with `GET http://127.0.0.1:8000/health`. Request bodies are bounded, API traffic is rate-limited per source IP in each worker, and response errors omit provider exception details. Rate limits are process-local; use gateway rate limiting for multi-worker or distributed deployments.

  ### Troubleshooting

  - `Missing required production configuration` means one or more production variables listed above are absent.
  - HTTP 401 indicates missing or incorrect HTTP Basic credentials; configure them at the browser prompt and verify TLS is enabled.
  - HTTP 429 indicates the app or provider rate limit was reached; wait and retry, or configure an upstream gateway limit for scaled deployments.
  - A `demo` memory source means Hindsight is not configured/available; the local fallback is volatile and limited to the current process.
  - A 503 from brief generation means Groq or another external dependency is unavailable; check server logs by request ID without logging credentials or prompts.
  - The startup warning that Hindsight is unavailable can also indicate that `hindsight-client` is not installed or its credentials/client configuration are invalid.

---

*Built for Hack With Hyderabad 3.0 — Theme: AI Agents That Learn Using Hindsight*
