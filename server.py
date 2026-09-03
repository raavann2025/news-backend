import asyncio
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from emergentintegrations.llm.chat import (LlmChat, StreamDone, TextDelta,
                                           UserMessage)
from emergentintegrations.llm.openai import (OpenAISpeechToText,
                                             OpenAITextToSpeech)
from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.cors import CORSMiddleware
from tavily import TavilyClient

from newsroom import PERSONAS as _NR_PERSONAS
from newsroom import run_newsroom as _run_newsroom
from research import (ResearchBundle, deep_research,
                      generate_insights_from_bundle, plan_rails)

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# Env
MONGO_URL = os.environ["MONGO_URL"]
DB_NAME = os.environ["DB_NAME"]
EMERGENT_LLM_KEY = os.environ["EMERGENT_LLM_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

# DB
client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

# App
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Tavily
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# OpenAI TTS + STT via Emergent key
tts_client = OpenAITextToSpeech(api_key=EMERGENT_LLM_KEY)
stt_client = OpenAISpeechToText(api_key=EMERGENT_LLM_KEY)


# ============ LLM model — OpenAI GPT-5.4 across the app ============
# Users don't pick — the product is opinionated. Latest & best chat model.
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-5.6-luna"


# ============ Models ============


class Insight(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    text: str
    headline: str = ""
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    rail: Optional[str] = None  # persona rail label (e.g., "Gossip Girl")
    interest: Optional[str] = None  # the single user interest this card is for
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class GenerateRequest(BaseModel):
    topic: str


class GenerateResponse(BaseModel):
    session_id: str
    topic: str
    insights: List[Insight]


class SaveInsightRequest(BaseModel):
    insight: Insight


class ChatRequest(BaseModel):
    session_id: str
    insight_id: str
    insight_text: str
    insight_headline: str = ""
    insight_topic: str = ""
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    history: List[dict] = Field(default_factory=list)  # [{role, content}]
    message: str


class RegenerateRequest(BaseModel):
    topic: str
    insight_id: str
    previous_text: str
    exclude: List[str] = Field(default_factory=list)


class MoreInsightsRequest(BaseModel):
    session_id: str
    topic: str
    previous_headlines: List[str] = Field(default_factory=list)
    steer: Optional[str] = None  # optional voice-steering instruction


class DirectionsRequest(BaseModel):
    topic: str
    interests: List[str] = Field(default_factory=list)
    is_news: bool = False


class DirectionOption(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    subtitle: str


class DirectionsResponse(BaseModel):
    topic: str
    directions: List[DirectionOption]


class RewriteRequest(BaseModel):
    session_id: str
    topic: str
    insight_id: str
    insight_headline: str = ""
    insight_text: str
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    transcript: str  # user's spoken request
    previous_headlines: List[str] = Field(default_factory=list)


class SourceInsight(BaseModel):
    """Compact representation of an insight used as context for a branch call."""

    id: Optional[str] = None
    topic: str = ""
    headline: str = ""
    text: str
    source_url: Optional[str] = None
    source_title: Optional[str] = None


class DeeperRequest(BaseModel):
    session_id: str
    topic: str
    source_insight: SourceInsight
    previous_headlines: List[str] = Field(default_factory=list)


class ExplainRequest(BaseModel):
    session_id: str
    topic: str
    source_insight: SourceInsight
    phrase: str  # the word or word-block user tapped


class QuestionsRequest(BaseModel):
    topic: str
    source_insight: SourceInsight


class QuestionsResponse(BaseModel):
    questions: List[str]


class AnswerRequest(BaseModel):
    session_id: str
    topic: str
    source_insight: SourceInsight
    question: str
    previous_headlines: List[str] = Field(default_factory=list)


class CommandRequest(BaseModel):
    """Unified command endpoint — accepts either voice transcript OR chat message. Backend
    classifies intent and streams back an appropriate response type."""

    session_id: str
    topic: str
    current_insight: SourceInsight  # the insight the user was viewing
    insights: List[SourceInsight] = Field(
        default_factory=list
    )  # full current visible deck
    message: str  # user's request (from mic or chat)
    channel: str = "text"  # "voice" | "text"
    history: List[dict] = Field(default_factory=list)


class BehaviorEvent(BaseModel):
    session_id: str
    insight_id: str
    topic: Optional[str] = None
    headline: Optional[str] = None
    event: str
    dwell_ms: Optional[int] = None


class TTSRequest(BaseModel):
    text: str
    voice: str = "sage"


class SavedInsight(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    topic: str
    text: str
    headline: str = ""
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    saved_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ============ Helpers ============

VAGUE_TOPICS = {"something", "anything", "stuff", "things", "idk", "nothing"}


def is_vague(topic: str) -> bool:
    t = topic.strip().lower()
    if len(t) < 4:
        return True
    if t in VAGUE_TOPICS:
        return True
    if len(t.split()) < 2 and t in VAGUE_TOPICS:
        return True
    return False


def legacy_context(results: List[dict], answer: str) -> str:
    parts = []
    if answer:
        parts.append(f"SEARCH SUMMARY:\n{answer}\n")
    parts.append("SOURCES:")
    for i, r in enumerate(results[:8], 1):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")[:800]
        parts.append(f"[{i}] {title}\n   URL: {url}\n   {content}\n")
    return "\n".join(parts)


async def build_behavior_context(session_id: str) -> str:
    """Turn user's behavior signals for this session into a natural-language bias hint."""
    if not session_id:
        return ""
    docs = await db.behaviors.find({"session_id": session_id}, {"_id": 0}).to_list(500)
    if not docs:
        return ""

    liked_topics, skipped_topics, asked_topics = [], [], []
    for d in docs:
        headline = (d.get("headline") or "").strip()
        if not headline:
            continue
        ev = d.get("event")
        if ev in ("save", "ask", "share"):
            liked_topics.append(headline)
        elif ev == "skip":
            skipped_topics.append(headline)
        if ev == "ask":
            asked_topics.append(headline)

    lines = []
    if liked_topics:
        lines.append(
            "The user ENGAGED with these insights (saved / asked follow-up / shared) — emphasize similar angles:"
        )
        for h in liked_topics[-8:]:
            lines.append(f"  ✓ {h}")
    if skipped_topics:
        lines.append(
            "The user SKIPPED these insights quickly (dwell < 4s) — de-emphasize this style/angle:"
        )
        for h in skipped_topics[-8:]:
            lines.append(f"  ✗ {h}")
    return "\n".join(lines)


def to_insights(
    items: List[dict],
    topic: str,
    sources: List[dict],
    preserve_id: Optional[str] = None,
) -> List[Insight]:
    """Map LLM output items to Insight model, resolving source_index → source_url/title.

    If an item already carries pre-resolved `_source_url` / `_source_title`
    (used by the multi-rail path where each rail resolves its own sources
    before the round-robin merge), those take precedence over source_index.
    """
    out: List[Insight] = []
    for i, item in enumerate(items):
        # Prefer pre-resolved fields when present (multi-rail path).
        src_url = item.get("_source_url")
        src_title = item.get("_source_title")
        if not src_url:
            idx = item.get("source_index")
            if isinstance(idx, int) and 1 <= idx <= len(sources):
                src = sources[idx - 1]
                src_url = src.get("url")
                src_title = src.get("title")
        kwargs = dict(
            topic=topic,
            headline=(item.get("headline") or "").strip(),
            text=(item.get("text") or "").strip(),
            source_url=src_url,
            source_title=src_title,
            rail=item.get("_rail"),
            interest=item.get("_interest"),
        )
        if preserve_id and i == 0:
            kwargs["id"] = preserve_id
        out.append(Insight(**kwargs))
    return out


# ============ Routes ============


@api_router.get("/")
async def root():
    return {"service": "Insight Swipe API", "status": "ok"}


@api_router.post("/insights/generate", response_model=GenerateResponse)
async def generate_insights(req: GenerateRequest):
    topic = req.topic.strip()
    if is_vague(topic):
        raise HTTPException(
            status_code=422,
            detail="Topic is too vague. Try something more specific — e.g. 'how Amazon became successful' or 'the semiconductor industry evolution'.",
        )

    # Deep research: multi-angle Tavily searches + extract + YouTube transcripts
    try:
        bundle = await deep_research(
            topic=topic,
            tavily_client=tavily,
            llm_key=EMERGENT_LLM_KEY,
            extract_top_n=4,
            youtube_top_n=2,
        )
    except Exception:
        logging.exception("deep_research failed")
        raise HTTPException(status_code=500, detail="Research pipeline failed")

    if not bundle.sources:
        raise HTTPException(status_code=500, detail="No sources found for topic")

    try:
        items = await generate_insights_from_bundle(
            bundle=bundle, llm_key=EMERGENT_LLM_KEY, count=12
        )
    except Exception:
        logging.exception("Insight generation failed")
        raise HTTPException(status_code=500, detail="AI response parsing failed")

    if not items:
        raise HTTPException(status_code=500, detail="Could not generate insights")

    insights = to_insights(items, topic, bundle.sources)
    session_id = str(uuid.uuid4())

    await db.sessions.insert_one(
        {
            "session_id": session_id,
            "topic": topic,
            "angles": bundle.angles,
            "sources_used": [s.get("url") for s in bundle.sources[:15]],
            "youtube_sources": [y["url"] for y in bundle.youtube],
            "insights": [i.model_dump() for i in insights],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    return GenerateResponse(session_id=session_id, topic=topic, insights=insights)


@api_router.post("/insights/save", response_model=SavedInsight)
async def save_insight(req: SaveInsightRequest):
    ins = req.insight
    saved = SavedInsight(
        id=ins.id,
        topic=ins.topic,
        headline=ins.headline,
        text=ins.text,
        source_url=ins.source_url,
        source_title=ins.source_title,
    )
    # Upsert by id
    await db.saved_insights.update_one(
        {"id": saved.id},
        {"$set": saved.model_dump()},
        upsert=True,
    )
    return saved


@api_router.get("/insights/saved", response_model=List[SavedInsight])
async def list_saved():
    docs = (
        await db.saved_insights.find({}, {"_id": 0}).sort("saved_at", -1).to_list(500)
    )
    return [SavedInsight(**d) for d in docs]


@api_router.delete("/insights/saved/{insight_id}")
async def delete_saved(insight_id: str):
    res = await db.saved_insights.delete_one({"id": insight_id})
    return {"deleted": res.deleted_count}


# ============ Phase 2 Endpoints ============

CHAT_SYSTEM_TEMPLATE = """You are the AI research companion inside Insight Swipe. The user is reading this specific insight:

TOPIC: {topic}
HEADLINE: {headline}
INSIGHT: {text}
{source_line}

Your job:
- Answer follow-up questions grounded in this insight.
- Give examples, comparisons, deeper mechanics, historical context, or counterpoints.
- Be concise. Use short paragraphs. Sound like a smart friend, not a textbook.
- Include quantified data whenever possible.
- If asked to challenge or steelman, do so honestly.
- Never invent sources. If you don't know, say so."""


@api_router.post("/insights/chat")
async def chat_stream(req: ChatRequest):
    source_line = (
        f"SOURCE: {req.source_title or ''} — {req.source_url}" if req.source_url else ""
    )
    system = CHAT_SYSTEM_TEMPLATE.format(
        topic=req.insight_topic,
        headline=req.insight_headline,
        text=req.insight_text,
        source_line=source_line,
    )

    session_id = f"chat-{req.insight_id}-{req.session_id}"
    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=session_id,
        system_message=system,
    ).with_model(LLM_PROVIDER, LLM_MODEL)

    # Prepend history as context in the user message
    history_txt = ""
    if req.history:
        parts = []
        for h in req.history[-6:]:
            role = h.get("role", "user").upper()
            parts.append(f"{role}: {h.get('content', '')}")
        history_txt = "PRIOR CONVERSATION:\n" + "\n".join(parts) + "\n\n"

    user_msg = UserMessage(text=f"{history_txt}CURRENT QUESTION: {req.message}")

    async def event_gen():
        try:
            async for ev in chat.stream_message(user_msg):
                if isinstance(ev, TextDelta):
                    payload = json.dumps({"delta": ev.content})
                    yield f"data: {payload}\n\n"
                elif isinstance(ev, StreamDone):
                    yield "data: [DONE]\n\n"
                    break
        except Exception as e:
            logging.exception("chat stream failed")
            err = json.dumps({"error": str(e)})
            yield f"data: {err}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@api_router.post("/insights/tts")
async def tts(req: TTSRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    if len(text) > 4000:
        text = text[:4000]
    try:
        audio_bytes = await tts_client.generate_speech(
            text=text, model="tts-1", voice=req.voice or "sage"
        )
    except Exception as e:
        logging.exception("TTS failed")
        raise HTTPException(status_code=500, detail="TTS generation failed")
    return Response(content=audio_bytes, media_type="audio/mpeg")


@api_router.post("/insights/stt")
async def stt(file: UploadFile = File(...)):
    """Transcribe uploaded audio (webm/wav/mp3/m4a). Returns {text}."""
    # Save to temp file (whisper needs a real path with known extension)
    suffix = ".webm"
    if file.filename:
        for ext in (".webm", ".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".mpga"):
            if file.filename.lower().endswith(ext):
                suffix = ext
                break
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        data = await file.read()
        tmp.write(data)
        tmp.flush()
        tmp.close()

        with open(tmp.name, "rb") as f:
            result = await stt_client.transcribe(
                file=f, model="whisper-1", response_format="text"
            )
        # result may be a string (text format) or object with .text
        text = (
            result if isinstance(result, str) else getattr(result, "text", str(result))
        )
        return {"text": (text or "").strip()}
    except Exception as e:
        logging.exception("STT failed")
        raise HTTPException(status_code=500, detail="Transcription failed")
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


@api_router.post("/insights/regenerate", response_model=Insight)
async def regenerate_insight(req: RegenerateRequest):
    """Regenerate one single insight for the same topic — different angle. Uses deep research."""
    try:
        bundle = await deep_research(
            topic=req.topic,
            tavily_client=tavily,
            llm_key=EMERGENT_LLM_KEY,
            extract_top_n=3,
            youtube_top_n=1,
        )
    except Exception:
        logging.exception("regenerate deep_research failed")
        raise HTTPException(status_code=500, detail="Research failed")

    exclude_list = (req.exclude + [req.previous_text])[-30:]
    items = await generate_insights_from_bundle(
        bundle=bundle,
        llm_key=EMERGENT_LLM_KEY,
        count=1,
        exclude_headlines=exclude_list,
    )
    if not items:
        raise HTTPException(status_code=500, detail="Could not regenerate")

    insights = to_insights(
        items[:1], req.topic, bundle.sources, preserve_id=req.insight_id
    )
    return insights[0]


@api_router.post("/insights/more", response_model=GenerateResponse)
async def more_insights(req: MoreInsightsRequest):
    """Continue the feed — 12 more insights (deep research), avoiding prior headlines, honoring steer + behavior."""
    research_topic = req.topic
    if req.steer:
        research_topic = f"{req.topic} — {req.steer}"

    try:
        bundle = await deep_research(
            topic=research_topic,
            tavily_client=tavily,
            llm_key=EMERGENT_LLM_KEY,
            extract_top_n=5,
            youtube_top_n=2,
        )
    except Exception:
        logging.exception("/more deep_research failed")
        raise HTTPException(status_code=500, detail="Research failed")

    behavior_ctx = await build_behavior_context(req.session_id)

    items = await generate_insights_from_bundle(
        bundle=bundle,
        llm_key=EMERGENT_LLM_KEY,
        count=12,
        behavior_context=behavior_ctx,
        steer=req.steer or "",
        exclude_headlines=req.previous_headlines,
    )
    insights = to_insights(items, req.topic, bundle.sources)
    return GenerateResponse(
        session_id=req.session_id, topic=req.topic, insights=insights
    )


@api_router.post("/insights/behavior")
async def record_behavior(ev: BehaviorEvent):
    """Persist user behavior signal for personalization."""
    doc = ev.model_dump()
    doc["timestamp"] = datetime.now(timezone.utc).isoformat()
    await db.behaviors.insert_one(doc)
    return {"ok": True}


# ============ Rabbit-hole directions ============

DIRECTIONS_SYSTEM = """You are Insight Swipe's research director. When a user submits a topic (or a set of interests), you generate 4 distinct "rabbit-hole" directions they can go deep in. Each direction is a specific, provocative angle — not a generic subcategory.

Rules:
- Each direction must open a genuinely non-obvious research thread.
- Prefer angles like: hidden history, contrarian economics, key characters, policy shifts, industry dynamics, psychological hooks.
- The 4 directions must be DIFFERENT from each other.
- HARD ANTI-BLEND RULE: if the user picked MULTIPLE interests, each direction must live INSIDE ONE of those interests — never fuse two interests into a single blended direction. Distribute the 4 directions across the user's interests so every interest gets its own dedicated rabbit-hole.
- Prefix the title with the interest label in brackets when multi-interest, e.g. `[Celebrity Gossip] Stan Army Economics` and `[AI] The RLHF Data Sweatshops`.
- `title`: 4-8 words after the optional bracketed prefix, punchy, curiosity-inducing. Do NOT restate the topic verbatim.
- `subtitle`: 1 short sentence (max 90 chars) that hints at what makes it interesting.
- Return STRICT JSON only, no prose."""


DIRECTIONS_USER_TEMPLATE = """TOPIC: {topic}
{interest_line}
Return strict JSON:
{{
  "directions": [
    {{"title": "...", "subtitle": "..."}},
    {{"title": "...", "subtitle": "..."}},
    {{"title": "...", "subtitle": "..."}},
    {{"title": "...", "subtitle": "..."}}
  ]
}}"""


@api_router.post("/insights/directions", response_model=DirectionsResponse)
async def get_directions(req: DirectionsRequest):
    topic = req.topic.strip()
    if is_vague(topic):
        raise HTTPException(
            status_code=422,
            detail="Topic is too vague. Try something more specific.",
        )

    interest_line = ""
    if req.interests and len(req.interests) > 1:
        interest_line = (
            "\nUSER PICKED MULTIPLE INTERESTS — treat each as its own lane: "
            + " · ".join(req.interests[:8])
            + "\nRule: produce 4 directions, one per interest as far as possible. Do NOT create fusion / cross-over directions that blend two interests into one. Each direction must clearly belong to ONE interest, and its title MUST start with `[<Interest Name>]`."
        )
    elif req.interests:
        interest_line = (
            "\nUSER INTEREST FOCUS: "
            + req.interests[0]
            + "\nEvery direction must sit inside this single interest lane."
        )
    if req.is_news:
        interest_line += (
            "\nTHE USER IS IN NEWS MODE — prefer directions that focus on RECENT (last 30 days) developments, "
            "current events, and today's headlines. Avoid evergreen or textbook framings."
        )

    session_id = f"dirs-{uuid.uuid4()}"
    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=session_id,
        system_message=DIRECTIONS_SYSTEM,
    ).with_model(LLM_PROVIDER, LLM_MODEL)

    try:
        text = await chat.send_message(
            UserMessage(
                text=DIRECTIONS_USER_TEMPLATE.format(
                    topic=topic, interest_line=interest_line
                )
            )
        )
    except Exception:
        logging.exception("directions LLM failed")
        raise HTTPException(status_code=500, detail="Could not generate directions")

    raw = text.strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()
    if "{" in raw:
        raw = raw[raw.find("{") : raw.rfind("}") + 1]
    try:
        parsed = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=500, detail="Directions parsing failed")

    directions = []
    for d in parsed.get("directions", [])[:4]:
        title = (d.get("title") or "").strip()
        subtitle = (d.get("subtitle") or "").strip()
        if title:
            directions.append(DirectionOption(title=title, subtitle=subtitle))

    if not directions:
        raise HTTPException(status_code=500, detail="No directions produced")

    return DirectionsResponse(topic=topic, directions=directions)


# ============ SSE Streaming (fixes 60s edge timeouts) ============


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _stream_generate(
    topic: str,
    session_id: Optional[str] = None,
    steer: Optional[str] = None,
    previous_headlines: Optional[List[str]] = None,
    direction: Optional[str] = None,
    interests: Optional[List[str]] = None,
    is_news: bool = False,
    extract_top_n: int = 4,
    youtube_top_n: int = 2,
):
    """Async generator producing SSE events for a deep-research + insight-generation flow.

    Two modes:
      - Single-rail (topic only, no interests): the original flow — one persona-less
        deep_research + insight synthesis pass.
      - Multi-rail (interests provided): interests are clustered into 1–4 audience
        personas ("rails") that run deep_research + synthesis IN PARALLEL. Their
        insights are then interleaved round-robin so swiping feels like flipping
        through curated persona lenses.
    """
    q: asyncio.Queue = asyncio.Queue()
    sid = session_id or str(uuid.uuid4())

    research_topic = topic
    if direction:
        research_topic = f"{topic} — specifically: {direction}"
    if steer:
        research_topic = f"{research_topic} — {steer}"
    if is_news:
        research_topic = f"{research_topic} — recent news, last 7 days, prioritize fresh developments"

    use_multirail = bool(interests) and not session_id  # only for first generation

    async def on_progress(ev: dict):
        await q.put(ev)

    async def rail_progress(rail_name: str):
        # Wrap on_progress so per-rail phases get tagged with the rail label,
        # which lets the Loading page show which persona is progressing.
        async def _cb(ev: dict):
            meta = dict(ev.get("meta") or {})
            meta["rail"] = rail_name
            await q.put({**ev, "meta": meta})

        return _cb

    def _combined_steer_for_rail(rail: Optional[dict]) -> str:
        parts = []
        if steer:
            parts.append(steer)
        if direction:
            parts.append(f"Focus rabbit-hole direction: {direction}")
        if rail:
            if rail.get("interests"):
                parts.append(
                    f"THIS RAIL'S SOLE INTEREST: {rail['interests'][0]}. Do NOT mix in "
                    f"any other topic the user picked — those run on separate rails."
                )
        elif interests:
            parts.append(f"User's curated interests: {', '.join(interests[:12])}")
        if is_news:
            parts.append(
                "NEWS MODE: every insight must be tied to something in the LAST 7 DAYS. "
                "Cite specific dates from this week. No evergreen textbook facts."
            )
        return " | ".join(parts)

    async def run_single_rail(rail: dict, target_count: int) -> tuple:
        """Runs deep_research + insight synthesis for one persona rail.
        Returns (rail, bundle, items).

        CRITICAL: each rail researches ONLY its own interest (never the
        joined interest string) so Celebrity Gossip rail never fetches
        AI content and vice-versa."""
        cb = await rail_progress(rail["name"])
        # Rail topic = this rail's single interest (never the joint topic).
        # If the user also picked a direction, append it as sub-focus for THIS rail.
        rail_interest = (rail.get("interests") or [""])[0] or research_topic
        if direction:
            rail_topic = f"{rail_interest} — specifically: {direction}"
        elif steer:
            rail_topic = f"{rail_interest} — {steer}"
        else:
            rail_topic = rail_interest
        if is_news:
            rail_topic = f"{rail_topic} — recent news, last 7 days, prioritize fresh developments"
        try:
            bundle = await deep_research(
                topic=rail_topic,
                tavily_client=tavily,
                llm_key=EMERGENT_LLM_KEY,
                extract_top_n=extract_top_n,
                youtube_top_n=youtube_top_n,
                on_progress=cb,
                persona=rail.get("persona"),
                angle_hint=rail.get("angle_hint"),
                search_hint=rail.get("search_hint"),
                is_news=is_news,
                news_days=7,
            )
        except Exception:
            logging.exception("rail %s: deep_research failed", rail["name"])
            return rail, ResearchBundle(topic=rail_topic), []

        if not bundle.sources:
            # Silent retry once for this rail (transient Tavily blip)
            try:
                await asyncio.sleep(1.0)
                bundle = await deep_research(
                    topic=rail_topic,
                    tavily_client=tavily,
                    llm_key=EMERGENT_LLM_KEY,
                    extract_top_n=extract_top_n,
                    youtube_top_n=youtube_top_n,
                    on_progress=cb,
                    persona=rail.get("persona"),
                    angle_hint=rail.get("angle_hint"),
                    search_hint=rail.get("search_hint"),
                    is_news=is_news,
                    news_days=7,
                )
            except Exception:
                pass

        if not bundle.sources:
            logging.warning("rail %s: 0 sources — dropping rail", rail["name"])
            return rail, bundle, []

        behavior_ctx = (
            ""  # multi-rail is only used on first generation (no history yet)
        )
        try:
            items = await generate_insights_from_bundle(
                bundle=bundle,
                llm_key=EMERGENT_LLM_KEY,
                count=target_count,
                behavior_context=behavior_ctx,
                steer=_combined_steer_for_rail(rail),
                exclude_headlines=previous_headlines or [],
                model_provider=LLM_PROVIDER,
                model_name=LLM_MODEL,
                persona=rail.get("persona"),
                voice_hint=rail.get("voice_hint"),
                angle_hint=rail.get("angle_hint"),
                is_news=is_news,
            )
        except Exception:
            logging.exception("rail %s: insight synthesis failed", rail["name"])
            items = []
        return rail, bundle, items

    async def worker():
        try:
            await q.put({"phase": "start", "message": "Starting deep research"})

            # ── MULTI-RAIL PATH ─────────────────────────────────────────
            if use_multirail:
                await q.put({"phase": "planning", "message": "Reading your interests"})
                rails = await plan_rails(
                    interests=interests or [],
                    is_news=is_news,
                    llm_key=EMERGENT_LLM_KEY,
                )
                if not rails:
                    # Fall back to single-rail if the planner failed
                    use_mr = False
                else:
                    use_mr = True
                    await q.put(
                        {
                            "phase": "rails_planned",
                            "message": f"Curated {len(rails)} persona lens{'es' if len(rails) != 1 else ''}",
                            "meta": {
                                "rails": [
                                    {"name": r["name"], "persona": r["persona"]}
                                    for r in rails
                                ]
                            },
                        }
                    )

                if use_mr:
                    # Distribute the 12 cards across rails. Give leftovers to the first rails.
                    total = 12
                    per_rail = [total // len(rails)] * len(rails)
                    for i in range(total - sum(per_rail)):
                        per_rail[i % len(rails)] += 1
                    # Buffer each rail's generation count so a rail that under-delivers
                    # (LLM shortfall or a Tavily miss) can be covered by neighbours.
                    # For N rails asking for `k` each, we generate `min(12, k + 3)` per
                    # rail. That guarantees at least 12 interleavable items even if one
                    # rail returns half of what it was asked for.
                    target_counts = [min(12, c + 3) for c in per_rail]

                    # Run rails in parallel
                    tasks = [
                        run_single_rail(r, tc) for r, tc in zip(rails, target_counts)
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=False)

                    # Round-robin interleave (Card 1 = Rail A, Card 2 = Rail B, …).
                    # CRITICAL: resolve each item's source_index against ITS OWN rail's
                    # sources here — otherwise once we merge sources across rails the
                    # indices point at the wrong rail's URLs.
                    per_rail_items: List[List[dict]] = []
                    per_rail_sources: List[List[dict]] = []
                    per_rail_labels: List[str] = []
                    for (rail, bundle, items), keep in zip(results, per_rail):
                        rail_sources = bundle.sources[:15]
                        tagged = []
                        # Keep ALL generated items for this rail — the round-robin
                        # interleave below will pull only what it needs, but any
                        # rail that under-delivers is compensated by neighbours.
                        for it in items:
                            it = dict(it)
                            # Resolve source now, while indices still refer to this rail's sources.
                            idx = it.get("source_index")
                            if isinstance(idx, int) and 1 <= idx <= len(rail_sources):
                                s = rail_sources[idx - 1]
                                it["_source_url"] = s.get("url")
                                it["_source_title"] = s.get("title")
                            # Nuke the now-meaningless index so to_insights won't re-resolve.
                            it["source_index"] = None
                            it["_rail"] = rail["name"]
                            it["_persona"] = rail["persona"]
                            it["_interest"] = (rail.get("interests") or [None])[0]
                            tagged.append(it)
                        per_rail_items.append(tagged)
                        per_rail_sources.append(rail_sources)
                        per_rail_labels.append(rail["name"])

                    interleaved: List[dict] = []
                    idxs = [0] * len(per_rail_items)
                    while len(interleaved) < total:
                        made_progress = False
                        for i, lst in enumerate(per_rail_items):
                            if idxs[i] < len(lst):
                                interleaved.append(lst[idxs[i]])
                                idxs[i] += 1
                                made_progress = True
                                if len(interleaved) >= total:
                                    break
                        if not made_progress:
                            break  # all rails exhausted

                    if not interleaved:
                        # Detect the specific LLM-budget-exhausted failure so the
                        # user gets an actionable message instead of a generic "try again".
                        budget_hit = False
                        # Check Tavily quota first (fastest signal — bundle carries the flag)
                        tavily_quota_hit = any(
                            (getattr(b, "quota_exceeded", False)) for _, b, _ in results
                        )
                        for _, bundle, items in results:
                            if items:
                                continue
                            # If NO rail produced items and NO sources either, the LLM
                            # planner most likely errored — often the Emergent LLM key
                            # hitting its monthly budget.
                        try:
                            # Re-run one cheap LLM ping to distinguish key-out-of-budget from Tavily
                            from emergentintegrations.llm.chat import \
                                LlmChat as _LC
                            from emergentintegrations.llm.chat import \
                                UserMessage as _UM

                            _c = _LC(
                                api_key=EMERGENT_LLM_KEY,
                                session_id="probe",
                                system_message="Reply with 'ok'",
                            ).with_model("openai", "gpt-5.6-luna")
                            await _c.send_message(_UM(text="ping"))
                        except Exception as e:
                            if (
                                "budget" in str(e).lower()
                                or "429" in str(e)
                                or "quota" in str(e).lower()
                            ):
                                budget_hit = True

                        if tavily_quota_hit:
                            await q.put(
                                {
                                    "phase": "error",
                                    "message": "Web-search quota exhausted (Tavily). Please upgrade your Tavily plan or contact support@tavily.com to continue.",
                                    "meta": {"reason": "tavily_quota_exceeded"},
                                }
                            )
                        elif budget_hit:
                            await q.put(
                                {
                                    "phase": "error",
                                    "message": "AI credits are exhausted. Please top up your Emergent LLM key (Profile → Universal Key → Add Balance).",
                                    "meta": {"reason": "llm_budget_exceeded"},
                                }
                            )
                        else:
                            await q.put(
                                {
                                    "phase": "error",
                                    "message": "Couldn't build a feed for these interests right now. Please try again.",
                                    "meta": {"reason": "all_rails_empty"},
                                }
                            )
                        return

                    # Merge sources from all rails (dedupe by URL, preserve order)
                    merged_sources: List[dict] = []
                    seen_urls = set()
                    for lst in per_rail_sources:
                        for s in lst:
                            u = s.get("url")
                            if not u or u in seen_urls:
                                continue
                            seen_urls.add(u)
                            merged_sources.append(s)

                    insights = to_insights(interleaved, topic, merged_sources)

                    await db.sessions.insert_one(
                        {
                            "session_id": sid,
                            "topic": topic,
                            "direction": direction,
                            "interests": interests,
                            "is_news": is_news,
                            "rails": [
                                {"name": r["name"], "persona": r["persona"]}
                                for r in rails
                            ],
                            "angles": [],
                            "sources_used": [s.get("url") for s in merged_sources[:15]],
                            "youtube_sources": [],
                            "insights": [i.model_dump() for i in insights],
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )

                    payload = {
                        "session_id": sid,
                        "topic": topic,
                        "rails": [
                            {"name": r["name"], "persona": r["persona"]} for r in rails
                        ],
                        "insights": [i.model_dump() for i in insights],
                    }
                    await q.put({"phase": "done", "data": payload})
                    return

            # ── SINGLE-RAIL PATH (topic-only OR /more/ endpoint) ─────────
            bundle = await deep_research(
                topic=research_topic,
                tavily_client=tavily,
                llm_key=EMERGENT_LLM_KEY,
                extract_top_n=extract_top_n,
                youtube_top_n=youtube_top_n,
                on_progress=on_progress,
                is_news=is_news,
                news_days=7,
            )
            # If Tavily gave us nothing on the first pass, retry once — often
            # a single flaky moment on Tavily's edge (RemoteDisconnected).
            if not bundle.sources:
                logging.warning(
                    "deep_research returned 0 sources for %r — retrying once",
                    research_topic,
                )
                await q.put(
                    {
                        "phase": "searching",
                        "message": "Retrying web search",
                        "meta": {"retry": 1},
                    }
                )
                await asyncio.sleep(1.5)
                bundle = await deep_research(
                    topic=research_topic,
                    tavily_client=tavily,
                    llm_key=EMERGENT_LLM_KEY,
                    extract_top_n=extract_top_n,
                    youtube_top_n=youtube_top_n,
                    on_progress=on_progress,
                    is_news=is_news,
                    news_days=7,
                )
            if not bundle.sources:
                # Real outage / rate limit — fail loudly so the user can retry,
                # instead of silently producing empty "no sources / no specifics" cards.
                logging.error(
                    "deep_research returned 0 sources after retry for %r — aborting",
                    research_topic,
                )
                if getattr(bundle, "quota_exceeded", False):
                    await q.put(
                        {
                            "phase": "error",
                            "message": "Web-search quota exhausted (Tavily). Please upgrade your Tavily plan or contact support@tavily.com to continue.",
                            "meta": {"reason": "tavily_quota_exceeded"},
                        }
                    )
                else:
                    await q.put(
                        {
                            "phase": "error",
                            "message": "Web search is temporarily unavailable. Please try again in a moment.",
                            "meta": {"reason": "no_sources"},
                        }
                    )
                return
            await q.put(
                {"phase": "synthesizing", "message": "Synthesizing 12 insights"}
            )
            behavior_ctx = await build_behavior_context(sid) if session_id else ""

            combined_steer = steer or ""
            if direction:
                combined_steer = (
                    f"{combined_steer + ' | ' if combined_steer else ''}"
                    f"Focus rabbit-hole direction: {direction}"
                )
            if interests:
                interests_txt = ", ".join(interests[:12])
                combined_steer = (
                    f"{combined_steer + ' | ' if combined_steer else ''}"
                    f"User's curated interests (deepest → broadest): {interests_txt}. "
                    f"Weave these interests into the insight angles — pull cross-connections, examples, and comparisons that touch these areas."
                )
            if is_news:
                combined_steer = (
                    f"{combined_steer + ' | ' if combined_steer else ''}"
                    "The user chose these from NEWS mode — prioritize CURRENT, FRESH, RECENT developments from the last 7 days. "
                    "Prefer 2025-2026 sources. Each insight should feel like a headline you'd read this week — cite specific dates, "
                    "companies, laws, launches, or events that happened recently. Avoid evergreen textbook facts."
                )

            items = await generate_insights_from_bundle(
                bundle=bundle,
                llm_key=EMERGENT_LLM_KEY,
                count=12,
                behavior_context=behavior_ctx,
                steer=combined_steer,
                exclude_headlines=previous_headlines or [],
                model_provider=LLM_PROVIDER,
                model_name=LLM_MODEL,
                is_news=is_news,
            )
            insights = to_insights(items, topic, bundle.sources)

            if not session_id:
                await db.sessions.insert_one(
                    {
                        "session_id": sid,
                        "topic": topic,
                        "direction": direction,
                        "angles": bundle.angles,
                        "sources_used": [s.get("url") for s in bundle.sources[:15]],
                        "youtube_sources": [y["url"] for y in bundle.youtube],
                        "insights": [i.model_dump() for i in insights],
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

            payload = {
                "session_id": sid,
                "topic": topic,
                "insights": [i.model_dump() for i in insights],
            }
            await q.put({"phase": "done", "data": payload})
        except Exception as e:
            logging.exception("stream worker failed")
            await q.put({"phase": "error", "message": str(e) or "Something went wrong"})
        finally:
            await q.put(None)

    task = asyncio.create_task(worker())
    try:
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=12)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            if item is None:
                break
            yield _sse(item)
    finally:
        if not task.done():
            task.cancel()


class GenerateStreamRequest(BaseModel):
    topic: str
    direction: Optional[str] = None
    interests: List[str] = Field(default_factory=list)
    is_news: bool = False


@api_router.post("/insights/generate/stream")
async def generate_stream(req: GenerateStreamRequest):
    topic = req.topic.strip()
    if is_vague(topic):

        async def err():
            yield _sse(
                {
                    "phase": "error",
                    "message": "Topic is too vague. Try something more specific — e.g. 'how Amazon became successful'.",
                }
            )

        return StreamingResponse(
            err(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        _stream_generate(
            topic=topic,
            direction=(req.direction or None),
            interests=(req.interests or None),
            is_news=req.is_news,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@api_router.post("/insights/more/stream")
async def more_stream(req: MoreInsightsRequest):
    return StreamingResponse(
        _stream_generate(
            topic=req.topic,
            session_id=req.session_id,
            steer=req.steer,
            previous_headlines=req.previous_headlines,
            extract_top_n=5,
            youtube_top_n=2,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============ Voice-driven rewrite / steer ============

REWRITE_SYSTEM = """You are the AI research companion inside Insight Swipe. The user is reading THIS specific insight and just spoke a request into their mic.

TOPIC: {topic}
CURRENT INSIGHT HEADLINE: {headline}
CURRENT INSIGHT TEXT: {text}
{source_line}

USER'S SPOKEN REQUEST: "{transcript}"

Rewrite the insight from scratch to satisfy the user's request while keeping the SAME wow-factor quality bar:
- Preserve or improve the insight's core surprising finding.
- Weave in what the user asked for (more depth, examples, comparison, challenge, specific angle, etc.).
- Keep it 2–4 sentences, dense but readable. No jargon.
- MUST include specific numbers / dates / named entities.
- Do NOT wrap in JSON. Do NOT preface with "Sure" or "Here's". Just output the new insight text, then on a new line "HEADLINE: <max 8 words>".

Stream the answer naturally, token by token."""


class SteerModeResp(BaseModel):
    mode: str  # "rewrite" or "steer"
    reason: str = ""


async def _classify_mic_intent(transcript: str, insight_headline: str) -> str:
    """LLM classifies the user's request into one of four modes:
    - "rewrite": modify just the CURRENT insight (deeper explanation of this one, or wording tweak)
    - "steer":   generate a NEW batch of 12 insights biased by the request
    - "transform": TRANSFORM every existing insight in place (add examples, use bullets, use simpler
                   language, arrange chronologically, restrict source type, etc.)
    - "chat":    conversational question that just wants an answer, not a feed change
    """
    system = """Classify the user's request into EXACTLY one of four modes:

- "rewrite": modify only the CURRENT insight the user is looking at (make THIS one deeper / clearer / different angle).
- "steer":   generate a NEW batch of upcoming insights biased by the request (e.g. "focus on Japan", "next topic", "show me recent news", "show me marketing examples").
- "transform": TRANSFORM every existing insight card in place — e.g. "change all the cards", "add examples on all cards", "arrange in chronological order", "use bullet points", "use simple words / analogies", "only use research papers as source", "add more numbers to every insight", "translate all to Hindi".
- "chat": a conversational question or clarification that just wants an ANSWER (no card changes) — e.g. "what does NPCI mean", "explain this to me", "why is that surprising".

Return strict JSON: {"mode": "rewrite" | "steer" | "transform" | "chat"}. No prose."""
    user = (
        f'CURRENT INSIGHT HEADLINE: "{insight_headline}"\n\nUSER SAID: "{transcript}"'
    )

    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=f"intent-{uuid.uuid4()}",
        system_message=system,
    ).with_model(LLM_PROVIDER, LLM_MODEL)
    try:
        raw = await chat.send_message(UserMessage(text=user))
        s = raw.strip().strip("`")
        if s.startswith("json"):
            s = s[4:].strip()
        if "{" in s:
            s = s[s.find("{") : s.rfind("}") + 1]
        parsed = json.loads(s)
        mode = (parsed.get("mode") or "").strip().lower()
        return mode if mode in ("rewrite", "steer", "transform", "chat") else "chat"
    except Exception:
        # Heuristic fallback
        t = transcript.lower()
        transform_words = (
            "all the cards",
            "every card",
            "all cards",
            "each insight",
            "arrange",
            "chronological",
            "bullet points",
            "simple words",
            "simpler language",
            "analogies",
            "use only",
            "only use",
            "add example",
            "add examples",
            "change all",
        )
        steer_words = (
            "focus on",
            "only show",
            "next topic",
            "different topic",
            "show me",
            "instead",
            "recent news",
        )
        rewrite_words = (
            "this insight",
            "this one",
            "explain this",
            "deeper on this",
            "clarify this",
            "elaborate this",
        )
        if any(w in t for w in transform_words):
            return "transform"
        if any(w in t for w in rewrite_words):
            return "rewrite"
        if any(w in t for w in steer_words):
            return "steer"
        return "chat"


@api_router.post("/insights/rewrite/stream")
async def rewrite_stream(req: RewriteRequest):
    """Mic-driven interaction. First emits {phase:'mode', mode:'rewrite'|'steer'}.
    If rewrite: streams {phase:'delta', delta: '...'} tokens then {phase:'done', data:{headline, text}}.
    If steer: falls back to full _stream_generate on the topic + transcript-as-steer."""

    transcript = (req.transcript or "").strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Empty transcript")

    mode = await _classify_mic_intent(transcript, req.insight_headline)

    async def rewrite_generator():
        # Announce mode first so client can pick the right UI
        yield _sse({"phase": "mode", "mode": "rewrite"})

        source_line = (
            f"SOURCE: {req.source_title or ''} — {req.source_url}"
            if req.source_url
            else ""
        )
        system = REWRITE_SYSTEM.format(
            topic=req.topic,
            headline=req.insight_headline,
            text=req.insight_text,
            source_line=source_line,
            transcript=transcript,
        )

        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"rewrite-{req.insight_id}-{uuid.uuid4()}",
            system_message=system,
        ).with_model(LLM_PROVIDER, LLM_MODEL)

        collected = ""
        try:
            async for ev in chat.stream_message(
                UserMessage(text=f"User wants: {transcript}. Rewrite the insight now.")
            ):
                if isinstance(ev, TextDelta):
                    collected += ev.content
                    yield _sse({"phase": "delta", "delta": ev.content})
                elif isinstance(ev, StreamDone):
                    break
        except Exception as e:
            logging.exception("rewrite stream failed")
            yield _sse({"phase": "error", "message": "Rewrite failed"})
            return

        # Parse headline from the last line if the model followed the format
        text_body = collected
        headline = req.insight_headline
        for line in reversed(collected.splitlines()):
            stripped = line.strip()
            if stripped.upper().startswith("HEADLINE:"):
                headline = stripped[len("HEADLINE:") :].strip().strip('"')
                # remove the last "HEADLINE:" line from body
                text_body = collected.rsplit(line, 1)[0].rstrip()
                break

        yield _sse(
            {
                "phase": "done",
                "data": {
                    "insight_id": req.insight_id,
                    "headline": headline,
                    "text": text_body.strip(),
                },
            }
        )

    async def steer_generator():
        yield _sse({"phase": "mode", "mode": "steer"})
        async for chunk in _stream_generate(
            topic=req.topic,
            session_id=req.session_id,
            steer=transcript,
            previous_headlines=req.previous_headlines,
            extract_top_n=5,
            youtube_top_n=2,
        ):
            yield chunk

    gen = rewrite_generator() if mode == "rewrite" else steer_generator()
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============ Branching endpoints (Go Deeper / Explain / Questions / Answer) ============

BRANCH_INSIGHT_RULES = """You are extending an Insight Swipe research session. The same non-negotiable quality bar applies:
- Every insight must contain a SPECIFIC number, year, percent, dollar/rupee figure, named person, named company, named study, or named regulation.
- No vague adjectives ("huge", "many", "significant"). If you cannot ground it, DO NOT write it.
- 2–4 sentences. Punchy. First sentence hits the striking fact.
- Non-obvious payoff — mechanism, not headline fact.
- Simple language. No jargon (or explain it in-line).
- headline: ≤ 8 words, hook-driven.
Return STRICT JSON only, no markdown, no prose."""


async def _stream_branch_generic(
    topic: str,
    session_id: str,
    source: SourceInsight,
    mode: str,  # "deeper" | "explain" | "answer"
    phrase: Optional[str] = None,
    question: Optional[str] = None,
    previous_headlines: Optional[List[str]] = None,
    count: int = 12,
):
    """Shared SSE producer for deeper/explain/answer branches. Uses the parent insight as context."""
    q: asyncio.Queue = asyncio.Queue()

    if mode == "deeper":
        research_focus = f"{topic} — go deeper on this specific insight: {source.headline}. {source.text}"
    elif mode == "explain":
        research_focus = (
            f"{topic} — explain the concept/entity '{phrase}' in the context of this insight: "
            f"{source.headline}. {source.text}"
        )
    else:  # answer
        research_focus = (
            f"{topic} — answer this question: {question}. Context insight: "
            f"{source.headline}. {source.text}"
        )

    async def on_progress(ev: dict):
        await q.put(ev)

    async def worker():
        try:
            await q.put({"phase": "start", "message": "Digging deeper"})
            bundle = await deep_research(
                topic=research_focus,
                tavily_client=tavily,
                llm_key=EMERGENT_LLM_KEY,
                extract_top_n=5 if mode == "deeper" else 3,
                youtube_top_n=2 if mode == "deeper" else 1,
                on_progress=on_progress,
            )

            # Compose a per-mode user prompt on top of the shared bundle context
            from research import build_research_context  # local import

            base_context = build_research_context(bundle)

            if mode == "deeper":
                user_prompt = f"""{base_context}

PARENT INSIGHT (expand different facets of THIS specific insight):
HEADLINE: {source.headline}
TEXT: {source.text}

Generate EXACTLY {count} NEW insights that go DEEPER on the parent insight. Each of the {count} must:
- Elaborate a distinct mechanism, sub-story, data point, or comparison referenced by or adjacent to the parent insight.
- NOT restate the parent insight.
- Follow the same quality bar (specific number + non-obvious payoff).
- Cover different angles across the {count}.
"""
            elif mode == "explain":
                user_prompt = f"""{base_context}

CONTEXT INSIGHT (user was reading this):
HEADLINE: {source.headline}
TEXT: {source.text}

The user tapped: "{phrase}"

Write EXACTLY 1 insight card that explains "{phrase}" in the context of the parent insight and the user's overall topic. The card must:
- Define / unpack what "{phrase}" actually is.
- Include the same quality bar (specific number, named entity, or study).
- 2–4 sentences.
- Feel like a Wikipedia-style expansion but keep the Insight-Swipe voice.
"""
            else:  # answer
                user_prompt = f"""{base_context}

CONTEXT INSIGHT (user was reading this):
HEADLINE: {source.headline}
TEXT: {source.text}

USER QUESTION: "{question}"

Write EXACTLY 1 insight card that answers this question specifically. The card must:
- Directly answer the question — no preamble.
- Include the same quality bar (specific number, named entity, study, or regulation).
- 2–4 sentences.
- Grounded in the sources above.
"""

            if previous_headlines:
                joined = "\n".join(f"- {h}" for h in previous_headlines[-30:])
                user_prompt += (
                    "\n\nDO NOT REPEAT or paraphrase these previously shown headlines:\n"
                    + joined
                )

            user_prompt += '\n\nStrict JSON only:\n{"insights": [{"headline": "...", "text": "...", "source_index": 1}, ...]}'

            await q.put(
                {
                    "phase": "synthesizing",
                    "message": "Writing insight" if count == 1 else "Writing insights",
                }
            )

            session_lbl = f"branch-{mode}-{uuid.uuid4()}"
            chat = LlmChat(
                api_key=EMERGENT_LLM_KEY,
                session_id=session_lbl,
                system_message=BRANCH_INSIGHT_RULES,
            ).with_model(LLM_PROVIDER, LLM_MODEL)

            raw = await chat.send_message(UserMessage(text=user_prompt))
            text = raw.strip().strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
            if "{" in text:
                text = text[text.find("{") : text.rfind("}") + 1]
            parsed = json.loads(text)

            items = parsed.get("insights", [])[:count]
            insights = to_insights(items, topic, bundle.sources)

            payload = {
                "session_id": session_id,
                "topic": topic,
                "insights": [i.model_dump() for i in insights],
                "mode": mode,
            }
            await q.put({"phase": "done", "data": payload})
        except Exception as e:
            logging.exception("branch worker (%s) failed", mode)
            await q.put({"phase": "error", "message": str(e) or "Failed to expand"})
        finally:
            await q.put(None)

    task = asyncio.create_task(worker())
    try:
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=12)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            if item is None:
                break
            yield _sse(item)
    finally:
        if not task.done():
            task.cancel()


@api_router.post("/insights/deeper/stream")
async def deeper_stream(req: DeeperRequest):
    """Double-tap Go Deeper → generate 12 insights that expand the source insight."""
    return StreamingResponse(
        _stream_branch_generic(
            topic=req.topic,
            session_id=req.session_id,
            source=req.source_insight,
            mode="deeper",
            previous_headlines=req.previous_headlines,
            count=12,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@api_router.post("/insights/explain/stream")
async def explain_stream(req: ExplainRequest):
    """Wikipedia-style: user tapped a word/phrase → generate 1 insight card explaining it in context."""
    return StreamingResponse(
        _stream_branch_generic(
            topic=req.topic,
            session_id=req.session_id,
            source=req.source_insight,
            mode="explain",
            phrase=req.phrase,
            count=1,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


QUESTIONS_SYSTEM = """You generate FIVE questions a curious reader would ask after reading a specific insight.
Rules:
- Questions must open genuinely new threads — not paraphrase the insight.
- Each question 6–14 words. Concrete. Not vague.
- No yes/no questions. No "what is X" if X is already defined in the insight.
- Cover different angles: mechanism, comparison, counter-example, historical parallel, current implication.
Return STRICT JSON: {"questions": ["...", "...", ...]}"""


@api_router.post("/insights/questions", response_model=QuestionsResponse)
async def get_questions(req: QuestionsRequest):
    src = req.source_insight
    user_msg = f"""TOPIC: {req.topic}

CURRENT INSIGHT:
HEADLINE: {src.headline}
TEXT: {src.text}

Return 5 questions strictly as JSON: {{"questions": [...]}}."""

    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=f"questions-{uuid.uuid4()}",
        system_message=QUESTIONS_SYSTEM,
    ).with_model(LLM_PROVIDER, LLM_MODEL)

    try:
        text = await chat.send_message(UserMessage(text=user_msg))
    except Exception:
        logging.exception("questions LLM failed")
        raise HTTPException(status_code=500, detail="Could not generate questions")

    raw = text.strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()
    if "{" in raw:
        raw = raw[raw.find("{") : raw.rfind("}") + 1]
    try:
        parsed = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=500, detail="Questions parse failed")

    questions = [
        (q or "").strip() for q in parsed.get("questions", []) if (q or "").strip()
    ][:5]
    if not questions:
        raise HTTPException(status_code=500, detail="No questions produced")
    return QuestionsResponse(questions=questions)


@api_router.post("/insights/answer/stream")
async def answer_stream(req: AnswerRequest):
    """User tapped one of the suggested questions → generate 1 insight card answering it."""
    return StreamingResponse(
        _stream_branch_generic(
            topic=req.topic,
            session_id=req.session_id,
            source=req.source_insight,
            mode="answer",
            question=req.question,
            previous_headlines=req.previous_headlines,
            count=1,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============ Unified command endpoint (mic OR chat) ============

TRANSFORM_SYSTEM = """You transform an existing set of Insight Swipe cards according to the user's instruction.

RULES:
1. Preserve the CORE data point / mechanism / entity of each insight — the transformation is about STYLE / SOURCE-TYPE / FORMAT, not throwing away the insight.
2. If the user asks to "add examples" — attach a concrete named example.
3. If the user asks for "bullet points" — write each insight as 2–3 short bullet lines (still dense).
4. If the user asks for "simpler language" or "analogies" — use plain English + one concrete analogy.
5. If the user asks to "arrange chronologically" — reorder the array from oldest event to most recent.
6. If the user asks to "only use research papers / studies / journals" — reference specific studies/papers (real ones, not invented) and keep only insights that can honor this.
7. If the user asks to "translate" — translate the text while preserving proper nouns and numbers.
8. Every transformed insight MUST STILL contain a specific number/year/named entity/study.
9. Keep the same number of insights unless the user explicitly asks to filter.

Return STRICT JSON:
{"insights": [{"headline": "...", "text": "...", "source_index": <int|null>}, ...]}"""


async def _stream_transform(
    session_id: str,
    topic: str,
    insights: List[SourceInsight],
    instruction: str,
):
    """Transform all current insights based on the user's instruction. Streams progress + done."""
    q: asyncio.Queue = asyncio.Queue()

    async def worker():
        try:
            await q.put({"phase": "start", "message": "Reshaping your feed"})
            # Build compact context of current insights
            lines = []
            for i, ins in enumerate(insights, 1):
                src_line = (
                    f"   SOURCE: {ins.source_title} — {ins.source_url}"
                    if ins.source_url
                    else ""
                )
                lines.append(
                    f"[{i}] HEADLINE: {ins.headline}\n    TEXT: {ins.text}\n{src_line}"
                )
            ctx = "\n\n".join(lines)

            await q.put({"phase": "synthesizing", "message": "Applying your changes"})

            user_msg = f"""TOPIC: {topic}

CURRENT INSIGHTS ({len(insights)} cards):
{ctx}

USER INSTRUCTION: "{instruction}"

Apply the instruction to ALL {len(insights)} insights. Return strict JSON only, same count as input (unless the instruction explicitly filters). Every insight MUST retain a specific number/year/named entity/study."""

            chat = LlmChat(
                api_key=EMERGENT_LLM_KEY,
                session_id=f"transform-{uuid.uuid4()}",
                system_message=TRANSFORM_SYSTEM,
            ).with_model(LLM_PROVIDER, LLM_MODEL)

            raw = await chat.send_message(UserMessage(text=user_msg))
            text = raw.strip().strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
            if "{" in text:
                text = text[text.find("{") : text.rfind("}") + 1]
            parsed = json.loads(text)
            items = parsed.get("insights", [])

            # Preserve original ids/source_url when possible (align by position)
            transformed: List[Insight] = []
            for i, item in enumerate(items):
                original = insights[i] if i < len(insights) else None
                transformed.append(
                    Insight(
                        id=(
                            original.id
                            if original and original.id
                            else str(uuid.uuid4())
                        ),
                        topic=topic,
                        headline=(item.get("headline") or "").strip(),
                        text=(item.get("text") or "").strip(),
                        source_url=(original.source_url if original else None),
                        source_title=(original.source_title if original else None),
                    )
                )

            payload = {
                "session_id": session_id,
                "topic": topic,
                "insights": [i.model_dump() for i in transformed],
                "mode": "transform",
            }
            await q.put({"phase": "done", "data": payload})
        except Exception as e:
            logging.exception("transform failed")
            await q.put({"phase": "error", "message": str(e) or "Transform failed"})
        finally:
            await q.put(None)

    task = asyncio.create_task(worker())
    try:
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=12)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            if item is None:
                break
            yield _sse(item)
    finally:
        if not task.done():
            task.cancel()


async def _stream_chat_answer(
    req: CommandRequest,
):
    """Stream a plain chat answer using the current insight as context — used when intent is 'chat'."""
    q: asyncio.Queue = asyncio.Queue()

    async def worker():
        try:
            source_line = (
                f"SOURCE: {req.current_insight.source_title or ''} — {req.current_insight.source_url}"
                if req.current_insight.source_url
                else ""
            )
            system = CHAT_SYSTEM_TEMPLATE.format(
                topic=req.current_insight.topic or req.topic,
                headline=req.current_insight.headline,
                text=req.current_insight.text,
                source_line=source_line,
            )
            chat = LlmChat(
                api_key=EMERGENT_LLM_KEY,
                session_id=f"chat-cmd-{uuid.uuid4()}",
                system_message=system,
            ).with_model(LLM_PROVIDER, LLM_MODEL)

            history_txt = ""
            if req.history:
                parts = []
                for h in req.history[-6:]:
                    role = h.get("role", "user").upper()
                    parts.append(f"{role}: {h.get('content', '')}")
                history_txt = "PRIOR CONVERSATION:\n" + "\n".join(parts) + "\n\n"

            um = UserMessage(text=f"{history_txt}CURRENT QUESTION: {req.message}")
            async for ev in chat.stream_message(um):
                if isinstance(ev, TextDelta):
                    await q.put({"phase": "delta", "delta": ev.content})
                elif isinstance(ev, StreamDone):
                    break
            await q.put({"phase": "done"})
        except Exception as e:
            logging.exception("chat command failed")
            await q.put({"phase": "error", "message": str(e) or "Chat failed"})
        finally:
            await q.put(None)

    task = asyncio.create_task(worker())
    try:
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=12)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            if item is None:
                break
            yield _sse(item)
    finally:
        if not task.done():
            task.cancel()


@api_router.post("/insights/command/stream")
async def command_stream(req: CommandRequest):
    """Unified endpoint for both chat and mic commands.
    First SSE event = {phase:'mode', mode:'chat'|'rewrite'|'steer'|'transform'} so the frontend
    can decide how to render subsequent events."""

    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message")

    mode = await _classify_mic_intent(message, req.current_insight.headline)

    async def wrap(sub_gen):
        # Emit mode first, then the sub-generator's events
        yield _sse({"phase": "mode", "mode": mode})
        async for chunk in sub_gen:
            yield chunk

    if mode == "chat":
        gen = wrap(_stream_chat_answer(req))
    elif mode == "rewrite":
        # Reuse the rewrite flow — build a synthetic RewriteRequest
        rw = RewriteRequest(
            session_id=req.session_id,
            topic=req.topic,
            insight_id=req.current_insight.id or "current",
            insight_headline=req.current_insight.headline,
            insight_text=req.current_insight.text,
            source_url=req.current_insight.source_url,
            source_title=req.current_insight.source_title,
            transcript=message,
            previous_headlines=[i.headline for i in req.insights if i.headline],
        )
        # Reuse the internal rewrite_generator by calling the SAME logic inline
        # (extracted below); we re-emit the mode ourselves
        gen = wrap(_stream_rewrite_body(rw))
    elif mode == "steer":
        # Use _stream_generate as a "more" batch with the message as steer
        gen = wrap(
            _stream_generate(
                topic=req.topic,
                session_id=req.session_id,
                steer=message,
                previous_headlines=[i.headline for i in req.insights if i.headline],
                extract_top_n=5,
                youtube_top_n=2,
            )
        )
    else:  # transform
        gen = wrap(
            _stream_transform(
                session_id=req.session_id,
                topic=req.topic,
                insights=req.insights or [req.current_insight],
                instruction=message,
            )
        )

    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _stream_rewrite_body(req: RewriteRequest):
    """Extracted rewrite generator so it can be reused by /command/stream. Emits delta+done events."""
    source_line = (
        f"SOURCE: {req.source_title or ''} — {req.source_url}" if req.source_url else ""
    )
    system = REWRITE_SYSTEM.format(
        topic=req.topic,
        headline=req.insight_headline,
        text=req.insight_text,
        source_line=source_line,
        transcript=req.transcript,
    )
    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=f"rewrite-{req.insight_id}-{uuid.uuid4()}",
        system_message=system,
    ).with_model(LLM_PROVIDER, LLM_MODEL)

    collected = ""
    try:
        async for ev in chat.stream_message(
            UserMessage(text=f"User wants: {req.transcript}. Rewrite the insight now.")
        ):
            if isinstance(ev, TextDelta):
                collected += ev.content
                yield _sse({"phase": "delta", "delta": ev.content})
            elif isinstance(ev, StreamDone):
                break
    except Exception:
        logging.exception("rewrite body failed")
        yield _sse({"phase": "error", "message": "Rewrite failed"})
        return

    text_body = collected
    headline = req.insight_headline
    for line in reversed(collected.splitlines()):
        stripped = line.strip()
        if stripped.upper().startswith("HEADLINE:"):
            headline = stripped[len("HEADLINE:") :].strip().strip('"')
            text_body = collected.rsplit(line, 1)[0].rstrip()
            break

    yield _sse(
        {
            "phase": "done",
            "data": {
                "insight_id": req.insight_id,
                "headline": headline,
                "text": text_body.strip(),
            },
        }
    )


# ============ Interest Onboarding (Spotify-style layered chips) ============

SEED_INTERESTS = [
    "AI",
    "Finance",
    "Business",
    "Communism",
    "Socialism",
    "Capitalism",
    "Governance",
    "Celebrity Gossip",
    "Music",
    "Sports",
    "Health",
    "Science",
    "Philosophy",
    "Technology",
    "History",
    "Design",
    "Psychology",
    "Space",
    "Climate",
    "Startups",
    "Art",
    "Literature",
    "Gaming",
    "Fitness",
    "Cooking",
    "Travel",
    "Cinema",
    "Photography",
]

# Cheap + fast LLM for chip expansion (independent of the main insight LLM)
INTEREST_LLM_PROVIDER = "openai"
INTEREST_LLM_MODEL = "gpt-5.6-luna"


class InterestExpandRequest(BaseModel):
    interest: str
    parent_chain: List[str] = Field(default_factory=list)


class InterestExpandResponse(BaseModel):
    interest: str
    sub_interests: List[str]
    from_cache: bool


class SaveInterestsRequest(BaseModel):
    device_id: str
    interests: List[str]


class UserInterestsResponse(BaseModel):
    device_id: str
    interests: List[str]
    updated_at: Optional[str] = None


def _extract_json_array(text: str):
    import re as _re

    fence = _re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, _re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    return None


@api_router.get("/interests/seed")
async def get_seed_interests():
    return {"interests": SEED_INTERESTS}


@api_router.post("/interests/expand", response_model=InterestExpandResponse)
async def expand_interest(req: InterestExpandRequest):
    key = " > ".join([*req.parent_chain, req.interest]).lower().strip()

    cached = await db.interest_cache.find_one({"key": key}, {"_id": 0})
    if cached and cached.get("sub_interests"):
        return InterestExpandResponse(
            interest=req.interest,
            sub_interests=cached["sub_interests"],
            from_cache=True,
        )

    parent_context = ""
    if req.parent_chain:
        parent_context = f" The user drilled down from: {' → '.join(req.parent_chain)}."

    system = (
        "You are a curious librarian who curates sub-topics for a knowledge discovery app. "
        "Given a topic, return 8 concise, distinct, and interesting sub-topics that a curious "
        "person might want to explore next. Each sub-topic must be 1-4 words, no punctuation. "
        "Return STRICT JSON only: an array of strings, no prose, no markdown fences."
    )
    user_text = (
        f'Topic: "{req.interest}".{parent_context}\n'
        'Return JSON array of 8 short sub-topics. Example: ["Bias", "Alignment", "Regulation", ...]'
    )

    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"expand-{uuid.uuid4()}",
            system_message=system,
        ).with_model(INTEREST_LLM_PROVIDER, INTEREST_LLM_MODEL)
        raw = await chat.send_message(UserMessage(text=user_text))
    except Exception as e:
        logging.exception("interest expand LLM failed")
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    subs = _extract_json_array(raw if isinstance(raw, str) else str(raw)) or []
    subs = [str(s).strip() for s in subs if isinstance(s, str) and str(s).strip()]
    subs = [s for s in subs if 1 <= len(s) <= 60][:8]

    if not subs:
        raise HTTPException(status_code=502, detail="Could not parse sub-interests")

    await db.interest_cache.update_one(
        {"key": key},
        {
            "$set": {
                "key": key,
                "interest": req.interest,
                "parent_chain": req.parent_chain,
                "sub_interests": subs,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        },
        upsert=True,
    )
    return InterestExpandResponse(
        interest=req.interest, sub_interests=subs, from_cache=False
    )


@api_router.get("/interests/me", response_model=UserInterestsResponse)
async def get_my_interests(device_id: str):
    doc = await db.user_interests.find_one({"device_id": device_id}, {"_id": 0})
    if not doc:
        return UserInterestsResponse(device_id=device_id, interests=[], updated_at=None)
    return UserInterestsResponse(
        device_id=device_id,
        interests=doc.get("interests", []),
        updated_at=doc.get("updated_at"),
    )


@api_router.post("/interests/me", response_model=UserInterestsResponse)
async def save_my_interests(req: SaveInterestsRequest):
    seen = set()
    cleaned = []
    for i in req.interests:
        s = str(i).strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            cleaned.append(s)
    updated_at = datetime.now(timezone.utc).isoformat()
    await db.user_interests.update_one(
        {"device_id": req.device_id},
        {
            "$set": {
                "device_id": req.device_id,
                "interests": cleaned,
                "updated_at": updated_at,
            }
        },
        upsert=True,
    )
    return UserInterestsResponse(
        device_id=req.device_id, interests=cleaned, updated_at=updated_at
    )


@api_router.delete("/interests/me")
async def delete_my_interests(device_id: str):
    await db.user_interests.delete_one({"device_id": device_id})
    return {"ok": True}


# ============ Streaming chip expansion + News mode ============

NEWS_SEED_INTERESTS = [
    "Recent AI Developments",
    "Stock Market Performance",
    "Celebrity Gossip",
    "Movies & TV Shows",
    "New Music Releases",
    "World Politics",
    "Sports Highlights",
    "Tech Product Launches",
    "Startup Funding News",
    "Climate & Environment",
    "Science Breakthroughs",
    "Crypto & Web3",
    "Gaming Releases",
    "Fashion & Culture",
    "Health & Wellness",
    "Space Exploration",
]


async def _stream_expand_chips(
    interest: str, parent_chain: List[str], mode: str = "interests"
):
    """Generator: streams sub-chip labels one by one as SSE events."""
    q: asyncio.Queue = asyncio.Queue()
    key = f"{mode}:" + " > ".join([*parent_chain, interest]).lower().strip()

    async def worker():
        try:
            # Cache check first
            cached = await db.interest_cache.find_one({"key": key}, {"_id": 0})
            if cached and cached.get("sub_interests"):
                await q.put({"phase": "start", "from_cache": True})
                for chip in cached["sub_interests"]:
                    await q.put({"phase": "chip", "label": chip})
                await q.put(
                    {
                        "phase": "done",
                        "sub_interests": cached["sub_interests"],
                        "from_cache": True,
                    }
                )
                return

            parent_context = ""
            if parent_chain:
                parent_context = (
                    f" The user drilled down from: {' → '.join(parent_chain)}."
                )

            if mode == "news":
                system = (
                    "You are a news curator for a discovery app. Given a broad news category, "
                    "return 8 distinct CURRENT and SPECIFIC news sub-topics or storylines a curious "
                    "person might want to swipe through. Each sub-topic must be 2-6 words, present-tense "
                    "or noun phrase (e.g. 'OpenAI GPT-6 launch', 'Fed rate decision', 'Taylor Swift tour', "
                    "'Nvidia earnings', 'Israel-Gaza ceasefire'). Return STRICT JSON: an array of strings. "
                    "No prose. No markdown."
                )
                user_text = (
                    f'News category: "{interest}".{parent_context}\n'
                    "Return JSON array of 8 short, current, SPECIFIC news storylines."
                )
            else:
                system = (
                    "You are a curious librarian who curates sub-topics for a knowledge discovery app. "
                    "Given a topic, return 8 concise, distinct, and interesting sub-topics that a curious "
                    "person might want to explore next. Each sub-topic must be 1-4 words, no punctuation. "
                    "Return STRICT JSON only: an array of strings, no prose, no markdown fences."
                )
                user_text = (
                    f'Topic: "{interest}".{parent_context}\n'
                    "Return JSON array of 8 short sub-topics."
                )

            await q.put({"phase": "start", "from_cache": False})

            chat = LlmChat(
                api_key=EMERGENT_LLM_KEY,
                session_id=f"expand-{uuid.uuid4()}",
                system_message=system,
            ).with_model(INTEREST_LLM_PROVIDER, INTEREST_LLM_MODEL)
            raw = await chat.send_message(UserMessage(text=user_text))

            subs = _extract_json_array(raw if isinstance(raw, str) else str(raw)) or []
            subs = [
                str(s).strip() for s in subs if isinstance(s, str) and str(s).strip()
            ]
            subs = [s for s in subs if 1 <= len(s) <= 80][:8]

            if not subs:
                await q.put(
                    {"phase": "error", "message": "Could not parse sub-interests"}
                )
                return

            # Stream chip-by-chip with tiny gaps so the UI reveals them progressively
            for chip in subs:
                await q.put({"phase": "chip", "label": chip})
                await asyncio.sleep(0.08)

            # Cache
            await db.interest_cache.update_one(
                {"key": key},
                {
                    "$set": {
                        "key": key,
                        "interest": interest,
                        "parent_chain": parent_chain,
                        "sub_interests": subs,
                        "mode": mode,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                },
                upsert=True,
            )

            await q.put({"phase": "done", "sub_interests": subs, "from_cache": False})
        except Exception as e:
            logging.exception("stream_expand_chips failed")
            await q.put({"phase": "error", "message": str(e)})
        finally:
            await q.put(None)

    task = asyncio.create_task(worker())
    try:
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=15)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                continue
            if item is None:
                break
            yield _sse(item)
    finally:
        if not task.done():
            task.cancel()


class InterestExpandStreamRequest(BaseModel):
    interest: str
    parent_chain: List[str] = Field(default_factory=list)
    mode: str = "interests"  # "interests" or "news"


@api_router.post("/interests/expand/stream")
async def expand_interest_stream(req: InterestExpandStreamRequest):
    return StreamingResponse(
        _stream_expand_chips(
            req.interest,
            req.parent_chain,
            req.mode if req.mode in ("interests", "news") else "interests",
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@api_router.get("/interests/news/seed")
async def get_news_seed_interests():
    return {"interests": NEWS_SEED_INTERESTS}


# ============ Hot Topics (Landing Pulse) ============
# Curated for 20-30 year-old urban Indians. Pulls today's viral moments
# across politics, Bollywood, cricket, tech, and social-media drama, then
# LLM-synthesises 10 wow-inducing insight cards each tagged with its own
# persona so the swipe deck alternates voices (Russian-roulette feel).


class HotCard(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str  # rabbit-hole topic launched when user picks this card
    headline: str
    text: str
    category: str  # "Politics" | "Bollywood" | "Cricket" | "Tech" | "Reddit" | "Drama"
    rail: str  # persona name
    interest: str
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    blueprint_index: int = 0  # canonical ordering (Politics=0, Geopolitics=1, …)


class HotResponse(BaseModel):
    cards: List[HotCard]
    generated_at: str


# ── Category → search + persona blueprint ───────────────────────────
# 12 blueprints — 2 per broad theme so the deck feels varied even inside
# a category. Each blueprint runs its own Tavily news search + persona
# rail so cards feel distinctly voiced.
HOT_BLUEPRINTS = [
    {
        "category": "Politics",
        "interest": "Indian Politics",
        "rail": "Foreign-Policy Desk",
        "query": "Dharmendra Pradhan resignation education minister India NEET exam leak protests youth aftermath",
        "fallback_query": "India education minister resignation NEET exam leak protests this week",
        "persona": "Ministry-of-External-Affairs desk officer covering Indian politics — reads The Hindu, Indian Express, PRS Legislative Research, ThePrint, Newslaundry, Scroll.in. Tracks Bill numbers, ministerial reshuffles, Lok Sabha proceedings, ED / CBI moves.",
        "search_hint": "The Hindu, Indian Express, ThePrint, Newslaundry, Scroll.in, PRS Legislative Research, LiveLaw, Bar and Bench",
        "voice_hint": "cites the exact minister name, the constituency, the specific act / notification, the parliamentary date; unafraid of BJP / INC / regional-party names",
        "angle_hint": "resignations as political shock-absorbers (not policy fixes), how the exam-leak scandal maps onto BJP's urban-youth vote arithmetic, NTA restructuring, coalition-partner reactions, upstream drift toward 2027 state polls",
    },
    {
        "category": "Geopolitics",
        "interest": "Global Politics",
        "rail": "World Watcher",
        "query": "geopolitics India this week diplomacy sanctions summit border",
        "fallback_query": "world politics news this week India relations",
        "persona": "Foreign-affairs analyst watching how the world moves India — reads Foreign Affairs, Foreign Policy, FT world, Reuters diplomacy, Politico, RAND. Tracks QUAD, BRICS, S. Jaishankar's calendar, US election arcs, China border pings.",
        "search_hint": "Foreign Affairs, Foreign Policy, FT world, Reuters diplomacy, Politico, RAND, CSIS, MEA India",
        "voice_hint": "measured, cites the summit communique, the sanctions list, the exact treaty article — names the ambassador and the exact date",
        "angle_hint": "QUAD/BRICS moves, chip / rare-earth export controls, border deployments, oil-import maths, backchannels, foreign-aid strings",
    },
    {
        "category": "Bollywood",
        "interest": "Bollywood Drama",
        "rail": "Bollywood Insider",
        "query": "Bollywood drama controversy actor actress feud this week India entertainment",
        "fallback_query": "Bollywood news celebrity this week",
        "persona": "Bollywood insider — reads Filmfare, Pinkvilla, Bollywood Hungama, Koimoi, r/BollyBlindsNGossip, The Film Companion, Anupama Chopra. Knows every Karan Johar dinner-party seating chart, PR-plant lore, and industry-plant theory.",
        "search_hint": "Pinkvilla, Bollywood Hungama, Filmfare, Koimoi, r/BollyBlindsNGossip, The Film Companion, Anupama Chopra",
        "voice_hint": "first-name basis with stars (Ranbir, Alia, SRK, Deepika), cites opening-weekend numbers in crore, distinguishes trade vs critic reception",
        "angle_hint": "PR fallouts, star-kid launches, IIFA / Filmfare politics, dating rumours with sourced sightings, feud timelines, breakup PR",
    },
    {
        "category": "Box Office",
        "interest": "Bollywood Business",
        "rail": "Trade Analyst",
        "query": "Bollywood box office collection weekend film India movie business",
        "fallback_query": "Indian film industry box office this week",
        "persona": "Trade analyst — reads Box Office India, Sacnilk, Ormax Media, Ormax Cinematix, Film Companion trade. Talks in crore day-1, weekend multiple, dropoff %, occupancy, ATP.",
        "search_hint": "Box Office India, Sacnilk, Ormax Media, Film Companion trade, Bollywood Hungama box office",
        "voice_hint": "cites the exact crore in India-nett, worldwide box office, ATP, occupancy %, dropoff Monday-vs-Friday, break-even threshold",
        "angle_hint": "which films actually recouped, OTT-release windowing, star fees vs opening, mass-vs-multiplex splits, regional-language crossovers",
    },
    {
        "category": "Cricket",
        "interest": "Cricket Match Analysis",
        "rail": "Cricket Analyst",
        "query": "India cricket match series latest this week batsman bowler performance",
        "fallback_query": "cricket India match report this week",
        "persona": "Cricket analyst — reads ESPNCricinfo, The Cricket Monthly, Cricbuzz, Wisden India, r/Cricket. Fluent in strike-rate splits, powerplay xR, spinner-friendly conditions.",
        "search_hint": "ESPNCricinfo, Cricbuzz, Wisden India, The Cricket Monthly, r/Cricket, Sportstar",
        "voice_hint": "cites the exact strike rate, economy, dot-ball %, DRS moment, boundary %, powerplay run rate",
        "angle_hint": "tactical experiments (impact player, floating batter), selection controversies, spin-vs-pace conditions, DRS controversies, form curves",
    },
    {
        "category": "IPL / BCCI",
        "interest": "Cricket Business",
        "rail": "Cricket Business Analyst",
        "query": "IPL auction BCCI franchise business media rights India cricket money",
        "fallback_query": "IPL BCCI news this week India cricket business",
        "persona": "Cricket business analyst — reads Sportstar, The Ken sports, Moneycontrol sport, The Hindu Business Line sport. Tracks IPL auction prices, media-rights inflation, franchise economics, BCCI notifications.",
        "search_hint": "Sportstar, The Ken sports, Moneycontrol sport, The Hindu Business Line sport, ESPNCricinfo money",
        "voice_hint": "cites the exact auction paddle price in crore, the media rights CAGR, the sponsor CAC, the franchise valuation multiple",
        "angle_hint": "auction economics, media-rights bidding wars, franchise valuation deltas, WPL economics, BCCI board politics",
    },
    {
        "category": "Startups",
        "interest": "Indian Startups",
        "rail": "Startup Insider",
        "query": "India startup funding round layoffs launch this week Bengaluru Mumbai",
        "fallback_query": "Indian startup news funding this week",
        "persona": "Indian-startup insider — reads Entrackr, Inc42, The Ken, Moneycontrol, YourStory, MorningContext, ETtech. Tracks every Series round, ESOP buyback, and DPIIT filing.",
        "search_hint": "Entrackr, Inc42, The Ken, Moneycontrol, MorningContext, ETtech, YourStory",
        "voice_hint": "cites the exact round size in crore or $, valuation multiple, ARR, lead investor (Peak XV, Accel, Lightspeed India), and the founder's name",
        "angle_hint": "funding rounds, secondaries, IPO plumbing (SEBI DRHP), layoff waves, PhonePe / Paytm / Zomato / Nykaa competitive moves",
    },
    {
        "category": "AI & Big Tech",
        "interest": "AI",
        "rail": "AI Researcher",
        "query": "AI model release breakthrough this week OpenAI Anthropic Google Meta",
        "fallback_query": "AI news this week new model release",
        "persona": "AI researcher / ML engineer — reads arXiv daily, tracks OpenAI, Anthropic, Google DeepMind, Meta FAIR. Listens to Dwarkesh Patel, Latent Space. Knows scaling laws, RLHF/DPO, benchmark leaderboards.",
        "search_hint": "arXiv, OpenAI blog, Anthropic research, Google DeepMind, Meta AI, Hugging Face, Import AI, Interconnects, Dwarkesh Patel",
        "voice_hint": "cites specific model versions, parameter counts, benchmark scores (MMLU, HumanEval, SWE-Bench), training compute in FLOPs",
        "angle_hint": "frontier-model releases, benchmark leapfrogs, safety/alignment breakthroughs, training-compute economics, agent frameworks, open-vs-closed weight wars",
    },
    {
        "category": "Reddit / X",
        "interest": "Internet Drama",
        "rail": "Reddit / X Native",
        "query": "India viral trending Twitter X Reddit hashtag drama controversy this week",
        "fallback_query": "India trending viral this week hashtag",
        "persona": "Chronically-online Reddit / X native — refreshes r/IndianDankMemes, r/india, r/librandu, r/IndiaSpeaks, X India-Twitter hashtags. Knows every micro-drama before it hits mainstream media.",
        "search_hint": "r/india, r/IndianDankMemes, X India trending, r/librandu, r/IndiaSpeaks, Reddit trending",
        "voice_hint": "screenshots-first, cites the exact thread, quote-tweet count, upvote ratio, and the reply-guy handle; casual Hinglish OK",
        "angle_hint": "why-is-this-trending explainers, quote-tweet dunk sequences, cancellation timelines, meme provenance, algorithm-boost mechanics",
    },
    {
        "category": "Instagram / Reels",
        "interest": "Creator Economy",
        "rail": "Reels Curator",
        "query": "India Instagram Reels viral creator influencer this week trending",
        "fallback_query": "India Instagram creator viral trending this week",
        "persona": "Reels-tab addict who reads Social Samosa, IndianCreatorHub, TheYouthMag, Homegrown. Tracks Meta creator payouts, brand-deal fees, follower-to-money ratios, and Reels-algorithm quirks.",
        "search_hint": "Social Samosa, Homegrown, TheYouthMag, IndianCreatorHub, Man's World India creator, Vogue India creator",
        "voice_hint": "cites the creator handle, follower count, brand-deal fee in lakh, view count in millions, format of the reel (trend, POV, day-in-life)",
        "angle_hint": "creator brand-deal economics, Reels-algorithm shifts, cross-platform migration (Reels vs YouTube Shorts), Bharat-tier creator boom",
    },
    {
        "category": "Culture",
        "interest": "Urban India Culture",
        "rail": "Gen-Z Reporter",
        "query": "India Gen Z youth culture trend viral urban cafe experience this week",
        "fallback_query": "India youth culture trend this week Bengaluru Mumbai Delhi",
        "persona": "Gen-Z / young-millennial reporter — reads Homegrown, Vogue India, GQ India, Fifty Two, LBB, Man's World India. Tracks creator-economy earnings, dating-app dynamics, cafe-scene expansion.",
        "search_hint": "Homegrown, LBB Local, Vogue India, GQ India, Fifty Two, TheYouthMag, Man's World India",
        "voice_hint": "cites the exact city (Bengaluru Koramangala, Mumbai Bandra, Delhi HKV), the creator's follower count, brand-deal fee, cafe / brand name",
        "angle_hint": "creator earnings, dating-app funnel data, cafe / D2C brand rollouts in metros, Tier-2 cultural exports, workation cities",
    },
    {
        "category": "Music",
        "interest": "Indian Music",
        "rail": "Music Head",
        "query": "India music release song artist viral this week hip hop pop Bollywood indie",
        "fallback_query": "India music new release song viral this week",
        "persona": "Music head — reads Pitchfork, Rolling Stone India, Homegrown music, Radio City India, Spotify RapCaviar India, r/IndianMusicRadio. Tracks Indian hip-hop scene, Bollywood-vs-indie streaming, indie-label economics.",
        "search_hint": "Rolling Stone India, Homegrown music, Radio City India, Spotify Charts India, r/IndianMusicRadio, Pitchfork",
        "voice_hint": "cites the producer credit, the sample source, the label, the streaming numbers on Spotify India / JioSaavn, YouTube views in millions",
        "angle_hint": "Indian hip-hop scene dynamics, indie-label economics, streaming payouts vs live/gig economics, Bollywood vs indie dominance, DHH / Divine / Seedhe Maut moves",
    },
]


# In-memory cache: hot cards freshness ~20 min
_HOT_CACHE: dict = {"cards": None, "ts": 0.0}
_HOT_CACHE_TTL_SEC = 20 * 60


async def _tavily_news_light(query: str) -> List[dict]:
    """Lightweight news search — 5 results, last 7 days, topic=news."""

    def _do():
        try:
            return tavily.search(
                query=query,
                search_depth="basic",
                topic="news",
                max_results=5,
                days=7,
                include_answer=False,
                include_raw_content=False,
            ).get("results", [])
        except Exception:
            return []

    return await asyncio.to_thread(_do)


async def _tavily_general_light(query: str) -> List[dict]:
    """Fallback general web search when news search is empty."""

    def _do():
        try:
            return tavily.search(
                query=query,
                search_depth="basic",
                topic="general",
                max_results=5,
                include_answer=False,
                include_raw_content=False,
            ).get("results", [])
        except Exception:
            return []

    return await asyncio.to_thread(_do)


HOT_CARD_SYSTEM = """You produce ONE ultra-engaging hot-topic insight card for Insight Swipe's Landing "Hot Now" feed.

READER: 20–30 year-old, urban Indian, chronically-online, reads NYRB / Stratechery / Deuxmoi / Stratechery / r/india / ThePrint. They already know the surface news. They open this app to find the ONE non-obvious thing about it that everyone else missed.

═══ NON-NEGOTIABLE QUALITY BAR ═══

Every single card MUST satisfy ALL of these:

1. **Non-obvious payoff, not headline fact.** Reject "X resigned" or "Y raised $100M". Only ship the *mechanism*, the hidden data point, the second-order consequence, the counter-intuitive comparison, or the historical parallel that reframes the story. If a smart friend already knows it after skimming Twitter for 3 minutes, DON'T write it.

2. **Contain a HARD SPECIFIC** grounded in the sources: a number, %, ₹/$ figure, exact date, named person, named regulation, named study, named company. Vague adjectives ("huge", "several", "significant") are banned.

3. **Punchy first sentence** — no throat-clearing. Start with the striking fact.

4. **HARD LENGTH CAP: 2 sentences, max 320 characters combined.** Punchy, not essayistic. If the mechanism needs more words, tighten the sentences. First sentence hits the mechanism, second (optional) delivers the specific.

5. **Persona voice.** Use the AUDIENCE PERSONA block below — their jargon, their references, their taste. If jargon is technical, in-line explain it.

6. **Never fabricate primary facts** (dates, names, ₹/$). If a source doesn't say it, don't write it — pick a different angle instead.

═══ HEADLINE + TOPIC RULES ═══

`headline`: ≤ 8 words, hook-driven. NEVER restate the surface news ("Minister resigns"). Compress the *insight* into the hook ("Resignation as pressure-valve, not policy change").

`topic`: 3–7 words describing the rabbit-hole this card opens (used to launch the deep-research flow when the user picks the card). Should be specific enough that a research LLM can dig further.

Return STRICT JSON: {"headline": "...", "text": "...", "topic": "...", "source_index": <int|null>}"""


async def _synth_hot_card_from_sources(bp: dict, results: List[dict]) -> Optional[dict]:
    """LLM synthesis given a persona blueprint + fetched sources."""
    if not results:
        return None
    lines = [f"CATEGORY: {bp['category']} · TARGET: young urban Indians"]
    for i, r in enumerate(results[:5], 1):
        title = r.get("title", "")
        url = r.get("url", "")
        content = (r.get("content") or "")[:600]
        lines.append(f"[{i}] {title}\n   URL: {url}\n   SNIPPET: {content}\n")
    ctx = "\n".join(lines)

    system = (
        HOT_CARD_SYSTEM
        + f"\n\nAUDIENCE PERSONA: {bp['persona']}"
        + f"\nVOICE / TONE: {bp['voice_hint']}"
        + f"\nANGLES TO PRIORITISE: {bp['angle_hint']}"
    )

    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=f"hot-{bp['category'].lower().replace(' ', '-')}-{uuid.uuid4()}",
        system_message=system,
    ).with_model(LLM_PROVIDER, LLM_MODEL)

    try:
        raw = await chat.send_message(
            UserMessage(text=ctx + "\n\nReturn exactly one card as strict JSON.")
        )
    except Exception:
        logging.exception("hot card LLM failed for %s", bp["category"])
        return None

    text = raw.strip().strip("`")
    if text.startswith("json"):
        text = text[4:].strip()
    if "{" in text:
        text = text[text.find("{") : text.rfind("}") + 1]
    try:
        parsed = json.loads(text)
    except Exception:
        logging.warning("hot card JSON parse fail: %s", text[:200])
        return None

    headline = (parsed.get("headline") or "").strip()
    body = (parsed.get("text") or "").strip()
    topic = (parsed.get("topic") or "").strip()
    if not headline or not body:
        return None
    if not topic:
        topic = headline

    src_url = None
    src_title = None
    idx = parsed.get("source_index")
    if isinstance(idx, int) and 1 <= idx <= len(results):
        src_url = results[idx - 1].get("url")
        src_title = results[idx - 1].get("title")
    elif results:
        src_url = results[0].get("url")
        src_title = results[0].get("title")

    return {
        "id": str(uuid.uuid4()),
        "topic": topic,
        "headline": headline,
        "text": body,
        "category": bp["category"],
        "rail": bp["rail"],
        "interest": bp["interest"],
        "source_url": src_url,
        "source_title": src_title,
    }


async def _synth_hot_card_llm_only(bp: dict) -> Optional[dict]:
    """Last-resort fallback: LLM-only card from the persona's core knowledge — used
    when Tavily returns nothing for both the primary and fallback query. Guarantees
    that no slot returns empty."""
    system = (
        HOT_CARD_SYSTEM
        + f"\n\nAUDIENCE PERSONA: {bp['persona']}"
        + f"\nVOICE / TONE: {bp['voice_hint']}"
        + f"\nANGLES TO PRIORITISE: {bp['angle_hint']}"
        + "\n\nNOTE: No live sources available — draw from the persona's core, well-known "
        "reference knowledge (established facts, publicly known data). Do NOT invent "
        "very recent (this-week) news. Prefer an evergreen non-obvious mechanism."
    )
    user = f"CATEGORY: {bp['category']}\nPRODUCE ONE INSIGHT CARD (strict JSON only) that a young urban Indian would find non-obvious. Use only widely-known references (no fabricated numbers)."

    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=f"hot-fb-{bp['category'].lower().replace(' ', '-')}-{uuid.uuid4()}",
        system_message=system,
    ).with_model(LLM_PROVIDER, LLM_MODEL)
    try:
        raw = await chat.send_message(UserMessage(text=user))
    except Exception:
        return None
    text = raw.strip().strip("`")
    if text.startswith("json"):
        text = text[4:].strip()
    if "{" in text:
        text = text[text.find("{") : text.rfind("}") + 1]
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    headline = (parsed.get("headline") or "").strip()
    body = (parsed.get("text") or "").strip()
    topic = (parsed.get("topic") or "").strip() or headline
    if not headline or not body:
        return None
    return {
        "id": str(uuid.uuid4()),
        "topic": topic,
        "headline": headline,
        "text": body,
        "category": bp["category"],
        "rail": bp["rail"],
        "interest": bp["interest"],
        "source_url": None,
        "source_title": None,
    }


async def _generate_one_hot_card(bp: dict, blueprint_index: int = 0) -> Optional[dict]:
    """Blueprint runner with 3-tier fallback so no slot returns empty:
       1) Primary news search + LLM synthesis
       2) Fallback broader query + LLM synthesis
       3) LLM-only from persona's core knowledge
    Only returns None if all 3 tiers fail (LLM + network both dead).
    Always stamps `blueprint_index` on the returned card so the frontend
    can render in canonical order regardless of SSE arrival order."""
    # Tier 1: primary news search
    results = await _tavily_news_light(bp["query"])
    card = await _synth_hot_card_from_sources(bp, results)
    if not card:
        # Tier 2: fallback broader query — try general search too
        fb_query = bp.get("fallback_query") or bp["query"]
        results = await _tavily_news_light(fb_query)
        if not results:
            results = await _tavily_general_light(fb_query)
        card = await _synth_hot_card_from_sources(bp, results)
    if not card:
        # Tier 3: LLM-only knowledge card
        card = await _synth_hot_card_llm_only(bp)
    if card:
        card["blueprint_index"] = blueprint_index
    return card


@api_router.get("/insights/hot", response_model=HotResponse)
async def hot_topics():
    """Curated 'Hot Now' feed for the Landing page.

    Priority order:
      1. Newsroom-generated cards from db.hot_now_cards (last 30h) — the 7-stage
         multi-agent pipeline output. This is the intended production path.
      2. Legacy blueprint-based fan-out (only as fallback if newsroom has never
         run OR its output has aged out).
    """
    # ── 1. Try newsroom cache first ──
    cutoff = datetime.now(timezone.utc).timestamp() - 30 * 3600
    nr_docs = (
        await db.hot_now_cards.find({}, {"_id": 0}).sort("published_at", -1).to_list(50)
    )
    fresh_nr = [
        d
        for d in nr_docs
        if datetime.fromisoformat(d["published_at"]).timestamp() >= cutoff
    ]
    if fresh_nr:
        # Keep the latest run only (dedupe by run_id: prefer most recent run_stamp)
        latest_run_stamp = max(d.get("run_stamp", d["published_at"]) for d in fresh_nr)
        latest = [
            d
            for d in fresh_nr
            if d.get("run_stamp", d["published_at"]) == latest_run_stamp
        ]
        cards_out = []
        persona_to_category = {
            "gossip_girl": "Culture",
            "startup_founder": "Startups",
            "seed_investor": "Business",
            "policy_wonk": "Geopolitics",
            "developer": "Tech",
            "contrarian_economist": "Markets",
            "culture_analyst": "Culture",
            "india_insider": "India",
        }
        for i, doc in enumerate(latest[:12]):
            if not doc.get("cards"):
                continue
            first = doc["cards"][0]
            src_url = first.get("source_url") or (doc.get("sources") or [None])[0]
            cards_out.append(
                {
                    "id": doc.get("id") or str(uuid.uuid4()),
                    "topic": doc["topic"],
                    "headline": first.get("headline", ""),
                    "text": first.get("text", ""),
                    "category": persona_to_category.get(
                        doc.get("persona_key", ""), "Insight"
                    ),
                    "rail": doc.get("persona_label", "Editor"),
                    "interest": doc.get("persona_label", "Editor"),
                    "source_url": src_url,
                    "source_title": first.get("source_title")
                    or doc.get("angle", {}).get("title", ""),
                    "blueprint_index": i,
                }
            )
        if cards_out:
            return HotResponse(
                cards=[HotCard(**c) for c in cards_out],
                generated_at=latest_run_stamp,
            )

    # ── 2. Legacy fan-out fallback ──
    import time

    now = time.time()
    cached = _HOT_CACHE.get("cards")
    if cached and (now - _HOT_CACHE.get("ts", 0)) < _HOT_CACHE_TTL_SEC:
        return HotResponse(
            cards=[HotCard(**c) for c in cached],
            generated_at=_HOT_CACHE.get("generated_at", ""),
        )

    # Fan out ALL 12 blueprints in parallel
    results = await asyncio.gather(
        *[_generate_one_hot_card(bp, i) for i, bp in enumerate(HOT_BLUEPRINTS)],
        return_exceptions=False,
    )
    cards = [c for c in results if c]
    # Canonical order: sort by blueprint_index so Politics (Pradhan) is always first.
    cards.sort(key=lambda c: c.get("blueprint_index", 0))

    if not cards:
        raise HTTPException(
            status_code=503,
            detail="Hot feed temporarily unavailable — please try again in a minute.",
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    _HOT_CACHE["cards"] = cards
    _HOT_CACHE["ts"] = now
    _HOT_CACHE["generated_at"] = generated_at

    return HotResponse(cards=[HotCard(**c) for c in cards], generated_at=generated_at)


@api_router.get("/insights/hot/stream")
async def hot_topics_stream(fresh: int = 0):
    """SSE variant of /insights/hot — emits each card as its blueprint
    finishes so the Landing page can progressively populate the deck
    instead of waiting for all 12 to finish.

    Query params:
        fresh=1  Bypass the 20-min in-memory cache and regenerate.
    """
    import time

    async def event_gen():
        yield _sse({"phase": "start", "total": len(HOT_BLUEPRINTS)})

        # ── Preferred path: stream newsroom-generated cards (always, even on
        # regenerate). Regenerate now rotates the deck rather than blocking on
        # a slow LLM fan-out — cards feel instant. A background newsroom run
        # keeps the buffer refreshed. ──
        try:
            cutoff = datetime.now(timezone.utc).timestamp() - 30 * 3600
            nr_docs = (
                await db.hot_now_cards.find({}, {"_id": 0})
                .sort("published_at", -1)
                .to_list(80)
            )
            fresh_nr = [
                d
                for d in nr_docs
                if datetime.fromisoformat(d["published_at"]).timestamp() >= cutoff
            ]
            if fresh_nr:
                latest_stamp = max(
                    d.get("run_stamp", d["published_at"]) for d in fresh_nr
                )
                latest = [
                    d
                    for d in fresh_nr
                    if d.get("run_stamp", d["published_at"]) == latest_stamp
                ]
                # On regenerate, rotate: skip the first N and cycle through
                if fresh and len(latest) > 0:
                    import random as _random

                    _random.shuffle(latest)
                persona_to_category = {
                    "gossip_girl": "Culture",
                    "startup_founder": "Startups",
                    "seed_investor": "Business",
                    "policy_wonk": "Geopolitics",
                    "developer": "Tech",
                    "contrarian_economist": "Markets",
                    "culture_analyst": "Culture",
                    "india_insider": "India",
                }
                yield _sse(
                    {"phase": "meta", "source": "newsroom", "run_stamp": latest_stamp}
                )
                for i, doc in enumerate(latest[:12]):
                    if not doc.get("cards"):
                        continue
                    first = doc["cards"][0]
                    card = {
                        "id": doc.get("id") or str(uuid.uuid4()),
                        "topic": doc["topic"],
                        "headline": first.get("headline", ""),
                        "text": first.get("text", ""),
                        "category": persona_to_category.get(
                            doc.get("persona_key", ""), "Insight"
                        ),
                        "rail": doc.get("persona_label", "Editor"),
                        "interest": doc.get("persona_label", "Editor"),
                        "source_url": first.get("source_url")
                        or (doc.get("sources") or [None])[0],
                        "source_title": first.get("source_title")
                        or doc.get("angle", {}).get("title", ""),
                        "blueprint_index": i,
                    }
                    yield _sse({"phase": "card", "card": card})
                yield _sse(
                    {
                        "phase": "done",
                        "generated_at": latest_stamp,
                        "source": "newsroom",
                    }
                )
                # Fire a background newsroom run on regenerate so tomorrow's
                # deck is fresh — user never waits for it.
                if fresh and not _NEWSROOM_STATE.get("running"):
                    asyncio.create_task(_do_newsroom_run())
                return
        except Exception:
            logging.exception("hot/stream newsroom read failed; falling back")

        # Fresh cache → emit everything at once then done (unless caller
        # explicitly asked for a regenerated batch).
        now = time.time()
        cached = _HOT_CACHE.get("cards")
        if (
            not fresh
            and cached
            and (now - _HOT_CACHE.get("ts", 0)) < _HOT_CACHE_TTL_SEC
        ):
            for c in cached:
                yield _sse({"phase": "card", "card": c})
            yield _sse(
                {"phase": "done", "generated_at": _HOT_CACHE.get("generated_at", "")}
            )
            return

        # Otherwise fan out blueprints and stream results as they finish
        tasks = [
            asyncio.create_task(_generate_one_hot_card(bp, i))
            for i, bp in enumerate(HOT_BLUEPRINTS)
        ]
        collected: List[dict] = []
        pending = set(tasks)
        try:
            while pending:
                done_set, pending = await asyncio.wait(
                    pending, timeout=6, return_when=asyncio.FIRST_COMPLETED
                )
                if not done_set:
                    # keep-alive comment so proxies don't idle-kill
                    yield ": keep-alive\n\n"
                    continue
                for t in done_set:
                    try:
                        card = t.result()
                    except Exception:
                        card = None
                    if card:
                        collected.append(card)
                        yield _sse({"phase": "card", "card": card})

            # Populate cache
            if collected:
                generated_at = datetime.now(timezone.utc).isoformat()
                _HOT_CACHE["cards"] = collected
                _HOT_CACHE["ts"] = time.time()
                _HOT_CACHE["generated_at"] = generated_at
                yield _sse({"phase": "done", "generated_at": generated_at})
            else:
                yield _sse(
                    {
                        "phase": "error",
                        "message": "Hot feed temporarily unavailable — please retry.",
                    }
                )
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============ Legacy endpoints ============
# ============================================================================
# Newsroom — admin endpoints + daily scheduler
# ============================================================================


class NewsroomRunResponse(BaseModel):
    ok: bool
    signals: Optional[int] = None
    kept: Optional[int] = None
    published: Optional[int] = None
    topics: Optional[List[dict]] = None
    reason: Optional[str] = None


_NEWSROOM_STATE: dict = {
    "running": False,
    "last_result": None,
    "last_started": None,
    "last_finished": None,
}


async def _do_newsroom_run() -> dict:
    if _NEWSROOM_STATE.get("running"):
        return {"ok": False, "reason": "a run is already in progress"}
    _NEWSROOM_STATE["running"] = True
    _NEWSROOM_STATE["last_started"] = datetime.now(timezone.utc).isoformat()
    try:
        result = await _run_newsroom(
            tavily=tavily, db=db, llm_key=EMERGENT_LLM_KEY, keep_n=15
        )
    except Exception as e:
        logging.exception("newsroom run failed")
        result = {"ok": False, "reason": str(e)}
    finally:
        _NEWSROOM_STATE["running"] = False
        _NEWSROOM_STATE["last_finished"] = datetime.now(timezone.utc).isoformat()
        _NEWSROOM_STATE["last_result"] = result
    return result


@api_router.post("/admin/newsroom/run", response_model=NewsroomRunResponse)
async def admin_newsroom_run(background: bool = True):
    """Trigger the 7-stage newsroom pipeline. Default: runs in background and
    returns immediately with the current state — poll /admin/newsroom/status
    for the result. Pass ?background=false to wait for the run to finish."""
    if background:
        asyncio.create_task(_do_newsroom_run())
        return NewsroomRunResponse(ok=True, reason="started in background")
    result = await _do_newsroom_run()
    return NewsroomRunResponse(**result)


@api_router.get("/admin/newsroom/status")
async def admin_newsroom_status():
    latest_run = None
    doc = await db.hot_now_cards.find_one({}, {"_id": 0}, sort=[("published_at", -1)])
    if doc:
        latest_run = {
            "run_stamp": doc.get("run_stamp") or doc.get("published_at"),
            "topic": doc.get("topic"),
            "persona": doc.get("persona_label"),
        }
    total_topics = await db.hot_now_cards.count_documents({})
    total_runs = await db.newsroom_runs.count_documents({})
    return {
        "running": _NEWSROOM_STATE.get("running", False),
        "last_started": _NEWSROOM_STATE.get("last_started"),
        "last_finished": _NEWSROOM_STATE.get("last_finished"),
        "last_result": _NEWSROOM_STATE.get("last_result"),
        "latest_cached_topic": latest_run,
        "total_topics_cached": total_topics,
        "total_runs_recorded": total_runs,
        "personas_available": list(_NR_PERSONAS.keys()),
    }


@api_router.get("/admin/newsroom/personas")
async def admin_newsroom_personas():
    """Full persona briefs — used by the admin page to render 'who is this
    persona, deeply' cards for the operator."""
    return {"personas": _NR_PERSONAS}


@api_router.get("/admin/newsroom/runs")
async def admin_newsroom_runs(limit: int = 20):
    """List of past newsroom runs (summary only, no full trace)."""
    cursor = (
        db.newsroom_runs.find(
            {},
            {
                "_id": 0,
                "run_id": 1,
                "started_at": 1,
                "finished_at": 1,
                "total_elapsed_sec": 1,
                "outcome": 1,
                "published_count": 1,
                "keep_n": 1,
                "stages.signal_scout.total_raw_signals": 1,
                "stages.signal_scout.clusters_formed": 1,
                "stages.topic_filter.kept_count": 1,
                "stages.topic_filter.scored_count": 1,
            },
        )
        .sort("started_at", -1)
        .limit(limit)
    )
    runs = await cursor.to_list(limit)
    for r in runs:
        stages = r.get("stages", {}) or {}
        scout = stages.get("signal_scout", {}) or {}
        flt = stages.get("topic_filter", {}) or {}
        r["summary"] = {
            "raw_signals": scout.get("total_raw_signals"),
            "clusters": scout.get("clusters_formed"),
            "scored": flt.get("scored_count"),
            "kept": flt.get("kept_count"),
            "published": r.get("published_count", 0),
        }
        r.pop("stages", None)
    return {"runs": runs}


@api_router.get("/admin/newsroom/runs/{run_id}")
async def admin_newsroom_run_detail(run_id: str):
    """Full trace of a single run — every stage, every signal, every angle,
    every score, every card. This is what powers the /admin factory-floor view."""
    doc = await db.newsroom_runs.find_one({"run_id": run_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Run not found")
    return doc


# ── Back Story: streaming plain-English explanation of a Hot Now card ──
class BackStoryReq(BaseModel):
    topic: str
    headline: str
    text: str
    source_title: Optional[str] = None
    source_url: Optional[str] = None


@api_router.post("/insights/hot/backstory/stream")
async def hot_backstory_stream(req: BackStoryReq):
    """Stream a Back Story: a 4-paragraph plain-English explainer of the insight
    with what/who/why/when + a follow-up prompt suggestion. Uses the existing
    LLM chat under the hood so it feels fast."""
    from emergentintegrations.llm.chat import LlmChat, UserMessage

    system = (
        "You write short, plain-English BACK STORY explainers for insight cards on a mobile app. "
        "Your reader is a curious 20-30 year old Indian who wants to understand the FULL story behind a short insight card. "
        "Keep it short but complete — 4 paragraphs, each 2-3 sentences, no jargon.\n\n"
        "STRUCTURE (use these exact 4 mini-headings, one line each, prefixed with — ):\n"
        "— What actually happened\n"
        "— Who is involved (name the key entities plainly)\n"
        "— Why it matters right now\n"
        "— The historic chronology (only 2-3 short lines — only if the story has a real historic thread; otherwise write 'This is a fresh, one-off shift — no historic chain to trace yet.')\n\n"
        "Rules:\n"
        "- No insider jargon. If you use a term like SEBI/RBI/EBITDA/IPO/S-1, explain in 3-4 words inline.\n"
        "- Name real entities the reader might not know (e.g., 'Hermès, the ₹200 billion French luxury brand').\n"
        "- Total length: ~180-250 words. Not longer.\n"
        "- End with the actual insight, then invite the reader to ask a follow-up.\n"
        "- Write in one continuous flow — Markdown '— heading' lines act as tiny section markers."
    )
    user_msg = (
        f"TOPIC: {req.topic}\n\nCARD HEADLINE: {req.headline}\n\nCARD INSIGHT: {req.text}\n\n"
        f"SOURCE: {req.source_title or ''} {req.source_url or ''}\n\n"
        "Write the back-story explainer now."
    )
    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=f"backstory-{uuid.uuid4()}",
        system_message=system,
    ).with_model("openai", "gpt-5.6-luna")

    async def gen():
        yield _sse({"phase": "start"})
        try:
            resp = await chat.send_message(UserMessage(text=user_msg))
            # Stream in chunks so the modal renders progressively
            chunk_size = 40
            for i in range(0, len(resp), chunk_size):
                yield _sse({"phase": "chunk", "text": resp[i : i + chunk_size]})
                await asyncio.sleep(0.02)
            yield _sse({"phase": "done"})
        except Exception as e:
            yield _sse({"phase": "error", "error": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream")


class BackStoryChatReq(BaseModel):
    topic: str
    headline: str
    text: str
    backstory: Optional[str] = None
    question: str


@api_router.post("/insights/hot/backstory/chat/stream")
async def hot_backstory_chat_stream(req: BackStoryChatReq):
    """User follow-up chat within the Back Story sheet."""
    from emergentintegrations.llm.chat import LlmChat, UserMessage

    system = (
        "You are a smart friend explaining an insight card to a curious 20-30 year old Indian in plain English. "
        "You have full context on the card + the back story that was just explained. Answer their follow-up in "
        "2-4 short paragraphs. No jargon. Cite specific entities/numbers/dates when relevant. If you don't know "
        "something for sure, say so honestly — never fabricate specific numbers."
    )
    user_msg = (
        f"CARD TOPIC: {req.topic}\nHEADLINE: {req.headline}\nCARD INSIGHT: {req.text}\n\n"
        + (f"BACK STORY CONTEXT:\n{req.backstory}\n\n" if req.backstory else "")
        + f"USER QUESTION: {req.question}"
    )
    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=f"backstory-chat-{uuid.uuid4()}",
        system_message=system,
    ).with_model("openai", "gpt-5.6-luna")

    async def gen():
        yield _sse({"phase": "start"})
        try:
            resp = await chat.send_message(UserMessage(text=user_msg))
            chunk_size = 40
            for i in range(0, len(resp), chunk_size):
                yield _sse({"phase": "chunk", "text": resp[i : i + chunk_size]})
                await asyncio.sleep(0.02)
            yield _sse({"phase": "done"})
        except Exception as e:
            yield _sse({"phase": "error", "error": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream")


# ── Single-card regenerate — swaps ONE Hot Now card without touching the rest ──
class RegenSingleReq(BaseModel):
    topic: str
    headline: str
    text: str
    source_title: Optional[str] = None
    steer: Optional[str] = ""


@api_router.post("/insights/hot/regenerate-single")
async def hot_regenerate_single(req: RegenSingleReq):
    """Regenerate a single Hot Now card on the same topic. Optional `steer`
    (voice or typed) reshapes the angle. Fast — single LLM call, no Tavily."""
    from emergentintegrations.llm.chat import LlmChat, UserMessage

    system = (
        "You are the DruvInsight card writer. Given an existing Hot Now insight "
        "card and (optionally) a user voice/text steer, produce a NEW single "
        "insight card on the SAME topic — with a different angle or fresh detail. "
        "Rules: 2-3 short sentences (~40-70 words), catchy 6-8 word headline, "
        "plain English (no jargon without inline explanation), starts with a "
        "concrete fact naming a real entity + date/number, ends leaving the "
        "reader wanting more.\n\n"
        "Return STRICT JSON only:\n"
        '{"headline": "...", "text": "..."}'
    )
    user_msg = (
        f"TOPIC: {req.topic}\n"
        f"OLD HEADLINE: {req.headline}\n"
        f"OLD TEXT: {req.text}\n"
        + (f"USER STEER: {req.steer}\n" if req.steer else "")
        + "Write the new card as strict JSON."
    )
    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=f"regen-single-{uuid.uuid4()}",
        system_message=system,
    ).with_model("openai", "gpt-5.6-luna")
    try:
        raw = await chat.send_message(UserMessage(text=user_msg))
        s = raw.strip().strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()
        if "{" in s:
            s = s[s.find("{") : s.rfind("}") + 1]
        data = json.loads(s)
        new_card = {
            "id": str(uuid.uuid4()),
            "topic": req.topic,
            "headline": data.get("headline", "").strip(),
            "text": data.get("text", "").strip(),
            "category": "Insight",
            "rail": "Editor",
            "interest": "Editor",
            "source_url": None,
            "source_title": req.source_title,
            "blueprint_index": 0,
        }
        return {"card": new_card}
    except Exception as e:
        logging.exception("regenerate-single failed")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/insights/hot/topic/{topic_id}")
async def hot_topic_full_sequence(topic_id: str):
    """Return the full 12-card sequence for a Hot Now topic (used when the
    user double-taps a hero card and enters the swipe feed)."""
    doc = await db.hot_now_cards.find_one({"id": topic_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Topic not found")
    cards = doc.get("cards", [])
    # Shape each card into the standard Insight response the feed already uses
    insights = []
    for c in cards:
        insights.append(
            {
                "id": str(uuid.uuid4()),
                "topic": doc["topic"],
                "headline": c.get("headline", ""),
                "text": c.get("text", ""),
                "source_url": c.get("source_url") or (doc.get("sources") or [None])[0],
                "source_title": c.get("source_title")
                or doc.get("angle", {}).get("title", ""),
                "rail": doc.get("persona_label"),
                "interest": doc.get("persona_label"),
                "created_at": doc.get("published_at", ""),
            }
        )
    return {
        "session_id": f"hot-{topic_id}",
        "topic": doc["topic"],
        "angle": doc.get("angle"),
        "persona": doc.get("persona_label"),
        "insights": insights,
    }


# ── Daily scheduler ──
# Runs the newsroom once every 24h. First trigger is on server startup after a
# short delay (so cold-start deploys get initial content), then every 24h.
_SCHEDULER_STARTED = False


async def _newsroom_daily_loop():
    # Initial delay: 60s after boot so we don't slam startup
    await asyncio.sleep(60)
    while True:
        try:
            fresh_exists = await db.hot_now_cards.find_one(
                {
                    "published_at": {
                        "$gte": (
                            datetime.now(timezone.utc).timestamp() - 20 * 3600
                        ).__str__()
                    }
                },
            )
            # Simpler check: any doc within the last 20h
            cutoff_iso = datetime.now(timezone.utc).isoformat()
            recent = await db.hot_now_cards.find_one({}, sort=[("published_at", -1)])
            needs_run = True
            if recent:
                try:
                    last_ts = datetime.fromisoformat(recent["published_at"]).timestamp()
                    if datetime.now(timezone.utc).timestamp() - last_ts < 20 * 3600:
                        needs_run = False
                except Exception:
                    pass
            if needs_run and not _NEWSROOM_STATE.get("running"):
                logging.info("[newsroom-scheduler] triggering daily run")
                await _do_newsroom_run()
        except Exception:
            logging.exception("newsroom scheduler tick failed")
        # Sleep 24h before next check
        await asyncio.sleep(24 * 3600)


@app.on_event("startup")
async def _start_newsroom_scheduler():
    global _SCHEDULER_STARTED
    if _SCHEDULER_STARTED:
        return
    _SCHEDULER_STARTED = True
    asyncio.create_task(_newsroom_daily_loop())
    logging.info("[newsroom-scheduler] started")


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
