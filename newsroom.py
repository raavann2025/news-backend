"""
DruvInsight Newsroom — 7-stage multi-agent content factory.

Every agent is a specialized expert with deep, grounded knowledge — not a title
wearing jargon. Every stage records a full trace to db.newsroom_runs so the
/admin page can replay the entire factory floor step-by-step.

Pipeline:
    Signal Scout → Topic Filter → Persona Router → Angle Ideator
        → Angle Critic (loops up to 3x) → Deep Researcher
        → Card Writer → Chief Editor (loops up to 2x) → cache in Mongo
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from emergentintegrations.llm.chat import LlmChat, UserMessage
from tavily import TavilyClient

from research import (ResearchBundle, deep_research,
                      generate_insights_from_bundle)

log = logging.getLogger("newsroom")

LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-5.6-luna"
LLM_MODEL_FAST = "gpt-5.6-luna"


# ============================================================================
# DEEP PERSONAS — not titles, but fully-realized humans with taste
# ============================================================================
# Each persona is written like a hiring brief: what they read, what rewards
# them at work, what makes them stop-scroll, what they text friends about,
# what kind of depth they crave, and what they immediately eye-roll at.
PERSONAS: Dict[str, Dict[str, Any]] = {
    "gossip_girl": {
        "label": "Gossip Girl",
        "reader": (
            "25-year-old woman in Bombay/Bangalore. Works in marketing, PR, brand strategy, "
            "or luxury retail. Grew up on Vogue Business, Deuxmoi, and Puck. Her Instagram close-friends "
            "list is 40 people. She doesn't just follow celebrities — she reads them like a business "
            "analyst reads S-1 filings, tracking whose ring is a real Cartier vs a rented Pomellato, "
            "which vacation is a paid brand trip vs an actual holiday, whose PR team just switched agencies."
        ),
        "reads": [
            "Deuxmoi + Deux U on Instagram",
            "Puck (Matt Belloni, Lauren Sherman)",
            "Vogue Business",
            "Business of Fashion — Imran Amed & Lauren Indvik",
            "Puck's What I'm Hearing podcast",
            "Page Six + The Cut",
            "Chris Rojek on celebrity sociology (undergrad reading)",
            "Emily Sundberg's Feed Me newsletter",
            "Rachel Karten's Link in Bio",
            "Popbitch (UK gossip with receipts)",
            "r/Fauxmoi + r/deuxmoi",
        ],
        "work_incentives": (
            "Rewarded for spotting cultural moments 3 weeks before they hit mainstream. Her manager "
            "trusts her because she called the Bollywood-designer wedding pivot two months before it "
            "went viral. She's collecting proof-of-taste to move from junior brand to celebrity accounts."
        ),
        "wow_factors": [
            "The actual receipts behind the story (invoices, cap tables, brand-deal fees)",
            "'This wedding cost more than a Series A' framings with real numbers",
            "Who's rising, who's falling in the celebrity power law — with proof",
            "The PR play behind the public breakdown",
            "'You thought it was organic — it was paid' reveals",
        ],
        "shareability_triggers": (
            "Anything that becomes a 3-minute voice note to her best friend at 11pm. She shares stories "
            "where the reveal makes her friends go 'WAIT what' — not stories that confirm what everyone "
            "already knows. Also: celebrity-as-brand teardowns her friends can quote at brunch."
        ),
        "depth_cravings": [
            "the money trail behind a wedding / divorce / label switch",
            "the 5-year career arc reframed with hindsight",
            "the family-office / management-firm behind the star",
            "brand-collab economics (how a swipe-up actually pays)",
            "PR-industrial-complex reveals",
        ],
        "eye_roll_triggers": [
            "generic paparazzi-style 'X spotted at Y' with no analysis",
            "moralistic takes on celebrities' personal choices",
            "obviously-planted PR framed as gossip",
            "recycled Twitter drama with nothing new",
        ],
        "voice": "Punchy, insider-y, slightly conspiratorial. Uses first names. Drops real numbers casually.",
        "search_hint": "Deuxmoi, Puck, Vogue Business, Page Six, Business of Fashion, Emily Sundberg, Feed Me, r/Fauxmoi",
    },
    "startup_founder": {
        "label": "Startup Founder",
        "reader": (
            "28-year-old founder or Head of Growth at an early-stage company. Has read every Paul Graham "
            "essay twice. Listens to Acquired on 2x during commutes. Follows @patio11, @Suhail, "
            "@paulg, @dhh, @nikitabier, @naval on Twitter. Reads Stratechery every Monday morning like "
            "religion. Would rather have 1 hour with Ben Thompson than a business degree."
        ),
        "reads": [
            "Stratechery (Ben Thompson) — daily",
            "The Information + The Diff (Byrne Hobart)",
            "Acquired (Ben Gilbert + David Rosenthal) — 4-hour episodes on repeat",
            "Business Breakdowns (Matt Russell)",
            "Y Combinator library + PG essays (yes, all of them)",
            "The Pragmatic Engineer (Gergely Orosz)",
            "Not Boring (Packy McCormick)",
            "Kunal Shah's tweets + Aravind Srinivas's launches",
            "First Round Review, a16z Future, Sequoia Perspectives",
            "SaaStr and Lenny's Newsletter",
            "S-1 filings on decoded terms (Airbnb, DoorDash, Figma, Instacart teardowns)",
        ],
        "work_incentives": (
            "Rewarded when their pattern-matching converts to real capital or a real hire. They're not "
            "trying to sound smart — they're trying to compress 10 years of experience into 3. Every "
            "insight they read is filed against 'can this change how I run my company Monday morning.'"
        ),
        "wow_factors": [
            "A first-principles reframe of something they thought they understood",
            "Concrete unit economics that break their prior model",
            "Timing-of-market analyses ('why now, why not 3 years ago')",
            "Distribution mechanics — the unlock most founders miss",
            "'The founder said X publicly but the S-1 says Y' reveals",
        ],
        "shareability_triggers": (
            "They share into their Slack/WhatsApp group of 6 co-founders when it's a mental-model upgrade, "
            "not news. If they can copy-paste one sentence and their friend says 'holy shit,' it ships."
        ),
        "depth_cravings": [
            "how a specific company found product-market fit (with the failed experiments in between)",
            "aggregation theory / bundling / distribution reframes",
            "the 'boring underneath' of a hyped company (economics, not hype)",
            "org-design failures and recoveries at scale",
            "the incentive structure behind a market shift",
        ],
        "eye_roll_triggers": [
            "'X company raised $Y' without market context",
            "recycled PG quotes with no new synthesis",
            "hero-worship profiles of founders",
            "posts that mistake VC-Twitter consensus for insight",
        ],
        "voice": "Blunt, first-principles, framework-forward. Every claim has a mechanism. No cheerleading.",
        "search_hint": "Stratechery, The Information, Byrne Hobart, Acquired, Business Breakdowns, YC library, S-1, Not Boring, Lenny's Newsletter",
    },
    "seed_investor": {
        "label": "Seed Investor",
        "reader": (
            "32-year-old angel investor or Principal at a seed fund. Comes from 4 years operating, now 4 "
            "years investing. Job is pattern-matching across ~800 pitches a year to find the 8 checks that "
            "matter. Reads investor letters religiously and can quote Fred Wilson from 2007 blog posts. "
            "Every trend she reads is subconsciously graded on 'is this a market big enough for a fund-returner.'"
        ),
        "reads": [
            "Fred Wilson (AVC) archives — 20 years deep",
            "USV (Union Square Ventures) thesis posts",
            "Sequoia Perspectives (data reports + LP letters)",
            "Not Boring (Packy) + The Generalist (Mario Gabriele)",
            "The Diff (Byrne Hobart) for capital-flow reads",
            "Chamath's memos + Bill Gurley's older Above the Crowd posts",
            "PitchBook, Crunchbase, CB Insights — the data underneath",
            "SaaStr, ARR benchmarks, Openview's SaaS reports",
            "Elad Gil's High Growth Handbook (mental model reference)",
            "IPO S-1s + Snowflake / Datadog / MongoDB teardowns",
            "Emerging-market investor letters (Kunal Bahl, Nikhil Kamath, Blume, Chiratae LP letters)",
        ],
        "work_incentives": (
            "Rewarded for calling the trend 12–24 months before her fund's investment committee sees it. "
            "Bad calls are forgiven; missed calls are career-ending. Every read is 'is this a signal I "
            "should be underwriting deals against, or noise I should skip.'"
        ),
        "wow_factors": [
            "Second-derivatives: not the trend, but why the trend is accelerating",
            "TAM re-estimations with new math",
            "Cohort economics that flip the received wisdom",
            "The invisible market that's about to bloom (with data)",
            "Comparative company teardowns showing which model wins",
        ],
        "shareability_triggers": (
            "Forwards to her fund's Slack with '👀' when it's a thesis-shaping datum. Shares to LP letters "
            "when it's a portfolio-defensible narrative."
        ),
        "depth_cravings": [
            "market-timing frameworks (why now)",
            "founder pattern-matching (what predicts breakout)",
            "capital-flow analyses (where is dumb money vs smart money going)",
            "distribution moats vs product moats",
            "the exit landscape 5-10 years out",
        ],
        "eye_roll_triggers": [
            "'X% CAGR' with no unit economics",
            "trends that are already priced in on Series B rounds",
            "founder profiles without the failure history",
            "market maps that just relist companies without a thesis",
        ],
        "voice": "Analytical, calm, quantitative. Talks in cohorts, second-derivatives, and 5-year windows.",
        "search_hint": "AVC, USV, Sequoia, Not Boring, The Diff, The Generalist, PitchBook, Crunchbase, SaaStr, Blume, Chiratae",
    },
    "policy_wonk": {
        "label": "Policy Wonk",
        "reader": (
            "30-year-old policy analyst at a think tank, foreign ministry desk officer, or geopolitics-obsessed "
            "consultant. Reads Foreign Affairs like others read Twitter. Owns hardbacks of Kissinger's Diplomacy, "
            "Tooze's Crashed and Shutdown. Would prefer a 90-minute Adam Tooze episode to a movie."
        ),
        "reads": [
            "Foreign Affairs, Foreign Policy magazine",
            "Adam Tooze — Chartbook newsletter + Ones and Tooze podcast",
            "Noah Smith (Noahpinion) for econ-policy synthesis",
            "The Economist — cover to cover, weekly",
            "Financial Times — Alphaville, Martin Wolf, Gideon Rachman",
            "Brookings, Carnegie, RUSI, CFR working papers",
            "Odd Lots (Joe Weisenthal + Tracy Alloway) — for cross-domain econ threads",
            "Zeihan on Geopolitics (skeptically) + Meaghan O'Sullivan on energy",
            "Central-bank statements — FOMC minutes, RBI monetary policy report, ECB press conferences",
            "IMF / World Bank / BIS working papers",
            "Ministry press releases (MEA, MoF), PIB India",
        ],
        "work_incentives": (
            "Rewarded for briefings that survive a red-team by senior colleagues. Bad briefings die from "
            "'we've seen this pattern before, in 1997, and here's what happened' takedowns. So every insight "
            "is stress-tested against 30 years of historical parallel."
        ),
        "wow_factors": [
            "A historical precedent nobody else drew (e.g., 1997 Asian Crisis → today's EM stress)",
            "A specific policy mechanism (sanctions, tariffs, capital controls) explained end-to-end",
            "A trade-flow / capital-flow reversal shown with numbers",
            "A central-bank language shift decoded",
            "A 'here's what happens 6 months out' predictive frame",
        ],
        "shareability_triggers": (
            "Forwards to her ministry Signal group when the framing is defensible. Shares to Twitter when "
            "the insight compresses a debate into one crisp claim."
        ),
        "depth_cravings": [
            "the mechanism behind a sanctions regime (SWIFT, secondary sanctions, licensing)",
            "the history of similar shocks (why does 1971, 1979, 2008 keep repeating)",
            "the domestic-politics-drives-foreign-policy read",
            "commodity flows (energy, food, chips) and chokepoints",
            "central bank psychology under pressure",
        ],
        "eye_roll_triggers": [
            "'nations acting in their interest' with no mechanism",
            "reheated Cold War framings applied to 2025",
            "policy claims without primary-source references",
            "geopolitics writing that ignores demographics and debt",
        ],
        "voice": "Historian's memory, economist's math. Every claim tied to a precedent, a mechanism, or a number.",
        "search_hint": "Foreign Affairs, Adam Tooze, Noah Smith, Odd Lots, Brookings, Carnegie, RUSI, FT, The Economist, IMF, World Bank, RBI, FOMC",
    },
    "developer": {
        "label": "Senior Developer",
        "reader": (
            "27-year-old senior engineer at a scale-up or platform team. Reads Hacker News every morning, "
            "then arxiv-sanity, then their team's Slack. Ships production code that thousands rely on daily. "
            "Reads engineering blogs to steal ideas, not to admire them. Skeptical of every new JS framework."
        ),
        "reads": [
            "Hacker News — front page + comments",
            "arxiv-sanity + arxiv daily for ML papers",
            "The Pragmatic Engineer (Gergely Orosz) — the gold standard for eng org insights",
            "Latent Space (Swyx) for AI-eng deep dives",
            "Engineering blogs: Uber Eng, Stripe Eng, Cloudflare, Netflix Tech Blog, Meta Eng",
            "Julia Evans's zines + Bytesize newsletter",
            "Simon Willison's blog (for LLM-eng ground truth)",
            "Dan Luu's essays (patient, contrarian, well-cited)",
            "Fabrice Bellard's projects + patio11's operator wisdom",
            "The Morning Paper (Adrian Colyer archives)",
            "SRE weekly, PostgreSQL Weekly, TL;DR newsletters",
        ],
        "work_incentives": (
            "Rewarded when their design docs survive senior review. Rewarded when a system they built runs "
            "boringly for 3 years. Not rewarded for hype — no one at their staff review cares that they read "
            "about a trendy framework. They care about mechanism, tradeoffs, and post-mortems."
        ),
        "wow_factors": [
            "Benchmark numbers that break received wisdom",
            "A post-mortem with real diagnostic depth (why Google's outage was actually about X)",
            "A paper that changes how they'd design a system",
            "An implementation trick that saves 10x latency",
            "A 'this is how X actually works underneath' teardown",
        ],
        "shareability_triggers": (
            "Slack DMs a colleague with 'read this, changes how I'd build our thing.' Or posts to team-eng "
            "channel with a one-line takeaway."
        ),
        "depth_cravings": [
            "how a specific system (Kafka, Postgres, DynamoDB) actually works under load",
            "the compiler / runtime / kernel-level explanation of a bug class",
            "distributed-systems tradeoff analyses (consistency, availability, latency)",
            "post-mortem reports from real production incidents",
            "the math behind a paper's headline claim",
        ],
        "eye_roll_triggers": [
            "'AI will change everything' with no benchmark",
            "framework announcements with no benchmark comparison",
            "junior-influencer content wrapped as technical insight",
            "opinion pieces posing as engineering analysis",
        ],
        "voice": "Precise, technical, dry. Cites papers and benchmarks. Skeptical of marketing framing.",
        "search_hint": "Hacker News, arxiv, The Pragmatic Engineer, Latent Space, Simon Willison, Dan Luu, engineering blogs, The Morning Paper",
    },
    "contrarian_economist": {
        "label": "Contrarian Economist",
        "reader": (
            "34-year-old buy-side analyst, macro fund associate, or activist-investor researcher. Grew up on "
            "Matt Levine's Money Stuff (never misses an issue), then FT Alphaville, then Bill Ackman's letters. "
            "Loves it most when the consensus is wrong and she can prove it with an SEC filing footnote."
        ),
        "reads": [
            "Matt Levine — Money Stuff (Bloomberg) — daily, twice",
            "FT Alphaville — Bryce Elder, Robin Wigglesworth",
            "The Economist — Buttonwood, Free Exchange columns",
            "Bloomberg Opinion + Odd Lots podcast",
            "Bill Ackman's activist letters + Third Point's + Elliott Management's",
            "Short-seller reports: Hindenburg, Muddy Waters, Kerrisdale Capital",
            "SEC filings (10-K, 10-Q, 8-K) — she READS them, not skims",
            "GMO's quarterly letters (Jeremy Grantham)",
            "Marc Rubinstein's Net Interest (banking depth)",
            "Barron's + WSJ Heard on the Street",
            "The Big Picture (Barry Ritholtz) archives",
        ],
        "work_incentives": (
            "Rewarded for calling the mispricing that the market later corrects. Punished for consensus takes "
            "that lose money. Reads every insight through 'what is the market pricing wrong here.'"
        ),
        "wow_factors": [
            "A footnoted filing detail that reframes a company",
            "A 'the entire narrative is backwards' contrarian read with data",
            "A short thesis with a specific catalyst and timeline",
            "A cross-market linkage (e.g., what treasury yields imply for equity multiples)",
            "A historical mispricing precedent (LTCM, 2008, GME, SVB, Adani)",
        ],
        "shareability_triggers": (
            "Screenshots to her buy-side WhatsApp group when it's a re-priceable claim. Quotes to the FT "
            "when it's dinner-party worthy."
        ),
        "depth_cravings": [
            "accounting-gymnastics teardowns (revenue recognition, capitalized R&D games)",
            "duration and rates math applied to real portfolios",
            "cap-structure arbitrage explanations",
            "central-bank reaction functions decoded",
            "activist campaigns — the mechanics of proxy fights",
        ],
        "eye_roll_triggers": [
            "consensus narratives repeated without new evidence",
            "TAM-based bull cases without unit economics",
            "'AI will fix earnings' takes",
            "posts that mistake momentum for thesis",
        ],
        "voice": "Dry, funny, quantitative. Style: Matt Levine. Footnote-driven skepticism.",
        "search_hint": "Money Stuff, FT Alphaville, Bloomberg Opinion, Odd Lots, Ackman letters, Hindenburg, Muddy Waters, 10-K filings, GMO, Net Interest",
    },
    "culture_analyst": {
        "label": "Culture Analyst",
        "reader": (
            "26-year-old strategist at a brand, agency, or trends consultancy. Job is to name the culture-shift "
            "6 months before it becomes a NYT trend piece. Reads NYT Styles, Business of Fashion, Business "
            "Insider trends, and 30 Substacks nobody in her family has heard of."
        ),
        "reads": [
            "NYT Styles + The Ethicist column",
            "Business of Fashion — daily brief",
            "Rest of World (RoW) — global consumer culture reporting",
            "Emily Sundberg — Feed Me (New York cultural pulse)",
            "Kyle Chayka — Filterworld + New Yorker essays",
            "Rachel Karten — Link in Bio (social strategy)",
            "Piper Sandler's Taking Stock of Teens (biannual survey — gold standard)",
            "TrendWatching + WGSN + The Business Research Company reports",
            "Ana Andjelic — The Sociology of Business (brand-strategy anthropologist)",
            "Terry Nguyen's Gen Yeet + Kaitlyn Tiffany's Girls newsletter",
            "Andreessen Horowitz's Cultural Currency essays",
        ],
        "work_incentives": (
            "Rewarded when the trend she named 6 months ago becomes the brief her CMO is asking about. "
            "Bad calls are 'you were early, doesn't count.' So she's obsessed with dating the trend precisely."
        ),
        "wow_factors": [
            "A trend named with a specific cohort attached (not 'Gen Z' but '19-22 college women in Tier-2 US')",
            "A quantifiable behavior shift (Piper Sandler-style survey deltas)",
            "The economic driver behind a cultural moment (why this, why now)",
            "A brand that quietly won the trend before anyone else noticed",
            "A cross-market parallel (this happened in Seoul in 2019, now Bombay in 2025)",
        ],
        "shareability_triggers": (
            "DMs into her strategy Slack with a screenshot. Includes it in her weekly deck if the trend is "
            "buyable."
        ),
        "depth_cravings": [
            "consumer-cohort economics (spending shifts by micro-generation)",
            "algorithm-shaped taste (how TikTok/Instagram sorted us into subcultures)",
            "brand-to-cohort fit case studies",
            "the tension between authenticity and manufactured trend",
            "trend-hosting platforms (Depop, RedNote, Discord, Reddit)",
        ],
        "eye_roll_triggers": [
            "'Gen Z is redefining X' without a survey or cohort behind it",
            "trend pieces that mistake influencer output for cultural signal",
            "boomer-written 'young people love X' articles",
            "brand-strategy talk without a customer",
        ],
        "voice": "Anthropologist meets strategist. Names the trend, quantifies it, predicts its trajectory.",
        "search_hint": "NYT Styles, Business of Fashion, Rest of World, Feed Me, Piper Sandler, WGSN, Ana Andjelic, Filterworld, RoW",
    },
    "india_insider": {
        "label": "India Insider",
        "reader": (
            "29-year-old India-based professional — founder, PM, consultant, analyst. Reads The Ken and Morning "
            "Context on paid subs. Follows Nithin Kamath, Kunal Shah, Sajith Pai on Twitter/X. Knows the specific "
            "difference between an RBI Master Circular and an RBI Notification. Explains India to Indians better "
            "than the Western press can."
        ),
        "reads": [
            "The Ken (Rohin Dharmakumar) — paid sub, reads every issue",
            "Morning Context (Ashish K Mishra) — India business depth",
            "The Signal (Zerodha's Nithin Kamath-adjacent) + Nithin's tweets",
            "Kunal Shah's threads + Sajith Pai (India-context essays)",
            "Aparna Chennapragada, Nikhil Kamath tweets",
            "RBI Master Circulars, SEBI orders, MoF budget speeches",
            "Bloomberg Quint (BQ Prime) + Moneycontrol Pro",
            "The Wire, Newslaundry, Scroll for policy critique",
            "PRS Legislative Research — bill tracking, gold standard",
            "LiveLaw and Bar & Bench for legal-tech reads",
            "Sajith Pai's 'India1/India2/India3' framework essays",
        ],
        "work_incentives": (
            "Rewarded when she calls the India-specific mechanism the Western press missed. Bad takes are "
            "'you're applying an American framework to a country with 22 income classes and 22 languages.'"
        ),
        "wow_factors": [
            "An India-native mechanism (UPI, ONDC, Account Aggregator) explained without dumbing down",
            "A specific RBI or SEBI move decoded — what it actually changes",
            "A state-level policy divergence (Karnataka vs Maharashtra vs TN)",
            "A demographic or income-class insight (India1 vs India2 vs India3)",
            "A founder story reframed with the regulatory/political backdrop",
        ],
        "shareability_triggers": (
            "Forwards to the India founder/PM WhatsApp group when it's a new mechanism. Retweets from the "
            "Ken account with 'This is exactly right.'"
        ),
        "depth_cravings": [
            "India's stack (UPI, Aadhaar, ONDC, AA) — how it actually works",
            "the political economy behind a market shift",
            "founder mortality and comeback arcs in Indian startups",
            "regulator psychology (what does the RBI/SEBI actually fear)",
            "the India-vs-global template contrast",
        ],
        "eye_roll_triggers": [
            "Western press applying US framings to India",
            "generic 'India is the next China' takes",
            "founder profiles without the regulatory backdrop",
            "diaspora takes disconnected from ground reality",
        ],
        "voice": "Confident, India-first, no diaspora apologia. Explains without patronizing.",
        "search_hint": "The Ken, Morning Context, Bloomberg Quint, Moneycontrol, RBI, SEBI, PRS Legislative, Nithin Kamath, Kunal Shah, Sajith Pai, LiveLaw",
    },
}


# ============================================================================
# LLM helpers
# ============================================================================
async def _llm_call_raw(
    system: str, user: str, *, model: str = LLM_MODEL, session_prefix: str = "nr"
) -> Tuple[str, float]:
    """Return (raw_text, elapsed_seconds)."""
    t0 = time.time()
    chat = LlmChat(
        api_key=os.environ["EMERGENT_LLM_KEY"],
        session_id=f"{session_prefix}-{uuid.uuid4()}",
        system_message=system,
    ).with_model(LLM_PROVIDER, model)
    try:
        raw = await chat.send_message(UserMessage(text=user))
    except Exception as e:
        log.exception("LLM call failed")
        return "", time.time() - t0
    return raw, time.time() - t0


async def _llm_json(
    system: str, user: str, *, model: str = LLM_MODEL, session_prefix: str = "nr"
) -> Tuple[Optional[dict], str, float]:
    """Call LLM, strip fences, parse JSON. Returns (parsed_or_None, raw_text, elapsed)."""
    raw, elapsed = await _llm_call_raw(
        system, user, model=model, session_prefix=session_prefix
    )
    if not raw:
        return None, "", elapsed
    s = raw.strip().strip("`")
    if s.lower().startswith("json"):
        s = s[4:].strip()
    if "{" in s:
        s = s[s.find("{") : s.rfind("}") + 1]
    try:
        return json.loads(s), raw, elapsed
    except Exception:
        log.warning("JSON parse failed for LLM output: %s", raw[:300])
        return None, raw, elapsed


# ============================================================================
# STAGE 1 — Signal Scout (multi-source aggregator)
# ============================================================================
async def _fetch_reddit_hot(
    subreddits: List[str], limit: int = 15
) -> List[Dict[str, Any]]:
    out = []
    headers = {"User-Agent": "DruvInsight-Newsroom/1.0"}
    async with httpx.AsyncClient(timeout=15, headers=headers) as c:
        for sub in subreddits:
            try:
                r = await c.get(
                    f"https://www.reddit.com/r/{sub}/hot.json?limit={limit}"
                )
                if r.status_code != 200:
                    continue
                for child in r.json().get("data", {}).get("children", []):
                    d = child.get("data", {})
                    if d.get("stickied"):
                        continue
                    out.append(
                        {
                            "source": f"reddit/r/{sub}",
                            "title": d.get("title", "")[:220],
                            "score": d.get("score", 0),
                            "num_comments": d.get("num_comments", 0),
                            "url": f"https://reddit.com{d.get('permalink', '')}",
                        }
                    )
            except Exception:
                log.warning("reddit fetch failed for %s", sub)
    return out


async def _fetch_hn_top(limit: int = 30) -> List[Dict[str, Any]]:
    out = []
    async with httpx.AsyncClient(timeout=15) as c:
        try:
            ids = (
                await c.get("https://hacker-news.firebaseio.com/v0/topstories.json")
            ).json()[:limit]
            for i in ids:
                try:
                    d = (
                        await c.get(
                            f"https://hacker-news.firebaseio.com/v0/item/{i}.json"
                        )
                    ).json()
                    if d and d.get("title"):
                        out.append(
                            {
                                "source": "hacker_news",
                                "title": d.get("title", "")[:220],
                                "score": d.get("score", 0),
                                "num_comments": d.get("descendants", 0),
                                "url": d.get("url")
                                or f"https://news.ycombinator.com/item?id={i}",
                            }
                        )
                except Exception:
                    continue
        except Exception:
            log.warning("HN fetch failed")
    return out


async def _fetch_tavily_trending(
    tavily: TavilyClient, queries: List[str]
) -> List[Dict[str, Any]]:
    out = []

    def _search(q):
        return tavily.search(
            query=q, search_depth="basic", max_results=8, topic="news", days=3
        )

    for q in queries:
        try:
            r = await asyncio.to_thread(_search, q)
            for it in r.get("results", []):
                out.append(
                    {
                        "source": "tavily_news",
                        "title": (it.get("title") or "")[:220],
                        "score": int((it.get("score") or 0) * 100),
                        "num_comments": 0,
                        "url": it.get("url", ""),
                    }
                )
        except Exception as e:
            log.warning("tavily trending failed for %s: %s", q, e)
    return out


async def signal_scout(
    tavily: TavilyClient,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Aggregate raw signals — India-first, persona-diverse.

    Every persona bucket pulls both India-native subs AND its trusted lenses,
    then Tavily queries are all India-anchored. Global stories only survive if
    they have a clear India read-through (Stage 2 enforces this).
    """
    t0 = time.time()

    # ── LEARNER-first subreddit set. Removed: gossip/meme subs (BollyBlindsNGossip,
    #    BollywoodMemes, IndianDankMemes, Chodi, librandu, Fauxmoi, popculturechat).
    #    Added: learner-oriented subs (AskIndia, OutOfTheLoop, ELI5, TIL,
    #    personalfinanceindia, IndianStockMarket, AskEconomics, csMajors).
    subreddit_groups: Dict[str, List[str]] = {
        # Culture ANALYST (not gossip) — sports depth, career shifts, "what's actually happening" explainers
        "culture_analyst": [
            "Cricket",
            "ipl",
            "IndianCricket",
            "OutOfTheLoop",
            "explainlikeimfive",
            "todayilearned",
            "dataisbeautiful",
        ],
        # Curious India — the "wait what" learning subs
        "india_insider": [
            "india",
            "AskIndia",
            "IndiaSocial",
            "CasualIndia",
            "UPSC",
            "JEENEETards",
            "csMajors",
            "developersIndia",
        ],
        # Policy DEPTH (not drama) — actual geopolitics + policy explainers
        "policy_wonk": [
            "geopolitics",
            "worldnews",
            "economics",
            "AskEconomics",
            "IndiaSpeaks",
        ],
        # Startup + tech culture
        "startup_founder": [
            "StartUpIndia",
            "startups",
            "IndianStartups",
            "SaaS",
            "Entrepreneur",
            "ycombinator",
        ],
        # Personal finance + investing learners
        "seed_investor": [
            "personalfinanceindia",
            "IndianStockMarket",
            "IndiaInvestments",
            "venturecapital",
            "IndiaGrowthStocks",
        ],
        # Developer + tech deep
        "developer": [
            "programming",
            "MachineLearning",
            "artificial",
            "IndiaTech",
            "learnprogramming",
        ],
        # Markets + contrarian
        "contrarian_economist": [
            "SecurityAnalysis",
            "stocks",
            "IndianStreetBets",
            "Valueinvesting",
        ],
        # Bollywood — but ONLY the depth-heavy subs (real biz coverage), not meme/gossip
        "gossip_girl": ["bollywood", "BollywoodRealism"],
    }

    # ── LEARNER-first Tavily queries. Reject "viral" — prefer "explained", "why",
    #    "shift", "mystery", "counter-intuitive", "how it actually works".
    persona_trending_queries: Dict[str, List[str]] = {
        "culture_analyst": [
            "India workforce trend shift explained this week LiveMint",
            "India career unusual success story analysis this week",
            "Indian urban lifestyle shift data this week",
            "India Gen Z spending analysis this week Ken",
        ],
        "policy_wonk": [
            "India policy mechanism explained this week Ken Morning Context",
            "India geopolitics analysis this week Adam Tooze",
            "India economic shift analysis this week LiveMint",
            "India regulation impact analysis this week",
        ],
        "india_insider": [
            "India business insider analysis this week The Ken",
            "India career mechanism explained this week",
            "India company insider story this week Morning Context",
            "India job market shift analysis this week",
        ],
        "startup_founder": [
            "Indian startup insight analysis this week Inc42 YourStory",
            "Indian founder statement analysis this week Moneycontrol",
            "India startup counter intuitive insight this week",
            "Indian company pivot story analysis this week",
            "Indian tech unusual mechanism this week",
        ],
        "seed_investor": [
            "India IPO analysis this week Moneycontrol Bloomberg Quint",
            "Indian company valuation analysis this week Livemint",
            "India VC market shift analysis this week",
            "Indian personal finance counter intuitive this week",
        ],
        "developer": [
            "AI India news deep analysis this week",
            "India tech infrastructure story this week",
            "India developer economy shift this week",
            "AI model release analysis this week Hacker News",
        ],
        "contrarian_economist": [
            "Indian stock market counter intuitive this week Moneycontrol",
            "India economic mystery explained this week Livemint",
            "India banking sector analysis this week",
            "Adani Ambani business analysis this week Ken",
        ],
        "gossip_girl": [
            "Bollywood industry business analysis this week Filmfare",
            "Indian celebrity brand economics this week",
        ],
    }

    # Flatten subs + track which persona each sub belongs to for trace
    all_subs: List[Tuple[str, str]] = []
    for persona, subs in subreddit_groups.items():
        for s in subs:
            all_subs.append((persona, s))

    async def _fetch_persona_reddit(persona: str, sub: str):
        r = await _fetch_reddit_hot([sub], limit=6)
        for item in r:
            item["_persona_hint"] = persona
        return persona, r

    async def _fetch_persona_tavily(persona: str, queries: List[str]):
        r = await _fetch_tavily_trending(tavily, queries)
        for item in r:
            item["_persona_hint"] = persona
        return persona, r

    reddit_tasks = [_fetch_persona_reddit(p, s) for p, s in all_subs]
    tavily_tasks = [
        _fetch_persona_tavily(p, qs) for p, qs in persona_trending_queries.items()
    ]
    hn_task = _fetch_hn_top(limit=15)

    reddit_results, tavily_results, hn_res = await asyncio.gather(
        asyncio.gather(*reddit_tasks, return_exceptions=False),
        asyncio.gather(*tavily_tasks, return_exceptions=False),
        hn_task,
        return_exceptions=False,
    )

    for item in hn_res:
        item["_persona_hint"] = "developer"

    per_persona_counts: Dict[str, Dict[str, int]] = {
        p: {"reddit": 0, "tavily": 0} for p in subreddit_groups
    }
    per_persona_counts["developer"]["hacker_news"] = len(hn_res)

    all_signals: List[Dict[str, Any]] = list(hn_res)
    for persona, items in reddit_results:
        per_persona_counts[persona]["reddit"] += len(items)
        all_signals.extend(items)
    for persona, items in tavily_results:
        per_persona_counts[persona]["tavily"] += len(items)
        all_signals.extend(items)

    raw_counts = {
        "reddit_total": sum(v["reddit"] for v in per_persona_counts.values()),
        "hacker_news": len(hn_res),
        "tavily_total": sum(v.get("tavily", 0) for v in per_persona_counts.values()),
    }

    # Cluster by rough title similarity
    clusters: Dict[str, Dict[str, Any]] = {}
    for s in all_signals:
        key_words = re.sub(r"[^\w\s]", "", s["title"].lower()).split()[:6]
        key = " ".join(key_words)
        if not key:
            continue
        if key not in clusters:
            clusters[key] = {
                "title": s["title"],
                "sources": set(),
                "score": 0,
                "urls": [],
                "raw_signals": [],
                "persona_hints": set(),
            }
        clusters[key]["sources"].add(s["source"])
        clusters[key]["score"] += s.get("score", 0)
        if s.get("url"):
            clusters[key]["urls"].append(s["url"])
        clusters[key]["raw_signals"].append(s)
        ph = s.get("_persona_hint")
        if ph:
            clusters[key]["persona_hints"].add(ph)

    ranked = []
    for c in clusters.values():
        ranked.append(
            {
                "title": c["title"],
                "signal_strength": c["score"],
                "source_diversity": len(c["sources"]),
                "sources": list(c["sources"]),
                "urls": c["urls"][:3],
                "raw_signals": c["raw_signals"][:5],
                "persona_hints": list(c["persona_hints"]),
            }
        )

    ranked.sort(
        key=lambda x: (x["source_diversity"], x["signal_strength"]), reverse=True
    )

    persona_buckets: Dict[str, List[Dict[str, Any]]] = {p: [] for p in subreddit_groups}
    persona_buckets["_unhinted"] = []
    for item in ranked:
        hints = item.get("persona_hints") or []
        placed = False
        for h in hints:
            if h in persona_buckets:
                persona_buckets[h].append(item)
                placed = True
                break
        if not placed:
            persona_buckets["_unhinted"].append(item)

    # Round-robin interleave — max 10 per persona bucket (culture_analyst has 8 subs, others less)
    max_per_persona = 10
    balanced: List[Dict[str, Any]] = []
    seen_titles = set()
    for i in range(max_per_persona):
        for p in list(subreddit_groups.keys()) + ["_unhinted"]:
            bucket = persona_buckets.get(p, [])
            if i < len(bucket):
                item = bucket[i]
                t = item["title"]
                if t not in seen_titles:
                    balanced.append(item)
                    seen_titles.add(t)
    for item in ranked:
        if len(balanced) >= 60:
            break
        if item["title"] not in seen_titles:
            balanced.append(item)
            seen_titles.add(item["title"])

    top = balanced[:50]

    trace = {
        "raw_source_counts": raw_counts,
        "per_persona_counts": per_persona_counts,
        "total_raw_signals": len(all_signals),
        "clusters_formed": len(clusters),
        "balancing": "India-first · round-robin, max 7 per persona bucket",
        "elapsed_sec": round(time.time() - t0, 2),
    }
    return top, trace
    t0 = time.time()

    # ── Persona-diverse subreddit sets ──
    subreddit_groups: Dict[str, List[str]] = {
        "gossip_girl": [
            "Fauxmoi",
            "popculturechat",
            "BollyBlindsNGossip",
            "entertainment",
            "celebrities",
        ],
        "startup_founder": [
            "startups",
            "ycombinator",
            "Entrepreneur",
            "SaaS",
            "smallbusiness",
        ],
        "seed_investor": [
            "venturecapital",
            "investing",
            "stocks",
            "SecurityAnalysis",
            "Valueinvesting",
        ],
        "policy_wonk": [
            "geopolitics",
            "worldnews",
            "economics",
            "IRstudies",
            "neoliberal",
        ],
        "developer": [
            "programming",
            "MachineLearning",
            "artificial",
            "compsci",
            "ExperiencedDevs",
        ],
        "contrarian_economist": [
            "wallstreetbets",
            "stocks",
            "options",
            "SecurityAnalysis",
            "Bogleheads",
        ],
        "culture_analyst": [
            "GenZ",
            "Millennials",
            "OutOfTheLoop",
            "TrueOffMyChest",
            "malefashionadvice",
            "femalefashionadvice",
        ],
        "india_insider": [
            "india",
            "IndianStreetBets",
            "unitedstatesofindia",
            "bollywood",
            "IndiaSpeaks",
        ],
    }

    # ── Persona-tuned trending queries (Tavily) ──
    persona_trending_queries: Dict[str, List[str]] = {
        "gossip_girl": [
            "celebrity brand deal news this week Vogue Business Puck",
            "Bollywood wedding brand collaboration news this week",
            "Hollywood couple breakup PR strategy news this week",
        ],
        "startup_founder": [
            "startup funding round news this week TechCrunch",
            "founder essay Stratechery Not Boring this week",
            "product launch tech news this week The Information",
        ],
        "seed_investor": [
            "venture capital trends this week Sequoia USV",
            "seed stage market shift analysis this week",
            "AI startup funding pattern this week Pitchbook",
        ],
        "policy_wonk": [
            "geopolitics analysis this week Foreign Affairs Adam Tooze",
            "central bank policy shift this week FT RBI Fed",
            "trade tariff policy news this week Odd Lots",
        ],
        "developer": [
            "engineering breakthrough news this week arxiv Hacker News",
            "AI model benchmark release this week Latent Space",
            "software incident postmortem this week Pragmatic Engineer",
        ],
        "contrarian_economist": [
            "market mispricing analysis this week Matt Levine Money Stuff",
            "short seller report this week Hindenburg Muddy Waters",
            "SEC filing surprise this week FT Alphaville",
        ],
        "culture_analyst": [
            "Gen Z spending trend this week Piper Sandler NYT Styles",
            "Business of Fashion trend brief this week",
            "Rest of World culture shift this week",
        ],
        "india_insider": [
            "India business news this week The Ken Morning Context",
            "RBI SEBI policy notification this week",
            "Indian startup founder news this week Bloomberg Quint",
        ],
    }

    # Flatten subs + track which persona each sub belongs to for trace
    all_subs: List[Tuple[str, str]] = []  # [(persona, sub), ...]
    for persona, subs in subreddit_groups.items():
        for s in subs:
            all_subs.append((persona, s))

    # Parallel fetches
    async def _fetch_persona_reddit(persona: str, sub: str):
        r = await _fetch_reddit_hot([sub], limit=6)
        for item in r:
            item["_persona_hint"] = persona
        return persona, r

    async def _fetch_persona_tavily(persona: str, queries: List[str]):
        r = await _fetch_tavily_trending(tavily, queries)
        for item in r:
            item["_persona_hint"] = persona
        return persona, r

    reddit_tasks = [_fetch_persona_reddit(p, s) for p, s in all_subs]
    tavily_tasks = [
        _fetch_persona_tavily(p, qs) for p, qs in persona_trending_queries.items()
    ]
    hn_task = _fetch_hn_top(limit=20)

    reddit_results, tavily_results, hn_res = await asyncio.gather(
        asyncio.gather(*reddit_tasks, return_exceptions=False),
        asyncio.gather(*tavily_tasks, return_exceptions=False),
        hn_task,
        return_exceptions=False,
    )

    # Tag HN as developer/founder-flavored
    for item in hn_res:
        item["_persona_hint"] = "developer"

    # Per-persona counts for trace
    per_persona_counts: Dict[str, Dict[str, int]] = {
        p: {"reddit": 0, "tavily": 0} for p in subreddit_groups
    }
    per_persona_counts["developer"]["hacker_news"] = len(hn_res)

    all_signals: List[Dict[str, Any]] = list(hn_res)
    for persona, items in reddit_results:
        per_persona_counts[persona]["reddit"] += len(items)
        all_signals.extend(items)
    for persona, items in tavily_results:
        per_persona_counts[persona]["tavily"] += len(items)
        all_signals.extend(items)

    raw_counts = {
        "reddit_total": sum(v["reddit"] for v in per_persona_counts.values()),
        "hacker_news": len(hn_res),
        "tavily_total": sum(v.get("tavily", 0) for v in per_persona_counts.values()),
    }

    # Cluster by rough title similarity (first 6 words as key)
    clusters: Dict[str, Dict[str, Any]] = {}
    for s in all_signals:
        key_words = re.sub(r"[^\w\s]", "", s["title"].lower()).split()[:6]
        key = " ".join(key_words)
        if not key:
            continue
        if key not in clusters:
            clusters[key] = {
                "title": s["title"],
                "sources": set(),
                "score": 0,
                "urls": [],
                "raw_signals": [],
                "persona_hints": set(),
            }
        clusters[key]["sources"].add(s["source"])
        clusters[key]["score"] += s.get("score", 0)
        if s.get("url"):
            clusters[key]["urls"].append(s["url"])
        clusters[key]["raw_signals"].append(s)
        ph = s.get("_persona_hint")
        if ph:
            clusters[key]["persona_hints"].add(ph)

    ranked = []
    for c in clusters.values():
        ranked.append(
            {
                "title": c["title"],
                "signal_strength": c["score"],
                "source_diversity": len(c["sources"]),
                "sources": list(c["sources"]),
                "urls": c["urls"][:3],
                "raw_signals": c["raw_signals"][:5],
                "persona_hints": list(c["persona_hints"]),
            }
        )

    # ── Persona-balanced ranking: cap dominance from any single persona hint ──
    # First sort within each persona bucket, then interleave round-robin so no
    # single persona takes more than its fair share of the top 40.
    ranked.sort(
        key=lambda x: (x["source_diversity"], x["signal_strength"]), reverse=True
    )

    persona_buckets: Dict[str, List[Dict[str, Any]]] = {p: [] for p in subreddit_groups}
    persona_buckets["_unhinted"] = []
    for item in ranked:
        hints = item.get("persona_hints") or []
        placed = False
        for h in hints:
            if h in persona_buckets:
                persona_buckets[h].append(item)
                placed = True
                break
        if not placed:
            persona_buckets["_unhinted"].append(item)

    # Round-robin interleave: take up to 6 from each persona's top before spilling
    max_per_persona = 6
    balanced: List[Dict[str, Any]] = []
    seen_titles = set()
    for i in range(max_per_persona):
        for p in list(subreddit_groups.keys()) + ["_unhinted"]:
            bucket = persona_buckets.get(p, [])
            if i < len(bucket):
                item = bucket[i]
                t = item["title"]
                if t not in seen_titles:
                    balanced.append(item)
                    seen_titles.add(t)
    # Fill any remaining slots with the strongest overall
    for item in ranked:
        if len(balanced) >= 50:
            break
        if item["title"] not in seen_titles:
            balanced.append(item)
            seen_titles.add(item["title"])

    top = balanced[:40]

    trace = {
        "raw_source_counts": raw_counts,
        "per_persona_counts": per_persona_counts,
        "total_raw_signals": len(all_signals),
        "clusters_formed": len(clusters),
        "balancing": "round-robin interleave, max 6 per persona bucket",
        "elapsed_sec": round(time.time() - t0, 2),
    }
    return top, trace


# ============================================================================
# STAGE 2 — Topic Filter (expert editor)
# ============================================================================
TOPIC_FILTER_SYSTEM = """You are the Managing Editor of DruvInsight — an insight app for CURIOUS EXPLORERS aged 20-35. Your reader is NOT a gossip consumer. Your reader is someone who reads Hacker News in the morning, listens to Acquired on their commute, and wants to LEARN something they didn't know before.

Your audience:
- Founders, product managers, engineers, analysts, consultants, MBAs, researchers, curious professionals
- They open the app to LEARN something new, not to catch up on celebrity news
- They love mysteries, mechanisms, counter-intuitive truths, hidden systems, "how does this ACTUALLY work" explainers
- They have zero interest in "X celebrity did Y" gossip, "who is this new viral person", or "meme moment reactions"

Score each signal 1-10 on 4 axes:

1. **teaches_something_new** — Does the topic promise a LEARNING moment? Would answering this topic reveal something a curious explorer didn't know? A gossip story about who dated whom = 1. A story revealing HOW a specific system works, WHY an unusual shift is happening, or WHAT'S actually behind a trend = 10.

2. **curious_learner_pull** — Would a 25-year-old founder / analyst / engineer WANT to click this on their morning commute? Something that gets them to a "wait, seriously?" moment = 10. A viral moment they've already seen on Insta = 2.

3. **is_shift_or_mystery** — Is this a real SHIFT (economic, technological, cultural, career-market) or MYSTERY (something unusual that begs explanation) or MECHANISM (how does something actually work)? A one-off event = 2. A structural shift with 5-year implications = 10.

4. **not_saturated** — Is there room for a non-obvious angle? Already fully-covered = 1. Genuine curiosity gap = 10.

═══ HARD REJECT (total below any threshold, or matches any of these patterns) ═══
- Pure celebrity gossip: "Kareena and Bipasha fallout" / "Niharika NM viral" / "Deepika at wedding"
- Viral moment reporting with no mechanism: "This reel got 120M views" / "Modi called GenZ friends"
- Meme trend commentary
- Politician-said-something-wild news
- One-off scandal, PR controversy, brand fight — UNLESS there's a real business/economic mechanism behind it
- "Who is X and why are they suddenly everywhere" pure-curiosity-about-a-person stories

═══ STRONGLY PREFER (aim for these) ═══
- IPO mysteries, market shifts, valuation changes (e.g., "12 firms eyeing ₹25,000 cr August IPO haul")
- Career mysteries and salary shifts (e.g., "₹4L to ₹30L at Qualcomm — how")
- Business trend explanations (e.g., "The rise of the adaptive workforce")
- Founder statements that reveal industry mechanics (e.g., "India shouldn't be an AI colony — Sarvam co-founder")
- Economic mysteries (e.g., "Sensex jumped 888 points despite US-Iran tensions — why")
- Layoff patterns with structural read (e.g., "Global layoffs reach Indian shores")
- Company pivots that reveal larger shifts (e.g., "Zepto pauses IPO — what changed")
- Regulatory / policy mechanisms with real-world impact
- Sports strategy analyses (WHY did IPL team X do move Y — not just who won)
- Technology infrastructure stories that affect daily life
- Historical patterns re-emerging (analyst-favourite territory)

═══ TOPIC FIELD RULE ═══
The `topic` field must be a **learner-style phrase** — something a curious explorer would want to click on. Not gossip. Not a headline.

- GOOD: "India's IPO market suddenly has 12 firms lining up for August — why now?"
- GOOD: "How did someone go from a ₹4 LPA offer to ₹30 LPA at Qualcomm"
- GOOD: "Why the adaptive workforce is becoming India's biggest hiring shift"
- GOOD: "Zepto paused its IPO — what actually changed in 60 days"
- GOOD: "Sensex rallied 888 points despite US-Iran — the hidden buyer"
- BAD: "Modi's new Insta reel is blowing up" (gossip)
- BAD: "Who is Niharika NM" (person-curiosity)
- BAD: "Kareena Bipasha fallout resurfaces" (pure gossip)

Return STRICT JSON only. No prose."""

TOPIC_FILTER_USER = """SIGNALS (JSON array):
{signals}

Return one score-row per input signal, preserving idx:
{{
  "scored": [
    {{"idx": 0, "teaches_something_new": N, "curious_learner_pull": N, "is_shift_or_mystery": N, "not_saturated": N, "total": N, "keep": true/false, "topic": "<learner-style topic phrase>", "reasoning": "<one line justification>"}}
  ]
}}"""


async def topic_filter(
    signals: List[Dict[str, Any]], keep_n: int = 15
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    t0 = time.time()
    if not signals:
        return [], {"elapsed_sec": 0, "reason": "no signals"}
    payload = [
        {"idx": i, "title": s["title"], "sources": s["sources"]}
        for i, s in enumerate(signals[:40])
    ]
    parsed, raw, elapsed = await _llm_json(
        TOPIC_FILTER_SYSTEM,
        TOPIC_FILTER_USER.format(signals=json.dumps(payload, ensure_ascii=False)),
        model=LLM_MODEL_FAST,
        session_prefix="filter",
    )
    scored_rows = (parsed or {}).get("scored", [])
    if not scored_rows:
        return [], {
            "elapsed_sec": round(elapsed, 2),
            "reason": "filter parse failed",
            "raw_head": (raw or "")[:400],
        }

    kept = []
    all_scored = []
    for row in scored_rows:
        idx = row.get("idx")
        if not isinstance(idx, int) or idx >= len(signals):
            continue
        base = signals[idx]
        scored_entry = {
            "title": base["title"],
            "topic": row.get("topic") or base["title"],
            "score": row.get("total", 0),
            "breakdown": {
                k: row.get(k)
                for k in (
                    "teaches_something_new",
                    "curious_learner_pull",
                    "is_shift_or_mystery",
                    "not_saturated",
                )
            },
            "keep": bool(row.get("keep")) and row.get("total", 0) >= 28,
            "reasoning": row.get("reasoning", ""),
            "sources": base["sources"],
        }
        all_scored.append(scored_entry)
        if scored_entry["keep"]:
            kept.append(
                {
                    **base,
                    "topic": scored_entry["topic"],
                    "quality_score": scored_entry["score"],
                    "score_breakdown": scored_entry["breakdown"],
                    "filter_reasoning": scored_entry["reasoning"],
                }
            )
    kept.sort(key=lambda x: x["quality_score"], reverse=True)
    trace = {
        "elapsed_sec": round(time.time() - t0, 2),
        "scored_count": len(all_scored),
        "kept_count": len(kept),
        "all_scored": all_scored,
    }
    return kept[:keep_n], trace


# ============================================================================
# STAGE 2.5 — Persona Router (expert audience matcher)
# ============================================================================
ROUTER_SYSTEM = """You are a Chief Audience Officer. You match every story to the ONE persona whose reader would find it MOST engaging and MOST likely to swipe and share.

Persona choices (pick exactly one persona_key):
{persona_details}

Rules:
- Pick the persona whose *deep sources* and *wow factors* best match this topic.
- If a story clearly touches multiple lanes, pick the one with the strongest 'shareability_triggers' fit — not the most obvious lane.
- A story about a Bollywood wedding with big brand-deal economics → gossip_girl (not culture_analyst).
- A story about India's UPI mechanics → india_insider (not startup_founder).
- A story about a paper on LLM benchmarks → developer (not startup_founder).

Return STRICT JSON: {{"persona_key": "<key>", "why": "<one line reason, name-check the persona's specific wow-factor that this story hits>"}}"""


async def persona_router(topic: str) -> Tuple[str, str, float]:
    """Returns (persona_key, reasoning, elapsed_sec)."""
    persona_details_list = []
    for k, v in PERSONAS.items():
        persona_details_list.append(
            f"- {k} ({v['label']}): reader={v['reader'][:200]}... "
            f"wow_factors={v['wow_factors'][:3]}"
        )
    persona_details = "\n".join(persona_details_list)
    parsed, raw, elapsed = await _llm_json(
        ROUTER_SYSTEM.format(persona_details=persona_details),
        f'STORY TOPIC: "{topic}"\n\nReturn strict JSON.',
        model=LLM_MODEL_FAST,
        session_prefix="router",
    )
    if not parsed:
        return "startup_founder", "router failed; defaulted", elapsed
    p = (parsed.get("persona_key") or "").strip()
    why = (parsed.get("why") or "").strip()
    if p not in PERSONAS:
        return "startup_founder", "invalid persona; defaulted", elapsed
    return p, why, elapsed


# ============================================================================
# STAGE 3 — Angle Ideator (expert content strategist per persona)
# ============================================================================
# ============================================================================
# STAGE 3 — Question Ladder (question generation → ranking → angle transformer)
# ============================================================================
#
# Great angles are great QUESTIONS framed as hooks. So we don't ask one Ideator
# for 5 angles — we ask 10 specialist Question Agents (each with a different
# lens), collect 25-30 candidate questions, rank them, and convert the top 5
# into angles.
#
# The lenses are drawn from what makes an angle genuinely engaging for a
# curious 25-year-old — invisible mechanism, follow-the-money, why-this-
# person, winners-and-losers, historical parallel, media-meta, insider-
# industry, official-vs-actual, second-order future, precedent-setting.
# ============================================================================

QUESTION_LENSES: Dict[str, Dict[str, str]] = {
    "follow_the_money": {
        "label": "Follow the Money",
        "description": (
            "Ask questions about the invisible cash flows behind the story. "
            "Every event has funders, beneficiaries, and quiet ROI. Big protests, "
            "big campaigns, big brand deals — all cost real money. Who's paying "
            "for what, and what return do they expect."
        ),
        "example_qs": (
            "Every major student protest in India costs ₹8-15 lakh in coordination — "
            "who typically funds these? Coaching-industry money? Rival political "
            "parties? What's the historical pattern of ROI on funding a protest?"
        ),
    },
    "invisible_mechanism": {
        "label": "Invisible Mechanism",
        "description": (
            "Ask how the system actually works BEHIND the surface event. "
            "What's the causal machinery — the rules, thresholds, escalation "
            "paths, decision-makers — that most coverage skips? Not 'what "
            "happened' but 'through what invisible levers did it happen.'"
        ),
        "example_qs": (
            "Do memes and reels actually change policy in India? If yes — "
            "what's the mechanism? Is there a threshold — 40-50% affected "
            "citizens on the streets? A viral view count? A specific number "
            "of press conferences? What actually moves the needle in Delhi?"
        ),
    },
    "winners_losers": {
        "label": "Winners & Losers",
        "description": (
            "Every event redistributes power, money, or attention. Ask who "
            "quietly WINS from the outcome and who LOSES. Winners and losers "
            "often fund political parties to protect or grow their position. "
            "Trace both sides."
        ),
        "example_qs": (
            "Every policy has a winner and a loser, and both fund political "
            "parties to grow/protect their business. In India's education "
            "policy fight, which industries LIVE on the old status quo "
            "(textbook publishers, coaching centers, hostel operators)? "
            "Which industries WIN if education gets privatized?"
        ),
    },
    "why_this_person": {
        "label": "Why This Person",
        "description": (
            "Every appointment, resignation, promotion is a strategic choice. "
            "Ask: why THIS specific person for THIS specific role at THIS "
            "specific moment? What was their unique skill set, network, or "
            "leverage that made them the right agent for what the org wanted?"
        ),
        "example_qs": (
            "What role does an education minister like Dharmendra Pradhan "
            "actually play? Why was HE picked in 2019, not someone else? "
            "Was his skill fundraising from a specific base? Was he the one "
            "who could switch education from a right into a supply chain?"
        ),
    },
    "historical_parallel": {
        "label": "Historical Parallel",
        "description": (
            "Every major event has happened before in some form. Ask: when "
            "did this pattern play out historically? What was the outcome? "
            "What can we PREDICT from that precedent about this time? "
            "Specific year, specific entity, specific consequence."
        ),
        "example_qs": (
            "Which past Indian student movements have forced a minister to "
            "resign? Anti-Mandal 1990, JP Movement 1974 — what were their "
            "funding structures, their eventual outcomes 5 years later, and "
            "what does that predict here?"
        ),
    },
    "precedent_setting": {
        "label": "Precedent Set",
        "description": (
            "Ask: what NEW rule or template does this event write for the "
            "future? Every big event either establishes or breaks a norm. "
            "What's the new playbook for the next actor who wants to do "
            "the same thing?"
        ),
        "example_qs": (
            "If a meme-driven student movement can now oust a Union Minister "
            "in 25 days, what NEW playbook does that write for the next "
            "activist group? What does the BJP do differently next time? "
            "What does the coaching industry learn?"
        ),
    },
    "media_meta": {
        "label": "Media & Algorithm",
        "description": (
            "Ask: what role did platforms, algorithms, and content types "
            "play in creating this outcome? Which platform benefited? "
            "How did the algorithm shape which stories reached which people? "
            "Was this event a manufactured algorithmic moment or organic?"
        ),
        "example_qs": (
            "The Cockroach reel got 120M views in 10 hours. Was that "
            "organic virality or Instagram Reels' recommendation choice? "
            "What did Meta's algorithm optimize for that gave this specific "
            "meme its lift? Which other movements did NOT get the algo boost?"
        ),
    },
    "insider_industry": {
        "label": "Insider Industry",
        "description": (
            "Every event affects specific industries whose survival depends "
            "on the old order. Ask: which businesses/professions are quietly "
            "affected? Which government-adjacent industries live or die by "
            "this decision? Get specific — name the industries."
        ),
        "example_qs": (
            "Which industries lived on the OLD Indian education system? "
            "NCERT textbook printers, government-funded IITs, coaching "
            "chains, hostel businesses, PSU-employer relationships. Who "
            "financially benefits from privatization? Who dies?"
        ),
    },
    "official_vs_actual": {
        "label": "Official vs Actual",
        "description": (
            "Ask: what's the OFFICIAL story vs what actually happened? "
            "What are the receipts, filings, or off-the-record details "
            "that don't match the press-release narrative? What are all "
            "the parties officially saying vs what they're actually doing?"
        ),
        "example_qs": (
            "The government's official reason for the minister's exit vs "
            "the real timeline. Were negotiations happening privately "
            "before the resignation looked forced? What's the gap between "
            "the PIB statement and what BJP insiders were doing?"
        ),
    },
    "second_order_future": {
        "label": "Second-Order Future",
        "description": (
            "Ask: what happens 6/12/36 months from now BECAUSE of this? "
            "What decision does someone now make that they wouldn't have "
            "before? What downstream event becomes probable? Predict with "
            "specifics — a specific actor, a specific action, a specific "
            "timeline."
        ),
        "example_qs": (
            "6 months from now, what does the BJP do differently in "
            "the next election cycle because of this? Does the coaching "
            "industry pivot? Do private colleges accelerate their hostel "
            "acquisitions? What's the specific probable next move?"
        ),
    },
}


QUESTION_AGENT_SYSTEM = """You are a senior editorial analyst specialized in ONE questioning lens: **{lens_label}**.

Your lens's job:
{lens_description}

Example of the KIND of question your lens produces (given as a reference — do NOT copy verbatim, produce fresh questions):
> {lens_example}

═══ YOUR TASK ═══
For the topic below, generate EXACTLY 3 questions from YOUR lens ONLY. Do NOT mix in other lenses.

Each question must:
- Be a genuinely non-obvious question a curious 25-year-old Indian professional would silently wonder about but has never seen asked directly.
- Be SPECIFIC — name real entities, real numbers, real dates, real industries wherever possible.
- Be answerable through research (not pure speculation).
- Have real "I've always wondered about this" energy — the kind of question your friend would ask you at 11pm over coffee.

AVOID:
- Generic questions ("What's the impact of X?", "How will this affect Y?")
- Questions with obvious 30-second Google answers
- Yes/no questions without a "how" or "why" follow-up
- Recycled Twitter takes rephrased as questions

Return STRICT JSON:
{{"questions": [
  {{"question": "<the actual question>", "curiosity_hook": "<what makes it interesting in one line>"}},
  {{...}},
  {{...}}
]}}"""


async def _question_from_lens(
    topic: str, lens_key: str
) -> Tuple[List[Dict[str, Any]], float]:
    lens = QUESTION_LENSES[lens_key]
    parsed, raw, elapsed = await _llm_json(
        QUESTION_AGENT_SYSTEM.format(
            lens_label=lens["label"],
            lens_description=lens["description"],
            lens_example=lens["example_qs"],
        ),
        f'TOPIC: "{topic}"\n\nReturn 3 questions from your lens as strict JSON.',
        model=LLM_MODEL_FAST,
        session_prefix=f"q-{lens_key}",
    )
    if not parsed:
        return [], elapsed
    out = []
    for q in parsed.get("questions", [])[:3]:
        out.append(
            {
                "question": q.get("question", ""),
                "curiosity_hook": q.get("curiosity_hook", ""),
                "lens": lens_key,
                "lens_label": lens["label"],
            }
        )
    return out, elapsed


async def generate_questions(topic: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run all 10 specialist question agents in parallel."""
    t0 = time.time()
    tasks = [_question_from_lens(topic, k) for k in QUESTION_LENSES]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    all_qs: List[Dict[str, Any]] = []
    per_lens_counts: Dict[str, int] = {}
    for (qs, elapsed), lens_key in zip(results, QUESTION_LENSES.keys()):
        per_lens_counts[lens_key] = len(qs)
        all_qs.extend(qs)
    return all_qs, {
        "elapsed_sec": round(time.time() - t0, 2),
        "per_lens_counts": per_lens_counts,
        "total_questions": len(all_qs),
    }


# ── Question Ranker ──────────────────────────────────────────────────────
QUESTION_RANKER_SYSTEM = """You are the Editor-in-Chief. You've been handed ~30 candidate questions on a topic, each from a different editorial lens. Your job: pick the 5 questions whose ANSWERS would most reward a curious 20-35 year old explorer.

A GREAT question passes these tests:
- Its answer is NOT commonly known — most people can't answer it off the top of their head
- Its answer is COUNTER-INTUITIVE — the reader's default assumption is likely wrong
- The question opens a rabbit hole (answering it opens 3 more sub-questions)
- The question can be grounded in real research (numbers, entities, dates)
- Answering the question TEACHES the reader something durable — a mental model, a mechanism, a historical pattern

A BAD question:
- Has an obvious 30-second Google answer
- Is essentially "who is X" / "what did X do" — pure name/event curiosity
- Rephrases what mainstream news is already saying
- Would be answered by "vibes" not evidence
- Is a yes/no question with no depth follow-up

Score each question 1-10 on these 5 axes:

1. **counter_intuitive** — Would the answer OVERTURN the average reader's default assumption? If yes = 10. If yes-but-obvious once you think about it = 5. If the answer confirms the default = 1.

2. **lesser_known_answer** — Is the answer NOT findable in the first 5 Google results? Would 90% of people not know this? Score 10 for genuinely obscure but well-documented insights. Score 1 for "everyone knows this."

3. **teaches_something_durable** — Does answering this question give the reader a MENTAL MODEL, MECHANISM, or PATTERN they can reuse elsewhere (not just a single fact)? Reusable insight = 10. One-off fact = 3.

4. **rabbit_hole_potential** — Does answering this question open 3+ new sub-questions the reader will now want answered? Explosive = 10. Dead-end = 1.

5. **answerability** — Can this actually be researched and answered with concrete sources/numbers/entities? Grounded = 10. Speculation only = 1.

Rules:
- Pick 5 questions from DIFFERENT lenses (diversity wins — never pick 3 from one lens).
- REJECT any question scoring below 38/50 total. Better to return 3 great questions than 5 mediocre ones.
- Reject questions with counter_intuitive < 7 or lesser_known_answer < 7. These two are non-negotiable.
- Between two similar questions, prefer the one with a sharper, more specific counter-intuition.

Return STRICT JSON:
{{
  "ranked": [
    {{"idx": N, "counter_intuitive": N, "lesser_known_answer": N, "teaches_something_durable": N, "rabbit_hole_potential": N, "answerability": N, "total": N, "note": "<one-line reason>"}},
    ...all questions ranked...
  ],
  "top_5_indices": [ints, ordered strongest first]
}}"""


async def rank_questions(
    questions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not questions:
        return [], {"elapsed_sec": 0, "reason": "no questions"}
    payload = [
        {
            "idx": i,
            "lens": q["lens_label"],
            "question": q["question"],
            "hook": q["curiosity_hook"],
        }
        for i, q in enumerate(questions)
    ]
    parsed, raw, elapsed = await _llm_json(
        QUESTION_RANKER_SYSTEM,
        f"CANDIDATE QUESTIONS:\n{json.dumps(payload, ensure_ascii=False, indent=2)}",
        model=LLM_MODEL,
        session_prefix="qranker",
    )
    if not parsed:
        return [], {
            "elapsed_sec": round(elapsed, 2),
            "reason": "ranker returned nothing",
        }
    ranked = parsed.get("ranked", [])
    top_ids = parsed.get("top_5_indices", [])[:5]
    top: List[Dict[str, Any]] = []
    for tid in top_ids:
        if isinstance(tid, int) and 0 <= tid < len(questions):
            entry = dict(questions[tid])
            score = next((r for r in ranked if r.get("idx") == tid), None)
            if score:
                entry["_rank_score"] = score
            top.append(entry)
    return top, {
        "elapsed_sec": round(elapsed, 2),
        "all_scored": ranked,
        "top_5_indices": top_ids,
    }


# ── Question → Angle Transformer ──────────────────────────────────────────
QUESTION_TO_ANGLE_SYSTEM = """You are a senior editorial packager. You receive 5 EXCELLENT editorial questions that survived a brutal ranking process. Your job: convert each into a proper ANGLE spec — WITHOUT losing the question's punch.

═══ THE #1 RULE — PRESERVE THE TENSION ═══
The reason each question survived ranking is that it's COUNTER-INTUITIVE and has a LESSER-KNOWN answer. The angle title you write MUST preserve that counter-intuition — it must feel like a hint of a surprising answer, not a smoothed-out headline.

- If the question is "Why did Sensex rise 888 pts despite US-Iran tensions?", the angle title should tease the *unexpected* answer (e.g., "The Buyer Nobody's Naming"), NOT flatten it to "Markets rise on positive sentiment".
- If the question is "How did someone go from ₹4L to ₹30L at Qualcomm?", the angle title should tease the *contrarian* answer (e.g., "The Rejection That Was Actually The Job Interview"), NOT flatten to "Success story of Qualcomm engineer".

═══ RULE 2 — HOOK FIRST, NAME AFTER ═══
The title should make the reader THINK "wait what — why?" before they've even read the thesis. If your title merely restates the topic, you've failed.

═══ WHAT TO PRODUCE ═══
For each question, produce an ANGLE with:
- **kind**: pick ONE that best fits — RECEIPTS_REVEAL, WTF_MOMENT, GROUP_CHAT_FORWARD, CONTRARIAN, HISTORICAL_PARALLEL, SECOND_ORDER, WHO_BENEFITS, WHAT_EVERYONES_MISSING
- **title**: 6-12 words. Counter-intuitive hint of the answer. Feels like a Ken subheading, not a newspaper headline.
- **thesis**: 1-2 sentences with a specific number/name/date the writer will build the story around. Must PROMISE the counter-intuitive answer.
- **anchor_data_point**: one concrete number, name, date, industry, or entity.
- **rabbit_hole_potential**: 3-4 line description of what 12 follow-up sub-cards could dig into.
- **why_this_reader**: one line — why this SPECIFIC persona will care AND why they'd learn from it.
- **underlying_question**: repeat the source question for traceability.

═══ TARGET READER (curious explorer, not gossip consumer) ═══
{persona_reader}

═══ PERSONA WOW FACTORS ═══
{persona_wow_factors}

Return STRICT JSON:
{{"angles": [
  {{"kind": "...", "title": "...", "thesis": "...", "anchor_data_point": "...", "rabbit_hole_potential": "...", "why_this_reader": "...", "underlying_question": "..."}},
  ...5 total, one per input question, each preserving the counter-intuition of its source question...
]}}"""


async def questions_to_angles(
    top_questions: List[Dict[str, Any]],
    persona_key: str,
    topic: str,
) -> Tuple[List[Dict[str, Any]], float]:
    p = PERSONAS[persona_key]
    payload = [
        {
            "idx": i,
            "lens": q["lens_label"],
            "question": q["question"],
            "hook": q["curiosity_hook"],
        }
        for i, q in enumerate(top_questions)
    ]
    parsed, raw, elapsed = await _llm_json(
        QUESTION_TO_ANGLE_SYSTEM.format(
            persona_reader=p["reader"],
            persona_wow_factors="\n".join(f"  • {w}" for w in p["wow_factors"]),
        ),
        f'TOPIC: "{topic}"\n\nSOURCE QUESTIONS:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\nConvert each question into a proper angle spec — DO NOT lose the counter-intuitive edge.',
        model=LLM_MODEL,
        session_prefix="q2angle",
    )
    if not parsed:
        return [], elapsed
    return parsed.get("angles", [])[:5], elapsed


# ── Existing Ideator kept as a fallback (unchanged interface) ──
IDEATOR_SYSTEM = """You are a senior editorial strategist with 8 years at Puck, then The Information, then The Economist. You know how to spin a topic into an angle that this SPECIFIC reader will find irresistible.

TARGET READER (write for this exact person):
{persona_reader}

WHAT THIS READER READS (source these mental models when picking angles):
{persona_reads}

WHAT REWARDS THIS READER AT WORK (this shapes what they value):
{persona_work_incentives}

WHAT MAKES THIS READER GO 'WHOA':
{persona_wow_factors}

WHAT THIS READER RABBIT-HOLES ON:
{persona_depth_cravings}

WHAT THIS READER EYE-ROLLS AT (avoid these framings):
{persona_eye_roll}

VOICE HINT (for internal reference, don't write in it here — the angle description is a brief):
{persona_voice}

═══ YOUR TASK ═══
Generate EXACTLY 5 completely different candidate angles for the topic. Each angle must be a genuine editorial hook a 25-year-old Indian would swipe on AND text to their friends.

Pick your 5 angles from this pool of 8 kinds (choose whichever fit the topic BEST — you don't need to hit every kind, but each angle you pick MUST be a different kind):

1. **RECEIPTS_REVEAL** — the money trail / private motive / actual receipts behind the public story. ("You thought X was Y. Here's what the receipts actually show.")
2. **WTF_MOMENT** — a wild, hard-to-believe fact from the story that most coverage is skipping. ("Wait — seriously?")
3. **GROUP_CHAT_FORWARD** — the one nugget from the story that would go straight into a WhatsApp group chat. ("You'll want to send this to your friend.")
4. **CONTRARIAN** — the mainstream framing is wrong, and here's the specific mispricing.
5. **HISTORICAL_PARALLEL** — a specific past event that mirrors this (with year, entity, outcome). Only pick this if the parallel is genuinely striking, not academic.
6. **SECOND_ORDER** — the ripple/downstream effect no one is naming yet.
7. **WHO_BENEFITS** — the entity quietly winning from this dynamic.
8. **WHAT_EVERYONES_MISSING** — the datum, mechanism, or actor the coverage is ignoring.

Priority: FOR HOT NOW, favour RECEIPTS_REVEAL, WTF_MOMENT, GROUP_CHAT_FORWARD, and WHO_BENEFITS — these are the shareable ones. Use CONTRARIAN / HISTORICAL_PARALLEL / SECOND_ORDER / WHAT_EVERYONES_MISSING only when the topic is genuinely dry (finance, policy) and needs a smart-angle rescue.

Each angle must have:
- title: 6-12 words, catchy hook that names a real entity or vivid detail. This is what the reader sees FIRST. Must feel like a WhatsApp forward, not a Bloomberg headline. NO metaphors, NO jargon.
- thesis: 1-2 sentences making a specific claim with a real number/name/date.
- anchor_data_point: one concrete number, name, date, or entity that grounds it.
- rabbit_hole_potential: what 12 follow-up sub-cards could dig into.
- why_this_reader: one line explaining why THIS SPECIFIC persona will care AND why they'd share.

{feedback_block}

Return STRICT JSON:
{{"angles": [
  {{"kind": "<one of the 8 kinds above>", "title": "...", "thesis": "...", "anchor_data_point": "...", "rabbit_hole_potential": "...", "why_this_reader": "..."}},
  ...5 total, EACH A DIFFERENT KIND...
]}}"""


async def angle_ideator(
    topic: str, persona_key: str, feedback: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], float, str]:
    """Returns (angles, elapsed_sec, raw_response)."""
    p = PERSONAS[persona_key]
    feedback_block = (
        f"═══ PRIOR CRITIQUE (address this in the new attempts) ═══\n{feedback}\n"
        if feedback
        else ""
    )
    parsed, raw, elapsed = await _llm_json(
        IDEATOR_SYSTEM.format(
            persona_reader=p["reader"],
            persona_reads="\n".join(f"  • {s}" for s in p["reads"][:10]),
            persona_work_incentives=p["work_incentives"],
            persona_wow_factors="\n".join(f"  • {w}" for w in p["wow_factors"]),
            persona_depth_cravings="\n".join(f"  • {d}" for d in p["depth_cravings"]),
            persona_eye_roll="\n".join(f"  • {e}" for e in p["eye_roll_triggers"]),
            persona_voice=p["voice"],
            feedback_block=feedback_block,
        ),
        f'STORY TOPIC: "{topic}"\n\nGenerate 5 angles as strict JSON.',
        model=LLM_MODEL,
        session_prefix="ideator",
    )
    if not parsed:
        return [], elapsed, raw
    return parsed.get("angles", [])[:5], elapsed, raw


# ============================================================================
# STAGE 4 — Angle Critic (hostile expert reader)
# ============================================================================
CRITIC_SYSTEM = """You are the most brutal reader on the internet. You've read Matt Levine, Ben Thompson, Adam Tooze, and Kyle Chayka every week for 5 years. You can spot a filler angle in one line.

Score EACH angle 1-10 on these 5 axes (total /50):

1. surprise_density — how many eyebrow-raising facts / re-framings per paragraph would this yield when written out. Boring angle = 1. Every sentence contains a fact I didn't have = 10.

2. non_obviousness — could a smart reader form this angle in 5 minutes of Googling? If yes, dead. 10 = angle is so non-obvious it required actual editorial insight to spot.

3. mechanism_not_headline — does it explain WHY (10) or just WHAT (1)? Headlines are cheap. Mechanisms are addictive.

4. narrative_tension — is there conflict, contradiction, reveal, or an unresolved question baked in? Angles without tension die in the swipe.

5. depth_runway — can this produce 12+ genuinely non-repeating follow-up rabbit-hole cards WITHOUT filler? Shallow angles hit a wall by card 4.

RULES:
- Any angle scoring below 35/50 total is dead.
- Pick the SINGLE strongest passing angle.
- If NONE pass, return winner_idx = -1 and write a SPECIFIC critique explaining what all 5 lacked in common (so the Ideator can regenerate with real signal, not vibes).

Return STRICT JSON:
{{
  "scores": [{{"idx": 0, "surprise_density": N, "non_obviousness": N, "mechanism_not_headline": N, "narrative_tension": N, "depth_runway": N, "total": N, "one_line_note": "..."}}, ...],
  "winner_idx": N_or_-1,
  "critique_if_all_fail": "<specific, actionable — name the pattern of weakness>"
}}"""


async def angle_critic(
    angles: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    if not angles:
        return None, "no angles supplied", {"elapsed_sec": 0}
    payload = [{"idx": i, **a} for i, a in enumerate(angles)]
    parsed, raw, elapsed = await _llm_json(
        CRITIC_SYSTEM,
        f"ANGLES:\n{json.dumps(payload, ensure_ascii=False, indent=2)}",
        model=LLM_MODEL,
        session_prefix="critic",
    )
    trace = {
        "elapsed_sec": round(elapsed, 2),
        "scores": (parsed or {}).get("scores", []),
        "raw_head": (raw or "")[:400],
    }
    if not parsed:
        return None, "critic returned nothing", trace
    winner_idx = parsed.get("winner_idx", -1)
    if isinstance(winner_idx, int) and 0 <= winner_idx < len(angles):
        winner = dict(angles[winner_idx])
        scores = next(
            (s for s in parsed.get("scores", []) if s.get("idx") == winner_idx), None
        )
        if scores:
            winner["_critic_score"] = scores
        return winner, None, trace
    return (
        None,
        parsed.get("critique_if_all_fail") or "all angles failed the bar",
        trace,
    )


async def pick_winning_angle(
    topic: str, persona_key: str, max_retries: int = 1
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Question Ladder → Angle Critic. Returns (winning_angle_or_None, full_trace)."""
    ladder_trace: Dict[str, Any] = {"pipeline": "question_ladder"}
    t0 = time.time()

    # ── Step 1: Generate ~30 candidate questions from 10 specialist lenses in parallel ──
    all_qs, qgen_trace = await generate_questions(topic)
    ladder_trace["step_1_question_generation"] = {
        **qgen_trace,
        "questions_by_lens": {},
    }
    for q in all_qs:
        lens = q["lens"]
        ladder_trace["step_1_question_generation"]["questions_by_lens"].setdefault(
            lens, []
        ).append(
            {
                "question": q["question"],
                "curiosity_hook": q["curiosity_hook"],
            }
        )
    log.info(
        "[newsroom]   ⚙ ladder: generated %d questions across %d lenses in %.1fs",
        len(all_qs),
        qgen_trace["total_questions"] > 0 and len(qgen_trace["per_lens_counts"]) or 0,
        qgen_trace["elapsed_sec"],
    )
    if not all_qs:
        ladder_trace["outcome"] = "no_questions"
        return None, ladder_trace

    # ── Step 2: Rank all questions, pick top 5 ──
    top_qs, ranker_trace = await rank_questions(all_qs)
    ladder_trace["step_2_question_ranking"] = {
        **ranker_trace,
        "top_5": [
            {
                "lens": q["lens_label"],
                "question": q["question"],
                "score": q.get("_rank_score", {}).get("total"),
                "curiosity_hook": q["curiosity_hook"],
            }
            for q in top_qs
        ],
    }
    log.info("[newsroom]   ⚙ ladder: ranked → top 5 questions selected")
    if not top_qs:
        ladder_trace["outcome"] = "no_top_questions"
        return None, ladder_trace

    # ── Step 3: Transform top-5 questions into angle specs ──
    angles, angle_elapsed = await questions_to_angles(top_qs, persona_key, topic)
    ladder_trace["step_3_angle_transform"] = {
        "elapsed_sec": round(angle_elapsed, 2),
        "angles": angles,
    }
    log.info(
        "[newsroom]   ⚙ ladder: transformed %d angles from top questions", len(angles)
    )
    if not angles:
        ladder_trace["outcome"] = "no_angles_from_questions"
        return None, ladder_trace

    # ── Step 4: Existing Angle Critic picks the strongest ──
    winner, critique, critic_trace = await angle_critic(angles)
    ladder_trace["step_4_critic"] = {
        **critic_trace,
        "winner_selected": bool(winner),
        "critique": critique,
    }
    if winner:
        winner["_attempts"] = 1
        winner["_via_question_ladder"] = True
        ladder_trace["final_winner"] = winner
        ladder_trace["outcome"] = "winner_selected"
        ladder_trace["total_elapsed_sec"] = round(time.time() - t0, 2)
        return winner, ladder_trace

    # ── Fallback: One more attempt with the old single-pass Ideator ──
    log.info("[newsroom]   ⚙ ladder critic REJECTED — fallback to single-pass ideator")
    angles_fb, fb_elapsed, _ = await angle_ideator(
        topic, persona_key, feedback=critique
    )
    ladder_trace["step_5_fallback_ideator"] = {
        "elapsed_sec": round(fb_elapsed, 2),
        "candidates": angles_fb,
    }
    winner_fb, fb_critique, fb_critic_trace = await angle_critic(angles_fb)
    ladder_trace["step_5_fallback_critic"] = fb_critic_trace
    if winner_fb:
        winner_fb["_attempts"] = 2
        winner_fb["_via_question_ladder"] = False
        ladder_trace["final_winner"] = winner_fb
        ladder_trace["outcome"] = "winner_selected_via_fallback"
        ladder_trace["total_elapsed_sec"] = round(time.time() - t0, 2)
        return winner_fb, ladder_trace

    ladder_trace["outcome"] = "no_winner"
    ladder_trace["total_elapsed_sec"] = round(time.time() - t0, 2)
    return None, ladder_trace


# ============================================================================
# STAGE 6 — Card Writer
# ============================================================================
CARD_WRITER_STEER_HEAD = """═══ WHO YOU'RE WRITING FOR ═══
Urban Indian 20-35 year olds reading on their phone during commutes, coffee breaks, and toilet trips. Curious, busy, phone-first. Class-12+ English but ZERO tolerance for jargon they didn't sign up for. They want to feel smart in 20 seconds, not overwhelmed in 2 minutes.

If a card is unclear in 3 seconds or uses jargon without a plain-English translation, they swipe past. Forever.

═══ THE ANGLE (persona-tuned — this is HOW you're framing, not WHO you're talking down to) ═══
This angle was chosen for readers with a **{persona_label}** interest lens, but the WRITING must be readable by any curious 22-year-old.

- angle kind: {angle_kind}
- angle title: {angle_title}
- angle thesis: {angle_thesis}
- anchor data point: {angle_anchor}

═══ NON-NEGOTIABLE RULES FOR EVERY CARD ═══

**RULE 1 — CATCHY, NON-REPETITIVE, "WAIT WHAT?" HEADLINE (max 8 words)**
Every one of the 12 headlines must be DIFFERENT in vibe from the other 11. No two headlines start the same way. No two headlines use the same construction ("X did Y", "The Z of W"). Vary rhythm, structure, and voice across the deck so the reader feels 12 distinct surprises.

The headline must give the reader a "wait — what?", "seriously?", or "no way" jolt. It should tease a surprising answer without revealing it. It should tell the reader WHAT the story is about (not tease with a metaphor).

GOOD (varied constructions, each a jolt):
  - "Zomato just quietly hiked delivery fees to 30%"
  - "The Adani deal Google didn't want anyone to see"
  - "1,072 Chrome bugs — in just two releases"
  - "India's IPO queue is 12 firms deep this month"
  - "Wait — Hermès really watches pork prices?"
  - "$130 billion in leases hid in Microsoft's small print"

BAD (never write like this):
  - "The bottleneck was clerical" (no topic, cryptic)
  - "The date that keeps returning" (metaphor)
  - "The X was Y" repeated across 4 different cards (repetitive)
  - "Everyone's talking AI. The missing word is..." (vague)

**RULE 2 — SENTENCE 1 = REAL-WORLD FACT WITH A REAL ENTITY, DATE, AND NUMBER**
The first sentence names WHAT HAPPENED and WHERE it happened. Ground the reader in real news within 15 words.

GOOD:
  - "Last week, Zomato raised its restaurant commissions from 22% to 30%."
  - "On July 28, Nithin Kamath's Zerodha said retail options trading fell 40% in one quarter."
  - "This week, Stripe agreed to buy AI-router startup OpenRouter for around $10 billion — 70x its annual revenue."

BAD:
  - "This week's tell is that the heaviest production compliance agents..." (starts with abstract)
  - "The market is moving from AI excitement to AI accountability..." (no entity)

**RULE 3 — PLAIN ENGLISH ONLY**
Every card is being read by a 22-year-old who does NOT work in finance / VC / policy / tech. If ANY of these terms appears WITHOUT an inline plain-English translation of 3-6 words, the card is dead:

FORBIDDEN unless immediately translated inline (in the same sentence):
FINANCE/VC: dry powder, exit market, secondary, offer-for-sale, OFS, cap table, dilution, follow-on, first cheque, Series A/B/C, IPO, unicorn, decacorn, TAM, cohort, retention, monetisation, cross-sell, downside protection, waterfall, LP, GP, DPI, IRR, MOIC, ARR, MRR, TCO, EBITDA, unit economics, aggregation theory, second-derivative, capital allocation, dry powder, exit velocity
REGULATION: SEBI (→ 'India's stock market regulator'), RBI (→ 'India's central bank'), MeitY (→ 'India's IT ministry'), CERT-In (→ 'the government's cyber-emergency team'), FIR (→ 'a police case'), KYC (→ 'ID verification'), Section 79 / safe harbour (→ 'the legal shield that means platforms aren't liable for user posts'), 10-K / S-1 (→ 'a company's official filing with regulators'), covenant leakage (→ 'lenders missing loan-violation red flags'), F&O (→ 'stock market bets called futures and options')
MARKET/POLICY: attribution (→ 'figuring out who was really behind it'), safe harbour, deemed dividend, resident-to-resident, slab rate, capital-flow, chokepoint, aggregation, unbundling, disintermediation, hedonic, arbitrage, tender, syndicate

ALSO AVOID as bare metaphors (only okay if immediately unpacked): bottleneck, plumbing, mechanism, dynamic, framework, lever, moat, headwinds, tailwinds, ripple effect

═══ TRANSLATE THE ANGLE ═══
The angle title/thesis above may contain jargon (e.g. "Dry Powder", "Exit Velocity", "S-1 Footnotes"). Your job is to DIRECTLY TRANSLATE these into everyday words in the actual cards. The angle is your *spine* — but the reader never sees the angle title, only your card headlines and body text. Never propagate the angle's jargon into a card.

GOOD ("SEBI" → "India's stock market regulator"):
  - "India's stock market regulator, SEBI, ordered Adani..."
BAD:
  - "The 8-K disclosure hints at capitalized R&D games."

**RULE 4 — 2-3 SHORT SENTENCES, ~40-70 WORDS TOTAL**
- Sentence 1: The concrete fact (who did what, when, with what number).
- Sentence 2: The insight — the "wait what" or the surprising angle nobody's saying.
- Sentence 3 (optional, only if it adds bite): The rabbit-hole hook that makes them want to swipe.

**RULE 5 — CONTAIN A SPECIFIC ANCHOR**
Every card MUST have at least one real: number, date, ₹/$/€ figure, %, named person, named company, named regulator, or named study.
BUT wrap it in narrative, not stats-blast.

**RULE 6 — DIFFICULTY GRADIENT (this is critical for Hot Now cards)**
The 12 cards form a LEARNING JOURNEY — start easy, end deep. NEVER start with the hardest card. Structure the sequence so a non-expert can enter and progressively go deeper.

- **Cards 1-3 (the HOOK)**: MUST be readable by anyone with no background knowledge. Introduce the topic, the surprising fact, and the counter-intuition. Assume ZERO expertise. Use everyday words. First sentence names the story in plain terms. Cards 1-3 are the "wait, seriously?" moment — the reader shouldn't need to know anything to feel the surprise.
- **Cards 4-8 (the BUILD)**: Introduce the mechanism, the actors, the moving parts. Add specific numbers, entities, and dates. If you need to use a technical term, explain it fully the first time. Cards here answer "how does this actually work?"
- **Cards 9-12 (the DEEP)**: For readers who stayed. Add nuance, second-order effects, historical parallel, contrarian objection. Card 12 opens a NEW rabbit-hole door — never a summary.

If a technical/insider term MUST appear, it appears in cards 5+, never in cards 1-3. Cards 1-3 should feel like a smart friend telling you the story over coffee.

**RULE 7 — CHAIN CARDS**
End each card at a point where the reader is silently forming their next question, then answer that question in the next card. Card 12 should open a new rabbit hole, not close the topic.

**RULE 8 — NEVER**
- Never write "In conclusion" / "In summary" / "Overall" / "This shows that"
- Never restate the topic name in the headline
- Never use passive voice as default
- Never open with a metaphor before the fact ("The tell was that..." "The real question is..." "Watch the...")
- Never use cliches: "the elephant in the room", "moving the needle", "seismic shift", "sea change"
- Never write a card the reader can't understand without a Wikipedia tab open

═══ HOW TO KNOW YOU'VE SUCCEEDED ═══
Read your card aloud. If a friend asked "what's the story?" while listening, would they know what real event you're talking about within 3 seconds? If no → rewrite.
"""


async def write_cards(
    topic: str,
    angle: Dict[str, Any],
    persona_key: str,
    tavily: TavilyClient,
    llm_key: str,
    writer_hint: Optional[str] = None,
    cached_bundle: Optional[ResearchBundle] = None,
) -> Tuple[List[Dict[str, Any]], ResearchBundle, Dict[str, Any]]:
    t0 = time.time()
    p = PERSONAS[persona_key]
    research_query = f"{topic} — {angle.get('title', '')} — {angle.get('thesis', '')}"

    research_trace = {
        "query": research_query,
        "attempts": [],
        "used_cached_bundle": bool(cached_bundle),
    }
    bundle: Optional[ResearchBundle] = cached_bundle
    if bundle is None or not bundle.sources:
        for is_news, days, extract_n, yt_n in [(True, 7, 5, 2), (False, 30, 4, 1)]:
            att_t0 = time.time()
            try:
                bundle = await deep_research(
                    topic=research_query,
                    tavily_client=tavily,
                    llm_key=llm_key,
                    extract_top_n=extract_n,
                    youtube_top_n=yt_n,
                    is_news=is_news,
                    news_days=days,
                )
            except Exception:
                log.exception("[newsroom] write_cards: deep_research raised")
                bundle = ResearchBundle(topic=research_query)
            research_trace["attempts"].append(
                {
                    "is_news": is_news,
                    "news_days": days,
                    "sources": len(bundle.sources) if bundle else 0,
                    "angles": len(bundle.angles) if bundle else 0,
                    "youtube": len(getattr(bundle, "youtube", []) or []),
                    "elapsed_sec": round(time.time() - att_t0, 2),
                }
            )
            if bundle and bundle.sources:
                break
            await asyncio.sleep(0.5)

    if not bundle or not bundle.sources:
        research_trace["final_sources"] = 0
        return (
            [],
            bundle or ResearchBundle(topic=research_query),
            {
                "research": research_trace,
                "writer": {"skipped": True, "reason": "no research sources"},
                "elapsed_sec": round(time.time() - t0, 2),
            },
        )

    research_trace["final_sources"] = len(bundle.sources)
    research_trace["source_urls"] = [s.get("url") for s in bundle.sources[:15]]
    research_trace["research_angles"] = [a.get("name") for a in bundle.angles]

    steer = CARD_WRITER_STEER_HEAD.format(
        persona_label=p["label"],
        angle_kind=angle.get("kind", ""),
        angle_title=angle.get("title", ""),
        angle_thesis=angle.get("thesis", ""),
        angle_anchor=angle.get("anchor_data_point", ""),
    )
    if writer_hint:
        steer += (
            "\n\n═══ EDITORIAL REWRITE HINT (address this NOW — the previous draft was rejected for these specific reasons) ═══\n"
            + writer_hint
            + "\n\nRewrite the flagged cards according to the hint above. Keep Rules 1-7 above intact — especially the plain-English and catchy-headline bar."
        )

    writer_t0 = time.time()
    items: List[Dict[str, Any]] = []
    try:
        items = await generate_insights_from_bundle(
            bundle=bundle,
            llm_key=llm_key,
            count=12,
            behavior_context="",
            steer=steer,
            exclude_headlines=[],
            model_provider=LLM_PROVIDER,
            model_name=LLM_MODEL,
            persona=p["label"],
            voice_hint=p["voice"],
            angle_hint=angle.get("title", ""),
            is_news=True,
        )
    except Exception:
        log.exception("[newsroom] card writer raised")

    writer_trace = {
        "cards_produced": len(items),
        "elapsed_sec": round(time.time() - writer_t0, 2),
    }
    return (
        items,
        bundle,
        {
            "research": research_trace,
            "writer": writer_trace,
            "elapsed_sec": round(time.time() - t0, 2),
        },
    )


# ============================================================================
# STAGE 6.5 — Readability + Engagement Critic (per-card)
# ============================================================================
READABILITY_CRITIC_SYSTEM = """You are 3 different readers reading the same 12-card sequence:

READER A — Priya, 25, marketing manager in Bombay. Reads Deuxmoi + The Ken. Zero patience for jargon.
READER B — Rohit, 28, engineer at a scale-up in Bangalore. Reads HN. Skeptical of hype but rewards specificity.
READER C — Ananya, 23, MBA student in Delhi. Reads Nithin Kamath tweets + The Signal. Wants to feel smart at dinner-parties.

For EACH card, silently ask (as all 3 readers):

1. **clarity** (1-10) — Do I know what real-world story this card is about within 3 seconds of reading the headline + first sentence? Score 10 if instantly obvious, 1 if I need to re-read twice.

2. **plain_english** (1-10) — Can I understand every word without a dictionary tab? Every acronym / insider term must be explained inline in plain English. Score 10 if reads like a smart friend telling me the story, 1 if reads like a Bloomberg terminal.

3. **engagement** (1-10) — Would I swipe to the next card because THIS one made me curious? Score 10 if I'm dying to know what's next, 1 if I'm swiping away out of boredom.

4. **wow_factor** (1-10) — Would I say "oh damn" or "wait what" or want to text this to a friend? Score 10 if this is a group-chat forward, 1 if it's a snooze.

═══ HARD BATCH RULES ═══

The batch FAILS the readability bar ONLY if the batch is systematically bad:
- 4 OR MORE cards score clarity < 6 (topic unclear to a 22-year-old non-expert)
- 4 OR MORE cards score plain_english < 6 (heavy jargon throughout)
- Average engagement across 12 cards < 5.5
- ZERO cards score wow_factor >= 7 (batch has no aha moment at all)

Otherwise PASS — even if some cards have minor jargon, at least the reader can extract meaning from most cards. FLAG problem card indices so we can log them.

Return STRICT JSON:
{{
  "per_card": [
    {{"idx": 0, "clarity": N, "plain_english": N, "engagement": N, "wow_factor": N, "problem": "<one-line diagnosis if any score < 6, else empty>"}},
    ...12 total...
  ],
  "avg_engagement": N,
  "wow_count": N,
  "problem_indices": [ints],
  "verdict": "PASS" | "REJECT",
  "rewrite_hint": "<specific instruction for the Writer to fix the flagged cards, e.g., 'Card 3 uses covenant leakage without explaining. Rewrite with plain language.'>"
}}"""


async def readability_engagement_critic(
    cards: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], float]:
    payload = [
        {"idx": i, "headline": c.get("headline", ""), "text": c.get("text", "")}
        for i, c in enumerate(cards)
    ]
    parsed, raw, elapsed = await _llm_json(
        READABILITY_CRITIC_SYSTEM,
        f"CARD SEQUENCE:\n{json.dumps(payload, ensure_ascii=False, indent=2)}",
        model=LLM_MODEL,
        session_prefix="readcritic",
    )
    if not parsed:
        return {
            "verdict": "REJECT",
            "problem_indices": [],
            "avg_engagement": 0,
            "wow_count": 0,
            "per_card": [],
            "rewrite_hint": "critic returned nothing",
        }, elapsed
    return parsed, elapsed


# ============================================================================
# STAGE 7 — Chief Editor
# ============================================================================
CHIEF_SYSTEM = """You are the Chief Editor of DruvInsight — the final gate before a card sequence reaches urban Indian 20-35 year olds scrolling on their phones.

Read the 12-card sequence as THE READER. Grade honestly:

1. **aha_count** — How many cards have a genuine "oh damn / wait what" moment? Count them.
2. **catchy_headlines_count** — How many of the 12 headlines are catchy hooks that name a real entity/number/event (NOT cryptic metaphors like "The bottleneck was clerical")? Count them.
3. **opens_curiosity_at_end** — Does card 12 open a new door or close the topic?
4. **chained** — Do most cards form a continuous chain?

Return PASS unless the batch is genuinely unusable:
- REJECT ONLY if: aha_count < 3 AND catchy_headlines_count < 6

Flag specific problem card indices (0-based) for logging.

Return STRICT JSON:
{{
  "verdict": "PASS" | "REJECT",
  "aha_count": N,
  "catchy_headlines_count": N,
  "opens_curiosity_at_end": true/false,
  "chained": true/false,
  "problem_indices": [ints],
  "notes": "<short editorial note>"
}}"""


async def chief_editor(cards: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], float]:
    payload = [
        {"idx": i, "headline": c.get("headline", ""), "text": c.get("text", "")}
        for i, c in enumerate(cards)
    ]
    parsed, raw, elapsed = await _llm_json(
        CHIEF_SYSTEM,
        f"CARD SEQUENCE:\n{json.dumps(payload, ensure_ascii=False, indent=2)}",
        model=LLM_MODEL,
        session_prefix="chief",
    )
    if not parsed:
        return {
            "verdict": "REJECT",
            "problem_indices": [],
            "notes": "chief returned nothing",
            "aha_count": 0,
            "opens_curiosity_at_end": False,
            "chained": False,
        }, elapsed
    return parsed, elapsed


# ============================================================================
# ORCHESTRATOR — one topic end-to-end, with full trace
# ============================================================================
async def process_one_topic(
    signal: Dict[str, Any],
    tavily: TavilyClient,
    llm_key: str,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    topic = signal.get("topic") or signal.get("title")
    topic_t0 = time.time()
    trace: Dict[str, Any] = {
        "topic": topic,
        "signal": {
            k: signal.get(k)
            for k in (
                "title",
                "quality_score",
                "score_breakdown",
                "sources",
                "filter_reasoning",
            )
        },
        "stages": {},
    }
    log.info("[newsroom] processing topic: %s", topic)

    persona_key, why, router_elapsed = await persona_router(topic)
    trace["stages"]["router"] = {
        "persona_key": persona_key,
        "persona_label": PERSONAS[persona_key]["label"],
        "why": why,
        "elapsed_sec": round(router_elapsed, 2),
    }
    log.info("[newsroom]   → persona: %s (%s)", persona_key, why)

    winner, ideation_trace = await pick_winning_angle(topic, persona_key, max_retries=2)
    trace["stages"]["ideation"] = ideation_trace
    if not winner:
        trace["outcome"] = "dropped_at_critic"
        trace["total_elapsed_sec"] = round(time.time() - topic_t0, 2)
        log.info("[newsroom]   ✗ no angle passed the critic — dropping")
        return None, trace
    log.info("[newsroom]   → angle: [%s] %s", winner.get("kind"), winner.get("title"))

    cards, bundle, writer_trace = await write_cards(
        topic, winner, persona_key, tavily, llm_key
    )
    trace["stages"]["writer"] = writer_trace
    if not cards:
        trace["outcome"] = "dropped_at_writer"
        trace["total_elapsed_sec"] = round(time.time() - topic_t0, 2)
        log.info("[newsroom]   ✗ writer returned nothing")
        return None, trace

    # ── Stage 6.5: Readability + Engagement Critic (one revision loop) ──
    read_v, read_elapsed = await readability_engagement_critic(cards)
    trace["stages"]["readability_v1"] = {
        **read_v,
        "elapsed_sec": round(read_elapsed, 2),
    }
    log.info(
        "[newsroom]   ⌛ readability v1: verdict=%s avg_eng=%.1f wow=%d problems=%s",
        read_v.get("verdict"),
        float(read_v.get("avg_engagement") or 0),
        int(read_v.get("wow_count") or 0),
        read_v.get("problem_indices"),
    )
    if read_v.get("verdict") != "PASS":
        log.info(
            "[newsroom]   ↺ readability REJECTED — revision with hint: %s",
            read_v.get("rewrite_hint", ""),
        )
        hint = (
            read_v.get("rewrite_hint")
            or "Rewrite the flagged cards in plainer English with clearer topic anchors."
        )
        cards2, bundle2, writer_trace2 = await write_cards(
            topic,
            winner,
            persona_key,
            tavily,
            llm_key,
            writer_hint=hint,
            cached_bundle=bundle,
        )
        trace["stages"]["writer_revision"] = writer_trace2
        if cards2:
            cards, bundle = cards2, bundle2
            read_v2, read_elapsed2 = await readability_engagement_critic(cards)
            trace["stages"]["readability_v2"] = {
                **read_v2,
                "elapsed_sec": round(read_elapsed2, 2),
            }
            log.info(
                "[newsroom]   ⌛ readability v2: verdict=%s", read_v2.get("verdict")
            )
            if read_v2.get("verdict") != "PASS":
                trace["outcome"] = "dropped_at_readability_critic"
                trace["total_elapsed_sec"] = round(time.time() - topic_t0, 2)
                log.info(
                    "[newsroom]   ✗ readability REJECTED on revision — dropping topic"
                )
                return None, trace

    verdict, chief_elapsed = await chief_editor(cards)
    trace["stages"]["chief_v1"] = {**verdict, "elapsed_sec": round(chief_elapsed, 2)}
    if verdict.get("verdict") != "PASS":
        log.info("[newsroom]   ↺ chief REJECTED — revision")
        chief_hint = (
            verdict.get("notes")
            or "Rewrite flagged cards with catchier headlines and plainer English."
        )
        cards, bundle, writer_trace3 = await write_cards(
            topic,
            winner,
            persona_key,
            tavily,
            llm_key,
            writer_hint=chief_hint,
            cached_bundle=bundle,
        )
        trace["stages"]["writer_revision2"] = writer_trace3
        if cards:
            verdict, chief_elapsed2 = await chief_editor(cards)
            trace["stages"]["chief_v2"] = {
                **verdict,
                "elapsed_sec": round(chief_elapsed2, 2),
            }
        if verdict.get("verdict") != "PASS":
            trace["outcome"] = "dropped_at_chief_editor"
            trace["total_elapsed_sec"] = round(time.time() - topic_t0, 2)
            log.info("[newsroom]   ✗ chief REJECTED on revision")
            return None, trace

    p = PERSONAS[persona_key]
    published = {
        "id": str(uuid.uuid4()),
        "topic": topic,
        "persona_key": persona_key,
        "persona_label": p["label"],
        "angle": winner,
        "cards": cards,
        "sources": [s.get("url") for s in bundle.sources[:15]],
        "quality_scores": {
            "filter": signal.get("score_breakdown"),
            "critic": winner.get("_critic_score"),
            "chief": {
                k: verdict.get(k)
                for k in ("aha_count", "opens_curiosity_at_end", "chained", "notes")
            },
        },
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    trace["outcome"] = "published"
    trace["published_id"] = published["id"]
    trace["total_elapsed_sec"] = round(time.time() - topic_t0, 2)
    log.info("[newsroom]   ✓ PUBLISHED (%.1fs)", trace["total_elapsed_sec"])
    return published, trace


# ============================================================================
# Daily run — persists to db.newsroom_runs (trace) + db.hot_now_cards (output)
# ============================================================================
async def run_newsroom(
    tavily: TavilyClient, db, llm_key: str, keep_n: int = 15
) -> Dict[str, Any]:
    run_id = str(uuid.uuid4())
    run_started_at = datetime.now(timezone.utc).isoformat()
    run_t0 = time.time()
    log.info("[newsroom] === run %s starting ===", run_id)

    run_doc: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": run_started_at,
        "keep_n": keep_n,
        "stages": {},
        "topics": [],
    }

    # Stage 1
    signals, scout_trace = await signal_scout(tavily)
    run_doc["stages"]["signal_scout"] = {
        **scout_trace,
        "signals": signals,
    }
    log.info("[newsroom] Stage 1: %d clustered signals", len(signals))
    if not signals:
        run_doc["finished_at"] = datetime.now(timezone.utc).isoformat()
        run_doc["total_elapsed_sec"] = round(time.time() - run_t0, 2)
        run_doc["outcome"] = "no_signals"
        await db.newsroom_runs.insert_one(run_doc)
        return {"ok": False, "reason": "no signals", "run_id": run_id}

    # Stage 2
    kept, filter_trace = await topic_filter(signals, keep_n=keep_n)
    run_doc["stages"]["topic_filter"] = filter_trace
    log.info("[newsroom] Stage 2: %d topics kept", len(kept))
    if not kept:
        run_doc["finished_at"] = datetime.now(timezone.utc).isoformat()
        run_doc["total_elapsed_sec"] = round(time.time() - run_t0, 2)
        run_doc["outcome"] = "no_topics_passed"
        await db.newsroom_runs.insert_one(run_doc)
        return {"ok": False, "reason": "no topics passed filter", "run_id": run_id}

    # Stages 3-7 in parallel (bounded)
    sem = asyncio.Semaphore(4)

    async def _bounded(sig):
        async with sem:
            try:
                return await process_one_topic(sig, tavily, llm_key)
            except Exception:
                log.exception("[newsroom] topic pipeline failed: %s", sig.get("topic"))
                return None, {"topic": sig.get("topic"), "outcome": "exception"}

    results = await asyncio.gather(
        *[_bounded(s) for s in kept], return_exceptions=False
    )

    published: List[Dict[str, Any]] = []
    for pub, tr in results:
        run_doc["topics"].append(tr)
        if pub:
            published.append(pub)

    run_stamp = datetime.now(timezone.utc).isoformat()
    if published:
        for doc in published:
            doc["run_id"] = run_id
            doc["run_stamp"] = run_stamp
        await db.hot_now_cards.insert_many(published)

    run_doc["finished_at"] = run_stamp
    run_doc["total_elapsed_sec"] = round(time.time() - run_t0, 2)
    run_doc["published_count"] = len(published)
    run_doc["outcome"] = "success" if published else "all_topics_dropped"
    await db.newsroom_runs.insert_one(run_doc)

    log.info(
        "[newsroom] === run %s done: %d published in %.1fs ===",
        run_id,
        len(published),
        run_doc["total_elapsed_sec"],
    )

    return {
        "ok": True,
        "run_id": run_id,
        "signals": len(signals),
        "kept": len(kept),
        "published": len(published),
        "elapsed_sec": run_doc["total_elapsed_sec"],
        "topics": [
            {
                "topic": r["topic"],
                "persona": r["persona_label"],
                "angle": r["angle"]["title"],
            }
            for r in published
        ],
    }
