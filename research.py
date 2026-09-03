"""
Deep research pipeline for Insight Swipe.

Runs a multi-stage research flow:
  1. LLM decomposes the topic into 5-6 distinct research angles.
  2. In parallel: for each angle, run a Tavily search (advanced depth) —
     plus one Tavily news search for current-shifts / recent-events angles.
  3. Optionally extract full page text for the highest-signal URLs (Tavily extract).
  4. Optionally fetch YouTube transcripts for any strong YouTube URLs
     that appeared in search results.
  5. Return a rich research bundle used to generate the 12 insights.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from emergentintegrations.llm.chat import LlmChat, UserMessage
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (NoTranscriptFound,
                                            TranscriptsDisabled,
                                            VideoUnavailable)

log = logging.getLogger(__name__)


ANGLE_PLANNER_SYSTEM = """You are the research planner behind Insight Swipe — a deep-research feed.

Given a user topic, produce 5–6 distinct research angles that together give a genuinely wow-inducing understanding of the topic. Cover, when relevant:
- Historical origins / founding story / hidden backstory
- Market structure, unit economics, revenue breakdown, size, growth data
- Key players, strategies, unusual competitive moves, business model quirks
- Consumer psychology, behavioural science, why people buy / adopt
- Current market shifts, recent policy / regulatory change, macro trends (last 2 years)
- Counter-intuitive data, surprising statistics, contrarian findings
- Future outlook / emerging technology / expected inflection points

Rules:
- Angles must be DIFFERENT from each other. No overlap.
- For each angle, produce a short, punchy Tavily-ready search query (like a smart Googler would type).
- Mark ONE angle as `is_current: true` — the one that most benefits from recent-news search.
- Output strict JSON only, no prose."""


RAIL_PLANNER_SYSTEM = """You are the audience director for Insight Swipe.

You receive ONE interest and must design ONE vivid AUDIENCE PERSONA who lives and breathes that interest — the exact archetype whose daily reading, jargon and taste should shape every insight card generated on this rail.

STRICT RULES (violating any = failure):
- Return exactly ONE persona. NEVER cluster with other interests. NEVER hedge with "and other topics".
- The persona is a VIVID archetype with a real reading list, not a generic label.
  Examples of the calibre expected:
    · Celebrity Gossip → "Gossip Girl-style narrator who refreshes Deuxmoi, PageSix, r/Fauxmoi, Pop Crave hourly and reads Vanity Fair long-reads on Sundays"
    · World Politics → "Ministry-of-External-Affairs desk officer who reads Foreign Affairs, FT, Politico, RAND, and every UNSC communiqué"
    · AI → "AI researcher who reads arXiv daily, tracks Anthropic/OpenAI/DeepMind, listens to Dwarkesh & Latent Space"
- Under NO circumstance does this persona also care about the user's OTHER picked interests. Content on this rail must feel like it was written FOR THIS PERSONA ONLY.

For the rail also produce:
- `search_hint`: comma-separated sites/creators/subreddits this persona trusts (used as a Tavily query prefix so search leans toward that persona's surfaces).
- `voice_hint`: how insights should SOUND — tone, jargon level, references, energy.
- `angle_hint`: the content angles this persona actively hunts for (drama-mechanics for gossip; sanctions/treaties for world politics; scaling-laws/benchmarks for AI; unit-economics for finance).

Output strict JSON only. No markdown fences. No prose outside JSON."""


RAIL_PLANNER_TEMPLATE = """USER INTEREST (single, do NOT blend): {interest}
NEWS MODE: {is_news}

Return strict JSON matching EXACTLY this shape (only ONE rail):
{{
  "rail": {{
    "name": "<persona label, 2-4 words>",
    "persona": "<vivid archetype paragraph — reading list, listening list, jargon, taste — specific to THIS one interest>",
    "search_hint": "<comma-separated trusted sites/creators/subreddits>",
    "voice_hint": "<tone, jargon level, references, energy>",
    "angle_hint": "<the specific content angles this persona hunts>"
  }}
}}"""


ANGLE_PLANNER_TEMPLATE = """TOPIC: {topic}

Return strict JSON matching:
{{
  "angles": [
    {{ "name": "Origin & hidden backstory", "query": "specific search query", "is_current": false }},
    {{ "name": "Market structure & unit economics", "query": "...", "is_current": false }},
    ...
  ]
}}
5–6 angles total."""


INSIGHT_SYSTEM_PROMPT = """You are the research engine behind Insight Swipe — a TikTok-style feed of insights that people would only discover after hours of deep research. Your output is used as an alternative to manual research on market dynamics, business strategy, case studies, government policy, behavioural shifts, and industry evolution.

═══ NON-NEGOTIABLE QUALITY BAR ═══

Every single insight MUST satisfy ALL of these:

1. **Contain a hard specific**: a number, percentage, dollar/rupee figure, exact year, market size, named study, specific person, specific company, specific regulation, or law. NO abstract adjectives like "huge", "many", "recently", "significant". If you cannot ground it in a specific, DO NOT write the insight — pick a different angle instead.

2. **Non-obvious payoff**: the insight must be something a smart casual reader would NOT already know. Reject headline facts (e.g. "Nintendo makes video games") in favor of mechanisms (e.g. "Nintendo bought 60% of the game industry's silicon chip supply in 1990 to starve competitors").

3. **Punchy first sentence**: no throat-clearing. Start with the striking fact, not "It is interesting to note that...".

4. **Simple language**: no jargon. If a term is technical, explain it in-line. Aim for a smart friend's voice, not a McKinsey deck.

5. **2–4 sentences**: dense but readable. No essays. No single-sentence tweets either.

6. **Grounded in the SOURCES list**: never invent numbers. If a specific isn't in the sources, use broader-knowledge only for the mechanism (source_index = null in that case), never for the number.

═══ THE 12 INSIGHTS TOGETHER MUST COVER ═══

You must produce a diverse batch. Across the 12, ensure you include:
  •  ≥ 1 historical-origin / founding-story angle
  •  ≥ 1 unit-economics / market-structure / revenue angle
  •  ≥ 1 consumer-psychology / behavioural-science angle
  •  ≥ 1 policy / regulation / government-action angle
  •  ≥ 1 current-shift (last 2 years) angle
  •  ≥ 1 counter-intuitive / contrarian / underrated angle
  •  ≥ 3 different named entities (people, companies, agencies)

If a mandatory angle has no supporting evidence in the sources, still cover it — but only with a real specific.

═══ OUTPUT FORMAT ═══

Return STRICT JSON only. No markdown fences. No prose outside JSON.
{
  "insights": [
    { "headline": "≤ 8-word hook", "text": "2–4 sentence insight", "source_index": 1, "angle": "history|economics|psychology|policy|current|contrarian|players|mechanism" },
    ...
  ]
}

`source_index` is 1-based into the SOURCES list, or null if drawn from general knowledge. `angle` tags the insight for angle diversity."""


@dataclass
class ResearchBundle:
    topic: str
    angles: List[dict] = field(default_factory=list)
    sources: List[dict] = field(default_factory=list)  # unique, top-N
    extracts: List[dict] = field(default_factory=list)  # {url, title, content}
    youtube: List[dict] = field(
        default_factory=list
    )  # {url, title, transcript_excerpt}
    tavily_answer: str = ""
    quota_exceeded: bool = False  # True if Tavily returned a plan/quota error


def _extract_json(raw: str) -> dict:
    s = raw.strip().strip("`")
    if s.startswith("json"):
        s = s[4:].strip()
    if "{" in s and "}" in s:
        s = s[s.find("{") : s.rfind("}") + 1]
    return json.loads(s)


async def plan_research_angles(
    topic: str,
    llm_key: str,
    model_provider: str = "openai",
    model_name: str = "gpt-5.6-luna",
    persona: Optional[str] = None,
    angle_hint: Optional[str] = None,
    is_news: bool = False,
) -> List[dict]:
    """Ask the LLM to decompose the topic into 5-6 research angles.

    If a persona is provided, the angles are tailored to what that specific
    audience persona would find engaging (e.g., a Gossip Enthusiast wants
    relationship drama / PR moves, not academic history)."""
    session_id = f"angles-{uuid.uuid4()}"

    system = ANGLE_PLANNER_SYSTEM
    if persona:
        system += (
            f"\n\nAUDIENCE PERSONA: {persona}\n"
            "Every angle you generate MUST be one this persona would eagerly consume. "
            "Do NOT produce generic textbook angles — produce angles calibrated to this "
            "persona's actual daily interests and reading habits."
        )
        if angle_hint:
            system += f"\nPRIORITIZE angles like: {angle_hint}"
    if is_news:
        system += (
            "\n\nNEWS MODE: at least 3 of the 5–6 angles must target the LAST 7 DAYS "
            "specifically. Their Tavily queries should include recent time cues "
            "(this week, past 7 days, latest, breaking) so the search surfaces "
            "fresh material — not evergreen background."
        )

    chat = LlmChat(
        api_key=llm_key, session_id=session_id, system_message=system
    ).with_model(model_provider, model_name)
    text = await chat.send_message(
        UserMessage(text=ANGLE_PLANNER_TEMPLATE.format(topic=topic))
    )
    try:
        parsed = _extract_json(text)
        angles = parsed.get("angles", [])
        # Sanity: keep only well-formed entries
        return [
            {
                "name": (a.get("name") or "").strip(),
                "query": (a.get("query") or "").strip(),
                "is_current": bool(a.get("is_current", False)),
            }
            for a in angles
            if a.get("query")
        ][:6]
    except Exception:
        log.exception("Angle planning JSON parse failed: %s", text[:400])
        return []


# ============ Hand-curated persona library ============
# Each key is normalized-lowercase of the interest label. If a match is
# found the rail uses this deterministic, high-fidelity persona instead of
# a fresh LLM call. Fall through to LLM planning for anything not listed.
PERSONA_LIBRARY: dict = {
    "ai": {
        "name": "AI Researcher",
        "persona": "AI researcher / ML engineer — reads arXiv daily, tracks OpenAI, Anthropic, Google DeepMind, Meta FAIR. Listens to Dwarkesh Patel, Latent Space, No Priors, Lex Fridman. Knows transformer internals, scaling laws, RLHF/DPO, MoE routing, benchmark leaderboards (MMLU, HumanEval, SWE-Bench).",
        "search_hint": "arXiv, OpenAI blog, Anthropic research, Google DeepMind, Meta AI, Hugging Face papers, LessWrong, Alignment Forum, Import AI, Interconnects, Dwarkesh Patel, Latent Space",
        "voice_hint": "sharp, technical when it earns the payoff, never gatekeeping. Cites specific model versions (Claude 4.5 Sonnet, GPT-5.4, Gemini 3.1), parameter counts, benchmark scores, dataset sizes, and training compute in FLOPs.",
        "angle_hint": "frontier-model releases, scaling-law inflections, benchmark leapfrogs, safety/alignment breakthroughs, training-compute economics, agent frameworks, open-vs-closed weight wars, evals-as-a-market",
    },
    "celebrity gossip": {
        "name": "Gossip Girl",
        "persona": "Gossip Girl-style narrator crossed with Deuxmoi's inbox. Refreshes Deuxmoi, PageSix, r/Fauxmoi, Pop Crave, Pop Base, TMZ, JustJared and DListed hourly. Reads Vanity Fair long-reads and Anne Helen Petersen on Sundays. Knows every situationship arc, PR crisis, hard-launch and industry-plant theory. On first-name basis with every A-lister and never explains who they are.",
        "search_hint": "Deuxmoi, PageSix, r/Fauxmoi, Pop Crave, Pop Base, TMZ, People, Us Weekly, JustJared, Vanity Fair, Vulture, DListed, Rob Shuter, The Cut",
        "voice_hint": "arch, knowing, receipt-dropping (dates, venues, insider tips). Zero cringe. Never over-explains. Drops names like 'Timmy' and 'Kylie' with easy familiarity.",
        "angle_hint": "PR-crisis playbooks, situationships / hard-launches / soft-launches, tour financials, feud timelines, industry-plant lore, stylist power moves, red-carpet strategy, brand deals & morality clauses, tabloid deal-flow",
    },
    "world politics": {
        "name": "Foreign-Policy Desk",
        "persona": "Ministry-of-External-Affairs / State-Department desk officer. Reads Foreign Affairs, Foreign Policy, The Economist, Reuters diplomacy wire, FT world, Politico, War on the Rocks, RAND, CSIS, Chatham House. Tracks bilateral summits, sanctions lists (SDN, Entity List), treaty texts, defence budgets and UNSC vote coalitions.",
        "search_hint": "Foreign Affairs, Foreign Policy, Reuters, Financial Times world, Politico, War on the Rocks, RAND, CSIS, Chatham House, MEA statements, State Department briefings, ORF, IISS",
        "voice_hint": "measured, treaty-article specific, cites the sanctions programme, the named diplomat, the summit date & communiqué wording. Casually uses NATO, QUAD, ASEAN, UNSC, MEA acronyms.",
        "angle_hint": "sanctions regimes, treaty-text disputes, back-channel diplomacy, defence pacts, aid flows and their strings, chip / rare-earth export controls, UN coalitions, foreign-election interference dynamics",
    },
    "politics": {
        "name": "Foreign-Policy Desk",
        "persona": "Same desk officer as World Politics — reads Foreign Affairs, FT, Politico, RAND, CSIS. Tracks summits, sanctions, treaty texts, defence budgets.",
        "search_hint": "Foreign Affairs, Foreign Policy, Reuters, FT world, Politico, War on the Rocks, RAND, CSIS, Chatham House, MEA, State Department",
        "voice_hint": "measured, treaty-specific, cites named diplomats and dates, uses NATO/UNSC/QUAD casually",
        "angle_hint": "sanctions, treaties, backchannels, defence pacts, foreign aid, coalition politics",
    },
    "finance": {
        "name": "Buyside Analyst",
        "persona": "Buyside analyst — Bloomberg terminal open, reads FT, WSJ, Matt Levine's Money Stuff, Bill Ackman letters, 10-Ks, 13-F filings, activist letters, earnings-call transcripts. Talks in bps, yields, spreads, multiples, and cap tables.",
        "search_hint": "FT, WSJ, Bloomberg, Matt Levine Money Stuff, SEC EDGAR, S&P Global, Moody's, Financial Times markets, Business Insider markets",
        "voice_hint": "precise, uses bps not %, cites tickers, fiscal year (FY26), and ratios (EV/EBITDA, P/S, ND/EBITDA). Comfortable with 'the Fed', 'the ECB', DXY, VIX, MOVE index.",
        "angle_hint": "rate cycles & duration risk, credit spreads and default cycles, activist positions, buyback vs dividend policy, cross-border M&A, IPO plumbing, distressed debt, fintech disruption",
    },
    "business": {
        "name": "Strategy Consultant",
        "persona": "Strategy consultant — reads HBR, Stratechery, The Information, McKinsey Quarterly, BCG Insights, CB Insights, Not Boring, Business of Fashion. Draws 2×2s in their sleep. Speaks Porter, Wardley, jobs-to-be-done.",
        "search_hint": "HBR, Stratechery, The Information, McKinsey Quarterly, BCG Insights, CB Insights, Not Boring by Packy McCormick, Business of Fashion",
        "voice_hint": "frameworks-first, backs claims with revenue splits, unit economics and market-share numbers. Cites named exec moves.",
        "angle_hint": "category creation, competitive moats, distribution monopolies, unbundling/rebundling, platform vs pipeline, org design, private-label vs house-brand, franchise economics",
    },
    "startups": {
        "name": "Early-Stage VC",
        "persona": "Early-stage VC / founder — TechCrunch, The Information, Sacra, Hacker News, YC Startup School, a16z, Sequoia, Benchmark, First Round Review. Tracks rounds, valuations, dilution, seed→A conversion.",
        "search_hint": "TechCrunch, The Information, Sacra, a16z blog, Sequoia perspectives, YC founder library, Hacker News, First Round Review, Not Boring",
        "voice_hint": "insider, uses ARR / MRR / CAC / LTV / burn multiple without over-explaining. Names funds (Benchmark, Founders Fund) and partners casually.",
        "angle_hint": "seed→Series-A conversion, category collapses, GTM motions (PLG vs sales-led), founder-market fit, cap-table disasters, secondaries, acqui-hires vs true acquisitions",
    },
    "communism": {
        "name": "Political Theorist",
        "persona": "Political theorist — has read Marx, Lenin, Mao, Gramsci, Žižek. Reads Jacobin, Verso Books, LRB, Monthly Review, Tribune, Chapo archives. Knows 20th-century labour history and current DSA / CPI(M) / KKE debates.",
        "search_hint": "Jacobin, LRB, Verso Books, Monthly Review, Tribune Magazine, Novara Media, Marxists.org, Chapo Trap House archives",
        "voice_hint": "theory-fluent, cites the essay title and year, distinguishes Trotskyism from Stalinism casually, uses 'praxis' with a straight face.",
        "angle_hint": "party-line splits, tankie-vs-libleft, worker-cooperative experiments, Cuba/Vietnam economic pivots, DSA electoral experiments, Kerala model, Chinese SOE reforms",
    },
    "socialism": {
        "name": "Democratic-Socialist Wonk",
        "persona": "Democratic-socialist policy wonk — Jacobin, Dissent, The Nation, In These Times, Nordic policy briefs. Tracks NHS-vs-privatization, UBI trials, worker co-ops, sectoral bargaining.",
        "search_hint": "Jacobin, Dissent, The Nation, In These Times, The Guardian opinion, LSE blogs, Nordic policy briefs, Roosevelt Institute",
        "voice_hint": "policy-first, cites the pilot city, the UBI amount, effective tax rates, Gini coefficients before/after.",
        "angle_hint": "UBI trials, sectoral bargaining, co-op finance models, healthcare-cost delta by system, wealth-tax design pitfalls, Nordic-model actualities vs myths",
    },
    "capitalism": {
        "name": "Free-Market Historian",
        "persona": "Free-market historian — WSJ opinion, The Economist, Reason, Cato, Mercatus, AEI, EconLog. Fluent in Hayek, Friedman, Coase, Ostrom. Believes prices carry information.",
        "search_hint": "The Economist, WSJ opinion, Reason, Cato, Mercatus, AEI, National Review economics, EconLog, Marginal Revolution",
        "voice_hint": "price-signals-first, cites Coasean externalities, elasticity numbers, named studies (Angrist, Nakamura, Chetty).",
        "angle_hint": "creative destruction case studies, regulatory capture, unintended consequences of price controls, property-rights innovation, spontaneous order in odd domains",
    },
    "governance": {
        "name": "Public-Policy Analyst",
        "persona": "Public-policy analyst — Brookings, PRS Legislative Research, IDFC Institute, Cato regulation, Ash Center Harvard, Governing.com, Aporia. Knows the difference between a Bill and an Act.",
        "search_hint": "Brookings, PRS Legislative Research, IDFC Institute, Cato regulation, Ash Center Harvard, Governing.com, Public Administration Review",
        "voice_hint": "cites the specific statute number, notification date, standing-committee report, and enforcement rate.",
        "angle_hint": "e-governance rollouts, judicial pendency data, ministerial reshuffles' policy impact, decentralisation experiments, RTI outcomes, procurement scandals",
    },
    "music": {
        "name": "Music Head",
        "persona": "Music head — Pitchfork, Anthony Fantano/The Needle Drop, Bandcamp Daily, Genius annotations, r/indieheads, r/hiphopheads, Stereogum, Rolling Stone longform. Knows why the snare is doing that.",
        "search_hint": "Pitchfork, Bandcamp Daily, Anthony Fantano, r/indieheads, r/hiphopheads, Stereogum, Rolling Stone, Genius annotations",
        "voice_hint": "cites the producer credit, sample source (WhoSampled-style), mix engineer, label history, chart position.",
        "angle_hint": "streaming payouts vs live economics, label buyouts and masters lawsuits, viral-song mechanics on TikTok, genre-splicing history, cult albums that hit late",
    },
    "sports": {
        "name": "Sports Analyst",
        "persona": "Sports analyst — The Athletic, ESPN Insider, FiveThirtyEight sports, Bill Simmons, ProFootballFocus, StatsBomb, Baseball Prospectus. Talks in xG, PER, WAR, VORP, EPA.",
        "search_hint": "The Athletic, ESPN Insider, FiveThirtyEight, ProFootballFocus, StatsBomb, Baseball Prospectus, The Ringer, ESPN Cricinfo",
        "voice_hint": "advanced-stats native, cites xG, salary-cap hit, draft capital, contract structure (guaranteed vs option year).",
        "angle_hint": "salary-cap loopholes, front-office rebuild timelines, betting-line movement, coaching-tree lineages, expansion-team economics, media-rights inflection points",
    },
    "health": {
        "name": "Health-Sciences Reader",
        "persona": "Health-sciences reader — STAT News, NEJM Journal Watch, The Atlantic health, Peter Attia, Andrew Huberman (skeptically), Cochrane reviews. Skeptical of single-study hype.",
        "search_hint": "STAT News, NEJM, The Lancet, JAMA, Cochrane, The Atlantic health, Peter Attia's blog, ScienceBasedMedicine",
        "voice_hint": "cites the trial name (RECOVERY, SELECT, DAPA-HF), phase, primary endpoint, effect size and 95% CI.",
        "angle_hint": "GLP-1 economics, healthcare-cost disease, insurance denial patterns, off-label prescribing waves, longevity biotech, mental-health prescription trends",
    },
    "science": {
        "name": "Science Journalist",
        "persona": "Science journalist — Quanta Magazine, Nature News, Science, Ars Technica science, Andy Matuschak & Michael Nielsen. Distinguishes preprint from peer-reviewed cleanly.",
        "search_hint": "Quanta Magazine, Nature News, Science News, Ars Technica science, arXiv, bioRxiv, PBS Space Time",
        "voice_hint": "cites the lab, the PI, the journal, the sigma level; comfortable with 'p-hacking' and effect sizes.",
        "angle_hint": "replication-crisis timelines, breakthrough experiments this year, funding-cliff politics, retracted-paper cascades, adjacent-field cross-pollination",
    },
    "philosophy": {
        "name": "Philosophy Nerd",
        "persona": "Philosophy nerd — Stanford Encyclopedia of Philosophy (SEP), Aeon, Philosophy Bites, LSE Philosophy blog, Daily Nous. Argues trolley problems unironically. Reads primary texts.",
        "search_hint": "Stanford Encyclopedia of Philosophy, Aeon, Philosophy Bites, Daily Nous, LSE Philosophy blog, Philosophy Now",
        "voice_hint": "cites philosopher and specific text (Kant CPR B75, Anscombe 1958), distinguishes analytic from continental without sneering.",
        "angle_hint": "virtue-ethics revival, effective-altruism debates, philosophy-of-AI, non-Western philosophy in mainstream syllabi, meta-ethics vs applied ethics",
    },
    "technology": {
        "name": "Tech Journalist",
        "persona": "Tech journalist — The Verge, Wired, MIT Tech Review, Rest of World, Casey Newton's Platformer, Ben Thompson's Stratechery, 404 Media. Tracks platform politics and antitrust fights.",
        "search_hint": "The Verge, Wired, MIT Tech Review, Rest of World, Platformer, Stratechery, Techmeme, 404 Media, Bloomberg tech",
        "voice_hint": "insider on platform politics — cites specific policy change, CEO memo leaks, FTC filings, antitrust case numbers.",
        "angle_hint": "platform-antitrust fights, chip supply chains, cross-border tech regulation, creator-economy monetization, e-waste geopolitics",
    },
    "history": {
        "name": "Popular Historian",
        "persona": "Popular historian — Dan Carlin's Hardcore History, The Rest is History podcast, Simon Sebag Montefiore, Ramachandra Guha, John Barry, Mary Beard. Loves the counterfactual.",
        "search_hint": "Hardcore History, The Rest is History podcast, Aeon Ideas, History Extra, JSTOR Daily, Ramachandra Guha essays",
        "voice_hint": "grounds every claim in a year, a person, a treaty, or a battle. Loves narrative causal chains, sceptical of great-man theory.",
        "angle_hint": "citable counterfactuals, hidden origins of familiar institutions, colonial economics data, plague-and-price-of-labour causal chains, subaltern histories",
    },
    "design": {
        "name": "Design Lead",
        "persona": "Design lead — It's Nice That, Sidebar, Designer Fund, Fast Company Design, Muzli, Michael Bierut, Khoi Vinh, Julie Zhuo. Knows Bauhaus from Swiss Style from Memphis.",
        "search_hint": "It's Nice That, Sidebar, Designer Fund, Fast Company Design, Muzli, Design Milk, Print Magazine, Brand New under Consideration",
        "voice_hint": "cites the studio (Pentagram, Wolff Olins), the year a logo shifted, the type foundry (Grilli, Klim, Colophon).",
        "angle_hint": "rebrand costs and payoffs, product-led design orgs, accessibility-driven aesthetics, AI in design workflows, physical/digital brand tensions",
    },
    "psychology": {
        "name": "Behavioural-Sciences Reader",
        "persona": "Behavioural-sciences reader — Kahneman, Cialdini, Robert Sapolsky, SciAm Mind, BPS Research Digest, Robb Willer papers, Ezra Klein interviews. Skeptical of pop-psych.",
        "search_hint": "SciAm Mind, Psychology Today, Aeon Psyche, BPS Research Digest, PsyArXiv, Kahneman archive",
        "voice_hint": "cites the seminal experiment, effect size, and the failed-replication caveat where relevant.",
        "angle_hint": "replication successes, loneliness / tech-use causal debates, mental-health economics, therapy-modality wars, gamification and habit science",
    },
    "space": {
        "name": "Space Enthusiast",
        "persona": "Space enthusiast — NASA Watch, SpaceNews, Ars Technica space (Eric Berger), Everyday Astronaut, Scott Manley. Knows every Starship IFT number by heart.",
        "search_hint": "NASA Watch, SpaceNews, Ars Technica space, Everyday Astronaut, Scott Manley, ESA, ISRO PIB releases",
        "voice_hint": "cites launch pad, payload mass to LEO, cost-per-kg, mission acronym (JWST, PSLV-C56, IFT-7).",
        "angle_hint": "reusability economics, national-space-budget politics, Starlink revenue lines, ISRO/CNSA/Roscosmos strategic pivots, deep-space mission bottlenecks",
    },
    "climate": {
        "name": "Climate Reporter",
        "persona": "Climate reporter — Carbon Brief, Inside Climate News, Grist, Vox Future Perfect, David Roberts' Volts, IPCC reports directly, Our World in Data.",
        "search_hint": "Carbon Brief, Inside Climate News, Grist, Volts by David Roberts, IPCC reports, Our World in Data, Nature Climate Change",
        "voice_hint": "cites tonnes CO₂e, LCOE per MWh, the specific IRA / EU-CBAM clause, the reference year.",
        "angle_hint": "grid-scale storage economics, methane-leak accounting, adaptation vs mitigation trade-offs, carbon-market fraud, climate-migration corridors",
    },
    "art": {
        "name": "Contemporary-Art Insider",
        "persona": "Contemporary-art insider — Artforum, ArtNet, Frieze, Hyperallergic, Ocula, Art Basel reports. Tracks auction records, biennial politics, gallery representation churn.",
        "search_hint": "Artforum, ArtNet, Frieze, Hyperallergic, Ocula, Art Basel reports, Sotheby's/Christie's, The Art Newspaper",
        "voice_hint": "cites the gallery (Gagosian, David Zwirner, Hauser & Wirth), auction estimate vs hammer, the biennial edition.",
        "angle_hint": "auction-market micro-cycles, artist-estate legal battles, NFT-market wreckage, museum-labour unionization, restitution politics",
    },
    "literature": {
        "name": "Book Critic",
        "persona": "Book critic — NYRB, LARB, The New Yorker fiction, LitHub, Granta, The Bookseller trade data, The Millions.",
        "search_hint": "NYRB, LARB, The New Yorker fiction, LitHub, Granta, The Millions, The Bookseller, Publishers Weekly",
        "voice_hint": "quotes a specific sentence, names the translator and imprint, cites advance size and print run.",
        "angle_hint": "translation-market growth, Booker / NBA / Nobel politics, big-five publisher M&A, literary-magazine collapses, backlist economics",
    },
    "gaming": {
        "name": "Games Journalist",
        "persona": "Games journalist — Kotaku, Polygon, Rock Paper Shotgun, Digital Foundry, GameDeveloper.com, Matt Piscatella (Circana). Deep in Metacritic-vs-user-score wars.",
        "search_hint": "Kotaku, Polygon, Rock Paper Shotgun, Digital Foundry, GameDeveloper.com, Circana / NPD, GameSpot",
        "voice_hint": "cites the studio, engine (Unreal 5.4, Godot 4.2), frame-rate mode, day-one player count, GaaS retention curve.",
        "angle_hint": "layoff waves and studio consolidation, live-service economics, GaaS burnout data, retro-remaster revenue, Chinese/Korean market carve-outs",
    },
    "fitness": {
        "name": "Evidence-Based Fitness Nerd",
        "persona": "Evidence-based fitness nerd — Alan Aragon, Menno Henselmans, Stronger by Science, MASS Research, Peter Attia, RP Strength. Reads meta-analyses.",
        "search_hint": "Stronger by Science, Alan Aragon Research Review, MASS Research, Menno Henselmans, RP Strength, PubMed exercise-science",
        "voice_hint": "cites the meta-analysis, sample size, effect size in kg or %1RM, and confidence intervals.",
        "angle_hint": "hypertrophy meta-analyses, HIIT-vs-LISS realities, protein-timing myths, wearable-data validity, GLP-1 vs training effect",
    },
    "cooking": {
        "name": "Serious Home Cook",
        "persona": "Serious home cook — Serious Eats (Kenji), Bon Appétit, NYT Cooking, Cook's Illustrated, Modernist Cuisine. Sous-vides eggs, has a Baking Steel.",
        "search_hint": "Serious Eats, NYT Cooking, Cook's Illustrated, Bon Appétit, Modernist Cuisine, Eater",
        "voice_hint": "cites exact temperatures, pan diameter, salt brand (Diamond Crystal vs Morton), hydration ratio.",
        "angle_hint": "restaurant-industry margins, ingredient-supply shocks, culinary-school ROI, chef-owner-vs-investor conflicts, TikTok-recipe science failures",
    },
    "travel": {
        "name": "Travel Journalist",
        "persona": "Travel journalist — Condé Nast Traveler, AFAR, Skift, One Mile at a Time (points), View from the Wing, Rick Steves, David Farley.",
        "search_hint": "Condé Nast Traveler, AFAR, Skift, One Mile at a Time, View from the Wing, The Points Guy, Lonely Planet",
        "voice_hint": "cites the airline route (Delta ATL-BLR), loyalty program tier, ADR (average daily rate), visa category and processing time.",
        "angle_hint": "airline route economics, hotel-loyalty devaluations, overtourism-vs-underdevelopment, visa-liberalisation politics, cruise-industry externalities",
    },
    "cinema": {
        "name": "Cinephile Critic",
        "persona": "Cinephile critic — Sight & Sound, Film Comment, IndieWire, Little White Lies, David Bordwell, MUBI Notebook. Tracks A24/Neon distribution deals and DoP moves.",
        "search_hint": "Sight & Sound, Film Comment, IndieWire, Little White Lies, MUBI Notebook, David Bordwell blog, Variety, Deadline",
        "voice_hint": "cites director, DoP, colourist, aspect ratio, box-office multiple (opening / domestic / worldwide).",
        "angle_hint": "A24/Neon distribution economics, streaming-vs-theatrical windowing, foreign-language Oscar campaigns, cinematographer-driven auteurism, restoration/repertory revivals",
    },
    "photography": {
        "name": "Serious Photographer",
        "persona": "Serious photographer — Magnum blog, PetaPixel, Fstoppers, LensCulture, Aperture. Knows the difference between a Zeiss Otus and a Sigma Art without looking.",
        "search_hint": "Magnum blog, PetaPixel, Fstoppers, LensCulture, Aperture, The New York Times Lens (archive), DPReview",
        "voice_hint": "cites sensor size (35mm, MF, 1-inch), print size, ISO / aperture / shutter, and museum acquisition.",
        "angle_hint": "photojournalism-vs-AI trust, medium-format resurgence, camera-market shrink data, photo-agency M&A, print-market economics",
    },
}


async def plan_rails(
    interests: List[str],
    is_news: bool,
    llm_key: str,
    model_provider: str = "openai",
    model_name: str = "gpt-5.6-luna",
) -> List[dict]:
    """Build ONE rail PER interest — NEVER cluster / mix interests.

    For each interest:
      1. If the interest is in PERSONA_LIBRARY, use the hand-curated persona
         (fast + deterministic + high quality).
      2. Otherwise ask the LLM to design a single vivid persona for that
         one interest (no mixing).

    Cap: up to 6 rails (first 6 interests). Each rail is a completely
    independent research pass — so Celebrity Gossip and World Politics
    never contaminate each other."""
    if not interests:
        return []

    # De-dupe while preserving order, cap 6 rails
    seen_lower = set()
    unique: List[str] = []
    for it in interests:
        k = (it or "").strip()
        if not k:
            continue
        lo = k.lower()
        if lo in seen_lower:
            continue
        seen_lower.add(lo)
        unique.append(k)
        if len(unique) >= 6:
            break

    async def build_one_rail(interest: str) -> Optional[dict]:
        # Library hit — deterministic, high-fidelity
        key = interest.strip().lower()
        lib = PERSONA_LIBRARY.get(key)
        if lib:
            return {
                "name": lib["name"],
                "persona": lib["persona"],
                "interests": [interest],
                "search_hint": lib["search_hint"],
                "voice_hint": lib["voice_hint"],
                "angle_hint": lib["angle_hint"],
            }

        # LLM-designed persona for a single interest (never clusters)
        session_id = f"rail-{uuid.uuid4()}"
        chat = LlmChat(
            api_key=llm_key,
            session_id=session_id,
            system_message=RAIL_PLANNER_SYSTEM,
        ).with_model(model_provider, model_name)
        text = ""
        try:
            text = await chat.send_message(
                UserMessage(
                    text=RAIL_PLANNER_TEMPLATE.format(
                        interest=interest,
                        is_news="yes" if is_news else "no",
                    )
                )
            )
            parsed = _extract_json(text)
            r = parsed.get("rail") or {}
            persona = (r.get("persona") or "").strip()
            if not persona:
                return None
            name = (r.get("name") or "").strip() or interest
            return {
                "name": name,
                "persona": persona,
                "interests": [interest],
                "search_hint": (r.get("search_hint") or "").strip(),
                "voice_hint": (r.get("voice_hint") or "").strip(),
                "angle_hint": (r.get("angle_hint") or "").strip(),
            }
        except Exception:
            log.exception(
                "Rail planning JSON parse failed for '%s': %s",
                interest,
                text[:400] if text else "<empty>",
            )
            # Last-resort persona so this interest still gets its own rail
            return {
                "name": interest,
                "persona": f"Deeply engaged reader whose entire content diet revolves around {interest}. Never wants generic filler — only material a specialist in this niche would find non-obvious.",
                "interests": [interest],
                "search_hint": "",
                "voice_hint": "specific, jargon-fluent when needed, cites names / dates / numbers",
                "angle_hint": "non-obvious mechanisms, specialist trivia, insider dynamics",
            }

    rails_raw = await asyncio.gather(*[build_one_rail(i) for i in unique])
    return [r for r in rails_raw if r]


async def _tavily_search_one(
    tavily_client,
    query: str,
    topic_type: str = "general",
    max_results: int = 6,
    days: Optional[int] = None,
) -> dict:
    """Single Tavily search wrapped in to_thread. Retries transient network errors
    (RemoteDisconnected, ConnectionError, ReadTimeout) up to 3 times with backoff
    so a flaky moment on Tavily's edge doesn't collapse the whole research batch.

    Returns a dict. On quota / plan limit, sets `_quota_exceeded: True` so the
    caller can surface a specific actionable message to the user."""
    import time as _time

    def _do():
        last_err = None
        for attempt in range(3):
            try:
                kwargs = dict(
                    query=query,
                    search_depth="advanced",
                    topic=topic_type,  # "general" or "news"
                    max_results=max_results,
                    include_answer=False,
                    include_raw_content=False,
                )
                if topic_type == "news" and days:
                    kwargs["days"] = days
                return tavily_client.search(**kwargs)
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                # Terminal plan / quota errors — no point retrying
                if (
                    "usage limit" in msg
                    or "plan" in msg
                    and "limit" in msg
                    or "forbidden" in msg
                    or "401" in msg
                    or "403" in msg
                ):
                    log.warning("Tavily blocked (quota/plan) for '%s': %s", query, e)
                    return {"results": [], "_quota_exceeded": True, "_error": str(e)}
                transient = (
                    "remotedisconnected" in msg
                    or "connection aborted" in msg
                    or "connection reset" in msg
                    or "read timed out" in msg
                    or "temporarily unavailable" in msg
                    or "502" in msg
                    or "503" in msg
                    or "504" in msg
                )
                if not transient or attempt == 2:
                    log.warning(
                        "Tavily search failed for '%s' (attempt %d): %s",
                        query,
                        attempt + 1,
                        e,
                    )
                    return {"results": []}
                _time.sleep(0.6 * (2**attempt))  # 0.6s, 1.2s
        log.warning("Tavily search exhausted retries for '%s': %s", query, last_err)
        return {"results": []}

    return await asyncio.to_thread(_do)


async def _tavily_extract(tavily_client, urls: List[str]) -> List[dict]:
    """Tavily extract for a batch of URLs. Returns list of {url, content}."""
    if not urls:
        return []

    def _do():
        try:
            resp = tavily_client.extract(urls=urls, extract_depth="advanced")
            # Response shape: {"results": [{"url":..., "raw_content":...}], "failed_results": [...]}
            return resp.get("results", [])
        except Exception as e:
            log.warning("Tavily extract failed: %s", e)
            return []

    return await asyncio.to_thread(_do)


YOUTUBE_URL_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([A-Za-z0-9_-]{6,15})"
)


def _extract_youtube_id(url: str) -> Optional[str]:
    if not url:
        return None
    m = YOUTUBE_URL_RE.search(url)
    return m.group(1) if m else None


async def _fetch_youtube_transcript(
    video_id: str, title: str, url: str
) -> Optional[dict]:
    """Fetch a YouTube transcript (best-effort). Returns a dict or None."""

    def _do():
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id, languages=["en", "en-US", "en-GB"]
            )
            # Join into a single text blob, keep first ~2500 chars
            text = " ".join(seg.get("text", "") for seg in transcript_list)
            text = re.sub(r"\s+", " ", text).strip()
            return {"url": url, "title": title, "transcript_excerpt": text[:2500]}
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
            return None
        except Exception as e:
            log.debug("YT transcript fetch failed for %s: %s", video_id, e)
            return None

    return await asyncio.to_thread(_do)


async def deep_research(
    topic: str,
    tavily_client,
    llm_key: str,
    extract_top_n: int = 6,
    youtube_top_n: int = 3,
    on_progress=None,
    persona: Optional[str] = None,
    angle_hint: Optional[str] = None,
    search_hint: Optional[str] = None,
    is_news: bool = False,
    news_days: int = 7,
) -> ResearchBundle:
    """Run the full deep-research pipeline. Returns a ResearchBundle.

    Persona-tailored parameters (all optional):
      persona:     vivid audience archetype string (e.g., "Gossip Enthusiast").
      angle_hint:  content angles this persona cares about (drama, cap tables, …).
      search_hint: site/forum/creator hint used as a soft query prefix.
      is_news:     when True, all Tavily calls use topic=news with a `days` window.
      news_days:   recency window in days (used only when is_news=True).

    on_progress: optional async callable taking a dict {phase, message, meta?}
    called at each stage boundary. Useful for driving SSE progress events.
    """

    async def emit(phase: str, message: str, meta: Optional[dict] = None):
        if on_progress:
            try:
                await on_progress(
                    {"phase": phase, "message": message, "meta": meta or {}}
                )
            except Exception:
                log.debug("progress callback error", exc_info=True)

    bundle = ResearchBundle(topic=topic)

    # 1) Plan angles (persona-aware if a persona is provided)
    await emit("planning", "Planning research angles")
    angles = await plan_research_angles(
        topic,
        llm_key,
        persona=persona,
        angle_hint=angle_hint,
        is_news=is_news,
    )
    if not angles:
        angles = [{"name": "general", "query": topic, "is_current": False}]
    bundle.angles = angles
    await emit(
        "angles_ready",
        f"Investigating {len(angles)} angles",
        {"angles": [a["name"] for a in angles]},
    )

    # 2) Parallel searches per angle. In news mode we force topic=news + days window.
    #    Persona search_hint (e.g., "Deuxmoi, Page Six, r/Fauxmoi") is appended to the
    #    Tavily query so the search leans toward that persona's trusted surfaces.
    await emit("searching", "Searching the web deeply")
    search_tasks = []
    for a in angles:
        q = a["query"]
        if search_hint:
            q = f"{q} ({search_hint})"
        if is_news:
            topic_type = "news"
            days = news_days
        else:
            topic_type = "news" if a["is_current"] else "general"
            days = news_days if topic_type == "news" else None
        search_tasks.append(
            _tavily_search_one(
                tavily_client,
                q,
                topic_type=topic_type,
                max_results=6,
                days=days,
            )
        )
    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # 3) Dedupe & rank sources by score
    seen_urls = set()
    ranked_sources: List[dict] = []
    youtube_candidates: List[dict] = []
    for angle, res in zip(angles, search_results):
        if isinstance(res, Exception):
            continue
        if isinstance(res, dict) and res.get("_quota_exceeded"):
            bundle.quota_exceeded = True
        for r in res.get("results", []):
            url = r.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            r["_angle"] = angle["name"]
            ranked_sources.append(r)
            vid = _extract_youtube_id(url)
            if vid:
                youtube_candidates.append(
                    {"video_id": vid, "url": url, "title": r.get("title", "")}
                )

    ranked_sources.sort(key=lambda x: x.get("score", 0), reverse=True)
    bundle.sources = ranked_sources[:20]
    await emit(
        "sources_ready",
        f"Found {len(bundle.sources)} sources across {len(angles)} angles",
        {"sources_count": len(bundle.sources)},
    )

    # 4) Deep-extract top URLs (skip YouTube — handled by transcript pass)
    top_urls_to_extract = [
        s["url"] for s in bundle.sources if _extract_youtube_id(s["url"]) is None
    ][:extract_top_n]

    if top_urls_to_extract:
        await emit("extracting", "Extracting full page content")
    yt_tasks = [
        _fetch_youtube_transcript(y["video_id"], y["title"], y["url"])
        for y in youtube_candidates[:youtube_top_n]
    ]
    extract_task = _tavily_extract(tavily_client, top_urls_to_extract)

    extracts_res, *yt_res = await asyncio.gather(extract_task, *yt_tasks)

    extract_map = {
        r["url"]: r.get("raw_content") or "" for r in extracts_res if r.get("url")
    }
    bundle.extracts = [
        {"url": u, "content": (extract_map.get(u, "") or "")[:6000]}
        for u in top_urls_to_extract
        if extract_map.get(u)
    ]

    yt_kept = [y for y in yt_res if y]
    bundle.youtube = yt_kept
    if yt_kept:
        await emit(
            "youtube",
            f"Read {len(yt_kept)} YouTube transcripts",
            {"videos": [y["title"][:60] for y in yt_kept]},
        )
    else:
        await emit("extracted", f"Extracted {len(bundle.extracts)} deep sources")

    return bundle


def build_research_context(bundle: ResearchBundle) -> str:
    """Assemble the LLM context string from the research bundle."""
    parts = []

    parts.append(f"USER TOPIC: {bundle.topic}\n")
    if bundle.angles:
        parts.append("RESEARCH ANGLES INVESTIGATED:")
        for i, a in enumerate(bundle.angles, 1):
            tag = " [current-news]" if a.get("is_current") else ""
            parts.append(f"  {i}. {a['name']}{tag} — {a['query']}")
        parts.append("")

    parts.append("SOURCES (numbered; use for source_index):")
    for i, s in enumerate(bundle.sources[:15], 1):
        title = s.get("title", "")
        url = s.get("url", "")
        content = (s.get("content") or "")[:900]
        angle = s.get("_angle", "")
        parts.append(
            f"[{i}] {title}   ({angle})\n   URL: {url}\n   SNIPPET: {content}\n"
        )

    if bundle.extracts:
        parts.append(
            "\nDEEP-EXTRACTED SOURCE CONTENT (fuller context from top sources):"
        )
        for e in bundle.extracts:
            parts.append(f"URL: {e['url']}\n---\n{e['content']}\n---")

    if bundle.youtube:
        parts.append("\nYOUTUBE TRANSCRIPTS (relevant videos found):")
        for y in bundle.youtube:
            parts.append(
                f"VIDEO: {y['title']}\nURL: {y['url']}\nTRANSCRIPT: {y['transcript_excerpt']}\n---"
            )

    return "\n".join(parts)


async def generate_insights_from_bundle(
    bundle: ResearchBundle,
    llm_key: str,
    count: int = 12,
    behavior_context: str = "",
    steer: str = "",
    exclude_headlines: Optional[List[str]] = None,
    model_provider: str = "openai",
    model_name: str = "gpt-5.4",
    persona: Optional[str] = None,
    voice_hint: Optional[str] = None,
    angle_hint: Optional[str] = None,
    is_news: bool = False,
) -> List[dict]:
    """Given a ResearchBundle, ask the LLM for `count` insights. Returns parsed list of dicts.

    Persona params tailor the *style* of the insights (tone, references, jargon,
    angle bias) without changing the hard specificity/quality bar."""
    ctx = build_research_context(bundle)

    system = INSIGHT_SYSTEM_PROMPT
    if persona:
        system += (
            f"\n\n═══ AUDIENCE PERSONA (drives voice, references, angle picks) ═══\n"
            f"You are writing FOR: {persona}\n"
            "Every insight must feel like it was hand-picked by an expert who knows this "
            "persona's daily reading list, jargon, and taste. If the sources contain material "
            "outside this persona's interest lane, IGNORE it — never pad the batch with "
            "off-persona filler.\n\n"
            "═══ HARD ANTI-MIX RULE ═══\n"
            "Do NOT blend this persona's interest with ANY other interest the user may have "
            "picked (they run on SEPARATE rails). Every card here is 100% inside this "
            "persona's world. If a source article touches an outside domain (e.g., a gossip "
            "story that also mentions AI startups), extract ONLY the gossip / persona-native "
            "angle. No cross-topic fusion cards. Ever."
        )
        if voice_hint:
            system += f"\nVOICE / TONE FOR THIS PERSONA: {voice_hint}"
        if angle_hint:
            system += f"\nPRIORITIZE ANGLES: {angle_hint}"
    if is_news:
        system += (
            "\n\n═══ NEWS MODE ═══\n"
            "The user chose NEWS mode. Every insight MUST be tied to something that happened "
            "in the LAST 7 DAYS — cite the specific date, week, or 'this week' explicitly. "
            "Absolutely NO evergreen textbook facts. If a source is older than 7 days, only "
            "use it as background for a fresh angle; the specific claim must be recent."
        )
    if steer:
        system += (
            f"\n\nUSER STEERING (spoken preference to honor in every insight): {steer}"
        )
    if behavior_context:
        system += (
            "\n\nUSER BEHAVIOUR SIGNAL FROM PRIOR BATCHES (bias the feed accordingly):\n"
            + behavior_context
        )

    exclude_block = ""
    if exclude_headlines:
        joined = "\n".join(f"- {h}" for h in exclude_headlines[-40:])
        exclude_block = (
            "\n\nDO NOT REPEAT or paraphrase any of these headlines / angles that were already shown:\n"
            + joined
        )

    user_msg = f"""{ctx}

Generate EXACTLY {count} distinct, high-quality insights. Every insight must be:
- surprising or non-obvious,
- backed by specific numbers / studies / dates / names drawn from the sources above,
- covering a DIFFERENT angle from every other insight in this batch.

{exclude_block}

Strict JSON only. No markdown."""

    session_id = f"insights-{uuid.uuid4()}"
    chat = LlmChat(
        api_key=llm_key, session_id=session_id, system_message=system
    ).with_model(model_provider, model_name)

    text = ""
    try:
        text = await chat.send_message(UserMessage(text=user_msg))
    except Exception:
        log.exception("Primary insight LLM call failed; retrying once")
        chat = LlmChat(
            api_key=llm_key, session_id=session_id, system_message=system
        ).with_model(model_provider, model_name)
        text = await chat.send_message(UserMessage(text=user_msg))

    parsed = _extract_json(text)
    return parsed.get("insights", [])[:count]
