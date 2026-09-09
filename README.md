# Project Specs: FinDocRAG & FinSight-Lite

Two standalone, fintech-flavored projects — one demonstrates RAG depth, the other demonstrates multi-agent + MCP orchestration. Built separately, shippable independently, each fills one resume slot.

---

# Project 1: FinDocRAG — SEC Filing Research Assistant

## The Problem

Public company filings (10-K annual reports, 10-Q quarterly reports) are 50–300 pages of dense legal and financial text. Investors, analysts, and students routinely need to answer specific questions like "what does NVIDIA list as its top risk factors?" or "how did gross margin change year over year?" — but reading full filings to find this is slow, and generic ChatGPT-style search either hallucinates numbers or has no access to the actual filing at all.

## What We're Building

A retrieval-augmented question-answering system focused on one company's filings at a time. A user uploads or selects a filing (10-K/10-Q), asks natural-language questions, and gets answers grounded in the actual document — with the exact source passage cited, not paraphrased from memory.

The differentiator from a "chat with PDF" tutorial: **retrieval quality is measured, not assumed.** You build an evaluation set of question–answer pairs with known correct source passages, and report precision/recall (or RAGAS metrics) — the same way you quantified your lunar segmentation project. That's what makes this defensible in an interview instead of "I followed a tutorial."

## How It Works (Pipeline)

```
1. INGESTION
   SEC EDGAR filing (10-K/10-Q, HTML or PDF)
        ↓
   Parse & clean (strip HTML/boilerplate, preserve section structure —
   Item 1A "Risk Factors", Item 7 "MD&A", Item 8 "Financial Statements")
        ↓
   Chunk (semantic/section-aware chunking, not naive fixed-size —
   keep tables and risk-factor bullets intact where possible)
        ↓
   Embed chunks (sentence-transformers or OpenAI/HF embeddings)
        ↓
   Store in vector DB + keep raw text indexed for BM25

2. RETRIEVAL (on each user query)
   User question
        ↓
   Query → BM25 search (keyword/exact-term match — good for tickers,
            dollar figures, exact phrases)
        +
   Query → Vector search (semantic match — good for "what are the risks
            related to competition")
        ↓
   Merge candidates → Rerank (cross-encoder reranker, e.g. bge-reranker)
        ↓
   Top-k reranked chunks passed to LLM

3. GENERATION
   LLM answers using ONLY retrieved chunks (strict grounding prompt)
        ↓
   Answer + cited source chunk(s) + page/section reference
        ↓
   If retrieval confidence is low → explicitly say "not found in filing"
   instead of guessing (this matters — hallucination control is a real
   skill signal)

4. EVALUATION (what makes this a "real" project)
   Build 20-30 question/answer pairs manually from 2-3 filings you know well
        ↓
   Run pipeline against each question
        ↓
   Score: did retrieval surface the correct passage? (Recall@k)
          did the generated answer match ground truth? (via RAGAS or manual)
        ↓
   Report numbers on the resume/README, same style as your mIoU scores
```

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Data source | SEC EDGAR API (free, no key needed) | Real filings, no scraping headaches |
| Parsing | `edgartools` or `BeautifulSoup` + custom section splitter | Filings are messy HTML; section-aware parsing is the actual engineering work here |
| Chunking | LangChain text splitters (customized for filing structure) | You already know LangChain |
| Embeddings | `sentence-transformers` (local, free) or OpenAI embeddings | Local = no API cost while iterating |
| Vector DB | ChromaDB or FAISS (both free, local, simple) | No infra overhead for an MVP |
| Keyword search | `rank_bm25` (lightweight, pure Python) | Needed for the hybrid retrieval story |
| Reranker | `bge-reranker-base` (HuggingFace, free) | This is the "not-basic" part of your RAG — most tutorial RAGs skip reranking entirely |
| LLM | Any (OpenAI API / local via Ollama if you want zero API cost) | Generation only, keep it swappable |
| Eval | RAGAS library, or a simple custom scoring script | Gives you defensible numbers |
| Backend | FastAPI | You already know this |
| Frontend | Streamlit | You already know this |
| Deployment | Docker → HuggingFace Spaces (same pattern as your other projects) | Consistent with your existing deploy story |

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌────────────────┐
│  Streamlit  │─────▶│   FastAPI    │─────▶│  Retrieval      │
│  Frontend   │◀─────│   Backend    │◀─────│  Pipeline       │
└─────────────┘      └──────────────┘      │  (BM25+Vector+  │
                             │              │   Reranker)     │
                             │              └────────┬────────┘
                             ▼                       ▼
                      ┌──────────────┐      ┌────────────────┐
                      │  LLM Answer   │◀─────│  ChromaDB /    │
                      │  Generator    │      │  FAISS Store   │
                      └──────────────┘      └────────────────┘
                             ▲
                             │
                      ┌──────────────┐
                      │  Ingestion    │
                      │  (EDGAR →     │
                      │  parse→chunk→ │
                      │  embed)       │
                      └──────────────┘
```

## End Goal

A working demo where a user picks a company (start with 3-5 pre-indexed companies, e.g. NVIDIA, Apple, Tesla), asks questions about their latest 10-K, and gets grounded, cited answers — plus a README/report showing retrieval evaluation metrics that prove the system actually works, not just looks like it does.

## Who Uses This

Framed for: retail investors doing quick research, students studying specific companies, or as a portfolio demo for recruiters — it doesn't need real users to be resume-valid, but designing it with a clear user in mind (e.g., "helps someone doing equity research skip reading 200 pages") makes the README and any interview explanation sharper.

## Deployment

Docker container → HuggingFace Spaces (free tier, matches your existing MusicLSTM/GRU deployment pattern) — keeps your portfolio consistent and all live-linkable from your resume.

## Build-From-Scratch Order (Realistic Path)

1. **Week 1** — EDGAR ingestion + section-aware parsing for one company's 10-K. Get clean, structured text out. This is unglamorous but is most of the real value — don't rush it.
2. **Week 1-2** — Chunking + embedding + basic vector search only (no BM25/reranker yet). Get a naive RAG answering questions end to end.
3. **Week 2** — Add BM25 + hybrid merge + reranker. Compare answer quality before/after — this comparison itself is a great README section.
4. **Week 3** — Build your eval set (20-30 Q&A pairs), run RAGAS or manual scoring, write up numbers.
5. **Week 3** — FastAPI + Streamlit wrapper, Docker, deploy, extend to 3-5 companies.

---

# Project 2: FinSight-Lite — Multi-Agent Financial Research System

## The Problem

Getting a well-rounded view on a stock means pulling from multiple disconnected sources — price/financials from one tool, recent news from another, and manually synthesizing it into a coherent view. This is exactly the kind of multi-step, multi-source task that's the current frontier use case for LLM agents — and exactly what's missing from your resume right now (you list LangGraph/agents as skills with nothing built to prove it).

## What We're Building

A small team of collaborating AI agents that, given a stock ticker, independently gather financial data and recent news, then synthesize a structured research report (bull case / bear case / key risks) — using real tool calls via MCP servers, not simulated ones.

Deliberately scoped to **3 agents, not 7** — an Orchestrator, a Financial Analyst Agent, and a News/Research Agent, feeding a Report Generator step. This is small enough to finish and polish, but still demonstrates true agent collaboration and tool use, which is what actually matters (a 7-agent version where most agents are shallow prompts is worse than a 3-agent version where every agent does real work).

## How It Works (Pipeline)

```
1. USER REQUEST
   "Give me a research report on NVIDIA"
        ↓
2. ORCHESTRATOR (LangGraph graph, supervisor pattern)
   Parses the request → decides which agents to invoke → manages state
   across the graph → handles failures (e.g., if one tool call fails,
   still produce a partial report rather than crashing)
        ↓
   ┌─────────────────────┬─────────────────────┐
   ▼                                            ▼
3a. FINANCIAL ANALYST AGENT             3b. NEWS/RESEARCH AGENT
    Calls Financial MCP Server:              Calls Web Search MCP Server:
    - get_stock_price(ticker)                - search_news(ticker)
    - get_financials(ticker)                 - get_recent_announcements(ticker)
    - get_key_ratios(ticker)                      ↓
         ↓                                   Summarizes recent developments,
    Structures: revenue trend,               sentiment, notable events
    margins, valuation multiples
        ↓                                            ↓
   └─────────────────────┬─────────────────────┘
                          ▼
4. REPORT GENERATOR (synthesis step)
   Combines both agents' outputs → produces structured report:
   - Financial Health summary
   - Recent News Summary
   - Bull Case
   - Bear Case
   - Key Risks
   - Sources (explicit: which tool/API/article each claim came from)
   - Disclaimer: "Not financial advice — for research purposes only"
        ↓
5. OUTPUT
   Rendered in Streamlit with the score-bar style visual (Financial
   Health ████████░░ etc.) for a polished demo feel
```

## Why MCP (not just plain function calling)

Instead of writing `get_stock_price()` as a Python function directly inside your agent code, you expose it as a tool on a standalone MCP server that the agent *connects to* over the protocol. This mirrors how production agent systems are actually being built industry-wide right now (Anthropic, OpenAI, and major agent frameworks have converged on MCP as the standard). It's a small extra step to build, but it's the difference between "I called a function" and "I understand the current tool-use architecture pattern" — which is precisely the skill you flagged as missing.

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Orchestration | LangGraph (supervisor/graph pattern) | Finishes what you paused mid-learning |
| Tool protocol | MCP (Model Context Protocol) | Current industry-standard tool-use pattern |
| Financial data | `yfinance` (free, reliable, no key needed) | Avoids flaky/paid financial APIs |
| News/web | Web search MCP tool (e.g. via a search API or existing MCP web-search server) | Skip building a "News MCP" from scratch — licensing/rate-limit headaches aren't worth it for an MVP |
| LLM | Any (OpenAI/Anthropic API) | Powers each agent's reasoning |
| Backend | FastAPI | Exposes the agent graph as an API |
| Frontend | Streamlit | Report rendering with visual score bars |
| Deployment | Docker → HuggingFace Spaces | Consistent with your existing projects |

## Architecture

```
┌─────────────┐      ┌──────────────┐
│  Streamlit  │─────▶│   FastAPI    │
│  Frontend   │◀─────│   Endpoint   │
└─────────────┘      └───────┬──────┘
                              ▼
                    ┌──────────────────┐
                    │   LangGraph       │
                    │   Orchestrator    │
                    │   (Supervisor)    │
                    └─────────┬─────────┘
                    ┌─────────┴─────────┐
                    ▼                   ▼
          ┌──────────────────┐ ┌──────────────────┐
          │ Financial Analyst │ │ News/Research     │
          │ Agent             │ │ Agent             │
          └────────┬──────────┘ └────────┬──────────┘
                    ▼                     ▼
          ┌──────────────────┐ ┌──────────────────┐
          │ Financial MCP      │ │ Web Search MCP    │
          │ Server (yfinance)  │ │ Server            │
          └──────────────────┘ └──────────────────┘
                    │                     │
                    └──────────┬──────────┘
                               ▼
                    ┌──────────────────┐
                    │ Report Generator  │
                    │ (synthesis node)  │
                    └──────────────────┘
```

## End Goal

A working demo: enter any stock ticker, get back a structured, cited research report generated by genuinely collaborating agents, in under ~30 seconds, deployed live and linkable from your resume/GitHub.

## Who Uses This

Framed as a research-assistance tool for retail investors or students learning equity analysis — again, doesn't need real users, but this framing keeps the report format and disclaimers grounded and makes it easy to explain in 30 seconds during an interview.

## Deployment

Docker container → HuggingFace Spaces, same pattern as your other live demos, so your GitHub/portfolio stays consistent.

## Build-From-Scratch Order (Realistic Path)

1. **Week 1** — Finish LangGraph learning through the supervisor/multi-agent pattern specifically (you're partway there already). Build a skeleton graph: Orchestrator → one dummy agent → output, just to get state passing working.
2. **Week 1-2** — Build the Financial MCP server wrapping `yfinance` (price, financials, ratios). Connect it to the Financial Analyst Agent. Test in isolation before adding the second agent.
3. **Week 2** — Add the News/Research Agent with a web-search MCP tool. Get both agents running in parallel under the Orchestrator.
4. **Week 3** — Build the Report Generator synthesis node — this is where "bull case / bear case / risks" structure comes from. Add the disclaimer and explicit source attribution.
5. **Week 3** — FastAPI wrapper, Streamlit UI with score-bar visuals, Docker, deploy.
6. **(Optional stretch, only if time allows)** — Add a lightweight verification step: cross-check one or two numeric claims in the report against the raw API response and flag mismatches. This alone is a strong differentiator if you have the extra week.

---

## Resume Bullets (once both are built)

**FinDocRAG:**
"Built hybrid RAG system for SEC filing analysis (BM25 + vector search + cross-encoder reranking); evaluated retrieval quality on a custom 30-question eval set, achieving [X]% recall@5 — deployed via FastAPI/Streamlit on HuggingFace Spaces."

**FinSight-Lite:**
"Built multi-agent financial research system using LangGraph supervisor orchestration and MCP tool servers; agents independently query real-time financial data and news to generate structured, source-cited investment research reports."

## Suggested Order to Build Them

RAG project first (FinDocRAG) — it's the smaller, faster win, replaces your weakest current project (GRU) sooner, and de-risks your timeline. Start FinSight-Lite once FinDocRAG is deployed and you have a working, gradeable result banked.
