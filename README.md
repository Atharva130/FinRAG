# 📊 FinRAG — SEC Filing Research Assistant

A retrieval-augmented question-answering system for SEC 10-K filings. Ask natural-language questions about a company's annual report and get answers grounded in the actual filing text — with exact source chunks cited, not paraphrased from memory.

**Live demo:** [https://atharva130-finrag-app-jrd75h.streamlit.app/]

---

## The Problem

10-K filings are 50–300 pages of dense legal and financial text. Generic LLM chat either hallucinates numbers or has no access to the actual document at all.

This project answers questions using **only** retrieved, cited passages from the real filing — and if the answer isn't in the retrieved context, it says so instead of guessing.

---

## What Makes This Different From a "Chat With PDF" Tutorial

Most tutorial RAG projects stop at:

> Embed chunks → vector search → ask LLM

This project adds two things most tutorials skip:

### 1. Hybrid retrieval, not just vector search

Pure semantic search alone can miss exact-term queries. For example, a query containing **"competition"** may fail to surface a chunk that only says **"competitors"** or **"competitive"**, even though it is clearly relevant.

This system combines:

- **BM25** — keyword-based retrieval
- **Vector search** — semantic retrieval
- **Reciprocal Rank Fusion (RRF)** — merges both result sets
- **Cross-encoder reranking** — performs a final relevance-based ranking

The reranker used is:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

This means the final ranking reflects actual query-document relevance rather than relying only on word overlap or embedding distance.

### 2. Measured evaluation, not assumed quality

A **12-question evaluation set** was built manually from the filing, with verified ground-truth source chunk IDs.

The pipeline is scored automatically on:

- **Recall@5** — Did retrieval surface the correct source chunk at all?
- **Answer keyword accuracy** — Did the generated answer contain the expected factual content?

### Evaluation Results

**Recall@5 = 62.5%**

**Answer keyword accuracy = 58.3%**

See `evaluation/eval_results.json` for the full per-question breakdown.

These numbers are reported honestly, including the questions where retrieval missed. This provides a more credible signal than an unverified claim that the system simply "works great."

---

## How It Works

```text
┌─────────────────────────────────────────────┐
│          OFFLINE / INGESTION PIPELINE       │
└─────────────────────────────────────────────┘

SEC EDGAR Filing (10-K, HTML)
                    │
                    ▼
          Parse & Clean Document
                    │
                    ▼
      Section-Aware Splitting
   (Item 1, Item 1A, Item 7, etc.)
                    │
                    ▼
      Paragraph-Aware Chunking
       (~1500 chars + overlap)
                    │
                    ▼
       Sentence Transformer
        all-MiniLM-L6-v2
                    │
                    ▼
              ChromaDB
          (Vector Storage)


┌─────────────────────────────────────────────┐
│              QUERY PIPELINE                 │
└─────────────────────────────────────────────┘

User Query
     │
     ├───────────────┐
     ▼               ▼
   BM25          Vector Search
 (Keyword)        (Semantic)
     │               │
     └───────┬───────┘
             ▼
 Reciprocal Rank Fusion
             │
             ▼
      Cross-Encoder
        Reranking
             │
             ▼
        Top-k Chunks
             │
             ▼
        LLM Generation
   Groq / openai-gpt-oss-20b
             │
             ▼
    Grounded Answer
             +
      chunk_id Citations
```

### Pipeline Summary

1. Fetch the SEC 10-K filing from EDGAR.
2. Parse and clean the HTML document.
3. Split the filing into meaningful SEC sections.
4. Create paragraph-aware chunks with overlap.
5. Generate embeddings using `all-MiniLM-L6-v2`.
6. Store embeddings and metadata in ChromaDB.
7. At query time, perform both BM25 and vector search.
8. Merge candidates using Reciprocal Rank Fusion.
9. Rerank candidates using a cross-encoder.
10. Pass the top-k chunks to the LLM.
11. Generate a grounded answer with `chunk_id` citations.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data source | SEC EDGAR API via `edgartools` |
| Parsing | Custom regex-based section splitter |
| Chunking | Custom paragraph-aware chunker with overlap |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB (local, persistent) |
| Keyword search | `rank_bm25` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq API (`openai/gpt-oss-20b`) |
| Evaluation | Custom scoring script — Recall@k + answer-keyword accuracy |
| Frontend | Streamlit |

---

## Project Structure

```text
FinRAG/
│
├── ingestion/
│   ├── fetch_filing.py
│   ├── section_splitter.py
│   └── chunker.py
│
├── retrieval/
│   └── retriever.py
│
├── generation/
│   └── generate_answer.py
│
├── evaluation/
│   ├── eval_set.json
│   ├── run_eval.py
│   └── eval_results.json
│
├── data/
│   └── processed/
│
├── rag_pipeline.py
├── app.py
└── README.md
```

### Key Files

| File | Purpose |
|---|---|
| `ingestion/fetch_filing.py` | Pull 10-K from SEC EDGAR |
| `ingestion/section_splitter.py` | Split filing into Item 1, 1A, 7, etc. |
| `ingestion/chunker.py` | Paragraph-aware chunking |
| `retrieval/retriever.py` | Hybrid BM25 + vector + reranker |
| `generation/generate_answer.py` | Grounded answer generation |
| `evaluation/eval_set.json` | 12 manually verified Q&A pairs |
| `evaluation/run_eval.py` | Evaluation/scoring script |
| `evaluation/eval_results.json` | Evaluation results |
| `rag_pipeline.py` | End-to-end RAG pipeline |
| `app.py` | Streamlit frontend |

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/Atharva130/FinRAG.git
cd FinRAG
```

### 2. Create a virtual environment

```bash
python -m venv ragenv
```

#### Windows

```powershell
.\ragenv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_key_here
```

Get a free API key from Groq.

### 5. Fetch and index a filing

The NVIDIA filing is already included in `data/processed/`.

To run the ingestion pipeline:

```bash
python ingestion/fetch_filing.py
python ingestion/section_splitter.py
python ingestion/chunker.py
python ingestion/embed_and_store.py
```

### 6. Run the application

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

### 7. Run the evaluation

```bash
python evaluation/run_eval.py
```

---

## Evaluation Results

The current evaluation uses a manually verified set of **12 questions**.

| Metric | Score |
|---|---:|
| Recall@5 | **62.5%** |
| Answer keyword accuracy | **58.3%** |
| Questions evaluated | **12** |

Full per-question results, including failed questions and their causes, are available in:

```text
evaluation/eval_results.json
```

---

## Known Limitations

- **Single-company scope:** Currently tested on the NVIDIA 10-K only. The architecture generalizes to other filings but has not yet been extended to multiple companies.

- **Financial statement tables:** Financial statements from Item 8 are indexed under a different section label (`Item 15`) due to how 10-Ks physically place financial statements later in the document. This is a known quirk of SEC filing structure, not a parsing bug.

- **General-purpose reranker:** The current reranker (`ms-marco-MiniLM-L-6-v2`) is not fine-tuned specifically on financial text and can occasionally rank boilerplate section headers above substantive content.

---

## Future Improvements

- Fine-tune or replace the reranker with a finance-domain model.
- Extend the system to support multiple companies and filings.
- Add RAGAS-based semantic answer scoring alongside keyword matching.
- Improve retrieval and reranking for financial statement tables.
- Expand the evaluation dataset with more questions and filing sections.

---

## Example Query Types

The system can answer questions about information contained in the indexed SEC filing, such as:

- What are the company's major risk factors?
- What factors affected revenue growth?
- What did management say about operating expenses?
- What are the company's main competitive risks?
- What were the major changes in the company's financial position?

Answers are generated from retrieved filing chunks and include source `chunk_id` citations.

---

## Project Goal

The goal of FinDocRAG is to demonstrate a **production-oriented RAG pipeline** for financial documents rather than a basic "chat with PDF" implementation.

The project focuses on:

- **Domain-specific document ingestion**
- **Section-aware and paragraph-aware chunking**
- **Hybrid keyword + semantic retrieval**
- **Reciprocal Rank Fusion**
- **Cross-encoder reranking**
- **Grounded LLM generation**
- **Source citations**
- **Quantitative retrieval evaluation**

---

## License

This project is intended for educational and portfolio purposes.
