# Token-Diet: Dynamic Context Compressor

A lightweight, local post-retrieval optimization layer for RAG (Retrieval-Augmented Generation) pipelines. Reduces LLM input tokens by stripping filler and redundant sentences from retrieved context **before** it reaches the LLM — cutting latency and API cost without an extra API call.

## The problem

Traditional RAG pipelines retrieve entire text chunks from a vector database and pass them straight into the LLM's context window. Most of that text is filler, redundant phrasing, or low-relevance to the actual query — but the LLM still has to process every token of it, which increases:

- **Time-to-first-token (TTFT)** latency
- **API cost** (billed per input token)

## The solution

Token-Diet sits between your vector DB and your LLM call:

```
User query → Vector DB retrieval → Token-Diet (local BM25 scoring) → Compressed context → LLM
```

1. Retrieved chunks are split into individual sentences
2. Each sentence is scored for relevance to the query using **BM25** (runs 100% locally, no API call)
3. Only the top-ranked, information-dense sentences are kept
4. The compressed context is passed to the LLM instead of the raw chunks

## Quick start

```bash
pip install -r requirements.txt
python token_diet.py
```

## Example output

```
Original words:    148
Compressed words:  58
Compression ratio: 60.8% tokens saved

Simulated latency before: 370.0 ms
Simulated latency after:  145.3 ms
```

## Usage in your own pipeline

```python
from token_diet import compress_context

result = compress_context(
    query="What causes high latency in RAG systems?",
    chunks=[retrieved_chunk_1, retrieved_chunk_2],
    keep_ratio=0.4,  # keep the top 40% most relevant sentences
)

# Pass result["compressed_text"] to your LLM instead of the raw chunks
```

## Roadmap

- Swap BM25 for a Cross-Encoder model (`ms-marco-MiniLM`) for higher-accuracy scoring
- Adaptive `keep_ratio` based on query complexity
- Per-domain tuning (legal/medical vs. casual text)
- Live dashboard for compression ratio + latency tracking in production

## Built for

vCET Hackathon 2026 — Pixels to Possibilities
