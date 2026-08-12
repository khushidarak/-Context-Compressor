"""
Token-Diet: Dynamic Context Compressor
----------------------------------------
A lightweight, local post-retrieval optimization layer for RAG pipelines.

Given a user query and one or more retrieved text chunks, this script:
 1. Splits chunks into individual sentences.
 2. Scores each sentence for relevance to the query using BM25.
 3. Keeps only the top-scoring, information-dense sentences.
 4. Reports the token/word savings and a simulated latency drop.

Usage:
    python token_diet.py

No external API calls are made -- scoring runs 100% locally.
"""

import re
import time
from rank_bm25 import BM25Okapi


def split_sentences(text: str):
    """Naive but dependency-free sentence splitter."""
    text = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())


def compress_context(query: str, chunks: list[str], keep_ratio: float = 0.5):
    """
    Compress retrieved chunks down to the most query-relevant sentences.

    Args:
        query: the user's question.
        chunks: list of raw retrieved text chunks (e.g. from a vector DB).
        keep_ratio: fraction of sentences to retain (0.5 = keep top 50%).

    Returns:
        dict with original text, compressed text, and metrics.
    """
    # Flatten all chunks into sentences, remembering original word count
    all_sentences = []
    for chunk in chunks:
        all_sentences.extend(split_sentences(chunk))

    original_text = " ".join(all_sentences)
    original_word_count = len(tokenize(original_text))

    if not all_sentences:
        return {
            "original_text": "",
            "compressed_text": "",
            "original_words": 0,
            "compressed_words": 0,
            "compression_ratio": 0.0,
            "simulated_latency_before_ms": 0,
            "simulated_latency_after_ms": 0,
        }

    # --- Local BM25 scoring (no API call, runs on-device) ---
    t0 = time.perf_counter()
    tokenized_corpus = [tokenize(s) for s in all_sentences]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenize(query))
    scoring_time = time.perf_counter() - t0

    # Rank sentences by relevance, keep top keep_ratio of them,
    # but preserve original order in the final output.
    n_keep = max(1, round(len(all_sentences) * keep_ratio))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep_indices = set(ranked_indices[:n_keep])
    compressed_sentences = [s for i, s in enumerate(all_sentences) if i in keep_indices]

    compressed_text = " ".join(compressed_sentences)
    compressed_word_count = len(tokenize(compressed_text))

    compression_ratio = 1 - (compressed_word_count / original_word_count) if original_word_count else 0

    # --- Simulated LLM latency model ---
    # Rough real-world heuristic: ~2.5ms of LLM time-to-first-token per input token,
    # used here only to *illustrate* the latency benefit, not as a measured API call.
    ms_per_token = 2.5
    simulated_latency_before = original_word_count * ms_per_token
    simulated_latency_after = compressed_word_count * ms_per_token + (scoring_time * 1000)

    return {
        "original_text": original_text,
        "compressed_text": compressed_text,
        "original_words": original_word_count,
        "compressed_words": compressed_word_count,
        "compression_ratio": round(compression_ratio * 100, 1),
        "simulated_latency_before_ms": round(simulated_latency_before, 1),
        "simulated_latency_after_ms": round(simulated_latency_after, 1),
    }


if __name__ == "__main__":
    query = "What causes high latency in RAG systems?"

    retrieved_chunks = [
        """Retrieval-augmented generation systems, often abbreviated as RAG, have become
        extremely popular in recent years for building question answering applications.
        High latency in RAG systems is often caused by passing overly long, unfiltered
        context chunks directly into the large language model. Many teams simply retrieve
        the top-k results from a vector database without any further processing. This
        balloons the number of input tokens the model has to process before it can even
        begin generating a response, which directly increases time-to-first-token.
        Additionally, redundant phrasing and filler sentences within retrieved documents
        add no new information but still consume valuable context window space and
        increase compute cost. It is worth noting that many teams are still exploring
        best practices in this fast-moving field. Furthermore, verbose retrieved passages
        often repeat the same facts using different wording across multiple chunks, which
        compounds the token bloat problem even further.""",
    ]

    result = compress_context(query, retrieved_chunks, keep_ratio=0.4)

    print("=" * 60)
    print("TOKEN-DIET: DYNAMIC CONTEXT COMPRESSOR")
    print("=" * 60)
    print(f"\nQuery: {query}\n")
    print(f"Original words:    {result['original_words']}")
    print(f"Compressed words:  {result['compressed_words']}")
    print(f"Compression ratio: {result['compression_ratio']}% tokens saved")
    print(f"\nSimulated latency before: {result['simulated_latency_before_ms']} ms")
    print(f"Simulated latency after:  {result['simulated_latency_after_ms']} ms")
    print("\n--- Compressed context passed to LLM ---")
    print(result["compressed_text"])
