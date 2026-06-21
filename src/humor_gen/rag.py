from __future__ import annotations

from pathlib import Path
from typing import Protocol

from humor_gen.data import load_dataset
from humor_gen.utils import read_jsonl


class RetrieverProtocol(Protocol):
    def retrieve(self, query: str, k: int) -> list[str]:
        ...


class TfidfRetriever:
    def __init__(self, documents: list[str]):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError as exc:
            raise RuntimeError("scikit-learn is not installed") from exc

        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(documents)
        self.cosine_similarity = cosine_similarity

    def retrieve(self, query: str, k: int) -> list[str]:
        query_vec = self.vectorizer.transform([query])
        scores = self.cosine_similarity(query_vec, self.matrix).ravel()
        best = scores.argsort()[::-1][:k]
        return [self.documents[idx] for idx in best if scores[idx] > 0][:k]


class KeywordRetriever:
    def __init__(self, documents: list[str]):
        self.documents = documents
        self.doc_terms = [_terms(doc) for doc in documents]

    def retrieve(self, query: str, k: int) -> list[str]:
        query_terms = _terms(query)
        scored = []
        for idx, terms in enumerate(self.doc_terms):
            overlap = len(query_terms & terms)
            scored.append((overlap, idx))
        scored.sort(reverse=True)
        return [self.documents[idx] for score, idx in scored if score > 0][:k]


class SentenceTransformerRetriever:
    def __init__(self, documents: list[str], model_name: str):
        from sentence_transformers import SentenceTransformer

        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(documents, normalize_embeddings=True, show_progress_bar=False)

    def retrieve(self, query: str, k: int) -> list[str]:
        import numpy as np

        query_embedding = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = np.dot(self.embeddings, query_embedding)
        best = np.argsort(scores)[::-1][:k]
        return [self.documents[int(idx)] for idx in best if scores[int(idx)] > 0][:k]


class HFWikipediaRetriever:
    def __init__(
        self,
        dataset_name: str,
        revision: str,
        split: str,
        embedding_model: str,
        text_column: str = "text",
        embedding_column: str = "embeddings",
        max_context_chars: int = 1200,
        query_device: str = "cpu",
        cache_dir: str | None = None,
    ):
        try:
            import numpy as np
            from datasets import load_dataset
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "HF Wikipedia RAG requires 'datasets', 'sentence-transformers', and 'numpy'."
            ) from exc

        load_kwargs = {"revision": revision, "split": split}
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        dataset = load_dataset(dataset_name, **load_kwargs)
        self.documents = [str(text) for text in dataset[text_column]]
        self.embeddings = np.array(dataset[embedding_column], dtype=np.float32)
        self.embeddings = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-12)
        self.model = SentenceTransformer(embedding_model, device=query_device)
        self.max_context_chars = max_context_chars
        self.np = np

    def retrieve(self, query: str, k: int) -> list[str]:
        query_embedding = self.model.encode([query]).astype(self.np.float32)
        query_embedding = query_embedding / (self.np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-12)
        if query_embedding.shape[1] != self.embeddings.shape[1]:
            raise ValueError(
                "Query embedding dimension does not match the HF Wikipedia embeddings. "
                "Use the same embedding model that produced the dataset embeddings."
            )
        scores = self.embeddings @ query_embedding[0]
        best = self.np.argsort(-scores)[:k]
        return _truncate_contexts([self.documents[int(idx)] for idx in best], self.max_context_chars)


def build_retriever(input_path: str, rag_cfg: dict, mock: bool = False) -> RetrieverProtocol:
    retriever_cfg = rag_cfg.get("retriever", {})
    backend = retriever_cfg.get("backend", "sentence-transformers")
    if mock or backend == "tfidf":
        documents = load_corpus(retriever_cfg.get("corpus_path"), input_path)
        return _tfidf_or_keyword(documents)
    if backend in {"hf-wikipedia", "huggingface-wikipedia"}:
        n_docs = int(retriever_cfg.get("n_docs", 25000))
        split = retriever_cfg.get("hf_split") or f"train[:{n_docs}]"
        return HFWikipediaRetriever(
            dataset_name=retriever_cfg.get("hf_dataset", "not-lain/wikipedia"),
            revision=retriever_cfg.get("hf_revision", "embedded"),
            split=split,
            embedding_model=retriever_cfg.get("embedding_model", "mixedbread-ai/mxbai-embed-large-v1"),
            text_column=retriever_cfg.get("text_column", "text"),
            embedding_column=retriever_cfg.get("embedding_column", "embeddings"),
            max_context_chars=int(retriever_cfg.get("max_context_chars", 1200)),
            query_device=retriever_cfg.get("query_device", "cpu"),
            cache_dir=retriever_cfg.get("cache_dir"),
        )
    documents = load_corpus(retriever_cfg.get("corpus_path"), input_path)
    try:
        model_name = retriever_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        return SentenceTransformerRetriever(documents, model_name)
    except Exception:
        fallback = retriever_cfg.get("fallback_backend", "tfidf")
        if fallback != "tfidf":
            raise
        return _tfidf_or_keyword(documents)


def load_corpus(corpus_path: str | None, input_path: str) -> list[str]:
    docs: list[str] = []
    if corpus_path and Path(corpus_path).exists():
        for row in read_jsonl(corpus_path):
            text = row.get("text") or row.get("joke") or row.get("headline")
            if text:
                docs.append(str(text))
    if not docs:
        for item in load_dataset(input_path):
            if item["input_type"] == "headline":
                docs.append(f"Headline humor example about: {item['headline']}")
            else:
                docs.append(f"Word-pair humor example using {item['word1']} and {item['word2']}.")
    if not docs:
        raise ValueError("RAG corpus is empty.")
    return docs


def _tfidf_or_keyword(documents: list[str]) -> RetrieverProtocol:
    try:
        return TfidfRetriever(documents)
    except RuntimeError:
        return KeywordRetriever(documents)


def _terms(text: str) -> set[str]:
    import re

    return {token for token in re.findall(r"[a-zA-Z]{3,}", text.casefold())}


def _truncate_contexts(documents: list[str], max_chars: int) -> list[str]:
    if not documents:
        return []
    per_document = max(1, max_chars // len(documents))
    contexts = []
    for document in documents:
        text = " ".join(document.split())
        if not text:
            continue
        contexts.append(text[:per_document])
    return contexts
