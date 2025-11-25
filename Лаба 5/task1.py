"""
ЛР5 — RAG для научного анализа (Пермский период)
Один файл: загрузка статей из Wikipedia -> chunking -> embeddings -> Chroma -> RetrievalQA -> оценка.
"""

import os
import json
import time
import shutil
import math
import requests
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

# langchain / vectorstore imports (may require installation)
try:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    # fallback: we'll implement simple split if langchain not available
    RecursiveCharacterTextSplitter = None

try:
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.llms import Ollama
except Exception:
    OllamaEmbeddings = None
    Ollama = None

try:
    from langchain_chroma import Chroma
except Exception:
    Chroma = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Конфигурация (параметры)
# ---------------------------
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
OUTPUT_DIR = "./permian_docs"
CHROMA_DIR = "./chroma_permian"
EMBED_MODEL = "nomic-embed-text"   # пример: модель Ollama для эмбеддингов
LLM_MODEL = "llama3.2"             # пример: локальный LLM в Ollama
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4  # сколько документов доставать из вектора при RAG

# ---------------------------
# Утилиты для работы с Wikipedia
# ---------------------------

def fetch_wikipedia_pages(topic: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Запрашивает список релевантных страниц по теме через MediaWiki API,
    затем загружает их содержимое. Возвращает список словарей {title, pageid, content}.
    """
    # Поиск страниц по запросу
    params = {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "srlimit": limit,
        "format": "json"
    }
    r = requests.get(WIKI_API_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    results = []
    for item in data.get("query", {}).get("search", []):
        title = item.get("title")
        pageid = item.get("pageid")
        # Получаем текст в формате wikitext или экстракт
        p = {
            "action": "query",
            "prop": "extracts",
            "explaintext": True,
            "pageids": pageid,
            "format": "json"
        }
        rr = requests.get(WIKI_API_URL, params=p, timeout=30)
        rr.raise_for_status()
        page_json = rr.json()
        page = page_json.get("query", {}).get("pages", {}).get(str(pageid), {})
        content = page.get("extract", "") or ""
        results.append({"title": title, "pageid": pageid, "content": content})
    return results


def save_pages_local(pages: List[Dict[str, Any]], out_dir: str = OUTPUT_DIR) -> None:
    """
    Сохраняет загруженные страницы в текстовые файлы (one-file per page).
    """
    os.makedirs(out_dir, exist_ok=True)
    for page in pages:
        safe_name = "".join(c if c.isalnum() or c in " _-." else "_" for c in page["title"])
        path = os.path.join(out_dir, f"{safe_name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Title: {page['title']}\n\n{page['content']}")


def dataset_statistics_from_folder(directory: str) -> Dict[str, Any]:
    """
    Возвращает статистику по наборам документов в директории: количество файлов, общее число символов/слов, средняя длина.
    """
    stats = {"num_documents": 0, "total_chars": 0, "total_words": 0, "avg_chars": 0, "avg_words": 0}
    files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    stats["num_documents"] = len(files)
    for fn in files:
        p = os.path.join(directory, fn)
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        stats["total_chars"] += len(text)
        stats["total_words"] += len(text.split())
    if stats["num_documents"] > 0:
        stats["avg_chars"] = stats["total_chars"] / stats["num_documents"]
        stats["avg_words"] = stats["total_words"] / stats["num_documents"]
    return stats

# ---------------------------
# Чанкеры / обработка текста
# ---------------------------

def simple_text_splitter(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Простейший windows-чанкер: разбивает текст по пробелам, формируя чанки длиной ~chunk_size с overlap.
    Возвращает список текстовых чанков.
    """
    if not text:
        return []
    words = text.split()
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        j = min(n, i + chunk_size)
        chunk = " ".join(words[i:j])
        chunks.append(chunk)
        # шаг назад на overlap слов
        i = max(i + chunk_size - chunk_overlap, j)
    return chunks


def documents_to_chunks_from_folder(directory: str) -> List[Dict[str, Any]]:
    """
    Читает все .txt в директории и возвращает список чанков в формате:
    [{'doc_id': filename, 'title': title, 'text': chunk_text, 'meta': {...}}, ...]
    """
    chunks = []
    files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    for fn in files:
        path = os.path.join(directory, fn)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        # отделяем заголовок, если есть
        if raw.startswith("Title:"):
            try:
                title, body = raw.split("\n\n", 1)
                title = title.replace("Title:", "").strip()
            except ValueError:
                title = fn
                body = raw
        else:
            title = fn
            body = raw
        cks = simple_text_splitter(body)
        for idx, ck in enumerate(cks):
            chunks.append({
                "doc_id": fn,
                "title": title,
                "chunk_id": f"{fn}__{idx}",
                "text": ck,
                "meta": {"source": fn, "title": title, "chunk_index": idx}
            })
    return chunks

# ---------------------------
# Embeddings & Vector Store
# ---------------------------

class VectorStoreBuilder:
    """
    Построение векторного индекса через OllamaEmbeddings -> Chroma.
    Если Ollama/Chroma не доступны, можно использовать TF-IDF + in-memory cosine.
    """
    def __init__(self, chunks: List[Dict[str, Any]], persist_dir: str = CHROMA_DIR, embed_model: str = EMBED_MODEL):
        self.chunks = chunks
        self.persist_dir = persist_dir
        self.embed_model = embed_model
        self.vectorstore = None
        self.embeddings = None

    def build_chroma(self):
        if Chroma is None or OllamaEmbeddings is None:
            raise RuntimeError("Chroma или OllamaEmbeddings не установлены in environment.")
        # очистка старой БД
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
        # создаём список документов/метаданных в формате, ожидаемом Chroma.from_documents
        docs = [c["text"] for c in self.chunks]
        metadatas = [c["meta"] for c in self.chunks]
        embeddings = OllamaEmbeddings(model=self.embed_model, base_url="http://localhost:11434", show_progress=True)
        vs = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=self.persist_dir, metadatas=metadatas)
        self.vectorstore = vs
        self.embeddings = embeddings
        vs.persist()
        return vs

    def build_tfidf(self):
        # Простая модель: TF-IDF + хранение текстов и метаданных
        docs = [c["text"] for c in self.chunks]
        vectorizer = TfidfVectorizer(max_features=32768)
        X = vectorizer.fit_transform(docs)
        self.vectorstore = {"type": "tfidf", "X": X, "vectorizer": vectorizer, "chunks": self.chunks}
        return self.vectorstore

# ---------------------------
# RAG System (Retrieval + LLM)
# ---------------------------

class PermianRAGSystem:
    """
    Обёртка над векторным хранилищем и LLM (Ollama) для выполнения Retrieval-Augmented Generation.
    Поддерживает два режима: 'chroma' (если есть Chroma) и 'tfidf' (fallback).
    """
    def __init__(self, vectorstore_obj: Any, llm_model: str = LLM_MODEL, mode: str = "chroma"):
        self.vs = vectorstore_obj
        self.mode = mode
        self.llm_model = llm_model
        if Ollama is not None:
            self.llm = Ollama(model=llm_model, base_url="http://localhost:11434", temperature=0.3, top_p=0.9, num_predict=512)
        else:
            self.llm = None

    def retrieve(self, query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Возвращает список найденных фрагментов (dict с keys: text, meta, score).
        Для TF-IDF возвращает косинусные похожести; для Chroma — использует .similarity_search.
        """
        if self.mode == "chroma":
            # ожидание, что Chroma API предоставляет метод as_retriever или similarity_search
            docs = self.vs.similarity_search(query, k=k)
            out = []
            for d in docs:
                out.append({"text": d.page_content if hasattr(d, "page_content") else d["text"], "meta": getattr(d, "metadata", d.metadata if hasattr(d, "metadata") else d.metadata if hasattr(d, "metadata") else d)})
            return out
        else:
            # TF-IDF fallback
            X = self.vs["X"]
            vectorizer = self.vs["vectorizer"]
            qv = vectorizer.transform([query])
            sims = cosine_similarity(qv, X)[0]
            top_idx = np.argsort(-sims)[:k]
            res = []
            for idx in top_idx:
                res.append({"text": self.vs["chunks"][idx]["text"], "meta": self.vs["chunks"][idx]["meta"], "score": float(sims[idx])})
            return res

    def generate_answer(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Формирует промпт и вызывает LLM (если доступен). Если LLM нет, возвращает конкатенацию контекста.
        """
        context_text = "\n\n".join([f"Source: {d.get('meta', {}).get('source','?')}\n{d.get('text','')[:1000]}" for d in context_docs])
        prompt = f"""You are an expert in paleontology and the Permian period.
Use ONLY the provided context to answer the question. If the answer is not present, say explicitly "This information is not available in the provided documents."

Context:
{context_text}

Question: {question}

Detailed Answer:"""
        if self.llm is None:
            # fallback: return top-k contexts + question (not a real generation)
            return "LLM not available — returning concatenated contexts:\n\n" + context_text
        resp = self.llm(prompt)
        return resp

    def answer_question(self, question: str, k: int = TOP_K) -> Dict[str, Any]:
        docs = self.retrieve(question, k=k)
        answer = self.generate_answer(question, docs)
        return {"question": question, "answer": answer, "sources": docs}

# ---------------------------
# Оценка качества (simple keyword-based & cosine)
# ---------------------------

def evaluate_keyword_coverage(answer: str, expected_keywords: List[str]) -> Dict[str, Any]:
    """
    Считает покрытие ключевых слов: доля ключевых слов, встреченных в ответе.
    """
    ans_lower = answer.lower()
    hit = [kw for kw in expected_keywords if kw.lower() in ans_lower]
    score = len(hit) / len(expected_keywords) if expected_keywords else 0.0
    return {"expected": expected_keywords, "hits": hit, "score": score}


def evaluate_cosine_with_reference(answer: str, reference_texts: List[str]) -> float:
    """
    Вычисляет косинусную схожесть между ответом и набором эталонных текстов (TF-IDF).
    Возвращает максимальную схожесть.
    """
    if not reference_texts:
        return 0.0
    vec = TfidfVectorizer(max_features=32768)
    docs = [answer] + reference_texts
    X = vec.fit_transform(docs)
    sims = cosine_similarity(X[0], X[1:])[0]
    return float(np.max(sims))

# ---------------------------
# Вспомогательные: визуализация результатов
# ---------------------------

def plot_evaluation(results: List[Dict[str, Any]]):
    """
    Построение bar-plot по метрикам: keyword score и cosine.
    results: list of dicts with keys 'question', 'keyword_score', 'cosine'
    """
    questions = [r["question"][:50] + ("..." if len(r["question"])>50 else "") for r in results]
    kw_scores = [r.get("keyword_score",0) for r in results]
    cos_scores = [r.get("cosine",0) for r in results]

    x = np.arange(len(questions))
    width = 0.35
    plt.figure(figsize=(10,5))
    plt.bar(x - width/2, kw_scores, width, label="Ключевые слова (доля)")
    plt.bar(x + width/2, cos_scores, width, label="Cosine")
    plt.xticks(x, questions, rotation=45, ha="right")
    plt.ylabel("Оценка")
    plt.title("Оценка качества ответов RAG")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------------------
# MAIN pipeline orchestration
# ---------------------------

def main_pipeline(topic: str = "Permian", num_pages: int = 8, use_chroma: bool = True):
    # 1. fetch pages
    print("Fetching Wikipedia pages...")
    pages = fetch_wikipedia_pages(topic, limit=num_pages)
    print(f"Fetched {len(pages)} pages.")
    save_pages_local(pages, OUTPUT_DIR)
    stats = dataset_statistics_from_folder(OUTPUT_DIR)
    print("Dataset stats:", stats)

    # 2. chunking
    print("Chunking documents...")
    chunks = documents_to_chunks_from_folder(OUTPUT_DIR)
    print(f"Total chunks: {len(chunks)}")

    # 3. build vectorstore
    vs_builder = VectorStoreBuilder(chunks, persist_dir=CHROMA_DIR)
    if use_chroma and Chroma is not None and OllamaEmbeddings is not None:
        print("Building Chroma vectorstore (OllamaEmbeddings)...")
        vs = vs_builder.build_chroma()
        mode = "chroma"
    else:
        print("Building TF-IDF fallback vectorstore...")
        vs = vs_builder.build_tfidf()
        mode = "tfidf"

    # 4. init RAG
    rag = PermianRAGSystem(vs, llm_model=LLM_MODEL, mode=mode)

    # 5. Example questions and evaluation
    questions = [
        {"question": "When did the Permian period start and end?", "expected_keywords": ["Permian", "252", "298", "million years"]},
        {"question": "What caused the Permian–Triassic extinction?", "expected_keywords": ["Siberian Traps", "volcanism", "climate change", "anoxia"]},
        {"question": "What were dominant terrestrial animals in the Permian?", "expected_keywords": ["therapsids", "reptiles", "pelycosaurs"]},
    ]

    results = []
    for q in questions:
        res = rag.answer_question(q["question"], k=TOP_K)
        ek = q.get("expected_keywords", [])
        kw = evaluate_keyword_coverage(res["answer"], ek)
        cos = evaluate_cosine_with_reference(res["answer"], [p["content"][:2000] for p in pages])
        output = {"question": q["question"], "answer": res["answer"], "sources": res["sources"], "keyword_score": kw["score"], "keyword_hits": kw["hits"], "cosine": cos}
        results.append(output)
        print("\n" + "="*80)
        print("Q:", q["question"])
        print("-"*80)
        print("Answer (truncated):", res["answer"][:800])

    # 6. plotting
    plot_evaluation(results)
    # 7. return structured results
    return {"stats": stats, "results": results, "mode": mode}

# ---------------------------
# Запуск
# ---------------------------

if __name__ == "__main__":
    out = main_pipeline(topic="Permian", num_pages=8, use_chroma=True)
    # сохранить результаты в JSON
    with open("permian_rag_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Done. Results saved to permian_rag_results.json")
