# If you 

import os
import json
import gc
import time
from pathlib import Path
from typing import List, Dict

# ===== NumPy 2.x patch =====
import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128

# ===== Key parameter =====
DASHSCOPE_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-6962d59e71694d45842e77d58315fab0")

JSONL_PATH = "rag_events.jsonl"          
CHROMA_DIR = "chroma_football_events"    
COLLECTION = "football_events"           

START_LINE = 740000       

EMBED_DIMS     = 1024
API_MAX_BATCH  = 10       
API_WORKERS    = 8      
DOC_BATCH      = 5000    
REBUILD = False

os.environ["CHROMA_TELEMETRY"] = "False"

from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.embeddings import Embeddings
from openai import OpenAI, APIConnectionError, RateLimitError

class TextV4Embeddings(Embeddings):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        dims: int = EMBED_DIMS,
        max_batch: int = API_MAX_BATCH,
        workers: int = API_WORKERS,
        timeout: float = 30.0,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = "text-embedding-v4"
        self.dims = dims
        self.max_batch = max_batch
        self.workers = workers

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        for attempt in range(4):
            try:
                resp = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                    dimensions=self.dims,
                    encoding_format="float",
                )
                return [d.embedding for d in resp.data]
            except (RateLimitError, APIConnectionError):
                time.sleep(0.5 * (2 ** attempt))

        resp = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dims,
            encoding_format="float",
        )
        return [d.embedding for d in resp.data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        chunks = [texts[i:i + self.max_batch] for i in range(0, len(texts), self.max_batch)]
        out_blocks: List[List[List[float]]] = [None] * len(chunks)  # type: ignore
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(self._embed_batch, ch): idx for idx, ch in enumerate(chunks)}
            for fut in as_completed(futs):
                idx = futs[fut]
                out_blocks[idx] = fut.result()
        return [v for block in out_blocks for v in block]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

embedding_fn = TextV4Embeddings(api_key=DASHSCOPE_KEY)

from langchain_community.vectorstores import Chroma

def make_vectordb():
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding_fn,
        persist_directory=CHROMA_DIR,
    )

def safe_str(x):
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return x.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

def iter_jsonl(path, start_line: int = 0):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue

def _parse_suffix(id_str: str, base: str) -> int:
    if id_str == base:
        return 1
    if id_str.startswith(base + "#"):
        try:
            return int(id_str.split("#", 1)[1])
        except Exception:
            return 0
    return 0

def _prime_base_counts_from_db(col, bases_batch: list[str]) -> dict[str, int]:
    out: dict[str, int] = {b: 0 for b in bases_batch}
    STEP = 40
    for i in range(0, len(bases_batch), STEP):
        group = bases_batch[i:i+STEP]
        where = {"$or": [{"id_odsp": {"$eq": b}} for b in group]}
        res = col.get(where=where, include=["metadatas"], limit=100000)
        ids = res.get("ids", []) or []
        metas = res.get("metadatas", []) or []
        for _id, m in zip(ids, metas):
            base = (m or {}).get("id_odsp")
            if not base or base not in out:
                continue
            s = _parse_suffix(_id, base)
            if s > out[base]:
                out[base] = s
    return out


from collections import defaultdict

def main():
    if REBUILD and Path(CHROMA_DIR).exists():
        import shutil
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    vectordb = make_vectordb()
    col = vectordb._collection  

    ids:   List[str]  = []
    docs:  List[str]  = []
    metas: List[dict] = []
    bases: List[str]  = []  

    id_counts = defaultdict(int)

    total = 0
    t0 = time.time()

    def flush_block():
        nonlocal ids, docs, metas, bases, total, id_counts
        if not ids:
            return

        unique_bases = sorted(set(bases))
        max_used = _prime_base_counts_from_db(col, unique_bases)  
        for b, mx in max_used.items():
            if mx > id_counts[b]:
                id_counts[b] = mx

        try:
            existing = set(col.get(ids=ids)["ids"])
        except Exception:
            existing = set()

        used = set(existing)  
        final_ids: List[str] = []

        for _id, base in zip(ids, bases):
            if id_counts[base] == 0:
                id_counts[base] = max(0, max_used.get(base, 0))
            new_id = _id
            while new_id in used:
                id_counts[base] += 1
                new_id = base if id_counts[base] == 1 else f"{base}#{id_counts[base]}"
            if new_id == base and (base not in max_used or max_used[base] == 0):
                id_counts[base] = max(1, id_counts[base])
            used.add(new_id)
            final_ids.append(new_id)

        embs = embedding_fn.embed_documents(docs)

        col.add(ids=final_ids, documents=docs, metadatas=metas, embeddings=embs)

        total += len(final_ids)
        print(f"... indexed {total}")

        ids.clear(); docs.clear(); metas.clear(); bases.clear()
        gc.collect()

    for i, row in enumerate(iter_jsonl(JSONL_PATH, start_line=START_LINE), START_LINE + 1):
        text = safe_str(row.get("text"))
        if not text:
            continue

        meta = {k: safe_str(row.get(k, "")) for k in
                ["id_odsp","event_team","season","shot_outcome",
                 "location","event_type","event_type2","player"]}

        base_id = meta.get("id_odsp") or f"row_{i:012d}"

        _id = base_id

        ids.append(_id)
        docs.append(text)
        metas.append(meta)
        bases.append(base_id)

        if len(ids) >= DOC_BATCH:
            flush_block()

    flush_block()

    print(f"Wrote：{total} 条 | time: {time.time()-t0:.1f}s | dir：{Path(CHROMA_DIR).resolve()} | collection：{COLLECTION}")

if __name__ == "__main__":
    main()
