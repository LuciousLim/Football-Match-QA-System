# -*- coding: utf-8 -*-
"""
只读查看现有 Chroma 向量库（适配 chromadb==0.4.x）
- 不再使用 chromadb.Client/Settings，避免 LEGACY 报错
- 直接用 LangChain 的 Chroma 打开持久化目录
- 打印集合大小与前几条文档（metadata + text）
"""

import os
from pathlib import Path
from langchain_community.vectorstores import Chroma

# ---- 基本参数（保持与构建时一致）----
CHROMA_DIR = "chroma_football_events"   # 你构建时的目录
COLLECTION = "football_events"          # 你构建时的集合名
N_SHOW = 5                              # 展示多少条

# ---- 关闭遥测噪音 ----
os.environ["CHROMA_TELEMETRY"] = "False"

# ---- 打开现有向量库（不提供 embedding_function 也能读取底层 collection）----
abs_dir = Path(CHROMA_DIR).resolve()
print(f"\n📁 读取目录: {abs_dir}\n")

vectordb = Chroma(
    collection_name=COLLECTION,
    persist_directory=CHROMA_DIR,
    embedding_function=None,  # 只读元数据/文本时不需要嵌入器
)

# ---- 底层原生 collection（chromadb.Collection），可直接 .count() / .get() ----
col = vectordb._collection
cnt = col.count()
print(f"✅ 集合 `{COLLECTION}` 文档总数: {cnt}\n")

if cnt == 0:
    print("⚠️ 集合为空：请检查你构建时的目录/集合名是否一致。")
else:
    res = col.get(limit=min(N_SHOW, cnt))
    docs = res.get("documents", []) or []
    metas = res.get("metadatas", []) or []
    ids = res.get("ids", []) or []
    print("📄 示例文档（最多前几条）：\n")
    for i, (doc, meta, _id) in enumerate(zip(docs, metas, ids), 1):
        text = (doc[:300] + "...") if isinstance(doc, str) and len(doc) > 300 else doc
        print(f"---- 文档 {i} | id={_id} ----")
        print("id_odsp      :", meta.get("id_odsp"))
        print("event_team   :", meta.get("event_team"))
        print("season       :", meta.get("season"))
        print("shot_outcome :", meta.get("shot_outcome"))
        print("location     :", meta.get("location"))
        print("event_type   :", meta.get("event_type"))
        print("event_type2  :", meta.get("event_type2"))
        print("player       :", meta.get("player"))
        print("text         :", text)
        print()
