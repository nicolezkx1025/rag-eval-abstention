"""BM25 检索(纯 numpy 实现)+ 检索质量指标。"""
from __future__ import annotations

import math

import numpy as np

from .corpus import tokenize


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.doc_ids = [d["id"] for d in docs]
        self.tokens = [tokenize(d["text"]) for d in docs]
        self.len = np.array([len(t) for t in self.tokens], float)
        self.avg_len = float(self.len.mean()) if len(self.len) else 0.0

        df = {}
        for toks in self.tokens:
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        self.df = df
        self.N = len(docs)
        # 倒排:词 → [(文档下标, 词频)]
        self.inv = {}
        for i, toks in enumerate(self.tokens):
            tf = {}
            for w in toks:
                tf[w] = tf.get(w, 0) + 1
            for w, c in tf.items():
                self.inv.setdefault(w, []).append((i, c))

    def idf(self, w):
        n = self.df.get(w, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, query):
        scores = np.zeros(self.N)
        for w in tokenize(query):
            if w not in self.inv:
                continue
            idf = self.idf(w)
            for i, tf in self.inv[w]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.len[i] / max(self.avg_len, 1e-9))
                scores[i] += idf * tf * (self.k1 + 1) / max(denom, 1e-9)
        return scores

    def search(self, query, k=5):
        s = self.score(query)
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx]


def _top_ids(rank, k=None):
    """兼容两种排序结果:BM25 的 (doc_id, score) 与重排后的 (doc_id, score, prob)。"""
    items = rank[:k] if k else rank
    return {int(it[0]) for it in items}


def recall_at_k(rankings, queries, k=5):
    hit = 0
    for rank, q in zip(rankings, queries):
        if _top_ids(rank, k) & q["gold"]:
            hit += 1
    return hit / len(queries) if queries else 0.0


def mrr(rankings, queries):
    total = 0.0
    for rank, q in zip(rankings, queries):
        for pos, item in enumerate(rank, start=1):
            if int(item[0]) in q["gold"]:
                total += 1.0 / pos
                break
    return total / len(queries) if queries else 0.0


def gold_recall_at_k(rankings, queries, k=5):
    """金标文档被召回的比例(不是"至少命中一个",而是覆盖多少)。"""
    vals = []
    for rank, q in zip(rankings, queries):
        vals.append(len(_top_ids(rank, k) & q["gold"]) / max(len(q["gold"]), 1))
    return float(np.mean(vals)) if vals else 0.0
