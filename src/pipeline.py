"""重排(词面特征逻辑回归,手写梯度) + 回答模拟器 + 拒答策略。

为什么要"模拟回答器":
    真实系统里这一步是 LLM,但 LLM 会把两件事混在一起——检索好不好、模型会不会编。
    为了把"检索质量 → 拒答 → 幻觉率"这条链单独量化出来,这里用一个**行为可控的
    回答模拟器**替代 LLM:它是否答错,只取决于"证据有没有被检索到、排在第几"。
    换成真实 LLM 时,只需替换 `answer_sim.answer` 一个函数。
"""
from __future__ import annotations

import numpy as np

from .corpus import tokenize


# ---------- 重排 ----------

def pair_features(query, doc, bm25_score):
    qt = set(tokenize(query))
    dt = set(tokenize(doc["text"]))
    inter = len(qt & dt)
    return np.array([
        bm25_score / 10.0,
        inter / max(len(qt), 1),
        inter / max(len(dt), 1),
        min(len(dt), 200) / 200.0,
        1.0 if inter > 0 else 0.0,
    ])


class Reranker:
    """逻辑回归:判断"这个文档是否真的相关"。

    注意两点,否则拒答信号会失效:
      * **特征标准化**:不做的话梯度下降收敛慢,输出概率会挤在很窄的一段里,
        导致"阈值扫描"退化成"要么全答、要么全拒";
      * **训练轮数足够**:概率要有区分度,才能拿来做拒答判据。
    """

    def __init__(self, dim=5, lr=0.3, epochs=300, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, dim)
        self.b = 0.0
        self.mu = np.zeros(dim)
        self.sd = np.ones(dim)
        self.lr, self.epochs = lr, epochs

    def _std(self, X):
        return (X - self.mu) / self.sd

    def prob(self, x):
        z = float(((x - self.mu) / self.sd) @ self.w + self.b)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-8
        Xs = self._std(X)
        for _ in range(self.epochs):
            p = 1.0 / (1.0 + np.exp(-(Xs @ self.w + self.b)))
            g = p - y
            self.w -= self.lr * (Xs.T @ g) / len(y)
            self.b -= self.lr * g.mean()
        return self

    def rerank(self, query, candidates, docs):
        scored = []
        for i, s in candidates:
            x = pair_features(query, docs[i], s)
            scored.append((i, s, self.prob(x)))
        scored.sort(key=lambda t: -t[2])
        return scored


# ---------- 回答模拟器 ----------

def answer_sim(rng, doc_rank, gold, top_prob):
    """返回 (answered_something, is_correct)。
    行为假设(显式写出,便于替换):
      * 金标在 top1 → 0.90 概率答对;
      * 金标在 top-k 内但不在 top1 → 0.55 概率答对;
      * 金标未被召回 → 0.80 概率给出"看起来合理但无依据"的答案(幻觉)。
    """
    if doc_rank is not None and doc_rank == 0:
        correct = rng.random() < 0.90
    elif doc_rank is not None:
        correct = rng.random() < 0.55
    else:
        correct = rng.random() < 0.20        # 没有证据时,偶尔"蒙对"
    return True, bool(correct)


# ---------- 拒答策略 ----------

def abstention_signals(ranked):
    """从重排结果里取出可用于拒答的信号。"""
    if not ranked:
        return {"top_prob": 0.0, "margin": 0.0}
    top_prob = ranked[0][2]
    margin = top_prob - (ranked[1][2] if len(ranked) > 1 else 0.0)
    return {"top_prob": top_prob, "margin": margin}


def decide(sig, tau):
    """置信度低于阈值就拒答。"""
    return sig["top_prob"] >= tau
