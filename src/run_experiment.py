"""主实验:检索质量 → 拒答阈值 → 覆盖率 / 精确率 / 幻觉率 的关系。

用法:
    python src/run_experiment.py
    python src/run_experiment.py --queries 200 --k 5
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.corpus import make_corpus, make_queries
from src.bm25 import BM25, recall_at_k, mrr, gold_recall_at_k
from src.pipeline import Reranker, pair_features, answer_sim, abstention_signals, decide

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=60)
    ap.add_argument("--queries", type=int, default=120)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    docs = make_corpus(rng, n_docs=args.docs)
    queries = make_queries(rng, docs, n_queries=args.queries)

    # 划分:一半训练重排器,一半测试
    n_tr = len(queries) // 2
    train_q, test_q = queries[:n_tr], queries[n_tr:]

    bm25 = BM25(docs)

    # ---------- 训练重排器 ----------
    X, y = [], []
    for q in train_q:
        for i, s in bm25.search(q["text"], k=8):
            X.append(pair_features(q["text"], docs[i], s))
            y.append(1.0 if i in q["gold"] else 0.0)
    reranker = Reranker().fit(X, y)
    print("重排器训练样本:", len(X), "| 正例比例 %.3f" % float(np.mean(y)))

    # ---------- 检索指标 ----------
    rank_bm25, rank_rr = [], []
    for q in test_q:
        cand = bm25.search(q["text"], k=8)
        rank_bm25.append(cand[:args.k])
        rank_rr.append([(i, s, p) for (i, s, p) in reranker.rerank(q["text"], cand, docs)][:args.k])

    print("\n=== 检索质量(测试集 %d 条查询)===" % len(test_q))
    print("%-16s %-12s %-12s %-12s" % ("", "Recall@k", "金标覆盖@k", "MRR"))
    print("%-16s %-12.4f %-12.4f %-12.4f" % ("BM25",
          recall_at_k(rank_bm25, test_q, args.k),
          gold_recall_at_k(rank_bm25, test_q, args.k),
          mrr(rank_bm25, test_q)))
    print("%-16s %-12.4f %-12.4f %-12.4f" % ("BM25 + 重排",
          recall_at_k(rank_rr, test_q, args.k),
          gold_recall_at_k(rank_rr, test_q, args.k),
          mrr([[(i, s) for i, s, _ in r] for r in rank_rr], test_q)))

    # ---------- 拒答阈值扫描 ----------
    taus = [round(0.05 * i, 2) for i in range(1, 20)]     # 0.05 → 0.95
    signals = []
    for q, ranked in zip(test_q, rank_rr):
        sig = abstention_signals(ranked)
        gold = q["gold"]
        doc_rank = None
        for pos, (i, _s, _p) in enumerate(ranked):
            if i in gold:
                doc_rank = pos
                break
        signals.append((sig, doc_rank, gold))

    rows = []
    print("\n=== 拒答阈值扫描(RAG 回答质量)===")
    print("%-8s %-10s %-12s %-10s %-10s %-12s" % ("阈值τ", "覆盖率", "回答精确率", "幻觉率", "拒答率", "风险(错误率)"))
    for tau in taus:
        answered = correct = halluc = 0
        for sig, doc_rank, gold in signals:
            if not decide(sig, tau):
                continue
            answered += 1
            _ok, is_correct = answer_sim(rng, doc_rank, gold, sig["top_prob"])
            if is_correct:
                correct += 1
            else:
                halluc += 1
        n = len(signals)
        cov = answered / n
        prec = correct / answered if answered else 0.0
        hal = halluc / n
        risk = (answered - correct) / answered if answered else 0.0
        rows.append({"tau": tau, "coverage": round(cov, 4),
                     "answer_precision": round(prec, 4),
                     "hallucination_rate": round(hal, 4),
                     "abstain_rate": round(1 - cov, 4),
                     "selective_risk": round(risk, 4)})
        print("%-8.2f %-10.4f %-12.4f %-10.4f %-10.4f %-12.4f" % (tau, cov, prec, hal, 1 - cov, risk))

    # 风险—覆盖率曲线下面积(越小越好):衡量"用置信度排序"的整体质量
    rc = sorted([(r["coverage"], r["selective_risk"]) for r in rows if r["coverage"] > 0])
    if len(rc) > 1:
        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")   # numpy 2.x 改名了
        aurc = float(_trapz([p[1] for p in rc], [p[0] for p in rc]))
    else:
        aurc = float("nan")

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "metrics.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n已写出:", path)
    print("风险—覆盖率曲线下面积 AURC = %.4f(越小越好)" % aurc)

    always = rows[0]
    valid = [r for r in rows if r["coverage"] > 0]
    best = max(valid, key=lambda r: r["answer_precision"]) if valid else always
    print("\n不拒答(τ=%.2f):精确率 %.4f,幻觉率 %.4f" % (
        always["tau"], always["answer_precision"], always["hallucination_rate"]))
    print("最佳 τ=%.2f:精确率 %.4f,幻觉率 %.4f,覆盖率 %.4f" % (
        best["tau"], best["answer_precision"], best["hallucination_rate"], best["coverage"]))


if __name__ == "__main__":
    main()
