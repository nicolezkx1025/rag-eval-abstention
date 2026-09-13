"""合成语料与查询:为 RAG 评测骨架提供可复现的数据。

说明:真实系统里的文档与查询来自公司内部,不能公开。这里用模板 + 主题词生成
一批"检索上确实有难度"的语料:同一主题的多个文档互为干扰、查询词与文档用词
并不完全一致(有同义替换),从而让检索质量本身成为一个变量。
"""
from __future__ import annotations

import numpy as np

TOPICS = {
    "监管报送": ["报送", "口径", "报表", "会计", "校验"],
    "风险指标": ["风险", "指标", "敞口", "限额", "预警"],
    "债券发行": ["债券", "发行", "承销", "利率", "募集"],
    "股权投资": ["股权", "投资", "估值", "退出", "持仓"],
    "数据治理": ["治理", "血缘", "质量", "主数据", "标准"],
    "模型对齐": ["对齐", "偏好", "奖励", "策略", "采样"],
    "检索增强": ["检索", "重排", "召回", "切片", "向量"],
    "不确定性": ["不确定", "置信", "校准", "覆盖率", "风险"],
    "状态估计": ["状态", "滤波", "协方差", "观测", "退化"],
    "多模态": ["多模态", "图文", "跨模态", "对齐", "表征"],
}

ASPECTS = ["定义", "口径", "实现细节", "常见问题", "评估方式", "演进历史", "工程权衡"]
SYNONYMS = {"报送": "上报", "指标": "度量", "债券": "固收产品", "治理": "管理",
            "检索": "召回查找", "不确定": "可靠程度", "状态": "位姿", "风险": "暴露"}


def _sentence(rng, topic, aspect, k):
    words = TOPICS[topic]
    w = str(rng.choice(words))
    if rng.random() < 0.5 and w in SYNONYMS:
        w = SYNONYMS[w]
    return "%s的%s：第%d条要点涉及%s与%s的配合。" % (topic, aspect, k, w, str(rng.choice(words)))


def make_corpus(rng, n_docs=60, sents_per_doc=6):
    topics = list(TOPICS)
    docs = []
    for i in range(n_docs):
        topic = topics[i % len(topics)]
        spans = [_sentence(rng, topic, str(rng.choice(ASPECTS)), k) for k in range(sents_per_doc)]
        docs.append({"id": i, "topic": topic, "text": " ".join(spans)})
    return docs


def make_queries(rng, docs, n_queries=120, gold_per_query=(1, 3)):
    by_topic = {}
    for d in docs:
        by_topic.setdefault(d["topic"], []).append(d["id"])
    queries = []
    for i in range(n_queries):
        topic = str(rng.choice(list(by_topic)))
        pool = by_topic[topic]
        n_gold = int(rng.integers(gold_per_query[0], gold_per_query[1] + 1))
        gold = list(rng.choice(pool, size=min(n_gold, len(pool)), replace=False))
        aspect = str(rng.choice(ASPECTS))
        # 查询用同义词措辞,制造词面不匹配(真实场景的常见难点)
        key = topic
        if rng.random() < 0.6:
            for a, b in SYNONYMS.items():
                key = key.replace(a, b)
        q = "%s 的 %s 是怎样的？" % (key, aspect)
        queries.append({"qid": i, "text": q, "gold": set(int(g) for g in gold), "topic": topic})
    return queries


def tokenize(text):
    """极简中文切分:按标点断句 + 2-gram 滑窗(够用且无依赖)。"""
    out = []
    buf = []
    for ch in text:
        if ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    grams = list(out)
    for tok in out:
        for i in range(len(tok) - 1):
            grams.append(tok[i:i + 2])
    return grams
