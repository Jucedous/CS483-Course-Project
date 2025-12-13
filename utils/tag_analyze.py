#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import csv
import math
from collections import Counter, defaultdict
from itertools import combinations
from typing import List, Dict, Any, Tuple

# ================= 配置（可按需调整） =================
MIN_TAG_DF = 5               # 标签最小文档频次（df）阈值，低于则不参与关联计算
MIN_COOCCUR = 2              # 标签对最小共现次数阈值，低于则不导出
TOP_ASSOC_PER_TAG = 20       # 每个标签导出的 Top-K 关联标签数
SORT_METRIC = "lift"         # Top-K 排序指标：lift|cosine|jaccard|phi|chi2|pmi|npmi|hypergeom_z
LIMIT_VOCAB_TOP_K = 5000     # 仅保留总体频次最高的前 K 个标签参与两两组合，控制爆炸
SMOOTH_EPS = 0.0             # PMI 平滑（默认 0：仅 cooccur>0 才计算）

# ================= 工具函数 =================
def gather_json_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path] if path.lower().endswith(".json") else []
    found = []
    for root, _, files in os.walk(path):
        for fn in files:
            if fn.lower().endswith(".json"):
                found.append(os.path.join(root, fn))
    return sorted(found)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def normalize_tag(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    t = " ".join(tag.strip().split())
    t = t.lower().replace(" ", "_")
    t = t.strip(",.;:()[]{}")
    return t

def unique_tags_for_message(tags: Any) -> List[str]:
    if not isinstance(tags, list):
        return []
    normed = [normalize_tag(t) for t in tags if isinstance(t, str)]
    # 去空、去重
    seen, out = set(), []
    for t in normed:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out

# ================= 频次与共现统计 =================
def collect_counts(files: List[str]) -> Tuple[int, Counter, Counter]:
    """返回 N, tag_df_counter, cooccur_counter"""
    tag_df = Counter()
    cooccur = Counter()
    N = 0

    # 首先统计每封邮件的标签集合（去重）
    all_msgs_tags: List[List[str]] = []

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        msgs = data.get("messages", [])
        for m in msgs:
            tags = unique_tags_for_message(m.get("tags"))
            if not tags:
                N += 1
                continue
            for t in tags:
                tag_df[t] += 1
            all_msgs_tags.append(tags)
            N += 1

    # 限定词表规模（防止两两组合爆炸）
    if LIMIT_VOCAB_TOP_K and len(tag_df) > LIMIT_VOCAB_TOP_K:
        top_vocab = set([t for t, _ in tag_df.most_common(LIMIT_VOCAB_TOP_K)])
    else:
        top_vocab = None

    # 统计共现
    for tags in all_msgs_tags:
        if top_vocab is not None:
            tags = [t for t in tags if t in top_vocab]
        # 应用 df 阈值
        tags = [t for t in tags if tag_df[t] >= MIN_TAG_DF]
        if len(tags) < 2:
            continue
        # 两两组合
        for a, b in combinations(sorted(tags), 2):
            cooccur[(a, b)] += 1

    return N, tag_df, cooccur

# ================= 关联指标计算 =================
def calc_metrics_for_pair(a: str, b: str, N: int, df_a: int, df_b: int, co: int) -> Dict[str, Any]:
    # 列联表
    # a: cooccur, b_only: df_b - cooccur, a_only: df_a - cooccur, neither: N - df_a - df_b + cooccur
    a_b = co
    a_only = max(df_a - co, 0)
    b_only = max(df_b - co, 0)
    neither = max(N - df_a - df_b + co, 0)

    # 基本概率
    p_a = df_a / N if N > 0 else 0.0
    p_b = df_b / N if N > 0 else 0.0
    p_ab = co / N if N > 0 else 0.0

    # jaccard
    denom_jac = (df_a + df_b - co)
    jaccard = (co / denom_jac) if denom_jac > 0 else 0.0

    # cosine
    denom_cos = math.sqrt(df_a * df_b) if df_a > 0 and df_b > 0 else 0.0
    cosine = (co / denom_cos) if denom_cos > 0 else 0.0

    # lift
    denom_lift = (df_a * df_b)
    lift = (co * N / denom_lift) if denom_lift > 0 and N > 0 else 0.0

    # PMI / NPMI（仅 co>0）
    if co > 0 and p_a > 0 and p_b > 0 and p_ab > 0:
        pmi = math.log2((p_ab + SMOOTH_EPS) / (p_a * p_b + SMOOTH_EPS))
        npmi = pmi / (-math.log2(p_ab)) if p_ab > 0 else 0.0
    else:
        pmi = 0.0
        npmi = 0.0

    # Phi 系数（Pearson correlation for binary）
    # phi = (ad - bc) / sqrt((a+b)(a+c)(b+d)(c+d))
    a_val = a_b
    b_val = a_only
    c_val = b_only
    d_val = neither
    denom_phi = math.sqrt((a_val + b_val) * (a_val + c_val) * (b_val + d_val) * (c_val + d_val))
    num_phi = (a_val * d_val - b_val * c_val)
    phi = (num_phi / denom_phi) if denom_phi > 0 else 0.0

    # Chi-square（Pearson, 1 df）
    # 期望值：E = row_sum * col_sum / N，sum((O-E)^2/E)
    # 等价快速公式：chi2 = N * (ad - bc)^2 / ((a+b)(a+c)(b+d)(c+d))
    denom_chi = (a_val + b_val) * (a_val + c_val) * (b_val + d_val) * (c_val + d_val)
    chi2 = (N * (num_phi ** 2) / denom_chi) if denom_chi > 0 else 0.0

    # 超几何基线的 Z（近似）
    # E = df_a * df_b / N
    # Var ≈ df_a*df_b*(N-df_a)*(N-df_b) / (N^2*(N-1))
    expected = (df_a * df_b) / N if N > 0 else 0.0
    var_num = df_a * df_b * (max(N - df_a, 0)) * (max(N - df_b, 0))
    var_den = (N * N * max(N - 1, 1))
    variance = (var_num / var_den) if var_den > 0 else 0.0
    std = math.sqrt(variance) if variance > 0 else 0.0
    z = ((co - expected) / std) if std > 0 else 0.0

    return {
        "tag_a": a,
        "tag_b": b,
        "cooccur": co,
        "df_a": df_a,
        "df_b": df_b,
        "N": N,
        "jaccard": jaccard,
        "cosine": cosine,
        "lift": lift,
        "pmi": pmi,
        "npmi": npmi,
        "phi": phi,
        "chi2": chi2,
        "hypergeom_z": z,
    }

# ================= 主流程 =================
def analyze(path: str, out_dir: str):
    files = gather_json_files(path)
    if not files:
        print(f"未找到 JSON：{path}")
        sys.exit(1)
    print(f"发现 {len(files)} 个 JSON，统计频次与共现…")

    N, tag_df, cooccur = collect_counts(files)
    print(f"总邮件数 N={N}，标签数={len(tag_df)}，标签对数={len(cooccur)}")

    # 频次导出
    freq_rows = [{"tag": t, "df": c, "df_pct": f"{(c / N * 100.0):.2f}" if N else "0.00"}
                 for t, c in tag_df.most_common()]
    ensure_dir(out_dir)
    write_csv(os.path.join(out_dir, "tag_frequency_summary.csv"), freq_rows, ["tag", "df", "df_pct"])

    # 关联度量
    pair_rows = []
    for (a, b), co in cooccur.items():
        df_a = tag_df[a]
        df_b = tag_df[b]
        if df_a < MIN_TAG_DF or df_b < MIN_TAG_DF or co < MIN_COOCCUR:
            continue
        m = calc_metrics_for_pair(a, b, N, df_a, df_b, co)
        pair_rows.append(m)

    # 导出全部标签对（按选择的排序指标）
    sort_key = SORT_METRIC if (pair_rows and SORT_METRIC in pair_rows[0]) else "lift"
    pair_rows.sort(key=lambda r: r.get(sort_key, 0.0), reverse=True)

    write_csv(
        os.path.join(out_dir, "associations_tag_pairs.csv"),
        pair_rows,
        ["tag_a", "tag_b", "cooccur", "df_a", "df_b", "N",
         "jaccard", "cosine", "lift", "pmi", "npmi", "phi", "chi2", "hypergeom_z"]
    )

    # 每标签 Top-K
    per_tag_map = defaultdict(list)
    for r in pair_rows:
        per_tag_map[r["tag_a"]].append(r)
        per_tag_map[r["tag_b"]].append(r)  # 让每个标签能看到另一侧的关联

    # 展开为 CSV 行
    top_rows = []
    for t, rows in per_tag_map.items():
        rows.sort(key=lambda x: x.get(SORT_METRIC, 0.0), reverse=True)
        for i, rr in enumerate(rows[:TOP_ASSOC_PER_TAG], 1):
            other = rr["tag_b"] if rr["tag_a"] == t else rr["tag_a"]
            top_rows.append({
                "tag": t,
                "rank": i,
                "assoc_tag": other,
                "cooccur": rr["cooccur"],
                "df_self": tag_df[t],
                "df_other": tag_df[other],
                SORT_METRIC: rr.get(SORT_METRIC, 0.0),
                "lift": rr["lift"],
                "cosine": rr["cosine"],
                "jaccard": rr["jaccard"],
                "phi": rr["phi"],
                "chi2": rr["chi2"],
                "pmi": rr["pmi"],
                "npmi": rr["npmi"],
                "hypergeom_z": rr["hypergeom_z"],
            })
    write_csv(
        os.path.join(out_dir, "top_associations_by_tag.csv"),
        top_rows,
        ["tag", "rank", "assoc_tag", "cooccur", "df_self", "df_other",
         SORT_METRIC, "lift", "cosine", "jaccard", "phi", "chi2", "pmi", "npmi", "hypergeom_z"]
    )

    # 简单统计摘要
    def quantiles(vals: List[float]) -> Tuple[float, float, float]:
        if not vals:
            return (0.0, 0.0, 0.0)
        xs = sorted(vals)
        def q(p):
            idx = max(0, min(len(xs)-1, int(p * (len(xs)-1))))
            return xs[idx]
        return (q(0.25), q(0.5), q(0.75))

    lifts = [r["lift"] for r in pair_rows]
    cosines = [r["cosine"] for r in pair_rows]
    jacs = [r["jaccard"] for r in pair_rows]
    phis = [r["phi"] for r in pair_rows]
    chis = [r["chi2"] for r in pair_rows]
    pmis = [r["pmi"] for r in pair_rows]
    npm = [r["npmi"] for r in pair_rows]
    zhs = [r["hypergeom_z"] for r in pair_rows]

    q_lift = quantiles(lifts)
    q_cos = quantiles(cosines)
    q_jac = quantiles(jacs)
    q_phi = quantiles(phis)
    q_chi = quantiles(chis)
    q_pmi = quantiles(pmis)
    q_npmi = quantiles(npm)
    q_zh = quantiles(zhs)

    with open(os.path.join(out_dir, "assoc_meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"Total messages: {N}\n")
        f.write(f"Unique tags: {len(tag_df)}\n")
        f.write(f"Tag pairs (after thresholds): {len(pair_rows)}\n")
        f.write(f"Thresholds: MIN_TAG_DF={MIN_TAG_DF}, MIN_COOCCUR={MIN_COOCCUR}, LIMIT_VOCAB_TOP_K={LIMIT_VOCAB_TOP_K}\n")
        f.write(f"Top metric: {SORT_METRIC}\n\n")

        def fmt_q(name, q):
            f.write(f"{name} quartiles: Q1={q[0]:.4f}, Q2={q[1]:.4f}, Q3={q[2]:.4f}\n")

        fmt_q("lift", q_lift)
        fmt_q("cosine", q_cos)
        fmt_q("jaccard", q_jac)
        fmt_q("phi", q_phi)
        fmt_q("chi2", q_chi)
        fmt_q("pmi", q_pmi)
        fmt_q("npmi", q_npmi)
        fmt_q("hypergeom_z", q_zh)

    print(f"完成，结果已导出至：{out_dir}")
    print("生成文件：associations_tag_pairs.csv, top_associations_by_tag.csv, tag_frequency_summary.csv, assoc_meta.txt")

def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

# ================= 入口 =================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python tag_association_stats.py <path_to_json_or_dir> [out_dir]")
        sys.exit(1)
    path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) >= 3 else "tag_assoc_out"
    analyze(path, out_dir)
