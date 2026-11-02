"""公開用の数値だけで再生成できる評価図。API と DB は使わない。"""

from pathlib import Path
from statistics import mean

from jp_doc_agent.evaluation.comparison import MODES, public_report

LABELS = {"no_rag": "No RAG", "single_rag": "Single-pass RAG", "agent": "Agent RAG"}
COLORS = {"no_rag": "#75859B", "single_rag": "#167D9A", "agent": "#DB783B"}


def plot_report(report: dict, output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError:
        raise ValueError(
            "図の生成には uv run --group eval doc-agent plot-evaluation を使ってください。"
        ) from None
    if report["trials"] and "result" in report["trials"][0]:
        report = public_report(report)
    if not report.get("summary") or not report.get("retrieval_curves"):
        raise ValueError("完了した対照評価レポートが必要です。")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#344054",
            "text.color": "#162B43",
            "axes.titleweight": "bold",
            "svg.hashsalt": "doc-agent-evaluation-v1",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]["modes"]
    repeats = report["configuration"]["repeats"]

    def range_by_repeat(mode, metric):
        values = []
        for repeat in range(repeats):
            trials = [
                x
                for x in report["trials"]
                if x["mode"] == mode
                and x["repeat"] == repeat
                and x["expected_status"] == "answered"
            ]
            values.append(mean((x["metrics"] or {}).get(metric, 0.0) for x in trials))
        value = summary[mode][metric]
        return [[max(0, value - min(values))], [max(0, max(values) - value)]]

    def percentage_axis(ax):
        ax.set_ylim(0, 1.14)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.grid(axis="y", color="#E4E9EF", zorder=0)
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.3))
    fig.suptitle(
        "RAG: quality, grounding, and generation cost", fontsize=20, fontweight="bold", y=0.99
    )
    fig.text(
        0.5,
        0.945,
        f"Same model | 10 fixed questions | {repeats} repeated trials per method",
        ha="center",
        color="#52637A",
    )
    ax = axes[0, 0]
    for i, mode in enumerate(MODES):
        value = summary[mode]["fact_f1"]
        variation = range_by_repeat(mode, "fact_f1")
        ax.bar(
            i,
            value,
            width=0.58,
            color=COLORS[mode],
            zorder=3,
            yerr=variation,
            capsize=5,
            error_kw={"linewidth": 1.3},
        )
        ax.text(
            i,
            min(1.08, value + variation[1][0] + 0.04),
            f"{value:.1%}",
            ha="center",
            fontweight="bold",
        )
    ax.set_xticks(range(3), [LABELS[m] for m in MODES])
    ax.set_title("A. Reference-guided Fact F1", loc="left")
    ax.set_ylabel("Mean over 8 answerable questions")
    percentage_axis(ax)

    ax = axes[0, 1]
    for i, mode in enumerate(MODES):
        p, r = summary[mode]["fact_precision"], summary[mode]["fact_recall"]
        ax.bar(i - 0.16, p, width=0.3, color=COLORS[mode], zorder=3)
        ax.bar(i + 0.16, r, width=0.3, color=COLORS[mode], alpha=0.44, zorder=3)
        ax.text(i - 0.16, p + 0.025, f"{p:.0%}", ha="center", fontsize=9)
        ax.text(i + 0.16, r + 0.025, f"{r:.0%}", ha="center", fontsize=9)
    ax.set_xticks(range(3), [LABELS[m] for m in MODES])
    ax.set_title("B. Factual precision / reference-fact recall", loc="left")
    ax.plot([], [], color="#52637A", linewidth=9, label="Precision (solid)")
    ax.plot([], [], color="#B2BDCA", linewidth=9, label="Recall (light)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    percentage_axis(ax)

    ax = axes[1, 0]
    metrics = ("faithfulness", "citation_support")
    for i, mode in enumerate(("single_rag", "agent")):
        for j, metric in enumerate(metrics):
            value = summary[mode][metric]
            if value is None:
                continue
            x = i + (j - 0.5) * 0.32
            ax.bar(x, value, width=0.3, color=COLORS[mode], alpha=1 if j == 0 else 0.44, zorder=3)
            ax.text(x, value + 0.025, f"{value:.0%}", ha="center", fontsize=10)
    ax.set_xticks(
        [0, 1],
        [
            f"{LABELS[m]}\n(n={summary[m]['grounding_scored_answers']})"
            for m in ("single_rag", "agent")
        ],
    )
    ax.set_title("C. Context / citation support (LLM judge)", loc="left")
    ax.set_ylabel("Valid scored outputs; No RAG is N/A")
    ax.plot([], [], color="#52637A", linewidth=9, label="Context support (solid)")
    ax.plot([], [], color="#B2BDCA", linewidth=9, label="Citation support (light)")
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
    percentage_axis(ax)
    ax.set_ylim(0, 1.3)

    ax = axes[1, 1]
    for i, mode in enumerate(MODES):
        x = summary[mode]["mean_model_tokens"]
        y = summary[mode]["mean_latency_seconds"]
        ax.scatter(x, y, s=155, color=COLORS[mode], edgecolors="white", linewidths=1.7, zorder=4)
        ax.annotate(
            LABELS[mode],
            (x, y),
            xytext=(8, 8 if i < 2 else -16),
            textcoords="offset points",
            fontsize=10,
        )
    max_x = max(summary[m]["mean_model_tokens"] for m in MODES)
    max_y = max(summary[m]["mean_latency_seconds"] for m in MODES)
    ax.set_xlim(-max_x * 0.08, max_x * 1.38)
    ax.set_ylim(0, max_y * 1.25)
    ax.set_xlabel("Mean generation tokens per question (input + output)")
    ax.set_ylabel("Mean generation latency (seconds)")
    ax.set_title("D. Generation latency and token use", loc="left")
    ax.grid(color="#E4E9EF", zorder=0)
    fig.text(
        0.03,
        0.015,
        "Fixed regression set; reference-guided LLM grading, not an independent human evaluation.\n"
        f"Error bars: min/max of {repeats} repeat means. "
        "Judge time and tokens excluded from generation cost.",
        fontsize=9,
        color="#52637A",
    )
    fig.tight_layout(rect=[0, 0.055, 1, 0.925], h_pad=3, w_pad=2.5)
    _save(fig, output_dir / "rag-comparison")
    plt.close(fig)

    curves = report["retrieval_curves"]
    points = []
    for k in range(1, 11):
        entries = [entry for curve in curves for entry in curve["points"] if entry["k"] == k]
        points.append(
            {
                "k": k,
                **{
                    key: mean(x[key] for x in entries)
                    for key in ("hit_rate", "mrr", "ndcg", "precision", "recall")
                },
            }
        )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    fig.suptitle("Retrieval curves: top-k sensitivity", fontsize=19, fontweight="bold", y=0.99)
    colors = ["#167D9A", "#DB783B", "#7357A2"]
    for ax, metrics in zip(
        axes, [("hit_rate", "mrr", "ndcg"), ("recall", "precision")], strict=True
    ):
        for color, metric in zip(colors, metrics, strict=False):
            label = {
                "hit_rate": "Hit@k",
                "mrr": "MRR@k",
                "ndcg": "nDCG@k",
                "recall": "Recall@k",
                "precision": "Precision@k",
            }[metric]
            ax.plot(
                [x["k"] for x in points],
                [x[metric] for x in points],
                marker="o",
                markersize=4.5,
                linewidth=2.1,
                color=color,
                label=label,
            )
        ax.axvline(5, color="#8B98AA", linestyle="--", linewidth=1, label="Generation uses k=5")
        ax.set_xlabel("Retrieved chunks (k)")
        ax.set_xticks(range(1, 11))
        ax.set_ylim(-0.02, 1.08)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(color="#E4E9EF")
        ax.legend(frameon=False, fontsize=9, loc="best")
    axes[0].set_title("A. Finding and ranking evidence", loc="left")
    axes[1].set_title("B. Coverage / noise tradeoff", loc="left")
    fig.text(
        0.03,
        0.015,
        f"{len(curves)} answerable questions; binary chunk relevance "
        "from annotated physical PDF pages.\n"
        "Page overlap is a relevance proxy, not a claim-level semantic relevance judgment.",
        fontsize=9,
        color="#52637A",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.92], w_pad=3)
    _save(fig, output_dir / "retrieval-curves")
    plt.close(fig)


def _save(figure, stem):
    figure.savefig(
        stem.with_suffix(".png"), dpi=190, facecolor="white", metadata={"Software": "Doc-Agent"}
    )
    figure.savefig(
        stem.with_suffix(".svg"), facecolor="white", metadata={"Creator": "Doc-Agent", "Date": None}
    )
    svg = stem.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
