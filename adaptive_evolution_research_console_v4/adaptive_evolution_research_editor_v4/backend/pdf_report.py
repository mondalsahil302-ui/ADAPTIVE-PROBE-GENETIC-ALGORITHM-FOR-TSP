"""
PDF Research Report Generator and Artifact Persistence Module.
Generates comprehensive ReportLab research reports with embedded matplotlib charts,
detailed metric tables, trajectory highlights, and execution logs.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


try:
    from .algorithm_core import best_known, gap_percent
except (ImportError, ValueError):
    from algorithm_core import best_known, gap_percent


def render_report_charts(st: Dict[str, Any], temp_dir: Path) -> Dict[str, Path]:
    """Generates high-resolution publication-quality research charts for the PDF report."""
    charts: Dict[str, Path] = {}
    records = st.get("records", [])
    history = st.get("history", [])
    inst = st["instance"]
    opt = st.get("config") and getattr(st["instance"], "name", None)

    # Optimum lookup
    optimum = best_known(inst.name) if hasattr(inst, "name") else None

    gens = [r["generation"] for r in records]
    cutoff_gen = st.get("generation", 0)

    # 1. Convergence Chart
    conv_path = temp_dir / "chart_convergence.png"
    plt.figure(figsize=(7.5, 3.2), dpi=200)
    plt.plot(range(len(history)), history, color="#1e40af", linewidth=2.0, label="Global Best Tour Cost")
    if optimum is not None and not np.isnan(optimum):
        plt.axhline(y=float(optimum), color="#dc2626", linestyle="--", linewidth=1.5, label=f"Known Optimum ({optimum:,.0f})")
    plt.axvline(x=cutoff_gen, color="#6b7280", linestyle=":", linewidth=1.5, label=f"Termination (Gen {cutoff_gen})")
    plt.title(f"APGGS Convergence Trajectory — {inst.name.upper()}", fontsize=11, fontweight="bold", pad=8)
    plt.xlabel("Generation", fontsize=9)
    plt.ylabel("Tour Cost", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(conv_path, bbox_inches="tight")
    plt.close()
    charts["convergence"] = conv_path

    # 2. Stagnation Patience Chart
    stag_path = temp_dir / "chart_stagnation.png"
    no_imp_counts = [r.get("no_improvement_count", 0) for r in records]
    patience_limit = int(st["config"].no_improvement_limit)
    plt.figure(figsize=(7.5, 2.8), dpi=200)
    plt.plot(gens, no_imp_counts, color="#d97706", linewidth=1.8, label="Consecutive Non-Improving Gens")
    plt.axhline(y=patience_limit, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Stagnation Limit ({patience_limit})")
    plt.title(f"Stagnation Patience Trajectory (Threshold: {patience_limit})", fontsize=11, fontweight="bold", pad=8)
    plt.xlabel("Generation", fontsize=9)
    plt.ylabel("Consecutive Generations", fontsize=9)
    plt.ylim(bottom=0, top=max(patience_limit + 5, max(no_imp_counts, default=0) + 2))
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="upper left", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(stag_path, bbox_inches="tight")
    plt.close()
    charts["stagnation"] = stag_path

    # 3. Improvement Chart
    imp_path = temp_dir / "chart_improvement.png"
    improvements = [r.get("improvement", 0.0) for r in records]
    plt.figure(figsize=(7.5, 2.6), dpi=200)
    plt.bar(gens, improvements, color="#059669", width=0.8, alpha=0.85, label="Global Best Improvement")
    plt.title("Per-Generation Solution Improvement Delta", fontsize=11, fontweight="bold", pad=8)
    plt.xlabel("Generation", fontsize=9)
    plt.ylabel("Cost Delta", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(imp_path, bbox_inches="tight")
    plt.close()
    charts["improvement"] = imp_path

    # 4. Execution Time Chart
    time_path = temp_dir / "chart_runtime.png"
    exec_times = [r.get("execution_time", 0.0) for r in records]
    cum_times = np.cumsum(exec_times)
    fig, ax1 = plt.subplots(figsize=(7.5, 2.8), dpi=200)
    ax1.plot(gens, exec_times, color="#6366f1", linewidth=1.4, alpha=0.7, label="Gen Duration (s)")
    ax1.set_xlabel("Generation", fontsize=9)
    ax1.set_ylabel("Generation Duration (s)", color="#6366f1", fontsize=9)
    ax1.tick_params(axis="y", labelcolor="#6366f1")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(gens, cum_times, color="#0f766e", linewidth=1.8, linestyle="--", label="Cumulative Time (s)")
    ax2.set_ylabel("Cumulative Time (s)", color="#0f766e", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#0f766e")

    plt.title("Generation Execution Time & Cumulative Progression", fontsize=11, fontweight="bold", pad=8)
    plt.tight_layout()
    plt.savefig(time_path, bbox_inches="tight")
    plt.close()
    charts["runtime"] = time_path

    # 5. Diversity & PMG Trust Chart
    div_path = temp_dir / "chart_diversity.png"
    divs = [r.get("diversity") if r.get("diversity") is not None else 0.0 for r in records]
    trusts = [r.get("memory_trust", 0.0) for r in records]
    plt.figure(figsize=(7.5, 2.8), dpi=200)
    plt.plot(gens, divs, color="#8b5cf6", linewidth=1.6, label="Population Diversity")
    plt.plot(gens, trusts, color="#ec4899", linewidth=1.6, linestyle="-.", label="PMG Memory Trust")
    plt.title("Population Diversity & Predictive Memory Graph Trust", fontsize=11, fontweight="bold", pad=8)
    plt.xlabel("Generation", fontsize=9)
    plt.ylabel("Index / Ratio", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="best", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(div_path, bbox_inches="tight")
    plt.close()
    charts["diversity"] = div_path

    return charts


def generate_pdf_report(
    st: Dict[str, Any],
    run_id: str,
    output_pdf_path: Path,
    app_version: str = "4.2",
) -> Path:
    """
    Creates a full ReportLab research PDF report document based on the
    persisted state of an Adaptive Evolution experiment.
    """
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    primary_color = colors.HexColor("#0f172a")
    accent_blue = colors.HexColor("#1e40af")
    muted_color = colors.HexColor("#64748b")
    card_bg = colors.HexColor("#f8fafc")
    border_color = colors.HexColor("#cbd5e1")

    # Typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=muted_color,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=accent_blue,
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=primary_color,
    )
    code_style = ParagraphStyle(
        "CodeText",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#334155"),
    )

    story: List[Any] = []

    inst = st["instance"]
    cfg = st["config"]
    records = st.get("records", [])
    history = st.get("history", [])
    pipe = st.get("pipe", {})

    optimum = best_known(inst.name) if hasattr(inst, "name") else None
    final_cost = float(st["global_best_cost"])
    gap = gap_percent(final_cost, optimum)
    gap_str = f"{gap:+.4f}%" if gap is not None and not np.isnan(gap) else "Unavailable"
    opt_str = f"{optimum:,.0f}" if optimum is not None else "Unknown"

    total_time = sum(r.get("execution_time", 0.0) for r in records)
    total_time += pipe.get("greedy_time", 0.0) + pipe.get("two_opt_local_time", 0.0) + pipe.get("two_opt_refined_time", 0.0) + pipe.get("population_time", 0.0)

    # -------------------------------------------------------------
    # HEADER BANNER
    # -------------------------------------------------------------
    header_data = [
        [
            Paragraph("<b>ADAPTIVE EVOLUTION RESEARCH CONSOLE</b>", ParagraphStyle("Hdr1", fontName="Helvetica-Bold", fontSize=9, textColor=accent_blue)),
            Paragraph(f"<b>Run ID:</b> {run_id}", ParagraphStyle("Hdr2", fontName="Helvetica", fontSize=8.5, alignment=2, textColor=muted_color)),
        ],
        [
            Paragraph("Adaptive Probe-Guided Genetic Algorithm (APGGS) — Experiment Report", title_style),
            Paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}<br/>Engine Version: v{app_version}", subtitle_style),
        ],
    ]
    header_table = Table(header_data, colWidths=[4.2 * inch, 3.0 * inch])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent_blue, spaceAfter=12))

    # -------------------------------------------------------------
    # SECTION 1: EXPERIMENT OVERVIEW
    # -------------------------------------------------------------
    story.append(Paragraph("1. Executive Summary & Dataset Metadata", h1_style))
    summary_data = [
        ["Dataset Name", str(inst.name).upper(), "Cities / Nodes", str(inst.n)],
        ["Edge Weight Type", str(inst.edge_weight_type), "Known Best Optimum", opt_str],
        ["Final Global Best Cost", f"{final_cost:,.2f}", "Optimality Gap", gap_str],
        ["Best Found Generation", str(st.get("best_generation", 0)), "Completed Generations", str(st.get("generation", 0))],
        ["Stagnation Streak", f"{st.get('no_improvement', 0)} / {cfg.no_improvement_limit}", "Stopping Condition", f"{cfg.no_improvement_limit} gens without improvement"],
        ["Experiment Status", str(st.get("state", "completed")).upper(), "Total Execution Time", f"{total_time:.2f} s"],
        ["Stop Reason", str(st.get("stop_reason", "—")), "Random Seed", str(cfg.seed)],
    ]
    summary_table = Table(summary_data, colWidths=[1.8 * inch, 1.9 * inch, 1.8 * inch, 1.7 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), card_bg),
        ("GRID", (0, 0), (-1, -1), 0.5, border_color),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), muted_color),
        ("TEXTCOLOR", (2, 0), (2, -1), muted_color),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # SECTION 2: INITIAL CONSTRUCTION PIPELINE
    # -------------------------------------------------------------
    story.append(Paragraph("2. Initial High-Quality Route Construction", h1_style))
    init_data = [
        ["Construction Stage", "Resulting Tour Cost", "Improvement vs Prev", "Duration (s)"],
        [
            "Stage 1: Greedy Nearest-Neighbor Tour",
            f"{pipe.get('greedy_cost', 0.0):,.2f}",
            "Baseline",
            f"{pipe.get('greedy_time', 0.0):.4f}",
        ],
        [
            "Stage 2: 2-Opt Local Search Improvement",
            f"{pipe.get('two_opt_local_cost', 0.0):,.2f}",
            f"-{max(0.0, pipe.get('greedy_cost', 0.0) - pipe.get('two_opt_local_cost', 0.0)):,.2f}",
            f"{pipe.get('two_opt_local_time', 0.0):.4f}",
        ],
        [
            "Stage 3: 2-Opt Deep Refinement Seed",
            f"{pipe.get('two_opt_refined_cost', 0.0):,.2f}",
            f"-{max(0.0, pipe.get('two_opt_local_cost', 0.0) - pipe.get('two_opt_refined_cost', 0.0)):,.2f}",
            f"{pipe.get('two_opt_refined_time', 0.0):.4f}",
        ],
        [
            "Stage 4: Initial 70-Individual Population (NN+Greedy+Random)",
            f"{records[0]['best_cost']:,.2f}" if records else "—",
            "Population Seeded",
            f"{pipe.get('population_time', 0.0):.4f}",
        ],
    ]
    init_table = Table(init_data, colWidths=[2.8 * inch, 1.6 * inch, 1.5 * inch, 1.3 * inch])
    init_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, border_color),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(init_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------
    # SECTION 3: ALGORITHM CONFIGURATION
    # -------------------------------------------------------------
    story.append(Paragraph("3. Genetic Algorithm & Adaptive Pipeline Configuration", h1_style))
    config_data = [
        ["Parameter", "Configured Value", "Parameter", "Configured Value"],
        ["Population Size", str(cfg.population_size), "Mutation Rate", f"{cfg.mutation_rate:.2f}"],
        ["Nearest-Neighbor Seeds", str(cfg.nn_count), "Swap Fraction", f"{cfg.swap_fraction:.2f}"],
        ["Greedy Edge Seeds", str(cfg.greedy_count), "Segment Crossover Prob", f"{cfg.segment_crossover_probability:.2f}"],
        ["Random Tour Seeds", str(cfg.random_count), "Offspring 2-Opt Prob", f"{cfg.offspring_local_search_probability:.2f}"],
        ["K-Nearest Neighbors", str(cfg.k_neighbors), "Adaptive Elites Count", str(cfg.adaptive_elite_count)],
        ["Tournament Size", str(cfg.tournament_size), "PMG Memory Decay", f"{cfg.memory_decay:.2f}"],
        ["Selective LKH Cadence", f"Every {cfg.lkh_cadence} gens", "Stagnation Window", str(cfg.stagnation_window)],
        ["Selective LKH Individuals", str(cfg.lkh_individuals), "Stagnation Patience Limit", f"{cfg.no_improvement_limit} consecutive gens"],
    ]
    config_table = Table(config_data, colWidths=[1.8 * inch, 1.8 * inch, 1.8 * inch, 1.8 * inch])
    config_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, border_color),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("PADDING", (0, 0), (-1, -1), 3.5),
    ]))
    story.append(config_table)

    # -------------------------------------------------------------
    # SECTION 4 & 5: RESEARCH CHARTS
    # -------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("4. Research Visualizations & Trajectory Curves", h1_style))

    # Render charts to temporary files and build doc while images exist
    with tempfile.TemporaryDirectory() as chart_tmp:
        chart_paths = render_report_charts(st, Path(chart_tmp))

        story.append(Paragraph("<b>Convergence Trajectory: Global Best Cost vs Generation</b>", ParagraphStyle("C1", fontName="Helvetica-Bold", fontSize=9, textColor=primary_color)))
        story.append(Image(str(chart_paths["convergence"]), width=7.2 * inch, height=3.0 * inch))
        story.append(Spacer(1, 8))

        story.append(Paragraph("<b>Stagnation Counter Progression (Target: 30 Consecutive Non-Improving Gens)</b>", ParagraphStyle("C2", fontName="Helvetica-Bold", fontSize=9, textColor=primary_color)))
        story.append(Image(str(chart_paths["stagnation"]), width=7.2 * inch, height=2.6 * inch))
        story.append(Spacer(1, 8))

        story.append(PageBreak())
        story.append(Paragraph("<b>Per-Generation Cost Improvement & Runtime Progression</b>", h1_style))

        story.append(Image(str(chart_paths["improvement"]), width=7.2 * inch, height=2.5 * inch))
        story.append(Spacer(1, 8))

        story.append(Image(str(chart_paths["runtime"]), width=7.2 * inch, height=2.7 * inch))
        story.append(Spacer(1, 8))

        story.append(Image(str(chart_paths["diversity"]), width=7.2 * inch, height=2.7 * inch))
        story.append(Spacer(1, 10))

        # -------------------------------------------------------------
        # SECTION 6: GENERATION HIGHLIGHTS TABLE
        # -------------------------------------------------------------
        story.append(PageBreak())
        story.append(Paragraph("5. Evolution Highlights (Global-Best Discoveries)", h1_style))

        improvements = st.get("improvements", [])
        if not improvements and records:
            improvements = [records[0]]

        hl_data = [
            ["Gen", "Global Best Cost", "Cost Reduction", "Gap (%)", "Operator / Method", "PMG Trust", "Non-Imp Streak", "Duration (s)"]
        ]
        for r in improvements[:25]:
            hl_data.append([
                str(r["generation"]),
                f"{r['best_cost']:,.2f}",
                f"{r.get('improvement', 0.0):,.2f}",
                f"{r['gap']:+.4f}%" if r.get("gap") is not None else "—",
                str(r.get("best_method", r.get("operator", "—"))),
                f"{r.get('memory_trust', 0.0):.2f}",
                str(r.get("no_improvement_count", 0)),
                f"{r.get('execution_time', 0.0):.4f}",
            ])
        if len(improvements) > 25:
            hl_data.append(["...", "...", "...", "...", f"[{len(improvements) - 25} more records]", "...", "...", "..."])

        # Ensure the final terminating generation is listed
        if records and records[-1]["generation"] != improvements[-1]["generation"]:
            final_rec = records[-1]
            hl_data.append([
                f"End: {final_rec['generation']}",
                f"{final_rec['best_cost']:,.2f}",
                "0.00",
                f"{final_rec['gap']:+.4f}%" if final_rec.get("gap") is not None else "—",
                f"Terminated ({final_rec.get('stop_reason', '—')})",
                f"{final_rec.get('memory_trust', 0.0):.2f}",
                str(final_rec.get("no_improvement_count", 0)),
                f"{final_rec.get('execution_time', 0.0):.4f}",
            ])

        hl_table = Table(hl_data, colWidths=[0.6 * inch, 1.2 * inch, 1.0 * inch, 0.9 * inch, 1.8 * inch, 0.7 * inch, 0.8 * inch, 0.8 * inch])
        hl_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, border_color),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("PADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(hl_table)
        story.append(Spacer(1, 12))

        # -------------------------------------------------------------
        # SECTION 7: FINAL RESULT & ROUTE SEQUENCE
        # -------------------------------------------------------------
        story.append(Paragraph("6. Final Discovered Tour & Solution Vector", h1_style))
        best_tour = st.get("global_best_tour", [])
        tour_preview = " → ".join(map(str, best_tour[:40]))
        if len(best_tour) > 40:
            tour_preview += f" → ... [{len(best_tour) - 40} additional cities] → {best_tour[0]}"

        story.append(Paragraph(
            f"<b>Optimized Tour Length:</b> {final_cost:,.2f} | <b>Optimality Gap:</b> {gap_str} | "
            f"<b>Discovered at Generation:</b> {st.get('best_generation', 0)}",
            body_style,
        ))
        story.append(Spacer(1, 4))
        story.append(Paragraph("<b>Permutation Sequence Preview:</b>", body_style))
        story.append(Paragraph(tour_preview, code_style))
        story.append(Spacer(1, 10))

        # -------------------------------------------------------------
        # SECTION 8: EXECUTION DIAGNOSTICS LOG
        # -------------------------------------------------------------
        story.append(Paragraph("7. Execution Event Log & Diagnostics", h1_style))
        diagnostics = st.get("diagnostics", [])
        log_text = "<br/>".join([f"• {msg}" for msg in diagnostics[-25:]]) if diagnostics else "No diagnostic warnings recorded."
        story.append(Paragraph(log_text, code_style))
        story.append(Spacer(1, 10))

        # -------------------------------------------------------------
        # SECTION 9: REPRODUCIBILITY MANIFEST
        # -------------------------------------------------------------
        story.append(Paragraph("8. Artifact Manifest & Research Reproducibility", h1_style))
        story.append(Paragraph(
            f"All raw experimental records have been saved to <code>runs/{run_id}/</code> including "
            "<code>summary.json</code>, <code>generation_history.jsonl</code>, <code>best_route.csv</code>, "
            "and <code>report.pdf</code>.",
            body_style,
        ))

        # Build PDF while chart images are available on disk
        doc.build(story)

    return output_pdf_path


def persist_run_artifacts(st: Dict[str, Any], run_id: Optional[str] = None, runs_base_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Saves complete experiment outputs to runs/<run_id>/ directory:
    - summary.json
    - generation_history.jsonl
    - best_route.csv
    - report.pdf
    - manifest.json
    """
    if not run_id:
        run_id = f"run_{int(time.time())}"

    if runs_base_dir is not None:
        runs_dir = Path(runs_base_dir) / run_id
    else:
        root_dir = Path(__file__).resolve().parents[1]
        runs_dir = root_dir / "runs" / run_id
    runs_dir.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, Path] = {}

    inst = st["instance"]
    cfg = st["config"]
    records = st.get("records", [])

    optimum = best_known(inst.name) if hasattr(inst, "name") else None
    final_cost = float(st["global_best_cost"])

    # 1. summary.json
    summary = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "name": inst.name,
            "cities": inst.n,
            "edge_weight_type": inst.edge_weight_type,
            "best_known": optimum,
        },
        "config": vars(cfg),
        "results": {
            "best_cost": final_cost,
            "gap_percent": gap_percent(final_cost, optimum),
            "best_generation": st.get("best_generation", 0),
            "completed_generations": st.get("generation", 0),
            "no_improvement_streak": st.get("no_improvement", 0),
            "stagnation_patience": cfg.no_improvement_limit,
            "converged": bool(st.get("converged", False)),
            "is_stopped": bool(st.get("is_stopped", False)),
            "state": st.get("state", "completed"),
            "stop_reason": st.get("stop_reason", ""),
            "total_execution_time": sum(r.get("execution_time", 0.0) for r in records),
        },
        "diagnostics": st.get("diagnostics", []),
    }
    summary_path = runs_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    artifacts["summary"] = summary_path

    # 2. generation_history.jsonl
    history_path = runs_dir / "generation_history.jsonl"
    with open(history_path, "w", encoding="utf-8") as f:
        for r in records:
            clean_r = {k: v for k, v in r.items() if k != "route"}
            f.write(json.dumps(clean_r) + "\n")
    artifacts["history"] = history_path

    # 3. best_route.csv
    route_path = runs_dir / "best_route.csv"
    best_tour = st.get("global_best_tour", [])
    coords = inst.coords
    with open(route_path, "w", encoding="utf-8") as f:
        if coords is not None:
            f.write("order,city_index,x,y\n")
            for order, city in enumerate(best_tour):
                f.write(f"{order},{city},{coords[city, 0]:.6f},{coords[city, 1]:.6f}\n")
        else:
            f.write("order,city_index\n")
            for order, city in enumerate(best_tour):
                f.write(f"{order},{city}\n")
    artifacts["route"] = route_path

    # 4. report.pdf
    pdf_path = runs_dir / "report.pdf"
    try:
        generate_pdf_report(st, run_id, pdf_path)
        artifacts["report"] = pdf_path
    except Exception as exc:
        print(f"Warning: PDF report generation failed: {exc}")

    # 5. manifest.json
    manifest = {
        "run_id": run_id,
        "created_at": time.time(),
        "files": {name: str(p.relative_to(runs_dir)) for name, p in artifacts.items()},
    }
    manifest_path = runs_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    artifacts["manifest"] = manifest_path

    return artifacts


def create_run_zip(run_dir: Path, zip_output_path: Path) -> Path:
    """Compresses an experiment's artifacts directory into a single ZIP file."""
    zip_output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in run_dir.rglob("*"):
            if f.is_file() and not f.name.endswith(".zip"):
                zf.write(f, arcname=f.relative_to(run_dir))
    return zip_output_path
