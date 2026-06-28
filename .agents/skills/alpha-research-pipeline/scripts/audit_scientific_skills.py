#!/usr/bin/env python3
"""Audit K-Dense scientific-agent-skills for LuminaQuant alpha-research relevance."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten
from typing import Any

UPSTREAM_COMMIT = "9c9bd2e92af12311ecd0c1a643e0931643f9ea04"

# Tier semantics:
# A: default alpha-research pipeline building block.
# B: conditional alpha method/tool when a matching research lane is selected.
# C: support skill for ingestion, reporting, infrastructure, or research management.
# D: reviewed but excluded by default; only use for explicit alternative data/domain transfer.
CLASSIFICATION: dict[str, tuple[str, str, str]] = {
    # Core research loop / method control
    "arbor": (
        "A",
        "optimization-loop",
        "Run hypothesis-tree refinement over concrete alpha artifacts with dev/test split discipline.",
    ),
    "hypothesis-generation": (
        "A",
        "ideation",
        "Turn observations into falsifiable alpha hypotheses, mechanisms, predictions, and disconfirming evidence.",
    ),
    "scientific-brainstorming": (
        "A",
        "ideation",
        "Generate diverse cross-domain alpha families before implementation.",
    ),
    "scientific-critical-thinking": (
        "A",
        "bias-control",
        "Attack leakage, p-hacking, confounding, survivorship, cost assumptions, and unsupported claims.",
    ),
    "experimental-design": (
        "A",
        "experiment-design",
        "Pre-register windows, embargoes, candidate budgets, controls, and DOE/ablation plans.",
    ),
    "statistical-analysis": (
        "A",
        "validation",
        "Choose tests, effect sizes, intervals, robustness checks, and multiple-comparison handling.",
    ),
    "hypogenic": (
        "A",
        "data-driven-hypotheses",
        "Generate/refine hypothesis banks from tabular market feature panels; exploration only, never proof.",
    ),
    # Literature / web / external data
    "paper-lookup": (
        "A",
        "literature",
        "Search academic paper APIs for alpha seeds and empirical finance/microstructure evidence.",
    ),
    "research-lookup": (
        "A",
        "literature",
        "Find current research, papers, and technical sources for alpha idea sourcing.",
    ),
    "literature-review": (
        "A",
        "literature",
        "Synthesize alpha-family evidence and gaps across sources before coding.",
    ),
    "database-lookup": (
        "A",
        "external-data",
        "Deterministic lookup across finance/economic endpoints such as FRED, SEC, ECB, BEA, BLS, Treasury.",
    ),
    "usfiscaldata": (
        "A",
        "macro-data",
        "Treasury/fiscal data for macro liquidity, rate, auction, and TGA factor research.",
    ),
    "parallel-web": (
        "B",
        "web-research",
        "Deep web/source extraction when local paper/database lookup is insufficient.",
    ),
    "exa-search": (
        "B",
        "web-research",
        "Semantic web retrieval for technical alpha sources and source extraction.",
    ),
    "bgpt-paper-search": (
        "B",
        "literature",
        "Structured paper extraction where available; useful for evidence fields, not required.",
    ),
    "citation-management": (
        "C",
        "provenance",
        "Citation quality, BibTeX, and source validation for literature-derived alphas.",
    ),
    "pyzotero": (
        "C",
        "provenance",
        "Zotero library integration for research provenance if the project uses Zotero.",
    ),
    "paperzilla": (
        "C",
        "research-management",
        "Project/paper recommendation management, optional.",
    ),
    "open-notebook": (
        "C",
        "research-management",
        "Organize paper/report corpora into notebooks; optional RAG workspace.",
    ),
    # Time series / statistical / ML analysis
    "aeon": (
        "A",
        "time-series-ml",
        "Time-series classification, anomaly detection, segmentation, motifs, similarity search, and temporal features.",
    ),
    "statsmodels": (
        "A",
        "econometrics",
        "OLS/GLM/ARIMA/VAR, cointegration, Granger-style tests, robust errors, diagnostics.",
    ),
    "scikit-learn": (
        "A",
        "ml",
        "Leakage-safe tabular ML baselines, selectors, clustering, anomaly detectors, and TimeSeriesSplit pipelines.",
    ),
    "shap": (
        "A",
        "explainability",
        "Feature attribution sanity checks and leakage-feature detection for ML alpha models.",
    ),
    "pymoo": (
        "A",
        "multi-objective-optimization",
        "Pareto search over return, drawdown, turnover, gross, stability, and implementation risk.",
    ),
    "pymc": (
        "B",
        "bayesian",
        "Bayesian regimes, uncertainty, hierarchical alpha strength, posterior predictive checks.",
    ),
    "timesfm-forecasting": (
        "B",
        "forecasting",
        "Zero-shot forecast intervals/anomaly features; never standalone promotion evidence.",
    ),
    "umap-learn": (
        "B",
        "regime-diagnostics",
        "Regime visualization/clustering diagnostics; proof must come from backtests/statistics.",
    ),
    "scikit-survival": (
        "B",
        "hazard-modeling",
        "Signal decay, stop-out, breakout failure, and time-to-event studies.",
    ),
    "networkx": (
        "B",
        "graph-alpha",
        "Lead-lag, correlation, cluster, asset graph, and propagation hypotheses.",
    ),
    "torch-geometric": (
        "B",
        "graph-ml",
        "Graph neural network alpha research; high leakage/overfit risk, strict gate required.",
    ),
    "pytorch-lightning": (
        "B",
        "deep-learning",
        "Structured deep learning experiments; use only with pre-registered splits and audits.",
    ),
    "transformers": (
        "B",
        "nlp-alt-data",
        "Text/news/filing sentiment or representation experiments; strict source/time provenance required.",
    ),
    "stable-baselines3": (
        "B",
        "rl",
        "Execution/sizing/risk-control simulation after environment fidelity is proven.",
    ),
    "pufferlib": (
        "B",
        "rl",
        "High-throughput RL for execution/sizing environments; not first-line alpha discovery.",
    ),
    "simpy": (
        "B",
        "execution-simulation",
        "Discrete-event fill, queue, latency, and slippage simulations.",
    ),
    "sympy": (
        "C",
        "math",
        "Symbolic derivations for transforms, constraints, or report equations.",
    ),
    "matlab": (
        "C",
        "numerics",
        "Translate/validate MATLAB/Octave quant prototypes when inherited.",
    ),
    # Data / compute / artifacts
    "get-available-resources": (
        "A",
        "preflight",
        "Check CPU/GPU/RAM/disk before heavy alpha sweeps.",
    ),
    "polars": (
        "A",
        "dataframe",
        "Primary LuminaQuant dataframe engine: lazy scans, streaming, joins, group-by, parquet.",
    ),
    "dask": (
        "B",
        "distributed-data",
        "Larger-than-RAM or distributed research scans when Polars alone is insufficient.",
    ),
    "vaex": (
        "B",
        "out-of-core-data",
        "Optional out-of-core tabular exploration for very large datasets.",
    ),
    "zarr-python": (
        "B",
        "array-storage",
        "Chunked factor tensors or cloud-native arrays when parquet is not enough.",
    ),
    "optimize-for-gpu": (
        "B",
        "gpu",
        "GPU/CUDA/cuDF/cuML/Numba acceleration for heavy feature/backtest research.",
    ),
    "modal": (
        "C",
        "cloud-compute",
        "Optional cloud/GPU batch execution; avoid unless explicitly authorized.",
    ),
    "xlsx": (
        "C",
        "document-ingest",
        "Ingest spreadsheets, broker extracts, and research workbooks.",
    ),
    "pdf": ("C", "document-ingest", "Extract tables/text from papers and reports."),
    "docx": ("C", "document-ingest", "Extract/edit Word research notes or reports."),
    "pptx": ("C", "document-ingest", "Extract slide decks or create research presentations."),
    "markitdown": (
        "C",
        "document-ingest",
        "Convert files into Markdown for auditable research ingestion.",
    ),
    "liteparse": (
        "C",
        "document-ingest",
        "OCR/layout-aware local PDF/document parsing for RAG inputs.",
    ),
    "lamindb": (
        "C",
        "lineage",
        "Lineage/lakehouse inspiration; prefer LuminaQuant native SHA/source artifact tracking first.",
    ),
    "autoskill": (
        "C",
        "skill-evolution",
        "Derive new local skills from repeated alpha workflows after patterns stabilize.",
    ),
    "pi-agent": (
        "C",
        "agent-platform",
        "Build/run Pi harnesses only if that platform is explicitly used.",
    ),
    # Visualization / reporting / review
    "exploratory-data-analysis": (
        "A",
        "eda",
        "Profile datasets, missingness, distributions, outliers, and quality before alpha fitting.",
    ),
    "scientific-visualization": (
        "A",
        "visualization",
        "Publication-quality diagnostics for returns, drawdowns, regimes, robustness, and costs.",
    ),
    "matplotlib": ("C", "visualization", "Low-level plot control for reports."),
    "seaborn": (
        "C",
        "visualization",
        "Fast statistical plots for distributions, correlations, and diagnostics.",
    ),
    "markdown-mermaid-writing": (
        "C",
        "reporting",
        "Write audit-friendly reports and pipeline diagrams.",
    ),
    "scientific-writing": ("C", "reporting", "Convert validated evidence into prose reports."),
    "scientific-slides": (
        "C",
        "reporting",
        "Research presentation deck creation after evidence exists.",
    ),
    "scientific-schematics": (
        "C",
        "reporting",
        "System/pipeline schematics and conceptual diagrams.",
    ),
    "infographics": ("C", "reporting", "Executive visual summaries after evidence exists."),
    "venue-templates": (
        "C",
        "reporting",
        "Formatting templates for formal papers/posters if needed.",
    ),
    "latex-posters": ("C", "reporting", "Poster output only."),
    "pptx-posters": ("C", "reporting", "Poster output only."),
    "market-research-reports": (
        "C",
        "macro-context",
        "Heavy macro/industry context reports; not proof of alpha.",
    ),
    "peer-review": (
        "A",
        "review",
        "Structured final methodology/reproducibility review before any promotion claim.",
    ),
    "scholar-evaluation": ("B", "review", "Score imported scholarly claims and paper quality."),
    "statistical-power": (
        "B",
        "validation",
        "Minimum detectable effect/sample-size checks for sparse trades or short windows.",
    ),
    "what-if-oracle": ("B", "scenario", "Scenario stress exploration; speculative, not proof."),
    "consciousness-council": (
        "B",
        "multi-perspective-review",
        "Multi-perspective debate for stuck research directions; not evidence.",
    ),
    "dhdna-profiler": (
        "D",
        "excluded-cognitive",
        "Reviewed; cognitive-style profiling is not needed for alpha research.",
    ),
    # Conditional alternative-data science skills
    "geomaster": (
        "B",
        "geospatial-alt-data",
        "Satellite/weather/supply-chain/geospatial alternative-data alpha research.",
    ),
    "geopandas": (
        "B",
        "geospatial-alt-data",
        "Vector geospatial joins/features for alternative datasets.",
    ),
    "neurokit2": (
        "D",
        "excluded-biosignal",
        "Could inspire signal processing, but biomedical signals are not default LuminaQuant data.",
    ),
    "fluidsim": (
        "D",
        "excluded-simulation",
        "Physics CFD not default; only analogy/commodity supply-chain special cases.",
    ),
    "astropy": ("D", "excluded-astronomy", "Astronomy workflows not default alpha research."),
    # Biomedical/chemistry/etc. default exclusions retained below by fallback.
}

DEFAULT_EXCLUDE_REASONS = {
    "bio": "Reviewed; biomedical/genomics workflow is not a default LuminaQuant alpha source.",
    "chem": "Reviewed; chemistry/drug-discovery workflow is not a default LuminaQuant alpha source.",
    "clinical": "Reviewed; clinical/medical workflow is not a default LuminaQuant alpha source.",
    "lab": "Reviewed; lab automation/integration workflow is not a default LuminaQuant alpha source.",
    "quantum": "Reviewed; quantum computing workflow is not needed for current alpha discovery.",
    "materials": "Reviewed; materials science workflow is not default unless explicit commodity/materials alt-data research is requested.",
    "other": "Reviewed; no direct role in finding, validating, analyzing, or reporting LuminaQuant trading alpha by default.",
}

BIO_PAT = re.compile(
    r"bio|gene|genom|rna|cell|protein|sequence|pathway|variant|phylo|omics|neuro|clinical|health|medical|dicom|pathology|treatment",
    re.I,
)
CHEM_PAT = re.compile(
    r"chem|drug|molec|protein|rdkit|smiles|docking|medchem|compound|metabol|spectr", re.I
)
LAB_PAT = re.compile(r"benchling|lab|opentrons|protocol|ginkgo|latch|dnanexus|omero", re.I)
QUANTUM_PAT = re.compile(r"quantum|qiskit|cirq|qutip|pennylane", re.I)
MATERIAL_PAT = re.compile(r"material|crystal|pymatgen", re.I)
CLINICAL_PAT = re.compile(
    r"clinical|patient|medical|treatment|health|DICOM|radiology|pathology", re.I
)


@dataclass(frozen=True)
class SkillMeta:
    folder: str
    name: str
    description: str
    compatibility: str
    reference_count: int


def parse_frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(errors="ignore")
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    header = text[3:end]
    fields: dict[str, str] = {}
    current_key = ""
    for raw in header.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.startswith(" ") and current_key:
            fields[current_key] += " " + raw.strip().strip('"')
            continue
        if ":" in raw:
            key, value = raw.split(":", 1)
            current_key = key.strip()
            fields[current_key] = value.strip().strip('"').strip("'")
    return fields


def load_skills(skills_dir: Path) -> list[SkillMeta]:
    metas: list[SkillMeta] = []
    for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
        fields = parse_frontmatter(skill_md)
        refs = (
            list((skill_md.parent / "references").glob("*.md"))
            if (skill_md.parent / "references").exists()
            else []
        )
        metas.append(
            SkillMeta(
                folder=skill_md.parent.name,
                name=fields.get("name") or skill_md.parent.name,
                description=" ".join((fields.get("description") or "").split()),
                compatibility=" ".join((fields.get("compatibility") or "").split()),
                reference_count=len(refs),
            )
        )
    return metas


def classify(meta: SkillMeta) -> tuple[str, str, str]:
    if meta.folder in CLASSIFICATION:
        return CLASSIFICATION[meta.folder]
    blob = f"{meta.folder} {meta.name} {meta.description}"
    if QUANTUM_PAT.search(blob):
        return "D", "excluded-quantum", DEFAULT_EXCLUDE_REASONS["quantum"]
    if CLINICAL_PAT.search(blob):
        return "D", "excluded-clinical", DEFAULT_EXCLUDE_REASONS["clinical"]
    if LAB_PAT.search(blob):
        return "D", "excluded-lab", DEFAULT_EXCLUDE_REASONS["lab"]
    if MATERIAL_PAT.search(blob):
        return "D", "excluded-materials", DEFAULT_EXCLUDE_REASONS["materials"]
    if CHEM_PAT.search(blob):
        return "D", "excluded-chemistry", DEFAULT_EXCLUDE_REASONS["chem"]
    if BIO_PAT.search(blob):
        return "D", "excluded-biomedical", DEFAULT_EXCLUDE_REASONS["bio"]
    return "D", "excluded-other", DEFAULT_EXCLUDE_REASONS["other"]


def build_audit(skills_dir: Path) -> dict[str, Any]:
    metas = load_skills(skills_dir)
    rows = []
    counts: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}
    for meta in metas:
        tier, category, alpha_use = classify(meta)
        counts[tier] += 1
        rows.append(
            {
                "skill": meta.folder,
                "declared_name": meta.name,
                "tier": tier,
                "category": category,
                "alpha_use": alpha_use,
                "description": meta.description,
                "compatibility": meta.compatibility,
                "reference_count": meta.reference_count,
            }
        )
    return {
        "source": "https://github.com/k-dense-ai/scientific-agent-skills",
        "source_commit": UPSTREAM_COMMIT,
        "reviewed_skill_count": len(rows),
        "tier_counts": counts,
        "tier_legend": {
            "A": "absorbed into default LuminaQuant alpha-research pipeline",
            "B": "conditional alpha method/tool",
            "C": "supporting research/reporting/infrastructure skill",
            "D": "reviewed and excluded by default",
        },
        "skills": rows,
    }


def markdown(audit: dict[str, Any]) -> str:
    rows: list[dict[str, Any]] = audit["skills"]
    out = []
    out.append("# Scientific Agent Skills Audit for LuminaQuant Alpha Research")
    out.append("")
    out.append(f"Source: `{audit['source']}`")
    out.append(f"Source commit reviewed: `{audit['source_commit']}`")
    out.append(f"Reviewed skills: **{audit['reviewed_skill_count']}**")
    out.append("")
    out.append("## Tier Legend")
    for tier, label in audit["tier_legend"].items():
        out.append(f"- **{tier}** — {label}")
    out.append("")
    out.append("## Counts")
    for tier in ["A", "B", "C", "D"]:
        out.append(f"- Tier {tier}: {audit['tier_counts'][tier]}")
    out.append("")
    for tier in ["A", "B", "C", "D"]:
        out.append(f"## Tier {tier}")
        out.append("")
        out.append("| Skill | Category | LuminaQuant alpha use / exclusion reason |")
        out.append("|---|---|---|")
        for row in rows:
            if row["tier"] != tier:
                continue
            use = row["alpha_use"].replace("|", "\\|")
            out.append(f"| `{row['skill']}` | {row['category']} | {use} |")
        out.append("")
    out.append("## Complete 147-Skill Review Table")
    out.append("")
    out.append("| # | Skill | Tier | Category | Upstream description |")
    out.append("|---:|---|---|---|---|")
    for idx, row in enumerate(rows, 1):
        desc = shorten((row["description"] or "").replace("|", "\\|"), width=180, placeholder="…")
        out.append(f"| {idx} | `{row['skill']}` | {row['tier']} | {row['category']} | {desc} |")
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skills-dir", type=Path, default=Path("/tmp/scientific-agent-skills/skills")
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()
    audit = build_audit(args.skills_dir)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(markdown(audit), encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
