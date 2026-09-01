# Scientific Agent Skills Audit for LuminaQuant Alpha Research

Source: `https://github.com/k-dense-ai/scientific-agent-skills`
Source commit reviewed: `9c9bd2e92af12311ecd0c1a643e0931643f9ea04`
Reviewed skills: **147**

## Tier Legend
- **A** — absorbed into default LuminaQuant alpha-research pipeline
- **B** — conditional alpha method/tool
- **C** — supporting research/reporting/infrastructure skill
- **D** — reviewed and excluded by default

## Counts
- Tier A: 22
- Tier B: 24
- Tier C: 27
- Tier D: 74

## Tier A

| Skill | Category | LuminaQuant alpha use / exclusion reason |
|---|---|---|
| `aeon` | time-series-ml | Time-series classification, anomaly detection, segmentation, motifs, similarity search, and temporal features. |
| `arbor` | optimization-loop | Run hypothesis-tree refinement over concrete alpha artifacts with dev/test split discipline. |
| `database-lookup` | external-data | Deterministic lookup across finance/economic endpoints such as FRED, SEC, ECB, BEA, BLS, Treasury. |
| `experimental-design` | experiment-design | Pre-register windows, embargoes, candidate budgets, controls, and DOE/ablation plans. |
| `exploratory-data-analysis` | eda | Profile datasets, missingness, distributions, outliers, and quality before alpha fitting. |
| `get-available-resources` | preflight | Check CPU/GPU/RAM/disk before heavy alpha sweeps. |
| `hypogenic` | data-driven-hypotheses | Generate/refine hypothesis banks from tabular market feature panels; exploration only, never proof. |
| `hypothesis-generation` | ideation | Turn observations into falsifiable alpha hypotheses, mechanisms, predictions, and disconfirming evidence. |
| `literature-review` | literature | Synthesize alpha-family evidence and gaps across sources before coding. |
| `paper-lookup` | literature | Search academic paper APIs for alpha seeds and empirical finance/microstructure evidence. |
| `peer-review` | review | Structured final methodology/reproducibility review before any promotion claim. |
| `polars` | dataframe | Primary LuminaQuant dataframe engine: lazy scans, streaming, joins, group-by, parquet. |
| `pymoo` | multi-objective-optimization | Pareto search over return, drawdown, turnover, gross, stability, and implementation risk. |
| `research-lookup` | literature | Find current research, papers, and technical sources for alpha idea sourcing. |
| `scientific-brainstorming` | ideation | Generate diverse cross-domain alpha families before implementation. |
| `scientific-critical-thinking` | bias-control | Attack leakage, p-hacking, confounding, survivorship, cost assumptions, and unsupported claims. |
| `scientific-visualization` | visualization | Publication-quality diagnostics for returns, drawdowns, regimes, robustness, and costs. |
| `scikit-learn` | ml | Leakage-safe tabular ML baselines, selectors, clustering, anomaly detectors, and TimeSeriesSplit pipelines. |
| `shap` | explainability | Feature attribution sanity checks and leakage-feature detection for ML alpha models. |
| `statistical-analysis` | validation | Choose tests, effect sizes, intervals, robustness checks, and multiple-comparison handling. |
| `statsmodels` | econometrics | OLS/GLM/ARIMA/VAR, cointegration, Granger-style tests, robust errors, diagnostics. |
| `usfiscaldata` | macro-data | Treasury/fiscal data for macro liquidity, rate, auction, and TGA factor research. |

## Tier B

| Skill | Category | LuminaQuant alpha use / exclusion reason |
|---|---|---|
| `bgpt-paper-search` | literature | Structured paper extraction where available; useful for evidence fields, not required. |
| `consciousness-council` | multi-perspective-review | Multi-perspective debate for stuck research directions; not evidence. |
| `dask` | distributed-data | Larger-than-RAM or distributed research scans when Polars alone is insufficient. |
| `exa-search` | web-research | Semantic web retrieval for technical alpha sources and source extraction. |
| `geomaster` | geospatial-alt-data | Satellite/weather/supply-chain/geospatial alternative-data alpha research. |
| `geopandas` | geospatial-alt-data | Vector geospatial joins/features for alternative datasets. |
| `networkx` | graph-alpha | Lead-lag, correlation, cluster, asset graph, and propagation hypotheses. |
| `optimize-for-gpu` | gpu | GPU/CUDA/cuDF/cuML/Numba acceleration for heavy feature/backtest research. |
| `parallel-web` | web-research | Deep web/source extraction when local paper/database lookup is insufficient. |
| `pufferlib` | rl | High-throughput RL for execution/sizing environments; not first-line alpha discovery. |
| `pymc` | bayesian | Bayesian regimes, uncertainty, hierarchical alpha strength, posterior predictive checks. |
| `pytorch-lightning` | deep-learning | Structured deep learning experiments; use only with pre-registered splits and audits. |
| `scholar-evaluation` | review | Score imported scholarly claims and paper quality. |
| `scikit-survival` | hazard-modeling | Signal decay, stop-out, breakout failure, and time-to-event studies. |
| `simpy` | execution-simulation | Discrete-event fill, queue, latency, and slippage simulations. |
| `stable-baselines3` | rl | Execution/sizing/risk-control simulation after environment fidelity is proven. |
| `statistical-power` | validation | Minimum detectable effect/sample-size checks for sparse trades or short windows. |
| `timesfm-forecasting` | forecasting | Zero-shot forecast intervals/anomaly features; never standalone promotion evidence. |
| `torch-geometric` | graph-ml | Graph neural network alpha research; high leakage/overfit risk, strict gate required. |
| `transformers` | nlp-alt-data | Text/news/filing sentiment or representation experiments; strict source/time provenance required. |
| `umap-learn` | regime-diagnostics | Regime visualization/clustering diagnostics; proof must come from backtests/statistics. |
| `vaex` | out-of-core-data | Optional out-of-core tabular exploration for very large datasets. |
| `what-if-oracle` | scenario | Scenario stress exploration; speculative, not proof. |
| `zarr-python` | array-storage | Chunked factor tensors or cloud-native arrays when parquet is not enough. |

## Tier C

| Skill | Category | LuminaQuant alpha use / exclusion reason |
|---|---|---|
| `autoskill` | skill-evolution | Derive new local skills from repeated alpha workflows after patterns stabilize. |
| `citation-management` | provenance | Citation quality, BibTeX, and source validation for literature-derived alphas. |
| `docx` | document-ingest | Extract/edit Word research notes or reports. |
| `infographics` | reporting | Executive visual summaries after evidence exists. |
| `lamindb` | lineage | Lineage/lakehouse inspiration; prefer LuminaQuant native SHA/source artifact tracking first. |
| `latex-posters` | reporting | Poster output only. |
| `liteparse` | document-ingest | OCR/layout-aware local PDF/document parsing for RAG inputs. |
| `markdown-mermaid-writing` | reporting | Write audit-friendly reports and pipeline diagrams. |
| `market-research-reports` | macro-context | Heavy macro/industry context reports; not proof of alpha. |
| `markitdown` | document-ingest | Convert files into Markdown for auditable research ingestion. |
| `matlab` | numerics | Translate/validate MATLAB/Octave quant prototypes when inherited. |
| `matplotlib` | visualization | Low-level plot control for reports. |
| `modal` | cloud-compute | Optional cloud/GPU batch execution; avoid unless explicitly authorized. |
| `open-notebook` | research-management | Organize paper/report corpora into notebooks; optional RAG workspace. |
| `paperzilla` | research-management | Project/paper recommendation management, optional. |
| `pdf` | document-ingest | Extract tables/text from papers and reports. |
| `pi-agent` | agent-platform | Build/run Pi harnesses only if that platform is explicitly used. |
| `pptx` | document-ingest | Extract slide decks or create research presentations. |
| `pptx-posters` | reporting | Poster output only. |
| `pyzotero` | provenance | Zotero library integration for research provenance if the project uses Zotero. |
| `scientific-schematics` | reporting | System/pipeline schematics and conceptual diagrams. |
| `scientific-slides` | reporting | Research presentation deck creation after evidence exists. |
| `scientific-writing` | reporting | Convert validated evidence into prose reports. |
| `seaborn` | visualization | Fast statistical plots for distributions, correlations, and diagnostics. |
| `sympy` | math | Symbolic derivations for transforms, constraints, or report equations. |
| `venue-templates` | reporting | Formatting templates for formal papers/posters if needed. |
| `xlsx` | document-ingest | Ingest spreadsheets, broker extracts, and research workbooks. |

## Tier D

| Skill | Category | LuminaQuant alpha use / exclusion reason |
|---|---|---|
| `adaptyv` | excluded-chemistry | Reviewed; chemistry/drug-discovery workflow is not a default LuminaQuant alpha source. |
| `anndata` | excluded-biomedical | Reviewed; biomedical/genomics workflow is not a default LuminaQuant alpha source. |
| `arboreto` | excluded-lab | Reviewed; lab automation/integration workflow is not a default LuminaQuant alpha source. |
| `astropy` | excluded-astronomy | Astronomy workflows not default alpha research. |
| `benchling-integration` | excluded-lab | Reviewed; lab automation/integration workflow is not a default LuminaQuant alpha source. |
| `bids` | excluded-clinical | Reviewed; clinical/medical workflow is not a default LuminaQuant alpha source. |
| `biopython` | excluded-chemistry | Reviewed; chemistry/drug-discovery workflow is not a default LuminaQuant alpha source. |
| `bioservices` | excluded-chemistry | Reviewed; chemistry/drug-discovery workflow is not a default LuminaQuant alpha source. |
| `bulk-rnaseq` | excluded-biomedical | Reviewed; biomedical/genomics workflow is not a default LuminaQuant alpha source. |
| `cellxgene-census` | excluded-biomedical | Reviewed; biomedical/genomics workflow is not a default LuminaQuant alpha source. |
| `cirq` | excluded-quantum | Reviewed; quantum computing workflow is not needed for current alpha discovery. |
| `clinical-decision-support` | excluded-clinical | Reviewed; clinical/medical workflow is not a default LuminaQuant alpha source. |
| `clinical-reports` | excluded-clinical | Reviewed; clinical/medical workflow is not a default LuminaQuant alpha source. |
| `cobrapy` | excluded-chemistry | Reviewed; chemistry/drug-discovery workflow is not a default LuminaQuant alpha source. |
| `datamol` | excluded-chemistry | Reviewed; chemistry/drug-discovery workflow is not a default LuminaQuant alpha source. |
| `deepchem` | excluded-chemistry | Reviewed; chemistry/drug-discovery workflow is not a default LuminaQuant alpha source. |
| `deeptools` | excluded-biomedical | Reviewed; biomedical/genomics workflow is not a default LuminaQuant alpha source. |
| `depmap` | excluded-chemistry | Reviewed; chemistry/drug-discovery workflow is not a default LuminaQuant alpha source. |
| `dhdna-profiler` | excluded-cognitive | Reviewed; cognitive-style profiling is not needed for alpha research. |
| `diffdock` | excluded-chemistry | Reviewed; chemistry/drug-discovery workflow is not a default LuminaQuant alpha source. |
| `dnanexus-integration` | excluded-lab | Reviewed; lab automation/integration workflow is not a default LuminaQuant alpha source. |
| `esm` | excluded-biomedical | Reviewed; biomedical/genomics workflow is not a default LuminaQuant alpha source. |
| `etetoolkit` | excluded-biomedical | Reviewed; biomedical/genomics workflow is not a default LuminaQuant alpha source. |
| `flowio` | excluded-other | Reviewed; no direct role in finding, validating, analyzing, or reporting LuminaQuant trading alpha by default. |
| `fluidsim` | excluded-simulation | Physics CFD not default; only analogy/commodity supply-chain special cases. |
| `generate-image` | excluded-chemistry | Reviewed; chemistry/drug-discovery workflow is not a default LuminaQuant alpha source. |
| `geniml` | excluded-biomedical | Reviewed; biomedical/genomics workflow is not a default LuminaQuant alpha source. |
| `gget` | excluded-biomedical | Reviewed; biomedical/genomics workflow is not a default LuminaQuant alpha source. |
| `ginkgo-cloud-lab` | excluded-lab | Reviewed; lab automation/integration workflow is not a default LuminaQuant alpha source. |
| `glycoengineering` | excluded-chemistry | Reviewed; chemistry/drug-discovery workflow is not a default LuminaQuant alpha source. |
| `gtars` | excluded-biomedical | Reviewed; biomedical/genomics workflow is not a default LuminaQuant alpha source. |
| `histolab` | excluded-lab | Reviewed; lab automation/integration workflow is not a default LuminaQuant alpha source. |
| `hugging-science` | excluded-materials | Reviewed; materials science workflow is not default unless explicit commodity/materials alt-data research is requested. |
| `imaging-data-commons` | excluded-clinical | Reviewed; clinical/medical workflow is not a default LuminaQuant alpha source. |
| `iso-13485-certification` | excluded-clinical | Reviewed; clinical/medical workflow is not a default LuminaQuant alpha source. |
| `labarchive-integration` | excluded-lab | Reviewed; lab automation/integration workflow is not a default LuminaQuant alpha source. |
| `latchbio-integration` | excluded-lab | Reviewed; lab automation/integration workflow is not a default LuminaQuant alpha source. |
| `matchms` | excluded-chemistry | Reviewed; chemistry/drug-discovery workflow is not a default LuminaQuant alpha source. |
| `medchem` | excluded-chemistry | Reviewed; chemistry/drug-discovery workflow is not a default LuminaQuant alpha source. |
| `molecular-dynamics` | excluded-chemistry | Reviewed; chemistry/drug-discovery workflow is not a default LuminaQuant alpha source. |
| `molfeat` | excluded-chemistry | Reviewed; chemistry/drug-discovery workflow is not a default LuminaQuant alpha source. |
| `neurokit2` | excluded-biosignal | Could inspire signal processing, but biomedical signals are not default LuminaQuant data. |
| `neuropixels-analysis` | excluded-biomedical | Reviewed; biomedical/genomics workflow is not a default LuminaQuant alpha source. |
| `nextflow` | excluded-biomedical | Reviewed; biomedical/genomics workflow is not a default LuminaQuant alpha source. |
| `omero-integration` | excluded-lab | Reviewed; lab automation/integration workflow is not a default LuminaQuant alpha source. |
| `opentrons-integration` | excluded-lab | Reviewed; lab automation/integration workflow is not a default LuminaQuant alpha source. |
| `pacsomatic` | excluded-biomedical | Reviewed; biomedical/genomics workflow is not a default LuminaQuant alpha source. |
| `pathml` | excluded-clinical | Reviewed; clinical/medical workflow is not a default LuminaQuant alpha source. |
| `pathway-enrichment` | excluded-biomedical | Reviewed; biomedical/genomics workflow is not a default LuminaQuant alpha source. |
| `pennylane` | excluded-quantum | Reviewed; quantum computing workflow is not needed for current alpha discovery. |
| `phylogenetics` | excluded-chemistry | Reviewed; chemistry/drug-discovery workflow is not a default LuminaQuant alpha source. |
| `polars-bio` | excluded-biomedical | Reviewed; biomedical/genomics workflow is not a default LuminaQuant alpha source. |
| `primekg` | excluded-chemistry | Reviewed; chemistry/drug-discovery workflow is not a default LuminaQuant alpha source. |
| `protocolsio-integration` | excluded-lab | Reviewed; lab automation/integration workflow is not a default LuminaQuant alpha source. |
| `pydeseq2` | excluded-biomedical | Reviewed; biomedical/genomics workflow is not a default LuminaQuant alpha source. |
| `pydicom` | excluded-clinical | Reviewed; clinical/medical workflow is not a default LuminaQuant alpha source. |
| `pyhealth` | excluded-clinical | Reviewed; clinical/medical workflow is not a default LuminaQuant alpha source. |
| `pylabrobot` | excluded-lab | Reviewed; lab automation/integration workflow is not a default LuminaQuant alpha source. |
| `pymatgen` | excluded-materials | Reviewed; materials science workflow is not default unless explicit commodity/materials alt-data research is requested. |
| `pyopenms` | excluded-lab | Reviewed; lab automation/integration workflow is not a default LuminaQuant alpha source. |
| `pysam` | excluded-biomedical | Reviewed; biomedical/genomics workflow is not a default LuminaQuant alpha source. |
| `pytdc` | excluded-chemistry | Reviewed; chemistry/drug-discovery workflow is not a default LuminaQuant alpha source. |
| `qiskit` | excluded-quantum | Reviewed; quantum computing workflow is not needed for current alpha discovery. |
| `qutip` | excluded-quantum | Reviewed; quantum computing workflow is not needed for current alpha discovery. |
| `rdkit` | excluded-chemistry | Reviewed; chemistry/drug-discovery workflow is not a default LuminaQuant alpha source. |
| `research-grants` | excluded-other | Reviewed; no direct role in finding, validating, analyzing, or reporting LuminaQuant trading alpha by default. |
| `rowan` | excluded-chemistry | Reviewed; chemistry/drug-discovery workflow is not a default LuminaQuant alpha source. |
| `scanpy` | excluded-biomedical | Reviewed; biomedical/genomics workflow is not a default LuminaQuant alpha source. |
| `scikit-bio` | excluded-biomedical | Reviewed; biomedical/genomics workflow is not a default LuminaQuant alpha source. |
| `scvelo` | excluded-biomedical | Reviewed; biomedical/genomics workflow is not a default LuminaQuant alpha source. |
| `scvi-tools` | excluded-biomedical | Reviewed; biomedical/genomics workflow is not a default LuminaQuant alpha source. |
| `tiledbvcf` | excluded-lab | Reviewed; lab automation/integration workflow is not a default LuminaQuant alpha source. |
| `torchdrug` | excluded-chemistry | Reviewed; chemistry/drug-discovery workflow is not a default LuminaQuant alpha source. |
| `treatment-plans` | excluded-clinical | Reviewed; clinical/medical workflow is not a default LuminaQuant alpha source. |

## Complete 147-Skill Review Table

| # | Skill | Tier | Category | Upstream description |
|---:|---|---|---|---|
| 1 | `adaptyv` | D | excluded-chemistry | How to use the Adaptyv Bio Foundry API and Python SDK for protein experiment design, submission, and results retrieval. Use this skill whenever the user mentions Adaptyv, Foundry… |
| 2 | `aeon` | A | time-series-ml | This skill should be used for time series machine learning tasks including classification, regression, clustering, forecasting, anomaly detection, segmentation, and similarity… |
| 3 | `anndata` | D | excluded-biomedical | Data structure for annotated matrices in single-cell analysis. Use when working with .h5ad files or integrating with the scverse ecosystem. This is the data format skill—for… |
| 4 | `arbor` | A | optimization-loop | Autonomously improve a real artifact (code, training recipe, agent harness, data pipeline, prompt) against an objective and an evaluator, using Hypothesis Tree Refinement (HTR)… |
| 5 | `arboreto` | D | excluded-lab | Infer gene regulatory networks (GRNs) from gene expression data using scalable algorithms (GRNBoost2, GENIE3). Use when analyzing transcriptomics data (bulk RNA-seq, single-cell… |
| 6 | `astropy` | D | excluded-astronomy | Core Python library for astronomy and astrophysics workflows that need Astropy APIs, including units/quantities, coordinates, FITS I/O, tables, time systems, WCS, and cosmology.… |
| 7 | `autoskill` | C | skill-evolution | Observe the user's screen via screenpipe, detect repeated research workflows, match them against existing scientific-agent-skills, and draft new skills (or composition recipes… |
| 8 | `benchling-integration` | D | excluded-lab | Benchling Python SDK and REST API integration for registry entities, inventory, ELN entries, workflows, Benchling Apps, and Data Warehouse queries. Use when automating lab data… |
| 9 | `bgpt-paper-search` | B | literature | Search scientific papers and retrieve structured experimental data extracted from full-text studies via the BGPT MCP server. Returns 25+ fields per paper including methods,… |
| 10 | `bids` | D | excluded-clinical | > Use this skill when working with Brain Imaging Data Structure (BIDS) datasets: organizing neuroscience and biomedical data (MRI, EEG, MEG, iEEG, PET, microscopy, NIRS, motion… |
| 11 | `biopython` | D | excluded-chemistry | Comprehensive molecular biology toolkit. Use for sequence manipulation, file parsing (FASTA/GenBank/PDB), phylogenetics, and programmatic NCBI/PubMed access (Bio.Entrez). Best for… |
| 12 | `bioservices` | D | excluded-chemistry | Unified Python interface to 40+ bioinformatics services. Use when querying multiple databases (UniProt, KEGG, ChEMBL, Reactome) in a single workflow with consistent API. Best for… |
| 13 | `bulk-rnaseq` | D | excluded-biomedical | End-to-end bulk RNA-seq orchestrator — takes raw FASTQ reads through QC and trimming (FastQC, fastp/Trim Galore), alignment and quantification (STAR, Salmon, featureCounts),… |
| 14 | `cellxgene-census` | D | excluded-biomedical | Query the CZ CELLxGENE Census programmatically for versioned public single-cell and spatial transcriptomics data. Use when you need population-scale cell metadata, gene expression… |
| 15 | `cirq` | D | excluded-quantum | Google quantum computing framework. Use when targeting Google Quantum AI hardware, designing noise-aware circuits, or running quantum characterization experiments. Best for Google… |
| 16 | `citation-management` | C | provenance | Comprehensive citation management for academic research. Search Google Scholar and PubMed for papers, extract accurate metadata, validate citations, and generate properly… |
| 17 | `clinical-decision-support` | D | excluded-clinical | Generate professional clinical decision support (CDS) documents for pharmaceutical and clinical research settings, including patient cohort analyses (biomarker-stratified with… |
| 18 | `clinical-reports` | D | excluded-clinical | Write comprehensive clinical reports including case reports (CARE guidelines), diagnostic reports (radiology/pathology/lab), clinical trial reports (ICH-E3, SAE, CSR), and patient… |
| 19 | `cobrapy` | D | excluded-chemistry | Constraint-based metabolic modeling (COBRA). FBA, FVA, gene knockouts, flux sampling, SBML models, for systems biology and metabolic engineering analysis. |
| 20 | `consciousness-council` | B | multi-perspective-review | Run a multi-perspective Mind Council deliberation on any question, decision, or creative challenge. Use this skill whenever the user wants diverse viewpoints, needs help making a… |
| 21 | `dask` | B | distributed-data | Distributed computing for larger-than-RAM pandas/NumPy workflows. Use when you need to scale existing pandas/NumPy code beyond memory or across clusters. Best for parallel file… |
| 22 | `database-lookup` | A | external-data | Deterministically query 78 public scientific, biomedical, materials science, regulatory, finance, and demographics databases through documented REST APIs. Use for reproducible… |
| 23 | `datamol` | D | excluded-chemistry | Pythonic wrapper around RDKit with simplified interface and sensible defaults. Preferred for standard drug discovery including SMILES parsing, standardization, descriptors,… |
| 24 | `deepchem` | D | excluded-chemistry | Molecular ML with diverse featurizers and pre-built datasets. Use for property prediction (ADMET, toxicity) with traditional ML or GNNs when you want extensive featurization… |
| 25 | `deeptools` | D | excluded-biomedical | NGS analysis toolkit. BAM to bigWig conversion, QC (correlation, PCA, fingerprints), heatmaps/profiles (TSS, peaks), for ChIP-seq, RNA-seq, ATAC-seq visualization. |
| 26 | `depmap` | D | excluded-chemistry | Query the Cancer Dependency Map (DepMap) for cancer cell line gene dependency scores (CRISPR Chronos), drug sensitivity data, and gene effect profiles. Use for identifying cancer-… |
| 27 | `dhdna-profiler` | D | excluded-cognitive | Extract cognitive patterns and thinking fingerprints from any text. Use this skill when the user wants to analyze how someone thinks, understand cognitive style, profile writing… |
| 28 | `diffdock` | D | excluded-chemistry | DiffDock and DiffDock-L molecular docking. Use for protein-small-molecule pose prediction from PDB or sequence plus SMILES/SDF/MOL2, batch docking, virtual screening, and pose-… |
| 29 | `dnanexus-integration` | D | excluded-lab | DNAnexus cloud genomics platform. Build apps/applets, manage data (upload/download), dxpy Python SDK, run workflows, FASTQ/BAM/VCF, for genomics pipeline development and execution. |
| 30 | `docx` | C | document-ingest | Use this skill whenever the user wants to create, read, edit, or manipulate Word documents (.docx files). Triggers include: any mention of 'Word doc', 'word document', '.docx', or… |
| 31 | `esm` | D | excluded-biomedical | Use when working directly with the `esm` Python SDK, ESM3 or ESMC model IDs, Forge/Biohub inference clients, or ESMFold2 folding workflows. |
| 32 | `etetoolkit` | D | excluded-biomedical | Phylogenetic tree toolkit (ETE). Tree manipulation (Newick/NHX), evolutionary event detection, orthology/paralogy, NCBI taxonomy, visualization (PDF/SVG), for phylogenomics. |
| 33 | `exa-search` | B | web-research | Web toolkit powered by Exa, tuned for scientific and technical content. Use this skill when the user needs to search the web or fetch/extract URL content. Covers: web search… |
| 34 | `experimental-design` | A | experiment-design | Design experiments and studies BEFORE data is collected — choosing a design, randomizing, blocking, and laying out treatment combinations so the results will actually be… |
| 35 | `exploratory-data-analysis` | A | eda | Perform comprehensive exploratory data analysis on scientific data files across 200+ file formats. This skill should be used when analyzing any scientific data file to understand… |
| 36 | `flowio` | D | excluded-other | Parse FCS (Flow Cytometry Standard) files v2.0-3.1. Extract events as NumPy arrays, read metadata/channels, convert to CSV/DataFrame, for flow cytometry data preprocessing. |
| 37 | `fluidsim` | D | excluded-simulation | Framework for computational fluid dynamics simulations using Python. Use when running fluid dynamics simulations including Navier-Stokes equations (2D/3D), shallow water… |
| 38 | `generate-image` | D | excluded-chemistry | Generate or edit images using AI models (FLUX, Nano Banana 2). Use for general-purpose image generation including photos, illustrations, artwork, visual assets, concept art, and… |
| 39 | `geniml` | D | excluded-biomedical | This skill should be used when working with genomic interval data (BED files) for machine learning tasks. Use for training region embeddings (Region2Vec, BEDspace), single-cell… |
| 40 | `geomaster` | B | geospatial-alt-data | Comprehensive geospatial science skill covering remote sensing, GIS, spatial analysis, machine learning for earth observation, and 30+ scientific domains. Supports satellite… |
| 41 | `geopandas` | B | geospatial-alt-data | Python library for working with geospatial vector data including shapefiles, GeoJSON, and GeoPackage files. Use when working with geographic data for spatial analysis, geometric… |
| 42 | `get-available-resources` | A | preflight | This skill should be used at the start of any computationally intensive scientific task to detect and report available system resources (CPU cores, GPUs, memory, disk space). It… |
| 43 | `gget` | D | excluded-biomedical | Fast CLI/Python queries to 20+ bioinformatics databases. Use for quick lookups: gene info, BLAST/BLAT, viral sequence downloads, AlphaFold structures, enrichment analysis,… |
| 44 | `ginkgo-cloud-lab` | D | excluded-lab | Submit and manage protocols on Ginkgo Bioworks Cloud Lab (cloud.ginkgo.bio), a web-based interface for autonomous lab execution on Reconfigurable Automation Carts (RACs). Use when… |
| 45 | `glycoengineering` | D | excluded-chemistry | Analyze and engineer protein glycosylation. Scan sequences for N-glycosylation sequons (N-X-S/T), predict O-glycosylation hotspots, and access curated glycoengineering tools… |
| 46 | `gtars` | D | excluded-biomedical | High-performance toolkit for genomic interval analysis in Rust with Python bindings. Use when working with genomic regions, BED files, coverage tracks, overlap detection,… |
| 47 | `histolab` | D | excluded-lab | Lightweight WSI tile extraction and preprocessing. Use for basic slide processing, tissue detection, tile extraction, and stain normalization for H&E images. Best for simple… |
| 48 | `hugging-science` | D | excluded-materials | Use when the user is doing AI/ML work in a scientific domain such as biology, chemistry, physics, astronomy, climate, genomics, materials, medicine, ecology, energy, engineering,… |
| 49 | `hypogenic` | A | data-driven-hypotheses | Automated LLM-driven hypothesis generation and testing on tabular datasets. Use when you want to systematically explore hypotheses about patterns in empirical data (e.g.,… |
| 50 | `hypothesis-generation` | A | ideation | Structured hypothesis formulation from observations. Use when you have experimental observations or data and need to formulate testable hypotheses with predictions, propose… |
| 51 | `imaging-data-commons` | D | excluded-clinical | Query and download public cancer imaging data from NCI Imaging Data Commons using idc-index. Use for accessing large-scale radiology (CT, MR, PET) and pathology datasets for AI… |
| 52 | `infographics` | C | reporting | Create professional infographics using Nano Banana Pro AI with smart iterative refinement. Uses Gemini 3 Pro for quality review. Integrates research-lookup and web search for… |
| 53 | `iso-13485-certification` | D | excluded-clinical | Comprehensive toolkit for preparing ISO 13485 certification documentation for medical device Quality Management Systems. Use when users need help with ISO 13485 QMS documentation,… |
| 54 | `labarchive-integration` | D | excluded-lab | Electronic lab notebook API integration. Access notebooks, manage entries/attachments, backup notebooks, integrate with Protocols.io/Jupyter/REDCap, for programmatic ELN workflows. |
| 55 | `lamindb` | C | lineage | Use when working with LaminDB, the open-source lineage-native lakehouse for biological datasets and models. Covers setup, artifact registration, query/search, lineage tracking,… |
| 56 | `latchbio-integration` | D | excluded-lab | Latch platform for bioinformatics workflows. Build pipelines with Latch SDK, @workflow/@task decorators, deploy serverless workflows, LatchFile/LatchDir, Nextflow/Snakemake… |
| 57 | `latex-posters` | C | reporting | Create professional research posters in LaTeX using beamerposter, tikzposter, or baposter. Support for conference presentations, academic posters, and scientific communication.… |
| 58 | `liteparse` | C | document-ingest | Local document and PDF parsing with spatial text and bounding boxes. Use for extracting text from PDFs, DOCX, Office files, and images; OCR on scans; layout-preserved JSON for… |
| 59 | `literature-review` | A | literature | Conduct comprehensive, systematic literature reviews using multiple academic databases (PubMed, arXiv, bioRxiv, Semantic Scholar, etc.). This skill should be used when conducting… |
| 60 | `markdown-mermaid-writing` | C | reporting | Comprehensive markdown and Mermaid diagram writing skill. Use when creating any scientific document, report, analysis, or visualization. Establishes text-based diagrams as the… |
| 61 | `market-research-reports` | C | macro-context | Generate comprehensive market research reports (50+ pages) in the style of top consulting firms (McKinsey, BCG, Gartner). Features professional LaTeX formatting, extensive visual… |
| 62 | `markitdown` | C | document-ingest | Convert files and office documents to Markdown. Supports PDF, DOCX, PPTX, XLSX, images (with OCR), audio (with transcription), HTML, CSV, JSON, XML, ZIP, YouTube URLs, EPubs and… |
| 63 | `matchms` | D | excluded-chemistry | Spectral similarity and compound identification for metabolomics. Use for comparing mass spectra, computing similarity scores (cosine, modified cosine), and identifying unknown… |
| 64 | `matlab` | C | numerics | MATLAB and GNU Octave numerical computing for matrix operations, data analysis, visualization, and scientific computing. Use when writing MATLAB/Octave scripts for linear algebra,… |
| 65 | `matplotlib` | C | visualization | Low-level plotting library for full customization. Use when you need fine-grained control over every plot element, creating novel plot types, or integrating with specific… |
| 66 | `medchem` | D | excluded-chemistry | Medicinal chemistry filters for compound triage. Apply drug-likeness rules (Lipinski, Veber, CNS), structural alert catalogs (PAINS, NIBR, ChEMBL), complexity metrics, and the… |
| 67 | `modal` | C | cloud-compute | Modal is a serverless cloud platform for running Python on demand, including on-demand GPUs. Use when deploying or serving AI/ML models, running GPU-accelerated workloads… |
| 68 | `molecular-dynamics` | D | excluded-chemistry | Run and analyze molecular dynamics simulations with OpenMM and MDAnalysis. Set up protein/small molecule systems, define force fields, run energy minimization and production MD,… |
| 69 | `molfeat` | D | excluded-chemistry | Molecular featurization for ML (100+ featurizers). ECFP, MACCS, descriptors, pretrained models (ChemBERTa), convert SMILES to features, for QSAR and molecular ML. |
| 70 | `networkx` | B | graph-alpha | Create, analyze, and visualize complex networks and graphs in Python with NetworkX. Use when working with network/graph data structures, computing graph algorithms (shortest… |
| 71 | `neurokit2` | D | excluded-biosignal | Comprehensive biosignal processing toolkit for analyzing physiological data including ECG, EEG, EDA, RSP, PPG, EMG, and EOG signals. Use this skill when processing cardiovascular… |
| 72 | `neuropixels-analysis` | D | excluded-biomedical | Analyze Neuropixels extracellular recordings end-to-end with SpikeInterface. Covers loading SpikeGLX/Open Ephys/NWB data, preprocessing, drift/motion correction, Kilosort4 (and… |
| 73 | `nextflow` | D | excluded-biomedical | Build, run, and debug Nextflow data pipelines and nf-core workflows end to end. Use whenever the user mentions Nextflow, nf-core, .nf files, nextflow.config, DSL2,… |
| 74 | `omero-integration` | D | excluded-lab | Microscopy data management platform. Access images via Python, retrieve datasets, analyze pixels, manage ROIs/annotations, batch processing, for high-content screening and… |
| 75 | `open-notebook` | C | research-management | Self-hosted, open-source alternative to Google NotebookLM for AI-powered research and document analysis. Use when organizing research materials into notebooks, ingesting diverse… |
| 76 | `opentrons-integration` | D | excluded-lab | Official Opentrons Protocol API for OT-2 and Flex robots. Use when writing protocols specifically for Opentrons hardware with full access to Protocol API v2 features. Best for… |
| 77 | `optimize-for-gpu` | B | gpu | GPU-accelerate Python code using CuPy, Numba CUDA, Warp, cuDF, cuML, cuGraph, KvikIO, cuCIM, cuxfilter, cuVS, cuSpatial, and RAFT. Use whenever the user mentions GPU/CUDA/NVIDIA… |
| 78 | `pacsomatic` | D | excluded-biomedical | Operator toolkit for nf-core/pacsomatic matched tumor-normal workflows from BAM inputs. Use this skill when the user needs to validate run inputs, generate pacsomatic-compliant… |
| 79 | `paper-lookup` | A | literature | Search 10 academic paper databases via REST APIs for research papers, preprints, and scholarly articles. Covers PubMed, PMC (full text), bioRxiv, medRxiv, arXiv, OpenAlex,… |
| 80 | `paperzilla` | C | research-management | Chat with your agent about projects, recommendations, and canonical papers in Paperzilla. Use when users ask for recent project recommendations, canonical paper details, markdown-… |
| 81 | `parallel-web` | B | web-research | All-in-one web toolkit powered by parallel-cli, with a strong emphasis on academic and scientific sources. Use this skill whenever the user needs to search the web, fetch/extract… |
| 82 | `pathml` | D | excluded-clinical | Full-featured computational pathology toolkit. Use for advanced WSI analysis including multiplexed immunofluorescence (CODEX, Vectra), nucleus segmentation, tissue graph… |
| 83 | `pathway-enrichment` | D | excluded-biomedical | Run pathway and gene-set enrichment analysis on gene lists or ranked gene data, then interpret the results. Use whenever the user has a set of genes (differentially expressed… |
| 84 | `pdf` | C | document-ingest | Use this skill whenever the user wants to do anything with PDF files. This includes reading or extracting text/tables from PDFs, combining or merging multiple PDFs into one,… |
| 85 | `peer-review` | A | review | Structured manuscript/grant review with checklist-based evaluation. Use when writing formal peer reviews with specific criteria methodology assessment, statistical validity,… |
| 86 | `pennylane` | D | excluded-quantum | Hardware-agnostic quantum ML framework with automatic differentiation. Use when training quantum circuits via gradients, building hybrid quantum-classical models, or needing… |
| 87 | `phylogenetics` | D | excluded-chemistry | Build and analyze phylogenetic trees using MAFFT (multiple alignment), IQ-TREE 2 (maximum likelihood), and FastTree (fast NJ/ML). Visualize with ETE3 or FigTree. For evolutionary… |
| 88 | `pi-agent` | C | agent-platform | Build with and use Pi, the minimal terminal coding harness. Use for installing Pi, configuring providers/models/settings, creating Pi skills/extensions/packages/themes/prompt… |
| 89 | `polars` | A | dataframe | High-performance DataFrame library for Python ETL, analytics, and pandas migration. Use for expression-based data manipulation with lazy query optimization, parallel execution,… |
| 90 | `polars-bio` | D | excluded-biomedical | High-performance genomic interval operations and bioinformatics file I/O on Polars DataFrames. Overlap, nearest, merge, coverage, complement, subtract for BED/VCF/BAM/GFF… |
| 91 | `pptx` | C | document-ingest | Use this skill any time a .pptx file is involved in any way — as input, output, or both. This includes: creating slide decks, pitch decks, or presentations; reading, parsing, or… |
| 92 | `pptx-posters` | C | reporting | Create research posters using HTML/CSS that can be exported to PDF or PPTX. Use this skill ONLY when the user explicitly requests PowerPoint/PPTX poster format. For standard… |
| 93 | `primekg` | D | excluded-chemistry | Query the Precision Medicine Knowledge Graph (PrimeKG) for multiscale biological data including genes, drugs, diseases, phenotypes, and more. |
| 94 | `protocolsio-integration` | D | excluded-lab | Integration with protocols.io API for managing scientific protocols. This skill should be used when working with protocols.io to search, create, update, or publish protocols;… |
| 95 | `pufferlib` | B | rl | High-performance reinforcement learning framework optimized for speed and scale. Use when you need fast parallel training, vectorized environments, multi-agent systems, or… |
| 96 | `pydeseq2` | D | excluded-biomedical | Differential gene expression analysis for bulk RNA-seq with PyDESeq2, including formulaic designs, Wald tests, FDR correction, LFC shrinkage, and result visualization. |
| 97 | `pydicom` | D | excluded-clinical | Python library for working with DICOM (Digital Imaging and Communications in Medicine) files. Use this skill when reading, writing, or modifying medical imaging data in DICOM… |
| 98 | `pyhealth` | D | excluded-clinical | Build clinical/healthcare deep-learning pipelines with PyHealth — loading EHR/signal/imaging datasets (MIMIC-III/IV, eICU, OMOP, SleepEDF, ChestXray14, EHRShot), defining tasks… |
| 99 | `pylabrobot` | D | excluded-lab | Vendor-agnostic lab automation framework. Use when controlling multiple equipment types (Hamilton, Tecan, Opentrons, plate readers, pumps) or needing unified programming across… |
| 100 | `pymatgen` | D | excluded-materials | Materials science toolkit. Crystal structures (CIF, POSCAR), phase diagrams, band structure, DOS, Materials Project integration, format conversion, for computational materials… |
| 101 | `pymc` | B | bayesian | Bayesian modeling with PyMC. Build hierarchical models, MCMC (NUTS), variational inference, LOO/WAIC comparison, posterior checks, for probabilistic programming and inference. |
| 102 | `pymoo` | A | multi-objective-optimization | Multi-objective optimization framework. NSGA-II, NSGA-III, MOEA/D, Pareto fronts, constraint handling, benchmarks (ZDT, DTLZ), for engineering design and optimization problems. |
| 103 | `pyopenms` | D | excluded-lab | Complete mass spectrometry analysis platform. Use for proteomics and metabolomics workflows—feature detection, peptide/protein identification, label-free and isobaric… |
| 104 | `pysam` | D | excluded-biomedical | Genomic file toolkit. Read/write SAM/BAM/CRAM alignments, VCF/BCF variants, FASTA/FASTQ sequences, extract regions, calculate coverage, for NGS data processing pipelines. |
| 105 | `pytdc` | D | excluded-chemistry | Therapeutics Data Commons. AI-ready drug discovery datasets (ADME, toxicity, DTI), benchmarks, scaffold splits, molecular oracles, for therapeutic ML and pharmacological… |
| 106 | `pytorch-lightning` | B | deep-learning | Deep learning framework (PyTorch Lightning / lightning package). Organize PyTorch code into LightningModules, configure Trainers for multi-GPU/TPU, implement data pipelines,… |
| 107 | `pyzotero` | C | provenance | Interact with Zotero reference management libraries using the pyzotero Python client. Retrieve, create, update, and delete items, collections, tags, and attachments via the Zotero… |
| 108 | `qiskit` | D | excluded-quantum | IBM quantum computing framework. Use when targeting IBM Quantum hardware, working with Qiskit Runtime for production workloads, or needing IBM optimization tools. Best for IBM… |
| 109 | `qutip` | D | excluded-quantum | Quantum physics simulation library for open quantum systems. Use when studying master equations, Lindblad dynamics, decoherence, quantum optics, or cavity QED. Best for physics… |
| 110 | `rdkit` | D | excluded-chemistry | Cheminformatics toolkit for fine-grained molecular control. SMILES/SDF parsing, descriptors (MW, LogP, TPSA), fingerprints, substructure search, 2D/3D generation, similarity,… |
| 111 | `research-grants` | D | excluded-other | Write competitive research proposals for NSF, NIH, DOE, DARPA, and Taiwan NSTC. Agency-specific formatting, review criteria, budget preparation, broader impacts, significance… |
| 112 | `research-lookup` | A | literature | Look up current research information using parallel-cli search (primary, fast web search), the Parallel Chat API (deep research), or Perplexity sonar-pro-search (academic paper… |
| 113 | `rowan` | D | excluded-chemistry | Rowan is a cloud-native molecular modeling and medicinal-chemistry workflow platform with a Python API. Use for pKa and macropKa prediction, conformer and tautomer ensembles,… |
| 114 | `scanpy` | D | excluded-biomedical | Standard single-cell RNA-seq analysis pipeline. Use for QC, normalization, dimensionality reduction (PCA/UMAP/t-SNE), clustering, differential expression, visualization, and… |
| 115 | `scholar-evaluation` | B | review | Systematically evaluate scholarly work using the ScholarEval framework, providing structured assessment across research quality dimensions including problem formulation,… |
| 116 | `scientific-brainstorming` | A | ideation | Creative research ideation and exploration. Use for open-ended brainstorming sessions, exploring interdisciplinary connections, challenging assumptions, or identifying research… |
| 117 | `scientific-critical-thinking` | A | bias-control | Evaluate scientific claims and evidence quality. Use for assessing experimental design validity, identifying biases and confounders, applying evidence grading frameworks (GRADE,… |
| 118 | `scientific-schematics` | C | reporting | Create publication-quality scientific diagrams using Nano Banana 2 AI with smart iterative refinement. Uses Gemini 3.1 Pro Preview for quality review. Only regenerates if quality… |
| 119 | `scientific-slides` | C | reporting | Build slide decks and presentations for research talks. Use this for making PowerPoint slides, conference presentations, seminar talks, research presentations, thesis defense… |
| 120 | `scientific-visualization` | A | visualization | Meta-skill for publication-ready figures. Use when creating journal submission figures requiring multi-panel layouts, significance annotations, error bars, colorblind-safe… |
| 121 | `scientific-writing` | C | reporting | Core skill for the deep research and writing tool. Write scientific manuscripts in full paragraphs (never bullet points). Use two-stage process with (1) section outlines with key… |
| 122 | `scikit-bio` | D | excluded-biomedical | Biological data toolkit. Sequence analysis, alignments, phylogenetic trees, diversity metrics (alpha/beta, UniFrac), ordination (PCoA), PERMANOVA, FASTA/Newick I/O, for microbiome… |
| 123 | `scikit-learn` | A | ml | Machine learning in Python with scikit-learn. Use when working with supervised learning (classification, regression), unsupervised learning (clustering, dimensionality reduction),… |
| 124 | `scikit-survival` | B | hazard-modeling | Comprehensive toolkit for survival analysis and time-to-event modeling in Python using scikit-survival. Use this skill when working with censored survival data, performing time-… |
| 125 | `scvelo` | D | excluded-biomedical | RNA velocity analysis with scVelo. Estimate cell state transitions from unspliced/spliced mRNA dynamics, infer trajectory directions, compute latent time, and identify driver… |
| 126 | `scvi-tools` | D | excluded-biomedical | Deep generative models for single-cell omics. Use when you need probabilistic batch correction (scVI), transfer learning, differential expression with uncertainty, or multi-modal… |
| 127 | `seaborn` | C | visualization | Statistical visualization with pandas integration. Use for quick exploration of distributions, relationships, and categorical comparisons with attractive defaults. Best for box… |
| 128 | `shap` | A | explainability | Model interpretability and explainability using SHAP (SHapley Additive exPlanations). Use this skill when explaining machine learning model predictions, computing feature… |
| 129 | `simpy` | B | execution-simulation | Process-based discrete-event simulation framework in Python. Use this skill when building simulations of systems with processes, queues, resources, and time-based events such as… |
| 130 | `stable-baselines3` | B | rl | Production-ready reinforcement learning algorithms (PPO, SAC, DQN, TD3, DDPG, A2C) with scikit-learn-like API. Use for standard RL experiments, quick prototyping, and well-… |
| 131 | `statistical-analysis` | A | validation | Guided statistical analysis with test selection and reporting. Use when you need help choosing appropriate tests for your data, assumption checking, power analysis, and APA-… |
| 132 | `statistical-power` | B | validation | Sample-size and statistical power calculations for planning studies. Use whenever someone asks "how many subjects/samples/replicates do I need", wants an a priori power analysis,… |
| 133 | `statsmodels` | A | econometrics | Statistical models library for Python. Use when you need specific model classes (OLS, GLM, mixed models, ARIMA) with detailed diagnostics, residuals, and inference. Best for… |
| 134 | `sympy` | C | math | Use when you need exact symbolic math in Python — algebra, calculus, equation solving, symbolic linear algebra, or code generation via lambdify/LaTeX. Prefer NumPy or SciPy when… |
| 135 | `tiledbvcf` | D | excluded-lab | Efficient storage and retrieval of genomic variant data using TileDB. Scalable VCF/BCF ingestion, incremental sample addition, compressed storage, parallel queries, and export… |
| 136 | `timesfm-forecasting` | B | forecasting | Zero-shot time series forecasting with Google's TimesFM foundation model. Use for any univariate time series (sales, sensors, energy, vitals, weather) without training a custom… |
| 137 | `torch-geometric` | B | graph-ml | PyTorch Geometric (PyG) for graph neural networks — node/link/graph classification, message passing (GCN, GAT, GraphSAGE, GIN), heterogeneous graphs, neighbor sampling, and custom… |
| 138 | `torchdrug` | D | excluded-chemistry | PyTorch-native graph neural networks for molecules and proteins. Use when building custom GNN architectures for drug discovery, protein modeling, or knowledge graph reasoning.… |
| 139 | `transformers` | B | nlp-alt-data | Hugging Face Transformers for loading Hub models, running pipeline inference, text generation, and Trainer fine-tuning on NLP, vision, audio, and multimodal tasks. Use when… |
| 140 | `treatment-plans` | D | excluded-clinical | Generate concise (3-4 page), focused medical treatment plans in LaTeX/PDF format for all clinical specialties. Supports general medical treatment, rehabilitation therapy, mental… |
| 141 | `umap-learn` | B | regime-diagnostics | Use UMAP-learn for nonlinear dimensionality reduction, 2D/3D embeddings, clustering preprocessing, supervised or semi-supervised UMAP, DensMAP, AlignedUMAP, and Parametric UMAP… |
| 142 | `usfiscaldata` | A | macro-data | Query the U.S. Treasury Fiscal Data REST API for federal financial data. No API key required. Use for national debt (Debt to the Penny), Daily Treasury Statements, Monthly… |
| 143 | `vaex` | B | out-of-core-data | Use this skill for processing and analyzing large tabular datasets (billions of rows) that exceed available RAM. Vaex excels at out-of-core DataFrame operations, lazy evaluation,… |
| 144 | `venue-templates` | C | reporting | Access comprehensive LaTeX templates, formatting requirements, and submission guidelines for major scientific publication venues (Nature, Science, PLOS, IEEE, ACM), academic… |
| 145 | `what-if-oracle` | B | scenario | Run structured What-If scenario analysis with 4–6 branch possibility exploration (best, likely, worst, wild card, contrarian, second-order). Use when the user asks speculative… |
| 146 | `xlsx` | C | document-ingest | Create, edit, analyze, or convert Excel spreadsheets (.xlsx, .xlsm) where the workbook file is the primary deliverable. Use for formulas, formatting, financial models, multi-sheet… |
| 147 | `zarr-python` | B | array-storage | Chunked N-D arrays for cloud storage (Zarr-Python 3). Compressed arrays, parallel I/O, S3/GCS via fsspec, NumPy/Dask/Xarray compatible, for large-scale scientific computing… |
