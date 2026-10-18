# Candidate Claims Novelty Verdicts

- **C1 (Transductive CV vs hidden test collapse)**: **Novel** [R07, R08, R09, R11].
  *Gap*: Prior work shows split leakage or negative transfer; none documents that complete held-out prompt blocks in CV provide near-oracle covariance causing large false gains (+0.06..+0.19) that collapse on sparse unseen test prompts.

- **C2 (API identity drift & silent aliasing)**: **Partially covered** [R03, R04, R05, R13, R14].
  *Gap*: Prior work detects benchmark drops or MMD shifts; none tracks longitudinal stylometric register drift (em-dash rates, syntax) across 18 frontier APIs with served-id verification and empirical aliasing logging (e.g. Gemini 3.7->3.8).

- **C3 (External data value & single-family prompt shortcuts)**: **Partially covered** [R02, R10, R12].
  *Gap*: Prior work shows topic confounding in AA; none analyzes the topic->label shortcut from single-family prompt coverage in external corpora, nor demonstrates hierarchical within-lab gating to harvest gains where flat training fails.

- **C4 (Cross-lab stylistic similarity & confusion clusters)**: **Novel** [R01, R15, R16, R17].
  *Gap*: Prior work shows isolated fingerprints or open weight-space lineage; none reveals empirical cross-lab stylistic confusion among black-box frontier LLMs where data-driven clusters cross lab lines (e.g. GLM with Claude, Kimi with GPT-5.5).
