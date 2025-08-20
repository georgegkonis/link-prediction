# Thesis Plan: Link Prediction on Text-Attributed Graphs
### Revision plan — from CascadeLP draft to evidence-first version

---

## 0. What stays, what changes — summary

| Element | Verdict | Why |
|---|---|---|
| DSAA 2023 as primary dataset | **Keep** | Real academic competition, checkable baselines, and the saturation phenomenon is your best research hook |
| CascadeLP tiered architecture (heuristics → POS/RF → Sentence-Transformer) | **Keep, reframed** | Sound engineering idea — but repositioned as a *diagnostic tool*, not a headline accuracy claim |
| SVM baseline | **Keep** | Legitimate, historically grounded (Al Hasan 2006), useful as a classic ML reference point |
| Abstract's claim "CascadeLP matches full-transformer accuracy" | **Remove until proven** | No experiments exist yet; this is currently an unearned claim |
| Citation [6] (Kim et al., arXiv:2401.04960) | **Remove or replace** | Could not verify this paper exists at this ID — likely fabricated |
| Citations [8], [10] titles | **Fix** | Real titles differ from what's in your bibliography (see §1) |
| "10,221 self-loop pairs" and other Section 1.3/2.6 specific numbers | **Re-derive from scratch, don't assume correct** | These read as placeholder numbers written before analysis — must be verified against actual data |
| Single-dataset scope | **Expand** | Add one comparison TAG dataset — required for the generalization question, and for the thesis's strongest contribution |
| "Beat the leaderboard" framing | **Replace** | Reframe around: why does DSAA saturate, does it leak, and does a near-perfect model generalize |

---

## 1. Citation fixes (do this first — it's fast and unblocks everything else)

Open every reference in your bibliography and check it against the primary source. Known issues so far:

- **[6] Kim et al., arXiv:2401.04960** — not found. Either locate the real paper you meant, or drop the claim it supports ("78% of new links involve a zero-common-neighbor node") until you find a real citation, or replace it with your own measurement on DSAA once you have the data (this is actually better — a number you measured yourself beats a borrowed one).
- **[8] Phan et al.** — correct title is *"Link Prediction for Wikipedia Articles as a Natural Language Inference Task"* (arXiv:2308.16469), not "Link prediction on text-attributed graphs using natural language inference."
- **[10] Tran et al.** — correct title is *"A Text-based Approach For Link Prediction on Wikipedia Articles"* (arXiv:2309.00317), not "Text-attributed graph link prediction via part-of-speech features and random forest."
- **All other references** — I have not independently verified [1]–[5], [7], [9], [11], [12]. Before submission, check each against Google Scholar/arXiv/ACM DL directly. Do not trust a bibliography entry just because it "looks right."

**Rule going forward:** every claim with a citation must be checked against the actual paper (abstract at minimum, ideally the relevant section) before it goes in the thesis. If you used an LLM to help draft related work, treat every citation it produced as unverified until you've opened the source yourself.

---

## 2. New central research question

> **Does near-perfect link prediction on the DSAA 2023 Wikipedia benchmark reflect genuine semantic reasoning, or does it reflect dataset-specific separability (or leakage) — and how much of that accuracy is actually needed, in a cost-aware sense, once trivial pairs are stripped out?**

This single question drives three components that were previously disconnected in the draft:
1. Dataset forensics (why does it saturate — leakage vs. separability)
2. Cross-dataset generalization (does it transfer)
3. Cost-aware cascade (what's the real marginal value of the expensive model, once you know which pairs are trivial)

---

## 3. Revised chapter plan

### Chapter 1 — Introduction
- Keep the motivation about TAGs and Wikipedia's relational structure — that part is fine.
- **Rewrite the "three observations" section (1.1)** to remove the unverified Kim et al. claim. Replace with what you can actually support: the documented near-perfect scores from Phan et al. and Tran et al. (real, verified), and the *open question* of why — not asserted as fact.
- **Rewrite Contributions (1.3)** to only list things you will actually have evidence for at submission. Suggested list:
  1. A train/test leakage audit of the DSAA 2023 dataset (exact pair overlap, reversed-pair overlap)
  2. A separability/difficulty characterization of the dataset (self-loops, common-neighbor distribution, text-similarity distribution for pos/neg pairs)
  3. A cross-dataset generalization test: train the best-performing text classifier on DSAA, evaluate zero-shot and fine-tuned on a second TAG dataset
  4. CascadeLP as a cost-aware diagnostic: quantify what fraction of pairs are resolved by cheap tiers, and characterize what remains in the "hard" residual
  5. An SVM + engineered-feature baseline, positioned as a classic-ML reference point throughout

### Chapter 2 — Background and Related Work
- Structure is fine (structural methods → semantic methods → hybrid architectures). Keep.
- Fix all citations (§1 above).
- **Add a subsection explicitly on evaluation integrity / leakage in link prediction benchmarks** — this is now load-bearing for your contribution, not a side note. Cite Yang, Chiang & Leskovec-style discussion of ogbl leakage issues, and Yao/Liben-Nowell type critiques of proximity-based leaderboards if available (verify before citing).
- **Add the second dataset here**, described alongside DSAA, so the comparison feels designed rather than bolted on later.

### Chapter 3 — Methodology
Reorganize into four concrete blocks:

**3.1 Dataset Forensics Protocol**
- Exact-match check: hash `(u,v)` pairs in train vs. test; report overlap count (expect 0, but check both orderings: `(u,v)` and `(v,u)`)
- Self-loop identification and count
- Common-neighbor count distribution for positive vs. negative training pairs
- Text-cosine-similarity (TF-IDF and Sentence-Transformer) distribution for positive vs. negative pairs
- Define a "trivial pair" criterion (e.g., self-loop, or common-neighbors > threshold, or text-similarity > threshold) and report what fraction of train/test falls into it

**3.2 CascadeLP (reframed as diagnostic architecture, not accuracy claim)**
- Keep Tier 0/1/2/3 structure as designed — it's good.
- Explicitly log, per pair, *which tier resolved it* and *what its difficulty label was* from 3.1 — this is what turns it from "we matched transformer accuracy" into "here's what fraction of the dataset actually required semantic reasoning."

**3.3 SVM/classic-ML baseline**
- Keep as designed (TF-IDF + SVM, POS + RF). Position explicitly as a reference point for "how much do you need PLMs at all" — ties into 3.1/3.2.

**3.4 Cross-dataset generalization protocol**
- Train best model (likely the Sentence-Transformer tier) on DSAA.
- Evaluate zero-shot on second dataset (no retraining).
- Evaluate after fine-tuning on second dataset's own training split.
- Report the delta — this is your generalization evidence.

### Chapter 4 — Experiments
Now has real content to slot in:
- 4.1 Datasets (DSAA + second dataset, with stats *you measured*, not copied)
- 4.2 Leakage audit results (from 3.1)
- 4.3 Separability characterization results (from 3.1)
- 4.4 CascadeLP tier-by-tier accuracy and call-rate results, broken down by difficulty tier
- 4.5 Cross-dataset generalization results
- 4.6 SVM baseline comparison across all of the above

### Chapter 5 — Analysis
- Ablations on cascade thresholds (as originally planned)
- **New:** analysis of what the "hard residual" (pairs that reach Tier 3 and are still uncertain) actually look like — read some of them qualitatively, characterize them
- Error analysis split by difficulty tier and by dataset

### Chapter 6 — Conclusion
- Only claim what chapters 4–5 actually show.
- Limitations section should explicitly name what you did *not* check (e.g., if you only test one second dataset, say generalization evidence is limited to that pair)
- Future work: additional datasets, GNN-based tiers, dynamic/temporal splits

---

## 4. Immediate next steps (in order)

1. **Fix citations** [6], [8], [10] and spot-check the rest (§1) — fast, unblocks credibility of everything else.
2. **Run the leakage audit** on the actual DSAA train.csv/test.csv (exact + reversed pair overlap). This is a 20-line script and settles the "were they accidentally given the test set" question empirically.
3. **Run the separability characterization** (self-loops, common-neighbor distribution, text-similarity distribution).
4. **Pick the second dataset** (Cora/PubMed for a fast pass, or CitationV8/ogbn-arxiv for closer scale/domain match to DSAA) and run the same forensics on it for comparability.
5. Only after 2–4: start writing Chapter 4 with real numbers, and only then write result-dependent parts of the Abstract/Intro/Contributions.

Once you have the leakage-audit and separability results in hand, we should revisit this plan — the findings there will determine how much weight the generalization experiment vs. the cascade experiment should carry in the final thesis.
