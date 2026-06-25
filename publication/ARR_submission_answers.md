# ARR Submission Form Answers

**Target:** ARR 2026 May → EMNLP 2026
**Paper:** Accuracy, Stability, and Repeated-Run Reliability of Large Language Models on Deterministic Programming Tasks
**Submission PDF:** `main-anon.pdf`

---

## Basic Information

| Field | Answer |
|---|---|
| **Title** | Accuracy, Stability, and Repeated-Run Reliability of Large Language Models on Deterministic Programming Tasks |
| **Paper Type** | Long |

---

## Authors

| Name | Affiliation | Email |
|---|---|---|
| Yongxi Zhou | Northeastern University | zhou.yongx@northeastern.edu |
| Lai Yun Choi | Northeastern University | (to fill) |
| Jiaxi Wen | Northeastern University | (to fill) |

---

## Abstract

```
Run-level pass rate overstates retry-free coverage by up to 17.8 percentage points---and the gap is largest precisely for mid-performing systems. We investigate this accuracy--stability relationship in large language model (LLM) evaluation for deterministic text-conditioned generation, using programming tasks as a concrete testbed. Standard code-generation benchmarks emphasize single-run accuracy or eventual success under repeated sampling, but many deployment settings also require stability: consistent outcomes across repeated invocations under the same task description. We present a repeated-run evaluation protocol with metrics for run-level accuracy, retry-free coverage, and per-problem variability. On a recency-based benchmark of 100 LeetCode-style problems, we evaluate 16 models from five provider families under two prompt templates with five repeated runs per problem, yielding 16,000 evaluation instances. Although run-level pass rate and perfect stability rate are strongly correlated (r=0.985), pass rate consistently exceeds retry-free coverage---a gap that reaches 17.8 percentage points and reverses model rankings even among closely matched systems. Prompt effects are model-dependent rather than uniformly beneficial. These results suggest that repeated-run stability analysis is a necessary complement to conventional accuracy reporting for deterministic text-conditioned generation tasks.
```

---

## TL;DR

Run-level pass rate overstates retry-free coverage by up to 17.8 pp across 16 LLMs on 100 programming tasks; the gap peaks in mid-accuracy models and can reverse rankings even among closely matched systems.

---

## Keywords

LLM evaluation, code generation, repeated-run reliability, stability, benchmark

---

## Research Area *(single select)*

- [x] **NLP and Code Models** — best fit for a code-generation evaluation paper
  - Alternative: **Resources and Evaluation** if you want to emphasize the evaluation methodology angle
  - Alternative: **Machine Learning for NLP** if you want to emphasize the model analysis angle

> **Recommendation:** Choose **NLP and Code Models** — it most directly matches the paper's content.

---

## Research Area Keywords *(comma-separated)*

From the official ARR area keywords list (https://aclrollingreview.org/areas):

```
Evaluation Methodologies, Resources and Benchmarking, Large Language Models, Code Generation
```

---

## Contribution Types *(multi-select)*

- [x] **NLP engineering experiment** — 16,000 evaluation instances across 16 models
- [x] **Data resources** — curated 100-problem benchmark dataset
- [x] **Publicly available software and/or pre-trained models** — leetcode_evaluator framework
- [x] **Data analysis** — systematic analysis of accuracy--stability gap

---

## Languages Studied

```
Python (code generation), English (problem descriptions)
```

---

## Preprint & Preprint Status

| Question | Answer |
|---|---|
| **Would the authors like ARR to release a public anonymous preprint?** | **yes** |
| **Is there a publicly available non-anonymous preprint, or do you plan to release one?** | We plan to release a non-anonymous preprint in the next two months (i.e., during the reviewing process). |
| **Existing Preprints (URLs)** | *(leave blank until arXiv is posted)* |

> ARR allows posting to arXiv at any time. Posting during reviewing does NOT violate anonymity.

---

## Previous Submission (Resubmission Fields)

| Field | Answer |
|---|---|
| **Previous URL** | *(leave blank — this is a new submission)* |
| **Explanation Of Revisions PDF** | *(leave blank)* |
| **Justification For Author Changes** | *(leave blank)* |

| Field | Answer |
|---|---|
| **Reassignment Request: Area Chair** | This is not a resubmission |
| **Reassignment Request: Reviewers** | This is not a resubmission |
| **Justification For Not Keeping AE/Reviewers** | *(leave blank)* |

---

## Software & Data Upload

| Field | Action |
|---|---|
| **Software (.tgz/.zip, max 200MB)** | Optional. Can upload the `leetcode_evaluator` codebase. |
| **Data (.tgz/.zip, max 200MB)** | Optional. Can upload the `dataset/main-dataset.json` benchmark. |

> Both are optional at submission time. You can skip for now.

---

## Preferred Venue

| Field | Answer |
|---|---|
| **Preferred Venue** | **EMNLP** |

---

## Consent & License

| Field | Answer |
|---|---|
| **Consent To Share Data** (anonymized metadata for public dataset) | **yes** |
| **Consent To Share Submission Details** (with other venue PCs for compliance verification) | On behalf of all authors, we agree to the terms above to share our submission details. |
| **ACL Blind Submission License Agreement** | On behalf of all authors, I agree |
| **License** | **CC BY 4.0** |

---

## EMNLP 2026 AI Reviewing Experiment

| Field | Answer |
|---|---|
| **Do you want to opt-in to the EMNLP 2026 AI Reviewing Experiment?** | **no** (conservative choice — ensures human reviewers only) |

> Read details: https://2026.emnlp.org/ai-reviewing-experiment/

---

## Author Submission Checklist

| Field | Answer |
|---|---|
| **I confirm that this submission adheres to ARR requirements.** | **yes** |

> Reference: https://aclrollingreview.org/authorchecklist

---

## Responsible NLP Checklist

### A. General

| ID | Question | Answer |
|----|----------|--------|
| **A1** | Does the paper have a dedicated Limitations section? | **This paper has a limitations section.** (Section "Limitations", after Conclusion) |
| **A2** | Did you discuss any potential risks of your work? | **Yes** |
| **A2 Elaboration** | *(COMPULSORY IF YES)* | **Section: Ethics Statement.** The paper acknowledges that stronger coding capabilities can increase both beneficial and harmful uses, and frames the contribution as evaluation methodology rather than endorsement of autonomous deployment. |

### B. Scientific Artifacts

| ID | Question | Answer |
|----|----------|--------|
| **B** | Did you use or create scientific artifacts? (code, datasets, models) | **Yes** |
| **B4** | Did you discuss steps to check for personally identifying info or offensive content in the data, and steps to protect/anonymize it? | **Yes** |
| **B4 Elaboration** | *(COMPULSORY IF YES)* | **Section: Experimental Setup, subsection Problem Dataset.** The benchmark consists of algorithmic LeetCode problem specifications. We state: "No personal data: problems consist of algorithmic specifications only and contain no personally identifying information." |
| **B6** | Did you report relevant statistics (number of examples, train/test/dev splits, etc.) for the data? | **Yes** |
| **B6 Elaboration** | *(COMPULSORY IF YES)* | **Section: Experimental Setup.** N=100 problems (20 Easy, 50 Medium, 30 Hard). No train/test split — this is an evaluation benchmark, not a training dataset. All 100 problems are used exclusively for evaluation. |

### C. Computational Experiments

| ID | Question | Answer |
|----|----------|--------|
| **C** | Did you run computational experiments? | **Yes** |
| **C2** | Did you discuss the experimental setup, including hyperparameter search and best-found hyperparameter values? | **Yes** |
| **C2 Elaboration** | *(COMPULSORY IF YES)* | **Section: Experimental Setup, subsection Repeated-Run Methodology.** Temperature=0.3, top-p=0.9, R=5 repeated runs, max_tokens=4096 (with noted exceptions for reasoning models). No hyperparameter search was performed — fixed decoding parameters were used for all models as the goal is to characterize variability under realistic stochastic decoding, not to optimize per-model performance. |
| **C3** | Did you report descriptive statistics about your results (error bars, summary statistics, transparent about max/mean/single run)? | **Yes** |
| **C3 Elaboration** | *(COMPULSORY IF YES)* | **Section: Results.** All RLPR and PSR values are point estimates; we state that 95% Wilson score intervals and bootstrap CIs are available. AV is explicitly shown with bootstrap CIs. We are transparent that all numbers are run-level averages across R=5 runs, not single-run or max. |

### D. Human Subjects

| ID | Question | Answer |
|----|----------|--------|
| **D** | Did you use human annotators or research with human subjects? | **No** |
| **D1** | Did you report full text of instructions given to participants? | **N/A** |
| **D2** | Did you report recruitment and payment information? | **N/A** |
| **D3** | Did you discuss consent from people whose data you're using? | **N/A** |
| **D4** | Was the data collection protocol approved by an ethics review board? | **N/A** |

### E. AI Assistants

| ID | Question | Answer |
|----|----------|--------|
| **E** | Did you use AI assistants (ChatGPT, Copilot, etc.) in your research, coding, or writing? | **Yes** |
| **E1** | If you used AI assistants, did you include information about their use? | **Yes** |
| **E1 Elaboration** | *(COMPULSORY IF YES)* | **Section: Acknowledgments.** "We thank Claude (Anthropic) for assistance in polishing the writing of this paper." AI assistance was limited to light copy-editing and prose polishing. All research design, experiments, data analysis, figures, tables, and intellectual contributions are entirely the authors' own work. |

---

## Quick Reference: All Required Fields Summary

| # | Field | Type | Answer |
|---|-------|------|--------|
| 1 | Title | text | Accuracy, Stability, and Repeated-Run Reliability of LLMs on Deterministic Programming Tasks |
| 2 | Authors | search | Yongxi Zhou, Lai Yun Choi, Jiaxi Wen (all Northeastern) |
| 3 | TL;DR | text | Run-level pass rate overstates retry-free coverage by up to 17.8 pp... |
| 4 | Abstract | textarea | *(see full abstract above)* |
| 5 | PDF | upload | main-anon.pdf |
| 6 | Paper Type | radio | Long |
| 7 | Keywords | text | LLM evaluation, code generation, repeated-run reliability, stability |
| 8 | Research Area | dropdown | NLP and Code Models |
| 9 | Research Area Keywords | text | Evaluation Methodologies, Resources and Benchmarking, Large Language Models |
| 10 | Contribution Types | multi-checkbox | NLP engineering experiment, Data resources, Publicly available software, Data analysis |
| 11 | Languages Studied | text | Python, English |
| 12 | Previous URL | text | *(blank)* |
| 13 | Reassignment Area Chair | radio | This is not a resubmission |
| 14 | Reassignment Reviewers | radio | This is not a resubmission |
| 15 | Preprint (ARR anonymous) | radio | yes |
| 16 | Preprint Status | radio | We plan to release a non-anonymous preprint in the next two months |
| 17 | Preferred Venue | dropdown | EMNLP |
| 18 | Consent To Share Data | radio | yes |
| 19 | Consent To Share Submission Details | radio | I agree |
| 20 | A1 Limitations | checkbox | checked |
| 21 | A2 Risks | radio | Yes |
| 22 | A2 Elaboration | text | Section: Ethics Statement |
| 23 | B Scientific Artifacts | radio | Yes |
| 24 | B4 PII Check | radio | Yes |
| 25 | B4 Elaboration | text | Section: Experimental Setup |
| 26 | B6 Statistics | radio | Yes |
| 27 | B6 Elaboration | text | Section: Experimental Setup |
| 28 | C Computational Experiments | radio | Yes |
| 29 | C2 Hyperparameters | radio | Yes |
| 30 | C2 Elaboration | text | Section: Repeated-Run Methodology |
| 31 | C3 Descriptive Stats | radio | Yes |
| 32 | C3 Elaboration | text | Section: Results |
| 33 | D Human Subjects | radio | No |
| 34 | D1–D4 | radio | N/A (hidden when D=No) |
| 35 | E AI Assistants | radio | Yes |
| 36 | E1 Info About Use | radio | Yes |
| 37 | E1 Elaboration | text | Section: Acknowledgments |
| 38 | Author Checklist | checkbox | yes |
| 39 | License Agreement | radio | On behalf of all authors, I agree |
| 40 | EMNLP AI Reviewing Experiment | radio | no |
| 41 | License | dropdown | CC BY 4.0 |
