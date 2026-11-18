# Selective classification, abstention, confidence scoring, and how to evaluate a reliability score

Method note: searches returned abstracts and metadata only. Full-text details (theorems, exact metric definitions) were not read, so such details are listed under Gaps or marked UNVERIFIED.

## 1. Key references and definitions

### Takeaway
The standard framework is: a score ranks samples, a threshold decides which to keep (coverage), and quality is the error rate among kept samples (selective risk) across all thresholds (risk-coverage curve, AURC). Failure prediction is judged by ranking (AUROC/AURC), not by whether the raw score value is calibrated.

### Cited Findings
- Geifman & El-Yaniv, "Selective Classification for Deep Neural Networks", NIPS 2017 (pp. 4878-4887). Turns a trained network into a selective classifier with a user-set risk level. Uses thresholded max-softmax ("softmax response", SR); MC-dropout is also usable as a selective score. Reported a guaranteed 2% top-5 ImageNet error with probability 99.9% at almost 60% coverage. — [arXiv](https://arxiv.org/abs/1705.08500), [NeurIPS PDF](https://papers.NeurIPS.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf)
- Hendrycks & Gimpel, "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks", ICLR 2017. Max softmax probability is a baseline: correct examples tend to have higher values than misclassified/OOD ones. Defines tasks and metrics for error/OOD detection; says the baseline can sometimes be beaten. — [arXiv](https://arxiv.org/abs/1610.02136), [MLAnthology](https://mlanthology.org/iclr/2017/hendrycks2017iclr-baseline)
- Corbiere et al., "Addressing Failure Prediction by Learning Model Confidence", NeurIPS 2019 (pp. 2902-2913). Proposes True Class Probability (TCP) as a better confidence target than Max Class Probability (MCP); the true label is unknown at test time so TCP is learned. Claims gains over MCP and Bayesian uncertainty. Reviewers noted the TCP and MCP coincide on training data that is fit almost perfectly, and the gain over a BCE baseline was modest (about 1-2.5% AUPR, as reported by the reviews page). — [NeurIPS abstract](https://papers.nips.cc/paper/2019/hash/757f843a169cc678064d9530d12a1881-Abstract.html), [reviews](https://papers.neurips.cc/paper_files/paper/2019/file/757f843a169cc678064d9530d12a1881-Reviews.html)
- Geifman, Uziel & El-Yaniv, "Bias-Reduced Uncertainty Estimation for Deep Neural Classifiers", ICLR 2019. Builds its performance measure from selective-prediction concepts; says confident-point uncertainty is biased by SGD dynamics. The search did not confirm that this paper names "AURC" explicitly. — [arXiv](https://arxiv.org/abs/1805.08206)
- Galil, Dabbah & El-Yaniv, "What Can We Learn From The Selective Prediction And Uncertainty Estimation Performance Of 523 ImageNet Classifiers?", ICLR 2023. Studies AUROC, ECE, AURC and coverage-for-selective-accuracy constraint across 523 classifiers; temperature scaling reported (via a third-party review) to improve AUROC and selective performance. — [ICLR page](https://iclr.cc/virtual/2023/poster/11783), [review](https://liner.com/review/what-can-we-learn-from-the-selective-prediction-and-uncertainty)
- Jaeger, Luth, Klein & Bungert, "A Call to Reflect on Evaluation Practices for Failure Detection in Image Classification", ICLR 2023 (oral). Benchmark of confidence scoring functions; simple softmax response baseline performed best overall; a review summary says it recommends AURC. — [arXiv](https://arxiv.org/abs/2211.15259v2), [ICLR session](https://iclr.cc/virtual/2023/session/13346)
- Conformal prediction: Vovk, Gammerman & Shafer, "Algorithmic Learning in a Random World", Springer 2005 (2nd ed. 2022) is the foundational monograph. — [Springer](https://link.springer.com/978-3-031-06648-1). Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification", arXiv 2107.07511 (2021, v6 2022): wraps any model to give sets with user-chosen coverage, non-asymptotic, distribution-free; also covers risk control such as FDR. — [arXiv](https://arxiv.org/abs/2107.07511). Romano, Sesia & Candes, "Classification with Valid and Adaptive Coverage", 2020: guaranteed marginal coverage, aims at approximate conditional coverage. — [arXiv](https://arxiv.org/abs/2006.02544)
- Ovadia et al., "Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift", NeurIPS 2019: in-distribution calibration/accuracy does not usually carry over to dataset shift; ensembles were most robust. — [arXiv](https://arxiv.org/abs/1906.02530)

### Inferences (standard definitions, from general knowledge, not read in the sources above; UNVERIFIED against text)
- Selective classifier = (f, g): predict f(x) if score g(x) >= tau, else abstain. Coverage = fraction accepted. Selective risk = error rate among accepted. Risk-coverage curve = selective risk vs coverage as tau varies. AURC = area under it (lower is better).
- AUROC for failure prediction = probability that a random correct sample scores higher than a random failed one. It depends only on ranking, so any monotone transform of the score leaves it unchanged. This is why ranking quality, not calibration of the raw value, is the right yardstick for "which scans to trust".
- AURC depends on both the base error rate and the ranking; compare it to the random-ranking line (flat at overall risk) and to the oracle curve. For the RRS-MBI case: the "risk" is localization error (or failure = LE > 15 mm), and "coverage" is the fraction of scans kept. A trivial baseline ("predict the centre") must also be run through the same curve.
- Conformal prediction is a different tool: it gives a guaranteed coverage set or risk bound under exchangeability. It would need a proper calibration split that is grouped by phantom (scans from one phantom are not exchangeable with scans from others). It is not a way to rank scores; use it only after the score is shown to carry signal.

### Gaps
- Exact AURC definition and AURC-vs-AUROC trade-off discussion from Jaeger et al. or Galil et al. were not read from full text.
- Exact statement of conformal risk control and exchangeability conditions in Angelopoulos & Bates were not confirmed (abstract only).
- Vovk et al. theorems not read.

## 2. Why a product of hand-normalized heuristic terms can fail or rank inversely

### Takeaway
I found no paper that studies product-form heuristic reliability scores directly. The mechanisms below are my own reasoning from general principles and are marked as inference. The closest cited support is that learned or simple scores often beat engineered ones.

### Cited Findings
- Simple softmax response beat many more elaborate confidence scoring functions in a large benchmark. — [Jaeger et al. 2023](https://arxiv.org/abs/2211.15259v2)
- The learned TCP target beat the MCP heuristic in the authors' experiments, although reviewers questioned the size of the gain. — [Corbiere et al. 2019](https://papers.nips.cc/paper/2019/hash/757f843a169cc678064d9530d12a1881-Abstract.html)
- Good calibration in-distribution does not carry over to dataset shift, so hand-set thresholds tuned on one set are fragile. — [Ovadia et al. 2019](https://arxiv.org/abs/1906.02530)
- Confidence from highly confident points is biased (SGD dynamics), an example of a score measuring a property of the method rather than correctness. — [Geifman et al. 2019](https://arxiv.org/abs/1805.08206)

### Inferences (all UNVERIFIED as published results; reasoning only)
- Error compounding / veto: in R = A*B*C*D, log R = sum of log terms. A term near 0 dominates the sum, so one badly behaved term decides the rank regardless of the others. If that term is anti-correlated with correctness, the product inherits the inversion.
- Saturation/clipping: clip(x, 0, 1) with a hand-picked cap maps all samples above the cap to the same value. This creates ties and destroys rank information precisely where many samples live. A saturation radius (R_sat 5 mm vs 20 mm) changed the repo's R results, which shows sensitivity to such a choice.
- Correlated components cancel: if two terms move together but enter with opposite sign (for example D rises with distance and 1-CC falls with it), the product can be flat or dominated by noise.
- Method-bias term: a component that measures where the reconstruction method tends to put its peak (central bias, shell ring) rather than whether it is right will rank by the method's artifact. In the repo log, D_norm and 1-CC were found inverted and R ranked inversely (rho +0.19 with LE).
- Product vs rank-based combination: rank averaging and a fitted logistic model have bounded sensitivity to any single term; the product does not.

### Gaps
- No published study on product-form scores found. A search aimed at that specific question was not run.

## 3. Why consistency / agreement-based measures work better

### Takeaway
Agreement across independent views estimates the variance part of the error and does not rely on a hand-set model of what a good image looks like. Ensembles and MC dropout are the standard published examples.

### Cited Findings
- Deep ensembles: Lakshminarayanan, Pritzel & Blundell, "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles", NIPS 2017. Simple, parallelizable, little tuning, high-quality uncertainty, shown on ImageNet. — [arXiv](https://arxiv.org/pdf/1612.01474)
- MC dropout: Gal & Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning", ICML 2016. Dropout at test time approximates Bayesian inference in deep Gaussian processes. — [arXiv](https://arxiv.org/pdf/1506.02142)
- Under dataset shift, ensembles (about 5 members) were robust and consistently strong across metrics. — [Ovadia et al. 2019](https://arxiv.org/abs/1906.02530)
- MC dropout was named by Geifman & El-Yaniv as a usable selective-prediction score. — [arXiv](https://arxiv.org/abs/1705.08500)

### Inferences
- Disagreement among K independent estimates is an estimate of the variance of the estimator. Large variance implies a likely large error; small variance does not guarantee correctness, because shared bias is invisible to agreement.
- For imaging: sub-aperture split-half, sub-band split, or bootstrap over antennas are the analogues of ensemble members. This matches the repo's finding that half-split and sub-band distances had the highest AUROC (about 0.59-0.60), though the CIs were close to 0.5.
- Limit: a method with a systematic bias (central pull) gives consistent but wrong answers across views, so agreement scores cap out.
- Deep ensembles/MC dropout papers are about neural nets; applying the same logic to beamforming sub-views is an analogy, not something these papers show.

### Gaps
- No paper found directly on split-half reliability or sub-aperture agreement for microwave imaging confidence. Not searched.
- Original exact quotes on the variance interpretation were not read from the full texts.

## 4. Statistical hygiene

### Takeaway
Use cluster (phantom-level) resampling, correct for the number of candidate scores, fit composites out-of-fold with phantom-grouped folds, and report AUROC with a CI. Choose thresholds on data not used to evaluate.

### Cited Findings
- Cluster bootstrap: Field & Welsh, "Bootstrapping clustered data", J. R. Statist. Soc. B 69(3), 369-390, 2007. Abstract: the cluster bootstrap gives consistent variance estimates under both the transformation and random-effect models, whereas the residual bootstrap is consistent only under the transformation model. — [ANU portal](https://researchportalplus.anu.edu.au/en/publications/bootstrapping-clustered-data/), [IDEAS](https://ideas.repec.org/a/bla/jorssb/v69y2007i3p369-390.html)
- Holm, "A Simple Sequentially Rejective Multiple Test Procedure", Scand. J. Statist. 6(2), 65-70, 1979. Rejects hypotheses one at a time until no more can be rejected. — [GESIS record](https://data.gesis.org/gesiskg/resource/zis-Holm1979A)
- Evaluation of failure detection is sensitive to protocol, motivating standard metrics and reflection on practice. — [Jaeger et al. 2023](https://arxiv.org/abs/2211.15259v2)

### Inferences (standard statistics, UNVERIFIED against the sources above)
- Scans from the same phantom are correlated, so resample phantoms (then scans within them) with replacement; scan-level iid bootstrap gives CIs that are too narrow. If the threshold or top-k selection is part of the procedure, redo it inside each bootstrap replicate.
- Multiplicity: with 6 candidates, Bonferroni uses alpha/6 (0.0083); Holm is uniformly more powerful and equally valid for any dependence. Equivalent for CIs: use 1 - alpha/6 intervals. In the repo notes, AUROC lower bounds of 0.509 and 0.514 at 95% would not survive this correction (stated in the repo log).
- Out-of-fold composites: fit weights/logistic coefficients on training phantoms, predict on held-out phantoms (GroupKFold), then compute AUROC on out-of-fold scores. Choosing which composite to report after looking at results is itself a candidate-selection step and needs the same correction, or a pre-registered choice.
- Thresholds: select tau (for example for 20% coverage) on a development set and report on a separate set; picking tau on the evaluation data inflates the apparent gain. Reporting the full risk-coverage curve avoids picking one tau.
- Always compare against a trivial predictor ("predict the centre") evaluated on the same accepted subset, because a score that selects easier cases (central tumours) lowers selective risk with no true reliability signal. The repo log found exactly this for composite A.
- Conformal calibration for this data should use phantom-grouped splits; treat exchangeability as violated across phantoms.

### Gaps
- Specific textbook citations for AUROC confidence intervals (DeLong, Hanley-McNeil), Bonferroni, and Benjamini-Hochberg were not searched; a search for Varoquaux & Colliot on AUC CIs returned nothing. Cite from standard texts after verifying.
- No source found on bootstrap re-selection inside replicates; that is my recommendation, not a cited result.
