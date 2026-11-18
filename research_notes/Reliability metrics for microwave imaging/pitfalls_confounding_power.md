# Pitfalls for per-scan reliability scores in DAS-based microwave breast imaging: central bias, clutter, confounding by depth/size, statistical power

Method note: about 17 tool calls. Earlier notes in `research_notes/RRS MBI reliability score fixes/` (three files) were read first and are not repeated; their open UNVERIFIED claims were re-checked where a primary source could be opened. Many full texts were NOT openable (PMC returned a bot check for PMC5017770 and one retry; arXiv 2409.19517 returned raw PDF bytes; Kirshin EuCAP 2011 failed DNS). Items seen only as a search snippet or abstract are marked UNVERIFIED. Items marked "SIM (mine)" are my own Monte Carlo computations (numpy, seed fixed), not literature. Items marked "general knowledge" are standard statistics stated from memory, not read in a source here.

Headline for the writer: the literature I could open gives mostly QUALITATIVE statements about wrong-peak causes (wrong permittivity, skin residual, heterogeneity) and almost no mm-per-unit sensitivities. I found no paper that reports a trivial "predict the centre" baseline, no paper that studies localization error versus radial position in UM-BMID Gen-2, and no paper that validates a no-ground-truth per-scan score against localization error. The statistics part (clustered AUROC, multiplicity, forking paths) is well supported by standard references plus a simulation matched to this dataset's size.

---

## Q1. Documented causes of wrong-peak behaviour in DAS microwave imaging, with quantitative values

### Takeaway
Papers consistently name five causes: skin/shell reflection and imperfect reference subtraction, wrong (constant) permittivity, tissue heterogeneity/dense breasts, antenna/geometry modelling errors, and weak deep responses. Only a few give numbers, and none that I could open gives mm-of-error per unit of epsilon or per ps of delay. The central-bias mechanism is still not stated as such in any source I opened. Residual clutter near the phantom centre is reported by Reimer 2020.

### Cited Findings
- Skin reflection dominates unprocessed focused images and completely masks tumour signals; tumours near the skin are hardest because skin clutter masks them; background (reference) subtraction removed skin clutter for a 4 mm tumour in one phantom — search summaries of Brunel (Zurich/Brunel pdfs: [bura.brunel.ac.uk C12_Zurich](https://bura.brunel.ac.uk/bitstream/2438/2662/1/C12_Zurich.pdf), [Brunel numerical analysis](https://bura.brunel.ac.uk/bitstream/2438/2614/1/Numerical%20analysis%20of%20microwave%20detection%20of%20breast%20tumours%20using%20synthetic%20focussing%20techniques.pdf)). UNVERIFIED: only snippet seen, not full text.
- Reimer, Solis-Nepote, Pistorius (2020), Diagnostics 10(6):411 (full text opened): tumour analogs were at antenna height with up to 10 mm position uncertainty; a single average propagation speed was used (adipose permittivity 6.4); no sensitivity analysis to propagation speed was reported (listed as future work); strong clutter sits near the air-tissue boundary and near the phantom centre, attributed to central fibroglandular tissue; in Class II and III the tumour response was not significantly above this central clutter — [PMC7345629](https://pmc.ncbi.nlm.nih.gov/articles/PMC7345629/)
- Same paper, SCR (dB) by density class, DAS: Class I 1 ± 2, II 0 ± 1, III 1 ± 1, IV 0 ± 2; localization error DAS/DMAS 11 ± 4 mm, no significant difference between the four algorithms; only 1 of 4 Class IV reconstructions had an identifiable tumour; the authors note a trend of decreasing SMR in denser breasts but call the sample small — same source. SCR near 0 dB means the global max is about as likely to be clutter as tumour.
- Moll, Kelly, Byrne, Sarafianou, Krozer, Craddock (2014), Int. J. Biomed. Imaging, "Microwave Radar Imaging of Heterogeneous Breast Tissue Integrating A Priori Information" (full text opened): with a constant average permittivity, DAS "fails" to localize the tumour in the dense heterogeneous phantom; with a 3D permittivity map the SCR was about 20 dB (Class-4). For Class-2 the localization error was 5.0 mm (their method) versus 13.1 mm for both the channel-ranked beamformer and the coherence-factor method. A global permittivity bias gave ambiguous detection (normalized peak-to-mean 0.529 versus 1.0 for the best case). The paper gives NO numeric localization error for constant-permittivity DAS and the bias sign is inconsistent between text (-1) and table (+1) — [PMC4243481](https://pmc.ncbi.nlm.nih.gov/articles/PMC4243481/)
- Lavoie, Okoniewski, Fear (2016), PLOS ONE, "Estimating the Effective Permittivity for Reconstructing Accurate Microwave-Radar Images" (full text opened via PLOS): incorrect permittivity moves the peaks off the scatterers and makes responses weaker and broader, but the paper gives no focal-shift in mm; in one 3D model the estimated tumour sat 6 mm further from the chest wall than the true location, called a slight discrepancy. Estimated interior permittivities in the examples span 2.1 to 7.8 (versus nominal 2.5 to 4.64 in simulation, 6.4 in one experiment), i.e. the estimator itself can be off by roughly 0.4 to 2 permittivity units in the cases they list; skin permittivity was consistently underestimated but has a much smaller effect than the interior value — [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0160849)
- Estimating a patient-specific average permittivity from focal-quality metrics is an active line because incorrect estimation hurts image quality and sensitivity: O'Loughlin et al. 2017 Sensors (focal quality metrics as fitness functions), EuCAP 2019 (first analysis of how realistic dielectric estimation affects sensitivity), Sensors 2021 (first application to patient images, 5 cases) — [Galway record](https://researchrepository.universityofgalway.ie/entities/publication/5915aeb8-f6e0-43d8-a60e-388e82a97c38), [EuCAP 2019 record](https://edepositireland.ie/handle/2262/102065). UNVERIFIED: abstracts/snippets only; no sensitivity numbers retrieved.
- Kirshin et al., EuCAP 2011 (McGill): DAS and DMAS give small localization error in the generated images but fail at high breast tissue density — snippet only, [PDF](https://www.tsp.ece.mcgill.ca/Networks/projects/pdf/Kirshin_EUCAP2011.pdf). UNVERIFIED (fetch failed).
- 2D reconstruction uses only in-plane data, so a target lying between two imaging slices may be located inaccurately; 3D reconstruction uses cross-plane measurements — search summary of a 3D breast-imaging paper (PMC3774049 listed in results; exact paper not opened). UNVERIFIED, no mm figure.
- Time-delay imprecision and its effect on image contrast in confocal microwave imaging is studied in [arXiv 2409.19517](https://arxiv.org/pdf/2409.19517); the abstract discusses contrast, not localization; full PDF could not be parsed here. UNVERIFIED, no numbers.
- Circular SAR/array point-spread functions are spatially varying, and a small radius with a wide antenna beamwidth gives sidelobes that degrade the image — [Muppala et al., IEEE TMTT 2025, via Fessler site](https://web.eecs.umich.edu/~fessler/papers/lists/files/jour/25/muppala-25-fic.pdf); resolution varies across the scene and closed forms are geometry dependent so they are evaluated numerically — [arXiv 1704.03303](https://arxiv.org/pdf/1704.03303). Both are snippet-level and not breast specific. UNVERIFIED relevance.
- Attenuation compensation in breast radar imaging is described as challenging and typically unstable; a regularized nonlinear compensation improved localization and resolution in simulated and experimental data — Kazemi and Fear, IEEE Open J. Signal Processing 2023 (snippet via [DOAJ](https://doaj.org/article/2e5137bc68a94478b3f6682000c49223)). UNVERIFIED, no numbers.
- Tumour phantoms with 15 mm diameter and MRI-derived position mismatch between MR and radar scans are described as a partial cause of position error in patient data (no mm) — Lavoie 2016 above.

### Inferences
- Sensitivity to assumed permittivity, derived by me from geometry (NOT from a paper): the one-way phase velocity is c/sqrt(er). For a tumour at distance d from an antenna, the two-way delay is 2 d sqrt(er)/c. A relative error in sqrt(er) of delta gives a delay error of about 2 d sqrt(er) delta/c. With er = 6.4, d = 0.1 m, an error of 1 unit of er (about 6 % in sqrt(er), i.e. delta about 0.06) gives roughly 2*0.1*2.53*0.06/3e8 = about 100 ps. At 4 GHz one period is 250 ps, so 100 ps is 0.4 cycle: enough to degrade coherence at the upper band, while the focal shift is a fraction of the aperture-averaged geometry rather than a fixed mm value. A 100 ps error equals about 6 mm of one-way path in er = 6.4 (c*100e-12/2.53 = 11.9 mm two-way, 5.9 mm one-way). This is back-of-envelope; no source gives mm per epsilon.
- Because all channels share roughly the same wrong speed, a permittivity error scales delays toward or away from the array centre, i.e. it shifts the best-focus radially and degrades the PSF; it does not by itself create a centre-peaked image. A centre peak needs a time-aligned residual (direct path, ring-symmetric shell echo, central fibroglandular clutter as Reimer 2020 observed).
- Reimer 2020 is the only opened source that ties a central maximum to a physical object (central fibroglandular tissue in the phantom). In the Gen-2 data used here the 'central clutter' is more likely residual shell echo with a centre-aligned delay; the literature does not confirm this.
- Tumour out of the antenna plane (3D versus 2D): a 2D DAS with a tumour offset by dz from the antenna plane sees a path length increased by sqrt(r^2 + dz^2) - r. For a 15 mm offset and r = 100 mm that is about 1.1 mm (small), but for tumours in the near field (r = 30 mm) it is about 7 mm of one-way path. Offsets matter most for tumours near the antennas and for the far-off-plane responses, which appear weaker (antenna pattern). This is my geometry only; I found no study quantifying it for DAS breast imaging.

### Gaps
- No published mm-per-permittivity-unit or mm-per-ps localization sensitivity for DAS breast imaging was found. The two papers most likely to have one (Lavoie 2016 full body figures, O'Loughlin 2017/2019) were not read beyond snippets/abstract-level text; Lavoie 2016 states the effect qualitatively.
- No source states the central bias of DAS on a ring array as a property of the method. The earlier note's mechanism remains an inference; the PSF-by-simulation test it proposes is still the right way to settle it.
- No quantitative study of 3D versus 2D out-of-plane localization error was found.
- The phase-centre and radius error sensitivity for UM-BMID is only in the project's own audit.

---

## Q2. Localization error versus tumour depth/position, breast size and density in phantom studies (UM-BMID and others)

### Takeaway
I found NO study that reports localization error as a function of radial position, depth or breast size for UM-BMID or another open phantom dataset. The nearest verified evidence is that SCR/SMR drop with breast density (Reimer 2020) and that DAS fails in dense heterogeneous cases. The project's own radial analysis therefore has no published counterpart to compare against.

### Cited Findings
- Reimer 2020 explicitly leaves a multivariate analysis of contrast versus tumour position out of scope; localization error is not related to density, position or depth; the density effect is on SCR/SMR only — [PMC7345629](https://pmc.ncbi.nlm.nih.gov/articles/PMC7345629/)
- Reimer's thesis reports a CNN AUC of 76 to 91 %, the upper end only when test phantom breast volumes are within the training bounds and the tumour is at the antenna height — [mspace.lib.umanitoba.ca hdl 1993/34970](https://mspace.lib.umanitoba.ca/handle/1993/34970), as cited in the earlier note (not re-read).
- The UM-BMID Gen-3 physics-driven AI paper uses a tolerance of tumour radius + 0.5 cm, with 0.5 cm the phantom position uncertainty, and does not break down error by depth — [arXiv 2401.02715](https://arxiv.org/pdf/2401.02715)
- Reporting practice: a 2023 scoping review of 184 papers states that most image analysis in breast microwave sensing is qualitative, that quantitative metrics mainly describe contrast, and recommends consistent metric definitions plus resolution, noise and artifacts; image-based sensitivities of 63 to 100 % in eleven trials, only four articles estimated specificity (20 to 65 %) — Reimer & Pistorius, Sensors 23(11):5123, [PMC10255334](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10255334/) (abstract/snippet level). UNVERIFIED in detail.
- Healthy-breast false-positive rates have not been systematically addressed in earlier MWBI studies, per a Wavelia preprint snippet — [preprints.org 202508.0289](https://www.preprints.org/manuscript/202508.0289). UNVERIFIED, preprint, snippet only.
- Coherence-based beamformers often put the dominant response in a place inconsistent with clinical reports and missed 2 of 5 patients (CF-DAS) — Elahi et al. 2018 (cited in earlier notes; [PMC6022049](https://pmc.ncbi.nlm.nih.gov/articles/PMC6022049/)); not re-read.

### Inferences
- With Reimer's 18 reconstructions as the only controlled localization numbers (11 ± 4 mm conditional on SCR > 0), the project's unconditional 30 mm error cannot be compared to the 11 mm. A like-for-like comparison would restrict to scans whose predicted peak lies inside the breast and SCR > 0 (point already made in the earlier note).
- Confounding by breast size is plausible but undocumented: in UM-BMID the antenna radius is fixed (21-23 cm) while the shell radius varies across the nine adipose shell designs, so tumour-to-shell distance and tumour-to-antenna distance covary with size. Any score that depends on the peak radius is therefore confounded by size and depth jointly. This is my reasoning; no source tests it.

### Gaps
- Localization error versus radial position for UM-BMID Gen-2 DAS: not found (search returned only the Gen-3 AI paper, Reimer 2020, and the Prykhodko MSc thesis, whose chapter list I saw but did not read).
- Breast size/volume effect on DAS localization error: not found.

---

## Q3. Are coherence/contrast confidence indices confounded by depth or size, and how do authors control for it?

### Takeaway
There is documented concern that coherence metrics reward non-lesion regions (Elahi 2018, from the earlier note), and that clutter near the skin or centre can exceed the tumour response (Reimer 2020). I found NO paper that formally shows a confidence index being confounded by depth or size, and almost nothing on depth/time-gain normalization in radar breast imaging; attenuation compensation is reported as hard and unstable.

### Cited Findings
- Attenuation compensation in breast microwave imaging is typically unstable; regularized nonlinear compensation improved localization and resolution — Kazemi & Fear 2023, IEEE OJSP (snippet via [DOAJ](https://doaj.org/article/2e5137bc68a94478b3f6682000c49223)). UNVERIFIED.
- Gain compensation to counteract microwave attenuation was used in thermoacoustic imaging, an adjacent modality — Wang 2001 (Caltech), [PDF](https://authors.library.caltech.edu/74037/1/Wang_2001p4.pdf). Snippet only, not breast radar. UNVERIFIED.
- Skin removal methods used as background/clutter controls: ideal skin suppression by subtracting the adipose-only phantom scan, rotation subtraction (relies on skin response being similar at neighbouring antenna positions), neighbourhood skin subtraction — [Reimer 2020 PMC7345629](https://pmc.ncbi.nlm.nih.gov/articles/PMC7345629/) (adipose-only subtraction, residual clutter still notable), [Brunel/Elahi snippets above](https://bura.brunel.ac.uk/bitstream/2438/2662/1/C12_Zurich.pdf). Rotation subtraction wording is from a search summary; UNVERIFIED.
- Reimer 2020 notes residual mismatch after adipose-only subtraction still produces notable clutter, especially in DAS and less in DMAS — [PMC7345629](https://pmc.ncbi.nlm.nih.gov/articles/PMC7345629/)
- Focal-quality metrics (O'Loughlin 2017) are used as fitness functions for estimating average permittivity; these are image-sharpness measures, so they are themselves image-property scores that depend on the permittivity assumption — [Galway record](https://researchrepository.universityofgalway.ie/entities/publication/5915aeb8-f6e0-43d8-a60e-388e82a97c38). Relevance as a confounding example is my inference.

### Inferences
- The usual defences in the literature are per-image (reference subtraction, skin removal), not per-score. No paper normalizes a quality score by the distribution of the same score on target-free scans at the same radius. That is exactly what the project's adipose-only scans (141 references) allow, and it would be a design contribution rather than a replication of published practice.
- The design rules that follow are in Q6.

### Gaps
- No paper found on depth/time-gain normalization specific to radar breast imaging quality metrics. No paper found that tests a coherence-type score against depth.
- Per-pixel normalization by an adipose-only reference image appears in practice as reference subtraction, not as division; I found no source that divides by a reference-image map.

---

## Q4. Statistical power: AUROC CI width, minimum detectable AUROC, multiplicity, forking paths

### Takeaway
With about 80 positives, 420 negatives and about 66 phantoms, a 95 % CI on AUROC is about 0.13 to 0.16 wide (plus/minus 0.065 to 0.08); AUROC near 0.58 to 0.60 cannot be distinguished from 0.5 once six candidates are screened. The minimum detectable AUROC for a single pre-specified score (two-sided alpha 0.05, 80 % power) is about 0.60 to 0.62; after screening K = 6 candidates the max-null 95th percentile alone reaches about 0.58 to 0.59.

### Cited Findings
- Hanley & McNeil (1982), Radiology 143:29-36, give closed-form SE for AUC: with Q1 = A/(2-A), Q2 = 2A^2/(1+A), SE = sqrt{[A(1-A) + (n1-1)(Q1-A^2) + (n0-1)(Q2-A^2)]/(n1 n0)}; later work found it can underestimate minimum sample size — [UC Davis copy](https://math.ucdavis.edu/~saito/data/roc/hanley-mcneil-auc.pdf), formula from search summary (UNVERIFIED against the PDF text; I did not open it).
- DeLong, DeLong, Clarke-Pearson (1988), Biometrics 44(3):837-845: nonparametric covariance of correlated AUCs via generalized U-statistics (paired comparison of AUCs from the same cases) — [Duke record](https://scholars.duke.edu/publication/712661). Abstract-level.
- Obuchowski (1997), "Nonparametric analysis of clustered ROC curve data", Biometrics 53:567-578, doi:10.2307/2533958: extends DeLong's structural components to clustered data using the design-effect/effective-sample-size ideas of Rao and Scott; does not require specifying the intra-cluster correlation structure; the estimator is the ordinary Mann-Whitney statistic with a cluster-corrected variance. Yougui Wu (2020, Comm. Stat.) argues the estimator is not efficient because it weights all pairwise rankings equally and proposes weighted alternatives — [Cleveland Clinic clusteredROC help](https://www.lerner.ccf.org/quantitative-health/documents/clusteredROC_help.pdf), [IDEAS Wu 2020](https://ideas.repec.org/a/taf/lstaxx/v49y2020i6p1446-1461.html). Snippet-level.
- Ying, Maguire, Glynn, Rosner (2021/22), Ophthalmic Epidemiology 29(2):117-127 (opened): ignoring within-cluster correlation gives narrower AUC CIs and overstates precision; the size of the underestimation depends on the correlation: with low correlation (kappa 0.23) naive width 0.149 versus clustered 0.156 versus bootstrap 0.152; with high correlation (kappa 0.98) naive 0.066 versus 0.093 (Obuchowski-type) and 0.090 (cluster bootstrap), i.e. naive about 30 % too narrow; cluster bootstrap resamples subjects and includes all their eyes; 1000 replications generally acceptable — [PMC8586066](https://pmc.ncbi.nlm.nih.gov/articles/PMC8586066)
- Field & Welsh (2007), JRSS-B 69(3):369-390: the cluster bootstrap is consistent under both transformation and random-effect models (cited in the earlier note) — [IDEAS](https://ideas.repec.org/a/bla/jorssb/v69y2007i3p369-390.html)
- Benjamini & Hochberg (1995), JRSS-B 57(1):289-300: step-up procedure controlling the false discovery rate, proved for independent test statistics; gains power over FWER control — [citation page](https://planosphere.stowers.org/pub/197), doi:10.1111/j.2517-6161.1995.tb02031.x. Holm (1979) Scand. J. Statist. 6(2):65-70 step-down FWER control (earlier note: [GESIS](https://data.gesis.org/gesiskg/resource/zis-Holm1979A)).
- Gelman & Loken (2013, "The garden of forking paths"): even a researcher who runs a single analysis can generate false positives when analysis choices (data exclusion, which comparison, which main effect/interaction) are made contingent on the data; the problem is not deliberate fishing; the responses discussed are pre-registration and multiverse analysis (Steegen et al. 2016) — [summary via Wikipedia "Forking paths problem"](https://en.wikipedia.org/wiki/Forking_paths_problem), [arXiv 1910.13602 listed in results](https://arxiv.org/pdf/1910.13602v2.pdf). I did not open the original PDF at stat.columbia.edu. UNVERIFIED wording.

### SIM (mine): clustered AUROC CI width and null distribution for this dataset's size
Setup (my own simulation, numpy; 66 clusters of about 7.6 scans, N = 504, 80 positives chosen as the top 80 of a latent propensity that may be cluster-correlated; score shift delta = 0.3 sd for positives; random cluster intercept in the score with variance share rho_b). 40 replicates per cell for CI widths, 1000 to 2000 for null distributions. Hanley-McNeil (HM) and cluster-bootstrap (cb) widths are for a 95 % interval.

| outcome cluster corr rho_y | score cluster corr rho_b | mean AUC | HM 95 % width | cluster-bootstrap width |
|---|---|---|---|---|
| 0 | 0 | 0.587 | 0.140 | 0.130 |
| 0 | 0.2 | 0.583 | 0.140 | 0.136 |
| 0 | 0.5 | 0.589 | 0.140 | 0.133 |
| 0.3 | 0 | 0.576 | 0.140 | 0.133 |
| 0.3 | 0.2 | 0.578 | 0.140 | 0.144 |
| 0.3 | 0.5 | 0.593 | 0.140 | 0.161 |

- Null (true AUC 0.5) sampling SD of a single AUROC: 0.034 (no clustering), 0.040 (rho_y = rho_b = 0.3), 0.044 (rho_y 0.3, rho_b 0.5). Two-sided 2.5/97.5 % null quantiles: [0.431, 0.566], [0.423, 0.580], [0.416, 0.583].
- Minimum detectable AUROC (single pre-specified score, two-sided alpha 0.05, 80 % power, normal approximation 0.5 + 2.8 SD): about 0.595 (SD 0.034) to 0.62 (SD 0.044). This ignores the alternative variance difference, so treat it as approximate.
- Max of K = 6 independent-ish null candidates: mean 0.545 to 0.550, 95th percentile 0.584 (no clustering) to 0.592 (rho 0.3). So a best-of-six AUROC of 0.585 to 0.59 is expected by chance about 1 time in 20 even with no signal. Candidates correlated with each other would give a smaller inflation; independent ones the larger one.
- Consistency check: the project's logged AUROCs, 0.585 [0.509, 0.665] and 0.602 [0.514, 0.682], have widths 0.156 and 0.168, matching the simulated cluster-bootstrap widths of 0.13 to 0.16 at moderate correlation. So the log is plausible, not too narrow.

### Inferences
- Design effect (general knowledge, Kish): DEFF = 1 + (m - 1) rho with average cluster size m about 7.6. If rho = 0.2 for the score within phantom, DEFF is about 2.3 and the effective n is about 220 of 504, but my simulation shows that the CI for AUROC widens much less than the DEFF would suggest when the outcome is only weakly clustered, because AUROC compares scans across clusters. Clustering matters most when positives concentrate in a few phantoms (rho_y large) AND the score has a phantom-level component (rho_b large): width rose from 0.13 to 0.16 (about 20 %).
- Holm adjusted lower bounds: with K = 6 two-sided, each 95 % interval should be a 1 - 0.05/6 = 99.17 % interval; the SE is about 0.04 so the z multiplier grows from 1.96 to about 2.64 and the half-width from about 0.08 to about 0.105. Lower bounds of 0.509 and 0.514 then fall below 0.5 (consistent with the project log).
- Forking paths risk is concrete here: the candidate list (6 scores), the composite (A rank-mean versus B logistic), the success threshold (15 mm), the coverage level (20 %), the saturation radius (5 versus 20 mm), the sign of each term, and the choice of region all offer degrees of freedom. The project's own earlier R_sat 5 versus 20 mm change is an example of one choice changing results (R mean 0.162 versus 0.085).
- A low AUROC with a CI that excludes 0.5 on the wrong side (R at 0.434, inverted) is itself informative: a score meant to be positively related to accuracy that ranks inversely invalidates the construct, which a pre-specified sign makes testable.

### Recommended practice (derived from the sources above; synthesis is mine)
1. Pre-specify before looking at outcomes: one primary score, its sign, one primary endpoint (for example AUROC for LE <= 15 mm), the 95 % CI method (phantom-level cluster bootstrap, B >= 1000, with any threshold or composite fitting redone inside each replicate), the analysis population (all 504 scans, no post-hoc exclusions).
2. Treat all other scores as secondary with Holm (FWER) or Benjamini-Hochberg (FDR) correction across them; report both raw and adjusted intervals.
3. Fit any composite out-of-fold with phantom-grouped folds and choose its form before seeing held-out results; report a held-out set of phantoms never used for design.
4. Report the trivial baseline in the same tables (see Q5) and the conditional AUROC within strata (tumour radial tertile, shell-radius tertile) to guard against a score that just encodes position.
5. Report power honestly: an AUROC under about 0.60 with these n cannot be separated from 0.5; state the minimum detectable effect.

### Gaps
- I did not open Obuchowski 1997 or Hanley-McNeil full text; formulas for the clustered variance are not reproduced here (only the idea and the Ying et al. numbers).
- I found no source on the effective rho for localization-success outcomes in phantom groups; the simulation values are assumptions, not estimates from this dataset. The actual within-phantom correlation can be estimated from the project data (ICC of LE <= 15 success per phantom) and the simulation re-run.
- Pre-registration practice specifically for imaging/biomarker score development (for example STARD, TRIPOD-AI) was not searched; cite from standard guidelines after verifying.
- Rutter (clustered ROC), mentioned in the assignment, was not searched.

---

## Q5. Is a trivial "always predict the array centre" baseline reported, and how should performance be benchmarked?

### Takeaway
I found no microwave breast imaging paper that reports a trivial constant-location (centre) baseline. Reporting norms are weak: most analysis is qualitative, specificity is rarely estimated, and performance numbers are conditional on identifiable subsets. The project's trivial-centre comparison (29.2 mm / 12.5 % versus pipeline 30.9 mm / 16.7 %) is therefore a methodological addition, not a replication of published practice.

### Cited Findings
- Image analysis in breast microwave sensing is mostly qualitative and quantitative metrics mostly describe contrast; consistent definitions are recommended — Reimer & Pistorius 2023, Sensors 23(11):5123, [PMC10255334](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10255334/) (snippet-level).
- Only four articles estimated specificity, 20 to 65 % — same source.
- Localization numbers in the opened paper are conditional on the identifiable subset: 12 of 18 reconstructions identifiable, error 11 ± 4 mm — [PMC7345629](https://pmc.ncbi.nlm.nih.gov/articles/PMC7345629/).
- Elahi 2018: in the patient with no disease, DAS and DMAS both showed a dominant response that would be a false positive without the mammogram (earlier note; [PMC6022049](https://pmc.ncbi.nlm.nih.gov/articles/PMC6022049/)). This is a qualitative equivalent of "the pipeline always produces a peak".
- Searches for 'false positives in tumour-free phantoms with DAS' returned a Sensors 2023 review ([doi 10.3390/s23115123](https://mdpi.com/1424-8220/23/11/5123)) and preprints, but no baseline-versus-method table.

### Inferences
- Benchmarking rules (my synthesis, standard in prediction-model evaluation but not from an MBI source): (1) report error of the constant predictor at the dataset centre and at the dataset mean tumour position; (2) report a paired phantom-cluster bootstrap CI of the difference (pipeline minus baseline); (3) report the same on the subset a reliability score accepts, because a score that merely selects central tumours lowers selective risk with no real information (the project's composite A finding: pipeline approximately equals trivial inside the accepted set); (4) report skill-score form 1 - LE_pipeline / LE_trivial; (5) test the null that the pipeline's predicted peak is independent of the true position (correlation of predicted and true x, y), which a centre-biased method fails.
- For detection at 15 mm, a trivial guess succeeds when the tumour is within 15 mm of the centre; the success rate depends only on the tumour position distribution, so it must be reported with the dataset's own distribution.

### Gaps
- No paper reporting a trivial or random-location baseline for breast microwave localization was found. A narrower search of Pistorius/Reimer papers and the 2025 IJMWT three-system comparison (Reimer, Fontaine, Eashour, Krenkevich, Pistorius; [Cambridge](https://resolve.cambridge.org/core/services/aop-cambridge-core/content/view/CD816E28EAFDCD815FC1108BAA40AC62/S1759078725000376a.pdf/image-quality-analysis-and-comparison-of-three-radar-based-breast-microwave-sensing-systems.pdf)) was not opened and may contain one. UNVERIFIED.

---

## Q6. Design rules for a replacement per-scan reliability score

### Takeaway
Rules follow from (a) the documented causes of wrong peaks, (b) the finding that image-shape scores can encode the method's bias, and (c) the power limits above. Most rules are my synthesis; the supporting facts are cited in Q1 to Q5.

### Cited Findings (supporting facts only)
- Coherence metrics may reward non-lesion regions; lower clutter did not guarantee correct localization — [Elahi 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6022049/).
- Central and boundary clutter can exceed the tumour response, particularly in dense phantoms — [Reimer 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7345629/).
- Permittivity misestimation moves and weakens peaks — [Lavoie 2016](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0160849); [Moll 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4243481/).
- Ignoring clustering overstates AUC precision — [Ying et al. 2021/22](https://pmc.ncbi.nlm.nih.gov/articles/PMC8586066).

### Inferences (design rules, all mine)
1. Score definition must not include a term that is a function of the predicted peak's radius alone (distance from centre, 'concentration near centre') unless it is evaluated against the target-free distribution at that radius; otherwise the score reproduces the method's location prior.
2. Normalize every image-derived term by its distribution on adipose-only (target-free) reference scans reconstructed with the same pipeline, per radial bin (and per shell-radius bin): report the score as a percentile or z-score relative to that null. This is the per-pixel/background normalization suggested in the question; it is not standard in the literature.
3. Prefer agreement-based terms (independent sub-apertures, sub-bands, bootstrap over antennas) over shape-based terms, since agreement estimates variance and does not need a model of a good image; remember agreement cannot detect shared bias (central pull), so combine with rule 2.
4. Control confounders explicitly: report AUROC within strata (tumour radius tertiles, shell-radius tertiles, density class) and a partial AUROC or logistic model adjusting for position; accept the score only if the within-stratum AUROCs stay above 0.5 with an interval that excludes it.
5. Pre-specify sign, primary endpoint, primary score, correction method and hold-out phantoms (Q4). With n = 504 scans from 66 phantoms, plan for a minimum detectable AUROC of about 0.60 to 0.62; any score expected to be weaker than that cannot be validated here.
6. Always benchmark selective-risk results against the trivial centre predictor evaluated on the same accepted subset (Q5).
7. Do not reuse the same 504 scans for both score design and validation: hold out whole phantoms, or use nested phantom-grouped cross-validation and report the nested estimate.

### Gaps
- None of these rules is validated in published microwave imaging work; they are recommendations to be tested on the project's data. The sub-aperture/sub-band consistency idea has no direct published counterpart that I found (as in the earlier note).

---

## Source reliability summary
- Opened and read in the page text returned: Reimer 2020 (PMC7345629), Moll 2014 (PMC4243481), Lavoie 2016 (PLOS ONE), Ying et al. 2021/22 (PMC8586066).
- Snippet-level or abstract-level only: Hanley-McNeil 1982, DeLong 1988, Obuchowski 1997, Wu 2020, Benjamini-Hochberg 1995, Holm 1979, Gelman & Loken 2013, Kirshin 2011, O'Loughlin 2017/2019/2021, Kazemi & Fear 2023, Reimer & Pistorius 2023, Brunel skin-clutter papers, Muppala 2025, arXiv 1704.03303, arXiv 2409.19517, Prykhodko thesis, Wavelia preprints.
- Not accessible: PMC5017770 (bot check; its content overlaps with Lavoie 2016 via PLOS), arXiv 2409.19517 (PDF bytes unparsed), Kirshin EuCAP 2011 (DNS error).
- SIM (mine) numbers come from a short numpy Monte Carlo; scripts were run in a temp folder and are not saved in the repo. They assume equal cluster sizes, a Gaussian score, and a delta of 0.3 sd; they are order-of-magnitude guides, not estimates for the real data.
