# Ground-truth-free confidence, quality and reliability metrics for microwave breast imaging (MBI)

Method note: about 14 web calls. This note extends the three earlier notes in `research_notes\RRS MBI reliability score fixes\` and does not repeat them (Reimer 2020 Diagnostics SCR/SMR definitions, Elahi 2018 Sensors, Klemm CF, Mallart-Fink / Li-Li GCF, selective-classification theory). What is new here: the Reimer and Pistorius 2023 review (read in part), the Reimer and Pistorius 2021 ORR paper (read), the GLRT/FDR detection framework, the UM-BMID generation facts, PCF/SCF in MBI. Anything from a search snippet or from my own background knowledge is marked UNVERIFIED. Two PDFs (Cambridge 2025, arXiv 2304.10386) could not be parsed.

## 1. Which MBI papers define ground-truth-free image-quality or confidence indices (CF, PCF, SCF, GCF, DMAS variants, MV/Capon, clutter estimates, multi-frequency / multi-view consistency, FP discrimination, decision rules)?

### Takeaway
The MBI literature defines almost no per-scan confidence score that works without ground truth. Beamformer-side quantities (CF, PCF, SCF, GCF, DMAS) are pixel weights, not trust scores. The only fully ground-truth-free decision rules found are: (a) a detection-theoretic GLRT with a false-discovery-rate threshold (Davis, Hagness, Van Veen 2005), (b) a MAX/AVG threshold on image statistics for patient images (Sani et al.), (c) an SCR threshold of 1.5 dB used in UM-BMID Gen-3 work (this SCR itself needs the tumour region, but the thresholded quantity can be recomputed at the predicted peak). The Reimer and Pistorius 2023 review says explicitly that current analyses rely on single-pixel intensity and/or full knowledge of true tissue properties and are unsuited to blind trials.

### Cited Findings

#### Review-level statements (verified from full text, partial)
- Reimer and Pistorius, "Review and Analysis of Tumour Detection and Image Quality Analysis in Experimental Breast Microwave Sensing", Sensors 2023, 23(11):5123 (PMID 37299852). Table 1 lists 11 metrics: SCR = Tmax/Cmax, SMR = Tmax/Cmean, MMR = Tmean/Cmean, TFRR = Tmax/Fmax (tumour to fibroglandular response), CCR (contrast-to-clutter ratio, formula garbled in extraction), C/Tt (clutter-to-tumour ratio at threshold t), LE = |r_max - r_tum|, MSE, SSIM, FWHM (area of pixels above 50 % of max), and the f1 / f metrics from their ref. 70: f1 = Imax/(N*A) with A the FWHM area and N the number of distinct areas inside it; f = f1*P with P = min(d/3 mm, 1) and d the distance from the imaging-domain boundary to the peak pixel. SCR appears in 22 of 76 quantitative papers and has several definitions. — [PMC10255334](https://pmc.ncbi.nlm.nih.gov/articles/PMC10255334/)
  - Note: f1 and f are the only listed metrics that need no tumour location (they use only peak, FWHM area and boundary distance). They are "focal quality metrics used as fitness functions" in the cited source (title seen in the reference list only: "focal quality metrics as fitness functions", ref. 98; not read).
- Same review: current metrics "rely on single-pixel intensity and/or full knowledge of true tissue properties" and so do not suit blind or double-blind trials; resolution, noise and artifacts are largely unaddressed; "tumour-like artifacts in reconstructions of healthy breasts" appear in several studies and are not adequately addressed; the authors recommend quantitative comparison of healthy and tumour-containing reconstructions. — [PMC10255334](https://pmc.ncbi.nlm.nih.gov/articles/PMC10255334/)
- Same review, detection criteria in phantom studies: SCR threshold 1.5 dB with a true positive requiring localization within the known tumour radius (O'Loughlin et al. and Reimer et al., their refs 100 and 120). O'Loughlin did not count hotspots within 10 mm of the imaging-domain boundary as false positives. Patient studies (Sani et al., refs 132-134) used a MAX/AVG threshold set from each dataset, which the review says depends on breast size. Moloney et al. (ref 94) used ROI thresholding with size, solidity and contrast criteria; 4 of 24 images had tumour-like responses at wrong positions. Half of the patient image-based studies used subjective clinician judgement, none blinded. — [PMC10255334](https://pmc.ncbi.nlm.nih.gov/articles/PMC10255334/)
- Same review: only 52 of 164 image-based papers (32 %) analysed healthy images or phantoms. In O'Loughlin et al., 94 % of images labelled tumour-free got that label via the boundary exception, and only 6 % through the SCR criterion; the review concludes localization was poor. — [PMC10255334](https://pmc.ncbi.nlm.nih.gov/articles/PMC10255334/)

#### Hypothesis-test / FDR detection (ground-truth-free by construction)
- Davis, Tandradinata, Hagness, Van Veen, "Ultrawideband microwave breast cancer detection: a detection-theoretic approach using the generalized likelihood ratio test", IEEE Trans. Biomed. Eng. 52(7):1237-1250, 2005, doi:10.1109/TBME.2005.847528. A GLRT per voxel tests H0 (no tumour) against H1; backscatter is modelled as Gaussian with a known clutter covariance estimated offline; clutter whitening; the voxel threshold is chosen to constrain the false discovery rate of the image; tumours smaller than 0.6 cm reported detected and localized in numerical and experimental phantoms. Conference precursor: Davis, Hagness, Van Veen, IEEE SSP Workshop 2003, pp. 617-620 (clutter covariance model, power analysis under matched and mismatched conditions). Patent US 8,050,740 B2. Seen via search abstracts and patent summary only (UNVERIFIED in detail). — [typeset record](https://typeset.io/pdf/ultrawideband-microwave-breast-cancer-detection-a-detection-nhkf26o9dm.pdf), [WARF record](https://minds.wisc.edu/items/33d223ac-9693-4407-928f-d3c32183cbe9)
  - Validated against detection/localization on numerical and experimental phantoms (per abstract); not tested on UM-BMID.

#### Coherence-family weights (pixel weights, not trust scores)
- Rasappan et al. 2022 (via Tareq Zanoon / AAUP repository listing; USM thesis), "Microwave 3D imaging system featuring phase coherence factor improved beamforming": PCF-weighted DAS, 32-antenna p-slot array, breast phantoms, detects 5 mm tumours in a dense 3D model. Snippet only. — [AAUP](https://repository.aaup.edu/handle/123456789/1590), [USM](https://erepo.usm.my/entities/publication/4581aae7-80e3-4913-8155-451a3da7cc32/full)
- "Sign coherence factor-based double-stage delay multiply and sum beamforming algorithm for breast cancer detection", Microwave and Optical Technology Letters, 2024 (SCF-DMAS; 3D-printed phantom with circular array). Snippet only; formula not seen. — [catalogue record](https://vufind.katalog.k.utb.cz/SummonRecord/FETCH-LOGICAL-c2272-11678bf375bf21b2a0f07713920d14fad462f52d967f7daa6ddf768f31c157833)
- Scaled CF-DMAS (NWPU 2024), iteratively corrected CF-DAS (IEEE Access 2019): both already listed in the earlier notes; they report SCR gains, not confidence evaluation. — [NWPU](https://pure.nwpu.edu.cn/en/publications/scaled-coherence-factor-based-delay-multiply-and-sum-beamforming-/)
- Standard formulas (background knowledge, UNVERIFIED against a primary text here; primary is Camacho, Parrilla, Fritsch, IEEE TUFFC 56(5):958, 2009 for PCF/SCF; Li and Li, IEEE TUFFC 50(2):128, 2003 for GCF): for N channel samples s_i(p) aligned at pixel p, CF = |sum s_i|^2 / (N sum |s_i|^2); PCF = max(0, 1 - sigma(phi)/sigma0) with phi the instantaneous phase of the aligned samples and sigma its standard deviation (circular-phase variants exist); SCF = 1 - sqrt(1 - (|sum b_i|/N)^2) with b_i = sign(s_i); GCF = fraction of the spatial-DFT energy of s_i(p) in a low-frequency band M0 over total energy. All are amplitude-independent, so they measure channel agreement, not tumour likelihood. The earlier note already makes the argument that CF can be high for coherent clutter.

#### Capon / MV
- I found no MBI paper that uses Capon/MV output as a per-image confidence. Only beamformer comparisons: coherent time-reversal minimum-variance beamformers on anatomically realistic 3D phantoms (UTS record) and a low-complexity DAS/DMAS paper that describes Capon/MVDR for context (arXiv 1709.02549). Elahi 2018 (in earlier note) found RCB placed a dominant scatterer near the skin and enhanced residual clutter. — [UTS](https://opus.cloud1.lib.uts.edu.au/handle/10453/138599), [arXiv 1709.02549](https://arxiv.org/pdf/1709.02549)

#### Sub-band / sub-aperture consistency
- Coherent time-reversal / C-DORT (Radio Science 2015, UTS): a focusing matrix maps frequency bins onto a consistent subspace because per-frequency non-coherent imaging is "often inconclusive" in clutter; robustness claim for dense phantoms is vs non-coherent DORT, not a confidence score. — [UTS record](https://opus.lib.uts.edu.au/handle/10453/33186)
- Foroozan, HosseinKhah, Adve: sub-aperture SVD and neighbour-subarray correlation to model clutter in coherent time-reversal breast imaging (slide-style excerpt only). — [sigport](https://sigport.org/sites/all/modules/pubdlcnt/pubdlcnt.php?fid=2838)
- 2025 PMC paper (PMC12349357): attention model that picks frequency sub-bands for localization and size estimation; simulation only. — [PMC12349357](https://pmc.ncbi.nlm.nih.gov/articles/PMC12349357)
- None of these uses agreement of sub-band or sub-aperture peaks as a reliability score, and none validates against error.

### Inferences
- Implementable ground-truth-free candidates that are supported at least by the literature's vocabulary, with what each needs (all computable from S21, reference scans and the reconstruction; none validated in MBI to my knowledge):
  1. Healthy-reference peak statistics: compute the same peak statistic (SCR-like Pmax/Cmax where the peak is taken at the predicted location and clutter is the max outside a mask around it, or Pmax/mean) on the adipose-only reference scans of the dataset; the per-scan score is the exceedance (percentile or FDR p-value) relative to that null distribution, ideally matched in radius. This is the direct analogue of the GLRT/FDR threshold, using empirical nulls instead of a Gaussian clutter model. The review's own recommendation (compare healthy and tumour images) and the Gen-3 SCR >= 1.5 dB rule both point here.
  2. Focal metric f1/f from the review (peak over N*A with FWHM area, and boundary-distance factor P): no tumour location needed.
  3. MAX/AVG over the breast mask (Sani et al.): the review says it depends on breast size, so normalise by mask area or by the null distribution of item 1.
  4. DAS-vs-DMAS-vs-CF peak agreement and sub-band/sub-aperture peak agreement: uncited in MBI, but the same logic as the selective-prediction notes; treat as hypotheses.
- Because the hot-spot artifact (section 3) is a DAS/DMAS property on healthy phantoms, any peak-height score has an unknown false-positive floor unless the healthy-reference null is used.

### Gaps
- I did not read Camacho 2009, Li and Li 2003, or the SCF-DMAS paper in full; formulas above are from background knowledge.
- The "focal quality metrics as fitness functions" paper (ref. 98 of the review) is a candidate source for f1/f and was not retrieved. Search next: "focal quality metric fitness function microwave imaging".
- Davis 2005 not read beyond abstract; whether any later work applied GLRT/FDR to UM-BMID was not found.
- The review text beyond character 200000 (about 75k characters, including conclusions and appendices) was not read.

## 2. Works that compare DAS vs DMAS vs other beamformers and use agreement as a reliability cue

### Takeaway
Comparisons exist (Elahi 2018, Reimer 2020, Reimer 2021) but all rank algorithms against ground truth. I found no work that uses between-beamformer agreement as a per-scan reliability cue. The closest hint is the reverse: a hot-spot artifact common to DAS and DMAS (so agreement between them would NOT flag it).

### Cited Findings
- Reimer and Pistorius 2021, "An Optimization-Based Approach to Radar Image Reconstruction in Breast Microwave Sensing", Sensors 21(24):8172, doi:10.3390/s21248172. 200 scans of MRI-derived phantoms (UM-BMID Gen-3) with adipose and healthy references. DAS sensitivity 19 %, specificity 44 %; ORR 27 % and 56 %; DMAS specificity 40 %. With adipose-only reference (ideal skin suppression) no beamformer reached practical sensitivity; DAS and DMAS specificity on healthy images was below a random classifier. The "hot-spot" artifact appears in DAS and DMAS healthy reconstructions as a localized high-intensity region not corresponding to any phantom feature, resembling a tumour response, and is not seen in ORR. The paper's own relative-improvement statements are inconsistent across sections (42 %/27 % in the abstract vs 42 %/35 % and 27 %/40 % in the conclusions). — [PMC8704509](https://pmc.ncbi.nlm.nih.gov/articles/PMC8704509/)
- Same paper, detection rule: image is positive if SCR >= 1.5 dB; true positive only if localization error <= tumour radius + positional uncertainty (4 mm in Gen-3); for healthy phantoms any image with SCR >= 1.5 dB is a false positive. ROC curves were built by sweeping 1000 thresholds from 0 to 30 dB; ORR beat DAS and DMAS at every threshold. — [PMC8704509](https://pmc.ncbi.nlm.nih.gov/articles/PMC8704509/)
- Review summary of the same family (their ref. 120): DAS up to 71 % sensitivity and up to 44 % specificity; DMAS up to 77 % and 40 %; ORR up to 82 % and 56 %. — [PMC10255334](https://pmc.ncbi.nlm.nih.gov/articles/PMC10255334/)
- Elahi 2018 (earlier note): DMAS ranked first in 5 patients; CF-DAS and IDAS suppressed clutter but misplaced the response; coherence metrics may reward non-lesion regions.
- Reimer 2020 Diagnostics (earlier note): itDAS/itDMAS vs DAS/DMAS, SCR about 0 to 5 dB, LE 11 +/- 4 mm on the 12/18 identifiable scans.

### Inferences
- The hot-spot artifact is shared by DAS and DMAS in UM-BMID Gen-3 healthy scans. Beamformer-agreement would therefore likely confirm artifacts. This is my inference from the two statements above; it has not been tested.
- A reliability score validated by AUROC on a healthy-phantom set (FP discrimination) is not found in the literature; Gen-3 has healthy references, which could provide it. For Gen-2 (the user's data) the adipose references are the only target-free scans and the earlier project audit shows their peaks sit near the centre.

### Gaps
- Numeric SCR/LE values for DAS, DMAS and ORR in the 2021 paper are only in Figures 2-3 and were not retrievable as text.
- The 2025 IJMWT paper (Reimer, Fontaine, Eashour, Krenkevich, Pistorius, "Image quality analysis and comparison of three radar-based breast microwave sensing systems", doi:10.1017/S1759078725000376, CC BY 4.0) is directly relevant to image-quality metrics across systems; only the PDF was reachable and the extraction failed. Read it next. — [Cambridge](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/CD816E28EAFDCD815FC1108BAA40AC62/S1759078725000376a.pdf/image-quality-analysis-and-comparison-of-three-radar-based-breast-microwave-sensing-systems.pdf)

## 3. Robustness / repeatability of localization across sub-bands, sub-apertures, antenna subsets, noise realisations

### Takeaway
I found no MBI study that measures localization repeatability across sub-bands, sub-apertures or noise realisations on UM-BMID or elsewhere. The review states that LE is "non-robust" and that diagnostic estimates depend strongly on the SCR threshold. Related repeatability evidence comes only from the dataset side (multiple scans of the same phantom) and from time-reversal papers.

### Cited Findings
- The review flags LE as non-robust and the MAX/AVG metric as breast-size dependent. — [PMC10255334](https://pmc.ncbi.nlm.nih.gov/articles/PMC10255334/)
- Reimer 2021: sensitivity/specificity depend on the SCR threshold; evaluated over a threshold sweep. — [PMC8704509](https://pmc.ncbi.nlm.nih.gov/articles/PMC8704509/)
- Review: several ML studies used multiple measurements of the same phantom (refs 53, 76, 79, 82, 104, 118, 119, 138) and data leakage occurred in several (refs 53, 112, 113, 188, 202; possible in 48, 82, 135, 138, 196). Only ref. 120 is said to meet basic ML methodological standards. — [PMC10255334](https://pmc.ncbi.nlm.nih.gov/articles/PMC10255334/)
- Sub-band/sub-array material: see section 1 (C-DORT, Foroozan et al., 2025 attention paper); none validates against error.

### Inferences
- Split-half/sub-band agreement as a reliability score looks new for MBI as far as these searches go (absence of evidence from about a dozen generic queries, not proof).
- Leakage across repeated scans of one phantom is an acknowledged problem in the field; the project's phantom-grouped folds follow the review's standard.

### Gaps
- No noise-realisation (bootstrap over antennas or frequencies) study found. Searches for "sub-aperture" and "sub-band" returned only the items above.

## 4. UM-BMID: what is published, headline numbers

### Takeaway
UM-BMID is documented by the EuCAP 2020 conference paper and IEEE DataPort/Zenodo/GitHub, not by a Scientific Data paper (I found none). Gen-1 and Gen-2 results in the literature are classifier (ML) results, with AUCs of 78-94 %; verified image-based DAS/DMAS numbers are Gen-3 only (sensitivity/specificity 19-27 % / 44-56 %, or 63-82 % in other systems), and no Gen-2 DAS unconditional localization-error or detection-rate figure was found. The project's 30 mm mean LE and 12-17 % detection at 15 mm cannot be placed against a published Gen-2 image baseline; against Gen-3 the closest comparison is that DAS sensitivity at the SCR >= 1.5 dB plus LE <= r+4 mm rule was 19 % with adipose-only reference.

### Cited Findings
- Dataset description: Reimer, Krenkevich, Pistorius, "An open-access experimental dataset for breast microwave imaging", EuCAP 2020, doi:10.23919/EuCAP48036.2020.9135659. Gen-1: 249 scans, Gen-2: 1008 scans (S11 and S21, 1-8 GHz, tumour diameters 10, 20, 30 mm in the cited table), Gen-3: 200 scans. IEEE DataPort doi:10.21227/1y0z-8t98; Zenodo mirrors; GitHub UManitoba-BMS/UM-BMID. Counts are from a search-result summary; no Scientific Data article found (the user's premise of one is unconfirmed). — [IEEE DataPort](https://ieee-dataport.org/node/4383), [GitHub](https://github.com/UManitoba-BMS/UM-BMID), [Zenodo](https://zenodo.org/record/5120981)
- Gen-3 (verified from the 2021 paper): 200 scans, 20 healthy phantom geometries, five tumour diameters (10, 15, 20, 25, 30 mm), each tumour scan with healthy and adipose references plus a shared empty-chamber reference; positioning uncertainty 4 mm; 1-9 GHz, 1001 frequencies, 72 antenna positions over 355 degrees. — [PMC8704509](https://pmc.ncbi.nlm.nih.gov/articles/PMC8704509/)
- ML on Gen-1: Reimer et al. (review ref. 118), 249 phantom scans: sensitivity 95 +/- 6 %, specificity 80 +/- 10 %, accuracy 85 +/- 4 %, AUC 94.4 +/- 0.5 %. ML on Gen-2: ref. 119, 1008 scans, mean AUC 78 +/- 3 %, rising to 90 +/- 3 % when the test set is constrained to matching breast volume and tumour height. The review states many ML results suffered leakage or lacked true test sets and are probably overestimates. — [PMC10255334](https://pmc.ncbi.nlm.nih.gov/articles/PMC10255334/)
- AAPM 2020 abstract from the same group: more than 1250 phantom scans; deep learning exploiting data structure AUC 86 +/- 1 %, shallow learning 81.3 +/- 0.1 %. — [AAPM](https://w3.aapm.org/meetings/2020AM/programInfo/programAbs.php?aid=53430)
- Image-based (Gen-3) numbers: see section 2. Reimer 2020 Diagnostics (18 Gen-style scans): LE 11 +/- 4 mm, 12/18 identifiable, SCR about 0-5 dB (earlier note).
- Review-wide image-based sensitivity range 63-100 %, specificity range 20-65 % (4 studies); authors conclude this does not show clinical utility. SCR 1.5 dB rule and 10 mm boundary exception: see section 1. — [PMC10255334](https://pmc.ncbi.nlm.nih.gov/articles/PMC10255334/), [PubMed 37299852](https://www.pubmed.gov/37299852/)
- Other UM-BMID users: a Gen-3 physics-driven AI paper (arXiv 2401.02715; tolerance = tumour radius + 0.5 cm; in earlier note) and an arXiv deep-learning classification paper (2304.10386; generation and numbers not extractable). A 3D CNN breast-density paper (Ronca 2026) appeared in search but was not opened. — [arXiv 2401.02715](https://arxiv.org/pdf/2401.02715), [arXiv 2304.10386](https://arxiv.org/pdf/2304.10386)
- The review's text, as retrieved (first 200k characters), never mentions "UM-BMID" by name; its refs 118-120 correspond to the Manitoba group's work per the matching numbers, but the name-to-dataset mapping is from my reading of the sensitivity numbers and the 249/1008 scan counts (249 = Gen-1, 1008 = Gen-2). Treat as likely, not verified.

### Inferences
- The project's detection at 15 mm (12-17 %) and the Gen-3 DAS sensitivity (19 %, adipose-only reference) are of the same low order, which is consistent with the 2021 paper's statement that with ideal skin suppression only no beamformer reached practical sensitivity. Rules differ (15 mm radius vs tumour radius + 4 mm; SCR gate vs none), so this is a context point, not a replication.
- The published Gen-2 ML AUC of 78 % (constrained 90 %) is a per-scan detect-vs-no-tumour classifier on raw data; it is a separate task from the project's localization-reliability task and cannot be used as a comparator for it.

### Gaps
- No Gen-2 DAS/DMAS localization-error table found. The Cambridge 2025 comparison paper may contain one (not retrieved).
- The EuCAP 2020 paper's text and the Reimer 2020 thesis tables were not read here.
- Cameron, Sadeghi, Kurrant, Santorelli, Elahi, Hall, Fear groups: I did not find specific papers using UM-BMID by these authors; do not infer any. The UM-BMID repo's scan_geometry.md was not fetched.

## 5. Published reliability metrics known NOT to work (negative results)

### Takeaway
Negative results exist, but mainly about image-quality and detection metrics, not about reliability scores: SCR-based detection gives specificity below chance on healthy phantoms; LE is non-robust; hot-spot artifacts in healthy DAS/DMAS images mimic tumours; coherence metrics may reward non-lesion regions; RLS skin removal also removes tumour response. No paper reporting a failed reliability score (e.g. one that ranks inversely) was found.

### Cited Findings
- DAS and DMAS specificity on healthy phantoms below a random classifier; ORR the only method above 50 %. — [PMC8704509](https://pmc.ncbi.nlm.nih.gov/articles/PMC8704509/)
- O'Loughlin et al.: only 6 % of tumour-free images failed the SCR criterion without the boundary exception; localization poor (review). Specificity as low as 20 %. — [PMC10255334](https://pmc.ncbi.nlm.nih.gov/articles/PMC10255334/)
- Elahi 2018: coherence-based metrics may reward non-lesion regions; CF-DAS and IDAS suppress clutter but misplace the response (earlier note, [PMC6022049](https://pmc.ncbi.nlm.nih.gov/articles/PMC6022049/)).
- Review: MAX/AVG thresholds depend on breast size; LE is non-robust; many ML results leak and over-estimate. — [PMC10255334](https://pmc.ncbi.nlm.nih.gov/articles/PMC10255334/)
- Reimer 2020 (earlier note): in dense classes tumour response not above central clutter; only 1 of 4 Class IV reconstructions identifiable.

### Inferences
- The project's own result (RRS-MBI ranks inversely, rho(R, LE) = +0.19; predictions about 11 mm from centre) fits this literature: image-intensity-based confidence is confounded by clutter and artifacts. Presenting it as a documented negative result is consistent with the review's own call for artifact-aware analysis. It has no direct published precedent that I found, so label it as new.

### Gaps
- No search was done for Bayesian/conformal uncertainty on UM-BMID or for uncertainty in MBI deep learning (Borghouts 2023 probability map is mentioned in the earlier note).
- No paper explicitly reports a negative AUROC or inverse correlation of a reliability score in MBI.

## Priority reading list for the next pass (not done here)
1. Reimer et al. 2025, IJMWT 17, doi:10.1017/S1759078725000376 (image quality across three systems; CC BY 4.0).
2. Davis et al. 2005, IEEE TBME 52(7):1237, doi:10.1109/TBME.2005.847528 (GLRT/FDR).
3. Review ref. 98 on focal quality metrics (f1/f) and ref. 99 "evaluation criteria" (O'Loughlin).
4. Camacho et al. 2009 and Li and Li 2003 for exact PCF/SCF/GCF formulas.
5. Review text after character 200000 (conclusions and appendix tables).
