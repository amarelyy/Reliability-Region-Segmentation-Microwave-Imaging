# Signal-level novelty / anomaly / model-misfit scores as ground-truth-free reliability indicators (S21, 1001 x 72, UM-BMID Gen-2)

Method note: about 17 tool calls. Web access was poor. PMC returned a bot check, ResearchGate and the arXiv PDF could not be parsed, and the IEEE and GitHub pages were summary-only. Much of what I found is therefore from search-result summaries and is marked UNVERIFIED. The prior notes in `research_notes/RRS MBI reliability score fixes/` (SMR/SCR metrics, selective-classification theory, DAS/CF central bias) were read and are not repeated. They already establish that no validated ground-truth-free per-scan reliability score exists in the microwave breast imaging literature. This file did not overturn that finding.

Convention: "Cited" means a source supports the claim. "Formula (ours)" means a NumPy recipe written by this researcher from standard definitions. It is NOT validated in the microwave imaging literature and has no citation for the microwave use case.

## Q1. Distance-from-healthy-reference scores (Mahalanobis, Wasserstein, energy distance, channel anomaly counts, band-limited and time-gated residual energy)

### Takeaway
I found no peer-reviewed paper that applies Mahalanobis, Wasserstein or energy-distance scores to the target-minus-reference residual of a microwave breast scan. These are statistically standard tools (the energy distance has a verified primary-source basis) but their use here is untested. The 4-6 GHz sub-band claim was not verified in any source I could open. In the user's own earlier work, a sub-band consistency score had only weak signal (AUROC 0.60), so treat sub-band energy as a candidate, not a result.

### Cited Findings
- The energy distance between two samples is built from within-sample and between-sample mean pairwise distances, scaled by n_i*n_j/(n_i+n_j). Significance is normally by permutation or bootstrap. The test is consistent against all alternatives when first moments are finite. Source: [R `energy` package eqdist.etest documentation](https://search.r-project.org/CRAN/refmans/energy/html/eqdist.etest.html) (snippet-level, via search).
- Energy distances are maximum mean discrepancies (MMD) for distance-induced kernels. Source: Sejdinovic et al. 2013, [arXiv 1207.6076](https://pdf.arxiv.org/pdf/1207.6076) and [arXiv 1205.0411](https://ar5iv.labs.arxiv.org/html/1205.0411) (snippet-level). This lets energy distance and kernel MMD be treated as one family.
- In the UM-BMID dataset the repository supplies `clean` (reference-subtracted) and `simple-clean` (no subtraction) versions of each generation. The reference for a tumour scan is the matching adipose-only (or fibroglandular-insert) scan. Source: [UM-BMID GitHub](https://github.com/UManitoba-BMS/UM-BMID). The page does not list metadata fields; the fields come from the user's own `metadata_gen_two.pickle` and the project notes.
- Reimer, Solis-Nepote, Pistorius (2020) used a clutter region and a tumour region on the image, not the raw data, for SCR and SMR. Source: [Diagnostics 10(6):411](https://pmc.ncbi.nlm.nih.gov/articles/PMC7345629/) (from the earlier note). This means image-domain scores need ground truth; the raw-data scores below do not.
- Elahi et al. (2018) report that with no disease present, DAS and DMAS still produced a dominant response (a false positive without the mammogram). They propose left/right breast comparison for rejection but call it future work. Source: [Sensors 18(6):1678](https://pmc.ncbi.nlm.nih.gov/articles/PMC6022049/) (from the earlier note). This is the closest published idea to a "novelty versus a paired reference" reliability test, and it is not implemented.
- I found no source for 4-6 GHz being informative in phantom studies. A search for it returned nothing usable. UNVERIFIED and unsupported.

### Formulas (ours; not validated in this application)
Notation: `S_t` target scan, shape (F=1001, A=72) complex. `S_r` reference scan for the same phantom (adipose-only, `adi_ref_id`). `D = S_t - S_r` is the residual. `f` in GHz.

1. Noise floor from repeatability. Pair scans of the same healthy phantom taken in the same session (or any two healthy scans with the same radius). `N_k = H_i - H_j`. Build a bank of residuals `N` of shape (K, F, A).
   ```python
   # channel-wise z-score against the healthy-minus-healthy noise floor
   mu = N.mean(0); sd = N.std(0) + 1e-12            # complex-safe: use real/imag separately
   z = (np.abs(D) - np.abs(N).mean(0)) / (np.abs(N).std(0) + 1e-12)
   anomaly_count = (z > 3).mean()                   # fraction of F*A channels beyond 3 sigma
   ```
   Caveat: K must be large relative to the number of tests. Apply a Holm or Benjamini-Hochberg correction if you test channels individually. A permutation null (below) avoids the normality assumption.

2. Mahalanobis distance of the residual feature vector. Reduce D to a feature vector `x` (for example band energies per antenna, or the first m PCA components of the noise bank, m << K).
   ```python
   from sklearn.covariance import LedoitWolf, MinCovDet
   cov = LedoitWolf().fit(Nfeat)                    # K x m features of healthy-minus-healthy residuals
   d2 = cov.mahalanobis(x[None])[0]                 # squared distance
   # reference: chi2.sf(d2, m) is only valid if features are near-Gaussian; otherwise use empirical rank in Nfeat
   ```
   MinCovDet is the robust alternative (Rousseeuw's MCD). I did not verify a primary MCD citation in this session. UNVERIFIED as a citation; the method is standard.
   Caveat: with K around 100 and m around 50 the sample covariance is unstable, so shrinkage is needed.

3. Energy distance between the target residual distribution and the noise bank. Treat the 72 antenna-wise feature vectors of D as a sample (n=72), and those of a noise residual as another.
   ```python
   from scipy.spatial.distance import cdist
   def energy_distance(X, Y):                       # X (n,d), Y (m,d)
       xy = cdist(X, Y).mean(); xx = cdist(X, X).mean(); yy = cdist(Y, Y).mean()
       return 2*xy - xx - yy
   # significance: permutation of pooled labels, p = (1 + #{e_perm >= e_obs}) / (1 + B)
   ```
   Antenna positions are spatially correlated, so permuting antennas violates exchangeability. Permute whole scans (target residual versus noise residuals) instead.

4. 1-D Wasserstein distance per frequency or per time bin. 
   ```python
   from scipy.stats import wasserstein_distance
   w = wasserstein_distance(np.abs(D[f_idx]).ravel(), np.abs(N[:, f_idx]).ravel())
   ```
   Wasserstein in 1D compares magnitude histograms only and ignores phase and antenna structure. It is a weak detector of a localized target echo and I do not recommend it as a primary score. No microwave breast source found.

5. Band-limited residual energy. Choose the band as a pre-registered window (do not tune it on the test labels).
   ```python
   fgrid = np.linspace(1, 8, 1001)
   band = (fgrid >= 4) & (fgrid <= 6)
   E_band = (np.abs(D[band])**2).sum() / (np.abs(D)**2).sum()   # fraction of residual energy in band
   z_band = (E_band - E_band_noise.mean()) / E_band_noise.std()  # versus noise bank
   ```

6. Time-gated residual energy in the expected return window. Convert to time domain with a windowed IFFT, or the repository's ICZT (0-6 ns, 1024 samples).
   ```python
   d_t = np.fft.ifft(D * np.hanning(F)[:, None], n=1024, axis=0)
   t = np.linspace(0, 6e-9, 1024)
   gate = (t > t_lo) & (t < t_hi)    # t_lo/t_hi from the delay model: 2*r_breast*sqrt(eps_eff)/c, +-margin
   E_gate = (np.abs(d_t[gate])**2).sum(); E_out = (np.abs(d_t[~gate])**2).sum()
   gate_ratio = E_gate / (E_out + 1e-30)
   ```
   Caveat: a gate set from the shell radius assumes the delay model is right. This conflicts with the point of the misfit score in Q2, so compute the gate from the data-estimated skin delay.

### Inferences
- The most defensible primary score is the scan-level z or permutation p of a scalar (band or gated residual energy) against the healthy-minus-healthy noise bank. It is cheap, has a clear null, and is stable with small K. Mahalanobis needs shrinkage. Wasserstein-1D is the weakest option.
- These scores measure "something differs from the reference", not "a tumour is present and was localized". A failed reference subtraction (session drift, a different radius) also produces a large residual. Any such score must be conditioned on the confounds in Q5, or it will rank reference mismatch above tumour presence.
- From the user's own logs (project CLAUDE.md): sub-band distance AUROC 0.602 [0.514, 0.682] with multiplicity unaddressed. That is the only local evidence on the 4-6 GHz idea, and it is weak.

### Gaps
- No published use of Mahalanobis, Wasserstein or energy distance on microwave breast residuals found.
- No source for the 4-6 GHz informativeness claim found.
- No primary citation retrieved for Rousseeuw's MCD or Mahalanobis outlier detection (the search covered only the energy-distance half).
- No verified paper on a healthy-only one-class model for microwave breast S-parameters; search returned only supervised autoencoder-plus-classifier work (see Q4).

## Q2. Time-domain and delay-model consistency scores

### Takeaway
Estimating the effective permittivity from the data and flagging mismatch to the assumed model is supported in principle: Lavoie, Okoniewski and Fear (2016) do it using image-quality fitness, not a skin-echo fit. Their method is image-domain (so beamforming-dependent) and they report wrong permittivity degrades images. A signal-domain circle/ellipse fit of skin-echo delays versus antenna angle is not found as a published reliability score.

### Cited Findings
- Lavoie, Okoniewski, Fear, "Estimating the Effective Permittivity for Reconstructing Accurate Microwave-Radar Images", PLOS ONE 11(9), 2016, doi:10.1371/journal.pone.0160849. They define a fitness function that scores image quality, using knowledge of how the images form, then fit a polynomial interpolant to locate the most likely permittivity. The interpolant uses locally and dimensionally adaptive sampling (stochastic collocation, polynomial chaos). Data were simulated, experimental and patient data from the Calgary TSAR system. They call the result "effective" because it is a weighted representative value, not a true average. The method works from only a broad permittivity range. Source: [PLOS ONE via PMC (abstract-level, via search)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5017770/). The full text was blocked; formulas and error numbers were NOT read. UNVERIFIED beyond the abstract.
- A search summary stated that Lavoie et al. note wrong permittivity degrades images and can create false detections. This came from a search-tool summary, not from the paper text. UNVERIFIED.
- Skin-response removal from the scan itself: a MIST-type patent fits the skin-breast interface response as a parametric function of skin thickness and permittivities by least squares, and a TSAR patent estimates skin reflection from neighbouring antennas. Sources: [US 7,570,063 (Space-time microwave imaging)](https://patents.google.com/patent/US7570063), [tissue sensing adaptive radar](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7454242) (patent summaries via search, UNVERIFIED in detail). The residual of the parametric fit is a natural misfit score, but the patents do not frame it as one.
- Time-domain gating of the transmitted signal to isolate antenna reflections has been used to estimate permittivity and conductivity (Fear group report, via search). Source: [MTT-S report](https://www.mtt.org/app/uploads/2019/01/04-Final-Report_David-Garrett.pdf) (snippet only; the attribution to a "Fear group report" is by the search summary). UNVERIFIED.
- Skin artifact varies across channels in realistic scenarios, and many removal algorithms assume it is common across channels. Source: [Artifact removal algorithms for microwave imaging of the breast, Univ. Galway](https://researchrepository.universityofgalway.ie/entities/publication/c31d84dc-95d1-4359-a0cc-f92eafb926b7) (snippet). This supports using the cross-channel spread of the skin echo as a quality cue.
- UM-BMID Gen-2 geometry (project file `scan_geometry.md`, per the project CLAUDE.md): `ant_rad` measured from the SMA connector; phase-centre correction `r = 0.97*(r_m - 0.106) + 0.148` (metres); 1-8 GHz, 1001 points. Not re-verified here from the primary doc.

### Formulas (ours; not validated)
1. Skin-echo delay per antenna. For each position a, take `d_t[:, a]` after reference subtraction is NOT applied (the skin echo is removed by subtraction), i.e. use the simple-clean target scan. Estimate the first strong arrival.
   ```python
   env = np.abs(hilbert(np.real(np.fft.ifft(S_t * w[:, None], n=1024, axis=0)), axis=0))
   tau = t[np.argmax(env[gate_early], axis=0)] + t[gate_early][0]     # (72,)
   # sub-sample refinement: parabolic interpolation around the peak
   ```
   For S21 (bistatic, ports offset by 60 degrees) the arrival is a transmit-leg plus receive-leg path, so the circle model is for the specular skin path, not a monostatic round trip.
2. Circle fit residual (monostatic-style approximation). Convert delay to range `rho = c*tau/(2*sqrt(eps_eff))`, antenna positions `p_a = r_ant*(cos th_a, sin th_a)`. Fit centre c0 and radius R of the skin boundary by minimising `sum_a (|p_a - c0| - (r_ant - rho_a))^2` (nonlinear least squares, `scipy.optimize.least_squares`). Outputs: `rms_resid`, `R_fit`, `c0` offset.
   ```python
   def resid(x, th, rho, r_ant):
       c0 = x[:2]; p = r_ant*np.c_[np.cos(th), np.sin(th)]
       return np.linalg.norm(p - c0, axis=1) - (r_ant - rho)   # distance from antenna to skin point
   ```
   A better test is: fit eps_eff as a free parameter and flag scans where the fitted eps_eff is outside a plausible band or where `rms_resid` exceeds that of the healthy-reference population. The project notes record a 13-15 mm centre offset found from an echo fit and an echo-based absolute radius biased about 15 percent low (project CLAUDE.md progress log). That is a local result, not a literature one.
3. Model-misfit score. `misfit = rms_resid / sigma_healthy` where `sigma_healthy` is the same quantity over healthy scans of the same radius. Cross-validate by phantom.
4. Dispersion. A frequency-dependent effective permittivity could be fitted by repeating the delay estimate in 2 GHz sub-bands (phase-slope group delay: `tau_g = -d(phase)/d(omega)`) and testing whether `tau_g` varies across bands more than in healthy scans.
   ```python
   ph = np.unwrap(np.angle(S_t), axis=0); tau_g = -np.gradient(ph, 2*np.pi*fgrid*1e9, axis=0)
   ```
   No source retrieved on dispersion-based reliability flags. Formula and use are ours.

### Inferences
- A signal-domain "model explains the data" score is plausible and cheap, and it addresses exactly the failure mode in the user's audit (wrong start angle, radius and centre, shell centre not at the chamber origin). It is unvalidated. If the fitted centre or radius is far from the assumed ones, the beamformed localization is likely unreliable, but whether this predicts localization error needs testing on the user's own 504 scans, with phantom-grouped resampling.
- Lavoie et al. is the strongest literature anchor that the permittivity assumption matters and can be estimated, but their estimator needs images, so it is partly circular for a pre-beamforming score.

### Gaps
- No verified paper fitting a circle or ellipse to skin-echo delays versus antenna angle as a quality score.
- Lavoie et al. numerical results (error versus permittivity) were not read.
- No verified source on dispersion or lossy-medium diagnostics used as reliability flags.
- Patent sources are not peer reviewed.

## Q3. SNR and drift diagnostics (reference repeatability, session drift, clutter-to-residual, SVD rank)

### Takeaway
SVD of the channel matrix is an established clutter-removal tool in microwave breast imaging (drop the two dominant singular components). I did not find its spectrum used as a reliability score. Repeatability and drift scores must be built from the metadata in the user's data; no source describes them for UM-BMID.

### Cited Findings
- I-MUSIC removes skin-interface clutter by SVD of the data matrix (N rows by A columns) and discarding the projections on the two dominant singular values. Source: [Columbia SLD, Pagliari et al. BioCAS 2015 (via search)](https://www.sld.cs.columbia.edu/pubs/pagliari_biocas15.pdf) (snippet; the attribution of the SVD description to this specific paper is by search-result adjacency, UNVERIFIED).
- A dry-setup study proposes an iterative adaptive SVD algorithm to handle stronger skin backscatter. Source: [ISCTE repository PDF](https://repositorio.iscte-iul.pt/bitstream/10071/20915/1/Heterogeneous%20breast%20MWI%20v5_3rdSub%20v2.pdf) (snippet). Search found no threshold rule from the spectrum shape (no energy-fraction or knee criterion).
- Cross-correlated back projection reduces clutter without SVD. Source: [PMC3035806](https://pmc.ncbi.nlm.nih.gov/articles/PMC3035806/) (snippet).
- Yin et al. 2015, "A robust and artifact resistant algorithm of ultrawideband imaging system for breast cancer detection", IEEE TBME. Source: [City Univ. repository PDF](https://openaccess.city.ac.uk/id/eprint/8364/1/Yin2015_ARobustAndArtifactResistantAlgorithm_IEEETBE.pdf) (title only, content not read). UNVERIFIED relevance.
- I found no TVSVD (total-variation SVD) source specific to microwave breast imaging.

### Formulas (ours; not validated)
```python
# SVD spectrum of the (frequency x antenna) residual
U, s, Vh = np.linalg.svd(D, full_matrices=False)          # D: (1001, 72) complex
p = s**2 / (s**2).sum()
eff_rank = np.exp(-(p*np.log(p+1e-30)).sum())             # participation / entropy rank
top1 = p[0]; top2 = p[:2].sum()
tail_ratio = (s[2:]**2).sum() / (s[:2]**2).sum()          # energy left after removing 2 dominant modes
# marchenko-pastur-style noise edge from noise bank: s_noise_max = np.median([svd(N_k)[1][0] for k])
# clutter-to-residual ratio (CRR) in the time domain
CRR = (np.abs(S_t_td[gate_skin])**2).sum() / (np.abs(D_td[gate_target])**2).sum()
# drift: residual energy on the early part of the time axis where the target cannot yet contribute
drift = (np.abs(D_td[t < t_skin_arrival - margin])**2).sum() / (np.abs(S_t_td[gate_skin])**2).sum()
```
Interpretation: if the target-minus-reference residual before the skin arrival (where nothing physical changed) is large, the reference scan mismatches the target (drift, repositioning, radius change). That is a reference-validity flag, independent of tumour presence. The reasoning is ours.

Repeatability: for phantoms scanned more than once, `rep = ||H_i - H_j||_F / ||H_i||_F`. The metadata in the user's data should be checked for duplicate scans and dates; I could not verify that UM-BMID has same-condition repeats.

### Inferences
- "Energy before the skin arrival" is the cleanest drift diagnostic, since it needs no model of the tumour. It is ours and untested.
- The SVD spectrum is easy to compute, but a high top-1 energy mostly reflects skin clutter and radius, so confounding by session or radius is likely (Q5).

### Gaps
- No source using SVD spectrum shape as a reliability or quality indicator.
- No TVSVD source found.
- Yin 2015 and Pagliari 2015 contents were not read.

## Q4. Machine-learning detection from S-parameters on open datasets: reported performance, split design, leakage

### Takeaway
I could not verify any UM-BMID ML paper's split design or its reported accuracy, so I cannot say whether reported accuracies survive phantom-grouped splitting. The one dataset-structure fact that matters is verified in snippets: scans are grouped by phantom, and Gen-1 scans are unique per phantom ID, tumour size and position, so a scan-level random split would put the same phantom in train and test. No author in what I could read audited this.

### Cited Findings
- UM-BMID: three generations. Gen-1: 249 scans (S11 only). Gen-2: 1008 scans (S11 and S21). Gen-3: 200 scans (1-9 GHz, tumour diameters 10 to 30 mm, 3D-printed phantoms). Total over 1250 scans. Source: [arXiv 2401.02715 summary via search](https://arxiv.org/pdf/2401.02715); [IEEE DataPort](https://ieee-dataport.org/node/4383). Described in Reimer, Krenkevich, Pistorius, "An Open-Access Experimental Dataset for Breast Microwave Imaging", EuCAP 2020: [IEEE](https://ieeexplore.ieee.org/document/9135659/).
- A published IEEE conference work on supervised ML for breast cancer detection in the frequency domain exists: [Supervised Machine Learning for Breast Cancer Detection Using Microwave Imaging in the Frequency Domain (IEEE 10501027)](https://ieeexplore.ieee.org/document/10501027/). The page was not readable. Dataset, accuracy and split are UNVERIFIED.
- Khalid et al., ISCAS 2023, "Efficient Deep Learning Approaches for Automated Tumor Detection, Classification, and Localization in Experimental Microwave Breast Imaging Data" (DBLP: [conf/iscas/KhalidHMARM0M23](https://dblp.uni-trier.de/rec/conf/iscas/KhalidHMARM0M23.html)). Only the bibliographic record was seen. Whether it uses UM-BMID and how it splits is UNVERIFIED.
- Al-Khatib et al., "A Deep Learning Framework for Breast Tumor Detection and Localization from Microwave Imaging Data" ([ResearchGate](https://www.researchgate.net/publication/357719606_A_Deep_Learning_Framework_for_Breast_Tumor_Detection_and_Localization_from_Microwave_Imaging_Data)); a search summary says it uses the dataset with tumour radius classification. Split design UNVERIFIED. Likewise [arXiv 2304.10386, Breast cancer detection using deep learning](https://arxiv.org/pdf/2304.10386): the PDF was unreadable, so its dataset and split are UNVERIFIED. Be careful: a search summary linked it to UM-BMID only by search ranking.
- Autoencoder-decoder plus probabilistic neural network on S21 magnitude from MammoWave data (not UM-BMID): about 60 percent accuracy, sensitivity and specificity, with class imbalance noted. Source: IEEE ISMICT 2024, DOI 10.1109/ISMICT61996.2024.10738023, via [search snippet](https://unpaywall.org/10.1109%2FISMICT61996.2024.10738023). UNVERIFIED (snippet).
- A textile-antenna phantom study reports 84 and 89 percent accuracy from S11 and S21 alone and 93 percent combined. Not UM-BMID. Source: [Cambridge Core](https://core-prod.cambridgecore.org/core/services/aop-cambridge-core/content/view/7FDA76F951C5315DA0AC6C37D49AABD7/S1759078722000319a.pdf/classification-of-scattering-parameters-of-body-embedded-wideband-textile-antennas-for-early-diagnosis-and-monitoring-of-breast-cancer.pdf) (snippet).
- General evidence that scan-level splits inflate test accuracy when the unit is the subject: Inflation of test accuracy due to data leakage in OCT deep learning, recommends per-subject splitting. Source: [PMC9500039](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9500039/). Different modality; supports the principle only.
- Local evidence (project, not literature): a U-Net trained with a row-order split leaked phantoms; with a phantom-grouped split, held-out Dice was 0.17 and mask-centroid detection at 15 mm was about 33 percent (5 seeds), not clearly better than predicting the centre. Source: the user's own progress log in `CLAUDE.md`.

### Inferences
- Any reported UM-BMID classification accuracy from papers that do not state a phantom-grouped split should be treated as an upper bound until proven otherwise. The dataset has many scans per phantom (rotations are the same phantom, and tumour positions vary within a phantom), so leakage is structurally easy. This is an inference from the dataset structure, not a finding from a specific paper.
- Gen-1 uses one scan per unique configuration; Gen-2 has 72-position scans of many phantom configurations sharing adipose shells. Even phantom-grouped splits may leak through the shared shell geometry or the shared reference scan. Group by shell and by session to be safe.

### Gaps
- No ML-on-UM-BMID paper's split design or numbers were verified; the key question (do accuracies hold up under phantom-grouped splits) is unanswered by sources.
- No source identifying session, date or antenna-radius confounding in UM-BMID ML work was found.
- The UM-BMID README (`docs/UM-BMID_README.txt`) and scan geometry doc were not read directly.

## Q5. Quantifying and removing acquisition-session confounds (antenna radius, date, session id)

### Takeaway
I found no UM-BMID-specific literature. Standard methods exist (grouped cross-validation, confound regression, conditional AUROC, stratified or matched evaluation), but I did not retrieve primary sources for them in this session. The recipes below are ours.

### Cited Findings
- Per-subject (grouped) splitting is recommended in place of scan-level random splitting to avoid leakage. Source: [PMC9500039](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9500039/) (OCT study; principle only).
- Per the project data checks: `ant_rad` takes values 21, 22 and 23 cm for all 504 scans, adi_x/adi_y are nonzero in 202 of 504, and shell centres are not at the chamber origin (project `CLAUDE.md` progress log). These are the local confounders to control. Not a literature source.

### Formulas (ours; not validated)
1. Measure confounding before adjusting. For each candidate score `r` and each nuisance variable `z` (radius class, date bin, session id, shell id):
   ```python
   from scipy.stats import kruskal, spearmanr
   kruskal(*[r[z == g] for g in np.unique(z)])           # categorical z
   # between-group variance share: eta2 = SSB/SST ; compare with eta2 for the outcome (tumour present or LE)
   ```
   A score that explains more variance of `z` than of the outcome is mostly a confound detector.
2. Conditional evaluation. Compute AUROC within each stratum of `z` and combine (for example a stratified Mann-Whitney or Mantel-Haenszel style average). The stratified AUROC is the honest number if the full-sample AUROC is much higher.
3. Residualisation. Regress the score on nuisance variables with phantom-grouped cross-fitting, then use the residual as the score.
   ```python
   from sklearn.model_selection import GroupKFold
   for tr, te in GroupKFold(5).split(X, y, groups=phantom_id):
       beta = np.linalg.lstsq(Z[tr], r[tr], rcond=None)[0]; r_adj[te] = r[te] - Z[te] @ beta
   ```
   Caveat: if tumour presence is correlated with `z` by design (for example tumours only scanned at certain radii), residualisation removes real signal. Check the cross-tabulation first.
4. Negative control. Compute the score on healthy-minus-healthy pairs with different `z` (for example different radius) and healthy-minus-healthy pairs with the same `z`. A large gap means the score responds to session change, not to a target.
5. Group everything by phantom (and by shell) in all resampling: permutation, bootstrap and cross-validation. The project already uses a phantom-cluster bootstrap.

### Inferences
- The negative control (item 4) is the most informative single test for the user's situation, because it uses only healthy data and needs no tumour labels.
- If a candidate score cannot beat the same-confound baseline, drop it. Stratified AUROC and the negative-control gap should be reported with CIs.

### Gaps
- No primary source on confound adjustment in microwave imaging or in medical-signal ML with session effects was retrieved; the methods are standard statistics from memory, not cited.
- Whether date and session id exist in `metadata_gen_two.pickle` was not checked here (no data access in this task).

## Overall summary for the report writer
- Nothing found that validates a ground-truth-free reliability score from raw S-parameters for microwave breast imaging. The candidates in Q1 to Q3 are plausible and cheap but untested. The user's earlier internal tests (AUROC about 0.6 for sub-aperture and sub-band consistency) are the only local evidence and are weak.
- Best-supported building blocks: energy-distance or MMD two-sample tests with permutation null (verified general theory); SVD clutter subspace removal (verified use, not as a score); permittivity estimation from data affecting image accuracy (Lavoie 2016, abstract-level).
- The largest unverified area is the ML-split question; the report writer should not state that published UM-BMID accuracies fail under phantom-grouped splits, because no paper was read that tested this.
- Multiple-comparison note: every extra candidate score tested on the same 504 scans adds to the multiplicity already recorded in the user's log.
