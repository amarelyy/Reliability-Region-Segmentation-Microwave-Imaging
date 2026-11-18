# Tumour-size information in wideband (1-8 GHz) microwave scattering data

Scope note: search tools returned mostly snippets/abstracts. Items marked UNVERIFIED were seen only as a snippet or abstract or via a secondary aggregator. Items under "Inferences" are my own reasoning or arithmetic (labelled "inference"); they are not found in sources. The earlier notes folders (reliability metrics, RRS-MBI fixes) were not repeated; I only checked that they concern reliability/central-bias, not size.

Conventions for all arithmetic below: c = 300 mm/ns; lambda_medium = 300 / (f[GHz] * sqrt(eps_r)) mm; "a" = tumour radius (5, 10, 15 mm = diameters 10, 20, 30 mm).

## Q1. Scattering regimes and amplitude-vs-size scaling

### Takeaway
For a 5-15 mm radius tumour in eps_r ~5-10 background, the size parameter k_b*a runs from about 0.2 (5 mm, 1 GHz) to about 8 (15 mm, 8 GHz), so the band spans Rayleigh at the low end through resonance (Mie) to near geometric optics at the high end. The amplitude exponent versus radius is therefore not a single power: about r^3 (Rayleigh/Born) at low frequency, falling toward r^1 to r^2 (reflection-dominated) at high frequency. Sources found state the Rayleigh and geometric limits only generically; I found no phantom paper that reports amplitude vs. radius for 5/10/15 mm tumours.

### Cited Findings
- Rayleigh limit for a small dielectric sphere: scattering cross-section sigma ~ (8*pi/3) k^4 a^6 |(eps-1)/(eps+2)|^2, i.e. a^6 in power (a^3 in amplitude) and f^4 in power; valid only for small size parameter. — [Wisconsin Mie lab notes](https://www.physics.wisc.edu/courses/home/spring2018/407/experiments/Mie/mie.pdf); [Purdue ECE 604 Lecture 34](https://engineering.purdue.edu/wcchew/ece604s19/Lecture%20Notes/Lect34.pdf) (seen via search snippets, content UNVERIFIED in full)
- With full-wave (Mie) theory the cross-section does not keep growing as f^4 a^6; for large spheres it tends to the geometric value, order pi*a^2; resonance-type enhancement occurs when (m^2+2) becomes small. — same snippets as above (UNVERIFIED in detail)
- A closed-form solution exists for the field scattered by a lossy dielectric spherical tumour of arbitrary diameter in a lossy fat medium, and the authors stress that the choice of polarization and operating frequency matters for detection (abstract only; UNVERIFIED) — surfaced in search (conference paper, title/URL not retrieved).
- A breast-imaging patent states that frequency-dependent scattering varies with tumour size and gives a means of assessing change over time (no quantitative relation given). — [US 7570063](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7570063)
- Malignant vs normal adipose-dominated tissue contrast can be as large as about 10:1 (Lazebnik et al. 2007, cancer-surgery study), and normal tissue is grouped by adipose content (0-30 %, 31-84 %, 85-100 %) with one-pole Cole-Cole fits; 354 normal samples. — [Lazebnik et al. PMB 2007 (UW-Madison PDF)](https://uwcem.ece.wisc.edu/pdf/lazebnik_pmb07br.pdf) (secondary snippet; PDF could not be parsed, so the actual adipose eps/sigma values are NOT confirmed here)
- Real breast contrast may be smaller than early studies assumed: a later tomography paper argues earlier work used too large a malignant/glandular contrast; its Table 1 gives adipose eps_inf ~2.3-4.1 and sigma ~0.002-0.084 S/m (Debye-type parameters) vs fibroglandular eps_inf ~16.8-29.1. — [PMC2921423](https://pmc.ncbi.nlm.nih.gov/articles/PMC2921423)
- In wave propagation the signal amplitude falls due to radial spreading of the impulse energy and path loss in tissue (so received amplitude depends on geometry/depth as well as size). — [Ahmad et al., URSI GA 2011 (snippet)](https://www.ursi.org/proceedings/procGA11/ursi/KBE-7.pdf) (UNVERIFIED beyond the snippet)
- No source found that reports measured response amplitude vs tumour radius for a 5/10/15 mm radius sweep in phantoms. (Search returned nothing direct.)

### Inferences
(all numbers derived by me from the formula above; not from a source)
- Wavelength in medium (mm), eps_r = 5 / 10: 1 GHz 134 / 95; 4.5 GHz 29.8 / 21.1; 8 GHz 16.8 / 11.9. Inside a tumour with eps_r ~50 (sqrt = 7.07): 42 mm at 1 GHz, 9.4 mm at 4.5 GHz, 5.3 mm at 8 GHz.
- Size parameter in the background, x = 2*pi*a/lambda_b, for eps_b = 5 / 10:
  - a = 5 mm: 1 GHz 0.23 / 0.33; 4.5 GHz 1.05 / 1.49; 8 GHz 1.87 / 2.64.
  - a = 10 mm: 0.47 / 0.66; 2.1 / 3.0; 3.7 / 5.3.
  - a = 15 mm: 0.70 / 1.0; 3.1 / 4.5; 5.6 / 7.9.
  Internal size parameter k_t*a (eps_t = 50): 0.75 (5 mm, 1 GHz) to 17.7 (15 mm, 8 GHz); internal resonances are expected for every size at 4.5-8 GHz because a/lambda_t is 0.5-2.8.
- Regime summary: Rayleigh (x<~0.5) holds only below ~2 GHz for 5-10 mm radius; at 4.5 GHz all three sizes are in the Mie/resonance region; at 8 GHz the 15 mm tumour is approaching optical (x ~6-8). The whole 1-8 GHz band thus straddles all regimes.
- Expected amplitude ratios (amplitude = sqrt of power, same depth, same material):
  - Rayleigh (a^3): 10 vs 5 mm = 8x (18 dB); 15 vs 10 mm = 3.4x (10.5 dB); 15 vs 5 mm = 27x (28.6 dB).
  - Reflection/geometric (RCS ~ a^2, amplitude ~ a): 10 vs 5 = 2x (6 dB); 15 vs 10 = 1.5x (3.5 dB); 15 vs 5 = 3x (9.5 dB).
  - Realistic band-integrated response: somewhere between these, so roughly 6-18 dB (5->10 mm) and 3.5-10.5 dB (10->15 mm). Resonance oscillations in x (Mie ripples) can add several dB of non-monotonic structure at fixed frequency, so single-frequency amplitude is a poor size cue; band-integrated energy is smoother.
- Born approximation caution (inference): Born needs weak contrast and small phase shift across the object. Here contrast (eps_t - eps_b)/eps_b ~ 4-9 and internal phase shift k_t*a up to ~18 rad, so Born validity fails for the upper end; the a^3 law is a low-frequency/small-object limit only. Tumour conduction loss also saturates the response for large radius (the object becomes shadowing, back face invisible).
- Depth/loss: for a lossy background, one-way attenuation alpha = (sigma/2)*(eta0/sqrt(eps_r)). With sigma ~0.1-0.3 S/m and eps_r ~5 at ~5 GHz (typical adipose-like values from my memory of Lazebnik, UNVERIFIED), alpha is about 1-3 dB/cm one-way. A 1 cm change in tumour depth therefore changes the two-way level by about 2-6 dB plus spreading, which is the same size as the 10->15 mm amplitude step (3.5-10 dB). So depth uncertainty of +/-1-2 cm is a first-order confound for amplitude-based size estimates.
- Tumour internal loss (eps ~50, sigma ~3-7 S/m at 5 GHz, typical literature values from memory, UNVERIFIED): alpha ~ 0.9-1 Np/cm, roughly 8 dB/cm one-way, so a wave crossing a 20-30 mm diameter tumour and back loses tens of dB. Back-face echoes are therefore unlikely to be usable, and the scatterer behaves like a front-surface reflector plus creeping wave; this weakens "ringing" or front/back-echo time-domain size cues for a lossy high-contrast tumour (if the phantom tumour is similarly lossy).

### Gaps
- No measured-phantom amplitude-vs-radius curve found.
- Actual Lazebnik adipose eps_r and sigma at 1/4.5/8 GHz not confirmed (PDF unreadable via tool); the UM-BMID phantom's true eps_r and tumour eps_r are not retrieved (dataset page does not state them).
- No dedicated Mie-resonance treatment of lossy tumour spheres retrieved (only the abstract of a closed-form spherical paper).

## Q2. Spectral and time-domain features used for size

### Takeaway
The best-documented size cues in the literature are (a) learned features from whole multichannel backscatter (PCA / local discriminant bases) in simulation, which classified size classes 0.5-2 cm at 10 dB SNR; (b) resonance-frequency shifts of self S-parameters and S21 changes in simulation; and (c) complex-domain (phase) transfer-function features in FDTD. All are simulation-based; I found no verified measured-phantom diameter-feature study in 1-8 GHz.

### Cited Findings
- Davis, Van Veen, Hagness, Kelcz (IEEE TBME 2008): numerical test of classifying target shape and size from 1-11 GHz multichannel UWB backscatter; four size classes 0.5-2 cm diameter; local discriminant bases and PCA feeding linear classifiers; at 10 dB SNR size classified with >97 % accuracy averaged over 360 targets, shape about 70 %. Simulation only. — surfaced via search ([typeset mirror](https://typeset.io/pdf/ultrawideband-microwave-breast-cancer-detection-a-detection-nhkf26o9dm.pdf), [IEEE record mirror](https://search.isc.ac/inventory/49/953455.htm)); content UNVERIFIED from abstract level.
- Ahmad et al. (URSI GA 2011): in a 2-D FDTD numerical breast, the phase of the complex transfer function H(z) of the reflected waveform decreases over the whole normalized frequency range as tumour size decreases; the authors conclude size strongly affects phase. — [URSI KBE-7](https://www.ursi.org/proceedings/procGA11/ursi/KBE-7.pdf) (snippet; UNVERIFIED beyond it)
- Contrast-enhanced variant (microbubbles; pole-zero and Bode analysis of differential response) claims viability for lesion-size categorisation; abstract reports no numeric metrics and states standard imaging struggles because dysplastic and normal tissue contrast is low. — [Northumbria portal](https://researchportal.northumbria.ac.uk/en/publications/feasibility-study-of-tumor-size-classification-via-contrast-enhan/)
- Simulated (CST, 8.5-11.5 GHz, three antennas, truncated hemisphere) study: self S-parameter resonant frequencies decrease as tumour length increases, and mutual S21 varies with size; features taken from 9-10 GHz (self) and 10-11 GHz (frequency-averaged mutual). Tumour size 0-20 mm, RMSE 1.5 mm, R2 0.96 (80/10/10 random split; simulation only; physics explanation thin). — [PMC12349357](https://pmc.ncbi.nlm.nih.gov/articles/PMC12349357)
- Brain-tumour antenna study (not breast) found impedance Z the most useful feature and S-parameters the weakest (84.21 % accuracy) for 3 vs 5 mm radius classes. — [STSA brain study summary](https://library.kab.ac.ug/Record/doaj-art-c00e4b1c84134036b15cbe1f97cdfee2/Description) (search summary; UNVERIFIED)
- Measured-phantom classification found: Conceicao et al. (Med. Phys. 2020) used 13 benign and 13 malignant tumour models 13-40 mm, 1-6 GHz monostatic prototype, kNN best at 96.2 % benign-vs-malignant (homogeneous phantom); this is class of lesion, not diameter. — [Univ. of Galway record](https://research.universityofgalway.ie/en/publications/classification-of-breast-tumor-models-with-a-prototype-microwave--6/) (abstract level; UNVERIFIED)
- Phantom study with a CIRS phantom and coax probe at 2.4 GHz found permittivity and loss increased with mass size. — surfaced in search (no URL retained); UNVERIFIED and not a scattering measurement.

### Inferences
- A frequency-slope feature (log amplitude vs log frequency over 2-4 GHz) is expected to be about +2 (amplitude ~ k^2) in Rayleigh for fixed r, flattening toward 0 as x grows; since x grows with both f and a, the slope should decrease with radius. Prediction: the 15 mm tumour should show a flatter/earlier-rolling spectrum than the 5 mm one, and the 2-3 GHz vs 7-8 GHz amplitude ratio should shrink with radius. This is only a prediction from scaling; I found no source confirming it for phantoms.
- Resonance notches for eps_t ~50 spheres in a eps_b ~5-10 background should lie at internal a/lambda_t ~0.5, i.e. frequencies (inference) of roughly 4-6 GHz for a=15 mm and 8+ GHz for a=5 mm; at 1-8 GHz bandwidth with VNA noise and clutter they are likely buried and also washed out by tumour loss.
- Time domain: pulse width 1/B ~ 0.14 ns (B=7 GHz); range resolution in medium c/(2B sqrt(eps)) ~ 9.6 mm (eps 5) or 6.8 mm (eps 10). A 10 mm diameter tumour is below this so imaging blob width carries almost no size information for the 5 mm radius class; blob width approaches size information only for 20-30 mm diameters, and then it is dominated by the point-spread function, array geometry and DAS blur.

### Gaps
- No verified measured-data (phantom) paper establishing a specific spectral slope, notch or ringing feature as a size predictor in 1-8 GHz.
- Davis et al. and Ahmad et al. not read in full; exact features, class boundaries and noise models not confirmed.

## Q3. ML size regression/classification on open phantom datasets (UM-BMID and others)

### Takeaway
The UM-BMID Gen-1/Gen-2 documentation and secondary papers list tumour diameters 10/20/30 mm (Gen-3 adds 15 and 25), but I found no verified paper that reports measured-data tumour-size classification with a phantom-grouped split or against a trivial baseline. The only reported "size classification" numbers on UM-BMID-derived data (98.35 %) are snippet-level, from MRI-derived signals, with no stated split, and should be treated as unreliable until checked.

### Cited Findings
- UM-BMID contains 1257 phantom scans; data stored as frequency-domain S-parameters; clean files have empty-chamber reference subtraction; repository README is the authority on structure; the repo page did not state tumour sizes, permittivities, or noise levels. — [UM-BMID GitHub](https://github.com/UManitoba-BMS/UM-BMID); [IEEE DataPort](https://ieee-dataport.org/node/4383)
- Gen-2: 66 distinct phantoms from 9 different polycarbonate-copolyester adipose shells; one preprint's table lists tumour diameters of 10, 20, 30 mm for Gen-1 and Gen-2, and 10, 15, 20, 25, 30 mm for Gen-3 (secondary source, not the dataset descriptor). — [Girish et al., arXiv 2304.10386](https://arxiv.org/abs/2304.10386)
- That preprint does image-level tumour detection with CNNs on MRI-derived-phantom reconstructions: best NASNetLarge accuracy 88.41 %, AUC 0.786; it does not (per abstract) do size estimation. — [arXiv abstract](https://arxiv.org/abs/2304.10386)
- Khalid et al., ISCAS 2023, "Efficient Deep Learning Approaches for Automated Tumor Detection, Classification, and Localization in Experimental Microwave Breast Imaging Data": bibliographic existence confirmed. — [dblp](https://dblp.uni-trier.de/rec/conf/iscas/KhalidHMARM0M23.html). A search snippet attributes to a related framework "RF-BreastNet" 97.19 % detection, R2 93.37 % localization, 98.35 % size classification on MRI-derived signals; one more snippet mentions a multitask tumour-radius/breast-shell model with "a substantial gap" for the radius task — both UNVERIFIED (full text not accessible; ResearchGate returned 403). Split design (by phantom or random) is not known.
- Dataset-descriptor: Reimer, Krenkevich, Pistorius, EuCAP 2020. — [IEEE Xplore 9135659](https://ieeexplore.ieee.org/document/9135659/) (not fetched; paywalled)
- Simulation-only DL size estimators (RMSE 1.5 mm, R2 0.96) use a random 80/10/10 split without cross-validation or phantom grouping, and have no measured test. — [PMC12349357](https://pmc.ncbi.nlm.nih.gov/articles/PMC12349357)
- Conceicao et al. tumour-model classification used 26 tumour models in two phantom types (benign/malignant); results are benign vs malignant, not size. (see Q2 citation)
- A Reimer talk title "The Diagnostic Capability of Machine Learning for Tumor ..." exists ([Bellairs 2020 slides](https://www.compem.ece.mcgill.ca/breast-cancer-detection/workshop/presentations/2020_Bellairs_TReimer.pdf)) but the host failed to load; content unknown.

### Inferences
- Because Gen-2 scans come from 66 phantoms made of 9 shells, with scan sessions and shells shared across tumour sizes, a random scan-level split will leak shell identity, session/day drift and clutter pattern; any reported high size accuracy from such a split should be treated as potentially due to shell/session identity. (Reasoning, plus the project's own noted leakage in U-Net training.)
- With three classes (10/20/30 mm), the chance level depends on class balance; before claiming skill the trivial majority-class and "predict the shell/phantom prior" baselines must be compared. No source quantifies these baselines.

### Gaps
- No verified accuracy of size classification/regression on measured UM-BMID Gen-1/2/3 with phantom-grouped validation.
- Could not read Khalid et al. or the RF-BreastNet/multitask papers; class definitions, split design and baselines unknown.
- Gen-3 literature not found at all in this pass.

## Q4. Confounds

### Takeaway
Depth, fibroglandular/phantom heterogeneity, antenna-to-skin distance, session effects and tumour shape/localisation error all act on the same quantity as size (received amplitude), and several of them change it by more than the 10->15 mm size step. Sources found support the existence of these confounds qualitatively; quantitative magnitudes are mostly my inference.

### Cited Findings
- Normal breast tissue shows large dielectric variation due to heterogeneity (354 samples, three adipose-content groups) — [Lazebnik et al. 2007](https://uwcem.ece.wisc.edu/pdf/lazebnik_pmb07br.pdf) (snippet-level).
- Amplitude is reduced by radial spreading and tissue path loss, so depth modulates received level — [URSI KBE-7](https://www.ursi.org/proceedings/procGA11/ursi/KBE-7.pdf) (snippet).
- Earlier malignant/normal contrast assumptions may have been too large and glandular/fat contrast too small, so the effective contrast for the same tumour size changes with density — [PMC2921423](https://pmc.ncbi.nlm.nih.gov/articles/PMC2921423).
- Tumour shape (smooth, microlobulated, spiculated) affects classification: shape ~70 % vs size >97 % in simulation, i.e. shape variation is a distinct nuisance variable — Davis et al. (see Q2).
- Frequency dependence: higher frequencies set resolution but reduce penetration — [PMC2921423](https://pmc.ncbi.nlm.nih.gov/articles/PMC2921423).
- The UM-BMID "clean" files use an empty-chamber reference subtraction; the repo lists no noise levels. — [UM-BMID GitHub](https://github.com/UManitoba-BMS/UM-BMID)

### Inferences
- Depth: about 2-6 dB per cm (two-way) from the loss estimate in Q1, comparable to the 3.5-10 dB 10->15 mm step; with the project's own finding of predictions clustering about 11 mm from centre regardless of truth, depth/position errors will also corrupt amplitude features.
- Antenna radius/standoff: 21/22/23 cm radii in the data (project CLAUDE.md) change spreading by about (23/21)^2 ~ 0.8 dB, small versus other terms but not zero; session/day drift in the empty reference or in phantom temperature (the permittivity of water-based phantoms is temperature- and age-dependent) can produce multi-dB level shifts (inference; no source found).
- Fibroglandular heterogeneity (eps_b 5 to 20+) changes the tumour-vs-background contrast factor |(eps_t-eps_b)/(eps_t+2 eps_b)| by perhaps 20-40 %, i.e. 2-3 dB, between adipose and glandular surroundings (inference from Rayleigh contrast factor, eps_t=50: eps_b=5 -> 0.72; eps_b=10 -> 0.51; eps_b=20 -> 0.30). Contrast factor in dB between eps_b=5 and 10: about 3 dB amplitude. Not negligible vs. the 3.5 dB 10->15 mm step in the reflection regime.
- Localisation error enters twice: through the wrong depth correction and through wrong beamforming delays (focus defocus).

### Gaps
- No source quantifying session/day effects on UM-BMID, or antenna-radius effect on amplitude.
- No source with measured density-confound magnitude in phantoms.

## Q5. Detectability: amplitude ratios vs noise and clutter

### Takeaway
The expected amplitude steps between 5, 10 and 15 mm tumours (about 3.5-18 dB depending on regime) are in principle detectable in a calibrated, low-noise VNA system, but they are of the same order as the unmodelled nuisance terms (depth, contrast, session), and the tumour response itself is typically far below the residual clutter. Sources for actual UM-BMID noise/clutter levels were not found, so a precise feasibility statement cannot be made from literature alone; the project's own data is needed.

### Cited Findings
- In numerical work, size class (four classes 0.5-2 cm) was recovered with >97 % accuracy at 10 dB SNR, using whole multichannel backscatter — Davis et al. (see Q2; abstract level).
- In differential imaging reconstructions, 20 dB SNR kept dense-tissue reconstructions accurate while 10 and 0 dB visibly degraded them (SNR defined relative to mean measured total field). — [PMC2921423](https://pmc.ncbi.nlm.nih.gov/articles/PMC2921423)
- Simulation size regressors reach RMSE 1.5 mm in noiseless/CST data; noise robustness not reported in what I read. — [PMC12349357](https://pmc.ncbi.nlm.nih.gov/articles/PMC12349357)
- Phantom studies in this field mostly use tumours of a few mm to 4 cm, with 12 mm inserts detectable in experimental phantoms; none reported size-from-amplitude trends. — [Yang thesis snippet via search](https://ninercommons.charlotte.edu/islandora/object/etd%3A1289/datastream/PDF/download/citation.pdf) (UNVERIFIED)

### Inferences
(order-of-magnitude, labelled as such)
- Signal step to resolve a class change: 5->10 mm radius is 6-18 dB, 10->15 mm is 3.5-10 dB in tumour response amplitude; a regressor needs an amplitude measurement uncertainty well below this, e.g. <1.5 dB, to separate 10 vs 15 mm.
- Typical VNA dynamic range is 80-100 dB, but clutter dominates: skin/shell reflection and antenna coupling are typically 30-60 dB above a tumour echo (general knowledge in the field; I did not retrieve a source), and after empty-chamber or adipose-reference subtraction the residual (shell misplacement, drift) sets the floor rather than thermal noise. Thus the effective SNR per scan is likely 0-20 dB, the regime where Davis-type simulation results (10 dB SNR) begin to degrade; plausible but not verified.
- Combined error budget (1-sigma, my estimate): depth/position +/-1-2 cm -> 3-8 dB; contrast/density -> 2-3 dB; session/reference drift -> 1-3 dB; residual clutter -> 2-5 dB. Root-sum-square about 5-10 dB. Against a 3.5-10 dB 10->15 mm step this gives per-scan separability (d') of roughly 0.5-1, i.e. weak; the 5->10 mm step (6-18 dB) is more feasible, d' of roughly 1-3. So a coarse "small vs large" call is feasible at best, fine three-class estimation is marginal without depth/density normalisation (e.g. adipose reference, matching tumour-bearing and tumour-free scans from the same session).
- Differential estimates (tumour scan minus its matching tumour-free scan of the same phantom) remove most shell, density and session terms and leave depth/shape; this is the setting in which a size-vs-energy relation has the best chance, and should be tested first as an upper bound.
- Trivial baselines to beat: majority class; phantom/shell identity; tumour-distance prior. Any size model should be evaluated grouped by phantom against those.

### Gaps
- No actual UM-BMID per-scan SNR, clutter-to-tumour ratio, or VNA dynamic-range figures found.
- All feasibility numbers above are inference and should be replaced by measurements from the project data (e.g. tumour-minus-reference energy per size class, grouped by phantom).
