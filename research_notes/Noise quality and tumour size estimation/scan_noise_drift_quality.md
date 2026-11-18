# Ground-truth-free scan-quality diagnostics (noise floor, drift, clutter rank, channel outliers) for VNA-based microwave breast imaging

Method note: about 16 tool calls. Earlier notes in `research_notes/Reliability metrics for microwave imaging/` and `research_notes/RRS MBI reliability score fixes/` were read (especially `signal_level_novelty_misfit.md`). Their established points are not repeated: no validated ground-truth-free per-scan reliability score exists; the energy-distance, Mahalanobis, band-energy and circle-fit recipes are already written there; the I-MUSIC "drop two dominant singular values" rule is already cited there; sub-band and sub-aperture consistency gave AUROC about 0.60 locally. Web access was poor: several hosts (Polito IRIS, ISCTE, Galway, URSI, IEEE Xplore) failed to load, so most primary papers below are seen as search-result summaries only and are marked UNVERIFIED. Pages actually fetched and read: the Keysight PNA noise-reduction help page, the UM-BMID GitHub README page, and the IEEE DataPort drift-dataset page.

Convention: "Cited" = a source supports it. "Formula (ours)" = NumPy recipe written here from standard definitions; NOT validated in the microwave breast literature. Notation: `S_t` target scan (F=1001, A=72) complex; `S_r` its reference scan; `D = S_t - S_r`; `fg = np.linspace(1, 8, 1001)` GHz; `t` is the ICZT/IFFT time axis (0-6 ns, 1024 samples).

---

## 1. Noise floor and SNR from a single scan; how papers quantify SNR; typical values

### Takeaway
The VNA noise floor is set by IF bandwidth and averaging (Keysight rule: 10x narrower IFBW gives about 10 dB lower floor). I found NO published measured per-scan SNR, noise floor or IFBW/averaging setting for UM-BMID in anything I could open, so typical UM-BMID values are an open gap (the EuCAP 2020 data descriptor and `docs/UM-BMID_README.txt` were not read). A single-scan noise estimate is computable from late-time and out-of-band content, but only as an upper bound on noise, because the late-time region also holds multipath, antenna ringing and chamber reflections.

### Cited Findings
- Keysight: "Reducing the IF receiver bandwidth reduces the effect of random noise on a measurement. Each tenfold reduction in IF bandwidth lowers the noise floor by 10 dB. However, narrower IF bandwidths cause longer sweep times." Averaging "reduces the effects of random noise"; "for minimizing very low noise, Averaging is more effective than reducing IF bandwidth"; changing IFBW after calibration makes calibration accuracy uncertain (C-delta correction). Averaging of an unratioed measurement does not lower the floor as it does for ratioed ones. Source: [Keysight PNA/E5080B help, Noise Reduction Techniques](https://helpfiles.keysight.com/csg/e5080b/s2_opt/trce_noise.htm) (fetched and read; vendor help, not an application note). The same help family says trace noise rises below about 400 MHz on 20 GHz+ models ([search summary](https://helpfiles.keysight.com/csg/e5080a/s2_opt/trce_noise.htm)); not relevant above 1 GHz.
- Keysight help pages do not give a formal dynamic-range definition; a Keysight/Rohde & Schwarz application note defining noise floor vs trace noise vs dynamic range was NOT found. UNVERIFIED for formal definitions.
- UM-BMID: Gen-2 is 1008 scans of 66 distinct phantoms built from 9 different adipose shells, 1-8 GHz (search summaries of [GitHub](https://github.com/UManitoba-BMS/UM-BMID), [IEEE DataPort](https://ieee-dataport.org/node/4383), [Zenodo](https://zenodo.org/record/5120981)). The GitHub page lists `clean/` (empty-chamber-reference-subtracted, S11) and `simple-clean/` (all scans, including empty-chamber references, plus metadata) and mentions only empty-chamber subtraction as processing; it gives no noise characteristics, VNA model, IFBW, averaging, repeat scans, known issues, or metadata field names ([GitHub page, fetched](https://github.com/UManitoba-BMS/UM-BMID)). A search summary says it could not confirm S21 files on that page; the project's own data does contain S21.
- Dataset descriptor to read next: Reimer, Krenkevich, Pistorius, EuCAP 2020, doi:10.23919/EuCAP48036.2020.9135659 ([IEEE](https://ieeexplore.ieee.org/document/9135659/)). NOT read.
- Other-system SNR/dynamic range figures found are datasheet-level, not scan-level: a low-cost receiver specified at noise floor -148.3 dBm/Hz, dynamic range 86 dB ([Pagliari et al., BioCAS 2015](https://www.sld.cs.columbia.edu/pubs/pagliari_biocas15.pdf), search summary, UNVERIFIED); a tomography phantom study quoting a VNA with >130 dB dynamic range (instrument ceiling, from a search summary, source not opened). MammoWave characterisation repeated an oil-phantom measurement four times and compared max-to-average values to judge repeatability ([LSBU CAMA extended paper](https://openresearch.lsbu.ac.uk/download/84c61e7580d4a682a598c502cc1b1b4d6b378fb706d085e8d19b269b671785bf/652811/CAMA-Extended-paper_Tiberi.pdf), search summary, UNVERIFIED).
- A four-antenna 3D-printed phantom dataset (Keysight MN7021A, 791 points 0.1-8 GHz) has 16 repeated sweeps per tumour position and 16 no-tumour baseline sweeps per session, so within-session noise is directly measurable there; the page reports no numeric noise results ([IEEE DataPort](https://ieee-dataport.org/documents/four-antenna-s-parameter-measurements-3d-printed-breast-phantoms-machine-learning-tumor), fetched).

### Formulas (ours; not validated)
```python
# --- time-domain late-window noise floor (upper bound on noise)
w = np.hanning(F)[:, None]                         # or Kaiser; state the window, it sets ENBW
x = np.fft.ifft(S * w, n=1024, axis=0)             # S: (F, A) -- use S_t (raw) AND D separately
late = t > t_late                                  # t_late > last physical return: 2*(2*r_ant)*sqrt(eps)/c + margin
P_noise = np.median(np.abs(x[late])**2, axis=0)    # (A,) median is robust to isolated multipath spikes
P_sig   = np.max(np.abs(x[~late])**2, axis=0)
snr_td_db = 10*np.log10(P_sig / P_noise)           # per-antenna peak SNR
# --- out-of-band / spectral-smoothness noise (no extra band needed: 1-8 GHz is the whole band)
# frequency-domain "high-quefrency" noise: energy of the TD samples that cannot be physical
# --- frequency-domain 2nd-difference noise (physical S21 is smooth over >= ~10 MHz):
d2 = np.diff(S, n=2, axis=0)                       # (F-2, A)
sigma_fd = 1.4826*np.median(np.abs(d2 - np.median(d2, 0)), axis=0) / np.sqrt(6)  # for white noise Var(d2)=6*sigma^2
snr_fd_db = 20*np.log10(np.sqrt((np.abs(S)**2).mean(0)) / sigma_fd)
```
The `sqrt(6)` factor is the exact variance gain of a second difference of white noise (1, -2, 1 -> 1+4+1 = 6). It underestimates noise if noise is correlated across adjacent frequencies and overestimates it if the true response has sharp resonances (the chamber/antenna may). Units and the Hann ENBW must be reported when comparing against a repeat-scan noise estimate.
Cross-check against the gold standard when a pair exists: for two scans of an unchanged scene, `sigma_rep = np.std(S_a - S_b, axis=0)/np.sqrt(2)`; compare `sigma_fd` and the late-window estimate with `sigma_rep` over healthy pairs to calibrate how biased the single-scan estimates are.
Noise-equivalent power at the VNA receiver is `N0*IFBW` (kTB-like) referred to the source power. Only a per-scan amplitude statistic is computable from S21; absolute NEP needs source power and IFBW from the instrument settings (not in the pickle as far as known).

### Inferences
- The two single-scan estimators (late-time median power, second-difference MAD) have distinct failure modes (multipath vs smoothness violations), so reporting both and flagging disagreement is cheap.
- Any per-scan SNR must be validated against repeatability pairs in the data before use; no paper validates this for UM-BMID.

### Gaps
- Typical SNR / noise floor / IFBW / averaging / sweep time for UM-BMID Gen-2 not found (EuCAP 2020 paper and README unread).
- No Keysight/R&S application note with formal noise floor vs trace noise vs dynamic range definitions retrieved.
- Whether UM-BMID contains repeated identical scans (needed for ground-truth noise) is unknown.

---

## 2. Drift and reference mismatch

### Takeaway
Drift is documented as a real problem in multistatic and time-domain systems and there are calibration-based fixes (Politecnico di Torino 2026), but I found no published scalar "drift score" computed from the target and its reference. The most defensible data-only drift flag is the residual energy `D` in the time window before the earliest physically possible change (formula ours, extends the earlier note). Rotation subtraction is a known artifact-removal method whose rotation angle and off-axis-tumour assumption can degrade images.

### Cited Findings
- A 2026 Politecnico di Torino paper proposes a simplified reflection-based calibration for a multistatic microwave imaging system using time-domain reflection signatures of the whole system plus an extended two-port calibration in an auxiliary channel to correct temporal drift. Source: [Polito IRIS PDF](https://iris.polito.it/retrieve/1332ab7b-5c02-40b9-a60b-72c61d10422a/Simplified_Reflection-Based_Calibration_for_a_Multistatic_Microwave_Imaging_System_Submitted.pdf) (search summary only; page failed to load; numbers not read; UNVERIFIED).
- A four-antenna dataset deliberately includes drift tests: cross-day measurement, fresh VNA calibration, antenna reattachment, antenna-position swap, canola-oil replacement, same-day reference set, and a null control; 32 sessions; each scenario is a "self-contained leave-one-session-out evaluation set". The page reports no drift magnitudes. Source: [IEEE DataPort](https://ieee-dataport.org/documents/four-antenna-s-parameter-measurements-3d-printed-breast-phantoms-machine-learning-tumor) (fetched). This is a candidate external dataset for validating a drift score; not UM-BMID.
- Rotational subtraction (subtract the signal recorded at a rotated position): a 2020 O'Loughlin et al. paper examines its theory and possible negative effects on image quality, concludes the rotation parameters need optimising and that the key assumption is that the tumour is off the rotation axis ([IEEE 9232244](https://ieeexplore.ieee.org/document/9232244), [URSI summary](https://2021.ursi.org/proceedings/procGA20/papers/YSASummaryOLoughlin.pdf); search summary only, pages did not load, UNVERIFIED). A 2021 EuCAP paper (Benchakroun, O'Halloran, O'Loughlin) uses simulation to show the chosen rotation angle can reduce image quality and detection ([Galway repository](https://research.universityofgalway.ie/en/publications/impact-of-rotational-artefact-removal-on-microwave-breast-images-2/), snippet, UNVERIFIED).
- Early-time artifact removal literature: Elahi et al. extend a monostatic adaptive artifact removal to multistatic signals, where the artifact varies more across channels and is harder to estimate ([BL catalogue](https://eld.bl.uk/catalog/100045370829_0x000001); [Galway](https://research.universityofgalway.ie/en/publications/adaptive-artifact-removal-for-selective-multistatic-microwave-bre-14/); snippets, UNVERIFIED). Hybrid artifact removal combining two methods: [Galway](https://research.universityofgalway.ie/en/publications/hybrid-artifact-removal-for-confocal-microwave-breast-imaging-18/) (snippet). Lavoie, Porter, Fear et al. compared radar algorithms on experimental phantoms with realistic early-time artifact removal ([CAS catalogue](https://katalog.lib.cas.cz/KNAV/EdsRecord/edsair,edsair.doi...........ff0b6fffaabfdc2e985810cf5a53db7f), snippet, UNVERIFIED).
- A search for Kurrant / Rodriguez-Herrera skin-artifact removal papers returned nothing from those authors; they are not verified here. The Rodriguez-Herrera phase-centre formula is only cited from the project's own CLAUDE.md.
- Microwave time-domain screening: a paper on "identification and compensation of measurement-induced uncertainties" exists ([catalogue record](https://bvbvf1.bib-bvb.de/vufind/PrimoRecord/cdi_gale_infotracmisc_A353518554), title only, UNVERIFIED).

### Formulas (ours; not validated)
```python
# 1. pre-signal residual energy (reference validity). t_skin_min = earliest possible echo, from 2*(r_ant - r_shell_max)/c*sqrt(eps_coupling)
pre = t < t_skin_min - margin
d_t = np.fft.ifft(D * w, n=1024, axis=0); s_t = np.fft.ifft(S_t * w, n=1024, axis=0)
drift_pre = (np.abs(d_t[pre])**2).mean(0) / (np.abs(s_t[~pre][:50])**2).max(0)   # (A,) normalized
# 2. low-time-lag drift: smooth gain/phase change between target and reference (cable / VNA drift is mostly a complex gain)
g = (np.conj(S_r) * S_t).sum(0) / (np.abs(S_r)**2).sum(0)          # per-antenna complex LS gain, (A,)
resid_after_gain = np.linalg.norm(S_t - S_r*g, axis=0) / np.linalg.norm(S_t, axis=0)
gain_dev = np.abs(g) - 1; phase_dev = np.angle(g)                   # report robust z across 72 antennas
# 3. delay drift (cable length / temperature): slope of phase(S_t/S_r) vs frequency
ph = np.unwrap(np.angle(S_t*np.conj(S_r)), axis=0)
tau_shift = -np.polyfit(2*np.pi*fg*1e9, ph, 1)[0]                   # (A,) seconds; fit over a band that contains the strong skin echo
# 4. reference gain-correction sanity: residual with gain-correction vs without; large improvement => drift present
improve = np.linalg.norm(S_t - S_r, axis=0) / np.linalg.norm(S_t - S_r*g, axis=0)
```
Caveat: item 2 (LS gain) can absorb tumour signal if the tumour response is correlated with the reference; fit the gain on the pre-skin window or on early time only, then apply. Item 3 mixes physical echo delay differences (different phantom radius) and drift; use only for same-shell references.
Repeatability (healthy-minus-healthy residual): `||H_i - H_j||_F / ||H_i||_F` for same-shell, same-radius healthy scans, as a noise-plus-drift floor. Whether UM-BMID has such pairs is unverified.

### Inferences
- Gain/phase/delay drift between target and reference are the physically expected VNA/cable drift modes (a complex scalar plus a small linear phase). They are separable from a tumour echo, which is localized in time and varies across antennas. A smooth-vs-localized decomposition of `D` is therefore a plausible drift test; it is untested.
- Rotation subtraction is a technique that needs a tumour off-axis. It is not applicable as is to bistatic UM-BMID with adipose-reference subtraction, but the same assumption failure (reference differs from target by more than the tumour) is the drift issue.

### Gaps
- No peer-reviewed paper quantifying VNA/cable/phantom drift between target and reference scans for UM-BMID or a similar system was read in full.
- Polito 2026 calibration paper, O'Loughlin 2020, Elahi adaptive artifact removal not read; Kurrant and Rodriguez-Herrera artifact methods not retrieved.
- No published threshold for "acceptable" pre-signal residual.

---

## 3. Clutter rank diagnostics (SVD / TVSVD / PCA)

### Takeaway
SVD clutter-subspace removal is established in breast microwave imaging (I-MUSIC, adaptive SVD) and in through-the-wall radar (Tivive and Bouzerdoum), but rank choice is mostly fixed-index or heuristic, and a 2025 radar paper states that the singular-value threshold choice is under-discussed. I found no paper that uses the number of removed singular values, the energy in the first singular vectors, or the spectrum slope as a scan-quality score, and no TVSVD microwave breast source.

### Cited Findings
- I-MUSIC (Pagliari et al.): SVD of the data matrix, discard the two dominant singular values to remove skin-interface clutter ([BioCAS 2015 PDF](https://www.sld.cs.columbia.edu/pubs/pagliari_biocas15.pdf), search summary; already in the earlier note).
- Dry-setup breast imaging: adaptive SVD artifact removal using the similarity between consecutive antennas' signals to remove the strong first (skin) reflection ([ISCTE PDF](https://repositorio.iscte-iul.pt/bitstream/10071/20915/1/Heterogeneous%20breast%20MWI%20v5_3rdSub%20v2.pdf), search summary; page failed to load; selection rule not read, UNVERIFIED). A related ISCTE conference paper: [ISCTE 35432](https://repositorio.iscte-iul.pt/bitstream/10071/35432/1/conferenceObject_112569.pdf) (title only).
- Subspace-based S-TwIST for microwave tomography proposes an adaptive strategy for choosing singular values (not fixed) ([PMC11902650](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11902650/), snippet, UNVERIFIED; tomography, not radar). A TSVD analysis of microwave inverse scattering for breast imaging exists ([record](https://vufind-test.katalog.k.utb.cz/SummonRecord/FETCH-LOGICAL-c574t-7e3f747ef018793936d73fb8543995ae7a3669ba64031bfa5d9a0135fbe13f443), title only). These are truncation of the inverse problem, not clutter-rank diagnostics.
- Through-wall radar: Tivive and Bouzerdoum, IEEE TGRS 2015, doi:10.1109/TGRS.2014.2355211: SVD of the space-frequency matrix estimates the wall subspace, orthogonal subspace projection removes it, no background scene needed, target-to-clutter improvement reported close to background subtraction ([UOW record](https://ro.uow.edu.au/eispapers/3043), summary). The SPAWC 2013 follow-up identifies dominant wall singular components from the singular-value spectrum ([UOW 1633](https://ro.uow.edu.au/eispapers/1633/)); a summary says an earlier approach treated the first singular value as wall and the second eigen-image as target ([core.ac.uk 36998005](https://core.ac.uk/download/36998005.pdf), fragmentary snippet, UNVERIFIED). HOSVD variants: [HBKU record](https://elmi.hbku.edu.qa/en/publications/wall-clutter-mitigation-using-hosvd-in-through-the-wall-radar-ima/).
- A 2025 adaptive-SVD radar clutter paper states the SVD threshold selection "has not been discussed in detail in the literature for radar signal processing" and proposes an SVM subspace classifier (search summary of [MDPI Remote Sensing 17, 2697](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-17-02697/article_deploy/remotesensing-17-02697.pdf); UNVERIFIED, not opened).
- GPR: Hankel-matrix SVD denoising selects effective singular values ([PMC6210738](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6210738/)); low-rank-plus-sparse decomposition with SVD thresholding ([BIT record](https://pure.bit.edu.cn/en/publications/clutter-removal-method-for-gpr-based-on-low-rank-and-sparse-decom)); window-length-optimised Hankel SVD ([Sensors 19(17):3807](https://mdpi.com/1424-8220/19/17/3807)). Search summaries only. No cumulative-energy (90/95 percent) rank rule was found validated.

### Formulas (ours; not validated as quality scores)
```python
U, s, Vh = np.linalg.svd(S_t, full_matrices=False)          # raw scan: (F, A); also do on D
p = s**2 / (s**2).sum()
e1, e2 = p[0], p[:2].sum()                                   # clutter concentration
k95 = int(np.searchsorted(np.cumsum(p), 0.95) + 1)           # rank for 95 % energy
slope = np.polyfit(np.log(np.arange(2, 20)), np.log(s[2:20]), 1)[0]   # decay of the tail after the 2 clutter modes
# noise edge: Marchenko-Pastur upper edge for a (F x A) white-noise matrix, sigma from section 1
# s_mp = sigma_fd * (np.sqrt(F) + np.sqrt(A)); rank_above_noise = (s > s_mp).sum()
# structure of the leading singular vectors: skin clutter => Vh[0] smooth over antenna angle
v0 = Vh[0]; smooth0 = np.abs(np.fft.fft(v0))[1:4].sum() / np.abs(np.fft.fft(v0))[1:].sum()
# separation of residual after removing k modes:
Dk = S_t - (U[:, :2] * s[:2]) @ Vh[:2]                       # 2 modes removed
frac_removed = 1 - (np.abs(Dk)**2).sum() / (np.abs(S_t)**2).sum()
```
A rotation-invariant physical expectation: a perfectly centred circular shell makes the skin clutter a near rank-1 (or rank-few) matrix across antennas; off-centre phantoms or position errors raise the rank. So `e1`, `k95`, and `rank_above_noise` of `S_t` partly measure shell eccentricity and antenna-position error, not only noise. This reasoning is ours, not from a source.

### Inferences
- Plausible diagnostics: `rank_above_noise` of `D` (should be small and low-rank for a single tumour), `e1` of `S_t` (skin dominance), `slope` and `frac_removed`. All are cheap. All are confounded by shell size and radius class, so evaluate against healthy-minus-healthy pairs of the same shell (negative control).
- A rank selection by Marchenko-Pastur edge is more principled than "remove 2" when `sigma` is estimated per scan, but it requires the Section 1 noise estimate to be unbiased.

### Gaps
- No microwave breast or radar paper found that uses removed-rank, energy fraction or spectrum slope as a scan-quality indicator.
- No TVSVD source for microwave breast imaging.
- None of the SVD papers' rank selection rules was read in full; Tivive 2015 selection rule not retrieved.

---

## 4. Antenna and coupling anomalies; outlier channels; failure modes in bistatic data

### Takeaway
No standard pre-reconstruction channel-quality check (per-channel SNR threshold, reciprocity, outlier rejection) was found for microwave breast imaging. Related literature exists in array diagnostics and in imaging-array element-failure analysis, and shows tolerance to failed channels is substantial. UM-BMID-specific failure modes are unknown from sources; the only facts are from the project's own audit.

### Cited Findings
- For active incoherent microwave imaging arrays, failure of individual receiver elements gives gradual image degradation rather than pixel loss, and RMSE below 10 percent was maintained with failure rates up to 25 percent in measured data. Source: [Element failures in active incoherent microwave imaging arrays (IEEE letter, 2019)](https://ersearch2.cvtisr.sk/vufind/SummonRecord/FETCH-LOGICAL-c293t-e091718b56089c23d4469bac8936fd11001a4919512d7cb245b543f76059629b3?lng=en) (search summary, UNVERIFIED; not a breast system).
- Blind diagnosis of large antenna arrays treats faults as a sparse deviation of the channel vector ([arXiv 2101.09952](https://arxiv.org/pdf/2101.09952), summary; mmWave communications).
- Array diagnostics from far-field measurements via Lebesgue-space inversion ([OJAP 2022](https://iris.unige.it/retrieve/ec4158c8-8f21-4020-9db0-1310f4c3fbca/OJAP2022_Array_Diagnostics.pdf), summary).
- Small antenna-modelling errors can significantly alter time-domain tomography reconstructions ([PMC3625614](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3625614/), summary). Supports checking antenna model and position errors.
- The skin artifact varies across channels in realistic scenarios and many removal algorithms assume it is common ([Galway repository, from the earlier note](https://researchrepository.universityofgalway.ie/entities/publication/c31d84dc-95d1-4359-a0cc-f92eafb926b7)).
- Project facts (not literature): audit found start-angle and radius-unit bugs, `ant_rad` 21/22/23 cm, shells not centred at the chamber origin (echo fit puts some shells 13-15 mm off), and tumours outside the shell radius in 64 of 504 scans (CLAUDE.md progress log).

### Formulas (ours; not validated)
```python
# robust z across the 72 positions of any per-antenna statistic q (A,)
def rz(q): med = np.median(q); mad = 1.4826*np.median(np.abs(q-med)); return (q-med)/(mad+1e-30)
# candidate per-antenna statistics (each is (A,))
q_dead  = np.log((np.abs(S_t)**2).mean(0))                   # dead / very weak channel => very negative z
q_noisy = np.log(sigma_fd)                                   # from Section 1
q_late  = np.log(P_noise)                                    # late-window noise
q_skin  = tau_skin                                           # skin delay vs angle; outliers = bad position / cable
q_coup  = (np.abs(np.fft.ifft(S_t*w, n=1024, axis=0)[t < t_direct])**2).sum(0)  # direct-coupling leakage window
bad = (np.abs(rz(q_dead)) > 3.5) | (np.abs(rz(q_noisy)) > 3.5)  # 3.5 is the usual Iglewicz-Hoaglin cutoff (convention, not microwave-validated)
frac_bad = bad.mean()
# cable motion / dropout: consecutive-antenna correlation of skin-echo shape
c = np.array([np.abs(np.vdot(s_t[:, a], s_t[:, (a+1)%72]))/np.linalg.norm(s_t[:, a])/np.linalg.norm(s_t[:, (a+1)%72]) for a in range(72)])
# neighbour-subtraction residual (adjacent positions have almost the same skin echo): large => abrupt change between positions
nbr = np.linalg.norm(s_t - np.roll(s_t, -1, axis=1), axis=0) / np.linalg.norm(s_t, axis=0)
```
Bistatic port offset: for S21 with port B offset by +60 degrees, neighbour subtraction of adjacent positions compares signals whose both legs move by 5 degrees; it is not a mirror-symmetric comparison like monostatic S11. A smooth `q_skin(angle)` sinusoid-plus-constant model (shell off-centre produces a first-harmonic variation) leaves a residual whose robust z flags position errors; fit with `np.linalg.lstsq` on `[1, cos, sin]` columns.

### Inferences
- Per-antenna robust z-scores are standard statistics and cheap. The count of flagged channels is a reasonable per-scan indicator, but with no published threshold for UM-BMID and the observation that imaging tolerates many failed channels, a scan with a few flagged channels need not be less reliable. This must be tested, not assumed.
- Shell eccentricity, a known local issue, imprints on `q_skin` and the SVD rank, so remove its first-harmonic before flagging outliers.

### Gaps
- No known UM-BMID channel failure catalogue; the repository issue tab shows one open issue whose content was not visible.
- Reciprocity check for S21 vs S12: UM-BMID Gen-2 apparently has S21 only (S12 not known); not available.
- No source on direct-coupling leakage diagnostics for breast imaging with a bistatic offset.

---

## 5. Evidence that scan-quality scores predict imaging success (and negative results)

### Takeaway
I found no paper in microwave breast imaging or radar showing that a ground-truth-free scan-quality score predicts detection, localization error or SCR. Closest evidence is that quality-sensitive pieces of the pipeline (effective permittivity, antenna model, artifact removal rotation angle, element failures) change image quality, which motivates but does not validate a scan-level score. The project's own tests are negative-to-weak.

### Cited Findings
- Effective-permittivity estimation from image-quality fitness affects accuracy of radar images (Lavoie, Okoniewski, Fear, PLOS ONE 2016, [PMC5017770](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5017770/), abstract-level; already in the earlier note).
- Rotation-angle choice in rotational artifact removal can reduce image quality and tumour detection (simulation) ([Galway repository](https://research.universityofgalway.ie/en/publications/impact-of-rotational-artefact-removal-on-microwave-breast-images-2/), snippet, UNVERIFIED).
- Imaging arrays degrade gradually with receiver failures (RMSE below 10 percent up to 25 percent failure) ([IEEE letter record](https://ersearch2.cvtisr.sk/vufind/SummonRecord/FETCH-LOGICAL-c293t-e091718b56089c23d4469bac8936fd11001a4919512d7cb245b543f76059629b3?lng=en), UNVERIFIED). This suggests a channel-outlier count alone may weakly predict success.
- Tivive and Bouzerdoum's through-wall SVD method reports target-to-clutter improvement close to background subtraction ([UOW record](https://ro.uow.edu.au/eispapers/3043)); this is a performance claim for the removal method, not a diagnostic that predicts success.
- Project's own results (CLAUDE.md): sub-aperture and sub-band consistency AUROC about 0.59-0.61 (CI lower bounds 0.51), peak margin, CF, artifact prior all CI containing 0.5; composite OOF AUROC 0.609; old RRS-MBI ranks inversely (AUROC 0.434) because accepted scans have more central tumours, where the pipeline equals the trivial "predict the centre" LE.

### Inferences
- Negative-control design is essential: a quality score is only useful if it predicts LE after conditioning on tumour distance from centre (central tumours are easy for "predict the centre"). Evaluate by stratifying on true distance, and compare with the trivial centre predictor.
- Given six candidates already tested on the same 504 scans, new scores in this note multiply multiplicity; pre-register at most 2-3 (suggested: late-window noise SNR, pre-signal drift residual after gain correction, `rank_above_noise` of `D`) and correct (Holm).

### Gaps
- No paper links any noise/drift/clutter-rank score to localization error or SCR in microwave breast imaging; absence of evidence is from limited access (many pages did not load), not a proven absence.
- Cross-field: radar-quality literature (e.g. through-wall SVD) frames rank selection as a processing choice, not as a score; no negative-result paper found.

---

## Overall summary for the report writer
- Well-supported building blocks: Keysight rule on IFBW/averaging vs noise floor (read directly); SVD clutter removal used in I-MUSIC (remove 2) and through-wall radar (dominant components identified from the spectrum); rotational subtraction rests on an off-axis assumption; adaptive artifact removal for multistatic signals; a public four-antenna drift-test dataset (cross-day, recalibration, reattachment, oil replacement) usable as an external check.
- Everything labelled "Formula (ours)" is a recipe, not validated in the literature. The strongest self-contained candidates: second-difference MAD noise estimate cross-checked against late-window median power; complex-gain-corrected residual and pre-skin residual for drift; Marchenko-Pastur-referenced rank of `D`; robust-z outlier counts after removing shell-eccentricity harmonics.
- Main unread primary sources that would close gaps: Reimer et al. EuCAP 2020 and `docs/UM-BMID_README.txt` (noise, IFBW, calibration, repeats); Polito 2026 drift calibration; O'Loughlin 2020 and Elahi adaptive artifact removal; Tivive and Bouzerdoum TGRS 2015; Pagliari BioCAS 2015.
- Every search summary used here is marked UNVERIFIED where the page did not load. The only fully read pages: Keysight noise-reduction help, UM-BMID GitHub README page, IEEE DataPort drift dataset page.
