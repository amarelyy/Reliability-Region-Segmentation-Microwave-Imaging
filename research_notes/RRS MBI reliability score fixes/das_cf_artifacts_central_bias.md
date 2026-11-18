# DAS beamforming and Coherence Factor in microwave breast imaging: central bias, shell/skin artifacts, and consequences for reliability scores

Research method note: about 13 web calls. Search tools returned summaries; several primary papers (Mallart & Fink 1994, Li & Li 2003, Camacho 2009, Lim 2008, Fear 2002, Bond 2003) were NOT read in full text. Their claims below are limited to what secondary pages stated. Anything else is UNVERIFIED or placed under Inferences (my own reasoning, not a literature claim).

## (a) Why DAS peaks near the array centre for weak or deep targets

### Takeaway
I found NO source that directly states "DAS images of a ring array are biased to the centre". The effect is an inference from the DAS summation geometry plus the documented facts that deep-tumour returns are weak and that residual clutter dominates. Treat the central-bias mechanism as a physics argument that needs a synthetic test, not as a literature result.

### Cited Findings
- DAS sidelobes are an inherent property of the scheme, and the sum is only partly coherent near the peak, degrading resolution — [Cross-correlated back projection, Improved Resolution and Reduced Clutter in UWB Microwave Imaging (PMC3035806)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3035806/)
- That paper also states that DAS clutter-like sidelobes arise along the cross-range direction and from contributions of each individual antenna signal, and that it assumes known, constant medium velocity — same source.
- Tumour backscatter signal-to-clutter ratio is expected to be extremely low clinically; late-time (deep) tumour responses can be very weak — search summary of [Elahi et al. 2018, Evaluation of Image Reconstruction Algorithms for Confocal Microwave Imaging: Application to Patient Data, Sensors 18(6):1678, doi:10.3390/s18061678](https://pmc.ncbi.nlm.nih.gov/articles/PMC6022049/) and related results (PMC3035806 listing). The precise wording was from a search snippet, so the exact source of the "late-time weak" sentence is UNVERIFIED.
- Scattering of real objects depends strongly on incidence/scattering angle; DAS assumes a constant antenna phase centre, only viable within the antenna main lobe — [PMC3035806](https://pmc.ncbi.nlm.nih.gov/articles/PMC3035806/)

### Inferences
- (My reasoning, UNVERIFIED in literature.) In a ring geometry, the sum of a noise or clutter-dominated signal over all Tx-Rx pairs has path-length distributions that are most symmetric at the centre: for the image point at the centre, every bistatic delay equals about 2*R/v for the same radius, so any residual that is a function of antenna-to-antenna range (direct-path leakage, antenna ringing, shell echo with the same time for every channel) aligns in time for the centre pixel. Off-centre pixels need different delays per channel, so such residuals add incoherently. Hence a ring-symmetric time-aligned residual produces a central maximum without any tumour.
- The same logic gives a bias for weak targets: if the tumour contribution is below the aligned residual, the image maximum is the residual's focus, which is the centre (or the shell ring if the shell echo is the strong aligned term). Your own audit data (predictions about 11 mm from centre at every true distance; fixed pipeline still about 9 mm) is consistent with this, and with the "trivial predict-centre" baseline matching the pipeline.
- Test to make this a result rather than a hypothesis: run DAS on target-free scans (adipose-only references) and on simulated noise; histogram the peak position (you already did: image peak at r/R < 0.25 in 62-65 % of adipose-only scans). Also run a point-spread-function map by simulation with one synthetic point scatterer at several radii (sim.py) and measure peak amplitude versus radius.

### Gaps
- No peer-reviewed statement of "central bias" in DAS breast imaging found. Search terms to try next: "point spread function ring array delay-and-sum position dependent", "spatially variant resolution microwave imaging", and Klemm 2010 IEEE TAP "Microwave radar-based differential breast cancer imaging".
- Elahi 2018 full text did not discuss depth effects directly (per fetch).

## (b) Skin/shell reflection and direct-path dominance; gating, adipose-reference subtraction, rotation subtraction

### Takeaway
Skin reflection is the dominant component of the backscattered pulse and must be removed; the literature offers time-shift/scaling subtraction, mean subtraction of aligned returns, neighbourhood-based estimation, RLS and Bond's adaptive artifact estimation. None of these is perfect: residual early-time clutter is itself reported to be amplified by several beamformers.

### Cited Findings
- Skin reflection is expected to be the dominant component of the backscattered pulse; an RLS filter has been used for effective skin-response subtraction in UWB phantom experiments, but as the filter order M was increased the RLS filter also removed tumour response — [PMC3035806](https://pmc.ncbi.nlm.nih.gov/articles/PMC3035806/)
- Antennas are offset from the breast, so the dominant component of recorded returns is the skin reflection; two reduction approaches compared: approximating with time-shifted scaled summed returns from a skin cylinder, and subtracting the mean of aligned returns — Fear & Stuchly, "Microwave detection of breast tumors: comparison of skin subtraction algorithms", SPIE 4129, 2000, [link](https://medicalimaging.spiedigitallibrary.org/conference-proceedings-of-spie/4129/0000/Microwave-detection-of-breast-tumors--comparison-of-skin-subtraction/10.1117/12.390618.full) (from search summary only)
- Fear, Li, Hagness, Stuchly, "Confocal Microwave Imaging for Breast Cancer Detection: Localization of Tumors in Three Dimensions", IEEE Trans. Biomed. Eng. 49(8):812-822, 2002 — [record](https://minds.wisc.edu/items/d9627322-5c45-452b-b9b0-cae5713d8024). The specific artifact-removal procedure in this paper was NOT verified (full text not read).
- Bond, Li, Hagness, Van Veen, "Microwave imaging via space-time beamforming for early detection of breast cancer", IEEE TAP 51(8):1690-1705, 2003, doi:10.1109/TAP.2003.815446 — [record](https://minds.wisc.edu/items/db1e8683-9311-4260-821d-af9eeefb8ffe). The related patent describes removing artifacts such as the skin response or antenna response by estimating it as a filtered combination of the signals at all other antennas — [US7570063B2](https://patents.google.com/patent/US7570063). Attribution to the TAP paper itself is UNVERIFIED.
- Neighbourhood-based skin subtraction (NSS) estimates the artifact at an antenna from a neighbourhood of antennas; "Both CR-DAS and RCB often enhanced the residual clutter from the preprocessing of the signals"; RCB placed a dominant scatterer near the skin — [Elahi et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6022049/)
- Offset-array recording then subtraction reduces skin reflections; background subtraction without tumour is only practical with phantoms — search summary of the review found via [PMC7345629](https://pmc.ncbi.nlm.nih.gov/articles/PMC7345629/) (UNVERIFIED which page said it).
- UM-BMID: "clean" files have empty-chamber reference subtraction applied; "simple-clean" files have none and include empty-chamber reference scans — [UM-BMID repo](https://github.com/UManitoba-BMS/UM-BMID). The repo page itself did not mention adipose references; the Gen-2 adipose-reference design is described elsewhere (UNVERIFIED here; Reimer, Krenkevich, Pistorius, "An open-access experimental dataset for breast microwave imaging", EuCAP 2020; dataset doi:10.21227/1y0z-8t98, [IEEE DataPort](https://ieee-dataport.org/node/4383)).
- Rotation subtraction (subtract the signal at a slightly rotated array/phantom position) matches "recording at an offset position" above but I found NO UM-BMID-specific source for it. UNVERIFIED.

### Inferences
- Early-time gating removes the direct path and the antenna ringing but also removes shallow tumour responses; the gate choice trades clutter for near-skin sensitivity.
- Adipose-reference subtraction removes the common shell echo only if the reference is aligned to the same shell geometry; the Gen-2 phantoms' shell radii differ in position, so misalignment leaves a residual ring. Your audit found the shell centre is not at the chamber origin (about 13-15 mm offset), which would leave exactly this kind of residual.
- A shell-centred residual that is coherent across channels is the primary candidate for a sharply focused ring in the image (your audit: 60 % of blobs outside R after fixes).

### Gaps
- No quantitative comparison of gating vs reference subtraction vs rotation subtraction on UM-BMID Gen-2 found.

## (c) What CF measures; why it can be high for coherent clutter and low for weak targets

### Takeaway
CF is the ratio of coherent-sum energy to incoherent-sum energy across channels; it measures channel-to-channel phase agreement at the assumed delays, not whether the source is a tumour. Anything coherent at the assumed geometry (shell echo, antenna residual, a systematic geometry error that is common to all channels) gets high CF, while a weak target buried under noise gets low CF.

### Cited Findings
- Mallart & Fink, "Adaptive focusing in scattering media through sound-speed inhomogeneities: The van Cittert-Zernike approach and focusing criterion", J. Acoust. Soc. Am. 96(6):3721-3732, 1994 — a focusing criterion based on the van Cittert-Zernike theorem; the focusing parameter C is called the coherence factor; in fully developed speckle without clutter C tends to 2/3 and drops when focus quality drops; VCZ is valid only for distortionless propagation — secondary summary via [IEEE IUS 2004 abstract list](https://ewh.ieee.org/conf/ius_2008/2004_ussymp_web/fri.pdf) and [arXiv 2103.02029](https://arxiv.org/pdf/2103.02029). Primary text NOT read.
- Li & Li, "Adaptive imaging using the generalized coherence factor", IEEE TUFFC 50(2):128-141, 2003, doi:10.1109/TUFFC.2003.1182117 — [NTU repository](https://scholars.lib.ntu.edu.tw/handle/123456789/318076). GCF uses the spatial DFT of aperture data and a low-frequency fraction to be more robust than CF; abstract-level only.
- CF per pixel = amplitude of coherent sum over incoherent sum; low CF is treated as clutter and the pixel is suppressed, so a real but weak or low-coherence scatterer is dimmed along with clutter — [ultrasound_metrics docs](https://forest-neurotech-ultrasound-metrics.readthedocs-hosted.com/latest/example_gallery/plot_coherence_factor.html) and search summary of [US7744532](https://patents.google.com/patent/US7744532).
- Camacho, Parrilla, Fritsch, "Phase coherence imaging", IEEE TUFFC 56(5):958-974, 2009 — phase coherence factor (PCF) and sign coherence factor (SCF), aimed at grating and side-lobe suppression in sparse arrays — [ITEFI record](https://www.itefi.csic.es/en/publicaciones/phase-coherence-imaging). DOI UNVERIFIED (not retrieved).
- In breast microwave imaging: CF was adapted from ultrasound; "the coherence metrics for IDAS and CF-DAS may reward responses from non-lesion regions within the breast"; coherence-based algorithms often fail to correctly localise the malignancy in multiple-lesion and heterogeneous breasts; they suppressed clutter but misplaced the dominant response — [Elahi et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6022049/)
- A coherence-factor weighted DAS reduces clutter and improves the peak clutter-to-target energy ratio by 3.1 dB in the hardest example in that work — search summary, source paper UNVERIFIED (a Klemm/Bristol study; likely [Klemm et al., Improved Delay-and-Sum Beamforming Algorithm for Breast Cancer Detection, Int. J. Antennas Propag. 2008, doi:10.1155/2008/761402](https://doaj.org/article/223c62b45caa43679505775429522dd9), attribution not confirmed).
- Lim, Nhung, Li, Thang, "Confocal microwave imaging for breast cancer detection: Delay-multiply-and-sum image reconstruction algorithm", IEEE Trans. Biomed. Eng. 55(6):1697-1704, 2008 (citation as listed in later papers; DOI not retrieved, UNVERIFIED). DMAS multiplies channel pairs ("virtually increases the number of signals from M to M choose 2") and gave better clutter suppression than DAS; SMR improved 44 % in one patient-data comparison — [Elahi et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6022049/)

### Inferences
- (Mathematical, from the CF definition.) CF = |sum s_i|^2 / (N * sum |s_i|^2) is invariant to the amplitude of the signal. A strong shell echo with a small timing error common to all channels still gives high CF at the shell pixels. A systematic geometry error (wrong start angle, wrong radius) shifts delays in the same direction for all channels; the CF then peaks at the displaced focus of the dominant coherent scatterer, which is the shell. High CF therefore confirms coherence, not correctness.
- Low CF for weak real targets: with SNR below 0 dB per channel, the coherent sum has random-phase terms and CF is near 1/N regardless of whether a target is there.
- Conditions where high CF plausibly indicates a correct detection (my reasoning, UNVERIFIED): (1) shell/direct-path residual removed to below the tumour level, e.g. verified on adipose-only scans (CF map flat); (2) geometry validated on a known point target (sim.py, calibration phantoms); (3) CF at the predicted point exceeds the CF distribution of target-free scans at that same radius. Your audit result that CF is about 3x the breast mean within 10 mm of the centre in adipose-only scans shows (3) is not met: CF has a location prior baked into it.

### Gaps
- No source found that explicitly evaluates CF on shell-dominated data or demonstrates "high CF clutter" in microwave breast imaging. Only Elahi 2018's warning (rewarding non-lesion responses) supports it.
- Primary papers for Mallart & Fink, Li & Li and Camacho not read.

## (d) Effect of wrong permittivity/geometry (phase-centre error, wrong radius, wrong start angle)

### Takeaway
The literature establishes that DAS assumes a known constant velocity and a fixed antenna phase centre; both fail in practice. Quantitative sensitivity to start angle or radius for UM-BMID is not in the sources I found, only in your own audit.

### Cited Findings
- Assumed uniform, known velocity; robustness to heterogeneous media untested — [PMC3035806](https://pmc.ncbi.nlm.nih.gov/articles/PMC3035806/)
- Fixed average permittivity (er = 9) "may not be an accurate estimate for all patients"; delays based on assumed average propagation speed; flagged as a limitation especially for dense breasts, effect not quantified — [Elahi et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6022049/)
- Constant phase centre irrespective of angle and frequency is viable only within the main lobe of the antenna pattern — [PMC3035806](https://pmc.ncbi.nlm.nih.gov/articles/PMC3035806/)
- Klemm et al. report inhomogeneity of the sensed volume is a bigger challenge than dielectric contrast — [Bristol Klemm record](https://research-information.bris.ac.uk/en/publications/microwave-radar-based-breast-cancer-detection-imaging-in-inhomoge/)
- UM-BMID scan geometry docs (start angle -130 deg, phase-centre formula): the repo README did not contain them (fetch result); the `UM-BMID/docs/scan_geometry.md` file was not fetched here. UNVERIFIED externally; the project's CLAUDE.md quotes it.

### Inferences
- (Geometry.) A start-angle error rotates the image by the same angle; a radius error of dr changes every two-way delay by about 2*dr/v, which at er about 9 (v about 1e8 m/s) is about 0.67 ns per cm... correction: 2*0.01 m / 1e8 m/s = 200 ps per cm, comparable to a fraction of the pulse width at 1-8 GHz, so a few mm already produce phase errors of order a cycle at 8 GHz (7 GHz-band wavelength in er 9 is about 14 mm; 1 cm radius error = 2 cm round trip = 1.4 wavelengths). Therefore a radius error of several mm is enough to destroy coherence at the upper band, which is why your 4-6 GHz sub-band helps (UNVERIFIED arithmetic assumption er = 9, check).
- A timing error that is the same for all channels (radius) moves the focus of ring-symmetric scatterers radially; a start-angle error moves it azimuthally. Only the shell and other ring-symmetric structures remain coherent under such errors, which again favours shell/centre peaks.

### Gaps
- No UM-BMID Gen-2 sensitivity-to-geometry study found.

## (e) Tumour-position-dependent SCR degradation

### Takeaway
Literature supports that near-skin tumours suffer from skin clutter and deep tumours from attenuation; my searches did not find a direct quantification of SCR versus radial position. Your own audit (SCR in GT disc -0.32 dB/mm, r -0.55 baseline) is the only quantification at hand.

### Cited Findings
- Skin is a major challenge; tumours close to the skin need additional consideration — search summary of [PMC7345629](https://pmc.ncbi.nlm.nih.gov/articles/PMC7345629/) (exact sentence source UNVERIFIED).
- Deep tumour late-time response can be very weak, so clinical SCR is expected to be extremely low — as in (a).
- Responses near the skin were a recurring clutter problem for beamformers (RCB "dominant scatterer near the skin") — [Elahi et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6022049/)
- RLS trade-off: raising filter order removes tumour response too — [PMC3035806](https://pmc.ncbi.nlm.nih.gov/articles/PMC3035806/)

### Inferences
- Two opposite position effects compete: near the skin, tumour response overlaps the skin/shell residual (clutter up, and subtraction may remove it); deep or central, attenuation reduces signal but the DAS geometry is most coherent. Which wins depends on the pipeline; for UM-BMID with fixed 21-23 cm antenna radius, tumours far from the centre are close to the shell and tend to coincide in time with the shell echo and its residual.

### Gaps
- No quantitative SCR-vs-radius study found.

## Key questions answered

### Is a "central concentration" penalty (CC) physically justified?

#### Takeaway
Not as a measure of tumour plausibility. Physics says a DAS+CF image of weak data concentrates at the centre because of method geometry; the data say the same. CC would then penalise outputs that are less biased, and reward the bias. Your audit found R ranks inversely (rho(R, LE) +0.19).

#### Cited Findings
- No source found that endorses a central-concentration penalty. Elahi 2018's warning that coherence-based metrics may reward non-lesion regions is the nearest relevant statement — [PMC6022049](https://pmc.ncbi.nlm.nih.gov/articles/PMC6022049/)

#### Inferences
- CC is defensible only as a detector of "image energy diffuse over many pixels" when the method has no spatial prior. Here the method has one. Use it only after normalizing against the same score on target-free (adipose-reference) scans at the same location.

### Is geometric plausibility (distance from centre / inside breast) a reasonable prior, and what does it do under centre bias?

#### Takeaway
A plausibility prior is reasonable in principle (tumours are inside the breast), but a prior that favours the centre rewards exactly the method's bias; it makes the score agree with the biased estimate and inflates apparent reliability for central predictions. Your data: 64/504 tumours lie outside their own recorded shell radius and the predicted points stay near 11 mm from centre.

#### Cited Findings
- Not found in literature; UNVERIFIED. Only supporting data is internal to the project.

#### Inferences
- A inside-the-breast hard constraint (do not predict outside the shell) is fine; a graded penalty that increases with radius is not, because the true distribution in UM-BMID spans the whole breast radius.

### Under what conditions does high CF indicate a correct detection vs coherent clutter?

See (c) Inferences: validated geometry, shell residual removed, and CF at the peak above the target-free CF distribution at the same radius. Literature support is limited to [Elahi et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6022049/) (coherence metrics may reward non-lesion regions) and the CF definition ([docs](https://forest-neurotech-ultrasound-metrics.readthedocs-hosted.com/latest/example_gallery/plot_coherence_factor.html)).

### What does the literature say about peak-to-sidelobe ("peak dominance") as a detection confidence?

#### Takeaway
Found nothing that validates peak dominance as a detection-confidence metric in microwave breast imaging. Related: DAS has inherent non-negligible sidelobes, and improvement metrics reported in the literature are clutter ratios and SCR/SMR, evaluated against known ground truth, not confidence scores.

#### Cited Findings
- Inherent side lobes of DAS — [PMC3035806](https://pmc.ncbi.nlm.nih.gov/articles/PMC3035806/)
- Reported figures of merit: SMR (signal-to-mean ratio; DMAS +44 % vs DAS) — [Elahi 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6022049/); peak clutter-to-target energy ratio (+3.1 dB for CF-DAS) — see (c), source UNVERIFIED.
- Elahi 2018: lower clutter did not always mean correct localisation (IDAS and CF-DAS suppressed clutter but misplaced the dominant response) — a direct warning against reading "dominant, clean peak" as correct.

#### Inferences
- Peak dominance is a property of the image, not of the truth. If the dominant peak is the shell or centre artifact, high dominance means high confidence in the wrong answer. Your sub-aperture/sub-band consistency candidates (AUROC about 0.59-0.60, weak) are in line with the better-motivated approach: reliability from independent-data agreement rather than image shape.

#### Gaps
- Searches for literature on peak-to-sidelobe ratio as a confidence/uncertainty measure for DAS breast imaging returned nothing; try "image quality assessment microwave breast imaging uncertainty", "false-positive rate breast microwave imaging DAS", and Reimer/Pistorius papers on UM-BMID reconstructions.

## Source reliability summary
- Read/fetched in some depth: Elahi et al. 2018 (PMC6022049), cross-correlated back projection paper (PMC3035806; authors/title/year not captured in the fetch, UNVERIFIED which paper exactly; URL given), UM-BMID repo page.
- Only abstract-level or secondary: Mallart & Fink 1994, Li & Li 2003, Camacho 2009, Lim 2008, Fear 2002, Bond 2003, Klemm 2008/2010.
- Search note: the suggested "O'Loughlin 2018" with UM-BMID was not found; UM-BMID is credited to Reimer, Krenkevich, Pistorius (EuCAP 2020). O'Loughlin appears in a 2024 EuCAP paper on open-source tools ([link](https://edepositireland.ie/handle/2262/108578)). An O'Loughlin 2018 paper may exist (e.g. on breast phantom/algorithm comparison) but was not confirmed.
