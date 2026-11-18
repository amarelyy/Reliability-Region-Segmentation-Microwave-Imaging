# Confidence / reliability / quality metrics for beamformed microwave breast images (UM-BMID, DAS/DMAS/CF)

Method note: about 10 web searches and fetches. Several full texts were blocked (MDPI 403, a PMC bot check, a repository 403), so those sources were seen only as search-snippet summaries. Anything not seen in a source is under Gaps or marked UNVERIFIED.

## Q1. Which published metrics measure reliability or confidence of a single microwave image, and how are they validated?

### Takeaway
The literature mostly uses ground-truth-based image-quality metrics (SCR, SMR, FWHM, localization error). These need the tumour location, so they cannot be computed for a new scan. I found no validated per-scan reliability or confidence score that works without ground truth. Detection is usually judged by whether the global maximum is the tumour (SCR > 0), or qualitatively against clinical reports.

### Cited Findings
- Reimer, Solis-Nepote, Pistorius (2020), "The Application of an Iterative Structure to the Delay-and-Sum and the Delay-Multiply-and-Sum Beamformers in Breast Microwave Imaging", Diagnostics 10(6):411. Sources: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7345629/), [MDPI](https://www.mdpi.com/2075-4418/10/6/411).
  - SMR = 20 log10(Smax / Cmean). Smax is the maximum response in the tumour region, a circle of radius r_t + 5 mm around the known tumour position. Cmean is the mean response in the clutter region.
  - SCR = 20 log10(Smax / Cmax), where Cmax is the maximum response in the clutter region.
  - SCR < 0 means the strongest response lies outside the tumour, so the tumour is "unidentifiable". Identifiable means SCR > 0.
  - Localization error is the distance between the image maximum and the known tumour centre.
- Same paper, results. The data were 18 scans of UM-BMID-type phantoms (9 adipose-fibroglandular combinations, 10 and 15 mm lesions).
  - Only 12 reconstructions per algorithm had an identifiable tumour response.
  - Only 1 of the 4 Class IV (densest) reconstructions was identifiable.
  - Mean SCR was about 0 to 5 dB for all four methods (DAS, DMAS, itDAS, itDMAS). For DAS and DMAS it was 0 to 2 dB.
  - SMR was 3 to 30 dB.
  - The authors note that in Class II and III, tumour response was not clearly above the high-intensity central clutter.
  - These numbers are conditional on the identifiable subset.
- Elahi et al. (2018), "Evaluation of Image Reconstruction Algorithms for Confocal Microwave Imaging: Application to Patient Data", Sensors 18(6):1678, doi:10.3390/s18061678. Source: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6022049/).
  - Metrics were SMR (ROI mean over whole-3D-image mean) and FWHM.
  - Detection was qualitative: a high-intensity response matching a clinically identified region in location. There was no numerical threshold.
  - SMR and FWHM were used only to rank algorithms.
  - In the patient with no disease, both DAS and DMAS showed a dominant response, which the authors say would be a false positive without the mammogram.
  - Proposed false-positive rejection (left-right breast comparison, classifying responses by magnitude and shape) is called future work needing large clinical studies.
  - CF-DAS improved average clutter suppression but often put its dominant response in a place inconsistent with the clinical reports. It missed the lesion in 2 of 5 patients (the table says "not detected").
- O'Loughlin, Glavin, Jones, O'Halloran (2018), "Evaluation of Experimental Microwave Radar-Based Images: Evaluation Criteria", IEEE APS-URSI. Source: [record](https://edepositireland.ie/handle/2262/102066). Per the search summary only (full page 403):
  - Some algorithms improve image quality when a tumour is present but can also increase false positives in phantoms with glandular structures.
- O'Loughlin et al. (2018), "Sensitivity and Specificity Estimation Using Patient-Specific Microwave Imaging in Diverse Experimental Breast Phantoms", IEEE TMI, doi:10.1109/TMI.2018.2864150. Source: [Galway record](https://research.universityofgalway.ie/en/publications/sensitivity-and-specificity-estimation-using-patient-specific-mic-3/). Per snippet:
  - Achieving high specificity given differences between individuals is a main challenge, and may be harder in dense breasts.
  - The paper has a detection-threshold / ROC style framework (the record is a summary only; details UNVERIFIED).
- Klemm et al., Bristol, IJAP 2008, doi:10.1155/2008/761402. Source: [DOI record](https://0-doi-org.brum.beds.ac.uk/10.1155/2008/761402).
  - A weighting factor improved SCR in focused images by 2.65 dB (10 mm tumour) and 4.4 dB (7 mm tumour).
  - The abstract says the algorithm is "effective in reducing clutter". I could not confirm that this entry is the coherence-factor paper.
- Klemm, Leendertz, Gibbins, Craddock, Preece, Benjamin (2009), "Microwave Radar-Based Breast Cancer Detection: Imaging in Inhomogeneous Breast Phantoms", IEEE AWPL 8:1349-1352. Source: [Bristol](https://research-information.bris.ac.uk/en/publications/microwave-radar-based-breast-cancer-detection-imaging-in-inhomoge/).
  - It reports a 3.1 dB gain in clutter-to-target energy ratio on the hardest phantom.
  - It concludes that inhomogeneity of the sensed volume, not contrast, is the main challenge.
  - It used a 31-antenna, 3 to 10 GHz system.
- Reimer and Pistorius (2023), "Review and Analysis of Tumour Detection and Image Quality Analysis in Experimental Breast Microwave Sensing", Sensors. Source: [PMC10255334](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10255334/).
  - The search summary says it notes that reporting standards vary across studies (this was stated about the Porter and O'Loughlin review). I could not open the full text.
- Physics-driven AI paper on UM-BMID Gen-3 (arXiv 2401.02715). Source: [arXiv](https://arxiv.org/pdf/2401.02715).
  - It counts a localization as correct within a tolerance of tumour radius + 0.5 cm, where 0.5 cm is the phantom position uncertainty.
  - This is a tolerance, not a confidence score.

### Inferences
- Published metrics are oracle metrics. A per-scan score such as RRS-MBI is a different kind of object. The nearest prior art is the SCR > 0 "identifiable" criterion, which also needs ground truth.
- SCR with a 5 mm-padded tumour disc and global clutter maximum is the de facto standard. Any SCR in the RRS work should state the region definitions explicitly, because the choice changes the value.

### Gaps
- I found no paper that validates a no-ground-truth reliability measure against localization error with AUROC or risk-coverage curves.
- I could not read the Reimer-Pistorius 2023 review in full.
- I could not read the O'Loughlin 2018 papers in full.

## Q2. Central bias, skin-reflection residue, and artifact removal in DAS images

### Takeaway
Papers consistently say that early-time skin reflection and residual artifact dominate DAS images unless removed. Reference-based removal (adipose-only phantom scan, empty chamber, neighbouring-antenna estimates) is the standard. Residual clutter, often central, remains and can mask tumours in dense breasts. I found no source that explicitly names "central bias" of the peak location.

### Cited Findings
- Reimer 2020 (Diagnostics 10:411, [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7345629/)):
  - "Ideal skin suppression" subtracts a scan of the adipose-only phantom (inner tissue removed) from the full scan.
  - There was no coupling medium, and air-phantom reflections are large enough to hide inner structures.
  - Residual mismatch still produced notable clutter in DAS, and to a lesser degree in DMAS.
  - Tumour response was not clearly above the high-intensity central clutter in Class II and III. The authors attribute this possibly to central fibroglandular tissue.
  - Tumours were at antenna height with up to 10 mm position uncertainty.
  - A single average propagation speed was assumed (adipose permittivity 6.4).
  - The forward model omits attenuation and inhomogeneous propagation.
- UM-BMID GitHub ([repo](https://github.com/UManitoba-BMS/UM-BMID)):
  - It ships "clean" files with empty-chamber reference subtraction applied, and "simple-clean" files with the reference scans kept.
  - The repo page does not describe tumour localization uncertainty.
- Reimer thesis, U. Manitoba 2020, [hdl:1993/34970](http://hdl.handle.net/1993/34970):
  - The inverse chirp z-transform reduced artifacts, improved contrast and increased detection compared with the inverse DFT.
  - Modeling the system output pulse gave the largest contrast gain.
  - The CNN AUC was 76 to 91 %. The upper end applies only when test phantoms have breast volumes within the training bounds and the tumour is at the same vertical position as the antennas.
- Elahi 2018 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6022049/)):
  - A calibration scan (same positions, no patient) is subtracted.
  - Neighbourhood-Based Skin Subtraction (NSS) estimates each antenna's skin artifact from neighbouring antennas.
  - Laser surface data restrict the imaging domain.
  - The article does not mention "rotation subtraction".
- Elahi's 2017 URSI paper and 2018 thesis ([Galway](https://research.universityofgalway.ie/en/publications/comparison-of-radar-based-microwave-imaging-algorithms-applied-to-11/), snippet only):
  - Clutter rejection was assessed with experimental noise and residual artifacts.
  - DMAS is said to give the best images, with correct detection and localisation in most cases.
- Klemm 2009 (AWPL 8:1349, [Bristol](https://research-information.bris.ac.uk/en/publications/microwave-radar-based-breast-cancer-detection-imaging-in-inhomoge/)): inhomogeneity of the imaged volume is the biggest challenge.

### Inferences
- The finding in this repo that predictions sit about 9 to 11 mm from the centre regardless of true distance matches the published statement that central fibroglandular and residual clutter can exceed the tumour response. It is consistent with the literature but not confirmed by a paper that measures it as a bias.
- Fewer published sources discuss rotation subtraction than reference subtraction. Treat rotation subtraction as UNVERIFIED in the literature.

### Gaps
- No source found that quantifies localization-error versus tumour radial position (centre versus boundary) for UM-BMID. I found only Reimer's remark about central clutter in dense breasts.
- Fear et al. 2002 (near-field imaging, tumour near skin is easier because of the early arrival) is not verified here. Do not cite it without checking the DOI. UNVERIFIED.

## Q3. Published localization error and detection rate (context for about 30 mm mean error, 12 to 17 % detection at 15 mm)

### Takeaway
The best verified figure is Reimer 2020: 11 ± 4 mm mean localization error (DAS and DMAS), but only on the 12 of 18 scans where the tumour was identifiable (SCR > 0). Published numbers are conditional and come from small samples. The repo's 30 mm unconditional error cannot be compared directly. No verified published unconditional UM-BMID Gen-2 DAS localization-error or detection-rate figure was found.

### Cited Findings
- Reimer 2020 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7345629/)):
  - Localization error was 11 ± 4 mm for DAS and DMAS, 12 ± 5 mm for itDAS, and 11 ± 4 mm for itDMAS. There was no significant difference.
  - 12 of 18 reconstructions were identifiable. A hit means the global maximum falls in the tumour region. 12/18 is about 67 %, but this is my arithmetic and uses a different success criterion.
  - Tumours were at antenna height.
- UM-BMID Gen-2 facts from the search summary: 66 phantoms from 9 adipose shell designs. Gen-1/Gen-2 were limited by target location uncertainty and reference scan acquisition, which motivated Gen-3. Source: the MDPI Sensors 21(24) paper snippet and [IEEE DataPort](https://ieee-dataport.org/node/4383). I could not open the full text. UNVERIFIED in detail.
- MDPI Sensors 21(24):8172, [link](https://www.mdpi.com/1424-8220/21/24/8172/xml) (title and authors UNVERIFIED; the page returned 403):
  - Per the snippet, it compared DAS, DMAS and ORR on SCR and localization error and evaluated sensitivity and specificity. ORR improved both over DAS and DMAS.
  - I could not extract the numbers.
- Elahi 2018: detection in 5 patients was qualitative. DMAS ranked first in all 5, DAS was missed or misplaced in one case (IDC) and CF-DAS missed 2 of 5. Source: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6022049/). The sample is tiny.
- Reimer thesis: CNN AUC 76 to 91 %, as above. Source: [mspace](https://mspace.lib.umanitoba.ca/handle/1993/34970).
- Related, from a snippet only: a handheld bistatic system had 100 % sensitivity in 5 patients, and SAFE (XGBoost) had 80 % sensitivity and 81 % specificity. Source: [PMC10255334](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10255334/).

### Inferences
- The repo's 12 to 17 % detection at 15 mm over all 504 scans, with no filter on identifiable scans, is not comparable with published numbers that exclude failures. A fairer comparison would be: of the scans where the maximum falls inside the breast and SCR > 0, what is the error?
- A 30 mm error is close to the "predict the centre" baseline (29 mm) in the repo. No published paper I found reports such a trivial-baseline comparison. That makes this a methodological point for the report.

### Gaps
- Gen-2 DAS-specific localization error: not found.
- Reimer, Krenkevich, Pistorius (2020) EuCAP, "An open-access experimental dataset for breast microwave imaging", doi:10.23919/EuCAP48036.2020.9135659: cited by the repo README, but its content was not retrieved.
- Santorelli et al. papers and any Hazarika UM-BMID paper: not found. Searching "Hazarika UM-BMID" returned no such author. UNVERIFIED.
- Santorelli's clinical-system numbers: not retrieved.

## Q4. Prior reliability score / confidence map resembling a product of peak dominance, coherence and plausibility terms

### Takeaway
I found no prior microwave-imaging paper that defines such a product score. The nearest ingredients exist separately: the coherence factor (a per-pixel coherence weight), SCR/SMR (peak dominance versus clutter), and proposed but untested false-positive rejection rules.

### Cited Findings
- The coherence factor is used as a per-pixel weight on DAS to suppress clutter and sidelobes (Klemm 2008/2009, above). Variants:
  - the phase coherence factor in a 2022 3D system (snippet, [link](https://repository.aaup.edu/bitstream/123456789/1590/1/D0005I%20%281%29.pdf));
  - an iteratively corrected data-adaptive CF-DAS (IEEE Access 2019, [DOAJ](https://doaj.org/article/66b920068032498f8750aea245ddb61d));
  - a scaled coherence factor for DMAS (2024, [NWPU](https://pure.nwpu.edu.cn/en/publications/scaled-coherence-factor-based-delay-multiply-and-sum-beamforming-/)), which reported 26 % SCR improvement over CF-DMAS on an experimental phantom.
  - In these papers CF is an image weight, not a score of whether the whole image can be trusted.
- Elahi 2018: false-positive rejection is proposed as future work (left-right comparison; classification by response magnitude and shape).
- Neyman-Pearson false positive / false negative trade-off in a detection framework: a Surrey record ([link](https://openresearch.surrey.ac.uk/esploro/outputs/journalArticle/Microwave-breast-cancer-detection-via-cost-sensitive/99548417702346)). This is classifier-based, not image-confidence.
- Reimer thesis and Borghouts et al. (Bioengineering 2023, [PMC10603986](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10603986/)):
  - Reimer: a CNN gives a tumour-presence AUC.
  - Borghouts et al.: a CNN outputs a probability map of tumour location. This is the closest published "confidence map", and it is a learned probability, not an analytic score.

### Inferences
- RRS-MBI can be positioned as new within this literature. The risk is that the earlier tests in this repo found it inversely related to error. The report should present it as a negative or inconclusive result rather than a validated score.
- The consistency measures in the repo's logic_refactor run (sub-aperture and sub-band split agreement, AUROC about 0.59 to 0.60) have no direct published counterpart that I found. UNVERIFIED as novel; I did not search specifically for sub-aperture or sub-band stability scores.

### Gaps
- No search was done for conformal prediction, uncertainty quantification in microwave imaging, or Bayesian deep learning confidence on UM-BMID.
- Fear 2002 and the Klemm CF papers' DOIs were not verified.
