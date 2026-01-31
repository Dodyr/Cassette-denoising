# Model Results Investigation Summary
## Denoising Model v15 Performance Analysis

**Generated from:** `investigate_model_results.ipynb`

---

## Overview

This analysis compares the original noisy audio recording (ELO_1_raw.wav) with the denoised output from Model v15 (1_test_v15_with_overlap.wav).

### Files Analyzed
- **Original (Noisy):** `ELO_1_raw.wav`
  - Duration: 298.18 seconds (~5 minutes)
  - Samples: 13,149,729
  - Sample rate: 44,100 Hz

- **Denoised (Model v15):** `1_test_v15_with_overlap.wav`
  - Duration: 297.22 seconds (~5 minutes)
  - Samples: 13,107,200
  - Sample rate: 44,100 Hz

---

## Key Findings

### ⚠️ CRITICAL ISSUE: Signal Inversion

The analysis reveals a **major problem** with the denoised output:

**The denoised audio is significantly LOUDER than the original, not quieter**, indicating the model is not correctly denoising the audio. This is the opposite of what we expect.

### Amplitude Analysis

| Metric | Original (Noisy) | Denoised (v15) | Change |
|--------|------------------|----------------|--------|
| **RMS** | 0.0944 | 0.9196 | +874% ↑ |
| **Peak Amplitude** | 0.4948 | 1.0000 (maxed out) | +102% ↑ |
| **Std Dev** | 0.0944 | 0.9196 | +874% ↑ |
| **dBFS (RMS)** | -20.50 dB | -0.73 dB | +19.77 dB ↑ |
| **dBFS (Peak)** | -6.11 dB | 0.00 dB (clipping) | +6.11 dB ↑ |

**Interpretation:** Instead of reducing amplitude (noise reduction), the model is **AMPLIFYING** the signal by ~10x. The denoised output is hitting digital clipping (1.0 amplitude limit), which will introduce severe distortion.

---

## Detailed Analysis

### 1. Spectral Analysis (FFT)

- **Original spectral energy:** 3.28 × 10²
- **Denoised spectral energy:** 3.24 × 10²
- **Energy reduction:** Only 1.20%

The frequency content is largely preserved, with minimal energy reduction. The model is not selectively removing high-frequency noise as expected.

**Observation:** The linear and log-scale FFT plots show that both the original and denoised signals have similar frequency distributions, dominated by low-frequency content. The denoised version shows slightly less high-frequency activity, but the overall effect is minimal.

---

### 2. Spectrogram Analysis

**Mel-Spectrogram Statistics (first 30 seconds):**

| Metric | Original | Denoised | Change |
|--------|----------|----------|--------|
| Mean | -47.48 dB | -51.83 dB | -4.35 dB ↓ |
| Std Dev | 14.58 dB | 23.83 dB | +9.25 dB ↑ |
| Range | [-80.00, 0.00] dB | [-80.00, 0.00] dB | Same |

**Observation:** The denoised spectrogram shows a **darker/quieter appearance** in the low-frequency regions (visible as black areas in the spectral visualization), which is unexpected. The increased standard deviation suggests higher variance across frequency bins.

The visual comparison shows:
- **Original:** Relatively uniform frequency distribution with purple/pink coloring throughout
- **Denoised:** More pronounced black (quiet) regions at low frequencies, with higher intensity peaks in the middle frequencies

---

### 3. Noise Floor Analysis

| Percentile | Original (noisy) | Denoised (v15) | Ratio |
|------------|------------------|----------------|-------|
| 5th | 0.000854 | 0.0508 | +59.4x |
| 10th | 0.003143 | 0.3191 | +101.5x |
| 25th | 0.013748 | 1.0000 | +7,173% |
| 50th | 0.044647 | 1.0000 | +2,242% |
| 75th | 0.101959 | 1.0000 | +880% |
| 90th | 0.164368 | 1.0000 | +509% |
| 95th | 0.203247 | 1.0000 | +392% |

**Estimated SNR:**
- Original: **31.12 dB**
- Denoised: **0.00 dB** (no signal-to-noise ratio possible when noise floor ≈ peak)
- Change: **-31.12 dB** ⚠️ **SEVERE DEGRADATION**

**Critical Finding:** The noise floor has dramatically increased from 0.0137 to 0.9999 (nearly the peak value). This means the model is essentially replacing the original quiet parts with high-amplitude content.

---

### 4. Waveform Visualization

The waveform comparison clearly shows:

**Original (Top Panel):**
- Relatively quiet signal with RMS of 0.094
- Peak amplitude of 0.495
- Lots of silence/quiet sections visible
- Shows the characteristics of quiet/soft audio

**Denoised (Bottom Panel):**
- Much louder, more energetic signal with RMS of 0.920
- Peak amplitude clipped at 1.0 (digital maximum)
- Fills the entire amplitude range
- Appears to be continuously at high volume with minimal dynamic range

---

## Summary

### Model Performance: ❌ FAILED

The Model v15 denoising attempt has **fundamentally failed** in the following ways:

1. **Signal Amplification Instead of Denoising:** The model amplified the signal by ~10x instead of reducing noise
2. **Digital Clipping:** The output is hitting the maximum digital level (1.0), causing clipping distortion
3. **SNR Degradation:** Signal-to-noise ratio decreased by 31.12 dB instead of improving
4. **Dynamic Range Collapse:** The output has essentially no dynamic range - most values are at or near maximum
5. **Noise Floor Elevation:** Instead of reducing the noise floor, it increased from 0.0137 to 0.9999

### Possible Causes

- **Training Issue:** The model may have been trained with reversed labels (treating noise as signal and signal as noise)
- **Incorrect Normalization:** The model might be normalizing to peak amplitude instead of RMS
- **Inverted Loss Function:** The training objective might be optimizing for the opposite of denoising
- **Preprocessing Mismatch:** Input normalization during training may differ from inference

### Recommendations

1. **Verify training data:** Check if the noisy/clean labels in the training data are correct
2. **Inspect model architecture:** Review the model configuration and loss function
3. **Test on known data:** Run the model on the original training data to verify it works
4. **Check inference pipeline:** Verify the audio loading, normalization, and output processing steps
5. **Review model weights:** Ensure the correct model weights are being loaded (v15 vs others)
6. **Consider earlier versions:** Test Model v14 and earlier to see when the problem started

---

## Files Referenced

- Input (noisy): `cassette_denoise/data/ELO_1_raw.wav`
- Output (denoised): `cassette_denoise/data/1_test_v15_with_overlap.wav`
- Analysis notebook: `cassette_denoise/investigate_model_results.ipynb`
