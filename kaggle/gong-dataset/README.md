# GONG multi-channel data for solar filament work

Data downloaded from the NSO GONG archives while working on the Kaggle solar filament
segmentation challenge (MAGFiLO 1.0). **This dataset holds no MAGFiLO images, annotations or
pseudo-labels**, only public GONG data and quantities computed from it.

## Contents

| Path                 | Files       | What it is                                                                                                                                                                                                                                                                                                                                                                                  |
|----------------------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `halpha_2023_2025/`  | 1,735 JPEG  | Unlabelled 2048×2048 full-disk GONG H-alpha frames, 2023–2025, converted to JPEG like MAGFiLO (see Provenance), site letters A B C L T U Z (no M frames in this period). Fetched as bursts of 5 frames 2 min apart (offsets −4…+4 min) on randomly sampled days, after a disk-brightness and dark-patch check. Name: YYYYMMDDHHMMSS + site letter + `h.jpeg`, e.g. `20240118183542Ch.jpeg`. |
| `halpha_bursts.json` | 1           | The burst groups and the reasons frames were rejected.                                                                                                                                                                                                                                                                                                                                      |
| `bzi_fits/`          | 640 FITS.gz | GONG line-of-sight magnetograms (`bzi`, 860×860, gauss) each taken within 60 min of a MAGFiLO 1.0 training observation from the same site. **Renamed to the matching H-alpha frame** (YYYYMMDDHHMMSS + site letter + `h.fits.gz`); the true observation time is in the `DATE-OBS` header.                                                                                                   |
| `bzi_matches.json`   | 1           | Match status for all 707 MAGFiLO training observations: time gap in minutes, plage correlation with the H-alpha frame, and `ok` / `too far` / `no frames`.                                                                                                                                                                                                                                  |
| `bzi_targets/`       | 628 NPY     | `(3, 512, 512)` uint8 (value / 255), on the H-alpha frame grid (centre 1024, radius 900, north up): 0 = proximity to a polarity inversion line `exp(-d / 10 Mm)` after 12 Mm smoothing, 1 = strong flux `tanh(                                                                                                                                                                              |B| / 100 G)`, 2 = valid-disk mask. Only the 628 matches that passed QC. |

Site letters: B Big Bear, C Cerro Tololo, L Learmonth, M Mauna Loa, T El Teide, U Udaipur. A and Z occur only in `halpha_2023_2025/` and are not in the MAGFiLO site mapping; check the FITS headers at the source before relying on their site.

## Registration notes

The H-alpha JPEGs are centred at (1024, 1024) with disk radius 900 px, north up. The bzi images
use the `FNDLMB*` header keys for centre and radius and are rotated 90° relative to the
H-alpha frames, with no mirror (median plage correlation 0.67 after alignment).

## Sources

- H-alpha: https://gong2.nso.edu/HA/haf/
- Magnetograms: https://nispdata.nso.edu/ftp/vmb/archive/bzi/

## Provenance

- H-alpha frames: `.fits.fz` downloaded 2026 from https://gong2.nso.edu/HA/haf/ and converted the way the MAGFiLO JPEGs were made: the MLEcoFi FITS-to-JPEG convertor (https://bitbucket.org/dataresearchlab/mleco-fits2jpegconvertor, `wavelet` then `fits2ppmColorscale grey 640 5335`), saved as 8-bit greyscale JPEG at quality 92.
- Magnetograms: downloaded 2026 from https://nispdata.nso.edu/ftp/vmb/archive/bzi/, FITS as served, renamed only.
- `bzi_targets`: computed from the magnetograms (resampled onto the H-alpha grid, 90° turn, PIL proximity, strong flux, valid mask).

## Coverage

- Time: H-alpha 2023-01 to 2025-12; magnetograms 2011-01 to 2022-08 (dates of the MAGFiLO training observations).
- Space: full solar disk as seen from Earth; GONG network sites worldwide.

## Citation

- **Data source:** NSO Integrated Synoptic Program (NISP), "Global Oscillation Network Group (GONG)". https://gong.nso.edu/ . Data were acquired by GONG instruments operated by NISP/NSO/AURA/NSF with contribution from NOAA.
- **GONG instrument paper:** Harvey, J. W. et al. (1996). "The Global Oscillation Network Group (GONG) Project". *Science* 272(5266), 1284–1286. https://doi.org/10.1126/science.272.5266.1284
- **H-alpha conversion:** MLEcoFi Team, "mleco-fits2jpegconvertor". https://bitbucket.org/dataresearchlab/mleco-fits2jpegconvertor
- **This dataset:** Hrabar, V. (2026). "GONG Multi-Channel Data for Solar Filaments" [Data set]. Kaggle. https://www.kaggle.com/datasets/vhrabar/gong-solar-filaments

## Acknowledgement and license

Data were acquired by GONG instruments operated by NISP/NSO/AURA/NSF with contribution from NOAA.

NSO publishes no formal license for GONG data. It asks users to acknowledge GONG; see the full
acknowledgement and the NSF sponsorship disclaimer at https://nso.edu/data/ ("Data Use Policy and
Acknowledgements"). Please include them when you publish work that uses this dataset.
