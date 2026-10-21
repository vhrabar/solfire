# GONG multi-channel data for solar filament work

Data downloaded from the NSO GONG archives while working on the Kaggle solar filament
segmentation challenge (MAGFiLO 1.0). **This dataset holds no MAGFiLO images, annotations or
pseudo-labels**, only public GONG data and quantities computed from it.

## Contents

Every FITS file is as served by NSO, only renamed after the H-alpha frame it belongs to
(YYYYMMDDHHMMSS + site letter + `h`). The original archive name of every renamed file is in the
matching JSON file, and the true observation time is in the `DATE-OBS` / `TIME-OBS` headers.

### Unlabelled H-alpha, 2023–2025

| Path                         | Files         | What it is                                                                                                                                                                                                                                                                  |
|------------------------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `halpha_2023_2025/`          | 1,300 JPEG    | Full-disk 2048×2048 GONG H-alpha frames converted to JPEG like MAGFiLO (see Provenance). 260 bursts of 5 frames 2 min apart (offsets −4…+4 min), one burst per randomly sampled day, every frame past a disk-brightness and dark-patch check. E.g. `20240118183542Ch.jpeg`. |
| `halpha_fits_2023_2025/`     | 1,300 FITS.fz | The raw `haf` FITS behind each JPEG (`.fits.fz`, image in HDU 1).                                                                                                                                                                                                           |
| `halpha_bursts.json`         | 1             | The 260 burst groups, and how many frames failed the checks (bursts with a failed frame are left out).                                                                                                                                                                      |
| `halpha_bzi/`                | 211 FITS.gz   | Line-of-sight magnetogram (`bzi`, 860×860, gauss) from the same site closest in time to each burst's centre frame, within 60 min (median gap 0.2 min).                                                                                                                      |
| `halpha_izi/`, `halpha_vzi/` | 211 each      | Continuum intensity (`izi`) and Doppler velocity (`vzi`) taken in the same minute as each `halpha_bzi` file.                                                                                                                                                                |
| `halpha_bzi_matches.json`    | 1             | Match status for all 260 burst centres: archive name, time gap, plage correlation with the H-alpha frame, `ok` / `too far` / `no frames`.                                                                                                                                   |
| `halpha_channels.json`       | 1             | Archive names of the `izi` / `vzi` files.                                                                                                                                                                                                                                   |

### Matched to the MAGFiLO 1.0 training observations, 2011–2022

| Path                     | Files       | What it is                                                                                                                                                                                                                                                                                                                           |
|--------------------------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `bzi_fits/`              | 640 FITS.gz | Same-site `bzi` magnetogram within 60 min of a MAGFiLO 1.0 training observation, named after that observation.                                                                                                                                                                                                                       |
| `izi_fits/`, `vzi_fits/` | 640 each    | Continuum intensity and Doppler velocity taken in the same minute as each `bzi_fits` file.                                                                                                                                                                                                                                           |
| `bzi_matches.json`       | 1           | Match status for all 707 MAGFiLO training observations: time gap, plage correlation, `ok` / `too far` / `no frames`.                                                                                                                                                                                                                 |
| `bzi_channels.json`      | 1           | Archive names of the `bzi` / `izi` / `vzi` files.                                                                                                                                                                                                                                                                                    |
| `bzi_targets/`           | 628 NPY     | `(3, 512, 512)` uint8 (value / 255) on the H-alpha grid (centre 1024, radius 900, north up): 0 = proximity to a polarity inversion line `exp(-d / 10 Mm)` after 12 Mm smoothing, 1 = strong flux `tanh(\|B\| / 100 G)`, 2 = valid-disk mask. Only the 628 matches that passed QC; QC and target settings in `bzi_targets/meta.json`. |

### Synoptic magnetograms

| Path                    | Files       | What it is                                                                                                                                                            |
|-------------------------|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `synoptic_zqs/`         | 957 FITS.gz | Hourly zero-point-corrected synoptic magnetograms (`mrzqs`, 360×180 Carrington longitude × sine latitude, gauss), original archive names.                             |
| `synoptic_matches.json` | 1           | The map closest in time to each of the 707 MAGFiLO training observations and 260 burst centres, within 6 h (median gap 20 min); some maps serve several observations. |

Site letters: B Big Bear, C Cerro Tololo, L Learmonth, M Mauna Loa, T El Teide, U Udaipur.
A and Z occur only in the 2023–2025 frames; their FITS headers give site codes `TC` and `TE`,
which have no `bzi` / `izi` / `vzi` archive, so their bursts have no magnetogram (`no frames`).

## Registration notes

The H-alpha JPEGs are centred at (1024, 1024) with disk radius 900 px, north up. The site
images (`bzi`, `izi`, `vzi`) use the `FNDLMB*` header keys for centre and radius and are rotated
90° relative to the H-alpha frames, with no mirror. Median plage correlation after alignment:
0.67 for the MAGFiLO matches, 0.44 for the 2023–2025 bursts (lowest 0.30).

## Sources

- H-alpha: https://gong2.nso.edu/HA/haf/
- Site magnetograms, intensity and velocity: https://nispdata.nso.edu/ftp/vmb/archive/ (`bzi`, `izi`, `vzi`)
- Synoptic magnetograms: https://gong2.nso.edu/oQR/zqs/
- H-alpha conversion: https://bitbucket.org/dataresearchlab/mleco-fits2jpegconvertor
- Matching times: MAGFiLO 1.0 training observation times (Kaggle solar filament segmentation challenge 2026); no MAGFiLO images or annotations are included.

## Provenance

- H-alpha frames: `.fits.fz` downloaded 2026 from https://gong2.nso.edu/HA/haf/ and converted the way the MAGFiLO JPEGs were made: the MLEcoFi FITS-to-JPEG convertor (`wavelet` then `fits2ppmColorscale grey 640 5335`), saved as 8-bit greyscale JPEG at quality 92.
- All other FITS files: downloaded 2026 from the archives above, as served, renamed only.
- `bzi_targets`: computed from the magnetograms (resampled onto the H-alpha grid, 90° turn, PIL proximity, strong flux, valid mask).

## Coverage

- Time: H-alpha and their site images 2023-01 to 2025-12; MAGFiLO-matched site images 2011-01 to 2022-08; synoptic maps for both.
- Space: full solar disk as seen from Earth (synoptic maps: whole Sun in Carrington coordinates); GONG network sites worldwide.

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
