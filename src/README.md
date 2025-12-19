# Source Code Documentation

This directory contains the Python source code for the Fourier Qualia Space (FQS) analysis tools developed for my PhD thesis.

## Module Overview

### Standalone Scripts

| Script | Description |
|--------|-------------|
| `amiot_entropy.py` | Computes Amiot's entropy for pitch-class sets using DFT |
| `fourier_coefficient_visualiser.py` | Interactive tool for exploring DFT coefficient contributions |
| `fourier_qualia_space_explorer.py` | Interactive visualisation of all set classes in FQS |
| `set_class_catalogue.py` | Complete catalogue of set classes with interval vector computation |

### Package: `coefficient_order_tests/`

Scripts for determining optimal DFT coefficient orderings in RadViz visualisation.

| Module | Description |
|--------|-------------|
| `dft_radviz_utils.py` | Shared utility functions for DFT and RadViz computations |
| `radviz_distance_correlation_coefficient_order.py` | Finds ordering that maximises Pearson correlation between 6D and 2D distances |
| `radviz_max_dispersion_coefficient_order.py` | Finds ordering that maximises point dispersion from origin |

### Package: `qualia_progression_analysis/`

Tools for analysing harmonic progressions using the Fourier Qualia Space framework.

| Module | Description |
|--------|-------------|
| `main.py` | Entry point for the complete analysis pipeline |
| `corpus.py` | Corpus loading, parsing, and data transformation utilities |
| `fourier_qualia_space.py` | DFT computation and RadViz projection functions |
| `segmentation.py` | Windowed and distance-sensitive segmentation strategies |
| `analysis.py` | Qualia classification, transition matrices, and statistical analysis |

### Package: `ycac_reader/`

Tools for processing the Yale Classical Archives Corpus.

| Module | Description |
|--------|-------------|
| `ycac_corpus_analyser.py` | Main script for YCAC corpus analysis |
| `ycac_analysis_utils.py` | Utility functions for YCAC data processing |

## Dependencies

See `requirements.txt` in the repository root. Key dependencies:

- **numpy**: Array operations and FFT
- **pandas**: Data manipulation
- **scipy**: Signal processing, clustering, spatial computations
- **matplotlib**: Visualisation
- **music21**: Music theory computations
- **tqdm**: Progress bars

## Licence

All code is licensed under the MIT licence. See `LICENSE` in the repository root.
