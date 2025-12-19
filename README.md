# Fourier Qualia Space Analysis: PhD Thesis Repository

[![DOI](https://zenodo.org/badge/DOI/PLACEHOLDER.svg)](https://doi.org/PLACEHOLDER)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This repository contains the code and datasets used in my PhD thesis on pitch-class set analysis using the Discrete Fourier Transform (DFT) and the Fourier Qualia Space (FQS).

## Overview

The research presented here develops analytical tools for studying harmonic content in music using mathematical approaches based on the Discrete Fourier Transform. The core methodology projects pitch-class sets onto a two-dimensional space (the Fourier Qualia Space) using RadViz visualisation, enabling the study of qualia progressions and their statistical properties.

## Repository Structure

```
phd-thesis-repository/
│
├── README.md                 # This file
├── LICENSE                   # MIT licence for code
├── LICENSE-DATA              # CC-BY-4.0 licence for data
├── CITATION.cff              # Citation information for Zenodo
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore patterns
│
├── data/                     # Datasets
│   ├── README.md             # Data documentation
│   ├── chapter_6/            # Chapter 6 analysis data
│   ├── chapter_7/            # Chapter 7 analysis data
│   │   ├── debussy/          # Debussy-specific analyses
│   │   └── ravel/            # Ravel-specific analyses
│   └── ycac_additions/       # Additional YCAC composer slices
│
└── src/                      # Source code
    ├── README.md             # Code documentation
    │
    ├── coefficient_order_tests/       # DFT coefficient ordering optimisation
    │   ├── dft_radviz_utils.py
    │   ├── radviz_distance_correlation_coefficient_order.py
    │   └── radviz_max_dispersion_coefficient_order.py
    │
    ├── qualia_progression_analysis/   # Harmonic progression analysis
    │   ├── analysis.py
    │   ├── corpus.py
    │   ├── fourier_qualia_space.py
    │   ├── main.py
    │   └── segmentation.py
    │
    ├── ycac_reader/                   # YCAC corpus utilities
    │   ├── ycac_analysis_utils.py
    │   └── ycac_corpus_analyser.py
    │
    ├── amiot_entropy.py               # Spectral entropy computation
    ├── fourier_coefficient_visualiser.py   # Interactive DFT visualiser
    ├── fourier_qualia_space_explorer.py    # FQS interactive explorer
    └── set_class_catalogue.py         # Set class definitions and utilities
```

## Installation

### Prerequisites

- Python 3.11 or higher (64-bit)
- pip (Python package manager)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/[username]/phd-thesis-repository.git
   cd phd-thesis-repository
   ```

2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Interactive Visualisers

**Fourier Coefficient Visualiser**: Explore how DFT coefficients capture pitch-class collections
```bash
python src/fourier_coefficient_visualiser.py
```

**Fourier Qualia Space Explorer**: Interactively explore set classes in FQS
```bash
python src/fourier_qualia_space_explorer.py
```

### YCAC Corpus Analysis

To analyse the Yale Classical Archives Corpus:

1. Download the YCAC corpus and place it in a known directory
2. Update the `CORPUS_DIRECTORY` path in `src/ycac_reader/ycac_corpus_analyser.py`
3. Run the analyser:
   ```bash
   python src/ycac_reader/ycac_corpus_analyser.py
   ```

### Qualia Progression Analysis

For harmonic progression analysis:

1. Update the `CORPUS_DIRECTORY` path in `src/qualia_progression_analysis/main.py`
2. Configure segmentation parameters as needed
3. Run the analysis:
   ```bash
   python src/qualia_progression_analysis/main.py
   ```

### Coefficient Order Optimisation

Tests to find optimal DFT coefficient orderings for the FQS:

```bash
# Maximum dispersion analysis
python src/coefficient_order_tests/radviz_max_dispersion_coefficient_order.py

# Distance correlation analysis
python src/coefficient_order_tests/radviz_distance_correlation_coefficient_order.py
```

## Data

The `data/` directory contains pre-computed analysis results in CSV format. See `data/README.md` for detailed descriptions of each dataset.

**Note**: The Yale Classical Archives Corpus (YCAC) itself is not included in this repository. It must be obtained separately from its original source.

## Licence

- **Code**: MIT Licence (see `LICENSE`)
- **Data**: Creative Commons Attribution 4.0 International (CC-BY-4.0) (see `LICENSE-DATA`)

## Citation

If you use this code or data in your research, please cite:

```bibtex
@phdthesis{author_year,
  author = {[Author Name]},
  title  = {[Thesis Title]},
  school = {[University]},
  year   = {[Year]},
  doi    = {[DOI from Zenodo]}
}
```

See `CITATION.cff` for machine-readable citation information.

## Contact

For questions or issues, please contact up201709719@up.pt.
