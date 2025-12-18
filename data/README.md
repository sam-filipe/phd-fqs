# Data Documentation

This directory contains the datasets used in the thesis analysis. All data is provided in CSV format.

## Directory Structure

```
data/
├── chapter_6/                  # Diachronic corpus analysis data
├── chapter_7/                  # Debussy and Ravel case studies
│   ├── debussy/                # Debussy analysis results
│   └── ravel/                  # Ravel analysis results
└── ycac_additions/             # Additional composer data for YCAC
```

## Chapter 6 Data

Data from the diachronic analysis of the Yale Classical Archives Corpus.

| Filename | Description |
|----------|-------------|
| `main_data.csv` | Primary analysis results including statistical measures for the corpus |
| `diachronic_zipfian_analysis.csv` | Zipf distribution analysis of qualia transitions across historical periods |
| `windowed_ambiguity_data.csv` | Temporal (windowed) analysis of harmonic ambiguity throughout pieces |
| `ambiguity_std_per_composer.csv` | Standard deviation of ambiguity measures grouped by composer |
| `mean_cardinality_per_composer.csv` | Average pitch-class set cardinality by composer |

## Chapter 7 Data

Detailed analysis of Ravel and Debussy's qualia sequences.

### Debussy Files

| Filename | Description |
|----------|-------------|
| `conditional_matrix.csv` | Conditional transition probabilities between qualia |
| `global_matrix.csv` | Raw transition counts between qualia pairs |
| `reflets_segments.csv` | Segmentation data for *Reflets dans l'eau* |
| `regression_data.csv` | Data for Zipf regression analysis |
| `transition_counts.csv` | Frequency counts of each transition type |

### Ravel Files

| Filename | Description |
|----------|-------------|
| `conditional_matrix.csv` | Conditional transition probabilities between qualia |
| `global_matrix.csv` | Raw transition counts between qualia pairs |
| `regression_data.csv` | Data for Zipf regression analysis |
| `transition_counts.csv` | Frequency counts of each transition type |

### Qualia Labels

The qualia classification system uses the following labels:

| Label | Name |
|-------|------|
| `C` | Chromaticity |
| `D` | Dyadicity |
| `T` | Triadicity |
| `O` | Octatonicity |
| `DT` | Diatonicity |
| `WT` | Whole-tone |
| `A` | Qualia Ambiguity |

## YCAC Additions

Additional vertical slice data extracted for composers not originally in the YCAC.

| Filename | Description |
|----------|-------------|
| `berg_slices.csv` | Vertical slices from Alban Berg's works |
| `messiaen_slices.csv` | Vertical slices from Olivier Messiaen's works |
| `schoenberg_slices.csv` | Vertical slices from Arnold Schoenberg's works |
| `webern_slices.csv` | Vertical slices from Anton Webern's works |

## Data Provenance

- **YCAC data**: Derived from the Yale Classical Archives Corpus. See the [YCAC project](https://ycac.yale.edu/) for original corpus documentation.
- **Analysis data**: Generated using the scripts in the `src/` directory of this repository.

## Licence

All data in this directory is licensed under the Creative Commons Attribution 4.0 International (CC-BY-4.0) licence. See `LICENSE-DATA` in the repository root.
