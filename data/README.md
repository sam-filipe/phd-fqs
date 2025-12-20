# Data Documentation

This directory contains the datasets used in the thesis analysis. All data is provided in CSV format.

## Directory Structure

```
data/
├── chapter_6/                  # Diachronic corpus analysis data
│   ├── transition_counts/      # Frequency tables for multiple composers
├── chapter_7/                  # Debussy and Ravel case studies
│   ├── debussy/                # Debussy analysis results
│   ├── ravel/                  # Ravel analysis results
│   └── dendrograms/            # Dendrograms for several composers
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

### Transition Counts Files

| Filename | Description |
|----------|-------------|
| `bach_transition_counts.csv` | Bach's frequency tables |
| `bartok_transition_counts.csv` | Bartók's frequency tables |
| `beethoven_transition_counts.csv` | Beethoven's frequency tables |
| `berlioz_transition_counts.csv` | Berlioz's frequency tables |
| `bizet_transition_counts.csv` | Bizet's frequency tables |
| `brahms_transition_counts.csv` | Brahms's frequency tables |
| `bruckner_transition_counts.csv` | Bruckner's frequency tables |
| `byrd_transition_counts.csv` | Byrd's frequency tables |
| `chopin_transition_counts.csv` | Chopin's frequency tables |
| `debussy_transition_counts.csv` | Debussy's frequency tables |
| `dvorak_transition_counts.csv` | Dvořák's frequency tables |
| `handel_transition_counts.csv` | Handel's frequency tables |
| `haydn_transition_counts.csv` | Haydn's frequency tables |
| `liszt_transition_counts.csv` | Liszt's frequency tables |
| `mahler_transition_counts.csv` | Mahler's frequency tables |
| `mendelssohn_transition_counts.csv` | Mendelssohn's frequency tables |
| `mozart_transition_counts.csv` | Mozart's frequency tables |
| `purcell_transition_counts.csv` | Purcell's frequency tables |
| `ravel_transition_counts.csv` | Ravel's frequency tables |
| `satie_transition_counts.csv` | Satie's frequency tables |
| `scarlatti_transition_counts.csv` | Scarlatti's frequency tables |
| `schubert_transition_counts.csv` | Schubert's frequency tables |
| `schumann_transition_counts.csv` | Schumann's frequency tables |
| `scriabin_transition_counts.csv` | Scriabin's frequency tables |
| `tchaikovsky_transition_counts.csv` | Tchaikovsky's frequency tables |
| `verdi_transition_counts.csv` | Verdi's frequency tables |
| `vivaldi_transition_counts.csv` | Vivaldi's frequency tables |
| `wagner_transition_counts.csv` | Wagner's frequency tables |


## Chapter 7 Data

Detailed analysis of Ravel and Debussy's qualia sequences.

### Debussy Files

| Filename | Description |
|----------|-------------|
| `d_complete_dataframe.csv` | All data produced by the algorithm |
| `d_conditional_matrix.csv` | Conditional transition probabilities between qualia |
| `d_global_matrix.csv` | Raw transition counts between qualia pairs |
| `d_reflets_segments.csv` | Segmentation data for *Reflets dans l'eau* |
| `d_regression_data.csv` | Data for Zipf regression analysis |
| `d_transition_counts.csv` | Frequency counts of each transition type |

### Ravel Files

| Filename | Description |
|----------|-------------|
| `r_conditional_matrix.csv` | Conditional transition probabilities between qualia |
| `r_global_matrix.csv` | Raw transition counts between qualia pairs |
| `r_regression_data.csv` | Data for Zipf regression analysis |
| `r_transition_counts.csv` | Frequency counts of each transition type |

### Dendrograms Files

| Filename | Description |
|----------|-------------|
| `bach_dendrogram.png` | Bach's dendrogram |
| `bartok_dendrogram.png` | Bartók's dendrogram |
| `beethoven_dendrogram.png` | Beethoven's dendrogram |
| `berlioz_dendrogram.png` | Berlioz's dendrogram |
| `bizet_dendrogram.png` | Bizet's dendrogram |
| `brahms_dendrogram.png` | Brahms's dendrogram |
| `bruckner_dendrogram.png` | Bruckner's dendrogram |
| `byrd_dendrogram.png` | Byrd's dendrogram |
| `chopin_dendrogram.png` | Chopin's dendrogram |
| `debussy_dendrogram.png` | Debussy's dendrogram |
| `dvorak_dendrogram.png` | Dvořák's dendrogram |
| `handel_dendrogram.png` | Handel's dendrogram |
| `haydn_dendrogram.png` | Haydn's dendrogram |
| `liszt_dendrogram.png` | Liszt's dendrogram |
| `mahler_dendrogram.png` | Mahler's dendrogram |
| `mendelssohn_dendrogram.png` | Mendelssohn's dendrogram |
| `mozart_dendrogram.png` | Mozart's dendrogram |
| `purcell_dendrogram.png` | Purcell's dendrogram |
| `ravel_dendrogram.png` | Ravel's dendrogram |
| `satie_dendrogram.png` | Satie's dendrogram |
| `scarlatti_dendrogram.png` | Scarlatti's dendrogram |
| `schubert_dendrogram.png` | Schubert's dendrogram |
| `schumann_dendrogram.png` | Schumann's dendrogram |
| `scriabin_dendrogram.png` | Scriabin's dendrogram |
| `tchaikovsky_dendrogram.png` | Tchaikovsky's dendrogram |
| `verdi_dendrogram.png` | Verdi's dendrogram |
| `vivaldi_dendrogram.png` | Vivaldi's dendrogram |
| `wagner_dendrogram.png` | Wagner's dendrogram |

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
