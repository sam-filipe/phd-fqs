#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YCAC Corpus Analyser
====================
Main script for analysing the Yale-Classical Archives Corpus (YCAC).

This script processes musical data from the YCAC corpus and computes various
analytical measures including:
- Discrete Fourier Transform (DFT) coefficients for pitch-class sets
- RadViz visualisation coordinates based on DFT magnitudes
- Qualia ambiguity metric
- Windowed temporal analysis

The script supports three analysis modes:
    Mode 1: Analyse works by specific composer(s)
    Mode 2: Analyse a single file
    Mode 3: Analyse the entire corpus

Output:
    - CSV and Excel files containing standard analysis data
    - Windowed ambiguity analysis data

Requirements:
    - Python 3.11.3+ (64-bit)
    - The YCAC corpus files and metadata (0_Metadata.csv) must be in the
      same directory specified by CORPUS_DIRECTORY

Author: Samuel Pereira
Date: 22/12/2025
Refactored for Thesis Repository
"""

import ycac_analysis_utils as utils


# =============================================================================
# CONFIGURATION
# =============================================================================

# System path to the YCAC corpus directory.
# IMPORTANT: Update this path to match your local system configuration.
# The metadata file (0_Metadata.csv) must be in this directory.
CORPUS_DIRECTORY = '/Users/sp/Documents/Investigação/#Code/YCAC corpus/'

# Order of DFT coefficients for RadViz visualisation.
# This arrangement [1, 5, 3, 4, 6, 2] positions harmonically related
# coefficients optimally around the RadViz circle for meaningful visual
# interpretation of pitch-class set distributions.
RADVIZ_COEFFICIENT_ORDER = [1, 5, 3, 4, 6, 2]

# Window size for temporal (windowed) analysis.
# Must be an even number >= 8. Default is 16 time slices.
WINDOW_SIZE = 16

# Maximum column width for DataFrame display.
DISPLAY_COLUMN_WIDTH = 30

# Base filename for exported data files.
OUTPUT_FILENAME = 'corpusData'


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function for the YCAC corpus analysis pipeline.
    
    The function orchestrates the complete analysis workflow:
    1. Mode selection and data loading
    2. Year estimation for undated compositions
    3. DFT coefficient computation
    4. RadViz coordinate calculation
    5. Piece-level statistics computation
    6. Qualia ambiguity calculation
    7. Windowed temporal analysis
    8. Data export
    """
    
    # -------------------------------------------------------------------------
    # STEP 1: MODE SELECTION AND DATA LOADING
    # -------------------------------------------------------------------------
    
    # Prompt user to select analysis mode (1: composers, 2: file, 3: corpus)
    mode = utils.mode_definition()
    
    # Set display options for DataFrame output
    utils.col_width(DISPLAY_COLUMN_WIDTH)
    
    # Load and process data based on selected mode
    if mode == 1:
        # MODE 1: ANALYSE SPECIFIC COMPOSER(S)
        # ------------------------------------
        # Prompts for composer names and filters the corpus accordingly.
        
        composers = utils.composer_definition()
        
        # Load metadata from the corpus
        metadata_filepath = CORPUS_DIRECTORY + '0_Metadata.csv'
        metadata = utils.read_metadata(metadata_filepath)
        
        # Validate that requested composers exist in the database
        composers = utils.check_composers(metadata, composers)
        
        # Determine which corpus files contain works by the selected composers
        files = utils.files_definition(CORPUS_DIRECTORY, composers)
        
        # Build the analysis database for selected composers
        db = utils.dataLoop_m1(CORPUS_DIRECTORY, composers, metadata, files)
        
    elif mode == 2:
        # MODE 2: ANALYSE A SINGLE FILE
        # ------------------------------
        # Prompts for a specific file and analyses only that file.
        
        file_name = utils.file_definition(CORPUS_DIRECTORY)
        
        # Build the analysis database for the selected file
        db = utils.dataLoop_m2(CORPUS_DIRECTORY, file_name)
        
    elif mode == 3:
        # MODE 3: ANALYSE ENTIRE CORPUS
        # ------------------------------
        # Processes all files in the YCAC corpus directory.
        
        metadata_filepath = CORPUS_DIRECTORY + '0_Metadata.csv'
        metadata = utils.read_metadata(metadata_filepath)
        
        # Build the analysis database for the entire corpus
        db = utils.dataLoop_m3(CORPUS_DIRECTORY, metadata)
    
    # -------------------------------------------------------------------------
    # STEP 2: YEAR ESTIMATION FOR UNDATED COMPOSITIONS
    # -------------------------------------------------------------------------
    # For pieces without a specific composition date, estimate the year using
    # the upper bound of the composition range (e.g., '1910-1920' → 1920).
    
    db = utils.year_definition(db)
    
    # -------------------------------------------------------------------------
    # STEP 3: DISCRETE FOURIER TRANSFORM (DFT) COMPUTATION
    # -------------------------------------------------------------------------
    # Compute the DFT coefficients for each pitch-class set. These coefficients
    # capture the interval-class content and are used for subsequent RadViz
    # visualisation and statistical analysis.
    
    db = utils.dft_data(db)
    
    # -------------------------------------------------------------------------
    # STEP 4: RADVIZ COORDINATE CALCULATION
    # -------------------------------------------------------------------------
    # Calculate RadViz coordinates using the six non-trivial DFT coefficient
    # magnitudes (|f1| through |f6|). The coefficient order determines their
    # placement around the RadViz circle.
    
    anchors = utils.radviz_anchors()
    db = utils.radviz_data(db, anchors, RADVIZ_COEFFICIENT_ORDER)
    
    # -------------------------------------------------------------------------
    # STEP 5: PIECE-LEVEL STATISTICS
    # -------------------------------------------------------------------------
    # Compute aggregate statistics for each piece:
    # - Mean cardinality of pitch-class sets
    # - Top 3 most frequent set classes
    # - Mean DFT coefficient magnitudes
    # - Tonal Index (TI = phase_f2 + phase_f3 - phase_f5)
    
    db = utils.piece_stats(db)
    
    # -------------------------------------------------------------------------
    # STEP 6: QUALIA AMBIGUITY
    # -------------------------------------------------------------------------
    # Ambiguity: Measures how close pitch-class sets cluster to the RadViz
    #            centre. Higher values indicate more ambiguous/chromatic harmony.
    
    db = utils.ambiguity(db)
    
    # -------------------------------------------------------------------------
    # STEP 7: WINDOWED TEMPORAL ANALYSIS
    # -------------------------------------------------------------------------
    # Perform sliding-window analysis to track how ambiguity and diversity
    # evolve throughout each piece. The window size determines temporal
    # granularity; overlap is half the window size.
    
    db_ambiguity = utils.windowed_analysis(db, WINDOW_SIZE)
    
    # Remove the RadViz coordinates column (no longer needed for export)
    db = db.drop(['RadViz>1'], axis=1)
    
    # -------------------------------------------------------------------------
    # STEP 8: DATA EXPORT
    # -------------------------------------------------------------------------
    # Export all analysis results to CSV and Excel formats.
    
    utils.export_data(db, db_ambiguity, OUTPUT_FILENAME)
    
    # -------------------------------------------------------------------------
    # FINAL OUTPUT
    # -------------------------------------------------------------------------
    # Display the complete analysis results to the console.
    
    print("\n\nYour data:")
    print("\n- Standard info:")
    print(db)
    print("\n\n- Windowed ambiguity analysis:")
    print(db_ambiguity)


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()

