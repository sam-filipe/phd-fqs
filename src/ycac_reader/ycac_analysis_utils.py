#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YCAC Analysis Utilities
=======================
Utility functions for analysing the Yale-Classical Archives Corpus (YCAC).

This module provides functions for:
- User interaction and input handling
- Metadata and corpus file reading
- Data processing and transformation
- DFT (Discrete Fourier Transform) computation for pitch-class sets
- RadViz coordinate calculation
- Statistical analysis (ambiguity, diversity, windowed analysis)
- Data export to CSV and Excel formats

The functions in this module are designed to support the main YCAC corpus
analyser script and maintain reproducibility for thesis analysis.

Requirements:
    - Python 3.11.3+ (64-bit)
    - pandas, numpy, music21, tqdm

Author: [Your Name]
Date: [Original Date]
Refactored for PhD Thesis Repository
"""

import os
import collections as c

import numpy as np
import pandas as pd
import music21 as m21
from tqdm import tqdm


# =============================================================================
# 1. USER INTERACTION FUNCTIONS
# =============================================================================

def mode_definition():
    """
    Prompt the user to select an analysis mode.
    
    Displays the available modes and validates user input.
    
    Returns
    -------
    int
        The selected mode:
        - 1: Analyse specific composer(s)
        - 2: Analyse a specific file
        - 3: Analyse the entire corpus
    
    Raises
    ------
    SystemExit
        If the user enters an invalid mode selection.
    """
    print("\nMode options:\n 1 - composer(s)\n 2 - file\n 3 - corpus")
    mode = input(" Please select a mode:")
    
    if mode.isdigit() and int(mode) in range(1, 4):
        return int(mode)
    else:
        print("\nInvalid input. Restart the program and please select a valid mode: 1, 2, or 3.")
        quit()


def col_width(width):
    """
    Set the maximum column width for pandas DataFrame display.
    
    Parameters
    ----------
    width : int
        Maximum number of characters to display in DataFrame columns.
    """
    pd.set_option('display.max_colwidth', width)


def composer_definition():
    """
    Prompt the user to enter composer names for Mode 1 analysis.
    
    The user should enter composer names separated by commas, with
    capitalised first letters (e.g., "Mozart, Bach, Satie").
    
    Returns
    -------
    list of str
        List of composer names entered by the user.
    """
    print("\n\nPlease write the composers' names separated by commas with "
          "capitalized first letter:\n"
          "(...example: Mozart, Bach, Satie, Messiaen...)\n")
    names = input("Composers' names: ")
    composers = [name.strip() for name in names.split(',')]
    return composers


def file_definition(directory):
    """
    Display available files and prompt the user to select one for Mode 2.
    
    Parameters
    ----------
    directory : str
        System path to the YCAC corpus directory.
    
    Returns
    -------
    str
        The filename selected by the user.
    
    Raises
    ------
    SystemExit
        If the user enters an invalid filename.
    """
    # List all files except the metadata file
    dir_files = os.listdir(directory)
    dir_files.remove('0_Metadata.csv')
    files = ', '.join(dir_files)
    
    print("\n\nThe YCAC directory in your system has the following files:\n")
    print(files)
    print("\nPlease select one file to analyze, writing its name as shown "
          "in the list above:\n")
    
    file_name = str(input("File name: "))
    
    if file_name in dir_files:
        return file_name
    else:
        print("\nThat's not a valid file name. Please check if you spelled "
              "the file name correctly and restart the program!\n")
        quit()


# =============================================================================
# 2. METADATA AND DATA LOADING FUNCTIONS
# =============================================================================

def read_metadata(filepath):
    """
    Read the YCAC metadata CSV file into a DataFrame.
    
    Loads the metadata file and extracts relevant columns for analysis.
    Removes quotation marks from filenames for consistency.
    
    Parameters
    ----------
    filepath : str
        System path to the metadata CSV file (0_Metadata.csv).
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing columns: 'Title', 'Composer', 'Date', 'Range',
        'Filename', and 'Comments'.
    """
    metadata = pd.read_csv(
        filepath,
        usecols=['Title', 'Composer', 'Date', 'Range', 'Filename', 'Comments'],
        encoding='utf8',
        encoding_errors='replace'
    )
    # Remove quotation marks from filenames for consistent matching
    metadata['Filename'] = metadata['Filename'].str.replace('"', '')
    return metadata


def check_composers(metadata, composers):
    """
    Validate that requested composers exist in the corpus metadata.
    
    Checks each composer name against the metadata and reports any that
    are not found. Removes invalid composers from the list.
    
    Parameters
    ----------
    metadata : pandas.DataFrame
        DataFrame containing the corpus metadata with a 'Composer' column.
    composers : list of str
        List of composer names to validate.
    
    Returns
    -------
    list of str
        Filtered list containing only valid composer names.
    
    Raises
    ------
    SystemExit
        If none of the requested composers exist in the database.
    """
    # Identify composers not found in the metadata
    non_existing = []
    for composer in composers:
        if not metadata['Composer'].eq(composer).any():
            non_existing.append(composer)
    
    # Remove invalid composers from the list
    composers = [x for x in composers if x not in non_existing]
    
    # Handle different validation outcomes
    if len(composers) == 0:
        # No valid composers found
        print("\nNone of the selected composers exist in the database. "
              "Please restart the program.\n")
        quit()
    elif len(non_existing) != 0:
        # Some composers not found - warn the user but continue
        print('\nSorry, but "' + ", ".join(non_existing[:-1]) + " and " +
              non_existing[-1].capitalize() + '" is/are not valid or '
              'existing composer(s) in the database.')
        print('Please check if the name is spelled correctly.\n')
    
    return composers


def files_definition(directory, composers):
    """
    Determine which corpus files to analyse based on selected composers.
    
    Maps composer names to their corresponding data files in the corpus
    directory. Uses filename prefixes to identify the correct files.
    
    Parameters
    ----------
    directory : str
        System path to the YCAC corpus directory.
    composers : list of str
        List of validated composer names.
    
    Returns
    -------
    list of str
        List of filenames corresponding to the selected composers.
    """
    dir_files = os.listdir(directory)
    dir_files.remove('0_Metadata.csv')
    
    # Match composers to files by prefix, falling back to first letter match
    files = [
        next(
            (f for f in dir_files if f.startswith(c)),
            next((f for f in dir_files if f.startswith(c[0]) and len(f) == 11), None)
        )
        for c in composers
    ]
    return files


# =============================================================================
# 3. DATA PROCESSING LOOPS (by analysis mode)
# =============================================================================

def _clean_filename_column(df):
    """
    Standardise the Filename column by removing file extensions and special characters.
    
    This is a helper function used by all dataLoop functions.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a 'Filename' column.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with cleaned 'Filename' column.
    """
    df['Filename'] = df['Filename'].str.replace('.mid', '')
    df['Filename'] = df['Filename'].str.replace('.mxl', '')
    df['Filename'] = df['Filename'].str.replace('"', '')
    df['Filename'] = df['Filename'].str.replace('\u00A0', '_')  # Non-breaking space
    return df


def _process_nf_pc_columns(df, label=""):
    """
    Convert string representations of NFs and PCs to proper Python lists.
    
    The YCAC corpus stores normal forms and pitch classes as string
    representations of lists. This function converts them to actual lists.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'NFs' and 'PCs' columns as strings.
    label : str, optional
        Label for progress bar display (e.g., composer name or filename).
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with 'NFs' and 'PCs' as proper list objects.
    """
    print('\n\nListing NFs and PCs')
    
    # Convert Normal Forms (NFs) from string to list
    print(f' - {label} NFs:')
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#f7f0f5')
    df['NFs'] = df['NFs'].progress_apply(eval)
    
    # Convert Pitch Classes (PCs) from string to list
    print(f' - {label} PCs:')
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#f7f0f5')
    df['PCs'] = df['PCs'].progress_apply(eval)
    
    return df


def _filter_by_cardinality(df, label=""):
    """
    Filter pitch-class sets by cardinality (number of pitch classes).
    
    Removes sets with cardinality 12 (chromatic aggregate) and creates
    filtered versions excluding sets with cardinality 1 (single notes).
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'NFs' and 'PCs' columns.
    label : str, optional
        Label for progress bar display.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with additional 'NFs>1' and 'PCs>1' columns containing
        only sets with cardinality > 1.
    """
    # Remove cardinality = 12 (chromatic aggregate, not analytically useful)
    print(f' - {label} deleting cardinality = 12 in PCs:')
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#f7f0f5')
    df['PCs'] = df['PCs'].progress_apply(
        lambda x: [sublst for sublst in x if len(sublst) != 12]
    )
    
    print(f' - {label} deleting cardinality = 12 in NFs:')
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#f7f0f5')
    df['NFs'] = df['NFs'].progress_apply(
        lambda x: [sublst for sublst in x if len(sublst) != 12]
    )
    
    # Create columns with cardinality > 1 (excluding single notes)
    print(f' - {label} deleting cardinality = 1 in PCs:')
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#f7f0f5')
    df['PCs>1'] = df['PCs'].progress_apply(
        lambda x: [sublst for sublst in x if len(sublst) > 1]
    )
    
    print(f' - {label} deleting cardinality = 1 in NFs:')
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#f7f0f5')
    df['NFs>1'] = df['NFs'].progress_apply(
        lambda x: [sublst for sublst in x if len(sublst) > 1]
    )
    
    # Convert empty lists to NaN for later removal
    df['PCs>1'] = df['PCs>1'].apply(lambda x: np.nan if x == [] else x)
    df['NFs>1'] = df['NFs>1'].apply(lambda x: np.nan if x == [] else x)
    
    # Remove rows where filtering left no data
    df = df.dropna().reset_index(drop=True)
    
    return df


def dataLoop_m1(directory, composers, metadata, files):
    """
    Process corpus data for Mode 1 (specific composers).
    
    Iterates through the selected composers' files, extracts relevant data,
    and combines it into a single DataFrame for analysis.
    
    Parameters
    ----------
    directory : str
        System path to the YCAC corpus directory.
    composers : list of str
        List of validated composer names.
    metadata : pandas.DataFrame
        DataFrame containing corpus metadata.
    files : list of str
        List of filenames corresponding to each composer.
    
    Returns
    -------
    pandas.DataFrame
        Combined DataFrame with columns: 'Composer', 'Title', 'Date',
        'Range', 'NFs>1', and 'PCs>1'.
    """
    db = pd.DataFrame()
    
    for i in range(len(files)):
        file = files[i]
        filepath = directory + file
        composer = composers[i]
        
        # Get metadata for this composer
        meta = metadata[metadata['Composer'] == composer]
        
        # Read the slice data file
        df = pd.read_csv(
            filepath,
            usecols=['NormalForm', 'PCsInNormalForm', 'file', 'Composer'],
            encoding='utf8',
            encoding_errors='replace'
        )
        
        # Filter to the current composer and remove 'Anonymous' entries
        df = df[df['Composer'] == composer]
        df = df[df['Composer'] != 'Anonymous']
        
        # Rename columns for consistency
        df = df.rename(columns={
            'file': 'Filename',
            'NormalForm': 'NFs',
            'PCsInNormalForm': 'PCs'
        })
        
        # Clean filename column
        df = _clean_filename_column(df)
        
        # Convert string representations to lists
        df = _process_nf_pc_columns(df, label=composer)
        
        # Group by piece (filename) and aggregate the NFs and PCs
        df = df.groupby('Filename').agg({
            'NFs': lambda x: x.tolist(),
            'PCs': lambda x: x.tolist(),
            'Composer': 'first'
        }).reset_index()
        df = df[['Composer', 'Filename', 'NFs', 'PCs']]
        df = df.sort_values(by=['Composer']).reset_index(drop=True)
        
        # Filter by cardinality
        df = _filter_by_cardinality(df, label=composer)
        
        # Merge with metadata to get Title, Date, Range
        df = pd.merge(
            df,
            meta[['Filename', 'Title', 'Date', 'Range']],
            on='Filename',
            how='left'
        )
        
        # Drop columns no longer needed
        df = df.drop(['Filename', 'PCs', 'NFs'], axis=1)
        
        # Append to main database
        db = pd.concat([db, df], ignore_index=True)
    
    return db


def dataLoop_m2(directory, file_name):
    """
    Process corpus data for Mode 2 (single file).
    
    Processes a single corpus file and extracts all relevant data
    for analysis.
    
    Parameters
    ----------
    directory : str
        System path to the YCAC corpus directory.
    file_name : str
        Name of the file to analyse.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: 'Composer', 'Title', 'Date', 'Range',
        'NFs>1', and 'PCs>1'.
    """
    filepath = directory + file_name
    meta_filepath = directory + '0_Metadata.csv'
    
    # Read metadata
    meta = pd.read_csv(
        meta_filepath,
        usecols=['Title', 'Composer', 'Date', 'Range', 'Filename'],
        encoding='utf8',
        encoding_errors='replace'
    )
    
    # Read the slice data file
    df = pd.read_csv(
        filepath,
        usecols=['NormalForm', 'file', 'Composer', 'PCsInNormalForm'],
        encoding='utf8',
        encoding_errors='replace'
    )
    
    # Remove 'Anonymous' entries
    df = df[df['Composer'] != 'Anonymous']
    
    # Rename columns for consistency
    df = df.rename(columns={
        'file': 'Filename',
        'NormalForm': 'NFs',
        'PCsInNormalForm': 'PCs'
    })
    
    # Clean filename column
    df = _clean_filename_column(df)
    
    # Convert string representations to lists
    df = _process_nf_pc_columns(df, label=file_name)
    
    # Group by piece and aggregate
    df = df.groupby('Filename').agg({
        'NFs': lambda x: x.tolist(),
        'PCs': lambda x: x.tolist(),
        'Composer': 'first'
    }).reset_index()
    df = df[['Composer', 'Filename', 'NFs', 'PCs']]
    df = df.sort_values(by=['Composer']).reset_index(drop=True)
    
    # Filter by cardinality
    df = _filter_by_cardinality(df, label=file_name)
    
    # Merge with metadata
    df = pd.merge(
        df,
        meta[['Filename', 'Title', 'Date', 'Range']],
        on='Filename',
        how='left'
    )
    
    # Drop columns no longer needed
    df = df.drop(['Filename', 'PCs', 'NFs'], axis=1)
    
    return df


def dataLoop_m3(directory, metadata):
    """
    Process corpus data for Mode 3 (entire corpus).
    
    Iterates through all files in the YCAC corpus directory and combines
    them into a single DataFrame for comprehensive analysis.
    
    Parameters
    ----------
    directory : str
        System path to the YCAC corpus directory.
    metadata : pandas.DataFrame
        DataFrame containing corpus metadata.
    
    Returns
    -------
    pandas.DataFrame
        Combined DataFrame with columns: 'Composer', 'Title', 'Date',
        'Range', 'NFs>1', and 'PCs>1'.
    """
    print('\n\n(...proceeding with processing; this task may take a few minutes...)\n\n')
    
    db = pd.DataFrame()
    
    # Get all corpus files (excluding metadata)
    dir_files = os.listdir(directory)
    dir_files.remove('0_Metadata.csv')
    
    for file in dir_files:
        filepath = directory + file
        
        # Read the slice data file
        df = pd.read_csv(
            filepath,
            usecols=['NormalForm', 'file', 'Composer', 'PCsInNormalForm'],
            encoding='utf8',
            encoding_errors='replace'
        )
        
        # Remove 'Anonymous' entries
        df = df[df['Composer'] != 'Anonymous']
        
        # Rename columns for consistency
        df = df.rename(columns={
            'file': 'Filename',
            'NormalForm': 'NFs',
            'PCsInNormalForm': 'PCs'
        })
        
        # Clean filename column
        df = _clean_filename_column(df)
        
        # Convert string representations to lists
        df = _process_nf_pc_columns(df, label=file)
        
        # Group by piece and aggregate
        df = df.groupby('Filename').agg({
            'NFs': lambda x: x.tolist(),
            'PCs': lambda x: x.tolist(),
            'Composer': 'first'
        }).reset_index()
        df = df[['Composer', 'Filename', 'NFs', 'PCs']]
        df = df.sort_values(by=['Composer']).reset_index(drop=True)
        
        # Filter by cardinality
        df = _filter_by_cardinality(df, label=file)
        
        # Merge with metadata
        df = pd.merge(
            df,
            metadata[['Filename', 'Date', 'Range', 'Title']],
            on='Filename',
            how='left'
        )
        
        # Drop columns no longer needed
        df = df.drop(['Filename', 'PCs', 'NFs'], axis=1)
        
        # Append to main database
        db = pd.concat([db, df], ignore_index=True)
    
    return db


# =============================================================================
# 4. YEAR ESTIMATION
# =============================================================================

def year_definition(df):
    """
    Estimate composition years for pieces with undefined dates.
    
    For pieces without a specific composition date, uses the upper bound
    of the composition range as an estimate (e.g., '1910-1920' → 1920).
    Pieces with no date or range are marked as 'Undetermined'.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'Date' and 'Range' columns.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with new 'Year' column containing estimated years.
    """
    print(' - determining undefined composition years:')
    
    # Extract the upper bound year from the Range column
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#f7f0f5')
    df['range_mean'] = df['Range'].progress_apply(
        lambda x: pd.to_numeric(x.split('-'))[1]
        if isinstance(x, str) and '-' in x else None
    )
    
    # Fill missing Date values with the estimated year from Range
    df['year_filled'] = df['Date'].fillna(df['range_mean'])
    df['year_filled'] = pd.to_numeric(df['year_filled'], errors='coerce')
    df['year_filled'] = df['year_filled'].fillna(0)
    
    # Convert to integer year, marking zeros as 'Undetermined'
    df['Year'] = df['year_filled'].astype(int)
    df['Year'] = df['Year'].apply(lambda x: 'Undetermined' if x == 0 else x)
    
    # Clean up intermediate columns
    df.drop(['range_mean', 'year_filled'], axis=1, inplace=True)
    
    return df


# =============================================================================
# 5. DISCRETE FOURIER TRANSFORM (DFT) FUNCTIONS
# =============================================================================

def binary_list(lst):
    """
    Convert a pitch-class list to a 12-element binary vector.
    
    Creates a binary representation where each position indicates
    whether that pitch class (0-11) is present in the input list.
    
    Parameters
    ----------
    lst : list of int
        List of pitch classes (integers 0-11).
    
    Returns
    -------
    list of int
        12-element binary list where 1 indicates presence of that pitch class.
    
    Examples
    --------
    >>> binary_list([0, 4, 7])
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    
    >>> binary_list([0, 11])
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    """
    return [1 if i in lst else 0 for i in range(12)]


def dft(lst):
    """
    Compute the Discrete Fourier Transform for a binary pitch-class vector.
    
    The DFT of a pitch-class set reveals its interval-class content
    through the magnitudes and phases of the Fourier coefficients.
    
    Parameters
    ----------
    lst : list of int
        12-element binary pitch-class vector.
    
    Returns
    -------
    list of complex
        12 complex DFT coefficients.
    """
    return np.fft.fft(lst).tolist()


def dft_data(df):
    """
    Compute DFT coefficients for all pitch-class sets in the DataFrame.
    
    Calculates both magnitudes and phases of the six non-trivial DFT
    coefficients (f1 through f6) for each pitch-class set. The trivial
    coefficients f0 (cardinality) and f7-f11 (conjugates) are excluded.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'PCs>1' column containing pitch-class sets.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with new columns:
        - 'DFTMag>1': Magnitudes of coefficients f1-f6
        - 'DFTPha>1': Phases of coefficients f1-f6
    """
    print('\n\nDFT Computation')
    
    # Convert pitch-class sets to binary vectors
    print(' - Computing binary vectors:')
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#fff3b0')
    df['PCsVct>1'] = df['PCs>1'].progress_apply(
        lambda x: [binary_list(sublst) for sublst in x]
    )
    
    # Compute DFT coefficient magnitudes
    print(' - Computing DFT coefficients magnitude:')
    df['DFTMag>1'] = df['PCsVct>1'].progress_apply(
        lambda x: [np.abs(dft(sublst)) for sublst in x]
    )
    
    # Compute DFT coefficient phases
    df['DFTPha>1'] = df['PCsVct>1'].progress_apply(
        lambda x: [np.angle(dft(sublst)) for sublst in x]
    )
    
    # Remove intermediate binary vector column
    df = df.drop(['PCsVct>1'], axis=1)
    
    # Extract only the non-trivial coefficients (indices 1-6, i.e., f1 to f6)
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#fff3b0')
    df['DFTMag>1'] = df['DFTMag>1'].progress_apply(
        lambda x: [sublist[1:7] for sublist in x]
    )
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#fff3b0')
    df['DFTPha>1'] = df['DFTPha>1'].progress_apply(
        lambda x: [sublist[1:7] for sublist in x]
    )
    
    return df


# =============================================================================
# 6. RADVIZ VISUALISATION FUNCTIONS
# =============================================================================

def radviz_anchors():
    """
    Calculate the anchor coordinates for a 6-dimensional RadViz visualisation.
    
    Places six anchor points equidistantly around a unit circle.
    These anchors correspond to the six non-trivial DFT coefficients.
    
    Returns
    -------
    list of list
        Six [x, y] coordinate pairs for the anchor points.
    """
    n = 6
    angle = 2 * np.pi / n
    anchors = [[np.cos(i * angle), np.sin(i * angle)] for i in range(n)]
    return anchors


def radviz_data(df, anchors, order):
    """
    Calculate RadViz coordinates for all pitch-class sets.
    
    Projects each pitch-class set onto a 2D plane using the RadViz
    algorithm, where the position is determined by a weighted average
    of anchor positions based on DFT coefficient magnitudes.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'DFTMag>1' column containing DFT magnitudes.
    anchors : list of list
        RadViz anchor coordinates from radviz_anchors().
    order : list of int
        Ordering of DFT coefficients around the RadViz circle.
        Typically [1, 5, 3, 4, 6, 2] for optimal visual arrangement.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with new 'RadViz>1' column containing [x, y] coordinates.
    """
    n = 6  # Number of DFT coefficients/anchors
    
    print('\n\nRadViz Data')
    
    # Rearrange DFT magnitudes according to specified order
    print(' - Rearranging elements to RadViz order:')
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#7adfbb')
    df['Order>1'] = df['DFTMag>1'].progress_apply(
        lambda x: [[x[i][j-1] for j in order] for i in range(len(x))]
    )
    
    # Calculate RadViz coordinates using weighted average formula
    # x = Σ(anchor_x * weight) / Σ(weight)
    # y = Σ(anchor_y * weight) / Σ(weight)
    print(' - Computing RadViz coordinates:')
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#7adfbb')
    df['RadViz>1'] = df['Order>1'].progress_apply(
        lambda x: [
            [
                sum(anchors[j][0] * x[i][j] for j in range(n)) / sum(x[i]),
                sum(anchors[j][1] * x[i][j] for j in range(n)) / sum(x[i])
            ]
            for i in range(len(x))
        ]
    )
    
    # Remove date-related columns (no longer needed after year estimation)
    df = df.drop(['Date', 'Range'], axis=1)
    
    return df


# =============================================================================
# 7. STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================

def sc_counter(nf_list):
    """
    Identify the three most frequent set classes in a piece.
    
    Counts normal form occurrences, converts them to prime form (set class),
    and returns the top 3 most frequent set classes with their counts.
    
    Parameters
    ----------
    nf_list : list of list
        List of normal forms (each a list of pitch classes).
    
    Returns
    -------
    list
        Flattened list: [SC1, count1, SC2, count2, SC3, count3]
        where each SC is a list representing the prime form.
        Missing entries are filled with None.
    """
    # Count occurrences of each normal form
    nf_tuples = [tuple(sublist) for sublist in nf_list]
    occurrences = c.Counter(nf_tuples)
    top10 = occurrences.most_common(10)
    
    # Flatten the top 10 results
    top10_occ = [item for sublist in top10 for item in sublist]
    top10_occurrences = [
        list(item) if isinstance(item, tuple) else item
        for item in top10_occ
    ]
    
    # Convert normal forms to set classes (prime forms)
    for i in range(len(top10_occurrences)):
        if i == 0 or i % 2 == 0:
            top10_occurrences[i] = m21.chord.Chord(top10_occurrences[i]).primeForm
    
    # Expand by repetition count for accurate set-class frequency
    expanded_list = []
    for i in range(len(top10_occurrences)):
        if i % 2 != 0:  # This is a count
            for _ in range(top10_occurrences[i]):
                expanded_list.append(top10_occurrences[i-1])
    
    # Re-count to get set-class frequencies
    sc_tuples = [tuple(sublist) for sublist in expanded_list]
    sc_occurrences = c.Counter(sc_tuples)
    sc_top3 = sc_occurrences.most_common(3)
    
    # Flatten the top 3 results
    sc_top3_occ = [item for sublist in sc_top3 for item in sublist]
    sc_top3_occurrences = [
        list(item) if isinstance(item, tuple) else item
        for item in sc_top3_occ
    ]
    
    # Pad with None if fewer than 3 set classes found
    while len(sc_top3_occurrences) != 6:
        sc_top3_occurrences.append(None)
    
    return sc_top3_occurrences


def piece_stats(df):
    """
    Compute aggregate statistics for each piece.
    
    Calculates:
    - Mean cardinality of pitch-class sets
    - Top 3 most frequent set classes
    - Mean DFT coefficient magnitudes (|f1| to |f6|)
    - Tonal Index (TI = phase_f2 + phase_f3 - phase_f5)
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with pitch-class and DFT data.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with statistics columns added.
    """
    print('\n\nPiece stats')
    
    # Mean cardinality
    print(' - Computing mean cardinality:')
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#313B72')
    df['CardMean'] = df['PCs>1'].progress_apply(
        lambda x: sum(len(sublist) for sublist in x) / len(x)
    )
    
    # Most frequent set classes
    print(' - Computing preponderant set classes:')
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#313B72')
    df['data'] = df['NFs>1'].progress_apply(sc_counter)
    
    # Extract individual set classes and counts
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#313B72')
    df['1SC'] = df['data'].progress_apply(lambda x: x[0])
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#313B72')
    df['#1SC'] = df['data'].progress_apply(lambda x: x[1])
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#313B72')
    df['2SC'] = df['data'].progress_apply(lambda x: x[2])
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#313B72')
    df['#2SC'] = df['data'].progress_apply(lambda x: x[3])
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#313B72')
    df['3SC'] = df['data'].progress_apply(lambda x: x[4])
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#313B72')
    df['#3SC'] = df['data'].progress_apply(lambda x: x[5])
    df = df.drop(['data'], axis=1)
    
    # Mean DFT coefficient magnitudes
    print(' - Computing mean DFT coefficient magnitudes:')
    magnitude_columns = ['|f1|', '|f2|', '|f3|', '|f4|', '|f5|', '|f6|']
    for i, column in enumerate(magnitude_columns):
        tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#313B72')
        df[column] = df['DFTMag>1'].progress_apply(
            lambda x: pd.Series(x).apply(lambda sublist: sublist[i]).mean()
        )
    
    # Tonal Index (uses phases of f2, f3, and f5)
    print(' - Computing the Tonal Index:')
    phase_columns = ['ph2', 'ph3', 'ph5']
    phase_indices = [1, 2, 4]  # Indices for f2, f3, f5 in the phase array
    
    for i, column in enumerate(phase_columns):
        k = phase_indices[i]
        tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#313B72')
        df[column] = df['DFTPha>1'].progress_apply(
            lambda x: pd.Series(x).apply(lambda sublist: sublist[k]).mean()
        )
    
    # Compute Tonal Index: TI = phase_f2 + phase_f3 - phase_f5
    df['TI'] = df['ph2'] + df['ph3'] - df['ph5']
    
    # Clean up and reorganise columns
    df = df.drop(['PCs>1', 'NFs>1', 'DFTPha>1'], axis=1)
    df = df[[
        'DFTMag>1', 'Order>1', 'RadViz>1', 'Composer', 'Title', 'Year',
        'CardMean', '1SC', '#1SC', '2SC', '#2SC', '3SC', '#3SC',
        '|f1|', '|f2|', '|f3|', '|f4|', '|f5|', '|f6|',
        'ph2', 'ph3', 'ph5', 'TI'
    ]]
    df = df.drop(['DFTMag>1', 'Order>1'], axis=1)
    
    return df


def ambiguity(df):
    """
    Calculate macroharmonic ambiguity for each piece.
    
    Ambiguity measures how close pitch-class sets cluster to the centre
    of the RadViz space. Higher values (closer to 1) indicate more
    ambiguous or chromatic harmony where no single pitch-class hierarchy
    dominates. Lower values indicate clearer tonal orientation.
    
    The formula is: Amb = 1 - mean(distance from origin)
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'RadViz>1' column containing coordinates.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with new 'Amb' column.
    """
    print('\n\nComputing macroharmonic ambiguity and diversity')
    print(' - Ambiguity:')
    
    tqdm.pandas(bar_format='{l_bar}{bar:50}{r_bar}', colour='#5d2e8c')
    df['Amb'] = df['RadViz>1'].progress_apply(
        lambda points: 1 - np.mean([np.linalg.norm(point) for point in points])
        if len(points) > 1 else 1
    )
    
    return df


# =============================================================================
# 8. WINDOWED TEMPORAL ANALYSIS
# =============================================================================

def windowed_analysis(df, win_size):
    """
    Perform sliding-window analysis of ambiguity over time.
    
    Divides each piece into overlapping windows and calculates ambiguity
    for each window, enabling analysis of how harmonic characteristics 
    evolve throughout a piece.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'RadViz>1' column and piece metadata.
    win_size : int
        Size of the analysis window (number of time slices).
        Must be an even number >= 8.
    
    Returns
    -------
    tuple of pandas.DataFrame
        (df_ambiguity):
        - df_ambiguity: Window-by-window ambiguity values
        Both include 'Composer', 'Title', and 'Year' columns.
    """
    # Initialise output DataFrames
    df_amb = pd.DataFrame()
    
    # Validate and adjust window size
    window_size = win_size
    if window_size % 2 != 0:
        window_size = win_size - 1  # Ensure even number
    if window_size <= 6:
        window_size = 8  # Minimum window size
    
    # Window overlap is half the window size
    interpolation_size = int(window_size / 2)
    
    # Origin point for ambiguity calculation
    centre = np.array([0, 0])
    
    print('\n\nPerforming windowed analysis:')
    
    for i in tqdm(range(len(df)),
                  bar_format='{l_bar}{bar:50}{r_bar}',
                  colour='#2a9d8f'):
        
        radviz_coords = df['RadViz>1'][i]
        n = len(radviz_coords)
        
        # Storage for this piece's windowed values
        row_amb_values = {}
        
        # Generate window indices with overlap
        windows_index = []
        start_num = 0
        
        while start_num <= n:
            end_num = min(start_num + window_size - 1, n - 1)
            windows_index.append([start_num, end_num])
            
            if end_num == n - 1 or end_num == n:
                break
            else:
                start_num += interpolation_size
        
        # Calculate ambiguity for each window
        for j, (win_start, win_end) in enumerate(windows_index):
            window_coords = radviz_coords[win_start:win_end + 1]
            
            # Ambiguity: 1 - std of distances from origin
            distances_from_origin = [
                np.linalg.norm(k - centre) for k in window_coords
            ]
            amb = 1 - np.std(distances_from_origin)
            
            column_name = f'w{j + 1}'
            row_amb_values[column_name] = amb
        
        # Append row to output DataFrames
        df_amb = pd.concat(
            [df_amb, pd.DataFrame([row_amb_values])],
            ignore_index=True
        )
    
    # Add metadata columns
    subset_df = df[['Composer', 'Title', 'Year']]
    df_amb = pd.concat([subset_df.reset_index(drop=True), df_amb], axis=1)
    
    return df_amb


# =============================================================================
# 9. DATA EXPORT FUNCTIONS
# =============================================================================

def export_data(db, db_amb, name):
    """
    Export analysis results to CSV and Excel files.
    
    Prompts the user to confirm export, then saves three datasets:
    - Standard analysis data (*_std.csv, *_std.xlsx)
    - Windowed ambiguity data (*_amb.csv, *_amb.xlsx)
    
    Files are saved in the same directory as the script.
    
    Parameters
    ----------
    db : pandas.DataFrame
        Main analysis DataFrame.
    db_amb : pandas.DataFrame
        Windowed ambiguity DataFrame.
    name : str
        Base filename for output files.
    """
    print('\n\nDo you want to export your data to csv and excel files?\n\nOptions: y/n')
    answer = input('\nYour answer: ').lower()
    
    if answer in ['y', 'yes']:
        data = [db, db_amb]
        suffixes = ['_std', '_amb']
        sheet_names = ['FMSData', 'AMBData']
        labels = ['standard', 'ambiguity']
        
        for k, (dataframe, suffix, sheet, label) in enumerate(
            zip(data, suffixes, sheet_names, labels)
        ):
            # Determine number of chunks for progress display
            chunks = np.array_split(dataframe.index, len(dataframe))
            
            # Get current directory for output path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            csv_name = f'{name}{suffix}.csv'
            excel_name = f'{name}{suffix}.xlsx'
            csv_path = os.path.join(current_dir, csv_name)
            excel_path = os.path.join(current_dir, excel_name)
            
            # --- Export to CSV ---
            # Remove existing file if present
            if os.path.exists(csv_path):
                os.remove(csv_path)
            
            print(f'\n\n - Exporting {label} data to .csv format:')
            for chunk, subset in enumerate(
                tqdm(chunks, bar_format='{l_bar}{bar:50}{r_bar}', colour='#fc7753')
            ):
                if chunk == 0:
                    dataframe.loc[subset].to_csv(csv_path, mode='w', index=False)
                else:
                    dataframe.loc[subset].to_csv(
                        csv_path, header=None, mode='a', index=False
                    )
            
            # --- Export to Excel ---
            # Remove existing file if present
            if os.path.exists(excel_path):
                os.remove(excel_path)
            
            print(f'\n- Exporting {label} data to .xlsx format:')
            
            # Create Excel file with header
            dataframe.iloc[:0].to_excel(excel_path, sheet_name=sheet, index=False)
            
            # Append data in chunks
            with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='overlay') as writer:
                for chunk, subset in enumerate(
                    tqdm(chunks, bar_format='{l_bar}{bar:50}{r_bar}', colour='#0c8346')
                ):
                    if chunk == 0:
                        dataframe.loc[subset].to_excel(
                            writer, sheet_name=sheet, index=False
                        )
                    else:
                        dataframe.loc[subset].to_excel(
                            writer, sheet_name=sheet, startrow=chunk + 1,
                            header=None, index=False
                        )
    
    elif answer in ['n', 'no']:
        pass
    else:
        print("\nInvalid answer. Please restart the program and try again.")
        quit()
