"""
Fourier Quotient Space Explorer for Pitch Class Set Classes
============================================================

This module provides an interactive visualisation of all pitch class set classes
positioned in a Fourier Quotient Space (FQS) using a RadViz-style representation.
It is part of the analytical tools developed for the author's PhD thesis on
pitch class set theory and Fourier analysis.

The explorer allows users to:
    - View all set classes plotted according to their Fourier coefficient magnitudes
    - Click on points to see which set classes occupy that position
    - Identify set classes that share the same FQS position (Z-related pairs, etc.)
    - Find set classes within the "ambiguity zone" (central region)
    - See related set classes by proximity in the space

Mathematical Background
-----------------------
The Fourier Quotient Space is constructed by computing the DFT of each set class
(in binary representation), extracting the magnitudes of coefficients 1-6, and
projecting them onto a hexagonal RadViz-style plot. Each vertex of the hexagon
corresponds to a Fourier coefficient (f1, f2, ..., f6), and points are positioned
as weighted averages of the vertex positions, where weights are the coefficient
magnitudes.

The "ambiguity zone" (inner circle with radius 1/6) contains set classes with
relatively balanced coefficient magnitudes, making them less distinctive in
terms of Fourier properties.

Usage
-----
Run this script directly to launch the interactive visualisation:
    $ python fourier_quotient_space_explorer.py

Dependencies
------------
- numpy: Numerical computations and FFT
- matplotlib: Plotting and interactive widgets
- scipy: KD-tree for efficient nearest-neighbour lookups

Author: [Thesis Author]
Created for: PhD Thesis on Pitch Class Set Theory and Fourier Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from math import pi, sin, cos
from scipy.spatial import KDTree


# =============================================================================
# Configuration Constants
# =============================================================================

# Fourier coefficient ordering for RadViz axes
# This ordering places related coefficients at appropriate positions on the hexagon
COEFFICIENT_ORDER = [1, 5, 3, 4, 6, 2]

# Number of Fourier coefficients (excluding the DC component, coefficient 0)
NUM_COEFFICIENTS = 6

# Angle between adjacent vertices of the hexagon
VERTEX_ANGLE = 2 * pi / NUM_COEFFICIENTS

# Radius of the "ambiguity zone" - set classes inside this circle have
# relatively balanced Fourier coefficients
AMBIGUITY_ZONE_RADIUS = 1 / 6


# =============================================================================
# Set Class Data
# =============================================================================

# Complete list of pitch class set classes in normal form
# Each set class is represented as a list of pitch class integers (0-11)
# This list includes all unique set classes under transposition and inversion
SET_CLASS_LIST = [
    # Cardinality 1 (monads)
    [0],
    
    # Cardinality 2 (dyads) and their complements (cardinality 10)
    [0, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 2], [0, 1, 2, 3, 4, 5, 6, 7, 8, 10],
    [0, 3], [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
    [0, 4], [0, 1, 2, 3, 4, 5, 6, 8, 9, 10],
    [0, 5], [0, 1, 2, 3, 4, 5, 7, 8, 9, 10],
    [0, 6], [0, 1, 2, 3, 4, 6, 7, 8, 9, 10],
    
    # Cardinality 3 (trichords) and their complements (cardinality 9)
    [0, 1, 2], [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [0, 1, 3], [0, 1, 2, 3, 4, 5, 6, 7, 9],
    [0, 1, 4], [0, 1, 2, 3, 4, 5, 6, 8, 9],
    [0, 1, 5], [0, 1, 2, 3, 4, 5, 7, 8, 9],
    [0, 1, 6], [0, 1, 2, 3, 4, 6, 7, 8, 9],
    [0, 2, 4], [0, 1, 2, 3, 4, 5, 6, 8, 10],
    [0, 2, 5], [0, 1, 2, 3, 4, 5, 7, 8, 10],
    [0, 2, 6], [0, 1, 2, 3, 4, 6, 7, 8, 10],
    [0, 2, 7], [0, 1, 2, 3, 5, 6, 7, 8, 10],
    [0, 3, 6], [0, 1, 2, 3, 4, 6, 7, 9, 10],
    [0, 3, 7], [0, 1, 2, 3, 5, 6, 7, 9, 10],
    [0, 4, 8], [0, 1, 2, 4, 5, 6, 8, 9, 10],
    
    # Cardinality 4 (tetrachords) and their complements (cardinality 8)
    [0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7],
    [0, 1, 2, 4], [0, 1, 2, 3, 4, 5, 6, 8],
    [0, 1, 2, 5], [0, 1, 2, 3, 4, 5, 7, 8],
    [0, 1, 2, 6], [0, 1, 2, 3, 4, 6, 7, 8],
    [0, 1, 2, 7], [0, 1, 2, 3, 5, 6, 7, 8],
    [0, 1, 3, 4], [0, 1, 2, 3, 4, 5, 6, 9],
    [0, 1, 3, 5], [0, 1, 2, 3, 4, 5, 7, 9],
    [0, 1, 3, 6], [0, 1, 2, 3, 4, 6, 7, 9],
    [0, 1, 3, 7], [0, 1, 2, 3, 5, 6, 7, 9],
    [0, 1, 4, 5], [0, 1, 2, 3, 4, 5, 8, 9],
    [0, 1, 4, 6], [0, 1, 2, 3, 4, 6, 8, 9],
    [0, 1, 4, 7], [0, 1, 2, 3, 5, 6, 8, 9],
    [0, 1, 4, 8], [0, 1, 2, 4, 5, 6, 8, 9],
    [0, 1, 5, 6], [0, 1, 2, 3, 4, 7, 8, 9],
    [0, 1, 5, 7], [0, 1, 2, 3, 5, 7, 8, 9],
    [0, 1, 5, 8], [0, 1, 2, 4, 5, 7, 8, 9],
    [0, 1, 6, 7], [0, 1, 2, 3, 6, 7, 8, 9],
    [0, 2, 3, 5], [0, 2, 3, 4, 5, 6, 7, 9],
    [0, 2, 3, 6], [0, 1, 3, 4, 5, 6, 7, 9],
    [0, 2, 3, 7], [0, 1, 2, 4, 5, 6, 7, 9],
    [0, 2, 4, 6], [0, 1, 2, 3, 4, 6, 8, 10],
    [0, 2, 4, 7], [0, 1, 2, 3, 5, 6, 8, 10],
    [0, 2, 4, 8], [0, 1, 2, 4, 5, 6, 8, 10],
    [0, 2, 5, 7], [0, 1, 2, 3, 5, 7, 8, 10],
    [0, 2, 5, 8], [0, 1, 2, 4, 5, 7, 8, 10],
    [0, 2, 6, 8], [0, 1, 2, 4, 6, 7, 8, 10],
    [0, 3, 4, 7], [0, 1, 3, 4, 5, 6, 8, 9],
    [0, 3, 5, 8], [0, 1, 3, 4, 5, 7, 8, 10],
    [0, 3, 6, 9], [0, 1, 3, 4, 6, 7, 9, 10],
    
    # Cardinality 5 (pentachords) and their complements (cardinality 7)
    [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6],
    [0, 1, 2, 3, 5], [0, 1, 2, 3, 4, 5, 7],
    [0, 1, 2, 3, 6], [0, 1, 2, 3, 4, 6, 7],
    [0, 1, 2, 3, 7], [0, 1, 2, 3, 5, 6, 7],
    [0, 1, 2, 4, 5], [0, 1, 2, 3, 4, 5, 8],
    [0, 1, 2, 4, 6], [0, 1, 2, 3, 4, 6, 8],
    [0, 1, 2, 4, 7], [0, 1, 2, 3, 5, 6, 8],
    [0, 1, 2, 4, 8], [0, 1, 2, 4, 5, 6, 8],
    [0, 1, 2, 5, 6], [0, 1, 2, 3, 4, 7, 8],
    [0, 1, 2, 5, 7], [0, 1, 2, 3, 5, 7, 8],
    [0, 1, 2, 5, 8], [0, 1, 2, 4, 5, 7, 8],
    [0, 1, 2, 6, 7], [0, 1, 2, 3, 6, 7, 8],
    [0, 1, 2, 6, 8], [0, 1, 2, 4, 6, 7, 8],
    [0, 1, 3, 4, 6], [0, 1, 2, 3, 4, 6, 9],
    [0, 1, 3, 4, 7], [0, 1, 2, 3, 5, 6, 9],
    [0, 1, 3, 4, 8], [0, 1, 2, 4, 5, 6, 9],
    [0, 1, 3, 5, 6], [0, 1, 2, 3, 4, 7, 9],
    [0, 1, 3, 5, 7], [0, 1, 2, 3, 5, 7, 9],
    [0, 1, 3, 5, 8], [0, 1, 2, 4, 5, 7, 9],
    [0, 1, 3, 6, 7], [0, 1, 2, 3, 6, 7, 9],
    [0, 1, 3, 6, 8], [0, 1, 2, 4, 6, 7, 9],
    [0, 1, 3, 6, 9], [0, 1, 3, 4, 6, 7, 9],
    [0, 1, 4, 5, 7], [0, 1, 4, 5, 6, 7, 9],
    [0, 1, 4, 5, 8], [0, 1, 2, 4, 5, 8, 9],
    [0, 1, 4, 6, 8], [0, 1, 2, 4, 6, 8, 9],
    [0, 1, 4, 6, 9], [0, 1, 3, 4, 6, 8, 9],
    [0, 1, 4, 7, 8], [0, 1, 2, 5, 6, 8, 9],
    [0, 1, 5, 6, 8], [0, 1, 2, 5, 6, 7, 9],
    [0, 2, 3, 4, 6], [0, 2, 3, 4, 5, 6, 8],
    [0, 2, 3, 4, 7], [0, 1, 3, 4, 5, 6, 8],
    [0, 2, 3, 5, 7], [0, 2, 3, 4, 5, 7, 9],
    [0, 2, 3, 5, 8], [0, 2, 3, 4, 6, 7, 9],
    [0, 2, 3, 6, 8], [0, 1, 3, 5, 6, 7, 9],
    [0, 2, 4, 5, 8], [0, 1, 3, 4, 5, 7, 9],
    [0, 2, 4, 6, 8], [0, 1, 2, 4, 6, 8, 10],
    [0, 2, 4, 6, 9], [0, 1, 3, 4, 6, 8, 10],
    [0, 2, 4, 7, 9], [0, 1, 3, 5, 6, 8, 10],
    [0, 3, 4, 5, 8], [0, 1, 3, 4, 5, 7, 8],
    
    # Cardinality 6 (hexachords)
    [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 6],
    [0, 1, 2, 3, 4, 7], [0, 1, 2, 3, 5, 6],
    [0, 1, 2, 3, 4, 8], [0, 1, 2, 4, 5, 6],
    [0, 1, 2, 3, 5, 7], [0, 1, 2, 3, 5, 8],
    [0, 1, 2, 4, 5, 7], [0, 1, 2, 3, 6, 7],
    [0, 1, 2, 3, 6, 8], [0, 1, 2, 4, 6, 7],
    [0, 1, 2, 3, 6, 9], [0, 1, 3, 4, 6, 7],
    [0, 1, 2, 3, 7, 8], [0, 1, 2, 5, 6, 7],
    [0, 1, 2, 4, 5, 8], [0, 1, 2, 4, 6, 8],
    [0, 1, 2, 4, 6, 9], [0, 1, 3, 4, 6, 8],
    [0, 1, 2, 4, 7, 8], [0, 1, 2, 5, 6, 8],
    [0, 1, 2, 4, 7, 9], [0, 1, 3, 5, 6, 8],
    [0, 1, 2, 5, 6, 9], [0, 1, 3, 4, 7, 8],
    [0, 1, 2, 5, 7, 8], [0, 1, 2, 5, 7, 9],
    [0, 1, 3, 5, 7, 8], [0, 1, 2, 6, 7, 8],
    [0, 1, 3, 4, 5, 7], [0, 2, 3, 4, 5, 8],
    [0, 1, 3, 4, 5, 8], [0, 1, 3, 4, 6, 9],
    [0, 1, 3, 4, 7, 9], [0, 1, 3, 5, 6, 9],
    [0, 1, 3, 5, 7, 9], [0, 1, 3, 6, 7, 9],
    [0, 1, 4, 5, 6, 8], [0, 1, 4, 5, 7, 9],
    [0, 1, 4, 5, 8, 9], [0, 1, 4, 6, 7, 9],
    [0, 2, 3, 6, 7, 9], [0, 2, 3, 4, 5, 7],
    [0, 2, 3, 4, 6, 8], [0, 2, 3, 4, 6, 9],
    [0, 2, 3, 5, 6, 8], [0, 2, 3, 5, 7, 9],
    [0, 2, 4, 5, 7, 9], [0, 2, 4, 6, 8, 10],
    
    # Cardinality 12 (aggregate)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
]


# =============================================================================
# Geometric Configuration
# =============================================================================

def compute_hexagon_vertices():
    """
    Compute the (x, y) coordinates of the hexagon vertices.
    
    Each vertex corresponds to one of the six Fourier coefficients (f1-f6).
    Vertices are positioned on the unit circle at equal angular intervals.
    
    Returns
    -------
    list of list
        List of [x, y] coordinates for each vertex.
    """
    return [[cos(i * VERTEX_ANGLE), sin(i * VERTEX_ANGLE)] 
            for i in range(NUM_COEFFICIENTS)]


def compute_label_positions(anchors):
    """
    Compute positions for coefficient labels, offset inwards from vertices.
    
    Parameters
    ----------
    anchors : list of list
        The hexagon vertex coordinates.
    
    Returns
    -------
    list of list
        List of [x, y] coordinates for label placement.
    """
    # Manual offsets to position labels clearly inside the hexagon
    offsets = [
        [-0.075, -0.01],   # f1
        [-0.04, -0.065],   # f5
        [0, -0.065],       # f3
        [0.035, -0.015],   # f4
        [0, 0.03],         # f6
        [-0.04, 0.03]      # f2
    ]
    
    return [[a + o for a, o in zip(anchors[i], offsets[i])] 
            for i in range(NUM_COEFFICIENTS)]


def compute_edge_midpoints(anchors):
    """
    Compute the midpoints of each hexagon edge.
    
    These are used for drawing radial guidelines from the centre.
    
    Parameters
    ----------
    anchors : list of list
        The hexagon vertex coordinates.
    
    Returns
    -------
    list of list
        List of [x, y] coordinates for edge midpoints.
    """
    return [[(a + anchors[(i + 1) % NUM_COEFFICIENTS][j]) / 2 
             for j, a in enumerate(anchors[i])] 
            for i in range(NUM_COEFFICIENTS)]


# =============================================================================
# Fourier Analysis Functions
# =============================================================================

def set_class_to_binary(set_class):
    """
    Convert a set class (normal form) to a 12-element binary representation.
    
    Parameters
    ----------
    set_class : list of int
        Pitch class set in normal form (e.g., [0, 4, 7] for major triad).
    
    Returns
    -------
    list of int
        12-element binary list where 1 indicates pitch class membership.
    
    Example
    -------
    >>> set_class_to_binary([0, 4, 7])
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    """
    binary = [0] * 12
    for pc in set_class:
        binary[pc] = 1
    return binary


def calculate_radviz_position(binary_pcset):
    """
    Calculate the RadViz-style FQS position for a binary pitch class set.
    
    The position is computed as a weighted average of hexagon vertex positions,
    where weights are the DFT coefficient magnitudes (f1-f6), reordered according
    to the COEFFICIENT_ORDER configuration.
    
    Parameters
    ----------
    binary_pcset : list of int
        12-element binary representation of the pitch class set.
    
    Returns
    -------
    list of float
        [x, y] coordinates in the Fourier Quotient Space.
    
    Notes
    -----
    The DFT is computed using numpy's FFT. Coefficients 1-6 are extracted
    (excluding the DC component at index 0 and the redundant higher coefficients).
    """
    # Compute the Discrete Fourier Transform
    dft_result = np.fft.fft(binary_pcset)
    
    # Extract magnitudes of coefficients 1-6 (indices 1 to 6 inclusive)
    magnitudes = np.abs(dft_result[1:7])
    
    # Reorder magnitudes according to COEFFICIENT_ORDER
    # This maps coefficient number to hexagon vertex position
    ordered_magnitudes = [magnitudes[i - 1] for i in COEFFICIENT_ORDER]
    
    # Compute weighted average position
    total_magnitude = sum(ordered_magnitudes)
    
    # Handle edge case where all magnitudes are zero
    if total_magnitude == 0:
        return [0, 0]
    
    # RadViz formula: position is weighted centroid of anchor points
    x = sum(ANCHORS[i][0] * ordered_magnitudes[i] 
            for i in range(NUM_COEFFICIENTS)) / total_magnitude
    y = sum(ANCHORS[i][1] * ordered_magnitudes[i] 
            for i in range(NUM_COEFFICIENTS)) / total_magnitude
    
    return [x, y]


def preprocess_all_set_classes(set_class_list):
    """
    Compute FQS positions for all set classes.
    
    Parameters
    ----------
    set_class_list : list of list
        List of all set classes in normal form.
    
    Returns
    -------
    positions : list of list
        [x, y] FQS positions for each set class.
    binary_representations : list of list
        Binary representations for each set class.
    set_classes : list of list
        The original set class list (returned for convenience).
    """
    positions = []
    binary_representations = []
    
    for set_class in set_class_list:
        binary = set_class_to_binary(set_class)
        position = calculate_radviz_position(binary)
        
        positions.append(position)
        binary_representations.append(binary)
    
    return positions, binary_representations, set_class_list


# =============================================================================
# Analysis Functions
# =============================================================================

def find_duplicate_positions(set_positions):
    """
    Identify positions in the FQS that contain multiple set classes.
    
    This reveals Z-related pairs and other set classes that share identical
    Fourier magnitude profiles.
    
    Parameters
    ----------
    set_positions : list of list
        FQS positions for all set classes.
    
    Returns
    -------
    dict
        Dictionary mapping position tuples to lists of set class indices
        (only for positions with multiple set classes).
    """
    position_counts = {}
    
    for i, pos in enumerate(set_positions):
        # Round to avoid floating-point comparison issues
        pos_tuple = (round(pos[0], 10), round(pos[1], 10))
        
        if pos_tuple in position_counts:
            position_counts[pos_tuple].append(i)
        else:
            position_counts[pos_tuple] = [i]
    
    # Filter for positions with multiple set classes
    duplicates = {pos: indices for pos, indices in position_counts.items() 
                  if len(indices) > 1}
    
    # Print statistics
    total_unique_positions = len(position_counts)
    total_positions_with_duplicates = len(duplicates)
    max_duplicates = max([len(indices) for indices in duplicates.values()]) if duplicates else 0
    
    print(f"Total unique positions: {total_unique_positions}")
    print(f"Positions with multiple set classes: {total_positions_with_duplicates}")
    print(f"Maximum number of set classes at a single position: {max_duplicates}")
    
    return duplicates


def find_set_classes_in_ambiguity_zone(set_positions, set_class_list, radius):
    """
    Identify set classes positioned within the central ambiguity zone.
    
    Set classes in this zone have relatively balanced Fourier coefficient
    magnitudes, making them less distinctive in Fourier terms.
    
    Parameters
    ----------
    set_positions : list of list
        FQS positions for all set classes.
    set_class_list : list of list
        The set classes in normal form.
    radius : float
        The radius of the ambiguity zone circle.
    
    Returns
    -------
    list of list
        Set classes within the ambiguity zone.
    """
    inside_circle = []
    
    for i, pos in enumerate(set_positions):
        distance_from_centre = np.sqrt(pos[0]**2 + pos[1]**2)
        if distance_from_centre < radius:
            inside_circle.append(set_class_list[i])
    
    print(f"Set classes inside the ambiguity zone (r = {radius}):")
    for sc in inside_circle:
        print(sc)
    
    return inside_circle


def find_set_classes_at_position(position, set_positions, tolerance=1e-10):
    """
    Find all set class indices at a given position (within tolerance).
    
    Parameters
    ----------
    position : list of float
        The [x, y] position to query.
    set_positions : list of list
        FQS positions for all set classes.
    tolerance : float, optional
        Maximum Euclidean distance to consider positions equal.
    
    Returns
    -------
    list of int
        Indices of set classes at this position.
    """
    same_position_indices = []
    
    for i, pos in enumerate(set_positions):
        distance = np.sqrt((pos[0] - position[0])**2 + (pos[1] - position[1])**2)
        if distance < tolerance:
            same_position_indices.append(i)
    
    return same_position_indices


# =============================================================================
# Visualisation Helper Functions
# =============================================================================

def create_circle_of_fifths_visualisation(ax, set_class, position_y, 
                                           highlight_color='red', 
                                           second_set=None, 
                                           second_color='blue'):
    """
    Create a Circle of Fifths diagram highlighting a set class.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    set_class : list of int
        The pitch class set to highlight.
    position_y : float
        Vertical position in axes coordinates for placement.
    highlight_color : str, optional
        Colour for the primary set class (default: 'red').
    second_set : list of int, optional
        A second set class to highlight (e.g., for comparison).
    second_color : str, optional
        Colour for the second set class (default: 'blue').
    
    Returns
    -------
    list
        List of matplotlib artists created (for later removal).
    """
    # Circle of fifths ordering: C, G, D, A, E, B, F#, Db, Ab, Eb, Bb, F
    # In pitch class integers: 0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5
    CIRCLE_OF_FIFTHS_ORDER = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    
    # Note names for labelling
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Circle positioning (to the right of the bar representation)
    centre_x = 0.42
    centre_y = position_y - 0.06
    radius = 0.06
    
    # Draw the main circle outline
    circle = plt.Circle((centre_x, centre_y), radius, fill=False, 
                        transform=ax.transAxes, color='black', linewidth=0.75)
    ax.add_patch(circle)
    
    # Track all created artists for potential removal
    artists = [circle]
    
    # Draw each pitch class position on the circle
    for i, pc in enumerate(CIRCLE_OF_FIFTHS_ORDER):
        # Calculate position on the circle
        angle = 2 * pi * i / 12
        x = centre_x + radius * cos(angle)
        y = centre_y + radius * sin(angle)
        
        # Determine colour based on set membership
        in_first_set = pc in set_class
        in_second_set = second_set is not None and pc in second_set
        
        if in_first_set and in_second_set:
            point_colour = 'purple'  # Overlap between both sets
        elif in_first_set:
            point_colour = highlight_color
        elif in_second_set:
            point_colour = second_color
        else:
            point_colour = 'lightgray'
        
        # Draw the pitch class point
        point = plt.Circle((x, y), 0.01, fill=True, 
                          transform=ax.transAxes, color=point_colour)
        ax.add_patch(point)
        artists.append(point)
        
        # Add note name label outside the circle
        label_distance = 1.2  # Relative to radius
        label_x = centre_x + label_distance * radius * cos(angle)
        label_y = centre_y + label_distance * radius * sin(angle)
        
        label_text = ax.text(label_x, label_y, NOTE_NAMES[pc], 
                            transform=ax.transAxes, fontsize=6, 
                            ha='center', va='center')
        artists.append(label_text)
    
    # Add diagram title
    title_text = ax.text(centre_x, centre_y + radius + 0.02, "Circle of Fifths",
                        transform=ax.transAxes, fontsize=8, ha='center')
    artists.append(title_text)
    
    # Add legend if comparing two sets
    if second_set is not None:
        legend_y = centre_y - radius - 0.02
        legend1 = ax.text(centre_x - radius, legend_y, "Original", 
                         color=highlight_color, transform=ax.transAxes, 
                         fontsize=6, ha='left')
        legend2 = ax.text(centre_x + radius, legend_y, "Transposed", 
                         color=second_color, transform=ax.transAxes, 
                         fontsize=6, ha='right')
        artists.extend([legend1, legend2])
    
    return artists


# =============================================================================
# Compute Global Data
# =============================================================================

# Pre-compute hexagon geometry
ANCHORS = compute_hexagon_vertices()
LABEL_POSITIONS = compute_label_positions(ANCHORS)
EDGE_MIDPOINTS = compute_edge_midpoints(ANCHORS)

# Coefficient labels for the hexagon vertices (in display order)
ANCHOR_LABELS = ['f1', 'f5', 'f3', 'f4', 'f6', 'f2']

# Pre-compute FQS positions for all set classes
SET_POSITIONS, BINARY_REPRESENTATIONS, SET_CLASSES = preprocess_all_set_classes(SET_CLASS_LIST)


# =============================================================================
# Interactive Visualisation Class
# =============================================================================

class FQSExplorer:
    """
    Interactive explorer for the Fourier Quotient Space.
    
    This class manages the matplotlib figure and handles user interactions
    for exploring set classes in the FQS.
    
    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The main figure.
    ax_radviz : matplotlib.axes.Axes
        The FQS plot axes.
    ax_info : matplotlib.axes.Axes
        The information panel axes.
    kdtree : scipy.spatial.KDTree
        KD-tree for efficient nearest-neighbour queries.
    """
    
    def __init__(self):
        """
        Initialise the FQS explorer and display the interactive figure.
        """
        # Create figure with two subplots: FQS plot and information panel
        self.fig, (self.ax_radviz, self.ax_info) = plt.subplots(1, 2, figsize=(15, 8))
        plt.subplots_adjust(bottom=0.15)  # Make room for status text
        
        # Build KD-tree for efficient nearest-neighbour lookups
        self.kdtree = KDTree(SET_POSITIONS)
        
        # Storage for dynamically created UI elements
        self.set_class_texts = []
        self.cardinality_texts = []
        self.binary_texts = []
        self.binary_rects_list = []
        
        # Setup the visualisation
        self._setup_radviz_plot()
        self._setup_info_panel()
        self._plot_all_set_classes()
        self._connect_event_handlers()
        
        # Run analysis functions
        print("\n" + "="*60)
        print("FOURIER QUOTIENT SPACE ANALYSIS")
        print("="*60 + "\n")
        
        find_set_classes_in_ambiguity_zone(SET_POSITIONS, SET_CLASS_LIST, 
                                           AMBIGUITY_ZONE_RADIUS)
        print()
        find_duplicate_positions(SET_POSITIONS)
        
        # Show initial selection
        if len(SET_POSITIONS) > 0:
            self._update_info_panel(0)
        
        # Display help text
        self.fig.text(0.5, 0.01,
                     "Click on any blue point to view all set classes at that position. "
                     "Some positions may contain multiple set classes.",
                     fontsize=9, ha='center')
        
        plt.show()
    
    def _setup_radviz_plot(self):
        """
        Configure the RadViz-style FQS plot with hexagon and reference circles.
        """
        # Draw the outer unit circle
        outer_circle = plt.Circle((0, 0), 1, fill=False, linewidth=0.5)
        self.ax_radviz.add_artist(outer_circle)
        
        # Draw the inner "ambiguity zone" circle
        inner_circle = plt.Circle((0, 0), AMBIGUITY_ZONE_RADIUS, 
                                   fill=False, linewidth=0.5, linestyle='--')
        self.ax_radviz.add_artist(inner_circle)
        
        # Plot hexagon vertices (anchor points)
        self.ax_radviz.scatter(*zip(*ANCHORS), color='black', s=5)
        
        # Add coefficient labels at each vertex
        for i, (x, y) in enumerate(LABEL_POSITIONS):
            self.ax_radviz.text(x, y, ANCHOR_LABELS[i], fontsize=10)
        
        # Draw hexagon edges
        for i in range(NUM_COEFFICIENTS):
            next_i = (i + 1) % NUM_COEFFICIENTS
            self.ax_radviz.plot(
                [ANCHORS[i][0], ANCHORS[next_i][0]],
                [ANCHORS[i][1], ANCHORS[next_i][1]],
                color='black', linewidth=0.75
            )
        
        # Draw dashed radial lines from centre to edge midpoints
        for x, y in EDGE_MIDPOINTS:
            self.ax_radviz.plot([0, x], [0, y], 
                               color='black', linestyle='--', 
                               linewidth=0.5, alpha=0.5)
        
        # Configure axes
        self.ax_radviz.set_xlim(-1.125, 1.125)
        self.ax_radviz.set_ylim(-1.125, 1.125)
        self.ax_radviz.set_xlabel("X-axis")
        self.ax_radviz.set_ylabel("Y-axis")
        self.ax_radviz.set_aspect('equal')
        self.ax_radviz.set_title("Radviz Space - Set Class Explorer")
    
    def _setup_info_panel(self):
        """
        Configure the information panel for displaying set class details.
        """
        self.ax_info.axis('off')
        
        # Position and count text displays
        self.position_text = self.ax_info.text(
            0.05, 0.95, "Position: ",
            transform=self.ax_info.transAxes, fontsize=10, verticalalignment='top'
        )
        self.count_text = self.ax_info.text(
            0.05, 0.9, "Number of set classes at this position: ",
            transform=self.ax_info.transAxes, fontsize=10, verticalalignment='top'
        )
        
        # Related set classes display
        self.nn_text = self.ax_info.text(
            0.05, 0.3, "Related Set Classes:",
            transform=self.ax_info.transAxes, fontsize=10, verticalalignment='top'
        )
        
        # Create text entries for nearest neighbours (top 5)
        self.nn_entries = []
        for i in range(5):
            entry = self.ax_info.text(
                0.05, 0.25 - i * 0.05, "",
                transform=self.ax_info.transAxes, fontsize=8, verticalalignment='top'
            )
            self.nn_entries.append(entry)
        
        # Selection point marker (initially invisible)
        self.point_scatter = self.ax_radviz.scatter(
            [0], [0], color='red', s=100, alpha=0
        )
        
        # Status text at bottom
        self.status_text = self.fig.text(
            0.5, 0.05, "Click on any blue point to select a set class",
            fontsize=12, ha='center'
        )
    
    def _plot_all_set_classes(self):
        """
        Plot all set class positions as clickable points.
        """
        self.all_set_points = self.ax_radviz.scatter(
            *zip(*SET_POSITIONS),
            color='blue',
            s=10,        # Point size
            alpha=0.3,   # Transparency
            picker=5     # Enable click detection with 5-pixel tolerance
        )
    
    def _connect_event_handlers(self):
        """
        Connect matplotlib event handlers for interactive features.
        """
        self.fig.canvas.mpl_connect('pick_event', self._on_pick)
    
    def _on_pick(self, event):
        """
        Handle click events on set class points.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.PickEvent
            The pick event containing information about the clicked point.
        """
        if event.artist == self.all_set_points:
            idx = event.ind[0]  # Index of the clicked point
            self._update_info_panel(idx)
            self.fig.canvas.draw_idle()
    
    def _clear_previous_visualisations(self):
        """
        Remove previously created dynamic UI elements.
        """
        # Remove text elements
        for text_list in [self.set_class_texts, self.cardinality_texts, 
                          self.binary_texts]:
            for text in text_list:
                text.remove()
        
        # Remove rectangle patches
        for rect_list in self.binary_rects_list:
            for rect in rect_list:
                rect.remove()
        
        # Reset storage lists
        self.set_class_texts = []
        self.cardinality_texts = []
        self.binary_texts = []
        self.binary_rects_list = []
    
    def _update_info_panel(self, idx):
        """
        Update the information panel for a selected set class.
        
        Parameters
        ----------
        idx : int
            Index of the selected set class.
        """
        position = SET_POSITIONS[idx]
        
        # Find all set classes at this position
        same_position_indices = find_set_classes_at_position(
            position, SET_POSITIONS
        )
        
        # Clear previous visualisations
        self._clear_previous_visualisations()
        
        # Update position information
        self.position_text.set_text(
            f"Position: ({position[0]:.3f}, {position[1]:.3f})"
        )
        self.count_text.set_text(
            f"Number of set classes at this position: {len(same_position_indices)}"
        )
        
        # Calculate layout for multiple set classes
        available_height = 0.5  # From 0.85 down to 0.35
        height_per_sc = min(0.15, available_height / len(same_position_indices))
        
        # Create visualisations for each set class at this position
        for i, sc_idx in enumerate(same_position_indices):
            set_class = SET_CLASSES[sc_idx]
            binary = BINARY_REPRESENTATIONS[sc_idx]
            
            # Calculate vertical position for this set class
            base_y = 0.85 - i * height_per_sc
            
            # Add set class text
            sc_text = self.ax_info.text(
                0.05, base_y, f"Set Class {i+1}: {set_class}",
                transform=self.ax_info.transAxes, fontsize=8, 
                verticalalignment='top'
            )
            self.set_class_texts.append(sc_text)
            
            # Add cardinality text
            card_text = self.ax_info.text(
                0.05, base_y - 0.03, f"Cardinality: {sum(binary)}",
                transform=self.ax_info.transAxes, fontsize=8, 
                verticalalignment='top'
            )
            self.cardinality_texts.append(card_text)
            
            # Create binary representation visualisation (bar chart)
            binary_rects = []
            for j in range(12):
                colour = 'blue' if binary[j] == 1 else 'gray'
                rect = plt.Rectangle(
                    (0.05 + j * 0.07, base_y - 0.09),  # Position
                    0.06,    # Width
                    0.025,   # Height
                    fc=colour
                )
                self.ax_info.add_patch(rect)
                binary_rects.append(rect)
            
            # Add pitch class labels (only for first set class to avoid clutter)
            if i == 0:
                for j in range(12):
                    self.ax_info.text(
                        0.05 + j * 0.07 + 0.03, base_y - 0.12,
                        str(j), ha='center', fontsize=8
                    )
            
            self.binary_rects_list.append(binary_rects)
        
        # Find and display related set classes (excluding current position)
        all_indices = set(range(len(SET_POSITIONS)))
        available_indices = list(all_indices - set(same_position_indices))
        
        if available_indices:
            # Build KD-tree for remaining positions
            other_positions = [SET_POSITIONS[i] for i in available_indices]
            kdtree_others = KDTree(other_positions)
            
            # Find 5 nearest neighbours
            k = min(5, len(other_positions))
            distances, indices = kdtree_others.query(position, k=k)
            
            # Convert back to original indices
            original_indices = [available_indices[i] for i in indices]
            
            # Update related set classes list
            for i, (entry, rel_idx, dist) in enumerate(
                zip(self.nn_entries, original_indices, distances)
            ):
                if i < len(self.nn_entries):
                    sc = SET_CLASSES[rel_idx]
                    entry.set_text(f"{i+1}. {sc} (distance: {dist:.4f})")
                else:
                    entry.set_text("")
        else:
            # No other positions available
            for entry in self.nn_entries:
                entry.set_text("")
        
        # Update the selection marker
        self.point_scatter.set_offsets([position])
        self.point_scatter.set_alpha(1)  # Make visible


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    explorer = FQSExplorer()
