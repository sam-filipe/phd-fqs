"""
DFT-RadViz Utilities Module
===========================

This module provides shared functions for analysing pitch-class set data using
Discrete Fourier Transform (DFT) magnitudes visualised via RadViz projection.

These utilities support the research methodology described in the PhD thesis,
where DFT coefficient magnitudes (indices 1-6 from a 12-point DFT) are used
as features for visualising pitch-class sets in a 2D RadViz space.
"""

from math import pi, sin, cos, sqrt
from itertools import permutations
from typing import List, Tuple

from scipy.fftpack import fft
from numpy import absolute


# =============================================================================
# PITCH-CLASS SET CLASS DATA
# =============================================================================

# Complete list of pitch-class set classes (in prime form) used for analysis.
# Excludes: set class (0) representing unison, and set class (0123456789TE)
# representing the aggregate, as both produce zero-magnitude DFT vectors [0,0,0,0,0,0].
SET_CLASS_LIST: List[List[int]] = [
    # Dyads (2-note sets) and their complements (10-note sets)
    [0, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 2], [0, 1, 2, 3, 4, 5, 6, 7, 8, 10],
    [0, 3], [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
    [0, 4], [0, 1, 2, 3, 4, 5, 6, 8, 9, 10],
    [0, 5], [0, 1, 2, 3, 4, 5, 7, 8, 9, 10],
    [0, 6], [0, 1, 2, 3, 4, 6, 7, 8, 9, 10],
    # Trichords (3-note sets) and their complements (9-note sets)
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
    # Tetrachords (4-note sets) and their complements (8-note sets)
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
    # Pentachords (5-note sets) and their complements (7-note sets)
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
    # Hexachords (6-note sets) - self-complementary or paired
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
]


# =============================================================================
# DFT COMPUTATION FUNCTIONS
# =============================================================================

def setclass_to_pcset_vector(setclass: List[int]) -> List[int]:
    """
    Convert a pitch-class set class to a 12-element binary vector.
    
    The vector has 1s at positions corresponding to pitch classes present
    in the set, and 0s elsewhere.
    
    Args:
        setclass: A list of pitch-class integers (0-11) in prime form.
    
    Returns:
        A 12-element binary list representing the pitch-class set.
    
    Example:
        >>> setclass_to_pcset_vector([0, 1])
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """
    pcset_vector = [0] * 12
    for pitch_class in setclass:
        pcset_vector[pitch_class] = 1
    return pcset_vector


def compute_dft_magnitudes(pcset_vectors: List[List[int]]) -> List[List[float]]:
    """
    Compute DFT magnitudes (coefficients 1-6) for a list of pcset vectors.
    
    Applies the Discrete Fourier Transform to each pcset vector and extracts
    the magnitudes of coefficients 1 through 6 (indices 1-6 of the 12-point DFT).
    Coefficient 0 is excluded as it represents the cardinality (sum) of the set.
    Coefficients 7-11 are redundant due to conjugate symmetry.
    
    Args:
        pcset_vectors: List of 12-element binary vectors representing pcsets.
    
    Returns:
        List of 6-element lists containing the DFT magnitudes for each pcset.
    """
    # Compute FFT for all pcset vectors at once
    fft_result = fft(pcset_vectors)
    
    # Extract magnitudes (absolute values) of the complex FFT output
    all_magnitudes = absolute(fft_result)
    
    # Keep only coefficients 1-6 (indices 1 to 6 inclusive)
    magnitudes_1_to_6 = []
    for magnitude_row in all_magnitudes:
        magnitudes_1_to_6.append([magnitude_row[i] for i in range(1, 7)])
    
    return magnitudes_1_to_6


def compute_all_dft_magnitudes() -> List[List[float]]:
    """
    Compute DFT magnitudes for all pitch-class set classes in SET_CLASS_LIST.
    
    Returns:
        List of 6-element magnitude vectors, one for each set class.
    """
    # Convert all set classes to binary pcset vectors
    pcset_vectors = [setclass_to_pcset_vector(sc) for sc in SET_CLASS_LIST]
    
    # Compute and return DFT magnitudes
    return compute_dft_magnitudes(pcset_vectors)


# =============================================================================
# RADVIZ PROJECTION FUNCTIONS
# =============================================================================

def compute_radviz_anchors(num_dimensions: int) -> List[Tuple[float, float]]:
    """
    Compute the (x, y) coordinates of RadViz anchor points on a unit circle.
    
    Anchor points are evenly distributed around a circle of radius 1,
    starting from angle 0 (rightmost point) and proceeding counter-clockwise.
    
    Args:
        num_dimensions: Number of anchor points (equals number of features).
    
    Returns:
        List of (x, y) coordinate tuples for each anchor point.
    """
    angle_step = 2 * pi / num_dimensions
    anchors = []
    for i in range(num_dimensions):
        x = cos(i * angle_step)
        y = sin(i * angle_step)
        anchors.append((x, y))
    return anchors


def project_to_radviz(
    magnitudes: List[List[float]],
    anchors: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """
    Project high-dimensional magnitude vectors onto 2D RadViz space.
    
    RadViz projection formula:
        For a point with feature values [v1, v2, ..., vn] and anchor 
        coordinates [(x1,y1), (x2,y2), ..., (xn,yn)]:
        
        x = sum(vi * xi) / sum(vi)
        y = sum(vi * yi) / sum(vi)
    
    Args:
        magnitudes: List of n-dimensional magnitude vectors.
        anchors: List of (x, y) anchor coordinates on the unit circle.
    
    Returns:
        List of (x, y) coordinate tuples in 2D RadViz space.
    """
    points = []
    num_dims = len(anchors)
    
    for magnitude_vector in magnitudes:
        # Compute weighted sums for x and y coordinates
        x_weighted_sum = 0.0
        y_weighted_sum = 0.0
        for j in range(num_dims):
            x_weighted_sum += anchors[j][0] * magnitude_vector[j]
            y_weighted_sum += anchors[j][1] * magnitude_vector[j]
        
        # Normalise by sum of magnitudes (total weight)
        total_weight = sum(magnitude_vector)
        x_coord = x_weighted_sum / total_weight
        y_coord = y_weighted_sum / total_weight
        
        points.append((x_coord, y_coord))
    
    return points


def reorder_magnitudes_by_permutation(
    magnitudes: List[List[float]],
    permutation: List[int]
) -> List[List[float]]:
    """
    Reorder magnitude vectors according to a coefficient permutation.
    
    The permutation specifies which original coefficient index should
    be placed at each position. Permutation values are 1-indexed
    (i.e., [1,2,3,4,5,6] means original order).
    
    Args:
        magnitudes: List of 6-element magnitude vectors in original order.
        permutation: A permutation of [1,2,3,4,5,6] specifying new order.
    
    Returns:
        List of reordered magnitude vectors.
    """
    reordered = []
    for mag_vector in magnitudes:
        # permutation[j]-1 converts 1-indexed to 0-indexed
        reordered_vector = [mag_vector[permutation[j] - 1] for j in range(6)]
        reordered.append(reordered_vector)
    return reordered


# =============================================================================
# DISTANCE COMPUTATION FUNCTIONS
# =============================================================================

def euclidean_distance_2d(
    point1: Tuple[float, float],
    point2: Tuple[float, float]
) -> float:
    """
    Compute Euclidean distance between two 2D points.
    
    Args:
        point1: First point as (x, y) tuple.
        point2: Second point as (x, y) tuple.
    
    Returns:
        The Euclidean distance between the two points.
    """
    return sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def euclidean_distance_6d(vector1: List[float], vector2: List[float]) -> float:
    """
    Compute Euclidean distance between two 6D vectors.
    
    Args:
        vector1: First 6-element vector.
        vector2: Second 6-element vector.
    
    Returns:
        The Euclidean distance between the two vectors in 6D space.
    """
    return sqrt(
        (vector2[0] - vector1[0])**2 +
        (vector2[1] - vector1[1])**2 +
        (vector2[2] - vector1[2])**2 +
        (vector2[3] - vector1[3])**2 +
        (vector2[4] - vector1[4])**2 +
        (vector2[5] - vector1[5])**2
    )


# =============================================================================
# PERMUTATION GENERATION
# =============================================================================

def generate_all_coefficient_permutations() -> List[List[int]]:
    """
    Generate all 720 permutations of the coefficient indices [1,2,3,4,5,6].
    
    These represent all possible orderings of the 6 DFT coefficients
    around the RadViz circle.
    
    Returns:
        List of 720 permutations, each as a list of 6 integers.
    """
    base_coefficients = [1, 2, 3, 4, 5, 6]
    all_perms = list(permutations(base_coefficients, 6))
    # Convert tuples to lists for consistency
    return [list(p) for p in all_perms]
