"""
RadViz Distance Correlation Coefficient Order Finder
====================================================

This script finds the optimal ordering of DFT (Discrete Fourier Transform)
coefficients for RadViz visualisation that best preserves the distance
relationships from the original 6-dimensional space.

Purpose:
    When projecting 6-dimensional DFT magnitude vectors to 2D RadViz space,
    some distance information is inevitably lost. However, the arrangement
    of anchor points affects how well the 2D representation preserves the
    original distance relationships. This script exhaustively tests all 720
    permutations to find the coefficient ordering that maximises the Pearson
    correlation between pairwise distances in 6D and 2D spaces.

Methodology:
    1. Convert all set classes to binary 12-element vectors
    2. Compute DFT magnitudes (coefficients 1-6) for each set class
    3. Compute all pairwise Euclidean distances in the original 6D space
    4. For each of the 720 coefficient permutations:
       a. Reorder magnitude vectors according to the permutation
       b. Project all points to 2D RadViz space
       c. Compute all pairwise Euclidean distances in 2D
       d. Calculate Pearson correlation between 6D and 2D distance matrices
    5. Identify the permutation yielding the highest correlation

Output:
    - The maximum Pearson correlation coefficient achieved
    - All coefficient orderings that achieve high correlation (≥ 0.729)

Note on Correlation Threshold:
    The threshold of 0.729 is used to identify permutations with notably
    high correlation values, allowing identification of multiple good
    orderings if they exist.

Dependencies:
    - scipy.fftpack: For computing the Discrete Fourier Transform
    - numpy: For computing magnitudes and Pearson correlation coefficients
    - dft_radviz_utils: Shared utility functions for this analysis
"""

from numpy import corrcoef

from dft_radviz_utils import (
    compute_all_dft_magnitudes,
    compute_radviz_anchors,
    project_to_radviz,
    reorder_magnitudes_by_permutation,
    euclidean_distance_2d,
    euclidean_distance_6d,
    generate_all_coefficient_permutations,
)


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Number of DFT coefficients used (coefficients 1 through 6)
NUM_COEFFICIENTS = 6

# Threshold for reporting high-correlation permutations
CORRELATION_THRESHOLD = 0.729


# =============================================================================
# DISTANCE MATRIX COMPUTATION FUNCTIONS
# =============================================================================

def compute_pairwise_distances_6d(magnitude_vectors: list) -> list:
    """
    Compute all pairwise Euclidean distances in 6D magnitude space.
    
    This creates a flattened distance matrix containing n² distances
    for n points, including self-distances (which are 0).
    
    Args:
        magnitude_vectors: List of 6-element DFT magnitude vectors.
    
    Returns:
        Flattened list of all pairwise distances (row-major order).
    """
    num_points = len(magnitude_vectors)
    distances = []
    
    for i in range(num_points):
        for k in range(num_points):
            dist = euclidean_distance_6d(
                magnitude_vectors[i],
                magnitude_vectors[k]
            )
            distances.append(dist)
    
    return distances


def compute_pairwise_distances_2d(radviz_points: list) -> list:
    """
    Compute all pairwise Euclidean distances in 2D RadViz space.
    
    Args:
        radviz_points: List of (x, y) coordinates in RadViz space.
    
    Returns:
        Flattened list of all pairwise distances (row-major order).
    """
    num_points = len(radviz_points)
    distances = []
    
    for i in range(num_points):
        for k in range(num_points):
            dist = euclidean_distance_2d(
                radviz_points[i],
                radviz_points[k]
            )
            distances.append(dist)
    
    return distances


# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def find_best_correlation_coefficient_order():
    """
    Find the coefficient ordering that best preserves 6D distances in 2D.
    
    This function evaluates all 720 permutations of the 6 DFT coefficients
    to determine which arrangement around the RadViz circle produces the
    highest Pearson correlation between original 6D pairwise distances and
    the resulting 2D RadViz pairwise distances.
    
    Returns:
        A tuple containing:
        - max_correlation: The highest Pearson correlation achieved
        - all_permutations: List of all 720 permutations tested
        - pearson_coefficients: List of correlation values for each permutation
    """
    # Step 1: Compute DFT magnitudes for all set classes (in original order)
    magnitudes_original_order = compute_all_dft_magnitudes()
    
    # Step 2: Compute 6D pairwise distances (constant across all permutations)
    # Note: This is computed once since it depends only on magnitude values,
    # not their ordering around the RadViz circle.
    distances_6d = compute_pairwise_distances_6d(magnitudes_original_order)
    
    # Step 3: Compute RadViz anchor positions on the unit circle
    anchors = compute_radviz_anchors(NUM_COEFFICIENTS)
    
    # Step 4: Generate all 720 permutations of coefficient indices
    all_permutations = generate_all_coefficient_permutations()
    
    # Step 5: Evaluate each permutation
    pearson_coefficients = []
    
    for permutation in all_permutations:
        # Reorder magnitudes according to current permutation
        reordered_magnitudes = reorder_magnitudes_by_permutation(
            magnitudes_original_order, permutation
        )
        
        # Project to 2D RadViz space
        radviz_points = project_to_radviz(reordered_magnitudes, anchors)
        
        # Compute 2D pairwise distances
        distances_2d = compute_pairwise_distances_2d(radviz_points)
        
        # Calculate Pearson correlation between 6D and 2D distance matrices
        correlation_matrix = corrcoef(distances_6d, distances_2d)
        pearson_r = correlation_matrix[0][1]
        pearson_coefficients.append(pearson_r)
    
    # Step 6: Find maximum correlation
    max_correlation = max(pearson_coefficients)
    
    return max_correlation, all_permutations, pearson_coefficients


def print_results(
    max_correlation: float,
    all_permutations: list,
    pearson_coefficients: list,
    threshold: float = CORRELATION_THRESHOLD
):
    """
    Display the analysis results in a formatted manner.
    
    Args:
        max_correlation: The highest Pearson correlation achieved.
        all_permutations: List of all 720 permutations tested.
        pearson_coefficients: List of correlation values for each permutation.
        threshold: Minimum correlation to include in detailed output.
    """
    print("")
    print("=" * 65)
    print("RADVIZ COEFFICIENT ORDER: DISTANCE CORRELATION ANALYSIS")
    print("=" * 65)
    print("")
    print("Objective: Find DFT coefficient arrangement that maximises the")
    print("           Pearson correlation between 6D and 2D pairwise distances.")
    print("")
    print("-" * 65)
    print("RESULTS")
    print("-" * 65)
    print("")
    print(f"Maximum Pearson correlation coefficient: {max_correlation:.6f}")
    print("")
    print(f"Permutations with correlation ≥ {threshold}:")
    print("")
    
    count = 0
    for i, pearson_r in enumerate(pearson_coefficients):
        if pearson_r >= threshold:
            count += 1
            print(f"  Correlation: {pearson_r:.6f}  |  Order: {all_permutations[i]}")
    
    if count == 0:
        print(f"  (No permutations achieved correlation ≥ {threshold})")
    
    print("")
    print(f"Total permutations meeting threshold: {count}")
    print("")
    print("=" * 65)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run the analysis
    max_corr, all_perms, pearson_values = find_best_correlation_coefficient_order()
    
    # Display results
    print_results(max_corr, all_perms, pearson_values)
