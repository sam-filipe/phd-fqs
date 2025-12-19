"""
RadViz Maximum Dispersion Coefficient Order Finder
==================================================

This script finds the optimal ordering of DFT (Discrete Fourier Transform)
coefficients for RadViz visualisation that maximises the dispersion of
pitch-class set points from the origin.

Purpose:
    In RadViz visualisation, the arrangement of anchor points (corresponding
    to DFT coefficients 1-6) around the circle significantly affects the
    spatial distribution of projected points. This script exhaustively tests
    all 720 possible permutations of the 6 coefficients to find the ordering
    that maximises the total sum of Euclidean distances from the origin.

Methodology:
    1. Convert all set classes to binary 12-element vectors
    2. Compute DFT magnitudes (coefficients 1-6) for each set class
    3. For each of the 720 coefficient permutations:
       a. Reorder magnitude vectors according to the permutation
       b. Project all points to 2D RadViz space
       c. Calculate Euclidean distance from origin for each point
       d. Sum all distances as the dispersion metric
    4. Identify the permutation(s) yielding maximum dispersion

Output:
    - The coefficient ordering that produces maximum point dispersion
    - The maximum sum of Euclidean distances achieved
    - All permutations that achieve the maximum (if ties exist)

Dependencies:
    - scipy.fftpack: For computing the Discrete Fourier Transform
    - numpy: For computing absolute values (magnitudes) of complex numbers
    - dft_radviz_utils: Shared utility functions for this analysis
"""

from dft_radviz_utils import (
    compute_all_dft_magnitudes,
    compute_radviz_anchors,
    project_to_radviz,
    reorder_magnitudes_by_permutation,
    euclidean_distance_2d,
    generate_all_coefficient_permutations,
)


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Number of DFT coefficients used (coefficients 1 through 6)
NUM_COEFFICIENTS = 6

# Origin point in 2D RadViz space (centre of the circle)
ORIGIN = (0.0, 0.0)


# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def compute_dispersion_sum(
    radviz_points: list,
    origin: tuple = ORIGIN
) -> float:
    """
    Compute the total dispersion as the sum of distances from the origin.
    
    This metric quantifies how spread out the points are from the centre
    of the RadViz visualisation. Higher values indicate better dispersion.
    
    Note: To ensure exact numerical reproducibility with the original
    implementation, distances are first collected into a list before
    summing (matching the original code's floating point behavior).
    
    Args:
        radviz_points: List of (x, y) coordinates in RadViz space.
        origin: Reference point, typically (0, 0).
    
    Returns:
        Sum of Euclidean distances from all points to the origin.
    """
    # First compute all individual distances (as in original code)
    individual_distances = []
    for point in radviz_points:
        individual_distances.append(euclidean_distance_2d(point, origin))
    
    # Then sum them (matching original code's floating point accumulation)
    return sum(individual_distances)


def find_max_dispersion_coefficient_order():
    """
    Find the coefficient ordering that maximises point dispersion in RadViz.
    
    This function exhaustively evaluates all 720 permutations of the
    6 DFT coefficients to determine which arrangement around the RadViz
    circle produces the greatest total distance from the origin.
    
    Returns:
        A tuple containing:
        - best_permutation: The coefficient ordering with maximum dispersion
        - max_dispersion: The maximum sum of Euclidean distances achieved
        - all_best_permutations: List of all permutations achieving the max
    """
    # Step 1: Compute DFT magnitudes for all set classes (in original order)
    magnitudes_original_order = compute_all_dft_magnitudes()
    
    # Step 2: Compute RadViz anchor positions on the unit circle
    anchors = compute_radviz_anchors(NUM_COEFFICIENTS)
    
    # Step 3: Generate all 720 permutations of coefficient indices
    all_permutations = generate_all_coefficient_permutations()
    
    # Step 4: Evaluate each permutation
    dispersion_sums = []
    
    for permutation in all_permutations:
        # Reorder magnitudes according to current permutation
        reordered_magnitudes = reorder_magnitudes_by_permutation(
            magnitudes_original_order, permutation
        )
        
        # Project to 2D RadViz space
        radviz_points = project_to_radviz(reordered_magnitudes, anchors)
        
        # Compute dispersion metric (sum of distances from origin)
        dispersion = compute_dispersion_sum(radviz_points)
        dispersion_sums.append(dispersion)
    
    # Step 5: Find maximum dispersion and corresponding permutation(s)
    max_dispersion = max(dispersion_sums)
    max_position = dispersion_sums.index(max_dispersion)
    best_permutation = all_permutations[max_position]
    
    # Find all permutations that achieve the maximum (to check for ties)
    all_best_permutations = []
    for i, dispersion in enumerate(dispersion_sums):
        if dispersion == max_dispersion:
            all_best_permutations.append(all_permutations[i])
    
    return best_permutation, max_dispersion, all_best_permutations


def print_results(
    best_permutation: list,
    max_dispersion: float,
    all_best_permutations: list
):
    """
    Display the analysis results in a formatted manner.
    
    Args:
        best_permutation: The first coefficient ordering found with max dispersion.
        max_dispersion: The maximum sum of Euclidean distances achieved.
        all_best_permutations: All permutations achieving the maximum.
    """
    print("")
    print("=" * 60)
    print("RADVIZ COEFFICIENT ORDER: MAXIMUM DISPERSION ANALYSIS")
    print("=" * 60)
    print("")
    print("Objective: Find DFT coefficient arrangement that maximises")
    print("           the sum of Euclidean distances from the origin.")
    print("")
    print("-" * 60)
    print("RESULTS")
    print("-" * 60)
    print("")
    print(f"Maximum dispersion sum: {max_dispersion:.6f}")
    print(f"Optimal coefficient order: {best_permutation}")
    print("")
    
    if len(all_best_permutations) > 1:
        print(f"Note: {len(all_best_permutations)} permutations achieve this maximum:")
        print("")
        for perm in all_best_permutations:
            print(f"  {perm}")
    else:
        print("This ordering is unique (no ties).")
    
    print("")
    print("=" * 60)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run the analysis
    best_perm, max_disp, all_best = find_max_dispersion_coefficient_order()
    
    # Display results
    print_results(best_perm, max_disp, all_best)
