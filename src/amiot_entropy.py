"""
Amiot's Entropy of Fourier Coefficients for Pitch-Class Sets
=============================================================

This module computes the spectral entropy of pitch-class sets (pcsets) using
the Discrete Fourier Transform (DFT), as formalised by Emmanuel Amiot in his
work on music theory and mathematical analysis of scales and chords.

Mathematical Background
-----------------------
Given a pitch-class set represented as a characteristic function (a 12-element
binary vector where 1 indicates presence of a pitch class), the DFT decomposes
it into Fourier coefficients. The magnitudes of these coefficients reveal the
intervallic structure of the set.

The entropy H is computed from the normalised squared magnitudes of the Fourier
coefficients (excluding the 0th coefficient, which merely counts cardinality):

    H = -Σ p_k * log(p_k)

where p_k = |a_k|² / (d * (n - d)), with:
    - a_k: the k-th Fourier coefficient
    - d: cardinality of the pcset (number of pitch classes present)
    - n: chromatic universe size (12 for standard equal temperament)

Higher entropy indicates a more "spread out" or balanced distribution of
intervallic content, whilst lower entropy indicates concentration of energy
in fewer Fourier components.

References
----------
Amiot, E. (2016). Music Through Fourier Space: Discrete Fourier Transform in
Music Theory. Springer.

Author: [Thesis Author]
Date: [Date]
"""

import numpy as np


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

CHROMATIC_CARDINALITY = 12  # Number of pitch classes in standard chromatic scale


# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------

def pcset_entropy_from_pitches(pitch_classes):
    """
    Compute Amiot's entropy from a list of pitch-class integers.

    This function takes pitch classes as integers (0-11) and computes the
    spectral entropy of the resulting set.

    Parameters
    ----------
    pitch_classes : array-like of int
        List of pitch-class integers in the range [0, 11].
        Example: [0, 2, 4, 5, 7, 9, 11] for the C major scale.

    Returns
    -------
    float
        The spectral entropy H of the pitch-class set.

    Examples
    --------
    >>> major_scale = [0, 2, 4, 5, 7, 9, 11]
    >>> entropy = pcset_entropy_from_pitches(major_scale)
    """
    n = CHROMATIC_CARDINALITY
    d = len(pitch_classes)  # Cardinality of the pcset

    # Compute Fourier coefficients manually using the DFT formula:
    # a_k = Σ exp(-2πi * k * p / n) for each pitch class p in the set
    fourier_coefficients = np.array([
        np.sum(np.exp(-2j * np.pi * k * np.array(pitch_classes) / n))
        for k in range(n)
    ])

    # Normalised squared magnitudes (excluding k=0, which is just the cardinality)
    # The normalisation factor d * (n - d) ensures values sum to n - 1
    normalised_magnitudes = np.abs(fourier_coefficients[1:]) ** 2 / (d * (n - d))

    # Compute entropy using the standard Shannon formula
    # Note: (pk == 0) adds 1 where pk is zero, preventing log(0) = -inf
    # This is equivalent to defining 0 * log(0) = 0 by convention
    entropy = -np.sum(
        normalised_magnitudes * np.log(normalised_magnitudes + (normalised_magnitudes == 0))
    )

    return entropy


def pcset_entropy_from_characteristic(characteristic_vector):
    """
    Compute Amiot's entropy from a characteristic function representation.

    This function takes a 12-element binary vector (characteristic function)
    and computes the spectral entropy using NumPy's FFT implementation.

    Parameters
    ----------
    characteristic_vector : array-like of int
        A 12-element binary vector where 1 indicates presence of a pitch class.
        Index 0 corresponds to pitch class 0 (C), index 1 to pitch class 1 (C#), etc.
        Example: [1,0,1,0,1,1,0,1,0,1,0,1] for the C major scale.

    Returns
    -------
    float
        The spectral entropy H of the pitch-class set.

    Examples
    --------
    >>> c_major = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]  # C major scale
    >>> entropy = pcset_entropy_from_characteristic(c_major)
    """
    n = CHROMATIC_CARDINALITY

    # Cardinality: count of non-zero elements (pitch classes present)
    d = sum(1 for x in characteristic_vector if x != 0)

    # Compute DFT using NumPy's efficient FFT algorithm
    dft_coefficients = np.fft.fft(characteristic_vector)

    # Normalised squared magnitudes (excluding the 0th coefficient)
    normalised_magnitudes = np.abs(dft_coefficients[1:]) ** 2 / (d * (n - d))

    # Compute entropy (with protection against log(0))
    entropy = -np.sum(
        normalised_magnitudes * np.log(normalised_magnitudes + (normalised_magnitudes == 0))
    )

    return entropy


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Example: A 10-note subset of the chromatic scale
    # Characteristic vector: pitch classes 0-9 are present, 10-11 are absent
    example_pcset = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]

    # Compute DFT for demonstration
    dft_result = np.fft.fft(example_pcset)
    print("DFT magnitudes:")
    print(np.abs(dft_result))

    # Compute normalised magnitudes
    cardinality = sum(1 for x in example_pcset if x != 0)
    normalised_mags = np.abs(dft_result[1:]) ** 2 / (cardinality * (12 - cardinality))
    print("\nNormalised squared magnitudes (excluding k=0):")
    print(normalised_mags)

    # Compute and display entropy
    entropy = pcset_entropy_from_characteristic(example_pcset)
    print(f"\nThe entropy of the pcset {example_pcset} is {entropy:.4f}")
