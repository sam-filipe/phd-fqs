"""
All Set Classes List and Calculate the Summed Interval Vector.

This script contains a list of all set classes and computes the sum of their interval 
vectors (excluding the empty set and the chromatic aggregate).

Notes
-----
- The unison set class (0) is included but the chromatic aggregate (0123456789TE)
  is excluded because its interval vector magnitudes are effectively [0,0,0,0,0,0]
  when considering complementary relationships.
- Set classes are represented in their prime form (most compact, starting from 0).
- Each set class and its complement are both included in this list.
"""

import numpy as np
from music21 import chord, pitch


# =============================================================================
# SET CLASS DEFINITIONS
# =============================================================================

# Complete list of prime-form set classes from cardinality 1 to 12.
# Organised as pairs: each set class followed by its complement (where applicable).
# Excludes the empty set () and the full chromatic aggregate (0123456789TE).
#
# Format: Each sublist represents pitch classes in prime form (integers 0-11,
# where 0=C, 1=C#/Db, 2=D, ..., 10=Bb, 11=B).

SET_CLASS_LIST = [
    # ---------------------------------------------------------------------
    # Cardinality 1 and 11 (complementary pairs)
    # ---------------------------------------------------------------------
    [0],                                        # 1-1: Single pitch class

    # ---------------------------------------------------------------------
    # Cardinality 2 and 10 (interval classes and their complements)
    # ---------------------------------------------------------------------
    [0, 1],                                     # 2-1: Minor second (ic1)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],              # 10-1: Complement

    [0, 2],                                     # 2-2: Whole tone (ic2)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 10],             # 10-2: Complement

    [0, 3],                                     # 2-3: Minor third (ic3)
    [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],             # 10-3: Complement

    [0, 4],                                     # 2-4: Major third (ic4)
    [0, 1, 2, 3, 4, 5, 6, 8, 9, 10],             # 10-4: Complement

    [0, 5],                                     # 2-5: Perfect fourth (ic5)
    [0, 1, 2, 3, 4, 5, 7, 8, 9, 10],             # 10-5: Complement

    [0, 6],                                     # 2-6: Tritone (ic6)
    [0, 1, 2, 3, 4, 6, 7, 8, 9, 10],             # 10-6: Complement

    # ---------------------------------------------------------------------
    # Cardinality 3 and 9 (trichords and their complements)
    # ---------------------------------------------------------------------
    [0, 1, 2],                                  # 3-1: Chromatic trichord
    [0, 1, 2, 3, 4, 5, 6, 7, 8],                 # 9-1: Complement

    [0, 1, 3],                                  # 3-2: Minor second + minor third
    [0, 1, 2, 3, 4, 5, 6, 7, 9],                 # 9-2: Complement

    [0, 1, 4],                                  # 3-3: Minor second + major third
    [0, 1, 2, 3, 4, 5, 6, 8, 9],                 # 9-3: Complement

    [0, 1, 5],                                  # 3-4: Minor second + perfect fourth
    [0, 1, 2, 3, 4, 5, 7, 8, 9],                 # 9-4: Complement

    [0, 1, 6],                                  # 3-5: Minor second + tritone
    [0, 1, 2, 3, 4, 6, 7, 8, 9],                 # 9-5: Complement

    [0, 2, 4],                                  # 3-6: Whole-tone trichord
    [0, 1, 2, 3, 4, 5, 6, 8, 10],                # 9-6: Complement

    [0, 2, 5],                                  # 3-7: Major second + perfect fourth
    [0, 1, 2, 3, 4, 5, 7, 8, 10],                # 9-7: Complement

    [0, 2, 6],                                  # 3-8: Major second + tritone
    [0, 1, 2, 3, 4, 6, 7, 8, 10],                # 9-8: Complement

    [0, 2, 7],                                  # 3-9: "Quartal" trichord (stacked 4ths)
    [0, 1, 2, 3, 5, 6, 7, 8, 10],                # 9-9: Complement

    [0, 3, 6],                                  # 3-10: Diminished triad
    [0, 1, 2, 3, 4, 6, 7, 9, 10],                # 9-10: Complement

    [0, 3, 7],                                  # 3-11: Minor/major triad
    [0, 1, 2, 3, 5, 6, 7, 9, 10],                # 9-11: Complement

    [0, 4, 8],                                  # 3-12: Augmented triad
    [0, 1, 2, 4, 5, 6, 8, 9, 10],                # 9-12: Complement

    # ---------------------------------------------------------------------
    # Cardinality 4 and 8 (tetrachords and their complements)
    # ---------------------------------------------------------------------
    [0, 1, 2, 3],                               # 4-1: Chromatic tetrachord
    [0, 1, 2, 3, 4, 5, 6, 7],                    # 8-1: Complement

    [0, 1, 2, 4],                               # 4-2
    [0, 1, 2, 3, 4, 5, 6, 8],                    # 8-2: Complement

    [0, 1, 2, 5],                               # 4-4
    [0, 1, 2, 3, 4, 5, 7, 8],                    # 8-4: Complement

    [0, 1, 2, 6],                               # 4-5
    [0, 1, 2, 3, 4, 6, 7, 8],                    # 8-5: Complement

    [0, 1, 2, 7],                               # 4-6
    [0, 1, 2, 3, 5, 6, 7, 8],                    # 8-6: Complement

    [0, 1, 3, 4],                               # 4-3
    [0, 1, 2, 3, 4, 5, 6, 9],                    # 8-3: Complement

    [0, 1, 3, 5],                               # 4-11
    [0, 1, 2, 3, 4, 5, 7, 9],                    # 8-11: Complement

    [0, 1, 3, 6],                               # 4-13
    [0, 1, 2, 3, 4, 6, 7, 9],                    # 8-13: Complement

    [0, 1, 3, 7],                               # 4-Z29
    [0, 1, 2, 3, 5, 6, 7, 9],                    # 8-Z29: Complement

    [0, 1, 4, 5],                               # 4-7
    [0, 1, 2, 3, 4, 5, 8, 9],                    # 8-7: Complement

    [0, 1, 4, 6],                               # 4-Z15
    [0, 1, 2, 3, 4, 6, 8, 9],                    # 8-Z15: Complement

    [0, 1, 4, 7],                               # 4-18
    [0, 1, 2, 3, 5, 6, 8, 9],                    # 8-18: Complement

    [0, 1, 4, 8],                               # 4-19
    [0, 1, 2, 4, 5, 6, 8, 9],                    # 8-19: Complement

    [0, 1, 5, 6],                               # 4-8
    [0, 1, 2, 3, 4, 7, 8, 9],                    # 8-8: Complement

    [0, 1, 5, 7],                               # 4-16
    [0, 1, 2, 3, 5, 7, 8, 9],                    # 8-16: Complement

    [0, 1, 5, 8],                               # 4-20
    [0, 1, 2, 4, 5, 7, 8, 9],                    # 8-20: Complement

    [0, 1, 6, 7],                               # 4-9
    [0, 1, 2, 3, 6, 7, 8, 9],                    # 8-9: Complement

    [0, 2, 3, 5],                               # 4-10
    [0, 2, 3, 4, 5, 6, 7, 9],                    # 8-10: Complement

    [0, 2, 3, 6],                               # 4-12
    [0, 1, 3, 4, 5, 6, 7, 9],                    # 8-12: Complement

    [0, 2, 3, 7],                               # 4-14
    [0, 1, 2, 4, 5, 6, 7, 9],                    # 8-14: Complement

    [0, 2, 4, 6],                               # 4-21: Whole-tone tetrachord
    [0, 1, 2, 3, 4, 6, 8, 10],                   # 8-21: Complement

    [0, 2, 4, 7],                               # 4-22
    [0, 1, 2, 3, 5, 6, 8, 10],                   # 8-22: Complement

    [0, 2, 4, 8],                               # 4-24
    [0, 1, 2, 4, 5, 6, 8, 10],                   # 8-24: Complement

    [0, 2, 5, 7],                               # 4-23
    [0, 1, 2, 3, 5, 7, 8, 10],                   # 8-23: Complement

    [0, 2, 5, 8],                               # 4-27
    [0, 1, 2, 4, 5, 7, 8, 10],                   # 8-27: Complement

    [0, 2, 6, 8],                               # 4-25: "French sixth" tetrachord
    [0, 1, 2, 4, 6, 7, 8, 10],                   # 8-25: Complement

    [0, 3, 4, 7],                               # 4-17: Major-minor tetrachord
    [0, 1, 3, 4, 5, 6, 8, 9],                    # 8-17: Complement

    [0, 3, 5, 8],                               # 4-26
    [0, 1, 3, 4, 5, 7, 8, 10],                   # 8-26: Complement

    [0, 3, 6, 9],                               # 4-28: Diminished seventh chord
    [0, 1, 3, 4, 6, 7, 9, 10],                   # 8-28: Complement

    # ---------------------------------------------------------------------
    # Cardinality 5 and 7 (pentachords and their complements)
    # ---------------------------------------------------------------------
    [0, 1, 2, 3, 4],                            # 5-1: Chromatic pentachord
    [0, 1, 2, 3, 4, 5, 6],                       # 7-1: Complement

    [0, 1, 2, 3, 5],                            # 5-2
    [0, 1, 2, 3, 4, 5, 7],                       # 7-2: Complement

    [0, 1, 2, 3, 6],                            # 5-4
    [0, 1, 2, 3, 4, 6, 7],                       # 7-4: Complement

    [0, 1, 2, 3, 7],                            # 5-5
    [0, 1, 2, 3, 5, 6, 7],                       # 7-5: Complement

    [0, 1, 2, 4, 5],                            # 5-3
    [0, 1, 2, 3, 4, 5, 8],                       # 7-3: Complement

    [0, 1, 2, 4, 6],                            # 5-9
    [0, 1, 2, 3, 4, 6, 8],                       # 7-9: Complement

    [0, 1, 2, 4, 7],                            # 5-Z36
    [0, 1, 2, 3, 5, 6, 8],                       # 7-Z36: Complement

    [0, 1, 2, 4, 8],                            # 5-13
    [0, 1, 2, 4, 5, 6, 8],                       # 7-13: Complement

    [0, 1, 2, 5, 6],                            # 5-6
    [0, 1, 2, 3, 4, 7, 8],                       # 7-6: Complement

    [0, 1, 2, 5, 7],                            # 5-14
    [0, 1, 2, 3, 5, 7, 8],                       # 7-14: Complement

    [0, 1, 2, 5, 8],                            # 5-Z38
    [0, 1, 2, 4, 5, 7, 8],                       # 7-Z38: Complement

    [0, 1, 2, 6, 7],                            # 5-7
    [0, 1, 2, 3, 6, 7, 8],                       # 7-7: Complement

    [0, 1, 2, 6, 8],                            # 5-15
    [0, 1, 2, 4, 6, 7, 8],                       # 7-15: Complement

    [0, 1, 3, 4, 6],                            # 5-10
    [0, 1, 2, 3, 4, 6, 9],                       # 7-10: Complement

    [0, 1, 3, 4, 7],                            # 5-16
    [0, 1, 2, 3, 5, 6, 9],                       # 7-16: Complement

    [0, 1, 3, 4, 8],                            # 5-Z17
    [0, 1, 2, 4, 5, 6, 9],                       # 7-Z17: Complement

    [0, 1, 3, 5, 6],                            # 5-Z12
    [0, 1, 2, 3, 4, 7, 9],                       # 7-Z12: Complement

    [0, 1, 3, 5, 7],                            # 5-24
    [0, 1, 2, 3, 5, 7, 9],                       # 7-24: Complement

    [0, 1, 3, 5, 8],                            # 5-27
    [0, 1, 2, 4, 5, 7, 9],                       # 7-27: Complement

    [0, 1, 3, 6, 7],                            # 5-19
    [0, 1, 2, 3, 6, 7, 9],                       # 7-19: Complement

    [0, 1, 3, 6, 8],                            # 5-29
    [0, 1, 2, 4, 6, 7, 9],                       # 7-29: Complement

    [0, 1, 3, 6, 9],                            # 5-31
    [0, 1, 3, 4, 6, 7, 9],                       # 7-31: Complement

    [0, 1, 4, 5, 7],                            # 5-Z18
    [0, 1, 4, 5, 6, 7, 9],                       # 7-Z18: Complement

    [0, 1, 4, 5, 8],                            # 5-21
    [0, 1, 2, 4, 5, 8, 9],                       # 7-21: Complement

    [0, 1, 4, 6, 8],                            # 5-30
    [0, 1, 2, 4, 6, 8, 9],                       # 7-30: Complement

    [0, 1, 4, 6, 9],                            # 5-32
    [0, 1, 3, 4, 6, 8, 9],                       # 7-32: Complement

    [0, 1, 4, 7, 8],                            # 5-22
    [0, 1, 2, 5, 6, 8, 9],                       # 7-22: Complement

    [0, 1, 5, 6, 8],                            # 5-20
    [0, 1, 2, 5, 6, 7, 9],                       # 7-20: Complement

    [0, 2, 3, 4, 6],                            # 5-8
    [0, 2, 3, 4, 5, 6, 8],                       # 7-8: Complement

    [0, 2, 3, 4, 7],                            # 5-11
    [0, 1, 3, 4, 5, 6, 8],                       # 7-11: Complement

    [0, 2, 3, 5, 7],                            # 5-23
    [0, 2, 3, 4, 5, 7, 9],                       # 7-23: Complement

    [0, 2, 3, 5, 8],                            # 5-25
    [0, 2, 3, 4, 6, 7, 9],                       # 7-25: Complement

    [0, 2, 3, 6, 8],                            # 5-28
    [0, 1, 3, 5, 6, 7, 9],                       # 7-28: Complement

    [0, 2, 4, 5, 8],                            # 5-26
    [0, 1, 3, 4, 5, 7, 9],                       # 7-26: Complement

    [0, 2, 4, 6, 8],                            # 5-33: Whole-tone pentachord
    [0, 1, 2, 4, 6, 8, 10],                      # 7-33: Complement

    [0, 2, 4, 6, 9],                            # 5-34
    [0, 1, 3, 4, 6, 8, 10],                      # 7-34: Complement

    [0, 2, 4, 7, 9],                            # 5-35: Pentatonic / diatonic subset
    [0, 1, 3, 5, 6, 8, 10],                      # 7-35: Diatonic scale complement

    [0, 3, 4, 5, 8],                            # 5-Z37
    [0, 1, 3, 4, 5, 7, 8],                       # 7-Z37: Complement

    # ---------------------------------------------------------------------
    # Cardinality 6 (hexachords) — self-complementary or paired
    # ---------------------------------------------------------------------
    [0, 1, 2, 3, 4, 5],                         # 6-1: Chromatic hexachord
    [0, 1, 2, 3, 4, 6],                         # 6-2
    [0, 1, 2, 3, 4, 7],                         # 6-Z36
    [0, 1, 2, 3, 5, 6],                         # 6-Z3
    [0, 1, 2, 3, 4, 8],                         # 6-Z37
    [0, 1, 2, 4, 5, 6],                         # 6-Z4
    [0, 1, 2, 3, 5, 7],                         # 6-9
    [0, 1, 2, 3, 5, 8],                         # 6-Z40
    [0, 1, 2, 4, 5, 7],                         # 6-Z11
    [0, 1, 2, 3, 6, 7],                         # 6-5
    [0, 1, 2, 3, 6, 8],                         # 6-Z41
    [0, 1, 2, 4, 6, 7],                         # 6-Z12
    [0, 1, 2, 3, 6, 9],                         # 6-Z42
    [0, 1, 3, 4, 6, 7],                         # 6-Z13
    [0, 1, 2, 3, 7, 8],                         # 6-Z6
    [0, 1, 2, 5, 6, 7],                         # 6-Z38
    [0, 1, 2, 4, 5, 8],                         # 6-15
    [0, 1, 2, 4, 6, 8],                         # 6-21
    [0, 1, 2, 4, 6, 9],                         # 6-Z45
    [0, 1, 3, 4, 6, 8],                         # 6-Z23
    [0, 1, 2, 4, 7, 8],                         # 6-Z17
    [0, 1, 2, 5, 6, 8],                         # 6-Z43
    [0, 1, 2, 4, 7, 9],                         # 6-Z46
    [0, 1, 3, 5, 6, 8],                         # 6-Z24
    [0, 1, 2, 5, 6, 9],                         # 6-Z44
    [0, 1, 3, 4, 7, 8],                         # 6-Z19
    [0, 1, 2, 5, 7, 8],                         # 6-18
    [0, 1, 2, 5, 7, 9],                         # 6-Z47
    [0, 1, 3, 5, 7, 8],                         # 6-Z25
    [0, 1, 2, 6, 7, 8],                         # 6-7
    [0, 1, 3, 4, 5, 7],                         # 6-Z10
    [0, 2, 3, 4, 5, 8],                         # 6-Z39
    [0, 1, 3, 4, 5, 8],                         # 6-14
    [0, 1, 3, 4, 6, 9],                         # 6-27
    [0, 1, 3, 4, 7, 9],                         # 6-Z48
    [0, 1, 3, 5, 6, 9],                         # 6-Z26
    [0, 1, 3, 5, 7, 9],                         # 6-34: Mystic chord / Prometheus
    [0, 1, 3, 6, 7, 9],                         # 6-Z49
    [0, 1, 4, 5, 6, 8],                         # 6-Z28
    [0, 1, 4, 5, 7, 9],                         # 6-Z50
    [0, 1, 4, 5, 8, 9],                         # 6-20: Hexatonic scale
    [0, 1, 4, 6, 7, 9],                         # 6-31
    [0, 2, 3, 6, 7, 9],                         # 6-Z29
    [0, 2, 3, 4, 5, 7],                         # 6-8
    [0, 2, 3, 4, 6, 8],                         # 6-Z22
    [0, 2, 3, 4, 6, 9],                         # 6-Z51
    [0, 2, 3, 5, 6, 8],                         # 6-16
    [0, 2, 3, 5, 7, 9],                         # 6-33
    [0, 2, 4, 5, 7, 9],                         # 6-32: Diatonic hexachord
    [0, 2, 4, 6, 8, 10],                        # 6-35: Whole-tone scale

    # ---------------------------------------------------------------------
    # Cardinality 12 (chromatic aggregate) — included for completeness
    # ---------------------------------------------------------------------
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],     # 12-1: Complete chromatic
]


# =============================================================================
# INTERVAL VECTOR COMPUTATION
# =============================================================================

def compute_interval_vector(pitch_class_set: list[int]) -> list[int]:
    """
    Compute the interval vector for a given pitch-class set using music21.

    Parameters
    ----------
    pitch_class_set : list[int]
        A list of pitch classes (integers 0-11).

    Returns
    -------
    list[int]
        A six-element interval vector [ic1, ic2, ic3, ic4, ic5, ic6].
    """
    pitches = [pitch.Pitch(pc) for pc in pitch_class_set]
    chord_obj = chord.Chord(pitches)
    return chord_obj.intervalVector


def compute_all_interval_vectors(set_classes: list[list[int]]) -> np.ndarray:
    """
    Compute interval vectors for all set classes in the provided list.

    Parameters
    ----------
    set_classes : list[list[int]]
        A list of pitch-class sets in prime form.

    Returns
    -------
    np.ndarray
        A 2D array where each row is the interval vector of a set class.
        Shape: (number_of_set_classes, 6)
    """
    interval_vectors = [
        compute_interval_vector(set_class)
        for set_class in set_classes
    ]
    return np.array(interval_vectors)


def sum_interval_vectors(interval_vectors: np.ndarray) -> np.ndarray:
    """
    Sum all interval vectors to produce a single aggregate vector.

    Parameters
    ----------
    interval_vectors : np.ndarray
        A 2D array of interval vectors (one per row).

    Returns
    -------
    np.ndarray
        A 1D array containing the sum across all interval vectors.
        Each element represents the total count of that interval class
        across all set classes.
    """
    return np.sum(interval_vectors, axis=0)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to compute and display the summed interval vector.

    This calculates the aggregate distribution of interval classes
    across the entire set-class universe.
    """
    # Compute interval vectors for all set classes
    all_interval_vectors = compute_all_interval_vectors(SET_CLASS_LIST)

    # Sum across all set classes
    summed_interval_vector = sum_interval_vectors(all_interval_vectors)

    # Display results
    print("Summed Interval Vector across all set classes:")
    print(f"  [ic1, ic2, ic3, ic4, ic5, ic6] = {summed_interval_vector}")


if __name__ == "__main__":
    main()