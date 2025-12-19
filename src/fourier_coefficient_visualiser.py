"""
Fourier Coefficient Visualiser for Pitch Class Sets
====================================================

This module provides an interactive visualisation tool for exploring how Discrete
Fourier Transform (DFT) coefficients capture pitch class collections. It is part
of the analytical tools developed for the author's PhD thesis on pitch class set
theory and Fourier analysis.

The visualiser allows users to:
    - Activate individual Fourier coefficients (1-6)
    - Adjust the magnitude and phase of each coefficient
    - Observe which pitch classes have positive amplitude in the combined waveform
    - Play the resulting pitch class set as a chord or arpeggio

Mathematical Background
-----------------------
The DFT of a pitch class set can be decomposed into sinusoidal components. Each
coefficient k corresponds to a periodicity that divides the octave into k equal
parts. The magnitude indicates how strongly the set exhibits that periodicity,
whilst the phase indicates the rotational position.

For a pitch class p and coefficient k:
    f(p) = magnitude * cos(2π * k * p / 12 + phase)

The combined waveform is the sum of all active sinusoids. Pitch classes where
this combined waveform is positive are highlighted.

Usage
-----
Run this script directly to launch the interactive visualisation:
    $ python fourier_coefficient_visualiser.py

Dependencies
------------
- numpy: Numerical computations
- matplotlib: Plotting and interactive widgets
- music21 (optional): Chord identification
- pygame (optional): Audio playback

Author: [Thesis Author]
Created for: PhD Thesis on Pitch Class Set Theory and Fourier Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button
import matplotlib.patches as patches

# -----------------------------------------------------------------------------
# Optional Dependency Handling
# -----------------------------------------------------------------------------

# Attempt to import music21 for chord identification functionality
try:
    from music21 import stream, note, chord, lily
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    print("Warning: music21 not installed. Musical notation will not be available.")

# Attempt to import pygame for audio playback functionality
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not installed. Audio playback will not be available.")


# =============================================================================
# Main Visualiser Class
# =============================================================================

class DFTPitchClassVisualiser:
    """
    Interactive visualiser for exploring DFT coefficients of pitch class sets.
    
    This class creates a matplotlib figure with:
        - A main plot showing individual coefficient sinusoids and their sum
        - Checkboxes to activate/deactivate coefficients 1-6
        - Sliders to adjust magnitude and phase for each active coefficient
        - Highlighted regions showing pitch classes with positive amplitude
        - Information panel displaying the resulting pitch class set
        - Audio playback buttons (if pygame is available)
    
    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The main figure containing all plot elements.
    ax : matplotlib.axes.Axes
        The main axes for plotting sinusoids.
    coefficients : dict
        Dictionary storing the state of each coefficient (1-6), including:
        - 'active': bool indicating if coefficient is enabled
        - 'magnitude': float between 0 and 1
        - 'phase': float between -π and π
        - 'magnitude_slider': Slider widget reference
        - 'phase_slider': Slider widget reference
    current_pc_set : set
        The current pitch class set determined by positive amplitude regions.
    
    Notes
    -----
    The chromatic scale is represented as pitch classes 0-11, where:
        0=C, 1=C#, 2=D, 3=D#, 4=E, 5=F, 6=F#, 7=G, 8=G#, 9=A, 10=A#, 11=B
    """
    
    # -------------------------------------------------------------------------
    # Initialisation
    # -------------------------------------------------------------------------
    
    def __init__(self):
        """
        Initialise the visualiser and display the interactive figure.
        """
        # Create the main figure with specified dimensions
        self.fig = plt.figure(figsize=(14, 10))
        
        # Main plot area - positioned to leave space for controls
        # [left, bottom, width, height] in figure coordinates (0-1)
        self.ax = plt.axes([0.20, 0.55, 0.70, 0.40])
        
        # Continuous pitch class values for smooth curve plotting (0 to 12)
        # Using 1000 points for visual smoothness
        self.pitch_classes = np.linspace(0, 12, 1000)
        
        # Discrete pitch class values for axis labels (0 to 12, inclusive)
        # Note: 12 is included for visual continuity (equivalent to 0)
        self.discrete_pitch_classes = np.arange(13)
        
        # Initialise coefficient parameters
        # Each coefficient (1-6) has magnitude, phase, and slider references
        self.coefficients = {
            i: {
                'active': False,           # Whether this coefficient is enabled
                'magnitude': 0.5,          # Initial magnitude (0 to 1)
                'phase': -np.pi,           # Initial phase (-π to π)
                'magnitude_slider': None,  # Will hold Slider widget reference
                'phase_slider': None       # Will hold Slider widget reference
            } for i in range(1, 7)
        }
        
        # Storage for matplotlib line objects (for updating plots)
        self.lines = {}
        self.combined_line = None
        
        # Distinct colours for each coefficient
        self.colours = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        # Current pitch class set (updated when coefficients change)
        self.current_pc_set = set()
        
        # Initialise pygame audio system if available
        if PYGAME_AVAILABLE:
            pygame.init()
            pygame.mixer.init(
                frequency=22050,   # Sample rate in Hz
                size=-16,          # 16-bit signed audio
                channels=2,        # Stereo
                buffer=512         # Buffer size
            )
        
        # Build the user interface components
        self.setup_plot()
        self.setup_checkboxes()
        self.setup_sliders()
        self.setup_buttons()
        self.setup_info_display()
        
        # Render initial state
        self.update_plot()
        
        # Display the interactive figure
        plt.show()
    
    # -------------------------------------------------------------------------
    # User Interface Setup Methods
    # -------------------------------------------------------------------------
    
    def setup_plot(self):
        """
        Configure the main plot axes with labels, grid, and reference lines.
        """
        self.ax.set_xlabel('Pitch Class', fontsize=12)
        self.ax.set_ylabel('Amplitude', fontsize=12)
        self.ax.set_title('DFT Coefficients for Pitch Class Sets', fontsize=14)
        self.ax.set_xlim(0, 12)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.grid(True, alpha=0.3)
        
        # Configure x-axis with pitch class labels
        self.ax.set_xticks(self.discrete_pitch_classes)
        self.ax.set_xticklabels(
            ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        )
        
        # Add vertical reference lines at each integer pitch class
        for pc in self.discrete_pitch_classes:
            self.ax.axvline(x=pc, color='gray', linestyle=':', alpha=0.5)
        
        # Add horizontal reference line at y=0 (the threshold for inclusion)
        self.ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    def setup_checkboxes(self):
        """
        Create checkboxes for activating/deactivating each coefficient.
        """
        # Position checkboxes in the upper-left area of the figure
        ax_checkbox = plt.axes([0.02, 0.65, 0.12, 0.25])
        
        self.checkbox = CheckButtons(
            ax_checkbox,
            ['Coeff 1', 'Coeff 2', 'Coeff 3', 'Coeff 4', 'Coeff 5', 'Coeff 6'],
            [False] * 6  # All coefficients initially inactive
        )
        self.checkbox.on_clicked(self.update_coefficient_active)
    
    def setup_sliders(self):
        """
        Initialise placeholder for dynamic slider creation.
        
        Sliders are created dynamically when coefficients are activated,
        rather than being pre-created for all coefficients.
        """
        self.slider_axes = []
        self.slider_labels = []
    
    def setup_buttons(self):
        """
        Create audio playback control buttons.
        
        Only creates buttons if pygame is available for audio playback.
        """
        if PYGAME_AVAILABLE:
            # "Play Chord" button - plays all pitch classes simultaneously
            ax_play_chord = plt.axes([0.02, 0.15, 0.10, 0.04])
            self.btn_play_chord = Button(ax_play_chord, 'Play Chord')
            self.btn_play_chord.on_clicked(self.play_chord)
            
            # "Play Arpeggio" button - plays pitch classes sequentially
            ax_play_arp = plt.axes([0.13, 0.15, 0.10, 0.04])
            self.btn_play_arp = Button(ax_play_arp, 'Play Arpeggio')
            self.btn_play_arp.on_clicked(self.play_arpeggio)
    
    def setup_info_display(self):
        """
        Create the text area for displaying pitch class set information.
        """
        self.ax_info = plt.axes([0.02, 0.22, 0.20, 0.20])
        self.ax_info.axis('off')  # Hide axes frame
        
        # Create text object for dynamic updates
        self.info_text = self.ax_info.text(
            0.05, 0.9, '',
            transform=self.ax_info.transAxes,
            fontsize=10,
            verticalalignment='top'
        )
    
    # -------------------------------------------------------------------------
    # Slider Management Methods
    # -------------------------------------------------------------------------
    
    def create_sliders_for_coefficient(self, coeff):
        """
        Dynamically create magnitude and phase sliders for all active coefficients.
        
        This method removes all existing sliders and recreates them for the
        currently active coefficients. Sliders are stacked vertically below
        the main plot.
        
        Parameters
        ----------
        coeff : int
            The coefficient number that was just toggled (used for callback
            context, though all active coefficients are processed).
        """
        # Remove all existing sliders before creating new ones
        self.clear_sliders()
        
        # Starting y-position for sliders (below the main plot)
        y_position = 0.48
        slider_height = 0.015
        slider_spacing = 0.022
        
        # Create sliders for each active coefficient
        for i in range(1, 7):
            if self.coefficients[i]['active']:
                # Magnitude slider (range: 0 to 1)
                ax_mag = plt.axes([0.30, y_position, 0.55, slider_height])
                mag_slider = Slider(
                    ax_mag,
                    f'Magnitude {i}',
                    valmin=0,
                    valmax=1,
                    valinit=self.coefficients[i]['magnitude'],
                    valstep=0.01,
                    color=self.colours[i-1]
                )
                # Use lambda with default argument to capture current i value
                mag_slider.on_changed(lambda val, c=i: self.update_magnitude(c, val))
                self.coefficients[i]['magnitude_slider'] = mag_slider
                
                # Phase slider (range: -π to π)
                y_position -= slider_spacing
                ax_phase = plt.axes([0.30, y_position, 0.55, slider_height])
                phase_slider = Slider(
                    ax_phase,
                    f'Phase {i}',
                    valmin=-np.pi,
                    valmax=np.pi,
                    valinit=self.coefficients[i]['phase'],
                    valstep=0.01,
                    color=self.colours[i-1]
                )
                phase_slider.on_changed(lambda val, c=i: self.update_phase(c, val))
                self.coefficients[i]['phase_slider'] = phase_slider
                
                y_position -= slider_spacing
    
    def clear_sliders(self):
        """
        Remove all existing slider widgets from the figure.
        """
        for coeff_data in self.coefficients.values():
            if coeff_data['magnitude_slider']:
                coeff_data['magnitude_slider'].ax.remove()
                coeff_data['magnitude_slider'] = None
            if coeff_data['phase_slider']:
                coeff_data['phase_slider'].ax.remove()
                coeff_data['phase_slider'] = None
    
    # -------------------------------------------------------------------------
    # Mathematical Computation Methods
    # -------------------------------------------------------------------------
    
    def calculate_sinusoid(self, coefficient, magnitude, phase):
        """
        Calculate the sinusoidal waveform for a given Fourier coefficient.
        
        The formula computes the real part of the DFT basis function:
            f(p) = magnitude × cos(2π × coefficient × p / 12 + phase)
        
        Parameters
        ----------
        coefficient : int
            The coefficient number (1-6), determining the periodicity.
        magnitude : float
            The amplitude of the sinusoid (0 to 1).
        phase : float
            The phase offset in radians (-π to π).
        
        Returns
        -------
        numpy.ndarray
            Array of amplitude values for each point in self.pitch_classes.
        """
        return magnitude * np.cos(
            2 * np.pi * coefficient * self.pitch_classes / 12 + phase
        )
    
    def calculate_combined_waveform(self):
        """
        Calculate the sum of all active sinusoidal components.
        
        Returns
        -------
        numpy.ndarray
            Array of combined amplitude values for each pitch class point.
        """
        combined = np.zeros_like(self.pitch_classes)
        
        for coeff in range(1, 7):
            if self.coefficients[coeff]['active']:
                combined += self.calculate_sinusoid(
                    coeff,
                    self.coefficients[coeff]['magnitude'],
                    self.coefficients[coeff]['phase']
                )
        
        return combined
    
    def find_positive_pitch_classes_combined(self):
        """
        Identify pitch classes where the combined waveform has positive amplitude.
        
        A pitch class is included in the resulting set if the combined waveform
        value at that integer pitch class position is greater than zero.
        
        Returns
        -------
        set
            Set of pitch class integers (0-11) with positive amplitude.
        """
        combined = self.calculate_combined_waveform()
        positive_pcs = set()
        
        for pc in range(12):
            # Calculate the array index corresponding to this pitch class
            # The pitch_classes array spans 0-12 with 1000 points
            idx = int(pc * len(self.pitch_classes) / 12)
            
            if combined[idx] > 0:
                positive_pcs.add(pc)
        
        return positive_pcs
    
    # -------------------------------------------------------------------------
    # Callback Methods (Event Handlers)
    # -------------------------------------------------------------------------
    
    def update_coefficient_active(self, label):
        """
        Handle checkbox click to toggle coefficient activation.
        
        Parameters
        ----------
        label : str
            The label of the clicked checkbox (e.g., 'Coeff 1').
        """
        # Extract coefficient number from label string
        coeff_num = int(label.split()[-1])
        
        # Toggle the active state
        self.coefficients[coeff_num]['active'] = not self.coefficients[coeff_num]['active']
        
        # Recreate sliders to reflect new active coefficients
        self.create_sliders_for_coefficient(coeff_num)
        
        # Redraw the plot
        self.update_plot()
    
    def update_magnitude(self, coeff, val):
        """
        Handle magnitude slider change for a coefficient.
        
        Parameters
        ----------
        coeff : int
            The coefficient number (1-6).
        val : float
            The new magnitude value (0 to 1).
        """
        self.coefficients[coeff]['magnitude'] = val
        self.update_plot()
    
    def update_phase(self, coeff, val):
        """
        Handle phase slider change for a coefficient.
        
        Parameters
        ----------
        coeff : int
            The coefficient number (1-6).
        val : float
            The new phase value in radians (-π to π).
        """
        self.coefficients[coeff]['phase'] = val
        self.update_plot()
    
    # -------------------------------------------------------------------------
    # Plot Update Methods
    # -------------------------------------------------------------------------
    
    def update_plot(self):
        """
        Redraw the plot based on current coefficient settings.
        
        This method:
            1. Clears previous plot elements (lines, patches)
            2. Plots individual sinusoids for each active coefficient
            3. Plots the combined waveform
            4. Highlights pitch classes with positive amplitude
            5. Updates the legend and y-axis limits
            6. Updates the information panel
        """
        # Clear previous rectangular patches (pitch class highlights)
        for patch in self.ax.patches[:]:
            if isinstance(patch, patches.Rectangle):
                patch.remove()
        
        # Clear previous line objects
        for line in self.lines.values():
            line.remove()
        self.lines.clear()
        
        # Clear previous combined line
        if self.combined_line:
            self.combined_line.remove()
            self.combined_line = None
        
        # Check if any coefficients are currently active
        any_active = any(self.coefficients[i]['active'] for i in range(1, 7))
        
        if any_active:
            # Plot individual sinusoids for each active coefficient
            legend_labels = []
            max_amplitude = 0
            
            for coeff in range(1, 7):
                if self.coefficients[coeff]['active']:
                    y_values = self.calculate_sinusoid(
                        coeff,
                        self.coefficients[coeff]['magnitude'],
                        self.coefficients[coeff]['phase']
                    )
                    
                    line, = self.ax.plot(
                        self.pitch_classes, y_values,
                        color=self.colours[coeff-1],
                        linewidth=1.5,
                        alpha=0.6,
                        label=f'Coefficient {coeff}'
                    )
                    self.lines[coeff] = line
                    legend_labels.append(f'Coefficient {coeff}')
                    max_amplitude = max(max_amplitude, self.coefficients[coeff]['magnitude'])
            
            # Plot the combined waveform (sum of all active sinusoids)
            combined = self.calculate_combined_waveform()
            self.combined_line, = self.ax.plot(
                self.pitch_classes, combined,
                color='black',
                linewidth=3,
                label='Combined',
                zorder=10  # Ensure combined line is drawn on top
            )
            legend_labels.append('Combined')
            
            # Determine which pitch classes have positive amplitude
            self.current_pc_set = self.find_positive_pitch_classes_combined()
            
            # Highlight positive pitch class regions with yellow rectangles
            for pc in self.current_pc_set:
                rect = patches.Rectangle(
                    (pc - 0.4, -1.5),  # (x, y) of lower-left corner
                    0.8,               # width
                    3,                 # height (spans full y-range)
                    linewidth=0,
                    edgecolor='none',
                    facecolor='yellow',
                    alpha=0.2
                )
                self.ax.add_patch(rect)
            
            # Adjust y-axis limits based on maximum possible combined amplitude
            total_max = sum(
                self.coefficients[i]['magnitude']
                for i in range(1, 7)
                if self.coefficients[i]['active']
            )
            self.ax.set_ylim(-total_max * 1.2, total_max * 1.2)
            
            # Update the legend
            self.ax.legend(legend_labels, loc='upper right')
            
            # Update the information display
            self.update_info_display()
        else:
            # No active coefficients - reset to default state
            self.ax.set_ylim(-1.5, 1.5)
            
            # Remove legend if present
            if self.ax.get_legend():
                self.ax.get_legend().remove()
            
            self.current_pc_set = set()
            self.update_info_display()
        
        # Request figure redraw
        self.fig.canvas.draw_idle()
    
    def update_info_display(self):
        """
        Update the information panel with the current pitch class set details.
        
        Displays:
            - The pitch class set in integer notation (e.g., {0, 4, 7})
            - The corresponding note names (e.g., {C, E, G})
            - The chord name (if music21 is available and set has ≥3 notes)
        """
        if self.current_pc_set:
            # Format pitch class set as sorted list
            pc_list = sorted(list(self.current_pc_set))
            pc_string = '{' + ', '.join(map(str, pc_list)) + '}'
            
            # Convert pitch classes to note names
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                          'F#', 'G', 'G#', 'A', 'A#', 'B']
            notes_string = '{' + ', '.join(note_names[pc] for pc in pc_list) + '}'
            
            info_text = f'Pitch Class Set:\n{pc_string}\n\nNotes:\n{notes_string}'
            
            # Attempt chord identification using music21
            if MUSIC21_AVAILABLE and len(pc_list) >= 3:
                try:
                    # Create chord from pitch classes (using middle C octave)
                    chord_notes = [note.Note(midi=60+pc) for pc in pc_list]
                    c = chord.Chord(chord_notes)
                    chord_type = c.pitchedCommonName
                    info_text += f'\n\nChord: {chord_type}'
                except:
                    pass  # Chord identification failed; continue without it
        else:
            info_text = 'Pitch Class Set:\n{}\n\nNotes:\n{}'
        
        self.info_text.set_text(info_text)
        self.fig.canvas.draw_idle()
    
    # -------------------------------------------------------------------------
    # Audio Playback Methods
    # -------------------------------------------------------------------------
    
    def generate_tone(self, frequency, duration=0.5, sample_rate=22050, timbre='piano'):
        """
        Generate an audio waveform for a single tone.
        
        Parameters
        ----------
        frequency : float
            The frequency of the tone in Hz.
        duration : float, optional
            Duration of the tone in seconds (default: 0.5).
        sample_rate : int, optional
            Sample rate in Hz (default: 22050).
        timbre : str, optional
            The timbre type: 'piano' or 'sine' (default: 'piano').
        
        Returns
        -------
        numpy.ndarray
            Stereo audio array of shape (frames, 2) with int16 values.
        """
        frames = int(duration * sample_rate)
        arr = np.zeros((frames, 2), dtype=np.int16)
        max_amplitude = 2000
        
        for i in range(frames):
            # Calculate amplitude envelope based on timbre
            if timbre == 'piano':
                # Piano-like ADSR (Attack-Decay-Sustain-Release) envelope
                if i < frames * 0.01:
                    # Attack phase: rapid rise
                    envelope = i / (frames * 0.01)
                elif i < frames * 0.1:
                    # Decay phase: gradual decrease
                    envelope = 1.0 - 0.3 * (i - frames * 0.01) / (frames * 0.09)
                elif i < frames * 0.7:
                    # Sustain phase: exponential decay
                    envelope = 0.7 * np.exp(-0.5 * (i - frames * 0.1) / (frames * 0.6))
                else:
                    # Release phase: final fade-out
                    envelope = 0.7 * np.exp(-0.5 * (frames * 0.7 - frames * 0.1) / (frames * 0.6)) * \
                              (frames - i) / (frames * 0.3)
            else:
                # Simple sine wave with linear fade-in/fade-out
                envelope = 1.0
                if i < frames * 0.1:
                    envelope = i / (frames * 0.1)
                elif i > frames * 0.9:
                    envelope = (frames - i) / (frames * 0.1)
            
            # Generate waveform sample
            if timbre == 'piano':
                # Add harmonics for richer piano-like sound
                value = 0
                harmonics = [1.0, 0.3, 0.1, 0.05, 0.02, 0.01]  # Harmonic amplitudes
                for h, amp in enumerate(harmonics, 1):
                    value += amp * np.sin(2 * np.pi * frequency * h * i / sample_rate)
                value = int(max_amplitude * envelope * value / sum(harmonics))
            else:
                # Pure sine wave
                value = int(max_amplitude * envelope * np.sin(2 * np.pi * frequency * i / sample_rate))
            
            # Write to both stereo channels
            arr[i] = [value, value]
        
        return arr
    
    def play_chord(self, event=None):
        """
        Play all pitch classes in the current set simultaneously as a chord.
        
        Parameters
        ----------
        event : optional
            Matplotlib button click event (not used, but required for callback).
        """
        if not PYGAME_AVAILABLE or not self.current_pc_set:
            return
        
        # Stop any currently playing audio
        pygame.mixer.stop()
        
        # Mix all tones together
        mixed = None
        for pc in self.current_pc_set:
            # Convert pitch class to MIDI note number (C4 = 60)
            midi_note = 60 + pc
            # Convert MIDI note to frequency (A4 = 440 Hz = MIDI 69)
            frequency = 440 * (2 ** ((midi_note - 69) / 12))
            
            tone = self.generate_tone(frequency, duration=1.0, timbre='piano')
            
            if mixed is None:
                mixed = tone.astype(np.float32)
            else:
                mixed += tone.astype(np.float32)
        
        if mixed is not None:
            # Normalise amplitude to prevent clipping
            mixed = mixed / len(self.current_pc_set)
            mixed = mixed.astype(np.int16)
            
            # Convert to pygame sound and play
            sound = pygame.sndarray.make_sound(mixed)
            sound.play()
    
    def play_arpeggio(self, event=None):
        """
        Play pitch classes sequentially as an ascending arpeggio.
        
        Parameters
        ----------
        event : optional
            Matplotlib button click event (not used, but required for callback).
        """
        if not PYGAME_AVAILABLE or not self.current_pc_set:
            return
        
        # Stop any currently playing audio
        pygame.mixer.stop()
        
        # Play each pitch class in ascending order with delays
        for i, pc in enumerate(sorted(self.current_pc_set)):
            # Convert pitch class to MIDI note and then to frequency
            midi_note = 60 + pc
            frequency = 440 * (2 ** ((midi_note - 69) / 12))
            
            tone = self.generate_tone(frequency, duration=0.3, timbre='piano')
            sound = pygame.sndarray.make_sound(tone)
            
            # Wait before playing next note (200ms between notes)
            pygame.time.wait(int(i * 200))
            sound.play()


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    app = DFTPitchClassVisualiser()
