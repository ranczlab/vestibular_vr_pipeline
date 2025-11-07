"""
GUI Annotation Tool for Saccade Classification

PyQt5-based GUI for manually annotating saccades into 4 classes:
- Compensatory
- Orienting (Purely Orienting)
- Saccade-and-Fixate
- Non-Saccade

Starts with rule-based classifications (2 classes) and allows manual correction.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

# PyQt5 imports
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
        QSplitter, QMessageBox, QLineEdit, QTextEdit, QGroupBox, QShortcut
    )
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtGui import QKeySequence
    PYQT5_AVAILABLE = True
except ImportError as e:
    PYQT5_AVAILABLE = False
    print(f"⚠️ PyQt5 not available: {e}")
    print("   Install with: pip install PyQt5")

# Matplotlib imports for plotting (only if PyQt5 is available)
if PYQT5_AVAILABLE:
    try:
        import matplotlib
        matplotlib.use('Qt5Agg')  # Use Qt5 backend (only if PyQt5 is available)
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        import matplotlib.pyplot as plt
        MATPLOTLIB_AVAILABLE = True
    except ImportError as e:
        MATPLOTLIB_AVAILABLE = False
        print(f"⚠️ matplotlib Qt5Agg backend not available: {e}")
else:
    MATPLOTLIB_AVAILABLE = False

# Import annotation storage
from sleap.annotation_storage import (
    save_annotation, load_annotations, get_annotation_stats, VALID_LABELS
)

# Global variable to track if a GUI window is already open
_active_gui_window = None


class TimeSeriesPlotWidget(QWidget):
    """Widget for displaying full time series with saccades highlighted."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6))  # Narrower: 12 -> 8
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        self.ax_position = None
        self.ax_velocity = None
        self.saccade_rects = []
        
    def plot_time_series(
        self,
        df: pd.DataFrame,
        saccades_df: pd.DataFrame,
        selected_saccade_id: Optional[int] = None,
        user_labels: Optional[Dict[int, str]] = None,
        window_duration: float = 6.0,
        vel_thresh: Optional[float] = None
    ):
        """
        Plot time series with saccades highlighted (6-second window centered on selected saccade).
        
        Parameters
        ----------
        df : pd.DataFrame
            Full time series DataFrame with 'Seconds', 'X_smooth', 'vel_x_smooth'
        saccades_df : pd.DataFrame
            DataFrame with saccade information (uses DataFrame index as saccade_id)
        selected_saccade_id : int, optional
            ID of currently selected saccade (will be centered in window)
        user_labels : dict, optional
            Dictionary mapping saccade_id to user_label for color coding
        window_duration : float
            Total window duration in seconds (default: 6.0, showing 3s before and 3s after)
        vel_thresh : float, optional
            Velocity threshold for detection (will be plotted as horizontal lines)
        """
        self.figure.clear()
        
        # Create subplots
        self.ax_position = self.figure.add_subplot(2, 1, 1)
        self.ax_velocity = self.figure.add_subplot(2, 1, 2)
        
        # Determine time window (centered on selected saccade)
        window_half = window_duration / 2.0  # 3 seconds before and after
        
        if selected_saccade_id is not None and len(saccades_df) > 0:
            # Find selected saccade time
            try:
                if selected_saccade_id in saccades_df.index:
                    selected_row = saccades_df.loc[[selected_saccade_id]]
                    if len(selected_row) > 0:
                        center_time = selected_row.iloc[0].get('time', selected_row.iloc[0].get('start_time', 0))
                        if pd.isna(center_time):
                            center_time = df['Seconds'].min()
                        window_start = max(df['Seconds'].min(), center_time - window_half)
                        window_end = min(df['Seconds'].max(), center_time + window_half)
                    else:
                        # Fallback: use full range
                        window_start = df['Seconds'].min()
                        window_end = df['Seconds'].max()
                else:
                    # Fallback: use full range
                    window_start = df['Seconds'].min()
                    window_end = df['Seconds'].max()
            except (KeyError, IndexError, AttributeError) as e:
                # Fallback on any error
                window_start = df['Seconds'].min()
                window_end = min(df['Seconds'].max(), df['Seconds'].min() + window_duration)
        else:
            # No selected saccade: show first 6 seconds
            window_start = df['Seconds'].min()
            window_end = min(df['Seconds'].max(), window_start + window_duration)
        
        # Filter data to window (use numpy for faster boolean indexing)
        try:
            window_mask = (df['Seconds'].values >= window_start) & (df['Seconds'].values <= window_end)
            df_window = df.loc[window_mask]
        except (KeyError, AttributeError):
            # Fallback if Seconds column missing
            df_window = df.iloc[:100] if len(df) > 100 else df
            window_start = df_window['Seconds'].min() if 'Seconds' in df_window.columns else 0
            window_end = df_window['Seconds'].max() if 'Seconds' in df_window.columns else 6.0
        
        if len(df_window) == 0:
            # If window is empty, use full data
            df_window = df
            if 'Seconds' in df.columns:
                window_start = df['Seconds'].min()
                window_end = df['Seconds'].max()
            else:
                window_start = 0
                window_end = 6.0
        
        # Calculate local Y limits from windowed data (autoscaling) - use numpy for speed
        try:
            pos_values = df_window['X_smooth'].values if 'X_smooth' in df_window.columns else np.array([0])
            vel_values = df_window['vel_x_smooth'].values if 'vel_x_smooth' in df_window.columns else np.array([0])
            y_position_min = float(np.nanmin(pos_values)) if len(pos_values) > 0 else 0.0
            y_position_max = float(np.nanmax(pos_values)) if len(pos_values) > 0 else 1.0
            y_velocity_min = float(np.nanmin(vel_values)) if len(vel_values) > 0 else -100.0
            y_velocity_max = float(np.nanmax(vel_values)) if len(vel_values) > 0 else 100.0
        except (KeyError, AttributeError, ValueError):
            # Fallback values
            y_position_min, y_position_max = 0.0, 1.0
            y_velocity_min, y_velocity_max = -100.0, 100.0
        
        # Add small padding (5%)
        pos_range = y_position_max - y_position_min
        vel_range = y_velocity_max - y_velocity_min
        # Handle zero range case
        if pos_range == 0:
            y_position_limits = (y_position_min - 1.0, y_position_max + 1.0)
        else:
            y_position_limits = (y_position_min - 0.05 * pos_range, y_position_max + 0.05 * pos_range)
        if vel_range == 0:
            y_velocity_limits = (y_velocity_min - 10.0, y_velocity_max + 10.0)
        else:
            y_velocity_limits = (y_velocity_min - 0.05 * vel_range, y_velocity_max + 0.05 * vel_range)
        
        # Plot position (thicker line) - use numpy arrays for faster plotting
        try:
            if 'Seconds' in df_window.columns and 'X_smooth' in df_window.columns:
                self.ax_position.plot(df_window['Seconds'].values, df_window['X_smooth'].values, 
                                     'b-', linewidth=2.0, alpha=0.7)
        except (KeyError, AttributeError, ValueError):
            pass  # Skip plotting if data missing
        
        self.ax_position.set_ylabel('Position (px)', fontsize=10)
        # Title removed per user request
        self.ax_position.grid(True, alpha=0.3)
        # Set local Y limits (autoscaling)
        self.ax_position.set_ylim(y_position_limits)
        
        # Plot velocity (thicker line) - use numpy arrays for faster plotting
        try:
            if 'Seconds' in df_window.columns and 'vel_x_smooth' in df_window.columns:
                self.ax_velocity.plot(df_window['Seconds'].values, df_window['vel_x_smooth'].values, 
                                     'r-', linewidth=2.0, alpha=0.7)
        except (KeyError, AttributeError, ValueError):
            pass  # Skip plotting if data missing
        self.ax_velocity.set_xlabel('Time (s)', fontsize=10)
        self.ax_velocity.set_ylabel('Velocity (px/s)', fontsize=10)
        self.ax_velocity.grid(True, alpha=0.3)
        # Set local Y limits (autoscaling)
        self.ax_velocity.set_ylim(y_velocity_limits)
        
        # Add horizontal dashed green line for detection threshold in velocity plot
        if vel_thresh is not None:
            self.ax_velocity.axhline(vel_thresh, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
            self.ax_velocity.axhline(-vel_thresh, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Set x-axis limits to window
        self.ax_position.set_xlim([window_start, window_end])
        self.ax_velocity.set_xlim([window_start, window_end])
        
        # Color mapping for labels (user classifications override rule-based)
        label_colors = {
            'compensatory': 'orange',
            'orienting': 'blue',
            'saccade_and_fixate': 'green',
            'non_saccade': 'red',
            None: 'gray'  # Unlabeled
        }
        
        # Rule-based classification colors (used when no user label exists)
        rule_based_colors = {
            'compensatory': 'orange',
            'orienting': 'blue',
            'unknown': 'gray'
        }
        
        # Highlight only the selected saccade with width bar
        if selected_saccade_id is not None and len(saccades_df) > 0:
            try:
                if selected_saccade_id in saccades_df.index:
                    selected_row = saccades_df.loc[[selected_saccade_id]]
                    if len(selected_row) > 0:
                        start_time = selected_row.iloc[0].get('start_time', selected_row.iloc[0].get('time', 0))
                        duration = selected_row.iloc[0].get('duration', 0.1)
                        if pd.isna(start_time):
                            start_time = window_start
                        if pd.isna(duration) or duration <= 0:
                            duration = 0.1
                        end_time = start_time + duration
                        
                        # Only highlight if saccade overlaps with window
                        if end_time >= window_start and start_time <= window_end:
                            # Get label for color: user label takes precedence over rule-based
                            user_label = user_labels.get(selected_saccade_id) if user_labels else None
                            
                            if user_label:
                                # Use user classification color
                                color = label_colors.get(user_label, 'gray')
                            else:
                                # Use rule-based classification color (if available)
                                rule_based_label = selected_row.iloc[0].get('saccade_type', 'unknown')
                                color = rule_based_colors.get(rule_based_label, 'gray')
                            
                            alpha = 0.3  # Selected saccade
                            linewidth = 1
                            
                            # Clip saccade times to window bounds for visualization
                            clip_start = max(start_time, window_start)
                            clip_end = min(end_time, window_end)
                            
                            # Position plot highlight (use edgecolor instead of color to avoid warning)
                            self.ax_position.axvspan(clip_start, clip_end, alpha=alpha, 
                                                    facecolor=color, edgecolor=color, linewidth=linewidth)
                            
                            # Velocity plot highlight
                            self.ax_velocity.axvspan(clip_start, clip_end, alpha=alpha,
                                                    facecolor=color, edgecolor=color, linewidth=linewidth)
            except (KeyError, IndexError, AttributeError, ValueError) as e:
                # Silently skip highlighting on error
                pass
        
        self.canvas.draw()
        
        # Return Y limits for use in segment plot
        return y_position_limits, y_velocity_limits
    
    def clear(self):
        """Clear the plot."""
        self.figure.clear()
        self.canvas.draw()


class PeriSaccadePlotWidget(QWidget):
    """Widget for displaying peri-saccade segment (aligned position/velocity)."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 4))  # Wider for side-by-side: 5 -> 10, shorter height: 6 -> 4
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
    def plot_segment(
        self,
        segment_df: pd.DataFrame,
        saccade_info: Optional[Dict] = None,
        y_position_limits: Optional[Tuple[float, float]] = None,
        y_velocity_limits: Optional[Tuple[float, float]] = None,
        user_label: Optional[str] = None,
        vel_thresh: Optional[float] = None
    ):
        """
        Plot peri-saccade segment aligned to threshold crossing (position and velocity side-by-side).
        
        Parameters
        ----------
        segment_df : pd.DataFrame
            Peri-saccade segment DataFrame with columns:
            'Time_rel_threshold', 'X_smooth_baselined', 'vel_x_smooth'
        saccade_info : dict, optional
            Dictionary with saccade information (amplitude, duration, etc.)
        y_position_limits : tuple, optional
            Y-axis limits for position plot (from time series for matching scale)
        y_velocity_limits : tuple, optional
            Y-axis limits for velocity plot (from time series for matching scale)
        vel_thresh : float, optional
            Velocity threshold for detection (will be plotted as horizontal lines)
        """
        self.figure.clear()
        
        # Create side-by-side subplots (1 row, 2 columns)
        ax_position = self.figure.add_subplot(1, 2, 1)
        ax_velocity = self.figure.add_subplot(1, 2, 2)
        
        # Use provided limits or calculate from segment data
        if y_position_limits is None:
            # Use raw position if available, otherwise baselined
            if 'X_raw' in segment_df.columns:
                pos_data = segment_df['X_raw']
            elif 'X_smooth' in segment_df.columns:
                pos_data = segment_df['X_smooth']
            elif 'X_smooth_baselined' in segment_df.columns:
                pos_data = segment_df['X_smooth_baselined']
            else:
                pos_data = pd.Series([0])
            pos_min, pos_max = pos_data.min(), pos_data.max()
            pos_range = pos_max - pos_min
            y_position_limits = (pos_min - 0.05 * pos_range, pos_max + 0.05 * pos_range)
        
        if y_velocity_limits is None:
            if 'vel_x_smooth' in segment_df.columns:
                vel_data = segment_df['vel_x_smooth']
            else:
                vel_data = pd.Series([0])
            vel_min, vel_max = vel_data.min(), vel_data.max()
            vel_range = vel_max - vel_min
            y_velocity_limits = (vel_min - 0.05 * vel_range, vel_max + 0.05 * vel_range)
        
        # Plot position - use X_smooth to match time series plot and preserve original direction
        # X_smooth preserves the absolute position values (just smoothed), maintaining directionality
        # X_raw is also fine but X_smooth matches the time series better
        # X_smooth_baselined removes the baseline offset and loses absolute direction - avoid this
        # Use numpy arrays for faster plotting
        try:
            if 'Time_rel_threshold' in segment_df.columns:
                if 'X_smooth' in segment_df.columns and len(segment_df) > 0:
                    # Use X_smooth to match time series plot - preserves direction (upward/downward)
                    ax_position.plot(segment_df['Time_rel_threshold'].values, segment_df['X_smooth'].values,
                                   'b-', linewidth=2)
                elif 'X_raw' in segment_df.columns and len(segment_df) > 0:
                    # Fallback to X_raw if X_smooth not available
                    ax_position.plot(segment_df['Time_rel_threshold'].values, segment_df['X_raw'].values,
                                   'b-', linewidth=2)
                elif 'X_smooth_baselined' in segment_df.columns and len(segment_df) > 0:
                    # Last resort: baselined (but this loses absolute direction - not ideal)
                    ax_position.plot(segment_df['Time_rel_threshold'].values, segment_df['X_smooth_baselined'].values,
                                   'b-', linewidth=2)
        except (KeyError, IndexError, AttributeError, ValueError):
            pass  # Skip plotting if data missing
        
        ax_position.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax_position.set_xlabel('Time relative to threshold crossing (s)', fontsize=10)
        ax_position.set_ylabel('Position (px)', fontsize=10)
        # Title removed per user request
        ax_position.grid(True, alpha=0.3)
        # Set Y limits (matching time series scale) - only if provided
        if y_position_limits is not None:
            ax_position.set_ylim(y_position_limits)
        
        # Plot velocity - use numpy arrays for faster plotting
        try:
            if 'Time_rel_threshold' in segment_df.columns and 'vel_x_smooth' in segment_df.columns and len(segment_df) > 0:
                ax_velocity.plot(segment_df['Time_rel_threshold'].values, segment_df['vel_x_smooth'].values,
                               'r-', linewidth=2)
        except (KeyError, IndexError, AttributeError, ValueError):
            pass  # Skip plotting if data missing
        
        ax_velocity.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax_velocity.set_xlabel('Time relative to threshold crossing (s)', fontsize=10)
        ax_velocity.set_ylabel('Velocity (px/s)', fontsize=10)
        # Title removed per user request
        ax_velocity.grid(True, alpha=0.3)
        # Set Y limits (matching time series scale) - only if provided
        if y_velocity_limits is not None:
            ax_velocity.set_ylim(y_velocity_limits)
        
        # Add saccade width bar (highlight the saccade period in the segment)
        if saccade_info is not None:
            duration = saccade_info.get('duration', 0)
            if duration > 0:
                # Saccade period is from 0 (threshold crossing) to duration
                saccade_start = 0
                saccade_end = duration
                
                # Get color for the saccade
                label_colors = {
                    'compensatory': 'orange',
                    'orienting': 'blue',
                    'saccade_and_fixate': 'green',
                    'non_saccade': 'red',
                }
                
                # Use user label if available, otherwise try segment metadata
                saccade_color = 'gray'
                if user_label:
                    saccade_color = label_colors.get(user_label, 'gray')
                elif 'saccade_type' in segment_df.columns and len(segment_df) > 0:
                    try:
                        seg_type = segment_df['saccade_type'].iloc[0]
                        if seg_type and not pd.isna(seg_type):
                            saccade_color = label_colors.get(seg_type, 'gray')
                    except (IndexError, KeyError, AttributeError):
                        pass
                
                # Highlight saccade period in position plot
                ax_position.axvspan(saccade_start, saccade_end, alpha=0.3, 
                                   facecolor=saccade_color, edgecolor=saccade_color, linewidth=1)
                
                # Highlight saccade period in velocity plot
                ax_velocity.axvspan(saccade_start, saccade_end, alpha=0.3,
                                   facecolor=saccade_color, edgecolor=saccade_color, linewidth=1)
        
        # Add saccade info text
        if saccade_info:
            info_text = f"Amplitude: {saccade_info.get('amplitude', 'N/A'):.1f} px | "
            info_text += f"Duration: {saccade_info.get('duration', 'N/A'):.3f} s | "
            info_text += f"Peak Velocity: {saccade_info.get('peak_velocity', 'N/A'):.1f} px/s"
            self.figure.suptitle(info_text, fontsize=10, y=0.98)
        
        self.canvas.draw()
    
    def clear(self):
        """Clear the plot."""
        self.figure.clear()
        self.canvas.draw()


class SaccadeAnnotationGUI(QMainWindow):
    """Main GUI window for saccade annotation."""
    
    def __init__(
        self,
        saccade_results: Dict,
        features_df: Optional[pd.DataFrame] = None,
        experiment_id: str = 'unknown',
        annotations_file_path: Union[str, Path] = 'saccade_annotations_master.csv',
        parent=None
    ):
        super().__init__(parent)
        
        if not PYQT5_AVAILABLE:
            raise ImportError("PyQt5 not available. Install with: pip install PyQt5")
        
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib not available. Install with: pip install matplotlib")
        
        self.saccade_results = saccade_results
        self.features_df = features_df
        self.experiment_id = experiment_id
        self.annotations_file_path = Path(annotations_file_path)
        
        # Get data
        self.df = saccade_results.get('df')
        self.all_saccades_df = saccade_results.get('all_saccades_df', pd.DataFrame())
        self.peri_saccades = saccade_results.get('peri_saccades', [])
        self.vel_thresh = saccade_results.get('vel_thresh', None)
        self.video_label = saccade_results.get('video_label', 'Unknown')
        
        # Eye label will be determined per-saccade from the DataFrame
        # Store default eye label for display
        self.default_eye_label = 'Unknown'
        if 'L:' in self.video_label or '(L:' in self.video_label:
            self.default_eye_label = 'L'
        elif 'R:' in self.video_label or '(R:' in self.video_label:
            self.default_eye_label = 'R'
        
        # Create combined saccades list with IDs
        self.saccades_list = []
        self._build_saccades_list()
        
        # Debug: Print summary of loaded data
        print(f"\n📊 GUI Initialized:")
        print(f"   Time series length: {len(self.df) if self.df is not None else 0}")
        print(f"   Total saccades: {len(self.saccades_list)}")
        print(f"   Total segments: {len(self.peri_saccades)}")
        if len(self.saccades_list) > 0 and 'eye' in self.saccades_list[0]:
            eye_counts = {}
            for saccade in self.saccades_list:
                eye = saccade.get('eye', 'Unknown')
                eye_counts[eye] = eye_counts.get(eye, 0) + 1
            print(f"   Saccades by eye: {dict(eye_counts)}")
        print()
        
        # Build segment cache for fast lookups (saccade_id -> segment DataFrame)
        self._segment_cache: Dict[int, pd.DataFrame] = {}
        self._build_segment_cache()
        
        # User labels (saccade_id -> user_label)
        self.user_labels: Dict[int, str] = {}
        self._load_existing_annotations()
        
        # Current selection
        self.current_index = 0
        
        # Flag to prevent recursive updates
        self._updating = False
        
        # Initialize UI
        self.init_ui()
        self.update_display()
        
    def _build_saccades_list(self):
        """Build list of saccades from all_saccades_df."""
        self.saccades_list = []
        
        if len(self.all_saccades_df) == 0:
            return
        
        # Use DataFrame index as saccade_id (all_saccades_df doesn't have a saccade_id column)
        for idx, row in self.all_saccades_df.iterrows():
            # Use the DataFrame index as the saccade_id
            saccade_id = int(idx)  # Convert to int for consistency
            
            # Get original_index for matching with segments
            # original_index should match the segment's saccade_id (from upward/downward DataFrames)
            if 'original_index' in row:
                original_index = row['original_index']
                # Convert to int if it's not already
                if pd.notna(original_index):
                    original_index = int(original_index)
                else:
                    original_index = idx
            else:
                # Fallback: use current index if original_index not available
                original_index = idx
            
            # Get eye label from DataFrame (if available), otherwise use default
            eye_label = row.get('eye', self.default_eye_label)
            if pd.isna(eye_label):
                eye_label = self.default_eye_label
            
            self.saccades_list.append({
                'saccade_id': saccade_id,
                'original_index': original_index,  # Store for segment matching
                'time': row.get('time', row.get('start_time', 0)),
                'start_time': row.get('start_time', row.get('time', 0)),  # Store start_time for time-based matching
                'amplitude': row.get('amplitude', 0),
                'duration': row.get('duration', 0),
                'peak_velocity': row.get('peak_velocity', row.get('velocity', 0)),
                'rule_based_label': row.get('saccade_type', 'unknown'),
                'rule_based_confidence': row.get('classification_confidence', 0.5),
                'direction': row.get('saccade_direction', row.get('direction', 'unknown')),
                'eye': eye_label,  # Store eye label (L or R)
                'index': idx  # Current DataFrame index in all_saccades_df
            })
        
        # Sort by time
        self.saccades_list.sort(key=lambda x: x['time'])
    
    def _build_segment_cache(self):
        """Build a cache dictionary mapping saccade_id to segment DataFrame for fast lookups."""
        self._segment_cache = {}
        self._segment_time_cache = {}  # Cache for time-based matching: time -> segment
        if len(self.peri_saccades) == 0:
            return
        
        for seg in self.peri_saccades:
            try:
                if 'saccade_id' in seg.columns and len(seg) > 0:
                    seg_saccade_id_val = seg['saccade_id'].iloc[0]
                    if pd.notna(seg_saccade_id_val):
                        seg_saccade_id = int(seg_saccade_id_val)
                        # Get eye label from segment (if available)
                        seg_eye = seg.get('eye', None)
                        if seg_eye is not None and len(seg) > 0:
                            if 'eye' in seg.columns:
                                seg_eye_val = seg['eye'].iloc[0]
                                if pd.notna(seg_eye_val):
                                    seg_eye = str(seg_eye_val)
                        
                        # Cache by saccade_id (original index from upward/downward DataFrames)
                        # Use composite key (eye, saccade_id) if eye is available to avoid collisions
                        if seg_eye is not None:
                            cache_key = (seg_eye, seg_saccade_id)
                        else:
                            cache_key = seg_saccade_id
                        self._segment_cache[cache_key] = seg
                        # Also cache by saccade_id alone for backward compatibility
                        self._segment_cache[seg_saccade_id] = seg
                        
                        # Also cache by time for primary matching
                        # Get the threshold crossing time from segment metadata
                        # Segments have Time_rel_threshold = 0 at threshold crossing (start_time)
                        # We need to find the absolute time at threshold crossing
                        if 'Seconds' in seg.columns and 'Time_rel_threshold' in seg.columns:
                            # Find where Time_rel_threshold is closest to 0 (threshold crossing)
                            time_rel_values = seg['Time_rel_threshold'].values
                            threshold_idx = np.argmin(np.abs(time_rel_values))
                            if threshold_idx < len(seg):
                                threshold_time = seg['Seconds'].iloc[threshold_idx]
                                # Also check if we can get start_time from segment metadata
                                # Some segments might have start_time stored directly
                                if 'start_time' in seg.columns:
                                    seg_start_time = seg['start_time'].iloc[0]
                                    if pd.notna(seg_start_time):
                                        threshold_time = seg_start_time
                                
                                # Round to nearest 0.001s for matching (1ms precision)
                                threshold_time_key = round(threshold_time, 3)
                                # Use composite key (eye, time) if eye is available to avoid collisions
                                if seg_eye is not None:
                                    time_cache_key = (seg_eye, threshold_time_key)
                                else:
                                    time_cache_key = threshold_time_key
                                # Store segment - if multiple segments have same time+eye, last one wins
                                self._segment_time_cache[time_cache_key] = seg
            except (IndexError, KeyError, ValueError, AttributeError):
                # Skip segments that can't be cached
                continue
    
    def _load_existing_annotations(self):
        """Load existing annotations for this experiment."""
        try:
            existing_annotations = load_annotations(self.annotations_file_path, experiment_id=self.experiment_id)
            for _, row in existing_annotations.iterrows():
                saccade_id = int(row['saccade_id'])
                user_label = row['user_label']
                self.user_labels[saccade_id] = user_label
        except Exception as e:
            print(f"⚠️ Could not load existing annotations: {e}")
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle(f"Saccade Annotation Tool - {self.experiment_id}")
        self.setGeometry(100, 100, 1000, 900)  # Narrower: 1400 -> 1000
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel: Saccade list and controls
        left_panel = self._create_left_panel()
        
        # Right panel: Plots
        right_panel = self._create_right_panel()
        
        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage(f"Ready - {len(self.saccades_list)} saccades loaded")
        
    def _create_left_panel(self) -> QWidget:
        """Create left panel with saccade list and controls."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Title
        title = QLabel(f"<h3>Saccades ({len(self.saccades_list)})</h3>")
        layout.addWidget(title)
        
        # Statistics
        stats_label = QLabel()
        self.stats_label = stats_label
        layout.addWidget(stats_label)
        
        # Saccade table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(['ID', 'Eye', 'Time', 'Amplitude', 'Duration', 'Rule-Based', 'User Label'])
        # Set all columns to auto-resize
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        layout.addWidget(self.table)
        
        # Classification buttons
        button_group = QGroupBox("Classification")
        button_layout = QVBoxLayout()
        
        # Color mapping for buttons (same as plot colors)
        button_colors = {
            'compensatory': 'orange',
            'orienting': 'blue',
            'saccade_and_fixate': 'green',
            'non_saccade': 'red'
        }
        
        self.btn_compensatory = QPushButton("1. Compensatory")
        self.btn_orienting = QPushButton("2. Orienting")
        self.btn_saccade_fixate = QPushButton("3. Saccade-and-Fixate")
        self.btn_non_saccade = QPushButton("4. Non-Saccade")
        
        # Set button colors
        self.btn_compensatory.setStyleSheet(f"background-color: {button_colors['compensatory']}; color: white; font-weight: bold;")
        self.btn_orienting.setStyleSheet(f"background-color: {button_colors['orienting']}; color: white; font-weight: bold;")
        self.btn_saccade_fixate.setStyleSheet(f"background-color: {button_colors['saccade_and_fixate']}; color: white; font-weight: bold;")
        self.btn_non_saccade.setStyleSheet(f"background-color: {button_colors['non_saccade']}; color: white; font-weight: bold;")
        
        self.btn_compensatory.clicked.connect(lambda: self._classify_saccade('compensatory'))
        self.btn_orienting.clicked.connect(lambda: self._classify_saccade('orienting'))
        self.btn_saccade_fixate.clicked.connect(lambda: self._classify_saccade('saccade_and_fixate'))
        self.btn_non_saccade.clicked.connect(lambda: self._classify_saccade('non_saccade'))
        
        button_layout.addWidget(self.btn_compensatory)
        button_layout.addWidget(self.btn_orienting)
        button_layout.addWidget(self.btn_saccade_fixate)
        button_layout.addWidget(self.btn_non_saccade)
        
        button_group.setLayout(button_layout)
        layout.addWidget(button_group)
        
        # Navigation buttons
        nav_group = QGroupBox("Navigation")
        nav_layout = QHBoxLayout()
        
        self.btn_previous = QPushButton("◄ Previous (P)")
        self.btn_next = QPushButton("Next (N) ►")
        self.btn_save = QPushButton("Save (S)")
        self.btn_exit = QPushButton("Exit")
        
        self.btn_previous.clicked.connect(self._previous_saccade)
        self.btn_next.clicked.connect(self._next_saccade)
        self.btn_save.clicked.connect(self._save_current_annotation)
        self.btn_exit.clicked.connect(self._exit_gui)
        
        nav_layout.addWidget(self.btn_previous)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addWidget(self.btn_save)
        nav_layout.addWidget(self.btn_exit)
        
        nav_group.setLayout(nav_layout)
        layout.addWidget(nav_group)
        
        # Notes
        notes_label = QLabel("Notes:")
        layout.addWidget(notes_label)
        self.notes_text = QTextEdit()
        self.notes_text.setMaximumHeight(80)
        layout.addWidget(self.notes_text)
        
        layout.addStretch()
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create right panel with plots."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Time series plot
        time_series_label = QLabel("<b>Time Series</b>")
        layout.addWidget(time_series_label)
        self.time_series_plot = TimeSeriesPlotWidget()
        layout.addWidget(self.time_series_plot)
        
        # Peri-saccade plot
        peri_label = QLabel("<b>Peri-Saccade Segment</b>")
        layout.addWidget(peri_label)
        self.peri_plot = PeriSaccadePlotWidget()
        layout.addWidget(self.peri_plot)
        
        return panel
    
    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Classification shortcuts
        QShortcut(QKeySequence("1"), self, lambda: self._classify_saccade('compensatory'))
        QShortcut(QKeySequence("2"), self, lambda: self._classify_saccade('orienting'))
        QShortcut(QKeySequence("3"), self, lambda: self._classify_saccade('saccade_and_fixate'))
        QShortcut(QKeySequence("4"), self, lambda: self._classify_saccade('non_saccade'))
        
        # Navigation shortcuts
        QShortcut(QKeySequence("."), self, self._next_saccade)  # Period for next
        QShortcut(QKeySequence(","), self, self._previous_saccade)  # Comma for previous
        QShortcut(QKeySequence("S"), self, self._save_current_annotation)
        
        # Exit shortcuts
        QShortcut(QKeySequence("Escape"), self, self._exit_gui)
        QShortcut(QKeySequence("Q"), self, self._exit_gui)
    
    def _update_table(self):
        """Update the saccade table."""
        # Block signals during update to prevent cascading updates
        self.table.blockSignals(True)
        
        # Only update if row count changed or this is first update
        if self.table.rowCount() != len(self.saccades_list):
            self.table.setRowCount(len(self.saccades_list))
        
        # Update all rows (necessary for user label changes)
        for i, saccade in enumerate(self.saccades_list):
            saccade_id = saccade['saccade_id']
            user_label = self.user_labels.get(saccade_id, '')
            
            # Get eye label from saccade dict
            eye_label = saccade.get('eye', self.default_eye_label)
            
            # Only update items if they don't exist or need updating
            if self.table.item(i, 0) is None:
                self.table.setItem(i, 0, QTableWidgetItem(str(saccade_id)))
                self.table.setItem(i, 1, QTableWidgetItem(str(eye_label)))
                self.table.setItem(i, 2, QTableWidgetItem(f"{saccade['time']:.2f}"))
                self.table.setItem(i, 3, QTableWidgetItem(f"{saccade['amplitude']:.1f}"))
                self.table.setItem(i, 4, QTableWidgetItem(f"{saccade['duration']:.3f}"))
                self.table.setItem(i, 5, QTableWidgetItem(saccade['rule_based_label']))
                self.table.setItem(i, 6, QTableWidgetItem(''))
            else:
                # Update eye label and user label columns if needed
                eye_item = self.table.item(i, 1)
                if eye_item is None:
                    eye_item = QTableWidgetItem(str(eye_label))
                    self.table.setItem(i, 1, eye_item)
                else:
                    eye_item.setText(str(eye_label))
                
                # Update only user label column (most likely to change)
                user_label_item = self.table.item(i, 6)
                if user_label_item is None:
                    user_label_item = QTableWidgetItem('')
                    self.table.setItem(i, 6, user_label_item)
                user_label_item.setText(user_label if user_label else 'Unlabeled')
                if user_label:
                    user_label_item.setBackground(Qt.green)
                else:
                    user_label_item.setBackground(Qt.white)
        
        # Select current row (only if not already selected to avoid unnecessary updates)
        if 0 <= self.current_index < len(self.saccades_list):
            try:
                current_selection = self.table.selectedIndexes()
                if not current_selection or current_selection[0].row() != self.current_index:
                    self.table.selectRow(self.current_index)
                    item = self.table.item(self.current_index, 0)
                    if item is not None:
                        # Scroll to item (use default behavior to prevent freezing)
                        self.table.scrollToItem(item)
            except (IndexError, AttributeError):
                # Silently handle selection errors
                pass
        
        # Re-enable signals
        self.table.blockSignals(False)
    
    def _update_statistics(self):
        """Update statistics display."""
        total = len(self.saccades_list)
        annotated = len(self.user_labels)
        remaining = total - annotated
        
        stats_text = f"<b>Statistics:</b><br>"
        stats_text += f"Total: {total} | Annotated: {annotated} | Remaining: {remaining}"
        
        # Show breakdown by eye if available
        if len(self.saccades_list) > 0 and 'eye' in self.saccades_list[0]:
            eye_counts = {}
            eye_annotated = {}
            for saccade in self.saccades_list:
                eye = saccade.get('eye', 'Unknown')
                eye_counts[eye] = eye_counts.get(eye, 0) + 1
                if saccade['saccade_id'] in self.user_labels:
                    eye_annotated[eye] = eye_annotated.get(eye, 0) + 1
            
            stats_text += "<br><b>By Eye:</b> "
            eye_stats = []
            for eye in sorted(eye_counts.keys()):
                count = eye_counts[eye]
                ann = eye_annotated.get(eye, 0)
                eye_stats.append(f"{eye}: {count} (annotated: {ann})")
            stats_text += " | ".join(eye_stats)
        
        if annotated > 0:
            by_class = {}
            for label in self.user_labels.values():
                by_class[label] = by_class.get(label, 0) + 1
            
            stats_text += "<br><b>By Class:</b> "
            stats_text += " | ".join([f"{k}: {v}" for k, v in sorted(by_class.items())])
        
        self.stats_label.setText(stats_text)
    
    def _update_plots(self):
        """Update the plots."""
        if self.current_index < 0 or self.current_index >= len(self.saccades_list):
            return
        
        current_saccade = self.saccades_list[self.current_index]
        saccade_id = current_saccade['saccade_id']
        original_index = current_saccade.get('original_index', saccade_id)  # Use stored original_index
        df_index = current_saccade['index']  # Current DataFrame index in all_saccades_df
        
        # Update time series plot and get Y limits
        y_position_limits = None
        y_velocity_limits = None
        try:
            if self.df is not None and len(self.df) > 0 and len(self.all_saccades_df) > 0:
                y_position_limits, y_velocity_limits = self.time_series_plot.plot_time_series(
                    self.df,
                    self.all_saccades_df,
                    selected_saccade_id=saccade_id,
                    user_labels=self.user_labels,
                    window_duration=6.0,  # ±3 seconds
                    vel_thresh=self.vel_thresh
                )
        except (KeyError, IndexError, AttributeError, ValueError) as e:
            # Log error but don't crash
            print(f"⚠️ Error updating time series plot: {e}")
            # Clear plots on error
            self.time_series_plot.clear()
        
        # Update peri-saccade plot
        if len(self.peri_saccades) > 0:
            # Find segment for current saccade
            # CRITICAL: Segments have saccade_id matching the original DataFrame index from upward/downward DataFrames
            # The original_index in all_saccades_df should match the segment's saccade_id
            # However, we should ALWAYS use time-based matching as the primary method since it's most reliable
            segment = None
            
            # PRIMARY METHOD: Match by time (most reliable)
            # Get start_time from DataFrame
            current_start_time = None
            try:
                if df_index in self.all_saccades_df.index:
                    row = self.all_saccades_df.loc[df_index]
                    current_start_time = row.get('start_time', row.get('time', None))
            except (KeyError, IndexError):
                pass
            
            # Fallback to current_saccade dict
            if current_start_time is None:
                current_start_time = current_saccade.get('start_time', current_saccade.get('time', None))
            
            if current_start_time is not None and hasattr(self, '_segment_time_cache'):
                # Get current saccade's eye label for matching
                current_eye = current_saccade.get('eye', None)
                
                # Try exact match first (rounded to 1ms precision)
                time_key = round(current_start_time, 3)
                
                # Try composite key (eye, time) first if eye is available
                if current_eye is not None:
                    composite_key = (current_eye, time_key)
                    if composite_key in self._segment_time_cache:
                        candidate_seg = self._segment_time_cache[composite_key]
                        # Verify the match
                        if candidate_seg is not None and len(candidate_seg) > 0 and 'saccade_id' in candidate_seg.columns:
                            seg_saccade_id = int(candidate_seg['saccade_id'].iloc[0])
                            # If original_index matches, this is a perfect match - use it immediately
                            if original_index is not None and seg_saccade_id == original_index:
                                segment = candidate_seg
                            elif original_index is None:
                                # No original_index to verify - use the time match
                                segment = candidate_seg
                
                # Fallback to time-only key if composite key didn't work
                if segment is None and time_key in self._segment_time_cache:
                    candidate_seg = self._segment_time_cache[time_key]
                    # Verify the match by checking the segment's saccade_id and eye match expected
                    if candidate_seg is not None and len(candidate_seg) > 0 and 'saccade_id' in candidate_seg.columns:
                        seg_saccade_id = int(candidate_seg['saccade_id'].iloc[0])
                        # Check if eye matches (if available)
                        seg_eye = None
                        if 'eye' in candidate_seg.columns:
                            seg_eye_val = candidate_seg['eye'].iloc[0]
                            if pd.notna(seg_eye_val):
                                seg_eye = str(seg_eye_val)
                        
                        eye_matches = (current_eye is None or seg_eye is None or current_eye == seg_eye)
                        
                        # If original_index matches and eye matches, this is a perfect match
                        if original_index is not None and seg_saccade_id == original_index and eye_matches:
                            segment = candidate_seg
                        # If no original_index but eye matches, use the time match
                        elif original_index is None and eye_matches:
                            segment = candidate_seg
                
                # If exact match failed or was invalid, try tolerance-based matching
                # Only do this if we don't have a segment yet
                if segment is None and len(self._segment_time_cache) > 0:
                    time_tolerance = 0.005  # 5ms tolerance (tighter than before)
                    best_match = None
                    best_time_diff = float('inf')
                    best_saccade_id_match = False
                    
                    # Limit search to reasonable number of candidates to prevent freezing
                    # Sort by time difference first to check closest matches first
                    candidates = []
                    for cached_time_key, seg in self._segment_time_cache.items():
                        # Handle composite keys (eye, time) or simple time keys
                        if isinstance(cached_time_key, tuple):
                            cached_eye, cached_time = cached_time_key
                            time_diff = abs(cached_time - current_start_time)
                            # Prefer candidates from same eye
                            eye_match = (current_eye is not None and cached_eye == current_eye)
                        else:
                            time_diff = abs(cached_time_key - current_start_time)
                            eye_match = False  # Unknown eye match
                        
                        if time_diff < time_tolerance:
                            candidates.append((time_diff, cached_time_key, seg, eye_match))
                    
                    # Sort by time difference (closest first), but prioritize eye matches
                    candidates.sort(key=lambda x: (not x[3], x[0]))  # eye_match first (True before False), then time_diff
                    
                    # Check candidates, prioritizing saccade_id and eye matches
                    for time_diff, cached_time_key, seg, eye_match in candidates:
                        # Check if saccade_id matches original_index (preferred)
                        seg_saccade_id = None
                        if len(seg) > 0 and 'saccade_id' in seg.columns:
                            seg_saccade_id = int(seg['saccade_id'].iloc[0])
                        
                        # Get segment eye
                        seg_eye = None
                        if len(seg) > 0 and 'eye' in seg.columns:
                            seg_eye_val = seg['eye'].iloc[0]
                            if pd.notna(seg_eye_val):
                                seg_eye = str(seg_eye_val)
                        
                        # Check eye match
                        eye_matches = (current_eye is None or seg_eye is None or current_eye == seg_eye)
                        
                        # Prefer matches where saccade_id matches original_index AND eye matches
                        saccade_id_matches = (original_index is not None and 
                                             seg_saccade_id is not None and 
                                             seg_saccade_id == original_index)
                        
                        # If we find a perfect match (saccade_id + eye + time), use it immediately
                        if saccade_id_matches and eye_matches:
                            best_match = seg
                            best_time_diff = time_diff
                            best_saccade_id_match = True
                            break  # Early exit - found perfect match
                        
                        # Otherwise, track best match (prefer eye match + saccade_id match)
                        if not best_saccade_id_match:
                            if eye_matches and saccade_id_matches:
                                # Eye + saccade_id match (but not both perfect)
                                best_match = seg
                                best_time_diff = time_diff
                            elif eye_matches and time_diff < best_time_diff:
                                # Eye match, better time
                                best_match = seg
                                best_time_diff = time_diff
                            elif not eye_matches and time_diff < best_time_diff:
                                # No eye match but better time (fallback)
                                best_match = seg
                                best_time_diff = time_diff
                    
                    if best_match is not None:
                        segment = best_match
            
            # FALLBACK: Try matching by original_index with eye (should match segment's saccade_id)
            if segment is None and original_index is not None:
                current_eye = current_saccade.get('eye', None)
                # Try composite key first
                if current_eye is not None:
                    composite_key = (current_eye, original_index)
                    segment = self._segment_cache.get(composite_key)
                # Fallback to original_index alone
                if segment is None:
                    segment = self._segment_cache.get(original_index)
            
            # FALLBACK: Try matching by current saccade_id (current DataFrame index)
            if segment is None:
                segment = self._segment_cache.get(saccade_id)
            
            if segment is not None:
                # Get saccade info directly from current_saccade (already cached)
                saccade_info = {
                    'amplitude': current_saccade.get('amplitude', 0),
                    'duration': current_saccade.get('duration', 0),
                    'peak_velocity': current_saccade.get('peak_velocity', 0)
                }
                
                # Only lookup peak_velocity from DataFrame if not already in current_saccade
                try:
                    if saccade_info['peak_velocity'] == 0 and df_index in self.all_saccades_df.index:
                        row = self.all_saccades_df.loc[df_index]
                        if 'peak_velocity' in row:
                            peak_vel = row['peak_velocity']
                            saccade_info['peak_velocity'] = peak_vel if not pd.isna(peak_vel) else 0
                        elif 'velocity' in row:
                            peak_vel = row['velocity']
                            saccade_info['peak_velocity'] = peak_vel if not pd.isna(peak_vel) else 0
                except (KeyError, IndexError, AttributeError):
                    # Keep default value on error
                    pass
                
                # Use Y limits from time series plot (for matching scale)
                # Also pass user label for color coding
                try:
                    user_label = self.user_labels.get(saccade_id) if saccade_id in self.user_labels else None
                    self.peri_plot.plot_segment(segment, saccade_info, 
                                              y_position_limits=y_position_limits,
                                              y_velocity_limits=y_velocity_limits,
                                              user_label=user_label,
                                              vel_thresh=self.vel_thresh)
                except (KeyError, IndexError, AttributeError, ValueError) as e:
                    # Log error but don't crash
                    print(f"⚠️ Error updating peri-saccade plot: {e}")
                    self.peri_plot.clear()
            else:
                self.peri_plot.clear()
        else:
            self.peri_plot.clear()
    
    def update_display(self):
        """Update all displays."""
        # Prevent recursive updates
        if self._updating:
            return
        self._updating = True
        try:
            self._update_table()
            self._update_statistics()
            self._update_plots()
        finally:
            self._updating = False
    
    def _on_table_selection_changed(self):
        """Handle table selection change."""
        selected_rows = self.table.selectedIndexes()
        if selected_rows:
            self.current_index = selected_rows[0].row()
            self.update_display()
    
    def _classify_saccade(self, label: str):
        """Classify current saccade."""
        if self.current_index < 0 or self.current_index >= len(self.saccades_list):
            return
        
        current_saccade = self.saccades_list[self.current_index]
        saccade_id = current_saccade['saccade_id']
        
        self.user_labels[saccade_id] = label
        self.update_display()
        
        # Auto-advance to next
        self._next_saccade()
    
    def _next_saccade(self):
        """Move to next saccade."""
        if self.current_index < len(self.saccades_list) - 1:
            self.current_index += 1
            self.update_display()
    
    def _previous_saccade(self):
        """Move to previous saccade."""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
    
    def _save_current_annotation(self):
        """Save current annotation."""
        if self.current_index < 0 or self.current_index >= len(self.saccades_list):
            return
        
        current_saccade = self.saccades_list[self.current_index]
        saccade_id = current_saccade['saccade_id']
        user_label = self.user_labels.get(saccade_id)
        
        if user_label is None:
            QMessageBox.warning(self, "No Label", "Please classify this saccade before saving.")
            return
        
        # Get notes
        notes = self.notes_text.toPlainText()
        
        # Save annotation
        try:
            save_annotation(
                annotations_file_path=self.annotations_file_path,
                experiment_id=self.experiment_id,
                saccade_id=saccade_id,
                user_label=user_label,
                time=current_saccade['time'],
                amplitude=current_saccade['amplitude'],
                duration=current_saccade['duration'],
                user_confidence=1.0,
                notes=notes
            )
            
            self.statusBar().showMessage(f"✅ Saved annotation: {saccade_id} -> {user_label}", 3000)
            self.notes_text.clear()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save annotation: {e}")
    
    def _exit_gui(self):
        """Exit the GUI application."""
        # Save any unsaved annotations before closing?
        # For now, just close the window
        self.close()
    
    def closeEvent(self, event):
        """Handle window close event."""
        global _active_gui_window
        # Clear the active window reference when closing
        if _active_gui_window is self:
            _active_gui_window = None
        # Save any unsaved annotations before closing?
        # For now, just close the window
        event.accept()


def launch_annotation_gui(
    saccade_results: Union[Dict, Dict[str, Dict]],
    features_df: Optional[pd.DataFrame] = None,
    experiment_id: str = 'unknown',
    annotations_file_path: Union[str, Path] = 'saccade_annotations_master.csv'
):
    """
    Launch the annotation GUI.
    
    Parameters
    ----------
    saccade_results : dict or dict of dicts
        Either:
        - Single eye: Dictionary from analyze_eye_video_saccades() containing:
          - 'df': Full time series DataFrame
          - 'all_saccades_df': DataFrame with all saccades
          - 'peri_saccades': List of peri-saccade segment DataFrames
          - 'video_label': Label for the video/eye
        - Both eyes: Dictionary with keys 'VideoData1' and/or 'VideoData2', each containing
          the above structure
    features_df : pd.DataFrame, optional
        Features DataFrame from extract_ml_features() (for future use)
    experiment_id : str
        Unique experiment identifier
    annotations_file_path : str or Path
        Path to master annotations CSV file
    """
    global _active_gui_window
    
    if not PYQT5_AVAILABLE:
        raise ImportError("PyQt5 not available. Install with: pip install PyQt5")
    
    # Check if a GUI window is already open
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Check if there's an active window and if it's still valid
    if _active_gui_window is not None:
        # Check if the window still exists and is visible
        try:
            if _active_gui_window.isVisible():
                # Show prominent warning dialog
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("⚠️ GUI Already Open")
                msg.setText("⚠️ WARNING: A Saccade Annotation GUI window is already open!")
                msg.setInformativeText(
                    "Please save your progress and close the existing GUI window before opening a new one.\n\n"
                    "Opening multiple GUI windows can cause conflicts and data loss."
                )
                msg.setStandardButtons(QMessageBox.Ok)
                msg.setDefaultButton(QMessageBox.Ok)
                
                # Make the dialog more prominent and ensure it's on top
                msg.setWindowFlags(msg.windowFlags() | Qt.WindowStaysOnTopHint)
                msg.setStyleSheet("""
                    QMessageBox {
                        font-size: 14px;
                        min-width: 500px;
                        background-color: #fff3cd;
                    }
                    QMessageBox QLabel {
                        font-size: 14px;
                        font-weight: bold;
                        color: #856404;
                    }
                    QPushButton {
                        font-size: 12px;
                        font-weight: bold;
                        min-width: 80px;
                        min-height: 30px;
                        background-color: #ffc107;
                        color: #000;
                    }
                    QPushButton:hover {
                        background-color: #ffca2c;
                    }
                """)
                
                # Bring existing window to front as well
                try:
                    _active_gui_window.raise_()
                    _active_gui_window.activateWindow()
                except (RuntimeError, AttributeError):
                    pass
                
                msg.exec_()
                return  # Don't open a new window
        except (RuntimeError, AttributeError):
            # Window was closed or invalid, clear the reference
            _active_gui_window = None
    
    # Debug: Check what structure we received
    print(f"\n🔍 Checking saccade_results structure...")
    print(f"   Type: {type(saccade_results)}")
    print(f"   Keys in saccade_results: {list(saccade_results.keys()) if isinstance(saccade_results, dict) else 'Not a dict'}")
    
    # Check if saccade_results contains multiple eyes (VideoData1/VideoData2 keys)
    # or is a single eye's results
    if isinstance(saccade_results, dict) and ('VideoData1' in saccade_results or 'VideoData2' in saccade_results):
        # Multiple eyes - combine them
        print(f"   ✅ Detected multiple eyes structure - combining...")
        combined_results = _combine_eyes_saccade_results(saccade_results)
    elif isinstance(saccade_results, dict) and 'df' in saccade_results and 'all_saccades_df' in saccade_results:
        # Single eye's results structure
        print(f"   ⚠️  Detected single eye structure")
        print(f"   ℹ️  To combine both eyes, pass the full saccade_results dict with 'VideoData1' and 'VideoData2' keys")
        print(f"   ℹ️  Example: launch_annotation_gui(saccade_results=saccade_results, ...) where saccade_results = {{'VideoData1': {...}, 'VideoData2': {...}}}")
        combined_results = saccade_results
    else:
        # Unknown structure - use as is
        print(f"   ⚠️  Unknown structure - using as is")
        combined_results = saccade_results
    
    window = SaccadeAnnotationGUI(
        saccade_results=combined_results,
        features_df=features_df,
        experiment_id=experiment_id,
        annotations_file_path=annotations_file_path
    )
    
    # Store reference to active window
    _active_gui_window = window
    
    window._setup_keyboard_shortcuts()
    window.show()
    
    sys.exit(app.exec_())


def _combine_eyes_saccade_results(saccade_results_dict: Dict[str, Dict]) -> Dict:
    """
    Combine saccade results from multiple eyes (VideoData1 and VideoData2).
    
    Parameters
    ----------
    saccade_results_dict : dict
        Dictionary with keys 'VideoData1' and/or 'VideoData2', each containing
        saccade results from analyze_eye_video_saccades()
    
    Returns
    -------
    dict
        Combined saccade results with:
        - 'df': Combined time series (if both eyes have same timebase, use first)
        - 'all_saccades_df': Combined DataFrame with all saccades from both eyes
        - 'peri_saccades': Combined list of peri-saccade segments
        - 'vel_thresh': Average threshold (or first available)
        - 'video_label': Combined label
    """
    combined = {
        'df': None,
        'all_saccades_df': pd.DataFrame(),
        'peri_saccades': [],
        'vel_thresh': None,
        'video_label': 'Combined'
    }
    
    eye_results = []
    if 'VideoData1' in saccade_results_dict:
        eye_results.append(('VideoData1', saccade_results_dict['VideoData1']))
    if 'VideoData2' in saccade_results_dict:
        eye_results.append(('VideoData2', saccade_results_dict['VideoData2']))
    
    if len(eye_results) == 0:
        raise ValueError("No valid eye data found in saccade_results_dict")
    
    # Combine DataFrames and segments
    all_saccades_dfs = []
    all_peri_saccades = []
    
    for eye_key, eye_data in eye_results:
        # Get eye label
        video_label = eye_data.get('video_label', eye_key)
        eye_label = 'Unknown'
        if 'L:' in video_label or '(L:' in video_label:
            eye_label = 'L'
        elif 'R:' in video_label or '(R:' in video_label:
            eye_label = 'R'
        
        # Add eye label column to saccades DataFrame
        # IMPORTANT: Preserve original_index for segment matching
        if 'all_saccades_df' in eye_data and len(eye_data['all_saccades_df']) > 0:
            saccades_df = eye_data['all_saccades_df'].copy()
            saccades_df['eye'] = eye_label
            saccades_df['eye_source'] = eye_key
            # Ensure original_index exists (for segment matching)
            if 'original_index' not in saccades_df.columns:
                # Use current index as original_index if not present
                saccades_df['original_index'] = saccades_df.index
            all_saccades_dfs.append(saccades_df)
        
        # Add segments with eye tracking
        if 'peri_saccades' in eye_data:
            for seg in eye_data['peri_saccades']:
                if len(seg) > 0:
                    seg_copy = seg.copy()
                    # Add eye label to segment as a column (DataFrame, not dict)
                    seg_copy['eye'] = eye_label
                    seg_copy['eye_source'] = eye_key
                    all_peri_saccades.append(seg_copy)
        
        # Use first eye's time series and threshold (assuming same timebase)
        if combined['df'] is None:
            combined['df'] = eye_data.get('df')
        if combined['vel_thresh'] is None:
            combined['vel_thresh'] = eye_data.get('vel_thresh')
    
    # Combine all saccades DataFrames
    if len(all_saccades_dfs) > 0:
        # Concatenate but preserve original_index for segment matching
        combined['all_saccades_df'] = pd.concat(all_saccades_dfs, ignore_index=True)
        # Sort by time first
        if 'time' in combined['all_saccades_df'].columns:
            combined['all_saccades_df'] = combined['all_saccades_df'].sort_values('time').reset_index(drop=True)
        # Now reset index to create new sequential IDs (this becomes the new saccade_id)
        # But keep original_index for segment matching
        # The original_index should match the segment's saccade_id from the original eye's DataFrame
    
    combined['peri_saccades'] = all_peri_saccades
    
    # Create combined label
    eye_labels = [eye_data.get('video_label', eye_key) for eye_key, eye_data in eye_results]
    combined['video_label'] = ' | '.join(eye_labels)
    
    # Debug output
    print(f"✅ Combined {len(eye_results)} eye(s): {[eye_key for eye_key, _ in eye_results]}")
    print(f"   Total saccades: {len(combined['all_saccades_df'])}")
    print(f"   Total segments: {len(combined['peri_saccades'])}")
    if 'eye' in combined['all_saccades_df'].columns:
        eye_counts = combined['all_saccades_df']['eye'].value_counts()
        print(f"   By eye: {dict(eye_counts)}")
    
    return combined


if __name__ == '__main__':
    # Test with dummy data
    print("Creating test GUI...")
    
    # Create dummy saccade_results
    import numpy as np
    
    # Dummy time series
    t = np.linspace(0, 100, 1000)
    position = np.cumsum(np.random.randn(1000) * 0.5)
    velocity = np.diff(position, prepend=0) * 10
    
    df = pd.DataFrame({
        'Seconds': t,
        'X_smooth': position,
        'vel_x_smooth': velocity
    })
    
    # Dummy saccades
    saccades = []
    for i in range(10):
        time = np.random.uniform(10, 90)
        saccades.append({
            'saccade_id': i + 1,
            'time': time,
            'start_time': time - 0.05,
            'end_time': time + 0.1,
            'amplitude': np.random.uniform(20, 60),
            'duration': np.random.uniform(0.05, 0.15),
            'peak_velocity': np.random.uniform(100, 300),
            'saccade_type': 'compensatory' if i < 5 else 'orienting',
            'classification_confidence': np.random.uniform(0.7, 0.95),
            'saccade_direction': 'upward' if i % 2 == 0 else 'downward'
        })
    
    all_saccades_df = pd.DataFrame(saccades)
    
    # Dummy peri-saccade segments
    peri_saccades = []
    for i, saccade in enumerate(saccades):
        t_rel = np.linspace(-0.15, 0.5, 65)
        segment = pd.DataFrame({
            'Time_rel_threshold': t_rel,
            'X_smooth_baselined': np.random.randn(65) * 5,
            'vel_x_smooth': np.random.randn(65) * 50,
            'saccade_id': saccade['saccade_id']
        })
        peri_saccades.append(segment)
    
    saccade_results = {
        'df': df,
        'all_saccades_df': all_saccades_df,
        'peri_saccades': peri_saccades
    }
    
    # Launch GUI
    launch_annotation_gui(
        saccade_results=saccade_results,
        experiment_id='test_experiment',
        annotations_file_path='test_annotations.csv'
    )

