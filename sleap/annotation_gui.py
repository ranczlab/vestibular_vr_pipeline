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
            if selected_saccade_id in saccades_df.index:
                selected_row = saccades_df.loc[[selected_saccade_id]]
                if len(selected_row) > 0:
                    center_time = selected_row.iloc[0].get('time', selected_row.iloc[0].get('start_time', 0))
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
        else:
            # No selected saccade: show first 6 seconds
            window_start = df['Seconds'].min()
            window_end = min(df['Seconds'].max(), window_start + window_duration)
        
        # Filter data to window
        window_mask = (df['Seconds'] >= window_start) & (df['Seconds'] <= window_end)
        df_window = df[window_mask].copy()
        
        if len(df_window) == 0:
            # If window is empty, use full data
            df_window = df.copy()
            window_start = df['Seconds'].min()
            window_end = df['Seconds'].max()
        
        # Calculate local Y limits from windowed data (autoscaling)
        y_position_min = df_window['X_smooth'].min()
        y_position_max = df_window['X_smooth'].max()
        y_velocity_min = df_window['vel_x_smooth'].min()
        y_velocity_max = df_window['vel_x_smooth'].max()
        
        # Add small padding (5%)
        pos_range = y_position_max - y_position_min
        vel_range = y_velocity_max - y_velocity_min
        y_position_limits = (y_position_min - 0.05 * pos_range, y_position_max + 0.05 * pos_range)
        y_velocity_limits = (y_velocity_min - 0.05 * vel_range, y_velocity_max + 0.05 * vel_range)
        
        # Plot position (thicker line)
        self.ax_position.plot(df_window['Seconds'], df_window['X_smooth'], 'b-', linewidth=2.0, alpha=0.7)
        self.ax_position.set_ylabel('Position (px)', fontsize=10)
        self.ax_position.set_title(f'Time Series: Position and Velocity (Window: {window_start:.1f}s - {window_end:.1f}s)', fontsize=12, fontweight='bold')
        self.ax_position.grid(True, alpha=0.3)
        # Set local Y limits (autoscaling)
        self.ax_position.set_ylim(y_position_limits)
        
        # Plot velocity (thicker line)
        self.ax_velocity.plot(df_window['Seconds'], df_window['vel_x_smooth'], 'r-', linewidth=2.0, alpha=0.7)
        self.ax_velocity.set_xlabel('Time (s)', fontsize=10)
        self.ax_velocity.set_ylabel('Velocity (px/s)', fontsize=10)
        self.ax_velocity.grid(True, alpha=0.3)
        # Set local Y limits (autoscaling)
        self.ax_velocity.set_ylim(y_velocity_limits)
        
        # Add velocity threshold lines if provided
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
        
        # Highlight saccades that fall within the window
        if len(saccades_df) > 0:
            for idx, row in saccades_df.iterrows():
                # Use DataFrame index as saccade_id
                saccade_id = int(idx)
                start_time = row.get('start_time', row.get('time', 0))
                end_time = row.get('end_time', start_time + row.get('duration', 0.1))
                
                # Only highlight if saccade overlaps with window
                if end_time >= window_start and start_time <= window_end:
                    # Get label for color: user label takes precedence over rule-based
                    user_label = user_labels.get(saccade_id) if user_labels else None
                    
                    if user_label:
                        # Use user classification color
                        color = label_colors.get(user_label, 'gray')
                    else:
                        # Use rule-based classification color (if available)
                        rule_based_label = row.get('saccade_type', 'unknown')
                        color = rule_based_colors.get(rule_based_label, 'gray')
                    
                    # Highlight selected saccade with higher transparency, non-selected more transparent
                    if saccade_id == selected_saccade_id:
                        alpha = 0.3  # Selected saccade
                    else:
                        alpha = 0.15  # Non-selected saccades more transparent
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
        
        # Remove dashed line for selected saccade (no special marking)
        
        self.canvas.draw()
    
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
            if 'X_smooth' in segment_df.columns:
                pos_data = segment_df['X_smooth']
            elif 'X_raw' in segment_df.columns:
                pos_data = segment_df['X_raw']
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
        
        # Plot position - use raw position to preserve original sign (downward/upward)
        # Prefer X_smooth (raw) over X_smooth_baselined to show actual direction
        if 'X_smooth' in segment_df.columns:
            ax_position.plot(segment_df['Time_rel_threshold'], segment_df['X_smooth'],
                           'b-', linewidth=2)
        elif 'X_raw' in segment_df.columns:
            ax_position.plot(segment_df['Time_rel_threshold'], segment_df['X_raw'],
                           'b-', linewidth=2)
        elif 'X_smooth_baselined' in segment_df.columns:
            # Fallback to baselined if raw not available
            ax_position.plot(segment_df['Time_rel_threshold'], segment_df['X_smooth_baselined'],
                           'b-', linewidth=2)
        
        ax_position.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax_position.set_xlabel('Time relative to threshold crossing (s)', fontsize=10)
        ax_position.set_ylabel('Position (px)', fontsize=10)
        ax_position.set_title('Peri-Saccade Segment: Position', fontsize=11, fontweight='bold')
        ax_position.grid(True, alpha=0.3)
        # Set Y limits (matching time series scale)
        ax_position.set_ylim(y_position_limits)
        
        # Plot velocity
        if 'vel_x_smooth' in segment_df.columns:
            ax_velocity.plot(segment_df['Time_rel_threshold'], segment_df['vel_x_smooth'],
                           'r-', linewidth=2)
        
        ax_velocity.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax_velocity.set_xlabel('Time relative to threshold crossing (s)', fontsize=10)
        ax_velocity.set_ylabel('Velocity (px/s)', fontsize=10)
        ax_velocity.set_title('Peri-Saccade Segment: Velocity', fontsize=11, fontweight='bold')
        ax_velocity.grid(True, alpha=0.3)
        # Set Y limits (matching time series scale)
        ax_velocity.set_ylim(y_velocity_limits)
        
        # Add velocity threshold lines if provided
        if vel_thresh is not None:
            ax_velocity.axhline(vel_thresh, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
            ax_velocity.axhline(-vel_thresh, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        
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
        
        # Determine eye label from video_label (e.g., "VideoData1 (L: Left)" -> "L")
        self.eye_label = 'Unknown'
        if 'L:' in self.video_label or '(L:' in self.video_label:
            self.eye_label = 'L'
        elif 'R:' in self.video_label or '(R:' in self.video_label:
            self.eye_label = 'R'
        
        # Create combined saccades list with IDs
        self.saccades_list = []
        self._build_saccades_list()
        
        # User labels (saccade_id -> user_label)
        self.user_labels: Dict[int, str] = {}
        self._load_existing_annotations()
        
        # Current selection
        self.current_index = 0
        
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
            original_index = row.get('original_index', idx) if 'original_index' in row else idx
            
            self.saccades_list.append({
                'saccade_id': saccade_id,
                'original_index': original_index,  # Store for segment matching
                'time': row.get('time', row.get('start_time', 0)),
                'amplitude': row.get('amplitude', 0),
                'duration': row.get('duration', 0),
                'rule_based_label': row.get('saccade_type', 'unknown'),
                'rule_based_confidence': row.get('classification_confidence', 0.5),
                'direction': row.get('saccade_direction', row.get('direction', 'unknown')),
                'index': idx  # Original DataFrame index
            })
        
        # Sort by time
        self.saccades_list.sort(key=lambda x: x['time'])
    
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
        self.table.horizontalHeader().setStretchLastSection(True)
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
        QShortcut(QKeySequence("N"), self, self._next_saccade)
        QShortcut(QKeySequence("P"), self, self._previous_saccade)
        QShortcut(QKeySequence("S"), self, self._save_current_annotation)
        
        # Exit shortcuts
        QShortcut(QKeySequence("Escape"), self, self._exit_gui)
        QShortcut(QKeySequence("Q"), self, self._exit_gui)
    
    def _update_table(self):
        """Update the saccade table."""
        self.table.setRowCount(len(self.saccades_list))
        
        for i, saccade in enumerate(self.saccades_list):
            saccade_id = saccade['saccade_id']
            user_label = self.user_labels.get(saccade_id, '')
            
            self.table.setItem(i, 0, QTableWidgetItem(str(saccade_id)))
            self.table.setItem(i, 1, QTableWidgetItem(self.eye_label))
            self.table.setItem(i, 2, QTableWidgetItem(f"{saccade['time']:.2f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{saccade['amplitude']:.1f}"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{saccade['duration']:.3f}"))
            self.table.setItem(i, 5, QTableWidgetItem(saccade['rule_based_label']))
            
            user_label_item = QTableWidgetItem(user_label if user_label else 'Unlabeled')
            if user_label:
                user_label_item.setBackground(Qt.green)
            self.table.setItem(i, 6, user_label_item)
        
        # Select current row
        if 0 <= self.current_index < len(self.saccades_list):
            self.table.selectRow(self.current_index)
            self.table.scrollToItem(self.table.item(self.current_index, 0))
    
    def _update_statistics(self):
        """Update statistics display."""
        total = len(self.saccades_list)
        annotated = len(self.user_labels)
        remaining = total - annotated
        
        stats_text = f"<b>Statistics:</b><br>"
        stats_text += f"Total: {total} | Annotated: {annotated} | Remaining: {remaining}"
        
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
        
        # Update time series plot
        if self.df is not None and len(self.all_saccades_df) > 0:
            self.time_series_plot.plot_time_series(
                self.df,
                self.all_saccades_df,
                selected_saccade_id=saccade_id,
                user_labels=self.user_labels,
                window_duration=6.0,  # ±3 seconds
                vel_thresh=self.vel_thresh
            )
        
        # Update peri-saccade plot
        if len(self.peri_saccades) > 0:
            # Find segment for current saccade
            # Segments have saccade_id matching original_index from upward/downward DataFrames
            segment = None
            matching_seg_id = None
            
            # DEBUG: Print matching information
            print(f"\n{'='*60}")
            print(f"DEBUG: Matching segment for saccade")
            print(f"{'='*60}")
            print(f"Current saccade_id (GUI): {saccade_id}")
            print(f"Current original_index (from saccades_list): {original_index}")
            print(f"Current df_index (all_saccades_df): {df_index}")
            
            # Check what's in all_saccades_df for this index
            if df_index in self.all_saccades_df.index:
                row_data = self.all_saccades_df.loc[df_index]
                print(f"all_saccades_df row data:")
                print(f"  - original_index: {row_data.get('original_index', 'N/A')}")
                print(f"  - is_upward: {row_data.get('is_upward', 'N/A')}")
                print(f"  - direction: {row_data.get('direction', 'N/A')}")
                print(f"  - time: {row_data.get('time', 'N/A')}")
            
            # Check all segments
            print(f"\nAvailable segments ({len(self.peri_saccades)} total):")
            for i, seg in enumerate(self.peri_saccades):
                if 'saccade_id' in seg.columns:
                    seg_id = seg['saccade_id'].iloc[0]
                    seg_dir = seg.get('saccade_direction', seg.get('saccade_direction', 'unknown'))
                    seg_time = seg.get('Time_rel_threshold', pd.Series([0])).iloc[0] if len(seg) > 0 else 'N/A'
                    match_status = "✓ MATCH" if seg_id == original_index else "✗"
                    print(f"  Segment {i}: saccade_id={seg_id}, direction={seg_dir}, match={match_status}")
            
            for seg in self.peri_saccades:
                if 'saccade_id' in seg.columns:
                    seg_saccade_id = seg['saccade_id'].iloc[0]
                    # Match by original_index (segments use original DataFrame indices)
                    if seg_saccade_id == original_index:
                        segment = seg
                        matching_seg_id = seg_saccade_id
                        print(f"\n✓ Found matching segment with saccade_id={matching_seg_id}")
                        break
            
            if segment is None:
                print(f"\n✗ No matching segment found for original_index={original_index}")
                print(f"  Trying fallback: matching by saccade_id={saccade_id}")
                # Fallback: try matching by current saccade_id
                for seg in self.peri_saccades:
                    if 'saccade_id' in seg.columns:
                        seg_saccade_id = seg['saccade_id'].iloc[0]
                        if seg_saccade_id == saccade_id:
                            segment = seg
                            matching_seg_id = seg_saccade_id
                            print(f"  ✓ Fallback match found with saccade_id={matching_seg_id}")
                            break
            
            print(f"{'='*60}\n")
            
            if segment is not None:
                # Get saccade info from DataFrame using index
                saccade_info = {
                    'amplitude': current_saccade.get('amplitude', 0),
                    'duration': current_saccade.get('duration', 0),
                }
                
                # Try to get peak_velocity from DataFrame
                if df_index in self.all_saccades_df.index:
                    if 'peak_velocity' in self.all_saccades_df.columns:
                        peak_vel = self.all_saccades_df.loc[df_index, 'peak_velocity']
                        saccade_info['peak_velocity'] = peak_vel if not pd.isna(peak_vel) else 0
                    elif 'velocity' in self.all_saccades_df.columns:
                        peak_vel = self.all_saccades_df.loc[df_index, 'velocity']
                        saccade_info['peak_velocity'] = peak_vel if not pd.isna(peak_vel) else 0
                    else:
                        saccade_info['peak_velocity'] = 0
                else:
                    saccade_info['peak_velocity'] = 0
                
                # Get Y limits from time series plot (for matching scale)
                time_series_pos_lims = self.time_series_plot.ax_position.get_ylim() if self.time_series_plot.ax_position else None
                time_series_vel_lims = self.time_series_plot.ax_velocity.get_ylim() if self.time_series_plot.ax_velocity else None
                
                self.peri_plot.plot_segment(segment, saccade_info, 
                                          y_position_limits=time_series_pos_lims,
                                          y_velocity_limits=time_series_vel_lims,
                                          vel_thresh=self.vel_thresh)
            else:
                self.peri_plot.clear()
        else:
            self.peri_plot.clear()
    
    def update_display(self):
        """Update all displays."""
        self._update_table()
        self._update_statistics()
        self._update_plots()
    
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


def launch_annotation_gui(
    saccade_results: Dict,
    features_df: Optional[pd.DataFrame] = None,
    experiment_id: str = 'unknown',
    annotations_file_path: Union[str, Path] = 'saccade_annotations_master.csv'
):
    """
    Launch the annotation GUI.
    
    Parameters
    ----------
    saccade_results : dict
        Dictionary from analyze_eye_video_saccades() containing:
        - 'df': Full time series DataFrame
        - 'all_saccades_df': DataFrame with all saccades
        - 'peri_saccades': List of peri-saccade segment DataFrames
    features_df : pd.DataFrame, optional
        Features DataFrame from extract_ml_features() (for future use)
    experiment_id : str
        Unique experiment identifier
    annotations_file_path : str or Path
        Path to master annotations CSV file
    """
    if not PYQT5_AVAILABLE:
        raise ImportError("PyQt5 not available. Install with: pip install PyQt5")
    
    app = QApplication(sys.argv)
    
    window = SaccadeAnnotationGUI(
        saccade_results=saccade_results,
        features_df=features_df,
        experiment_id=experiment_id,
        annotations_file_path=annotations_file_path
    )
    
    window._setup_keyboard_shortcuts()
    window.show()
    
    sys.exit(app.exec_())


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

