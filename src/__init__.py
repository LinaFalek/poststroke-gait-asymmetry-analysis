"""
Gait Analysis - Python Implementation
======================================

A Python implementation of IMU-based gait analysis, converted from MATLAB.

Core Modules:
- gait_functions: Core processing functions (filtering, calibration, swing detection)
- gait_processing: Main processing pipeline

Author: Converted from MATLAB (Huiseok MOON)
Date: 2026-02-15
"""

from .gait_functions import *
from .gait_processing import *

__version__ = "1.0.0"
__author__ = "Python conversion from MATLAB (H. MOON)"
