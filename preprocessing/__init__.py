"""Signal preprocessing subpackage: ECG, PPG, cross-modal synchronization, and signal quality."""
from .signal_quality import SignalQualityResult, check_feasibility_rules, adaptive_template_sqi, assess_segment_quality
