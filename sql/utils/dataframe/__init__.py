"""Utility per manipolazione e analisi DataFrame."""
from sql.utils.dataframe.converter import DataFrameConverter
from sql.utils.dataframe.analyzer import DataFrameAnalyzer
from sql.utils.dataframe.year_detector import YearDetector, YearDetectionResult

__all__ = ["DataFrameConverter", "DataFrameAnalyzer", "YearDetector", "YearDetectionResult"]
