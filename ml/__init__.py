"""
AI/ML Recommendation & Student Performance Package.
Provides Personalized Hybrid Recommender and Student Academic Performance Scoring.
"""

from .recommender import LibraryRecommender
from .student_scorer import StudentPerformanceAnalyzer

__all__ = ['LibraryRecommender', 'StudentPerformanceAnalyzer']
