"""
AI Student Performance & Learning Behavior Scorer.
Evaluates student academic reading scores (0-100) based on book selection quality,
study timings/turnaround, reading frequency, and domain diversity.
"""

import pandas as pd
import numpy as np

class StudentPerformanceAnalyzer:
    def __init__(self, books_df, users_df, borrowing_df):
        self.books_df = books_df.copy()
        self.users_df = users_df.copy()
        self.borrowing_df = borrowing_df.copy()
        
        # Ensure correct types
        self.books_df['book_id'] = self.books_df['book_id'].astype(int)
        self.users_df['user_id'] = self.users_df['user_id'].astype(int)
        self.borrowing_df['user_id'] = self.borrowing_df['user_id'].astype(int)
        self.borrowing_df['book_id'] = self.borrowing_df['book_id'].astype(int)

    def calculate_student_score(self, user_id):
        """
        Calculates a comprehensive multi-factor score (0-100) for a student:
        1. Book Selection Quality (30 pts)
        2. Reading Frequency & Volume (30 pts)
        3. Study Timing & Return Turnaround (25 pts)
        4. Domain Diversity & Depth (15 pts)
        """
        user_history = self.borrowing_df[self.borrowing_df['user_id'] == user_id]
        user_row = self.users_df[self.users_df['user_id'] == user_id]
        
        user_name = user_row['name'].values[0] if not user_row.empty else f"User {user_id}"
        dept = user_row['department'].values[0] if not user_row.empty else "General"

        # Cold start case
        if user_history.empty:
            return {
                'user_id': user_id,
                'name': user_name,
                'department': dept,
                'total_score': 35.0,
                'tier': '🌱 Emerging Reader',
                'badge_color': '#64748B',
                'selection_score': 10.0,
                'frequency_score': 5.0,
                'timing_score': 15.0,
                'diversity_score': 5.0,
                'books_count': 0,
                'distinct_genres': 0,
                'est_study_hours_week': '2 - 4 hrs',
                'primary_domain': 'Unassigned',
                'ai_strengths': 'Newly registered student account.',
                'ai_recommendations': 'Start by borrowing introductory Data Structures or Programming textbooks.'
            }

        # 1. Book Selection Quality (max 30 pts)
        borrowed_book_ids = list(user_history['book_id'])
        borrowed_books = self.books_df[self.books_df['book_id'].isin(borrowed_book_ids)]
        
        avg_rating = borrowed_books['rating'].mean() if not borrowed_books.empty else 4.0
        # Advanced technical subjects have higher weight
        rigorous_genres = ['AI/ML', 'Computer Science', 'Electronics', 'Databases', 'Networking']
        rigor_count = borrowed_books['genre'].isin(rigorous_genres).sum() if not borrowed_books.empty else 0
        rigor_ratio = (rigor_count / len(borrowed_books)) if len(borrowed_books) > 0 else 0
        
        selection_score = min(30.0, (avg_rating / 5.0) * 18.0 + (rigor_ratio * 12.0))

        # 2. Reading Frequency & Volume (max 30 pts)
        books_count = len(user_history)
        frequency_score = min(30.0, books_count * 7.5)

        # 3. Study Timing & Return Turnaround (max 25 pts)
        # Returned on time vs currently Borrowed
        returned_count = len(user_history[user_history['status'] == 'Returned'])
        return_rate = (returned_count / books_count) if books_count > 0 else 1.0
        timing_score = 15.0 + (return_rate * 10.0)

        # 4. Domain Diversity & Depth (max 15 pts)
        distinct_genres = borrowed_books['genre'].nunique() if not borrowed_books.empty else 0
        diversity_score = min(15.0, distinct_genres * 5.0)

        total_score = round(selection_score + frequency_score + timing_score + diversity_score, 1)
        total_score = min(100.0, max(10.0, total_score))

        # Determine Tier
        if total_score >= 88:
            tier = '🏆 Elite Scholar'
            badge_color = '#10B981' # Emerald
        elif total_score >= 75:
            tier = '🌟 Proficient Scholar'
            badge_color = '#6366F1' # Indigo
        elif total_score >= 60:
            tier = '🚀 Active Learner'
            badge_color = '#38BDF8' # Cyan
        else:
            tier = '🌱 Developing Reader'
            badge_color = '#F59E0B' # Amber

        # Primary Domain & Estimated Weekly Study Timings
        top_genre = borrowed_books['genre'].mode()[0] if not borrowed_books.empty and not borrowed_books['genre'].mode().empty else 'General Tech'
        est_hours = f"{min(25, max(6, books_count * 4))} - {min(30, max(8, books_count * 5))} hrs/week"

        # AI Qualitative Insights
        if 'AI/ML' in list(borrowed_books['genre']):
            ai_strengths = f"Exemplary focus on advanced AI/ML architectures and predictive models. High book selection quality (avg ⭐ {avg_rating:.1f}/5.0)."
            ai_recommendations = "Explore practical embedded systems or cloud deployment books to complement theoretical AI modeling."
        elif 'Computer Science' in list(borrowed_books['genre']):
            ai_strengths = f"Strong grounding in core algorithmic problem solving, BST indexing, and data structure principles."
            ai_recommendations = "Branch into Machine Learning applications and modern software design pattern literature."
        else:
            ai_strengths = f"Consistent engineering coursework reading with steady library loan turnaround."
            ai_recommendations = "Increase reading diversity across emerging domains like Data Science & Cloud Computing."

        return {
            'user_id': user_id,
            'name': user_name,
            'department': dept,
            'total_score': total_score,
            'tier': tier,
            'badge_color': badge_color,
            'selection_score': round(selection_score, 1),
            'frequency_score': round(frequency_score, 1),
            'timing_score': round(timing_score, 1),
            'diversity_score': round(diversity_score, 1),
            'books_count': books_count,
            'distinct_genres': distinct_genres,
            'est_study_hours_week': est_hours,
            'primary_domain': top_genre,
            'ai_strengths': ai_strengths,
            'ai_recommendations': ai_recommendations
        }

    def get_all_students_leaderboard(self):
        """Generates a ranked leaderboard of all students."""
        records = []
        for u_id in self.users_df['user_id']:
            records.append(self.calculate_student_score(u_id))
        df = pd.DataFrame(records)
        return df.sort_values(by='total_score', ascending=False).reset_index(drop=True)
