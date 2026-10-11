"""
LibAI: AI-Based Library Recommendation & Data Structures System
Department of E&TC - Autonomous Mini Project
Complete Standalone Single-File Implementation.

Contains:
1. Singly Linked List (Chronological Borrowing History)
2. Binary Search Tree (BST) (Book ID Indexing & In-Order Traversal)
3. Hash Table with Separate Chaining (O(1) ISBN Fast Search)
4. FIFO Queue (Book Reservation Waiting List)
5. Hybrid AI Recommendation Engine (Content-Based TF-IDF + Collaborative Filtering)
6. AI Student Academic Performance & Reading Behavior Scorer
7. Streamlit Interactive Cyber-Clean Dashboard UI
"""

import os
import sys
import io
import time
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# 1. CUSTOM DATA STRUCTURE IMPLEMENTATIONS (FROM SCRATCH)
# ==============================================================================

# --- A. Singly Linked List ---
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
        self._count = 0

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
        else:
            curr = self.head
            while curr.next:
                curr = curr.next
            curr.next = new_node
        self._count += 1

    def prepend(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
        self._count += 1

    def to_list(self):
        elements = []
        curr = self.head
        while curr:
            elements.append(curr.data)
            curr = curr.next
        return elements

    def size(self):
        return self._count

    def is_empty(self):
        return self.head is None

    def __len__(self):
        return self._count

    def __iter__(self):
        curr = self.head
        while curr:
            yield curr.data
            curr = curr.next


# --- B. Binary Search Tree (BST) ---
class BSTNode:
    def __init__(self, key, value):
        self.key = int(key)     # Key for binary search (Book ID)
        self.value = value      # Book details dictionary payload
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
        self._count = 0

    def insert(self, key, value):
        self.root = self._insert_rec(self.root, int(key), value)

    def _insert_rec(self, node, key, value):
        if node is None:
            self._count += 1
            return BSTNode(key, value)
        if key < node.key:
            node.left = self._insert_rec(node.left, key, value)
        elif key > node.key:
            node.right = self._insert_rec(node.right, key, value)
        else:
            node.value = value  # Duplicate key updates record
        return node

    def search(self, key):
        return self._search_rec(self.root, int(key))

    def _search_rec(self, node, key):
        if node is None or node.key == key:
            return node.value if node else None
        if key < node.key:
            return self._search_rec(node.left, key)
        return self._search_rec(node.right, key)

    def inorder(self):
        result = []
        self._inorder_rec(self.root, result)
        return result

    def _inorder_rec(self, node, result):
        if node:
            self._inorder_rec(node.left, result)
            result.append((node.key, node.value))
            self._inorder_rec(node.right, result)

    def search_by_title_substring(self, query):
        all_books = self.inorder()
        q_lower = str(query).lower()
        matched = []
        for key, val in all_books:
            title = str(val.get('title', '')).lower()
            author = str(val.get('author', '')).lower()
            genre = str(val.get('genre', '')).lower()
            if q_lower in title or q_lower in author or q_lower in genre:
                matched.append(val)
        return matched

    def size(self):
        return self._count


# --- C. Hash Table with Separate Chaining ---
class HashNode:
    def __init__(self, key, value):
        self.key = str(key)
        self.value = value
        self.next = None

class HashTable:
    def __init__(self, capacity=64):
        self.capacity = capacity
        self.buckets = [None] * self.capacity
        self._size = 0

    def _hash(self, key):
        key_str = str(key)
        hash_val = 0
        p = 31
        m = 10**9 + 9
        p_pow = 1
        for char in key_str:
            hash_val = (hash_val + (ord(char) - ord('a') + 1) * p_pow) % m
            p_pow = (p_pow * p) % m
        return hash_val % self.capacity

    def put(self, key, value):
        key_str = str(key)
        index = self._hash(key_str)
        head = self.buckets[index]

        curr = head
        while curr:
            if curr.key == key_str:
                curr.value = value
                return
            curr = curr.next

        new_node = HashNode(key_str, value)
        new_node.next = head
        self.buckets[index] = new_node
        self._size += 1

    def get(self, key):
        key_str = str(key)
        index = self._hash(key_str)
        curr = self.buckets[index]
        while curr:
            if curr.key == key_str:
                return curr.value
            curr = curr.next
        return None

    def values(self):
        vals = []
        for bucket in self.buckets:
            curr = bucket
            while curr:
                vals.append(curr.value)
                curr = curr.next
        return vals

    def size(self):
        return self._size


# --- D. FIFO Queue ---
class QueueNode:
    def __init__(self, data):
        self.data = data
        self.next = None

class Queue:
    def __init__(self):
        self.front = None
        self.rear = None
        self._count = 0

    def enqueue(self, data):
        new_node = QueueNode(data)
        if self.rear is None:
            self.front = self.rear = new_node
        else:
            self.rear.next = new_node
            self.rear = new_node
        self._count += 1

    def dequeue(self):
        if self.is_empty():
            return None
        temp = self.front
        self.front = temp.next
        if self.front is None:
            self.rear = None
        self._count -= 1
        return temp.data

    def peek(self):
        return self.front.data if self.front else None

    def is_empty(self):
        return self.front is None

    def size(self):
        return self._count

    def to_list(self):
        elements = []
        curr = self.front
        while curr:
            elements.append(curr.data)
            curr = curr.next
        return elements


# ==============================================================================
# 2. AI / ML ENGINE IMPLEMENTATIONS
# ==============================================================================

class LibraryRecommender:
    def __init__(self, books_df, users_df, borrowing_df):
        self.books_df = books_df.copy()
        self.users_df = users_df.copy()
        self.borrowing_df = borrowing_df.copy()
        
        self.books_df['book_id'] = self.books_df['book_id'].astype(int)
        self.users_df['user_id'] = self.users_df['user_id'].astype(int)
        self.borrowing_df['user_id'] = self.borrowing_df['user_id'].astype(int)
        self.borrowing_df['book_id'] = self.borrowing_df['book_id'].astype(int)

        self._build_content_model()
        self._build_collaborative_model()

    def _build_content_model(self):
        self.books_df['content_features'] = (
            self.books_df['genre'].fillna('') + " " +
            self.books_df['author'].fillna('') + " " +
            self.books_df['description'].fillna('')
        )
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.books_df['content_features'])
        self.content_sim_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def _build_collaborative_model(self):
        if not self.borrowing_df.empty:
            user_item = self.borrowing_df.groupby(['user_id', 'book_id']).size().unstack(fill_value=0)
            user_item = (user_item > 0).astype(int)
            self.user_item_matrix = user_item
            if user_item.shape[0] > 1:
                user_sim = cosine_similarity(user_item)
                self.user_sim_df = pd.DataFrame(user_sim, index=user_item.index, columns=user_item.index)
            else:
                self.user_sim_df = pd.DataFrame()
        else:
            self.user_item_matrix = pd.DataFrame()
            self.user_sim_df = pd.DataFrame()

    def get_borrowed_books_for_user(self, user_id):
        user_history = self.borrowing_df[self.borrowing_df['user_id'] == user_id]
        return list(user_history['book_id'].unique())

    def recommend_books_for_user(self, user_id, top_n=5):
        borrowed_ids = self.get_borrowed_books_for_user(user_id)
        all_book_ids = list(self.books_df['book_id'].unique())
        candidate_ids = [b_id for b_id in all_book_ids if b_id not in borrowed_ids]

        if not borrowed_ids:
            top_books = self.books_df.sort_values(by=['rating', 'publication_year'], ascending=False).head(top_n).copy()
            top_books['recommendation_score'] = top_books['rating'] / 5.0
            top_books['reason'] = "Popular & High Rated Book (Popularity Default)"
            return top_books[['book_id', 'title', 'author', 'genre', 'rating', 'recommendation_score', 'reason']]

        collab_scores = {b_id: 0.0 for b_id in candidate_ids}
        if user_id in self.user_sim_df.index:
            similar_users = self.user_sim_df[user_id].sort_values(ascending=False)
            similar_users = similar_users[similar_users.index != user_id]
            for sim_user_id, sim_score in similar_users.items():
                if sim_score <= 0:
                    continue
                sim_user_borrowed = self.get_borrowed_books_for_user(sim_user_id)
                for b_id in sim_user_borrowed:
                    if b_id in candidate_ids:
                        collab_scores[b_id] += sim_score

        content_scores = {b_id: 0.0 for b_id in candidate_ids}
        book_id_to_idx = {b_id: idx for idx, b_id in enumerate(self.books_df['book_id'])}
        
        for borrowed_id in borrowed_ids:
            if borrowed_id in book_id_to_idx:
                borrowed_idx = book_id_to_idx[borrowed_id]
                sim_vector = self.content_sim_matrix[borrowed_idx]
                for candidate_id in candidate_ids:
                    cand_idx = book_id_to_idx[candidate_id]
                    content_scores[candidate_id] = max(content_scores[candidate_id], sim_vector[cand_idx])

        hybrid_scores = []
        max_collab = max(collab_scores.values()) if collab_scores and max(collab_scores.values()) > 0 else 1.0
        
        for b_id in candidate_ids:
            c_score = content_scores.get(b_id, 0.0)
            col_score = collab_scores.get(b_id, 0.0) / max_collab
            combined = 0.6 * c_score + 0.4 * col_score
            hybrid_scores.append({
                'book_id': b_id,
                'hybrid_score': combined,
                'content_score': c_score,
                'collab_score': col_score
            })

        hybrid_df = pd.DataFrame(hybrid_scores)
        hybrid_df = hybrid_df.sort_values(by='hybrid_score', ascending=False).head(top_n)

        result_df = pd.merge(hybrid_df, self.books_df, on='book_id')
        result_df['recommendation_score'] = result_df['hybrid_score'].round(3)

        def get_reason(row):
            if row['collab_score'] > 0.3:
                return "Borrowed by readers with similar reading habits"
            elif row['content_score'] > 0.3:
                return "Similar content/genre to books in your borrowing history"
            else:
                return "Matches your overall library reading profile"

        result_df['reason'] = result_df.apply(get_reason, axis=1)
        return result_df[['book_id', 'title', 'author', 'genre', 'rating', 'recommendation_score', 'reason']]


class StudentPerformanceAnalyzer:
    def __init__(self, books_df, users_df, borrowing_df):
        self.books_df = books_df.copy()
        self.users_df = users_df.copy()
        self.borrowing_df = borrowing_df.copy()
        
        self.books_df['book_id'] = self.books_df['book_id'].astype(int)
        self.users_df['user_id'] = self.users_df['user_id'].astype(int)
        self.borrowing_df['user_id'] = self.borrowing_df['user_id'].astype(int)
        self.borrowing_df['book_id'] = self.borrowing_df['book_id'].astype(int)

    def calculate_student_score(self, user_id):
        user_history = self.borrowing_df[self.borrowing_df['user_id'] == user_id]
        user_row = self.users_df[self.users_df['user_id'] == user_id]
        
        user_name = user_row['name'].values[0] if not user_row.empty else f"User {user_id}"
        dept = user_row['department'].values[0] if not user_row.empty else "General"

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

        borrowed_book_ids = list(user_history['book_id'])
        borrowed_books = self.books_df[self.books_df['book_id'].isin(borrowed_book_ids)]
        
        avg_rating = borrowed_books['rating'].mean() if not borrowed_books.empty else 4.0
        rigorous_genres = ['AI/ML', 'Computer Science', 'Electronics', 'Databases', 'Networking']
        rigor_count = borrowed_books['genre'].isin(rigorous_genres).sum() if not borrowed_books.empty else 0
        rigor_ratio = (rigor_count / len(borrowed_books)) if len(borrowed_books) > 0 else 0
        
        selection_score = min(30.0, (avg_rating / 5.0) * 18.0 + (rigor_ratio * 12.0))
        books_count = len(user_history)
        frequency_score = min(30.0, books_count * 7.5)

        returned_count = len(user_history[user_history['status'] == 'Returned'])
        return_rate = (returned_count / books_count) if books_count > 0 else 1.0
        timing_score = 15.0 + (return_rate * 10.0)

        distinct_genres = borrowed_books['genre'].nunique() if not borrowed_books.empty else 0
        diversity_score = min(15.0, distinct_genres * 5.0)

        total_score = round(selection_score + frequency_score + timing_score + diversity_score, 1)
        total_score = min(100.0, max(10.0, total_score))

        if total_score >= 88:
            tier = '🏆 Elite Scholar'
            badge_color = '#10B981'
        elif total_score >= 75:
            tier = '🌟 Proficient Scholar'
            badge_color = '#6366F1'
        elif total_score >= 60:
            tier = '🚀 Active Learner'
            badge_color = '#38BDF8'
        else:
            tier = '🌱 Developing Reader'
            badge_color = '#F59E0B'

        top_genre = borrowed_books['genre'].mode()[0] if not borrowed_books.empty and not borrowed_books['genre'].mode().empty else 'General Tech'
        est_hours = f"{min(25, max(6, books_count * 4))} - {min(30, max(8, books_count * 5))} hrs/week"

        if 'AI/ML' in list(borrowed_books['genre']):
            ai_strengths = f"Exemplary focus on advanced AI/ML architectures and predictive models (avg ⭐ {avg_rating:.1f}/5.0)."
            ai_recommendations = "Explore practical embedded systems or cloud deployment books to complement theoretical AI modeling."
        elif 'Computer Science' in list(borrowed_books['genre']):
            ai_strengths = "Strong grounding in core algorithmic problem solving, BST indexing, and data structure principles."
            ai_recommendations = "Branch into Machine Learning applications and modern software design pattern literature."
        else:
            ai_strengths = "Consistent engineering coursework reading with steady library loan turnaround."
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
        records = [self.calculate_student_score(u_id) for u_id in self.users_df['user_id']]
        return pd.DataFrame(records).sort_values(by='total_score', ascending=False).reset_index(drop=True)


# ==============================================================================
# 3. EMBEDDED DATASET & FILE RESOLUTION (PREVENTS FILE-NOT-FOUND ERRORS)
# ==============================================================================

DEFAULT_BOOKS_CSV = """book_id,title,author,genre,isbn,rating,copies_available,publication_year,description
101,Introduction to Algorithms,Thomas H. Cormen,Computer Science,9780262033848,4.9,5,2009,Comprehensive guide to modern algorithms graph theory and sorting algorithms.
102,Data Structures and Algorithm Analysis in C++,Mark Allen Weiss,Computer Science,9780132847377,4.7,3,2013,Detailed analysis of binary search trees hash tables and linked lists in C++.
103,Artificial Intelligence: A Modern Approach,Stuart Russell,AI/ML,9780134610993,4.8,4,2020,The standard textbook on machine learning neural networks and intelligent agents.
104,Deep Learning,Ian Goodfellow,AI/ML,9780262035613,4.6,2,2016,In-depth coverage of mathematical foundations deep neural networks and autoencoders.
105,Pattern Recognition and Machine Learning,Christopher Bishop,AI/ML,9780387310732,4.8,3,2006,Comprehensive introduction to Bayesian pattern recognition and statistical learning.
106,Computer Networks,Andrew S. Tanenbaum,Networking,9780132126953,4.5,4,2010,Classic exploration of network layers routing algorithms and wireless protocols.
107,Database System Concepts,Abraham Silberschatz,Databases,9780073523323,4.6,6,2019,Fundamental concepts of relational database indexing query optimization and SQL.
108,Microelectronic Circuits,Adel S. Sedra,Electronics,9780190853464,4.7,4,2019,Standard textbook on semiconductor devices operational amplifiers and analog circuits.
109,Digital Signal Processing,John G. Proakis,Electronics,9780131873742,4.4,2,2006,Comprehensive coverage of z-transforms digital filter design and DFT algorithms.
110,Clean Code: A Handbook of Agile Software Craftsmanship,Robert C. Martin,Computer Science,9780132350884,4.7,5,2008,Best practices for writing maintainable scalable and clean code.
"""

DEFAULT_USERS_CSV = """user_id,name,department,email,year
201,Aarav Sharma,E&TC,aarav.sharma@nmiet.edu.in,SE
202,Priya Patel,Computer Science,priya.patel@nmiet.edu.in,TE
203,Rohan Verma,AI & Data Science,rohan.verma@nmiet.edu.in,SE
204,Ananya Rao,E&TC,ananya.rao@nmiet.edu.in,BE
205,Aditya Deshmukh,Computer Science,aditya.deshmukh@nmiet.edu.in,TE
"""

DEFAULT_BORROW_CSV = """borrow_id,user_id,book_id,borrow_date,return_date,status
301,201,101,2026-02-01,2026-02-15,Returned
302,201,102,2026-02-16,2026-03-01,Returned
303,201,103,2026-03-02,,Borrowed
304,202,103,2026-01-10,2026-01-25,Returned
305,202,104,2026-02-05,2026-02-20,Returned
306,202,105,2026-02-22,,Borrowed
307,203,101,2026-02-10,2026-02-24,Returned
308,203,107,2026-02-25,,Borrowed
309,204,108,2026-01-15,2026-01-30,Returned
310,204,109,2026-02-01,,Borrowed
311,205,102,2026-02-12,2026-02-26,Returned
312,205,110,2026-02-27,,Borrowed
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
BOOKS_CSV_PATH = os.path.join(DATA_DIR, 'books.csv')
USERS_CSV_PATH = os.path.join(DATA_DIR, 'users.csv')
BORROW_CSV_PATH = os.path.join(DATA_DIR, 'borrowing_history.csv')

def load_initial_dataframe(file_path, default_csv_content):
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception:
            pass
    return pd.read_csv(io.StringIO(default_csv_content.strip()))


# ==============================================================================
# 4. STREAMLIT FRONTEND & INTERACTIVE DASHBOARD
# ==============================================================================

st.set_page_config(
    page_title="LibAI | Single-File Data Structures & AI System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Titan Clean Light Cyber Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .stApp {
        background-color: #F8FAFC;
    }
    .hero-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #FFFFFF;
        padding: 2.2rem;
        border-radius: 18px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
        margin-bottom: 2rem;
    }
    .stat-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
    }
    .stat-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
    }
    .stat-lbl {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .ds-card {
        background: #FFFFFF;
        border-left: 5px solid #2563EB;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to sync session state DataFrames into custom Data Structures
def sync_data_structures():
    st.session_state.bst = BinarySearchTree()
    st.session_state.hash_table = HashTable(capacity=64)
    st.session_state.linked_list = LinkedList()

    # Build BST & Hash Table
    for _, row in st.session_state.books_df.iterrows():
        b_dict = row.to_dict()
        st.session_state.bst.insert(int(row['book_id']), b_dict)
        st.session_state.hash_table.put(str(row['isbn']), b_dict)

    # Build Linked List
    for _, row in st.session_state.borrow_df.iterrows():
        st.session_state.linked_list.append(row.to_dict())

    # Build ML Engines
    st.session_state.recommender = LibraryRecommender(
        st.session_state.books_df, st.session_state.users_df, st.session_state.borrow_df
    )
    st.session_state.scorer = StudentPerformanceAnalyzer(
        st.session_state.books_df, st.session_state.users_df, st.session_state.borrow_df
    )

# Initialize Data
if 'ds_initialized' not in st.session_state:
    st.session_state.books_df = load_initial_dataframe(BOOKS_CSV_PATH, DEFAULT_BOOKS_CSV)
    st.session_state.users_df = load_initial_dataframe(USERS_CSV_PATH, DEFAULT_USERS_CSV)
    st.session_state.borrow_df = load_initial_dataframe(BORROW_CSV_PATH, DEFAULT_BORROW_CSV)

    st.session_state.queue = Queue()
    st.session_state.queue.enqueue({'user_id': 203, 'user_name': 'Rohan Verma', 'book_id': 103, 'book_title': 'Artificial Intelligence: A Modern Approach'})
    st.session_state.queue.enqueue({'user_id': 205, 'user_name': 'Aditya Deshmukh', 'book_id': 101, 'book_title': 'Introduction to Algorithms'})

    sync_data_structures()
    st.session_state.ds_initialized = True

books_df = st.session_state.books_df
users_df = st.session_state.users_df
borrow_df = st.session_state.borrow_df
bst = st.session_state.bst
hash_table = st.session_state.hash_table
linked_list = st.session_state.linked_list
queue = st.session_state.queue
recommender = st.session_state.recommender
scorer = st.session_state.scorer

# Sidebar
with st.sidebar:
    st.markdown("### 🛡️ **LibAI Control Hub**")
    st.caption("NMIET Dept of E&TC • Autonomous")
    st.markdown("---")

    st.markdown("#### ⚡ **Data Structure Status**")
    st.markdown(f"- **BST Nodes**: `{bst.size()}`")
    st.markdown(f"- **Hash Table Keys**: `{hash_table.size()}`")
    st.markdown(f"- **Linked List Nodes**: `{linked_list.size()}`")
    st.markdown(f"- **Queue Length**: `{queue.size()}`")

    st.markdown("---")
    st.markdown("#### 💾 **Data Persistence**")
    if st.button("💾 Sync Data Structures", use_container_width=True, type="primary"):
        os.makedirs(DATA_DIR, exist_ok=True)
        st.session_state.books_df.to_csv(BOOKS_CSV_PATH, index=False)
        st.session_state.users_df.to_csv(USERS_CSV_PATH, index=False)
        st.session_state.borrow_df.to_csv(BORROW_CSV_PATH, index=False)
        st.success("✅ Synced to disk CSVs!")

# Main Dashboard Header
st.markdown("""
<div class="hero-card">
    <h1 style="margin:0; font-size:2.4rem; font-weight:800;">📚 LibAI: Smart Recommendation & Data Structures System</h1>
    <p style="margin-top:8px; opacity:0.85; font-size:1.05rem;">
        Integrated Singly Linked List, BST, Hash Table, FIFO Queue, TF-IDF Hybrid AI Recommender & Student Scorer.
    </p>
</div>
""", unsafe_allow_html=True)

# Overview Metrics
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="stat-box"><div class="stat-val">{len(books_df)}</div><div class="stat-lbl">Books Indexed (BST)</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="stat-box"><div class="stat-val">{len(users_df)}</div><div class="stat-lbl">Registered Students</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="stat-box"><div class="stat-val">{len(borrow_df)}</div><div class="stat-lbl">Loan History (List)</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="stat-box"><div class="stat-val">{queue.size()}</div><div class="stat-lbl">Waiting Queue (FIFO)</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 DS Engine (BST & Hash Table)",
    "📜 Loan Log (Linked List)",
    "⏳ Reservation Queue (FIFO)",
    "🤖 AI Book Recommender",
    "📊 Student Performance Scorer"
])

# --- TAB 1: DATA STRUCTURES ENGINE ---
with tab1:
    st.subheader("⚡ Data Structures Demonstration: BST & Hash Table")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        <div class="ds-card">
            <h4>🌲 Binary Search Tree (BST) Indexing</h4>
            <p>Provides <b>O(log N)</b> average search time for books by unique Book ID.</p>
        </div>
        """, unsafe_allow_html=True)

        search_id = st.number_input("Enter Book ID to search via BST:", min_value=101, max_value=999, value=103)
        if st.button("🌲 Execute BST Search", type="primary"):
            t0 = time.perf_counter()
            book_found = bst.search(int(search_id))
            t1 = time.perf_counter()
            exec_time = (t1 - t0) * 1000

            if book_found:
                st.success(f"✅ **BST Search Hit** (Time: `{exec_time:.4f} ms`)")
                st.json(book_found)
            else:
                st.error(f"❌ Book ID `{search_id}` not found in BST.")

        st.markdown("##### 📜 In-Order Traversal (Sorted by Book ID)")
        inorder_data = bst.inorder()
        inorder_df = pd.DataFrame([val for key, val in inorder_data])
        st.dataframe(inorder_df[['book_id', 'title', 'author', 'genre', 'rating']], use_container_width=True)

    with col_right:
        st.markdown("""
        <div class="ds-card" style="border-left-color: #10B981;">
            <h4>🔑 Hash Table (Polynomial Rolling Hash)</h4>
            <p>Provides instantaneous <b>O(1)</b> lookup for book metadata via ISBN string.</p>
        </div>
        """, unsafe_allow_html=True)

        search_isbn = st.text_input("Enter ISBN to lookup via Hash Table:", value="9780134610993")
        if st.button("🔑 Execute Hash Lookup", type="primary"):
            t0 = time.perf_counter()
            book_found = hash_table.get(search_isbn.strip())
            t1 = time.perf_counter()
            exec_time = (t1 - t0) * 1000

            if book_found:
                st.success(f"✅ **Hash Table Hit** (Time: `{exec_time:.4f} ms`)")
                st.json(book_found)
            else:
                st.error(f"❌ ISBN `{search_isbn}` not found in Hash Table.")

        st.markdown("##### ➕ Add New Book Record (Updates BST & Hash Table)")
        with st.form("add_book_form"):
            new_id = st.number_input("Book ID", min_value=111, max_value=999, value=111)
            new_title = st.text_input("Book Title", value="Operating System Concepts")
            new_author = st.text_input("Author", value="Abraham Silberschatz")
            new_genre = st.selectbox("Genre", ["Computer Science", "AI/ML", "Electronics", "Databases", "Networking"])
            new_isbn = st.text_input("ISBN", value="9781118063330")
            new_rating = st.slider("Rating", 1.0, 5.0, 4.7)

            if st.form_submit_button("🚀 Add Book to DS Engine"):
                book_payload = {
                    'book_id': int(new_id),
                    'title': new_title,
                    'author': new_author,
                    'genre': new_genre,
                    'isbn': new_isbn,
                    'rating': new_rating,
                    'copies_available': 3,
                    'publication_year': 2021,
                    'description': f"{new_title} textbook for engineering study."
                }
                # Update Session DataFrame
                st.session_state.books_df = pd.concat([st.session_state.books_df, pd.DataFrame([book_payload])], ignore_index=True)
                sync_data_structures()
                st.success(f"✅ Book '{new_title}' added to BST & Hash Table successfully!")
                st.rerun()


# --- TAB 2: LINKED LIST LOAN LOG ---
with tab2:
    st.subheader("📜 Singly Linked List: Dynamic Borrowing History")
    st.caption("Tracks borrowing history dynamically in memory using Linked Nodes.")

    list_elements = linked_list.to_list()
    if list_elements:
        history_df = pd.DataFrame(list_elements)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("Linked List is currently empty.")

    st.markdown("##### ➕ Issue Book (Appends Node to Linked List)")
    col_u, col_b = st.columns(2)
    with col_u:
        u_select = st.selectbox("Select Student:", users_df['name'])
    with col_b:
        b_select = st.selectbox("Select Book:", books_df['title'])

    if st.button("📌 Issue Book (Append to Linked List)"):
        u_id = users_df[users_df['name'] == u_select]['user_id'].values[0]
        b_id = books_df[books_df['title'] == b_select]['book_id'].values[0]

        new_loan = {
            'borrow_id': 300 + len(borrow_df) + 1,
            'user_id': int(u_id),
            'book_id': int(b_id),
            'borrow_date': '2026-03-10',
            'return_date': '',
            'status': 'Borrowed'
        }
        st.session_state.borrow_df = pd.concat([st.session_state.borrow_df, pd.DataFrame([new_loan])], ignore_index=True)
        sync_data_structures()
        st.success(f"✅ Appended new borrowing record for **{u_select}** to Singly Linked List!")
        st.rerun()


# --- TAB 3: FIFO QUEUE WAITING LIST ---
with tab3:
    st.subheader("⏳ FIFO Queue: Reservation Waiting List")
    st.caption("First-In, First-Out (FIFO) queue for managing book issue requests when copies are unavailable.")

    q_items = queue.to_list()
    if q_items:
        st.table(pd.DataFrame(q_items))
    else:
        st.info("Queue is currently empty.")

    q_col1, q_col2 = st.columns(2)
    with q_col1:
        st.markdown("##### 📥 Enqueue Request")
        req_name = st.text_input("Student Name", value="Sneha Kulkarni")
        req_book = st.selectbox("Book to Reserve", books_df['title'])
        if st.button("📥 Enqueue to Waiting List"):
            b_id = books_df[books_df['title'] == req_book]['book_id'].values[0]
            queue.enqueue({'user_id': 206, 'user_name': req_name, 'book_id': int(b_id), 'book_title': req_book})
            st.success(f"✅ Added {req_name} to FIFO Queue!")
            st.rerun()

    with q_col2:
        st.markdown("##### 📤 Dequeue Request")
        if st.button("📤 Serve Next In Queue (Dequeue)", type="primary"):
            dequeued = queue.dequeue()
            if dequeued:
                st.balloons()
                st.success(f"🎉 **Served Request**: {dequeued['user_name']} for *{dequeued['book_title']}*!")
                st.rerun()
            else:
                st.warning("Queue is empty.")


# --- TAB 4: AI BOOK RECOMMENDER ---
with tab4:
    st.subheader("🤖 Hybrid AI Book Recommendation Engine")
    st.caption("Combines Content-Based TF-IDF Cosine Similarity with User-User Collaborative Filtering.")

    sel_student = st.selectbox("Select Student Profile to Generate Recommendations:", users_df['name'], index=0)
    student_id = users_df[users_df['name'] == sel_student]['user_id'].values[0]

    st.markdown(f"#### 🎯 Recommended Books for **{sel_student}** (User ID: `{student_id}`)")
    recs = recommender.recommend_books_for_user(int(student_id), top_n=5)
    st.dataframe(recs, use_container_width=True)


# --- TAB 5: STUDENT PERFORMANCE SCORER ---
with tab5:
    st.subheader("📊 AI Student Academic Performance Scorer")
    st.caption("Evaluates student reading quality, study turnaround timing, volume, and domain diversity.")

    score_card = scorer.calculate_student_score(int(student_id))

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("Academic Score", f"{score_card['total_score']} / 100")
    with sc2:
        st.metric("Reading Tier", score_card['tier'])
    with sc3:
        st.metric("Estimated Study Time", score_card['est_study_hours_week'])

    st.markdown("##### 💡 AI Qualitative Feedback & Guidance")
    st.info(f"**Strengths**: {score_card['ai_strengths']}")
    st.success(f"**AI Recommendations**: {score_card['ai_recommendations']}")

    st.markdown("---")
    st.markdown("#### 🏆 All Students Academic Leaderboard")
    leaderboard = scorer.get_all_students_leaderboard()
    st.dataframe(leaderboard[['user_id', 'name', 'department', 'total_score', 'tier', 'primary_domain', 'est_study_hours_week']], use_container_width=True)
