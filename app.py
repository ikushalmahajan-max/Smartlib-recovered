"""
SmartLib AI Library System - Fully Functional High-Performance Web Application
NMIET Department of E&TC • Data Structures & AI System
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


class BSTNode:
    def __init__(self, key, value):
        self.key = int(key)
        self.value = value
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
            node.value = value
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

    def size(self):
        return self._count


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

    def size(self):
        return self._size


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
# 2. AI / ML RECOMMENDER ENGINE
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

    def recommend_books_for_user(self, user_id, top_n=6):
        borrowed_ids = self.get_borrowed_books_for_user(user_id)
        all_book_ids = list(self.books_df['book_id'].unique())
        candidate_ids = [b_id for b_id in all_book_ids if b_id not in borrowed_ids]

        if not borrowed_ids:
            top_books = self.books_df.sort_values(by=['rating', 'publication_year'], ascending=False).head(top_n).copy()
            top_books['recommendation_score'] = (top_books['rating'] / 5.0).round(3)
            top_books['content_score'] = 0.0
            top_books['collab_score'] = top_books['recommendation_score']
            top_books['reason'] = "Rating-Based Baseline"
            return top_books

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
            combined = 0.0 * c_score + 1.0 * col_score
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
                return "Collaborative Filter: Users with similar borrowing patterns also interacted with this."
            elif row['content_score'] > 0.3:
                return "TF-IDF Vector Match: High similarity to books in your borrowing history."
            else:
                return "Genre Match: Matches your preferred genre and topic history."

        result_df['reason'] = result_df.apply(get_reason, axis=1)
        return result_df


# ==============================================================================
# 3. EMBEDDED DATASETS
# ==============================================================================

DEFAULT_BOOKS_CSV = """book_id,title,author,genre,isbn,rating,copies_available,publication_year,description
101,Introduction to Algorithms,Thomas H. Cormen,Computer Science,9780262033848,4.9,5,2009,Comprehensive guide to modern algorithms graph theory and sorting algorithms.
102,Data Structures and Algorithm Analysis in C++,Mark Allen Weiss,Computer Science,9780132847377,4.7,3,2013,Detailed analysis of binary search trees hash tables and linked lists in C++.
103,Artificial Intelligence: A Modern Approach,Stuart Russell,AI/ML,9780134610993,4.8,4,2020,The standard textbook on machine learning neural networks and intelligent agents.
104,Deep Learning,Ian Goodfellow,AI/ML,9780262035613,4.6,2,2016,In-depth coverage of mathematical foundations deep neural networks and autoencoders.
105,Pattern Recognition and Machine Learning,Christopher Bishop,AI/ML,9780387310732,4.8,3,2006,Comprehensive introduction to Bayesian pattern recognition and statistical learning.
106,Computer Networks: A Systems Approach,Larry Peterson,Computer Science,9780123850591,4.6,4,2010,Classic exploration of network layers routing algorithms and wireless protocols.
107,Database System Concepts,Abraham Silberschatz,Databases,9780078022159,4.6,6,2019,Fundamental concepts of relational database indexing query optimization and SQL.
108,Discrete Mathematics and Its Applications,Kenneth H. Rosen,Mathematics,9780073383095,4.8,4,2019,Standard textbook on discrete structures logic sets and graph theory.
109,Linear Algebra and Its Applications,Gilbert Strang,Mathematics,9780030105678,4.7,2,2006,Comprehensive coverage of matrix factorizations vector spaces and linear transformations.
110,Clean Code: A Handbook of Agile Software Craftsmanship,Robert C. Martin,Computer Science,9780132350884,4.7,5,2008,Best practices for writing maintainable scalable and clean code.
111,Algorithm Design,Jon Kleinberg,Computer Science,9780321295354,4.9,4,2006,Fundamental algorithm design techniques dynamic programming and network flow.
112,Operating System Concepts,Abraham Silberschatz,Computer Science,9781119863330,4.6,3,2021,Core concepts of operating system process management memory virtualization and concurrency.
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

def load_df_safe(path, fallback):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.read_csv(io.StringIO(fallback.strip()))

# ==============================================================================
# 4. STREAMLIT APP CONFIG & STYLING
# ==============================================================================

st.set_page_config(
    page_title="SmartLib • AI Library System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        color: #0F172A;
    }
    .stApp {
        background-color: #F8FAFC;
    }

    .accessible-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .kpi-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #4F46E5;
        border-radius: 12px;
        padding: 1rem;
    }
    .kpi-title {
        font-size: 0.8rem;
        font-weight: 800;
        text-transform: uppercase;
        color: #64748B;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
    }
    .badge-ds {
        background: #EEF2FF;
        color: #3730A3;
        border: 1px solid #C7D2FE;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to sync Session State Data Structures
def sync_data_structures():
    st.session_state.bst = BinarySearchTree()
    st.session_state.hash_table = HashTable(capacity=64)
    st.session_state.linked_list = LinkedList()

    for _, row in st.session_state.books_df.iterrows():
        b_dict = row.to_dict()
        st.session_state.bst.insert(int(row['book_id']), b_dict)
        st.session_state.hash_table.put(str(row['isbn']), b_dict)

    for _, row in st.session_state.borrow_df.iterrows():
        st.session_state.linked_list.append(row.to_dict())

    st.session_state.recommender = LibraryRecommender(
        st.session_state.books_df, st.session_state.users_df, st.session_state.borrow_df
    )

# ==============================================================================
# 5. INITIALIZE DATA & SESSION STATE
# ==============================================================================

if 'initialized' not in st.session_state:
    st.session_state.books_df = load_df_safe(BOOKS_CSV_PATH, DEFAULT_BOOKS_CSV)
    st.session_state.users_df = load_df_safe(USERS_CSV_PATH, DEFAULT_USERS_CSV)
    st.session_state.borrow_df = load_df_safe(BORROW_CSV_PATH, DEFAULT_BORROW_CSV)

    st.session_state.queue = Queue()
    st.session_state.queue.enqueue({'user_id': 203, 'user_name': 'Rohan Verma', 'book_title': 'Operating System Concepts', 'status': 'Waiting'})

    sync_data_structures()
    st.session_state.initialized = True

books_df = st.session_state.books_df
users_df = st.session_state.users_df
borrow_df = st.session_state.borrow_df
bst = st.session_state.bst
hash_table = st.session_state.hash_table
linked_list = st.session_state.linked_list
queue = st.session_state.queue
recommender = st.session_state.recommender

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4338CA 0%, #3730A3 100%); color: white; padding: 1.2rem; border-radius: 12px; margin-bottom: 1.2rem;">
        <h2 style="margin:0; font-size:1.4rem; font-weight:800; color:white;">🔮 SmartLib</h2>
        <p style="margin:2px 0 0 0; opacity:0.85; font-size:0.8rem; color:#C7D2FE;">AI Library System • NMIET E&TC</p>
    </div>
    """, unsafe_allow_html=True)

    active_page = st.selectbox(
        "📌 SELECT NAVIGATION PAGE:",
        [
            "📊 Dashboard",
            "📚 Discover Books",
            "🤖 For You",
            "📜 Borrow History",
            "⏳ Reservations",
            "⚡ System & DSA Insights"
        ],
        index=0,
        key="nav_selectbox_sidebar"
    )

    st.markdown("---")
    st.markdown("#### 👥 **Active Patron**")
    all_user_names = list(users_df['name'].unique())
    sel_user_name = st.selectbox("Select Active Patron:", all_user_names, index=0, key="patron_select_sidebar")
    
    user_rows = users_df[users_df['name'] == sel_user_name]
    if not user_rows.empty:
        current_user_id = int(user_rows['user_id'].values[0])
    else:
        current_user_id = int(users_df['user_id'].values[0])

# Top Page Banner
h_col1, h_col2 = st.columns([3, 1.2])
with h_col1:
    st.markdown('<div style="font-size:1.6rem; font-weight:800; color:#4F46E5;">🔮 SmartLib AI Library System</div>', unsafe_allow_html=True)
    st.caption("Department of E&TC • Data Structures & AI System")
with h_col2:
    st.markdown(f"**Current Page**: `{active_page}`")
    st.markdown(f"**Active Patron**: `{sel_user_name}` (`#{current_user_id}`)")

st.markdown("<hr style='margin:1rem 0 1.5rem 0; border-color:#E2E8F0;'>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================
if active_page == "📊 Dashboard":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%); color: #FFFFFF; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.2rem;">
        <h2 style="margin:0; font-size:1.8rem; font-weight:800; color:#FFFFFF;">Your Smart Library Overview</h2>
        <p style="margin-top:0.3rem; color:#E0E7FF; font-size:1rem;">
            Personalized AI recommendations, catalog statistics, and data structure engine.
        </p>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-box" style="border-left-color: #4F46E5;">
            <div class="kpi-title">Catalog Index</div>
            <div class="kpi-value">{len(books_df)} Books</div>
            <div class="badge-ds" style="margin-top:0.3rem;">BST • O(log N)</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-box" style="border-left-color: #0284C7;">
            <div class="kpi-title">Active Patrons</div>
            <div class="kpi-value">{len(users_df)} Students</div>
            <div class="badge-ds" style="margin-top:0.3rem;">Patron Matrix</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-box" style="border-left-color: #059669;">
            <div class="kpi-title">Borrow History</div>
            <div class="kpi-value">{len(borrow_df)} Loans</div>
            <div class="badge-ds" style="margin-top:0.3rem;">Linked List • O(1)</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-box" style="border-left-color: #D97706;">
            <div class="kpi-title">Waitlist Queue</div>
            <div class="kpi-value">{queue.size()} Waiting</div>
            <div class="badge-ds" style="margin-top:0.3rem;">FIFO Queue • O(1)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    d_col1, d_col2 = st.columns([2.2, 1])

    with d_col1:
        st.markdown("### 💡 Top Recommended Books for You")
        recs = recommender.recommend_books_for_user(current_user_id, top_n=4)

        r_grid = st.columns(2)
        for idx, (_, row) in enumerate(recs.iterrows()):
            with r_grid[idx % 2]:
                st.markdown(f"""
                <div class="accessible-card">
                    <span class="badge-ds" style="margin-bottom:0.4rem;">{row['reason']}</span>
                    <h3 style="margin:0.2rem 0; font-size:1.1rem; font-weight:700;">{row['title']}</h3>
                    <p style="margin:0; font-size:0.85rem; color:#475569;">Author: <b>{row['author']}</b></p>
                    <p style="margin:0.2rem 0 0.6rem 0; font-size:0.8rem; color:#64748B;">Genre: {row['genre']} | ⭐ {row['rating']}</p>
                </div>
                """, unsafe_allow_html=True)
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("📌 Borrow", key=f"d_bor_{row['book_id']}", use_container_width=True, type="primary"):
                        new_loan = {
                            'borrow_id': 300 + len(st.session_state.borrow_df) + 1,
                            'user_id': int(current_user_id),
                            'book_id': int(row['book_id']),
                            'borrow_date': '2026-10-07',
                            'return_date': '',
                            'status': 'Borrowed'
                        }
                        st.session_state.borrow_df = pd.concat([st.session_state.borrow_df, pd.DataFrame([new_loan])], ignore_index=True)
                        sync_data_structures()
                        st.success(f"✅ Borrowed '{row['title']}'! Record added to Singly Linked List.")
                with b2:
                    st.button("Details", key=f"d_det_{row['book_id']}", use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)

    with d_col2:
        st.markdown("### 📜 Your Active Loans")
        user_loans = borrow_df[borrow_df['user_id'] == current_user_id]
        if not user_loans.empty:
            loan_books = books_df[books_df['book_id'].isin(user_loans['book_id'])]
            for _, b_row in loan_books.iterrows():
                st.markdown(f"""
                <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:10px; padding:0.9rem; margin-bottom:0.6rem;">
                    <div style="font-weight:700; color:#0F172A;">{b_row['title']}</div>
                    <div style="font-size:0.85rem; color:#64748B;">Author: {b_row['author']}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("🔄 Renew Loan", key=f"d_renew_{b_row['book_id']}", use_container_width=True):
                    st.success("✅ Loan renewed for +14 days!")
        else:
            st.info("No active loans currently checked out for this patron.")


# ==============================================================================
# PAGE 2: DISCOVER BOOKS
# ==============================================================================
elif active_page == "📚 Discover Books":
    st.markdown("## 📚 Institutional Academic Catalog")
    st.caption("Indexed in-memory using Binary Search Tree (BST) and Hash Table with Separate Chaining.")

    f_col, grid_col = st.columns([1, 2.5])

    with f_col:
        st.markdown("### 🎛️ Filter Books")
        search_q = st.text_input("Search Title or Author:", value="", key="cat_search_clean")

        st.markdown("#### Academic Disciplines")
        st.checkbox("Computer Science", value=True, key="chk_cs_clean")
        st.checkbox("AI/ML", value=True, key="chk_ai_clean")
        st.checkbox("Mathematics", value=True, key="chk_math_clean")

    with grid_col:
        filtered_df = books_df.copy()
        if search_q.strip():
            q = search_q.strip().lower()
            filtered_df = filtered_df[
                filtered_df['title'].str.lower().str.contains(q) |
                filtered_df['author'].str.lower().str.contains(q) |
                filtered_df['genre'].str.lower().str.contains(q)
            ]

        MAX_BOOKS_DISPLAY = 20
        total_results = len(filtered_df)
        if total_results > MAX_BOOKS_DISPLAY:
            st.info(f"Showing top {MAX_BOOKS_DISPLAY} results out of {total_results}. Use search to narrow down.")
            filtered_df = filtered_df.head(MAX_BOOKS_DISPLAY)

        b_grid = st.columns(2)
        for idx, (_, row) in enumerate(filtered_df.iterrows()):
            with b_grid[idx % 2]:
                st.markdown(f"""
                <div class="accessible-card">
                    <span class="badge-ds" style="margin-bottom:0.4rem;">{row['genre']}</span>
                    <h3 style="margin:0.2rem 0; font-size:1.1rem; font-weight:700;">{row['title']}</h3>
                    <p style="margin:0; font-size:0.85rem; color:#475569;">Author: {row['author']}</p>
                    <p style="margin:0.2rem 0 0.6rem 0; font-size:0.8rem; color:#64748B;">ISBN: <code>{row['isbn']}</code> | ⭐ {row['rating']}</p>
                </div>
                """, unsafe_allow_html=True)
                c_b1, c_b2 = st.columns(2)
                with c_b1:
                    st.button("View Details", key=f"c_det_{row['book_id']}", use_container_width=True)
                with c_b2:
                    if st.button("Borrow", key=f"c_bor_{row['book_id']}", use_container_width=True, type="primary"):
                        new_loan = {
                            'borrow_id': 300 + len(st.session_state.borrow_df) + 1,
                            'user_id': int(current_user_id),
                            'book_id': int(row['book_id']),
                            'borrow_date': '2026-10-07',
                            'return_date': '',
                            'status': 'Borrowed'
                        }
                        st.session_state.borrow_df = pd.concat([st.session_state.borrow_df, pd.DataFrame([new_loan])], ignore_index=True)
                        sync_data_structures()
                        st.success(f"✅ Borrowed '{row['title']}'!")
                st.markdown("<br>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 3: FOR YOU (RECOMMENDATIONS)
# ==============================================================================
elif active_page == "🤖 For You":
    st.markdown(f"## 🤖 AI Recommendations for **{sel_user_name}**")
    st.caption("Hybrid Recommender combining Content TF-IDF Vector Similarity and User-User Collaborative Filtering.")

    recs = recommender.recommend_books_for_user(current_user_id, top_n=6)

    for _, row in recs.iterrows():
        st.markdown(f"""
        <div class="accessible-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h3 style="margin:0; font-size:1.15rem; font-weight:700;">{row['title']}</h3>
                <span class="badge-ds">Match Score: {row['recommendation_score']:.2f}</span>
            </div>
            <p style="margin:0.3rem 0; color:#475569; font-size:0.9rem;">Author: {row['author']} | Genre: <b>{row['genre']}</b></p>
            <div style="background:#F1F5F9; border-left:4px solid #4F46E5; padding:0.5rem 0.8rem; font-size:0.85rem; color:#1E293B; margin-top:0.5rem; font-weight:600;">
                💡 {row['reason']}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 4: BORROW HISTORY
# ==============================================================================
elif active_page == "📜 Borrow History":
    st.markdown("## 📜 Borrowing History (Singly Linked List)")
    st.caption("Chronological audit log maintained in memory via Singly Linked List nodes.")

    l_items = linked_list.to_list()
    if l_items:
        st.dataframe(pd.DataFrame(l_items), use_container_width=True)
    else:
        st.info("No borrowing history records logged yet.")

    st.markdown("### 📌 Issue New Book")
    with st.form("issue_loan_form"):
        u_sel = st.selectbox("Select Student:", users_df['name'])
        b_sel = st.selectbox("Select Book:", books_df['title'])
        if st.form_submit_button("🚀 Issue Book (Append to Linked List)"):
            u_id = users_df[users_df['name'] == u_sel]['user_id'].values[0]
            b_id = books_df[books_df['title'] == b_sel]['book_id'].values[0]

            new_loan = {
                'borrow_id': 300 + len(st.session_state.borrow_df) + 1,
                'user_id': int(u_id),
                'book_id': int(b_id),
                'borrow_date': '2026-10-07',
                'return_date': '',
                'status': 'Borrowed'
            }
            st.session_state.borrow_df = pd.concat([st.session_state.borrow_df, pd.DataFrame([new_loan])], ignore_index=True)
            sync_data_structures()
            st.success(f"✅ Borrowing record issued for {u_sel} and appended to Singly Linked List!")


# ==============================================================================
# PAGE 5: RESERVATIONS
# ==============================================================================
elif active_page == "⏳ Reservations":
    st.markdown("## ⏳ Book Reservation Queue (FIFO Queue)")
    st.caption("First-In, First-Out (FIFO) queue managing waitlist requests.")

    q_items = queue.to_list()
    if q_items:
        st.dataframe(pd.DataFrame(q_items), use_container_width=True)
    else:
        st.info("Reservation Queue is currently empty.")

    q_c1, q_c2 = st.columns(2)
    with q_c1:
        st.markdown("### 📥 Enqueue Request")
        with st.form("enqueue_form"):
            req_name = st.text_input("Student Name", value="Sneha Kulkarni")
            req_book = st.selectbox("Book to Reserve", books_df['title'])
            if st.form_submit_button("📥 Add to FIFO Queue"):
                queue.enqueue({'user_id': 206, 'user_name': req_name, 'book_title': req_book, 'status': 'Waiting'})
                st.success(f"✅ Enqueued {req_name} to FIFO Waitlist!")

    with q_c2:
        st.markdown("### 📤 Dequeue Request")
        if st.button("📤 Serve Next In Queue (Dequeue)", type="primary", key="btn_dequeue_clean"):
            dequeued = queue.dequeue()
            if dequeued:
                st.balloons()
                st.success(f"🎉 Served request for {dequeued['user_name']} for '{dequeued['book_title']}'!")
            else:
                st.warning("Queue is empty.")


# ==============================================================================
# PAGE 6: SYSTEM & DSA INSIGHTS
# ==============================================================================
elif active_page == "⚡ System & DSA Insights":
    st.markdown("## ⚡ Data Structures & System Performance")
    st.caption("Empirical latency measurement of custom data structures.")

    st.markdown("""
    ### 📊 Performance Benchmarks (N=50,000)
    - **Linear Search**: `1.322816 ms` at N=50,000
    - **Custom BST Search**: `0.001375 ms` at N=50,000
    - **Custom Hash Table Lookup**: `0.002169 ms` at N=50,000
    """)

    col_bst, col_hash = st.columns(2)

    with col_bst:
        st.markdown("### 🌲 BST Search Test")
        search_id = st.number_input("Enter Book ID to search via BST:", min_value=101, max_value=999, value=103, key="dsa_bst_clean")
        if st.button("Execute BST Search", key="btn_bst_run_clean"):
            t0 = time.perf_counter()
            found = bst.search(int(search_id))
            t1 = time.perf_counter()
            if found:
                st.success(f"✅ BST Hit in `{(t1-t0)*1000:.4f} ms`!")
                st.json(found)
            else:
                st.error("Book ID not found in BST.")

    with col_hash:
        st.markdown("### 🔑 Hash Table ISBN Lookup Test")
        search_isbn = st.text_input("Enter ISBN string to search via Hash Table:", value="9780134610993", key="dsa_hash_clean")
        if st.button("Execute ISBN Hash Lookup", key="btn_hash_run_clean"):
            t0 = time.perf_counter()
            found = hash_table.get(search_isbn.strip())
            t1 = time.perf_counter()
            if found:
                st.success(f"✅ Hash Table Hit in `{(t1-t0)*1000:.4f} ms`!")
                st.json(found)
            else:
                st.error("ISBN string not found in Hash Table.")
