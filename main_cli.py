"""
Terminal CLI Application for AI Library Recommendation System.
Provides an interactive menu for searching books, viewing user history, reservation queues, and AI recommendations.
"""

import pandas as pd
import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ds import LinkedList, BinarySearchTree, HashTable, Queue
from ml import LibraryRecommender

def main():
    print("=" * 65)
    print("   AI-BASED LIBRARY RECOMMENDATION SYSTEM (NMIET TOPIC NO. 3)")
    print("=" * 65)

    # 1. Load Data
    books_df = pd.read_csv('data/books.csv')
    users_df = pd.read_csv('data/users.csv')
    borrow_df = pd.read_csv('data/borrowing_history.csv')

    # 2. Instantiate Data Structures
    bst = BinarySearchTree()
    hash_table = HashTable()
    linked_list_history = LinkedList()
    reservation_queue = Queue()

    for _, row in books_df.iterrows():
        b_dict = row.to_dict()
        bst.insert(int(row['book_id']), b_dict)
        hash_table.put(str(row['isbn']), b_dict)

    for _, row in borrow_df.iterrows():
        linked_list_history.append(row.to_dict())

    # Pre-populate sample queue
    reservation_queue.enqueue({'user_id': 203, 'book_id': 103, 'user_name': 'Rohan Verma'})
    reservation_queue.enqueue({'user_id': 205, 'book_id': 101, 'user_name': 'Aditya Deshmukh'})

    # 3. Instantiate Recommender
    recommender = LibraryRecommender(books_df, users_df, borrow_df)

    while True:
        print("\n---------------- MENU ----------------")
        print("1. View Book Catalog (BST In-Order Traversal)")
        print("2. Search Book by ID or Title (BST)")
        print("3. Instant Book Lookup by ISBN (Hash Table O(1))")
        print("4. View User Borrowing History (Linked List)")
        print("5. View / Process Reservation Queue (FIFO Queue)")
        print("6. Get AI Personalized Book Recommendations")
        print("7. Exit")

        choice = input("Enter choice (1-7): ").strip()

        if choice == '1':
            print("\n=== BOOK CATALOG (BST SORTED BY BOOK ID) ===")
            all_books = bst.inorder()
            for b_id, b in all_books:
                print(f" ID: {b_id} | {b['title'][:32]:<32} | {b['genre']:<15} | Rating: {b['rating']}/5.0")

        elif choice == '2':
            query = input("Enter Book ID or Title query: ").strip()
            if query.isdigit():
                b = bst.search(int(query))
                if b:
                    print(f"\n[BST Found] ID: {b['book_id']} | Title: {b['title']} | Author: {b['author']} | Genre: {b['genre']}")
                else:
                    print("Book ID not found in BST.")
            else:
                matches = bst.search_by_title_substring(query)
                print(f"\nFound {len(matches)} matching books in BST:")
                for b in matches:
                    print(f" - ID: {b['book_id']} | Title: {b['title']} | Author: {b['author']}")

        elif choice == '3':
            isbn = input("Enter ISBN (e.g. 978-0134610993): ").strip()
            b = hash_table.get(isbn)
            if b:
                print(f"\n[Hash Table O(1) HIT] ISBN: {isbn}")
                print(f" Title: {b['title']}\n Author: {b['author']}\n Genre: {b['genre']}\n Rating: {b['rating']}")
            else:
                print(f"ISBN '{isbn}' not found in Hash Table.")

        elif choice == '4':
            print("\n=== USER BORROWING HISTORY (LINKED LIST) ===")
            u_id = input("Enter User ID (e.g. 201, 202, 207) or press Enter for all: ").strip()
            records = linked_list_history.to_list()
            if u_id.isdigit():
                records = [r for r in records if int(r['user_id']) == int(u_id)]
            
            print(f"Total Transactions: {len(records)}")
            for r in records:
                print(f" Tx #{r['transaction_id']} | User ID: {r['user_id']} | Book ID: {r['book_id']} | Date: {r['borrow_date']} | Status: {r['status']}")

        elif choice == '5':
            print("\n=== RESERVATION QUEUE ===")
            q_items = reservation_queue.to_list()
            if not q_items:
                print("Queue is empty.")
            else:
                for idx, item in enumerate(q_items, 1):
                    print(f" Position #{idx}: User ID {item['user_id']} ({item.get('user_name', '')}) -> Requested Book ID {item['book_id']}")
                
                sub = input("\nProcess (dequeue) top request? (y/n): ").strip().lower()
                if sub == 'y':
                    processed = reservation_queue.dequeue()
                    print(f"[DEQUEUED] Processed reservation for User {processed['user_id']} (Book ID {processed['book_id']})")

        elif choice == '6':
            u_id = input("Enter User ID for AI Recommendations (e.g. 201, 202, 203, 207): ").strip()
            if u_id.isdigit():
                user_id = int(u_id)
                recs = recommender.recommend_books_for_user(user_id, top_n=5)
                print(f"\n=== AI RECOMMENDATIONS FOR USER {user_id} ===")
                for idx, row in recs.iterrows():
                    print(f" [{row['book_id']}] {row['title']} ({row['genre']})")
                    print(f"     Match Score: {row['recommendation_score']} | Reason: {row['reason']}")

        elif choice == '7':
            print("Exiting CLI application. Goodbye!")
            break
        else:
            print("Invalid option. Please choose 1-7.")

if __name__ == "__main__":
    main()
