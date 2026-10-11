/*
 * NMIET Second Year Mini Project - Department of E&TC
 * Topic 3: AI-Based Library Recommendation System & Data Structures Engine
 * Pure C++ Implementation (Single File, Custom Data Structures, No Demo Data)
 * 
 * Implemented Data Structures:
 *  1. Singly Linked List: Dynamic Borrowing History Log
 *  2. Binary Search Tree (BST): Book Catalog Indexing & Sorted Traversal (O(log N))
 *  3. Hash Table with Separate Chaining: Fast ISBN Lookup (O(1))
 *  4. FIFO Queue: Book Reservation Waiting List
 */

#include <iostream>
#include <string>
#include <iomanip>

using namespace std;

// ==============================================================================
// 1. DATA MODELS
// ==============================================================================

struct Book {
    int id;
    string title;
    string author;
    string genre;
    string isbn;
    double rating;
};

struct BorrowRecord {
    int transaction_id;
    int user_id;
    int book_id;
    string borrow_date;
};

// ==============================================================================
// 2. SINGLY LINKED LIST IMPLEMENTATION (FOR BORROWING HISTORY)
// ==============================================================================

struct LLNode {
    BorrowRecord record;
    LLNode* next;
    LLNode(BorrowRecord rec) : record(rec), next(nullptr) {}
};

class LinkedList {
private:
    LLNode* head;
    int count;

public:
    LinkedList() : head(nullptr), count(0) {}

    ~LinkedList() {
        LLNode* curr = head;
        while (curr) {
            LLNode* temp = curr;
            curr = curr->next;
            delete temp;
        }
    }

    void append(BorrowRecord rec) {
        LLNode* newNode = new LLNode(rec);
        if (!head) {
            head = newNode;
        } else {
            LLNode* temp = head;
            while (temp->next) {
                temp = temp->next;
            }
            temp->next = newNode;
        }
        count++;
    }

    void displayAll() const {
        if (!head) {
            cout << "\n[LINKED LIST] Borrowing history is empty." << endl;
            return;
        }
        cout << "\n=======================================================" << endl;
        cout << "   BORROW TRANSACTION LOG (SINGLY LINKED LIST)        " << endl;
        cout << "=======================================================" << endl;
        LLNode* curr = head;
        while (curr) {
            cout << "  [Tx #" << setw(3) << curr->record.transaction_id << "] "
                 << "User ID: " << setw(4) << curr->record.user_id
                 << " | Book ID: " << setw(4) << curr->record.book_id
                 << " | Date: " << curr->record.borrow_date << endl;
            curr = curr->next;
        }
    }

    int size() const { return count; }
};

// ==============================================================================
// 3. BINARY SEARCH TREE (BST) IMPLEMENTATION (FOR BOOK CATALOG)
// ==============================================================================

struct BSTNode {
    Book book;
    BSTNode* left;
    BSTNode* right;

    BSTNode(Book b) : book(b), left(nullptr), right(nullptr) {}
};

class BinarySearchTree {
private:
    BSTNode* root;
    int count;

    BSTNode* insertRec(BSTNode* node, Book b) {
        if (!node) {
            count++;
            return new BSTNode(b);
        }
        if (b.id < node->book.id) {
            node->left = insertRec(node->left, b);
        } else if (b.id > node->book.id) {
            node->right = insertRec(node->right, b);
        } else {
            node->book = b; // Update duplicate ID
        }
        return node;
    }

    BSTNode* searchRec(BSTNode* node, int bookId) const {
        if (!node || node->book.id == bookId) return node;
        if (bookId < node->book.id) return searchRec(node->left, bookId);
        return searchRec(node->right, bookId);
    }

    void inorderRec(BSTNode* node) const {
        if (node) {
            inorderRec(node->left);
            cout << "  [ID: " << setw(4) << node->book.id << "] "
                 << setw(30) << left << node->book.title
                 << " | Genre: " << setw(15) << node->book.genre
                 << " | ISBN: " << setw(14) << node->book.isbn
                 << " | Rating: " << fixed << setprecision(1) << node->book.rating << "/5.0" << endl;
            inorderRec(node->right);
        }
    }

    void clearRec(BSTNode* node) {
        if (node) {
            clearRec(node->left);
            clearRec(node->right);
            delete node;
        }
    }

public:
    BinarySearchTree() : root(nullptr), count(0) {}
    ~BinarySearchTree() { clearRec(root); }

    void insert(Book b) { root = insertRec(root, b); }

    Book* search(int bookId) const {
        BSTNode* res = searchRec(root, bookId);
        return res ? &(res->book) : nullptr;
    }

    void displayInOrder() const {
        if (!root) {
            cout << "\n[BST] Book catalog is empty." << endl;
            return;
        }
        cout << "\n=======================================================" << endl;
        cout << "   BOOK CATALOG (BST IN-ORDER TRAVERSAL SORTED BY ID) " << endl;
        cout << "=======================================================" << endl;
        inorderRec(root);
    }

    int size() const { return count; }
};

// ==============================================================================
// 4. HASH TABLE IMPLEMENTATION (SEPARATE CHAINING FOR O(1) ISBN LOOKUP)
// ==============================================================================

struct HashNode {
    string isbn;
    Book book;
    HashNode* next;

    HashNode(string k, Book b) : isbn(k), book(b), next(nullptr) {}
};

class HashTable {
private:
    static const int TABLE_SIZE = 37;
    HashNode* buckets[TABLE_SIZE];

    int hashFunction(const string& key) const {
        unsigned long hash = 5381;
        for (char c : key) {
            hash = ((hash << 5) + hash) + c; // djb2 polynomial hash
        }
        return hash % TABLE_SIZE;
    }

public:
    HashTable() {
        for (int i = 0; i < TABLE_SIZE; i++) buckets[i] = nullptr;
    }

    ~HashTable() {
        for (int i = 0; i < TABLE_SIZE; i++) {
            HashNode* curr = buckets[i];
            while (curr) {
                HashNode* temp = curr;
                curr = curr->next;
                delete temp;
            }
        }
    }

    void insert(string isbn, Book b) {
        int index = hashFunction(isbn);
        HashNode* head = buckets[index];
        HashNode* curr = head;

        while (curr) {
            if (curr->isbn == isbn) {
                curr->book = b;
                return;
            }
            curr = curr->next;
        }

        HashNode* newNode = new HashNode(isbn, b);
        newNode->next = head;
        buckets[index] = newNode;
    }

    Book* get(string isbn) const {
        int index = hashFunction(isbn);
        HashNode* curr = buckets[index];
        while (curr) {
            if (curr->isbn == isbn) return &(curr->book);
            curr = curr->next;
        }
        return nullptr;
    }
};

// ==============================================================================
// 5. FIFO QUEUE IMPLEMENTATION (FOR RESERVATION WAITING LIST)
// ==============================================================================

struct QueueNode {
    int user_id;
    int book_id;
    string student_name;
    QueueNode* next;

    QueueNode(int u, int b, string name) : user_id(u), book_id(b), student_name(name), next(nullptr) {}
};

class Queue {
private:
    QueueNode* front;
    QueueNode* rear;
    int count;

public:
    Queue() : front(nullptr), rear(nullptr), count(0) {}

    ~Queue() {
        while (!isEmpty()) dequeue();
    }

    void enqueue(int userId, int bookId, string name) {
        QueueNode* newNode = new QueueNode(userId, bookId, name);
        if (isEmpty()) {
            front = rear = newNode;
        } else {
            rear->next = newNode;
            rear = newNode;
        }
        count++;
        cout << "✅ [QUEUE] Added " << name << " (User ID: " << userId << ") to waiting queue for Book ID " << bookId << "." << endl;
    }

    bool dequeue() {
        if (isEmpty()) return false;
        QueueNode* temp = front;
        cout << "🎉 [QUEUE DEQUEUE] Served request for " << temp->student_name << " (User ID: " << temp->user_id << ") for Book ID " << temp->book_id << "." << endl;
        front = front->next;
        if (!front) rear = nullptr;
        delete temp;
        count--;
        return true;
    }

    bool isEmpty() const { return front == nullptr; }

    void displayQueue() const {
        if (isEmpty()) {
            cout << "\n[QUEUE] Book reservation queue is currently empty." << endl;
            return;
        }
        cout << "\n=======================================================" << endl;
        cout << "   BOOK RESERVATION WAITING LIST (FIFO QUEUE)          " << endl;
        cout << "=======================================================" << endl;
        QueueNode* curr = front;
        int pos = 1;
        while (curr) {
            cout << "  Position #" << pos++ << " -> Student: " << setw(20) << left << curr->student_name
                 << " (User ID: " << curr->user_id << ") | Reserved Book ID: " << curr->book_id << endl;
            curr = curr->next;
        }
    }

    int size() const { return count; }
};

// ==============================================================================
// 6. MAIN INTERACTIVE CONSOLE SYSTEM
// ==============================================================================

int main() {
    BinarySearchTree bst;
    LinkedList history;
    HashTable hashTable;
    Queue reservationQueue;

    int choice;
    int autoTxId = 1001;

    cout << "=================================================================\n";
    cout << "   NMIET DEPARTMENT OF E&TC - SECOND YEAR MINI PROJECT           \n";
    cout << "   Topic 3: AI-Based Library System (Data Structures Engine)    \n";
    cout << "=================================================================\n";

    do {
        cout << "\n=================== MAIN MENU ===================\n";
        cout << " 1. Add New Book (Insert into BST & Hash Table)\n";
        cout << " 2. Search Book by ID (BST O(log N))\n";
        cout << " 3. Lookup Book by ISBN (Hash Table O(1))\n";
        cout << " 4. Display All Books (BST In-Order Traversal)\n";
        cout << " 5. Issue Book (Append to Singly Linked List)\n";
        cout << " 6. Display Borrowing History Log (Linked List)\n";
        cout << " 7. Add Book Reservation (Enqueue to FIFO Queue)\n";
        cout << " 8. Serve Next Reservation (Dequeue from Queue)\n";
        cout << " 9. Display Reservation Waiting List (FIFO Queue)\n";
        cout << " 0. Exit Application\n";
        cout << "-------------------------------------------------\n";
        cout << "Enter your choice [0-9]: ";
        if (!(cin >> choice)) {
            cin.clear();
            cin.ignore(10000, '\n');
            continue;
        }

        switch (choice) {
            case 1: {
                Book b;
                cout << "\n--- Add New Book ---\n";
                cout << "Enter Book ID (e.g., 101): ";
                cin >> b.id;
                cin.ignore();
                cout << "Enter Book Title: ";
                getline(cin, b.title);
                cout << "Enter Author Name: ";
                getline(cin, b.author);
                cout << "Enter Genre: ";
                getline(cin, b.genre);
                cout << "Enter ISBN String: ";
                getline(cin, b.isbn);
                cout << "Enter Rating (1.0 - 5.0): ";
                cin >> b.rating;

                bst.insert(b);
                hashTable.insert(b.isbn, b);
                cout << "✅ Book '" << b.title << "' added successfully to BST & Hash Table!\n";
                break;
            }
            case 2: {
                int searchId;
                cout << "\n--- BST Search by Book ID ---\n";
                cout << "Enter Book ID: ";
                cin >> searchId;
                Book* found = bst.search(searchId);
                if (found) {
                    cout << "✅ [BST HIT] Found Book:\n";
                    cout << "   ID: " << found->id << " | Title: " << found->title 
                         << " | Author: " << found->author << " | Genre: " << found->genre 
                         << " | Rating: " << found->rating << "/5.0\n";
                } else {
                    cout << "❌ Book ID " << searchId << " not found in BST.\n";
                }
                break;
            }
            case 3: {
                string searchIsbn;
                cout << "\n--- Hash Table Lookup by ISBN ---\n";
                cout << "Enter ISBN: ";
                cin.ignore();
                getline(cin, searchIsbn);
                Book* found = hashTable.get(searchIsbn);
                if (found) {
                    cout << "✅ [HASH TABLE O(1) HIT] Found Book:\n";
                    cout << "   ISBN: " << found->isbn << " | Title: " << found->title 
                         << " | Author: " << found->author << " | Rating: " << found->rating << "/5.0\n";
                } else {
                    cout << "❌ ISBN '" << searchIsbn << "' not found in Hash Table.\n";
                }
                break;
            }
            case 4:
                bst.displayInOrder();
                break;
            case 5: {
                BorrowRecord rec;
                rec.transaction_id = autoTxId++;
                cout << "\n--- Issue Book ---\n";
                cout << "Enter Student User ID: ";
                cin >> rec.user_id;
                cout << "Enter Book ID: ";
                cin >> rec.book_id;
                cout << "Enter Borrow Date (YYYY-MM-DD): ";
                cin.ignore();
                getline(cin, rec.borrow_date);

                history.append(rec);
                cout << "✅ Issued Book ID " << rec.book_id << " to User ID " << rec.user_id << " (Tx #" << rec.transaction_id << "). Added to Linked List!\n";
                break;
            }
            case 6:
                history.displayAll();
                break;
            case 7: {
                int uId, bId;
                string name;
                cout << "\n--- Add Reservation to FIFO Queue ---\n";
                cout << "Enter Student User ID: ";
                cin >> uId;
                cin.ignore();
                cout << "Enter Student Name: ";
                getline(cin, name);
                cout << "Enter Book ID to Reserve: ";
                cin >> bId;

                reservationQueue.enqueue(uId, bId, name);
                break;
            }
            case 8:
                if (!reservationQueue.dequeue()) {
                    cout << "⚠️ Reservation queue is empty. Nothing to dequeue.\n";
                }
                break;
            case 9:
                reservationQueue.displayQueue();
                break;
            case 0:
                cout << "\nExiting LibAI Data Structures Engine. Goodbye!\n";
                break;
            default:
                cout << "Invalid choice! Please select 0 to 9.\n";
        }
    } while (choice != 0);

    return 0;
}
