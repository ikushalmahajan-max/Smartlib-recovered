/*
 * C++ Data Structures Empirical Performance Benchmark Executable
 * Compares BST Search, Hash Table Lookup, and Queue Enqueue/Dequeue operations.
 * Outputs CSV formatted timing data for Python vs C++ benchmarking.
 */

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>

using namespace std;
using namespace std::chrono;

// --- 1. DATA STRUCTURES ---
struct Book {
    int id;
    string title;
    string isbn;
};

struct BSTNode {
    int key;
    Book value;
    BSTNode* left;
    BSTNode* right;
    BSTNode(int k, Book v) : key(k), value(v), left(nullptr), right(nullptr) {}
};

class BinarySearchTree {
public:
    BSTNode* root;
    BinarySearchTree() : root(nullptr) {}

    void insert(int key, Book value) {
        BSTNode* newNode = new BSTNode(key, value);
        if (!root) {
            root = newNode;
            return;
        }
        BSTNode* curr = root;
        while (curr) {
            if (key < curr->key) {
                if (!curr->left) { curr->left = newNode; return; }
                curr = curr->left;
            } else if (key > curr->key) {
                if (!curr->right) { curr->right = newNode; return; }
                curr = curr->right;
            } else {
                curr->value = value;
                return;
            }
        }
    }

    Book* search(int key) const {
        BSTNode* curr = root;
        while (curr) {
            if (key == curr->key) return const_cast<Book*>(&(curr->value));
            if (key < curr->key) curr = curr->left;
            else curr = curr->right;
        }
        return nullptr;
    }
};

struct HashNode {
    string key;
    Book value;
    HashNode* next;
    HashNode(string k, Book v) : key(k), value(v), next(nullptr) {}
};

class HashTable {
private:
    int capacity;
    vector<HashNode*> buckets;

    int hashFunc(const string& key) const {
        unsigned long hash = 5381;
        for (char c : key) hash = ((hash << 5) + hash) + c;
        return hash % capacity;
    }

public:
    HashTable(int cap = 64) : capacity(cap), buckets(cap, nullptr) {}

    void put(string key, Book value) {
        int idx = hashFunc(key);
        HashNode* curr = buckets[idx];
        while (curr) {
            if (curr->key == key) { curr->value = value; return; }
            curr = curr->next;
        }
        HashNode* newNode = new HashNode(key, value);
        newNode->next = buckets[idx];
        buckets[idx] = newNode;
    }

    Book* get(const string& key) const {
        int idx = hashFunc(key);
        HashNode* curr = buckets[idx];
        while (curr) {
            if (curr->key == key) return const_cast<Book*>(&(curr->value));
            curr = curr->next;
        }
        return nullptr;
    }
};

int main(int argc, char* argv[]) {
    int size = (argc > 1) ? atoi(argv[1]) : 10000;
    int iterations = (argc > 2) ? atoi(argv[2]) : 1000;

    mt19937 rng(42);

    vector<Book> books;
    for (int i = 1; i <= size; ++i) {
        int b_id = i + 100;
        string isbn = "978-013" + to_string(i);
        books.push_back({b_id, "Synthetic Book " + to_string(i), isbn});
    }

    BinarySearchTree bst;
    HashTable ht(max(64, static_cast<int>(size * 1.5)));

    vector<Book> shuffled = books;
    shuffle(shuffled.begin(), shuffled.end(), rng);

    for (const auto& b : shuffled) bst.insert(b.id, b);
    for (const auto& b : books) ht.put(b.isbn, b);

    uniform_int_distribution<int> dist(0, size - 1);
    vector<int> target_ids;
    vector<string> target_isbns;
    for (int i = 0; i < iterations; ++i) {
        int idx = dist(rng);
        target_ids.push_back(books[idx].id);
        target_isbns.push_back(books[idx].isbn);
    }

    // Benchmark BST Search
    auto t0 = high_resolution_clock::now();
    for (int t_id : target_ids) {
        volatile Book* b = bst.search(t_id);
        (void)b;
    }
    auto t1 = high_resolution_clock::now();
    double bst_ms = duration<double, milli>(t1 - t0).count();

    // Benchmark Hash Lookup
    t0 = high_resolution_clock::now();
    for (const auto& t_isbn : target_isbns) {
        volatile Book* b = ht.get(t_isbn);
        (void)b;
    }
    t1 = high_resolution_clock::now();
    double hash_ms = duration<double, milli>(t1 - t0).count();

    cout << fixed << setprecision(6);
    cout << "C++,BST Search," << size << "," << iterations << "," << (bst_ms / iterations) << "\n";
    cout << "C++,Hash Lookup," << size << "," << iterations << "," << (hash_ms / iterations) << "\n";

    return 0;
}
