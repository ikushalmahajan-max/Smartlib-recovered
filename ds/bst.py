"""
Custom Binary Search Tree (BST) Implementation.
Used to index books by unique Book ID for O(log N) search and sorted in-order traversal.
Uses iterative insertion and search for maximum efficiency and stack safety.
"""

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
        """Insert a key-value pair into the BST iteratively."""
        key = int(key)
        new_node = BSTNode(key, value)
        
        if self.root is None:
            self.root = new_node
            self._count += 1
            return

        curr = self.root
        while curr:
            if key < curr.key:
                if curr.left is None:
                    curr.left = new_node
                    self._count += 1
                    return
                curr = curr.left
            elif key > curr.key:
                if curr.right is None:
                    curr.right = new_node
                    self._count += 1
                    return
                curr = curr.right
            else:
                # Key already exists: update payload value
                curr.value = value
                return

    def search(self, key):
        """Search for a book by key in O(log N) average time iteratively."""
        key = int(key)
        curr = self.root
        while curr:
            if key == curr.key:
                return curr.value
            elif key < curr.key:
                curr = curr.left
            else:
                curr = curr.right
        return None

    def inorder(self):
        """Return all items in sorted order of key via iterative In-Order Traversal."""
        result = []
        stack = []
        curr = self.root

        while curr or stack:
            while curr:
                stack.append(curr)
                curr = curr.left
            curr = stack.pop()
            result.append((curr.key, curr.value))
            curr = curr.right

        return result

    def search_by_title_substring(self, query):
        """Find books where query matches part of title, author, or genre."""
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
