"""
Custom Hash Table Implementation with Separate Chaining.
Used for O(1) instantaneous lookup of book metadata using ISBN or Book ID.
"""

class HashNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None

class HashTable:
    def __init__(self, capacity=64):
        self.capacity = capacity
        self.buckets = [None] * self.capacity
        self._size = 0

    def _hash(self, key):
        """Polynomial rolling hash function."""
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
        """Insert or update key-value pair into hash table."""
        index = self._hash(key)
        head = self.buckets[index]

        curr = head
        while curr:
            if curr.key == key:
                curr.value = value
                return
            curr = curr.next

        new_node = HashNode(key, value)
        new_node.next = head
        self.buckets[index] = new_node
        self._size += 1

    def get(self, key):
        """Retrieve value by key in O(1) average time."""
        index = self._hash(key)
        curr = self.buckets[index]
        while curr:
            if curr.key == key:
                return curr.value
            curr = curr.next
        return None

    def contains(self, key):
        return self.get(key) is not None

    def remove(self, key):
        """Delete key from hash table."""
        index = self._hash(key)
        curr = self.buckets[index]
        prev = None

        while curr:
            if curr.key == key:
                if prev:
                    prev.next = curr.next
                else:
                    self.buckets[index] = curr.next
                self._size -= 1
                return True
            prev = curr
            curr = curr.next
        return False

    def values(self):
        """Return all stored values."""
        vals = []
        for bucket in self.buckets:
            curr = bucket
            while curr:
                vals.append(curr.value)
                curr = curr.next
        return vals

    def size(self):
        return self._size
