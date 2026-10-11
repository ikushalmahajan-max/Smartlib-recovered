"""
Custom Singly Linked List Implementation with O(1) Tail Pointer.
Used to track dynamic user borrowing history and activity logs in chronological order.
"""

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self._count = 0

    def append(self, data):
        """Append data to the end of the linked list in O(1) time using tail pointer."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self._count += 1

    def prepend(self, data):
        """Prepend data to the head of the linked list in O(1) time."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head = new_node
        self._count += 1

    def search(self, match_fn):
        """Search for nodes matching a predicate function."""
        results = []
        curr = self.head
        while curr:
            if match_fn(curr.data):
                results.append(curr.data)
            curr = curr.next
        return results

    def delete(self, match_fn):
        """Delete the first node matching a predicate function, updating head and tail correctly."""
        curr = self.head
        prev = None
        while curr:
            if match_fn(curr.data):
                if prev:
                    prev.next = curr.next
                else:
                    self.head = curr.next
                
                # If deleted node was tail, update tail pointer to prev
                if curr == self.tail:
                    self.tail = prev
                
                self._count -= 1
                return True
            prev = curr
            curr = curr.next
        return False

    def clear(self):
        """Clear all nodes from the linked list."""
        self.head = None
        self.tail = None
        self._count = 0

    def to_list(self):
        """Convert linked list elements into a Python list."""
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
