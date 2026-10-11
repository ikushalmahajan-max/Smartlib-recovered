"""
Custom FIFO Queue Implementation using Linked Nodes & Per-Book Reservation Queue Manager.
Used for FIFO Book Reservation management when requested books are currently borrowed.
"""

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
        """Add an item to the rear of the queue in O(1) time."""
        new_node = QueueNode(data)
        if self.rear is None:
            self.front = self.rear = new_node
        else:
            self.rear.next = new_node
            self.rear = new_node
        self._count += 1

    def dequeue(self):
        """Remove and return the front item from the queue in O(1) time."""
        if self.is_empty():
            return None
        temp = self.front
        self.front = temp.next
        if self.front is None:
            self.rear = None
        self._count -= 1
        return temp.data

    def peek(self):
        """Look at the front item without removing it. O(1)."""
        return self.front.data if self.front else None

    def is_empty(self):
        return self.front is None

    def size(self):
        return self._count

    def to_list(self):
        """Return list of elements from front to rear."""
        elements = []
        curr = self.front
        while curr:
            elements.append(curr.data)
            curr = curr.next
        return elements


class BookReservationManager:
    """
    Manages per-book FIFO reservation queues (book_id -> Queue).
    Preserves O(1) enqueue and dequeue performance for each book's queue.
    """
    def __init__(self):
        self.queues = {}  # book_id -> Queue

    def get_queue(self, book_id):
        book_id = int(book_id)
        if book_id not in self.queues:
            self.queues[book_id] = Queue()
        return self.queues[book_id]

    def enqueue_reservation(self, book_id, user_id, user_name=""):
        """Enqueue a user reservation for a specific book_id in O(1)."""
        book_id = int(book_id)
        q = self.get_queue(book_id)
        record = {
            'book_id': book_id,
            'user_id': int(user_id),
            'user_name': user_name
        }
        q.enqueue(record)
        return record

    def dequeue_reservation(self, book_id):
        """Dequeue the next user in line for a specific book_id in O(1)."""
        book_id = int(book_id)
        if book_id not in self.queues or self.queues[book_id].is_empty():
            return None
        return self.queues[book_id].dequeue()

    def peek_reservation(self, book_id):
        """Peek at the next user in line for a specific book_id."""
        book_id = int(book_id)
        if book_id not in self.queues:
            return None
        return self.queues[book_id].peek()

    def queue_size(self, book_id):
        """Get length of reservation queue for a given book_id."""
        book_id = int(book_id)
        if book_id not in self.queues:
            return 0
        return self.queues[book_id].size()

    def get_all_active_reservations(self):
        """Return mapping of book_id -> list of reservation records."""
        result = {}
        for b_id, q in self.queues.items():
            if not q.is_empty():
                result[b_id] = q.to_list()
        return result
