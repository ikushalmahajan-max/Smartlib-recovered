"""
Unit tests for Custom Data Structures:
LinkedList with O(1) Tail Pointer, BST, Hash Table, Queue, and BookReservationManager.
"""

import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ds import LinkedList, BinarySearchTree, HashTable, Queue, BookReservationManager


class TestDataStructures(unittest.TestCase):

    def test_linked_list_tail_behavior(self):
        ll = LinkedList()
        self.assertTrue(ll.is_empty())
        self.assertEqual(ll.size(), 0)
        self.assertIsNone(ll.head)
        self.assertIsNone(ll.tail)

        # Append first item: head and tail should be identical
        ll.append({'tx': 1, 'user': 201})
        self.assertEqual(ll.size(), 1)
        self.assertEqual(ll.head.data['tx'], 1)
        self.assertEqual(ll.tail.data['tx'], 1)
        self.assertIs(ll.head, ll.tail)

        # Append second item: tail should update to new node
        ll.append({'tx': 2, 'user': 202})
        self.assertEqual(ll.size(), 2)
        self.assertEqual(ll.head.data['tx'], 1)
        self.assertEqual(ll.tail.data['tx'], 2)

        # Prepend item: head updates, tail remains item 2
        ll.prepend({'tx': 0, 'user': 200})
        self.assertEqual(ll.size(), 3)
        self.assertEqual(ll.head.data['tx'], 0)
        self.assertEqual(ll.tail.data['tx'], 2)

        # Delete middle item (tx 1)
        deleted = ll.delete(lambda item: item['tx'] == 1)
        self.assertTrue(deleted)
        self.assertEqual(ll.size(), 2)
        self.assertEqual(ll.head.data['tx'], 0)
        self.assertEqual(ll.tail.data['tx'], 2)

        # Delete tail item (tx 2): tail should update to tx 0
        deleted_tail = ll.delete(lambda item: item['tx'] == 2)
        self.assertTrue(deleted_tail)
        self.assertEqual(ll.size(), 1)
        self.assertEqual(ll.head.data['tx'], 0)
        self.assertEqual(ll.tail.data['tx'], 0)
        self.assertIs(ll.head, ll.tail)

        # Clear list
        ll.clear()
        self.assertTrue(ll.is_empty())
        self.assertIsNone(ll.head)
        self.assertIsNone(ll.tail)

    def test_binary_search_tree(self):
        bst = BinarySearchTree()
        bst.insert(105, {'title': 'Deep Learning'})
        bst.insert(101, {'title': 'Algorithms'})
        bst.insert(110, {'title': 'Operating Systems'})

        self.assertEqual(bst.size(), 3)

        val = bst.search(101)
        self.assertIsNotNone(val)
        self.assertEqual(val['title'], 'Algorithms')

        val_missing = bst.search(999)
        self.assertIsNone(val_missing)

        inorder = bst.inorder()
        self.assertEqual(len(inorder), 3)
        self.assertEqual(inorder[0][0], 101)  # Sorted order check
        self.assertEqual(inorder[1][0], 105)
        self.assertEqual(inorder[2][0], 110)

    def test_hash_table(self):
        ht = HashTable(capacity=16)
        ht.put("978-0134610993", {'title': 'AI Book'})
        ht.put("978-0262033848", {'title': 'Algorithms Book'})

        self.assertEqual(ht.size(), 2)
        self.assertTrue(ht.contains("978-0134610993"))

        val = ht.get("978-0134610993")
        self.assertEqual(val['title'], 'AI Book')

        removed = ht.remove("978-0134610993")
        self.assertTrue(removed)
        self.assertIsNone(ht.get("978-0134610993"))
        self.assertEqual(ht.size(), 1)

    def test_queue_and_per_book_reservation_manager(self):
        # Basic Queue test
        q = Queue()
        self.assertTrue(q.is_empty())

        q.enqueue({'user': 201, 'book': 101})
        q.enqueue({'user': 202, 'book': 102})
        self.assertEqual(q.size(), 2)

        top = q.peek()
        self.assertEqual(top['user'], 201)

        item1 = q.dequeue()
        self.assertEqual(item1['user'], 201)
        self.assertEqual(q.size(), 1)

        # Per-Book Reservation Manager test (book_id -> Queue)
        res_mgr = BookReservationManager()
        res_mgr.enqueue_reservation(book_id=101, user_id=201, user_name="Aarav")
        res_mgr.enqueue_reservation(book_id=101, user_id=202, user_name="Priya")
        res_mgr.enqueue_reservation(book_id=103, user_id=203, user_name="Rohan")

        self.assertEqual(res_mgr.queue_size(101), 2)
        self.assertEqual(res_mgr.queue_size(103), 1)
        self.assertEqual(res_mgr.queue_size(999), 0)

        # Dequeue per-book
        dequeued_101 = res_mgr.dequeue_reservation(101)
        self.assertEqual(dequeued_101['user_id'], 201)
        self.assertEqual(res_mgr.queue_size(101), 1)

        peek_101 = res_mgr.peek_reservation(101)
        self.assertEqual(peek_101['user_id'], 202)


if __name__ == '__main__':
    unittest.main()
