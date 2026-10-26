import unittest

from tracking import TrackHistory


class TrackHistoryTests(unittest.TestCase):
    def test_limits_trail_length_without_mixing_ids(self):
        history = TrackHistory(length=2, max_age=3)
        history.update(1, {7: (10, 20), 8: (90, 80)})
        history.update(2, {7: (11, 21)})
        history.update(3, {7: (12, 22)})
        self.assertEqual(list(history.points[7]), [(11, 21), (12, 22)])
        self.assertEqual(list(history.points[8]), [(90, 80)])

    def test_expires_lost_tracks_even_on_empty_frames(self):
        history = TrackHistory(length=30, max_age=2)
        history.update(1, {7: (10, 20)})
        history.update(3, {})
        self.assertIn(7, history.points)
        history.update(4, {})
        self.assertNotIn(7, history.points)
        history.update(5, {7: (50, 60)})
        self.assertEqual(list(history.points[7]), [(50, 60)])

    def test_rejects_invalid_limits(self):
        for length, age in [(0, 10), (10, -1)]:
            with self.assertRaises(ValueError):
                TrackHistory(length=length, max_age=age)


if __name__ == '__main__':
    unittest.main()
