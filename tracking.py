"""Bounded trajectory history, independent of the detector."""
from collections import deque


class TrackHistory:
    def __init__(self, length=60, max_age=30):
        if length < 1 or max_age < 0:
            raise ValueError('length must be positive and max_age nonnegative')
        self.length = length
        self.max_age = max_age
        self.points = {}
        self.last_seen = {}

    def update(self, frame, centers):
        for track_id in list(self.points):
            if frame - self.last_seen[track_id] > self.max_age:
                del self.points[track_id]
                del self.last_seen[track_id]
        for track_id, center in centers.items():
            self.points.setdefault(track_id, deque(maxlen=self.length)).append(center)
            self.last_seen[track_id] = frame
