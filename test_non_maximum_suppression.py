from unittest import TestCase
import numpy as np

from src.non_maximum_suppression import non_max_suppression_fast


class Test(TestCase):
    def test_non_max_suppression_fast(self):

        # 3 boxes which overlap at 1,1 and another 3 boxes at 4,4
        boxes = np.array([
            [1.0, 1.0, 2, 2],
            [1.1, 1.1, 2, 2],
            [1.1, 1.1, 1.9, 1.9],

            [4.0, 4.0, 2, 2],
            [4.1, 4.1, 2, 2],
            [4.1, 4.1, 1.9, 1.9]
        ])

        picks = non_max_suppression_fast(boxes)
        self.assertEqual(len(picks), 2)

