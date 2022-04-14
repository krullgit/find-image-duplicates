import unittest
import utils
import collections
import os 
import numpy as np

class Test(unittest.TestCase):
    def test_get_all_images_in_folder(self):
        self.assertIsInstance(utils.get_all_images_in_folder(os.getcwd()), collections.abc.Sequence)
        self.assertRaises(TypeError, utils.get_all_images_in_folder, None)
        self.assertRaises(FileNotFoundError, utils.get_all_images_in_folder, "jk567jh%&")

    def test_pairwise_combs_numba(self):
        self.assertEqual(utils.pairwise_combs_numba(np.array([1,2,3,4,5])).shape,(10,2))

if __name__ == '__main__':
    unittest.main()