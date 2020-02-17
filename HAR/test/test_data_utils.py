import unittest
from HAR.data_utils import UCI


class UCITestCase(unittest.TestCase):

    def setUp(self):
        self.dir = '../../data/UCI HAR Dataset'
        self.train_loader, self.test_loader = UCI.load_dataset(batch_size=1, data_dir=self.dir)
        self.train_X, self.train_y = iter(self.train_loader).next()
        self.test_X, self.test_y = iter(self.train_loader).next()
        self.n_classes = 6

    def test_load(self):
        self.assertIsNotNone(self.train_loader)
        self.assertIsNotNone(self.test_loader)

    def test_X_shape(self):
        self.assertTupleEqual(self.train_X.shape, (1, 9, 128))
        self.assertTupleEqual(self.test_X.shape, (1, 9, 128))

    def test_y_shape(self):
        self.assertTupleEqual(self.train_y.shape, (1,))
        self.assertTupleEqual(self.test_y.shape, (1,))

    def test_y_range(self):
        d = [0 for _ in range(self.n_classes)]
        for data in self.train_loader:
            _, y = data
            d[y.item()] += 1

        self.assertFalse(any([a == 0 for a in d]))


if __name__ == '__main__':
    unittest.main()
