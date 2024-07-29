# test_model.py

import unittest
from iris_model import model, X_test, y_test
from sklearn.metrics import accuracy_score

class TestIrisModel(unittest.TestCase):

    def test_accuracy(self):
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        self.assertEqual(accuracy, 1.0)

if __name__ == '__main__':
    unittest.main()
