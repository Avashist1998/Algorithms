import os
import unittest
from utils import my_train_test_split, encoder, file_prep
from machine_learning.decision_stamp import decision_stamp

'''
    Testing the Decision Stamp behavior
'''
PATH = os.getcwd()
DATAPATH = os.path.join(PATH, 'Dataset')
DATAFILES = os.listdir(DATAPATH) 
FILESPATH = os.path.join(DATAPATH, DATAFILES[0])
X, y, features = file_prep(FILESPATH)
X_train, X_test, y_train, y_test = my_train_test_split(
    X, y, test_size=0.20, random_state=42)

class TestDecisionTree(unittest.TestCase):
    '''
    Class to test the decision tree
    '''
    def test_initial_clf(self):
        '''
        Function to test the inital state of the classifer
        '''
        my_stamp = decision_stamp()
        self.assertIsNone(my_stamp.feature_index)
        self.assertIsNone(my_stamp.weights)
        self.assertEqual(my_stamp.threshold, 0)
        self.assertEqual(my_stamp.pairty, 0)
        self.assertEqual(my_stamp.weight_error,0)
    
    def test_prediction(self):
        '''
        Function to test the behavior of the classifer
        '''
        my_stamp = decision_stamp()
        my_stamp.fit(X_train,y_train)
        prediction = my_stamp.predict(X_test)
        score = my_stamp.score(prediction,y_test)
        self.assertEqual(my_stamp.feature_index, 4)
        self.assertAlmostEqual(score, 0.9262106072252114)

if __name__ == "__main__":
   unittest.main()
