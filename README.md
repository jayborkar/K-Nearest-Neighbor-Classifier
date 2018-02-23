# K-Nearest-Neighbor-Classifier

I have implemented my own k-Nearest Neighbor (KNN) classifiers on the NEWS DATASET (http://scikit-learn.org/stable/datasets/twenty_newsgroups.html).

I have used the 4 classes (’alt.atheism’,’talk.religion.misc’,’comp.graphics’,’sci.space’) among the original 20 classes.

Make a folder called hw1student and add features.py, train.pkl, test.pkl, knn.ipynb and knn-test.ipnb files in that folder. Make a new folder in this hw1student folder name it cs536-1, add a blank -init-.py and make a new folder in this cs536-1 folder and name it models. Add -init-.py and k-nearest-neigbhor.py files in this folder.

Run: hw1student → python features.py

Working files :
hw1 student → knn.ipynb

hw1 student → cs536 1 → models → k nearest neighbor.py

You can change the train ratio from 0.1 to 1.0(interval step : 0.1).

When you finish your implementation, you will find validation accuracy rates for the different Ks at the train ratio.


Prediction on Test Set :

Working files:
hw1 student → knn-test.ipynb
hw1 student → cs536-1 → models → k-nearest-neighbor.py

You can change the train ratio from 0.1 to 1.0 (interval step : 0.1)

When you finish your implementation, you will find test accuracy rates for the different Ks at the train ratio.

# IrisKNN
IrisKNN.py is program where I have imported iris data from sklearn and classified using two features-sepal length (cm) and sepal width (cm). I have imported KNeighborsClassifier from sklearn.neighbors and imported train_test split from sklearn.cross_validation for classification.
