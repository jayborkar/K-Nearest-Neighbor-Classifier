import numpy as np
from scipy.spatial import distance
from collections import Counter
###Distance Metric : L2 ###
class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k):
    """
    STUDENT CODE
    """
    dists = self.compute_distances(X)
    return self.predict_labels(dists, k=k)
   
 
  def compute_distances(self, X):
    """
    STUDENT CODE
    """
    val=X.toarray()
    train=self.X_train.toarray()
    dists=[[] for i in range(len(val))]

    for i in range(0,len(val)):
      for j in range(0,len(train)):
        distances=distance.euclidean(val[i],train[j])
        dists[i].append([distances,self.y_train[j]]) 

    return dists

  def predict_labels(self, dists,k):
    """
    STUDENT CODE
    """
    y_pred=[]
    
    for i in range(0,len(dists)):
      res_list=[]
      k_small=[]

      dists[i]=sorted(dists[i],key=lambda x: x[0])

      for j in range(0,k):
        k_small.append(dists[i][j])
      res_list=[x[1] for x in k_smallest]
      count=Counter([x[1] for x in k_small])
      y_pred.append(count.most_common()[0][0])
      
    return y_pred

