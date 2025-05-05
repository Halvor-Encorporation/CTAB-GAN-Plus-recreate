import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture





class GMM_transformer():
    def __init__(self):
        pass
        

    def fit(self,data_col,n_clusters,eps):
        gm = BayesianGaussianMixture(
            n_components = n_clusters, 
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001, 
            max_iter=100,n_init=1, random_state=42)
        gm.fit(data_col.reshape([-1, 1]))
        mode_freq = (pd.Series(gm.predict(data_col.reshape([-1, 1]))).value_counts().keys())
        
        old_comp = gm.weights_ > eps
        comp = []
        for i in range(n_clusters):
            if (i in (mode_freq)) & old_comp[i]:
                comp.append(True)
            else:
                comp.append(False)
         
        output_info = [(1, 'tanh','no_g'), (np.sum(comp), 'softmax')]
        output_dim = 1 + np.sum(comp)
        return gm, comp, output_info, output_dim