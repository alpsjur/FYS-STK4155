'''
Mitt forsøk på k-fold cross-validation
'''
import numpy as np
from sklearn.utils import shuffle

def k_fold_cross_validation(x_grid, y_grid, z, k=10):
    evaluation_scores=np.zeros(k)

    #flatten the data set and shuffle it
    x_shuffle, y_shuffle, z_shuffle = shuffle(x_grid.flatten(), y_grid.flatten(), z.flatten(), random_state=0)

    #split the data into k folds
    x_split = np.array_split(x_shuffle, k)
    y_split = np.array_split(y_shuffle, k)
    z_split = np.array_split(z_shuffle, k)

    #loop through the folds
    for i in len(k):
        #pick out the test fold from data
        x_test = x_model_split[i]
        y_test = y_model_split[i]
        z_test = z_model_split[i]

        #pick out the remaining data as training data
        mask = np.ones(len(z_model_split), dtype=bool)
        mask[i] = False
        x_train = x_model_split[mask]
        y_train = y_model_split[mask]
        z_train = z_model_split[mask]

        #fit a model to the training set
        '''
        Her må vi bruke enten OLS, Ridges eller Lasso
        '''

        #evaluate the model on the test set
        '''
        Her må vi bruke modellen til å beregne z_tilde for (x_test, y_test)
        og sammenligne med z_test ved MSE eller R_2_score
        '''

    return np.mean(evaluation_scores)
