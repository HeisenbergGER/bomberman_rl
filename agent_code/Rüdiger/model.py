"""
Regression Forest for Bomber man
"""


from sklearn.ensemble import RandomForestRegressor


class REG_FOR():
    gamma = 0.99
    
    min_samples = 10
    N_trees = 50
    


    def __init__(self):
        self.regr = RandomForestRegressor(n_estimators=self.N_trees, min_samples_split=self.min_samples, oob_score=True)

    def fit(self,X,y):
        return self.regr.fit(X,y)
        

    def pred(self,X):
        return self.regr.predict(X)
    def oob_score(self):
        return self.regr.oob_score_



