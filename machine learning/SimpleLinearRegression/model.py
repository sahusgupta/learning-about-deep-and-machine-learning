import numpy as np
import pandas as pd

class SimpleLinearRegression:
    def __init__(self) -> None:
        self.model = None
        self.__m = 0
        self.__b = 0
        self.pred = None
        self.predicted = False
        
    
    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        max_err = 0.01
        iters = 10000
        for _ in range(iters):
            preds = self.__b + self.__m * x
            err = preds - y
            self.__b -= max_err * np.mean(err)
            self.__m -= max_err * np.mean(err*x)
            
    
    def predict(self, x):
        preds = self.__b + self.__m * x
        self.pred = preds
        self.predicted = True
        return preds
    
    def err(self, y):
        if self.predicted:
            return np.sum(np.abs(y-self.pred))
        else: return np.sum(y ** 2)
        
    def _coefs(self):
        return (self.__m, self.__b)