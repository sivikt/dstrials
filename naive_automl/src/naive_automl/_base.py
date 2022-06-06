import numpy as np
import pandas as pd

from typing import (
    Dict
)


class Estimator:
    """Base class for all estimators"""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Fits on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : np.ndarray
            Ground truth labels as a numpy array of 0-s and 1-s.
 
 
        Examples
        --------
        >>> X = pd.DataFrame({'feat1': ['a', 'b', 'a'], 'feat2': [1, 2, 3]})
        >>> y = np.array([0, 0, 1])
        >>> est = Estimator()
        >>> est.fit(X, y)
        """
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class labels on new data.

        Parameters
        ----------
        X : pd.DataFrame
            Input features


        Return
        ------
        np.ndarray
            Predicted class labels


        Examples
        --------
        >>> X = pd.DataFrame({'feat1': ['a', 'b', 'a'], 'feat2': [1, 2, 3]})
        >>> est = Estimator()
        >>> est.predict(X)
        np.array([0, 0, 1])
        """
        return np.array([0, 0, 1])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts the probability of each label.

        Parameters
        ----------
        X : pd.DataFrame
            Input features


        Return
        ------
        np.ndarray
            Predicted probability of each label


        Examples
        --------
        >>> X = pd.DataFrame({'feat1': ['a', 'b', 'a'], 'feat2': [1, 2, 3]})
        >>> est = Estimator()
        >>> est.predict_proba(X)
        np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]])
        """
        return np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]])
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Evaluates "under the hood" model.
        
        Providing features X and Ground truth labels gets 
        the value of the following metrics: 
            1. `F1-score <https://en.wikipedia.org/wiki/F1_score>`_
            2. `LogLoss <https://en.wikipedia.org/wiki/Loss_functions_for_classification#Logistic_loss>`_

        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : np.ndarray
            Ground truth labels as a numpy array of 0-s and 1-s.

        Return
        ------
        np.ndarray
            Predicted probability of each label

        Examples
        --------
        >>> X = pd.DataFrame({'feat1': ['a', 'b', 'a'], 'feat2': [1, 2, 3]})
        >>> y = np.array([0, 0, 1])
        >>> est = Estimator()
        >>> est.evaluate(X, y)
        {'f1_score': 0.3, 'logloss': 0.7}
        """
        return {'f1_score': 0.3, 'logloss': 0.7}

    def tune_parameters(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Dict]:
        """Tunes parameters of "under the hood" model.
        
        Finds the best hyperparameters using K-Fold cross-validation for evaluation.
        The user is not required to provide a parameter search space. This Estimator 
        picks a search space on its own.

        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : np.ndarray
            Ground truth labels as a numpy array of 0-s and 1-s.

        Return
        ------
        np.ndarray
            Output the best parameters and the mean CV score they achieve.

        Examples
        --------
        >>> X = pd.DataFrame({'feat1': ['a', 'b', 'a'], 'feat2': [1, 2, 3]})
        >>> y = np.array([0, 0, 1])
        >>> est = Estimator()
        >>> est.tune_parameters(X, y)
        {
            'best_parameters': {'C': 1.0, 'fit_intercept': False},
            'best_scores': {'f1_score': 0.3, 'logloss': 0.7},
        }
        """
        return {
            'best_parameters': {'C': 1.0, 'fit_intercept': False},
            'best_scores': {'f1_score': 0.3, 'logloss': 0.7},
        }
