import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, log_loss, make_scorer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold

from typing import (
    Dict
)

from logit._base import Estimator
from logit._exploratory_data_analist import ExploratoryDataAnalyst, PreprocessingExpert


class LogRegClassifier(Estimator):
    """An automated version of Regularized Logistic Regression based on 
       scikit-learn SGDClassifier classifier.
    """

    def __init__(self, random_state: int = None):
        """Basic constructor.

        Parameters
        ----------
        random_state : int, default=None
            Random seed to force reproducibility while testing


        Examples
        --------
        >>> cls = LogRegClassifier(random_seed=242)
        >>> X = pd.DataFrame({'feat1': ['a', 'b', 'a'], 'feat2': [1, 2, 3]})
        >>> y = np.array([0, 0, 1])
        >>> cls.fit(X,y)
        """
        self._clf_pipeline = None
        self._expected_objects_features = None
        self._rseed = random_state
        self._epsilon = 1e-14

    def _assert_objects_features(self, X: pd.DataFrame):
        if list(X.columns) != self._expected_objects_features:
            raise Exception('Input features do not match expected list ' + self._expected_objects_features)

    def _assert_is_binary_target(self, y: np.ndarray):
        if len(np.unique(y)) != 2:
            raise Exception('Only Binary classification is supported')

    def _create_base_model(self):
        #         return SGDClassifier(
        #             loss='log',
        #             penalty='l2',
        #             fit_intercept=False,
        #             max_iter=1000,
        #             tol=1e-4,
        #             epsilon=self._epsilon,
        #             shuffle=True,
        #             random_state=self._rseed,
        #             learning_rate='optimal'
        #         )
        return LogisticRegression(
            penalty='l2',
            dual=False,
            tol=1e-4,
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            n_jobs=-1,
            random_state=self._rseed
        )

    def _create_default_pipeline(self):
        return Pipeline(steps=[
            ('preprocessings', PreprocessingExpert()),
            ('logit_classify', self._create_base_model())
        ])

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self._assert_is_binary_target(y)

        self._expected_objects_features = list(X.columns)

        data_report = ExploratoryDataAnalyst().analyze(X, y)
        self._clf_pipeline = self._create_default_pipeline()

        self._clf_pipeline.fit(X, y, preprocessings__data_report=data_report)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._assert_objects_features(X)
        return self._clf_pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._assert_objects_features(X)
        return self._clf_pipeline.predict_proba(X)

    def evaluate(self, X: pd.DataFrame, y_true: np.ndarray) -> Dict[str, float]:
        self._assert_objects_features(X)
        self._assert_is_binary_target(y_true)

        y_pred = self._clf_pipeline.predict(X)
        y_pred_proba = self._clf_pipeline.predict_proba(X)

        return {
            'f1_score': f1_score(y_true=y_true, y_pred=y_pred, average='binary'),
            'logloss': log_loss(y_true=y_true, y_pred=y_pred_proba, eps=self._epsilon)
        }

    def tune_parameters(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Dict]:
        self._assert_is_binary_target(y)

        self._expected_objects_features = list(X.columns)

        data_report = ExploratoryDataAnalyst().analyze(X, y)
        preprocessings = PreprocessingExpert().fit(X, y, data_report=data_report)
        possible_strategies = preprocessings.get_possible_strategies()

        dict_merge = lambda a, b: a.update(b) or a

        preprocessings_params = {
            'preprocessings__correct_misses_numerics_strategy': possible_strategies['correct_misses'][
                'numerics_strategy'],
            'preprocessings__correct_misses_category_strategy': possible_strategies['correct_misses'][
                'category_strategy'],
            'preprocessings__transform_feat_numerics_strategy': possible_strategies['transform_feat'][
                'numerics_strategy'],
            'preprocessings__transform_feat_category_strategy': possible_strategies['transform_feat'][
                'category_strategy']
        }

        base_params = dict_merge(preprocessings_params, {
            # 'logit_classify__dual': [True, False],
            'logit_classify__C': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 5, 10, 50, 100, 1000]
        })

        param_grid = [
            dict_merge(base_params, {
                # 'logit_classify__solver': ['liblinear', 'saga'],
                # 'logit_classify__penalty': ['l1', 'l2']
            })
            #dict_merge(base_params, {
            #   'logit_classify__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
            #   'logit_classify__penalty': ['l2']
            #})
        ]

        print(param_grid)

        # n_splits should be adaptive
        kf = KFold(n_splits=10, shuffle=True, random_state=self._rseed)

        scoring = {
            'AUC': 'roc_auc',
            'f1': make_scorer(f1_score, needs_proba=False, average='binary'),
            'logloss': make_scorer(log_loss, needs_proba=True, eps=self._epsilon)
        }

        self._clf_pipeline = self._create_default_pipeline()

        gs = GridSearchCV(
            self._clf_pipeline,
            param_grid,
            cv=kf,
            n_jobs=-1,
            verbose=1,
            scoring=scoring,
            refit='AUC',
            return_train_score=True
        )

        gs.fit(X, y)
        self._clf_pipeline = gs.best_estimator_

        return {
            'best_parameters': gs.best_params_,
            'best_scores': {
                'f1_score': gs.cv_results_['mean_test_f1'][gs.best_index_],
                'logloss': gs.cv_results_['mean_test_logloss'][gs.best_index_]
            }
        }
