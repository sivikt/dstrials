import os
import pathlib
import pytest
import numpy as np

from numpy.testing import assert_array_equal
from sklearn.model_selection import train_test_split

from logit._log_reg_classifier import LogRegClassifier
from logit.test._base import load_lending_club


def fixture_path(file_name: str):
    return str(pathlib.Path(os.path.dirname(__file__)) / 'fixture' / file_name)


@pytest.fixture
def lending_club_data():
    return load_lending_club()


@pytest.fixture
def lending_club_categor_features():
    return [
        'emp_length',
        'home_ownership',
        'verification_status',
        'pymnt_plan',
        'purpose_cat',
        'zip_code',
        'addr_state',
        'initial_list_status',
        'policy_code'
    ]


@pytest.fixture
def random_state():
    return 242


@pytest.fixture
def new_unique_category():
    return 'aee8eab3aeb0d2f20146ce8a4c1ebbb5'


def test_fit_predict(random_state, lending_club_data):
    X, y = lending_club_data
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)
    
    cls = LogRegClassifier(random_state=random_state)
    cls.fit(X_train, y_train)
    
    pred = cls.predict(X_test)
    expected = np.load(fixture_path('LogRegClassifier_fit_predict.npz'), 'r')['arr_0']
        
    assert_array_equal(pred, expected)


def test_fit_predict_with_new_categories(random_state, new_unique_category, lending_club_data, lending_club_categor_features):
    X, y = lending_club_data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)

    cls = LogRegClassifier(random_state=random_state)
    cls.fit(X_train, y_train)

    X_test = X_test.copy(deep=True)

    for c in lending_club_categor_features:
        X_test.loc[:, c] = new_unique_category

    pred_proba = cls.predict_proba(X_test)
    expected = np.load(fixture_path('LogRegClassifier_fit_predict_proba_with_new_categories.npz'), 'r')['arr_0']

    assert_array_equal(pred_proba, expected)


def test_fit_predict_proba(random_state, lending_club_data):
    X, y = lending_club_data
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)
    
    cls = LogRegClassifier(random_state=random_state)
    cls.fit(X_train, y_train)
    
    pred_proba = cls.predict_proba(X_test)
    expected = np.load(fixture_path('LogRegClassifier_fit_predict_proba.npz'), 'r')['arr_0']
        
    assert_array_equal(pred_proba, expected)


def test_evaluate(random_state, lending_club_data):
    X, y = lending_club_data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)

    cls = LogRegClassifier(random_state=random_state)
    cls.fit(X_train, y_train)

    eval_res = cls.evaluate(X_test, y_test)

    assert eval_res == {'f1_score': 0.21739130434782608, 'logloss': 0.3369127601475267}


def test_tune_parameters(random_state, lending_club_data):
    X, y = lending_club_data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)

    cls = LogRegClassifier(random_state=random_state)

    best_params = cls.tune_parameters(X_train, y_train)
    eval_res = cls.evaluate(X_test, y_test)

    assert best_params == {
        'best_parameters': {
            'logit_classify__C': 0.1,
            'preprocessings__correct_misses_category_strategy': 'most_frequent',
            'preprocessings__correct_misses_numerics_strategy': 'median',
            'preprocessings__transform_feat_category_strategy': 'ohe',
            'preprocessings__transform_feat_numerics_strategy': 'standardize'
        },
        'best_scores': {
            'f1_score': 0.15587968850860037,
            'logloss': 0.3573686967964508
        }
    }

    assert eval_res == {'f1_score': 0.14155251141552513, 'logloss': 0.338917581762135}
