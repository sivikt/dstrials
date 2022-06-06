import os
import numpy as np
import pathlib
import pytest

from unittest.mock import patch
from numpy.testing import assert_array_equal
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

from logit._exploratory_data_analist import (
    FeatureStats,
    ExploratoryReport,
    ExploratoryDataAnalyst,
    PreprocessingExpert
)

from logit.test._base import load_lending_club


dict_merge = lambda a,b: a.update(b) or a


def fixture_path(file_name: str):
    return str(pathlib.Path(os.path.dirname(__file__)) / 'fixture' / file_name)


@pytest.fixture
def lending_club_data():
    return load_lending_club()


@pytest.fixture
def random_state():
    return 242


@pytest.fixture
def exploratory_test_report():
    numeric_features = [
        FeatureStats(
            name='num_test'+str(i),
            index=i,
            unique_num=42*i,
            missed_num=66*i) for i in range(2)]
 
    categor_features = [
        FeatureStats(
            name='cat_test'+str(i),
            index=len(numeric_features)+i,
            unique_num=42*(len(numeric_features)+i),
            missed_num=66*(len(numeric_features)+i)) for i in range(2)]

    return ExploratoryReport(
        objects_num=11, 
        numeric_features=numeric_features, 
        categor_features=categor_features
      )


@pytest.fixture
def lending_club_numeric_features():
    return [
        FeatureStats(name='Id', index=0, unique_num=10000, missed_num=0), 
        FeatureStats(name='annual_inc', index=3, unique_num=1902, missed_num=1), 
        FeatureStats(name='debt_to_income', index=9, unique_num=2585, missed_num=0), 
        FeatureStats(name='delinq_2yrs', index=10, unique_num=11, missed_num=5), 
        FeatureStats(name='inq_last_6mths', index=11, unique_num=21, missed_num=5), 
        FeatureStats(name='mths_since_last_delinq', index=12, unique_num=92, missed_num=6316), 
        FeatureStats(name='mths_since_last_record', index=13, unique_num=95, missed_num=9160), 
        FeatureStats(name='open_acc', index=14, unique_num=37, missed_num=5), 
        FeatureStats(name='pub_rec', index=15, unique_num=5, missed_num=5), 
        FeatureStats(name='revol_bal', index=16, unique_num=8130, missed_num=0), 
        FeatureStats(name='revol_util', index=17, unique_num=1028, missed_num=26), 
        FeatureStats(name='total_acc', index=18, unique_num=76, missed_num=5), 
        FeatureStats(name='collections_12_mths_ex_med', index=20, unique_num=2, missed_num=32), 
        FeatureStats(name='mths_since_last_major_derog', index=21, unique_num=3, missed_num=0)
    ]
    

@pytest.fixture
def lending_club_categor_features():
    return [
        FeatureStats(name='emp_length', index=1, unique_num=14, missed_num=0), 
        FeatureStats(name='home_ownership', index=2, unique_num=5, missed_num=0), 
        FeatureStats(name='verification_status', index=4, unique_num=3, missed_num=0), 
        FeatureStats(name='pymnt_plan', index=5, unique_num=2, missed_num=0), 
        FeatureStats(name='purpose_cat', index=6, unique_num=27, missed_num=0), 
        FeatureStats(name='zip_code', index=7, unique_num=720, missed_num=0), 
        FeatureStats(name='addr_state', index=8, unique_num=50, missed_num=0), 
        FeatureStats(name='initial_list_status', index=19, unique_num=2, missed_num=0), 
        FeatureStats(name='policy_code', index=22, unique_num=5, missed_num=0)
    ]
    

@pytest.fixture
def adjusted_data_report_after_correct_misses_step():
    return ExploratoryReport(
        objects_num=10000, 
        numeric_features=[FeatureStats(name='Id', index=0, unique_num=10000, missed_num=0), 
                          FeatureStats(name='annual_inc', index=1, unique_num=1902, missed_num=1), 
                          FeatureStats(name='debt_to_income', index=2, unique_num=2585, missed_num=0), 
                          FeatureStats(name='delinq_2yrs', index=3, unique_num=11, missed_num=5), 
                          FeatureStats(name='inq_last_6mths', index=4, unique_num=21, missed_num=5), 
                          FeatureStats(name='mths_since_last_delinq', index=5, unique_num=92, missed_num=6316), 
                          FeatureStats(name='mths_since_last_record', index=6, unique_num=95, missed_num=9160), 
                          FeatureStats(name='open_acc', index=7, unique_num=37, missed_num=5), 
                          FeatureStats(name='pub_rec', index=8, unique_num=5, missed_num=5), 
                          FeatureStats(name='revol_bal', index=9, unique_num=8130, missed_num=0), 
                          FeatureStats(name='revol_util', index=10, unique_num=1028, missed_num=26), 
                          FeatureStats(name='total_acc', index=11, unique_num=76, missed_num=5), 
                          FeatureStats(name='collections_12_mths_ex_med', index=12, unique_num=2, missed_num=32), 
                          FeatureStats(name='mths_since_last_major_derog', index=13, unique_num=3, missed_num=0)], 
        categor_features=[FeatureStats(name='emp_length', index=14, unique_num=14, missed_num=0), 
                          FeatureStats(name='home_ownership', index=15, unique_num=5, missed_num=0), 
                          FeatureStats(name='verification_status', index=16, unique_num=3, missed_num=0), 
                          FeatureStats(name='pymnt_plan', index=17, unique_num=2, missed_num=0), 
                          FeatureStats(name='purpose_cat', index=18, unique_num=27, missed_num=0), 
                          FeatureStats(name='zip_code', index=19, unique_num=720, missed_num=0), 
                          FeatureStats(name='addr_state', index=20, unique_num=50, missed_num=0), 
                          FeatureStats(name='initial_list_status', index=21, unique_num=2, missed_num=0), 
                          FeatureStats(name='policy_code', index=22, unique_num=5, missed_num=0)]
    )


preprocessing_expert_cm_numerics_strategy_params = pytest.mark.parametrize('cm_numerics_strategy', [None, 'mean', 'median'])
preprocessing_expert_cm_category_strategy_params = pytest.mark.parametrize('cm_category_strategy', [None, 'most_frequent'])
preprocessing_expert_tf_numerics_strategy_params = pytest.mark.parametrize('tf_numerics_strategy', [None, 'standardize', 'minmax'])
preprocessing_expert_tf_category_strategy_params = pytest.mark.parametrize('tf_category_strategy', [None, 'ohe'])

preprocessing_expert_cm_numerics_strategy_params_no_none = pytest.mark.parametrize('cm_numerics_strategy', ['mean', 'median'])
preprocessing_expert_cm_category_strategy_params_no_none = pytest.mark.parametrize('cm_category_strategy', ['most_frequent'])
preprocessing_expert_tf_numerics_strategy_params_no_none = pytest.mark.parametrize('tf_numerics_strategy', ['standardize', 'minmax'])
preprocessing_expert_tf_category_strategy_params_no_none = pytest.mark.parametrize('tf_category_strategy', ['ohe'])


@pytest.fixture
def default_preprocessing_expert_strategies():
    return {
        'correct_misses': {
            'numerics_strategy': ['mean', 'median'],
            'category_strategy': ['most_frequent']
        },
        'transform_feat': {
            'numerics_strategy': ['standardize', 'minmax'],
            'category_strategy': ['ohe']
        }
    }


def test_feature_stats():
    fs = FeatureStats(
            name='test',
            index=21,
            unique_num=42,
            missed_num=66
         )
    
    assert fs.name == 'test'
    assert fs.index == 21
    assert fs.unique_num == 42
    assert fs.missed_num == 66
    assert str(fs) == 'name=test\nindex=21\nunique_num=42\nmissed_num=66'
            

def test_feature_stats_reindex():
    fs = FeatureStats(
            name='test',
            index=21,
            unique_num=42,
            missed_num=66
         )

    fs2 = fs.reindex(index=33)
    
    assert fs != fs2
    assert fs2.index == 33
    

def test_exploratory_report(exploratory_test_report):
    er = exploratory_test_report
    
    numeric_features = [
        FeatureStats(name='num_test0',index=0,unique_num=0,missed_num=0), 
        FeatureStats(name='num_test1',index=1,unique_num=42,missed_num=66)
    ]
    
    categor_features = [
        FeatureStats(name='cat_test0',index=2,unique_num=84,missed_num=132), 
        FeatureStats(name='cat_test1',index=3,unique_num=126,missed_num=198)
    ]
    
    numeric_features_idx = {f.name: f for f in numeric_features}
    categor_features_idx = {f.name: f for f in categor_features}
        
    assert er.objects_num == 11
    assert er.numeric_features == numeric_features
    assert er.categor_features == categor_features
    assert er.numeric_features_idx == numeric_features_idx
    assert er.categor_features_idx == categor_features_idx
    assert str(er) == "objects_num=11\nnumeric_features=[FeatureStats(name='num_test0', index=0, unique_num=0, missed_num=0), FeatureStats(name='num_test1', index=1, unique_num=42, missed_num=66)]\ncategor_features=[FeatureStats(name='cat_test0', index=2, unique_num=84, missed_num=132), FeatureStats(name='cat_test1', index=3, unique_num=126, missed_num=198)]"
    
    
def test_exploratory_data_analist(lending_club_data, lending_club_numeric_features, lending_club_categor_features):
    X, y = lending_club_data
    data_report = ExploratoryDataAnalyst().analyze(X, y)
            
    assert data_report.objects_num == 10000
    assert data_report.numeric_features == lending_club_numeric_features
    assert data_report.categor_features == lending_club_categor_features


@preprocessing_expert_tf_numerics_strategy_params
@preprocessing_expert_tf_category_strategy_params
def test_preprocessing_expert_plan_transform_values_strategy(lending_club_data,
                                                             tf_numerics_strategy, 
                                                             tf_category_strategy):
    X, y = lending_club_data
    data_report = ExploratoryDataAnalyst().analyze(X, y)

    pe = PreprocessingExpert(transform_feat_numerics_strategy=tf_numerics_strategy, 
                             transform_feat_category_strategy=tf_category_strategy) 

    pipeline, dr = pe.plan_transform_values_strategy(data_report=data_report)
    
    expected_parameters = {
        'None_None'  : ("{'n_jobs': None, 'remainder': 'drop', 'sparse_threshold': 0.3, 'transformer_weights': None, 'transformers': [('num_transform_standardize', StandardScaler(copy=True, with_mean=True, with_std=True), [0, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21]), ('cat_transform_oneh_encode', OneHotEncoder(categories='auto', drop=None, dtype=<class 'numpy.float64'>,\n"
                        "              handle_unknown='ignore', sparse=False), [1, 2, 4, 5, 6, 7, 8, 19, 22])], 'verbose': False, 'num_transform_standardize': StandardScaler(copy=True, with_mean=True, with_std=True), 'cat_transform_oneh_encode': OneHotEncoder(categories='auto', drop=None, dtype=<class 'numpy.float64'>,\n"
                        "              handle_unknown='ignore', sparse=False), 'num_transform_standardize__copy': True, 'num_transform_standardize__with_mean': True, 'num_transform_standardize__with_std': True, 'cat_transform_oneh_encode__categories': 'auto', 'cat_transform_oneh_encode__drop': None, 'cat_transform_oneh_encode__dtype': <class 'numpy.float64'>, 'cat_transform_oneh_encode__handle_unknown': 'ignore', 'cat_transform_oneh_encode__sparse': False}"),        
        'minmax_None': ("{'n_jobs': None, 'remainder': 'drop', 'sparse_threshold': 0.3, 'transformer_weights': None, 'transformers': [('num_transform_minmax', MinMaxScaler(copy=True, feature_range=(0, 1)), [0, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21]), ('cat_transform_oneh_encode', OneHotEncoder(categories='auto', drop=None, dtype=<class 'numpy.float64'>,\n"
                        "              handle_unknown='ignore', sparse=False), [1, 2, 4, 5, 6, 7, 8, 19, 22])], 'verbose': False, 'num_transform_minmax': MinMaxScaler(copy=True, feature_range=(0, 1)), 'cat_transform_oneh_encode': OneHotEncoder(categories='auto', drop=None, dtype=<class 'numpy.float64'>,\n"
                        "              handle_unknown='ignore', sparse=False), 'num_transform_minmax__copy': True, 'num_transform_minmax__feature_range': (0, 1), 'cat_transform_oneh_encode__categories': 'auto', 'cat_transform_oneh_encode__drop': None, 'cat_transform_oneh_encode__dtype': <class 'numpy.float64'>, 'cat_transform_oneh_encode__handle_unknown': 'ignore', 'cat_transform_oneh_encode__sparse': False}")
    }
    
    expected_parameters = dict_merge(expected_parameters, {
        'None_ohe': expected_parameters['None_None'],
        'standardize_None': expected_parameters['None_None'],
        'standardize_ohe': expected_parameters['None_None'],
        'minmax_ohe': expected_parameters['minmax_None']
    })
    
    params_combi = str(tf_numerics_strategy) + '_' + str(tf_category_strategy)
    
    assert type(pipeline) == ColumnTransformer
    assert repr(pipeline.get_params()) == expected_parameters[params_combi]
    assert dr is None

    
@preprocessing_expert_cm_numerics_strategy_params
@preprocessing_expert_cm_category_strategy_params
def test_preprocessing_expert_plan_missing_values_strategy(lending_club_data,
                                                           adjusted_data_report_after_correct_misses_step,
                                                           cm_numerics_strategy, 
                                                           cm_category_strategy):
    X, y = lending_club_data
    data_report = ExploratoryDataAnalyst().analyze(X, y)

    pe = PreprocessingExpert(correct_misses_numerics_strategy=cm_numerics_strategy, 
                             correct_misses_category_strategy=cm_category_strategy) 

    pipeline, dr = pe.plan_missing_values_strategy(data_report=data_report)
    
    expected_parameters = {
        'None_None'  : ("{'n_jobs': None, 'remainder': 'drop', 'sparse_threshold': 0.3, 'transformer_weights': None, 'transformers': [('num_nan_miss_vals_imputer', SimpleImputer(add_indicator=False, copy=True, fill_value=None,\n"
                        "              missing_values=nan, strategy='mean', verbose=0), [0, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21]), ('cat_nan_miss_vals_imputer', SimpleImputer(add_indicator=False, copy=True, fill_value=None,\n"
                        "              missing_values=nan, strategy='most_frequent', verbose=0), [1, 2, 4, 5, 6, 7, 8, 19, 22])], 'verbose': False, 'num_nan_miss_vals_imputer': SimpleImputer(add_indicator=False, copy=True, fill_value=None,\n"
                        "              missing_values=nan, strategy='mean', verbose=0), 'cat_nan_miss_vals_imputer': SimpleImputer(add_indicator=False, copy=True, fill_value=None,\n"
                        "              missing_values=nan, strategy='most_frequent', verbose=0), 'num_nan_miss_vals_imputer__add_indicator': False, 'num_nan_miss_vals_imputer__copy': True, 'num_nan_miss_vals_imputer__fill_value': None, 'num_nan_miss_vals_imputer__missing_values': nan, 'num_nan_miss_vals_imputer__strategy': 'mean', 'num_nan_miss_vals_imputer__verbose': 0, 'cat_nan_miss_vals_imputer__add_indicator': False, 'cat_nan_miss_vals_imputer__copy': True, 'cat_nan_miss_vals_imputer__fill_value': None, 'cat_nan_miss_vals_imputer__missing_values': nan, 'cat_nan_miss_vals_imputer__strategy': 'most_frequent', 'cat_nan_miss_vals_imputer__verbose': 0}"),        
        'median_None': ("{'n_jobs': None, 'remainder': 'drop', 'sparse_threshold': 0.3, 'transformer_weights': None, 'transformers': [('num_nan_miss_vals_imputer', SimpleImputer(add_indicator=False, copy=True, fill_value=None,\n"
                        "              missing_values=nan, strategy='median', verbose=0), [0, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21]), ('cat_nan_miss_vals_imputer', SimpleImputer(add_indicator=False, copy=True, fill_value=None,\n"
                        "              missing_values=nan, strategy='most_frequent', verbose=0), [1, 2, 4, 5, 6, 7, 8, 19, 22])], 'verbose': False, 'num_nan_miss_vals_imputer': SimpleImputer(add_indicator=False, copy=True, fill_value=None,\n"
                        "              missing_values=nan, strategy='median', verbose=0), 'cat_nan_miss_vals_imputer': SimpleImputer(add_indicator=False, copy=True, fill_value=None,\n"
                        "              missing_values=nan, strategy='most_frequent', verbose=0), 'num_nan_miss_vals_imputer__add_indicator': False, 'num_nan_miss_vals_imputer__copy': True, 'num_nan_miss_vals_imputer__fill_value': None, 'num_nan_miss_vals_imputer__missing_values': nan, 'num_nan_miss_vals_imputer__strategy': 'median', 'num_nan_miss_vals_imputer__verbose': 0, 'cat_nan_miss_vals_imputer__add_indicator': False, 'cat_nan_miss_vals_imputer__copy': True, 'cat_nan_miss_vals_imputer__fill_value': None, 'cat_nan_miss_vals_imputer__missing_values': nan, 'cat_nan_miss_vals_imputer__strategy': 'most_frequent', 'cat_nan_miss_vals_imputer__verbose': 0}")
    }
    
    expected_parameters = dict_merge(expected_parameters, {
        'mean_None': expected_parameters['None_None'],
        'None_most_frequent': expected_parameters['None_None'],
        'mean_most_frequent': expected_parameters['None_None'],
        'median_most_frequent': expected_parameters['median_None']
    })
        
    assert type(pipeline) == ColumnTransformer
    assert repr(pipeline.get_params()) == expected_parameters[str(cm_numerics_strategy) + '_' + str(cm_category_strategy)]
    assert dr is not None
    assert dr != data_report
    assert dr == adjusted_data_report_after_correct_misses_step


# TODO probably unsafe test, how to do it better?
@patch.object(PreprocessingExpert, 'plan_transform_values_strategy')
@patch.object(PreprocessingExpert, 'plan_missing_values_strategy')
@preprocessing_expert_cm_numerics_strategy_params
@preprocessing_expert_cm_category_strategy_params
@preprocessing_expert_tf_numerics_strategy_params
@preprocessing_expert_tf_category_strategy_params
def test_preprocessing_expert_plan(plan_transform_values_strategy_mock,
                                   plan_missing_values_strategy_mock,
                                   lending_club_data,
                                   cm_numerics_strategy, 
                                   cm_category_strategy, 
                                   tf_numerics_strategy, 
                                   tf_category_strategy):
    X, y = lending_club_data
    data_report = ExploratoryDataAnalyst().analyze(X, y)

    plan_transform_values_strategy_mock.return_value = FunctionTransformer(), data_report
    plan_missing_values_strategy_mock.return_value = FunctionTransformer(), data_report
    
    pe = PreprocessingExpert(correct_misses_numerics_strategy=cm_numerics_strategy, 
                             correct_misses_category_strategy=cm_category_strategy, 
                             transform_feat_numerics_strategy=tf_numerics_strategy, 
                             transform_feat_category_strategy=tf_category_strategy) 
    
    pipeline = pe.plan(data_report=data_report)
    
    plan_missing_values_strategy_mock.assert_called_with(data_report=data_report)
    plan_transform_values_strategy_mock.assert_called_with(data_report=data_report)
    
    assert type(pipeline) == Pipeline
    assert repr(pipeline.get_params()) == ("{'memory': None, 'steps': [('correct_misses', FunctionTransformer(accept_sparse=False, check_inverse=True, func=None,\n"
                                           "                    inv_kw_args=None, inverse_func=None, kw_args=None,\n"
                                           "                    validate=False)), ('transform_feat', FunctionTransformer(accept_sparse=False, check_inverse=True, func=None,\n"
                                           "                    inv_kw_args=None, inverse_func=None, kw_args=None,\n"
                                           "                    validate=False))], 'verbose': False, 'correct_misses': FunctionTransformer(accept_sparse=False, check_inverse=True, func=None,\n"
                                           "                    inv_kw_args=None, inverse_func=None, kw_args=None,\n"
                                           "                    validate=False), 'transform_feat': FunctionTransformer(accept_sparse=False, check_inverse=True, func=None,\n"
                                           "                    inv_kw_args=None, inverse_func=None, kw_args=None,\n"
                                           "                    validate=False), 'correct_misses__accept_sparse': False, 'correct_misses__check_inverse': True, 'correct_misses__func': None, 'correct_misses__inv_kw_args': None, 'correct_misses__inverse_func': None, 'correct_misses__kw_args': None, 'correct_misses__validate': False, 'transform_feat__accept_sparse': False, 'transform_feat__check_inverse': True, 'transform_feat__func': None, 'transform_feat__inv_kw_args': None, 'transform_feat__inverse_func': None, 'transform_feat__kw_args': None, 'transform_feat__validate': False}")

    
@preprocessing_expert_cm_numerics_strategy_params
@preprocessing_expert_cm_category_strategy_params
@preprocessing_expert_tf_numerics_strategy_params
@preprocessing_expert_tf_category_strategy_params
def test_preprocessing_expert_get_possible_strategies_without_fit(default_preprocessing_expert_strategies,
                                                                  cm_numerics_strategy, 
                                                                  cm_category_strategy, 
                                                                  tf_numerics_strategy, 
                                                                  tf_category_strategy):    
    pe = PreprocessingExpert(correct_misses_numerics_strategy=cm_numerics_strategy, 
                             correct_misses_category_strategy=cm_category_strategy, 
                             transform_feat_numerics_strategy=tf_numerics_strategy, 
                             transform_feat_category_strategy=tf_category_strategy) 
    
    possible_strategies1 = pe.get_possible_strategies()
    assert possible_strategies1 == default_preprocessing_expert_strategies 
    

@preprocessing_expert_cm_numerics_strategy_params
@preprocessing_expert_cm_category_strategy_params
@preprocessing_expert_tf_numerics_strategy_params
@preprocessing_expert_tf_category_strategy_params
def test_preprocessing_expert_get_possible_strategies_immutable(default_preprocessing_expert_strategies,
                                                                cm_numerics_strategy, 
                                                                cm_category_strategy, 
                                                                tf_numerics_strategy, 
                                                                tf_category_strategy):    
    pe = PreprocessingExpert(correct_misses_numerics_strategy=cm_numerics_strategy, 
                             correct_misses_category_strategy=cm_category_strategy, 
                             transform_feat_numerics_strategy=tf_numerics_strategy, 
                             transform_feat_category_strategy=tf_category_strategy) 
    
    possible_strategies1 = pe.get_possible_strategies()
    possible_strategies2 = pe.get_possible_strategies()
    
    assert possible_strategies1 == default_preprocessing_expert_strategies
    assert possible_strategies2 == default_preprocessing_expert_strategies
    assert possible_strategies1 is not possible_strategies2
    assert possible_strategies1 == possible_strategies2
            
    possible_strategies1['correct_misses']['numerics_strategy'].append('test')
    assert possible_strategies1 != possible_strategies2
    possible_strategies1['correct_misses']['numerics_strategy'].remove('test')
    assert possible_strategies1 == possible_strategies2

    possible_strategies1['correct_misses']['category_strategy'].append('test')
    assert possible_strategies1 != possible_strategies2
    possible_strategies1['correct_misses']['category_strategy'].remove('test')
    assert possible_strategies1 == possible_strategies2
    
    possible_strategies1['transform_feat']['numerics_strategy'].append('test')
    assert possible_strategies1 != possible_strategies2
    possible_strategies1['transform_feat']['numerics_strategy'].remove('test')
    assert possible_strategies1 == possible_strategies2
    
    possible_strategies1['transform_feat']['category_strategy'].append('test')
    assert possible_strategies1 != possible_strategies2
    possible_strategies1['transform_feat']['category_strategy'].remove('test')
    assert possible_strategies1 == possible_strategies2
    
    
@preprocessing_expert_cm_numerics_strategy_params
@preprocessing_expert_cm_category_strategy_params
@preprocessing_expert_tf_numerics_strategy_params
@preprocessing_expert_tf_category_strategy_params
def test_preprocessing_expert_get_possible_strategies_with_fit(default_preprocessing_expert_strategies,
                                                               lending_club_data,
                                                               cm_numerics_strategy, 
                                                               cm_category_strategy, 
                                                               tf_numerics_strategy, 
                                                               tf_category_strategy):
    X, y = lending_club_data
    data_report = ExploratoryDataAnalyst().analyze(X, y)
    
    pe = PreprocessingExpert(correct_misses_numerics_strategy=cm_numerics_strategy, 
                             correct_misses_category_strategy=cm_category_strategy, 
                             transform_feat_numerics_strategy=tf_numerics_strategy, 
                             transform_feat_category_strategy=tf_category_strategy) 
        
    pe.fit(X, y, data_report=data_report)
    
    possible_strategies = pe.get_possible_strategies()
    assert possible_strategies == default_preprocessing_expert_strategies 
        

@preprocessing_expert_cm_numerics_strategy_params_no_none
@preprocessing_expert_cm_category_strategy_params_no_none
@preprocessing_expert_tf_numerics_strategy_params_no_none
@preprocessing_expert_tf_category_strategy_params_no_none
def test_preprocessing_expert_fit_transform(lending_club_data,
                                            cm_numerics_strategy, 
                                            cm_category_strategy, 
                                            tf_numerics_strategy, 
                                            tf_category_strategy):        
    X, y = lending_club_data
    data_report = ExploratoryDataAnalyst().analyze(X, y)

    pe = PreprocessingExpert(correct_misses_numerics_strategy=cm_numerics_strategy, 
                             correct_misses_category_strategy=cm_category_strategy, 
                             transform_feat_numerics_strategy=tf_numerics_strategy, 
                             transform_feat_category_strategy=tf_category_strategy) 
        
    pe.fit(X, y, data_report=data_report)
        
    expected_file_name = f'cm__{cm_numerics_strategy}-{cm_category_strategy}__ft__{tf_numerics_strategy}-{tf_category_strategy}.npz'
    expected = np.load(fixture_path(expected_file_name), 'r')['arr_0']
    
    assert_array_equal(pe.transform(X), expected)
