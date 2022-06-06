import copy
import numpy as np
import pandas as pd

from dataclasses import dataclass, field, replace

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from typing import (
    Dict,
    List,
    Tuple
)


@dataclass
class FeatureStats:
    """Holds summary stats about feature values"""
    
    name: str
    index: int
    unique_num: int = 0
    missed_num: int = 0
    
    def reindex(self, index):        
        return replace(self, index=index)
    
    def __str__(self):
        return (
            f'name={self.name}\n'
            f'index={self.index}\n'
            f'unique_num={self.unique_num}\n'
            f'missed_num={self.missed_num}'
        )
    
    
@dataclass
class ExploratoryReport:
    """Holds summary of data set main characteristics""" 

    objects_num: int = 0
    numeric_features: List[FeatureStats] = field(default_factory=list)
    categor_features: List[FeatureStats] = field(default_factory=list)
    numeric_features_idx: Dict[str, FeatureStats] = field(init=False, compare=False, repr=False, hash=None, default_factory=dict) 
    categor_features_idx: Dict[str, FeatureStats] = field(init=False, compare=False, repr=False, hash=None, default_factory=dict)
    
    def __post_init__(self):
        self.numeric_features_idx = {f.name: f for f in self.numeric_features}
        self.categor_features_idx = {f.name: f for f in self.categor_features}

    def __str__(self):
        return (
            f'objects_num={self.objects_num}\n'
            f'numeric_features={self.numeric_features}\n'
            f'categor_features={self.categor_features}'
        )
    
    
class ExploratoryDataAnalyst:
    """Base class for different approaches to analyzing data sets to summarize 
       their main characteristics
    """    
    
    def analyze(self, X: pd.DataFrame, y: np.ndarray) -> ExploratoryReport:
        """Analyzes provided dataset.

        Parameters
        ----------
        X: pd.DataFrame
            A collection of objects (objects-features matrix)
        y: np.ndarray
            Target variables (ground truth labels).
            

        Return
        ------
        ExploratoryReport


        Examples
        --------
        >>> X = pd.DataFrame({'feat1': ['a', 'b', 'a'], 'feat2': [1, 2, 3]})
        >>> y = np.array([0, 0, 1])
        >>> report = ExploratoryDataAnalyst.analyze(X, y)
        """
        
        numeric_cols = X.select_dtypes(include=['number']).columns
        categor_cols = X.select_dtypes(include=['object', 'bool']).columns
        
        numeric_features = [
            FeatureStats(
                name=c,
                index=X.columns.get_loc(c),
                unique_num=len(c_values.unique()),
                missed_num=c_values.isna().sum()
            ) for (c, c_values) in ((c, X[c]) for c in numeric_cols)]

        categor_features = [
            FeatureStats(
                name=c,
                index=X.columns.get_loc(c),
                unique_num=len(c_values.unique()),
                missed_num=c_values.isna().sum()
            ) for (c, c_values) in ((c, X[c]) for c in categor_cols)]

        return ExploratoryReport(
            objects_num=len(X), 
            numeric_features=numeric_features, 
            categor_features=categor_features
        )
        

class PreprocessingExpert(BaseEstimator, TransformerMixin):  
    """Plans a strategy of data preprocessing."""

    def __init__(self, 
                 correct_misses_numerics_strategy: str = 'mean', 
                 correct_misses_category_strategy: str = 'most_frequent', 
                 transform_feat_numerics_strategy: str = 'standardize', 
                 transform_feat_category_strategy: str = 'ohe'):       
        """Creates auto-preprocessing estimator and transformer which makes preprocessing decisions on its own.

        Parameters
        ----------
        correct_misses_numerics_strategy : one from {'mean', 'median'}, default 'mean'. 
            If multiple strategies is selected than correct_misses_numerics_columns must be specified.
                
        correct_misses_category_strategy : list of items from {'most_frequent'}, default ['most_frequent']. 
            If multiple strategies is selected than correct_misses_category_columns must be specified. 
        
        transform_feat_numerics_strategy : list of items from {'standardize', 'minmax'}, default ['standardize']. 
            If multiple strategies is selected than transform_feat_numerics_columns must be specified. 
                
        transform_feat_category_strategy : list of items from {'ohe'}, default ['ohe']. 
            If multiple strategies is selected than transform_feat_numerics_columns must be specified.  
        """
        
        # this default parameters map can be changed to reflect input data characteristics after
        # fit method is called
        self._possible_strategies = {
            'correct_misses': {
                'numerics_strategy': ['mean', 'median'],
                'category_strategy': ['most_frequent']
            },
            'transform_feat': {
                'numerics_strategy': ['standardize', 'minmax'],
                'category_strategy': ['ohe']
            }
        }
                
        self.correct_misses_numerics_strategy = correct_misses_numerics_strategy if correct_misses_numerics_strategy else 'mean'
        self.correct_misses_category_strategy = correct_misses_category_strategy if correct_misses_category_strategy else 'most_frequent'
        self.transform_feat_numerics_strategy = transform_feat_numerics_strategy if transform_feat_numerics_strategy else 'standardize'
        self.transform_feat_category_strategy = transform_feat_category_strategy if transform_feat_category_strategy else 'ohe'
                    
            
    def fit(self, X, y=None, data_report: ExploratoryReport = None):
        if not data_report:
            data_report = ExploratoryDataAnalyst().analyze(X, y)
            
        self._preprocessing_strategy = self.plan(data_report)
        self._preprocessing_strategy.fit(X, y)
        
        return self

    def transform(self, X):
        return self._preprocessing_strategy.transform(X)

    def get_possible_strategies(self):
        return copy.deepcopy(self._possible_strategies)
    
    def plan(self, data_report: ExploratoryReport) -> Pipeline:
        """Plans a strategy of dataset preprocessing based on its data report.

        Parameters
        ----------
        data_report : ExploratoryReport
            Dataset description with different statistics


        Return
        ------
        Pipeline
            The suggested strategy represented as Pipeline


        Examples
        --------
        >>> X = pd.DataFrame({'feat1': ['a', 'b', 'a'], 'feat2': [1, 2, 3]})
        >>> y = np.array([0, 0, 1])
        >>> dr = ExploratoryDataAnalyst.analyze(X, y)
        >>> self.plan(data_report=dr)
        """
        
        correct_misses_strategy, adj_report1 = self.plan_missing_values_strategy(data_report=data_report)
        transform_feat_strategy, adj_report2 = self.plan_transform_values_strategy(data_report=adj_report1)
    
        return Pipeline(steps=[
            ('correct_misses', correct_misses_strategy),
            ('transform_feat', transform_feat_strategy)
        ])
        
    def plan_missing_values_strategy(self, data_report: ExploratoryReport) -> Tuple[Pipeline, ExploratoryReport]:
        """This method owns knowledge about how to deal with missing values.

        Plans a strategy of dealing with missing values for dataset based on its data report.
        It tries to guess what are the possible strategies and what is the best strategy 
        to apply to such dataset. Different strategies will lead to different output shape,
        so planning also includes adjusting the input data report to correctly handle output 
        data in further preprocessing.

        Strategies depend on data statistics, selected model, etc. and can be various:
           - remove objects
           - remove feature
           - remove feature if collinear with other features
           - impute mean, moda
           - impute using KNN
          ...
          
          
        Parameters
        ----------
        data_report : ExploratoryReport
            Dataset description with different statistics
            
            
        Return
        ------
        Tuple[Pipeline, ExploratoryReport]
            the first tuple component is the suggested strategy represented as Pipeline
            the second tuple component is the adjusted data report for future use
        """
            
        numeric_cols = [f.index for f in data_report.numeric_features]
        categor_cols = [f.index for f in data_report.categor_features]
        
        nums_sz = len(numeric_cols)
        cats_sz = len(categor_cols)
        
        num_nan_miss_vals_imputer = SimpleImputer(missing_values=np.nan, strategy=self.correct_misses_numerics_strategy)
        cat_nan_miss_vals_imputer = SimpleImputer(missing_values=np.nan, strategy=self.correct_misses_category_strategy)
        
        adjusted_report = replace(
            data_report, 
            numeric_features=[f.reindex(index=i) 
                              for i, f in zip(range(nums_sz), data_report.numeric_features)], 
            categor_features=[f.reindex(index=i) 
                              for i, f in zip(range(nums_sz, nums_sz+cats_sz), data_report.categor_features)]
        )
        
        ct = ColumnTransformer([
            # these transformers do not change the input shape
            # TODO: open question is what to do if the input shape changed and we
            #       can not rely on data_report in next transformers
            ('num_nan_miss_vals_imputer', num_nan_miss_vals_imputer, numeric_cols),
            ('cat_nan_miss_vals_imputer', cat_nan_miss_vals_imputer, categor_cols)
        ])
        
        return ct, adjusted_report
 
    def plan_transform_values_strategy(self, data_report: ExploratoryReport) -> Tuple[Pipeline, ExploratoryReport]:
        """This method owns knowledge about how to transform different features.

        Strategies depend on data statistics and can be various for different feature
        scales - absolute aka numeric, interval, ordinal, nominal aka categorical - and 
        admissible set of operations defined for each scale.

        Upproaches can vary depending on selected model, data sparsity, number of outliers, 
        correlation and others.
        
        
        Parameters
        ----------
        report : ExploratoryReport
            Dataset description with different statistics
            
            
        Return
        ------
        Tuple[Pipeline, ExploratoryReport]
            the first tuple component is the suggested strategy represented as Pipeline
            the second tuple component is the adjusted data report for future use
        """
    
        numeric_cols = [f.index for f in data_report.numeric_features]
        categor_cols = [f.index for f in data_report.categor_features]
        
        if self.transform_feat_numerics_strategy == 'standardize':
            numerics_step = ('num_transform_standardize', StandardScaler(), numeric_cols)
        else:
            numerics_step = ('num_transform_minmax', MinMaxScaler(), numeric_cols)
            
        ct = ColumnTransformer([
            numerics_step,
            
            # these transformer will change the input shape, so futher
            # one can not use categor_cols from data_report
            ('cat_transform_oneh_encode', OneHotEncoder(sparse=False, handle_unknown='ignore'), categor_cols)
        ])

        return ct, None
