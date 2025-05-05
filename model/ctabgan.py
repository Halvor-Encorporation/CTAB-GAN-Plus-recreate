"""
Generative model training algorithm based on the CTABGANSynthesiser

"""
import pandas as pd
import time
from model.pipeline.data_preparation import DataPrep
from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer

from model.pipeline2.data_preparation import DataPrep as DataPrep2

import warnings

warnings.filterwarnings("ignore")

class CTABGAN():

    def __init__(self,
                 df,
                 test_ratio = 0.20,
                 categorical_columns = [ 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'], 
                 log_columns = [],
                 mixed_columns= {'capital-loss':[0.0],'capital-gain':[0.0]},
                 general_columns = ["age"],
                 non_categorical_columns = [],
                 integer_columns = ['age', 'fnlwgt','capital-gain', 'capital-loss','hours-per-week'],
                 problem_type= {"Classification": "income"}):

        self.__name__ = 'CTABGAN'
              
        self.synthesizer = CTABGANSynthesizer()
        self.raw_df = df
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.general_columns = general_columns
        self.non_categorical_columns = non_categorical_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
                
    def fit(self,epochs= 100):
        
        start_time = time.time()
        self.data_prep = DataPrep(self.raw_df,self.categorical_columns,self.log_columns,self.mixed_columns,self.general_columns,self.non_categorical_columns,self.integer_columns,self.problem_type,self.test_ratio)
        self.data_prep2 = DataPrep2(self.raw_df,self.categorical_columns,self.log_columns)
        self.prepared_data = self.data_prep2.preprocesses_transform(self.raw_df)

        self.synthesizer.fit(train_data=self.prepared_data, 
                     categorical=self.categorical_columns, 
                     mixed=self.mixed_columns, 
                     general=self.general_columns, 
                     non_categorical=self.non_categorical_columns, 
                     type=self.problem_type, 
                     epochs=epochs)
        end_time = time.time()
        print('Finished training in',end_time-start_time," seconds.")


    def generate_samples(self,n=1000):
        
        sample_transformed = self.synthesizer.sample(n)#self.synthesizer.sample(n, column_index, column_value_index)
        sample_transformed = pd.DataFrame(sample_transformed, columns=self.prepared_data.columns)
        
        sample = self.data_prep2.preprocesses_inverse_transform(sample_transformed)
        
        return sample
