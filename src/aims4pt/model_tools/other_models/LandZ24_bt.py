"""Li & Zhang (2022) biotite thermobarometry model wrappers."""

from __future__ import annotations


from aims4pt.model_tools.ModelManager import ModelManager
from aims4pt.model_tools.rpy_tools.rds_model import r_model


class LandZ24_bt_base_model(ModelManager):
    '''
    Base class for LandZ24 Biotite Thermobarometry models.
    Li, X., & Zhang, C. (2022). Machine Learning Thermobarometry for Biotite-Bearing Magmas. 
    Journal of Geophysical Research: Solid Earth, 127(9), e2022JB024137. https://doi.org/10.1029/2022jb024137

    '''
    
    def __init__(self, model, comments=None):
        '''
        Initialize with a model object and standard columns.
        
        Parameters:
            model Model object:
                could be a Python model or a wrapped R model (r_model).
            comments (str): 
                Additional comments or metadata for the model.
        '''
        super().__init__(model, comments=comments)
        self.model_name = 'Li & Zhang, 2022 (10kbar recalibrated; bt-only)'
        self.bt_names_ = [
            'SiO2_bt', 'TiO2_bt', 'Al2O3_bt', 'FeOt_bt', 'MnO_bt', 'MgO_bt',
            'CaO_bt', 'Na2O_bt', 'K2O_bt', 'F_bt', 'Cl_bt'
        ]
        self.bt_names = [
            'Bt.SiO2', 'Bt.TiO2', 'Bt.Al2O3', 'Bt.FeOt', 'Bt.MnO', 'Bt.MgO',
            'Bt.CaO', 'Bt.Na2O', 'Bt.K2O', 'Bt.F', 'Bt.Cl'
        ]
        self.standard_columns = self.bt_names
        
        # Load feature information from template CSV for consistent formatting
        # super().load_set_feature_info(r'C:\Users\13493\OneDrive - The University of Hong Kong - Connect\Document\aims4pt\files\LandZ24_bt_template.csv')


    def predict(self, X):
        '''
        Predict using the model.
        
        Parameters:
            X (pd.DataFrame): 
                Input DataFrame. The order of the columns does not matter, but it is recommended to use the standard column names.
                理论上说只要包含了所有所需特征就可以了（不要求顺序，也不要求名称完全一致），不过最好是按照标准列名的顺序。

        Returns:
            pd.Series: The predicted values.    
        '''
        X = X.copy()
        prediction = super().predict(X)
        return prediction


class LandZ24_bt_T_model(LandZ24_bt_base_model):
    '''
    Temperature prediction model for LandZ24 Biotite Thermobarometry.
    '''
    def __init__(self, rds_path_T, comments=None):
        '''
        Initialize with the temperature model.
        
        Parameters:
            rds_path_T (str): 
                Path to the RDS file for the temperature model.
            comments (str): 
                Additional comments or metadata for the model.
        '''
        T_model = r_model.load(rds_path_T)
        super().__init__(T_model, comments=comments)
        self.prediction_column_name = 'T.C'
        self.T_P = "T"
        self.bt_only = True
        


class LandZ24_bt_P_model(LandZ24_bt_base_model):
    '''
    Pressure prediction model for LandZ24 Biotite Thermobarometry.
    '''
    def __init__(self, rds_path_P, comments=None):
        '''
        Initialize with the pressure model.    
        
        Parameters:
            rds_path_P (str): 
                Path to the RDS file for the pressure model.
            comments (str): 
                Additional comments or metadata for the model.
        '''
        P_model = r_model.load(rds_path_P)
        super().__init__(P_model, comments=comments)
        self.prediction_column_name = 'P.Kbar'
        self.T_P = "P"
        self.bt_only = True
