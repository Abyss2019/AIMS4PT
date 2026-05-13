
"""Higgins et al. (2021) clinopyroxene-only thermobarometry model wrapper."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import aims4pt.model_tools.data.Higgins21 as higgins_data
from aims4pt.model_tools.ModelManager import ModelManager
from aims4pt.model_tools.model_registry import register_model
from aims4pt.model_tools.rpy_tools.r_env import (
    configure_r_environment,
    format_rpy2_setup_error,
)
from aims4pt.toolkit_utils import get_file_path


# print("check")

@register_model
class Higgins21(ModelManager):
    '''
    Higgins, O., Sheldrake, T., & Caricchi, L. (2021). Machine learning thermobarometry and chemometry using amphibole and clinopyroxene: A window into the roots of an arc volcano (Mount Liamuiga, Saint Kitts). Contributions to Mineralogy and Petrology, 177(1). https://doi.org/10.1007/s00410-021-01874-6

clinopyroxene_SiO2	clinopyroxene_Al2O3	clinopyroxene_TiO2	clinopyroxene_CaO	clinopyroxene_Na2O	clinopyroxene_FeO	clinopyroxene_MgO	clinopyroxene_MnO	clinopyroxene_Cr2O3

    '''

    def __init__(self, T_P: str, comments: Optional[str] = None):
        '''
        Initialize with a model object and standard columns.

        Parameters:
            T_P (str):
                "T" for temperature, "P" for pressure.
            comments (str): 
                Additional comments or metadata for the model.
        '''

        super().__init__(comments=comments)
        self.model_name = "Higgins et al., 2021"

        self.require_water = False
        if T_P == "T":
            self.prediction_column_name = "T_C"
            self.uncertainty = 57
        else:
            self.prediction_column_name = "P_kbar"
            self.uncertainty = 2.3

        # melt (SiO2, TiO2, Al2O3, FeOt, MnO, MgO, CaO, Na2O, K2O, Cr2O3, P2O5, and H2O) and clinopyroxene (SiO2, TiO2, Al2O3, FeOt, MnO, MgO, CaO, Na2O, K2O, and Cr2O3)
        self.cpx_names = ["SiO2_cpx", "Al2O3_cpx", "TiO2_cpx", "CaO_cpx",  "Na2O_cpx", "FeO_cpx", "MgO_cpx", 
                          "MnO_cpx", "Cr2O3_cpx" ]

        standard_columns = self.cpx_names

        
        self.standard_columns = standard_columns

        self.model_save_path = None # Disposable path for pkl use only.
        self.models_dir = get_file_path(
            higgins_data, "model")

        self.T_P = T_P
        self.cpx_only = True

        self.if_support_hydrous = True

        X_cpx_train_pkl_name = "datapkl/X_cpx_train.pkl"
        self.X_cpx_train_pkl_path = get_file_path(
            higgins_data, X_cpx_train_pkl_name)
        
        X_liq_train_pkl_name = "datapkl/X_liq_train.pkl"
        self.X_liq_train_pkl_path = get_file_path(
            higgins_data, X_liq_train_pkl_name)

        self.initialize_model(cpx_training_path=self.X_cpx_train_pkl_path,
                              liq_training_path=self.X_liq_train_pkl_path)

        r_model_file = "model/T_cats_C.Rdata" if T_P == "T" else "model/P_cats_C.Rdata"
        self.r_model_path = get_file_path(higgins_data, r_model_file)

    def format_input(self, X_cpx: pd.DataFrame) -> pd.DataFrame:
        '''
        Sample	clinopyroxene_SiO2	clinopyroxene_Al2O3	clinopyroxene_TiO2	clinopyroxene_CaO	clinopyroxene_Na2O	clinopyroxene_FeO	clinopyroxene_MgO	clinopyroxene_MnO	clinopyroxene_Cr2O3

        '''
        from aims4pt.utils import normalize_column_names
        X_cpx_norm_col = ["Sample"]
        X_cpx_oxides = ["clinopyroxene_SiO2", "clinopyroxene_Al2O3", "clinopyroxene_TiO2",
                        "clinopyroxene_CaO", "clinopyroxene_Na2O", "clinopyroxene_FeO",
                        "clinopyroxene_MgO", "clinopyroxene_MnO", "clinopyroxene_Cr2O3"]
        X_cpx_norm = normalize_column_names(X_cpx, X_cpx_oxides,)
        X_cpx_norm_col.extend(X_cpx_oxides)
        X_cpx_norm["Sample"] = np.arange(len(X_cpx_norm)) + 1
        X_cpx_norm.fillna(0, inplace=True)
        X_cpx_processed = X_cpx_norm[X_cpx_norm_col]

        return X_cpx_processed


    def save(self) -> None:
        pass

    def predict(self, X_cpx: pd.DataFrame, X_liq = None) -> pd.Series:
        '''
        Predict using the model.

        Parameters:
            X (pd.DataFrame): 
                Input DataFrame. The order of the columns does not matter, but it is recommended to use the standard column names.
                In principle, it only needs to contain all required features; order and exact names are not required,
                but using the standard column order is recommended.

        Returns:
            pd.Series: The predicted values.    
        '''
        configure_r_environment()
        try:
            from rpy2.robjects import pandas2ri
            import rpy2.robjects as robjects
            from rpy2.robjects.conversion import localconverter
            from rpy2.robjects import default_converter
        except Exception as exc:
            raise RuntimeError(format_rpy2_setup_error(exc)) from exc

        X_cpx = X_cpx.copy()
        X_input = self.format_input(X_cpx=X_cpx)

        # load libraries
        robjects.r(r'''
            
            options(repos = c(CRAN = "https://cloud.r-project.org")) 
            options(java.parameters = "-Xmx8g")

            pack0 <- suppressWarnings(require(rJava))
            if (!pack0) {
            install.packages("rJava", dependencies=TRUE, type = "win.binary")
            library(rJava)
            }

                   
            pack1 <- suppressWarnings(require(extraTrees))
            if (!pack1) {
            install.packages(
            "https://cran.r-project.org/src/contrib/Archive/extraTrees/extraTrees_1.0.5.tar.gz",
            repos = NULL, type = "source")
            library(extraTrees)
            }
            
            pack2 <- suppressWarnings(require(EnvStats))
            if (!pack2) {
            install.packages("EnvStats", dependencies=TRUE, type = "win.binary")
            library(EnvStats)
            }
            
            rm(pack1, pack2)
        ''')



        with localconverter(default_converter + pandas2ri.converter):
            robjects.globalenv["input"] = pandas2ri.py2rpy(X_input)

        Oxi_Weight_path = get_file_path(higgins_data, "model/OxiWeight.Rdata")
        robjects.r['load'](str(Oxi_Weight_path))

        robjects.r(
        r'''
        #Make a list of all possible oxiides that can be dealt with
        all.ox <- rownames(OxiWeight)
        #Locate oxide columns
        headers <- colnames(input)
        input.ox <- headers[which(headers %in% paste0("clinopyroxene_",rownames(OxiWeight)))]
        ox <- gsub(x = input.ox,pattern = "clinopyroxene_", replacement = "")
        rm(headers)
        #The number of oxygens to calculate on the basis of which depends upon the mineral formula you are calculating
        OxNum <- 6
        #This is the numbers from Ox.frame relevant to the raw data inputted
        Ox <- OxiWeight$Ox[match(ox, all.ox)]
        #Take the each oxide and divide by its appropriate atomic weight as taken from the data OxiWeight
        molprop <- apply(input[, input.ox], MARGIN = 1, function(x) x / round(OxiWeight[ox, 'OWeight'], digits = 2))
        #calculate the atomic proportion of oxygen. This is the molar proportion multiplied by the number of oxygens in the oxide
        AtPropOx <- apply(molprop, MARGIN = 2, function(x) x * Ox)
        #Sum the atomic proportion of oxygen to give the total atomic proportion of oxygen
        TotalOx <- apply(AtPropOx, MARGIN = 2, function(x) sum(x))
        #Calculate the value to use in the recalculation
        OxRecalc <- OxNum / TotalOx
        #Calculate the number of anions on the basis of the desired number of oxygens
        AnionsPerOxNum <- apply(AtPropOx, MARGIN = 1, function(x) x * OxRecalc)
        CatRecalc <- OxiWeight$Cat[match(ox, all.ox)]
        #Calculate the number of cations in the desired formula
        cations <- apply(AnionsPerOxNum, MARGIN = 1, function(x) x / CatRecalc)
        #Transpose the result to allow simpler calculation of mineral ratios etc
        cations <- as.data.frame(t(cations))
        #Select the cation names
        elems <- OxiWeight$element[match(ox, all.ox)]
        colnames(cations) <- elems
        #Bind together to make a final input dataframe
        dat <- cbind(input, cations)
        #calculate cation sum
        dat$CatSum <- apply(cations, 1, sum)
        #Clean environment (all except dat)
        rm(list = ls()[which(ls()!="dat")])
        '''
        )

        # load the R models
        robjects.r['load'](str(self.r_model_path))
        # predict
        robjects.r(r'''
        id.cats <- c("Si", "Al", "Ti", "Ca", "Na", "Fe", "Mg", "Mn", "Cr")
        ''')

        if self.T_P == "T":
            robjects.r(r'''
            T <- predict(T_cats_C, newdata = dat[, id.cats], allValues = T)
            dat$T <- round(apply(T, 1, median), 0)
            dat$T_uncer <- round(apply(T,1,IQR),0)/2
            ''')
            # to np.ndarray
            dat = robjects.globalenv['dat']
            T_col = dat.rx2('T')           # Equivalent to dat$T.
            prediction = np.array(T_col)   # Convert to a NumPy array.

        else:
            robjects.r(r'''
            P <- predict(P_cats_C, newdata = dat[, id.cats], allValues = T)
            dat$P <- round(apply(P, 1, median), 1)
            dat$P_uncer <- round(apply(P,1,IQR),1)/2
            ''')

            # to np.ndarray
            dat = robjects.globalenv['dat']
            P_col = dat.rx2('P')           # Equivalent to dat$P.
            prediction = np.array(P_col)   # Convert to a NumPy array.

        # to pd.Series
        prediction = pd.Series(prediction, name=self.prediction_column_name, index=X_cpx.index)
        return prediction



# test code
if __name__ == "__main__":
    print("Testing Higgins21 model...")
    model = Higgins21(T_P="T")
    prediction = model.predict(X_cpx=model.X_cpx_training)
    print(prediction)
