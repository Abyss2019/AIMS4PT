from __future__ import annotations

"""Utilities for eruption classification models."""

import numpy as np
import pandas as pd

from aims4pt.data_tools.data_engineering import (
    add_ratio,
    add_ratio_keep_nan_keep_inverse,
    add_ratio_keep_nan_no_inverse,
)
from aims4pt.model_tools.ModelManager import ModelManager




class EruptionClassification_ModelManager(ModelManager):
    """Base class for eruption classification models."""

    def __init__(self, model, features: list[str], comments: str | None = None):
        """
        Initialize with a model object and standard columns.

        Parameters
        ----------
        model : object
            Underlying estimator (Python model or wrapped R model).
        features : list[str]
            Feature names expected by the model in standard order.
        comments : str, optional
            Additional comments or metadata for the model.
        """
        super().__init__(model, comments=comments)
        # Load feature information from template CSV for consistent formatting
        self.standard_columns = features
        

    def predict_and_get_df(self, X: pd.DataFrame):
        """Predict and return both the original and formatted DataFrames.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features; order is flexible as long as required columns are
            present.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame]
            Original DataFrame with predictions appended, and a formatted copy
            ready for persistence.
        """
        X = X.copy()
        prediction = self.predict(X)
        X[self.prediction_column_name] = prediction

        # Get formatted X for saving
        formatted_X = super().format_input(X)
        formatted_X[self.prediction_column_name] = prediction

        return X, formatted_X

    def predict(self, X: pd.DataFrame, if_origin: bool = False):
        """Predict using the wrapped model.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features; column order is flexible when names are present.
        if_origin : bool, default False
            If True, return labels after inverse-transforming via the label
            encoder; otherwise return encoded predictions.
        """
        if self.model is None:
            raise ValueError('Model has not been trained yet.')

        X = self.preprocess(X)
        prediction = self.model.predict(X)
        prediction_origin = self.label_encoder.inverse_transform(prediction)
        if if_origin:
            return prediction_origin
        else:
            return prediction
    
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select and order the expected columns for prediction."""
        try:
            return X[self.standard_columns]
        except KeyError:
            raise KeyError('Input columns do not match the expected standard columns.')

 

    def save(self, path):
        '''
        Save the model to a file.

        Parameters:
            path (str): 
                File path to save the model.
        '''
        import pickle
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        '''
        Load the model from a file.

        Parameters:
            path (str): 
                File path to load the model.

        Returns:
            EruptionClassificationModel: The loaded model.
        '''
        import pickle
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model


class EruptionClassification_xgboost (EruptionClassification_ModelManager):

    '''
    Class for Eruption Classification models using XGBoost.
    '''

    def __init__(self, features, comments=None, with_ratio=True,  with_texture=False, add_ratio_method=add_ratio, balance_training_data=None, weight=None, eruption_order=None):
        '''
        initialize the model with XGBoost and train the model with the data.

        Target column: 'Eruption'
        Textural position column: 'textural position'

        Parameters:

            features (list): 
                List of feature names in the order expected by the model.
            random_state (int):
                Random seed for reproducibility.
            with_ratio (bool):
                Whether to add ratio features to the data.
            with_texture (bool):
                Whether to add texture positions features.
            add_ratio_method (function):
                The method to add ratio features.
            balance_training_data (bool):
                The method to balance the training data.
            weight (dict):
                The weight of each class for imbalance data.

        '''
        self.prediction_column_name = 'Eruption'
        self.with_ratio = with_ratio
        self.with_texture = with_texture
        self.add_ratio_method = add_ratio_method
        self.balance_training_data = balance_training_data
        self.weight = weight
        self.eruption_order = eruption_order
        

        super().__init__(None, features, comments=comments)

    def __str__(self):
        return f'\nModel: XGBoost' + f'\nFeatures: {self.standard_columns}' + f'\nWith ratio: {self.with_ratio}' + f'\nWith texture: {self.with_texture}' + f'\nAdd ratio method: {self.add_ratio_method.__name__}' 


    def train(self, data, searching_times=100, random_state=42, searching_space=None):
        '''
        initialize the model with XGBoost and train the model with the data.

        Target column: 'Eruption'

        Parameters:

            features (list): 
                List of feature names in the order expected by the model.
            random_state (int): 
                Random seed for reproducibility.
            with_ratio (bool): 
                Whether to add ratio features to the data.
            comments (str):
                Additional comments or metadata for the model.
            searching_times (int):
                The number of searching times for hyperparameter tuning.
            with_texture (bool):
                Whether to add texture positions features.



        '''        
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import KFold
        from sklearn.metrics import make_scorer
        from sklearn.preprocessing import LabelEncoder
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        from ray.air import session
        import logging
        from ray import init, shutdown

        # logging.basicConfig(level=logging.ERROR)
        data = data.copy()

        
        with_ratio = self.with_ratio
        features = self.standard_columns

        # set random seed
        np.random.seed(random_state)

        # do not print warnings
        # init(ignore_reinit_error=True)

        # if in Jupyter Notebook, use IPython.display
        # try:
        #     from IPython import get_ipython
        #     if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':  # Jupyter Notebook             
        #         from IPython.display import display
        #         from IPython import display as ipydisplay
        #         from ipywidgets import widgets
                
        # except:
        #     pass



        all_columns = data.columns
        not_features = [col for col in all_columns if col not in features]

        label = LabelEncoder()

        num_classes = len(data['Eruption'].unique())

        if with_ratio:
            df_ratio = self.add_ratio_method(data, features, random_state)
        else:
            df_ratio = data

        df_ratio['label'] = label.fit_transform(df_ratio['Eruption'])


        self.label_encoder = label
        label_mapping = pd.DataFrame(
            {'Eruption': label.classes_, 'label': label.transform(label.classes_)})
        
        

        X = df_ratio[[col for col in df_ratio.columns if col not in not_features]]
        X = X.drop('label', axis=1)
        if self.with_texture:
            X["textural position"] = data["textural position"]
            # encode textural position
            self.textural_position_encoder = LabelEncoder()
            X["textural position"] = self.textural_position_encoder.fit_transform(X["textural position"])



        y = df_ratio['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state)
        
        # store test dataset
        self.X_test = X_test
        self.y_test = y_test
        
        # balance the training data
        if self.balance_training_data:
            X_train, y_train = self.balance_training_data(X_train, y_train, random_state=random_state)


        def train_xgboost_with_cv(config):

                # 根据分类任务动态设置 objective 和 num_class
            if num_classes > 2:
                config["objective"] = "multi:softprob"
                config["num_class"] = num_classes
            else:
                config["objective"] = "binary:logistic"

            config["n_estimators"] = int(config["n_estimators"])
            config["max_depth"] = int(config["max_depth"])
            config["eval_metric"] = "auc"

            # weight
            if self.weight:
                model = xgb.XGBClassifier(**config, scale_pos_weight=self.weight)
            # 定义 XGBoost 模型，超参数由 config 传入
            model = xgb.XGBClassifier(**config)

            # 10折交叉验证
            kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
            scores = cross_val_score(
                model, X, y, cv=kf, scoring=make_scorer(accuracy_score))

            # 计算 10 折平均准确率
            mean_accuracy = scores.mean()

            # 上报平均准确率给 Ray Tune
            session.report({"accuracy": mean_accuracy})

        # 3. 定义搜索空间
        if searching_space is None:

            search_space = {
            # 树模型参数
            "max_depth": tune.randint(3, 10),            # 树的最大深度
            "min_child_weight": tune.uniform(1.0, 10.0),  # 最小叶子节点样本权重和
            "gamma": tune.uniform(0, 0.5),               # 最小分裂损失
            "subsample": tune.uniform(0.5, 1.0),         # 每棵树的样本采样比例
            "colsample_bytree": tune.uniform(0.5, 1.0),  # 每棵树的特征采样比例

            # 学习率和迭代次数
            "learning_rate": tune.loguniform(0.001, 0.1),  # 学习率（eta）
            "n_estimators": tune.randint(1, 500),  # 迭代次数

            # 正则化参数
            "reg_alpha": tune.loguniform(1e-8, 10),      # L1 正则化项
            "reg_lambda": tune.loguniform(1e-6, 10),     # L2 正则化项

            # 验证指标
            # "eval_metric": tune.choice(["logloss", "auc", "error"]),
        }
        else:
            search_space = searching_space

        # 4. 配置调度器（如 ASHA）
        scheduler = ASHAScheduler(
            metric="accuracy",
            mode="max",
            max_t=50,  # 最大训练周期
            grace_period=30,  # 早停的宽限期
        )

        # 5. 使用 Ray Tune 进行超参数调优
        tuner = tune.run(
            #

            train_xgboost_with_cv,  # 训练函数
            config={
                **search_space,
                "random_state": random_state,  # 在搜索空间中设置随机种子


            },
            num_samples=searching_times,  # 搜索次数
            scheduler=scheduler,  # 使用调度器
            verbose=0,
            log_to_file=True,
            trial_dirname_creator=custom_trial_dirname
        )

        # 6. 获取最佳结果
        best_trial = tuner.get_best_trial(metric="accuracy", mode="max")
        # best config
        best_config = best_trial.config

        from aims4pt.toolkit_utils import ray_searching_curve

        ray_searching_curve(tuner.results_df, "accuracy")

        model = xgb.XGBClassifier(**best_config)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # confusion matrix
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        cm = confusion_matrix(y_test, y_pred)
        self.test_cm = cm
        plt.figure(figsize=(10, 8), dpi=150)
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
        plt.xticks(ticks=np.arange(len(self.label_encoder.classes_)) + 0.5,
            labels=self.label_encoder.classes_, rotation=45)
        plt.yticks(ticks=np.arange(len(self.label_encoder.classes_)) + 0.5,
                   labels=self.label_encoder.classes_, rotation=45)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

        train_accuracy = accuracy_score(y_train, model.predict(X_train))

        print(f"Train accuracy: {train_accuracy:.2f}")
        print("Best trial config:", best_trial.config)
        print("Best trial final accuracy:", best_trial.last_result["accuracy"])
        print(label_mapping)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        
        # calculate feature importance
        from xgboost import plot_importance

        # in ratio

        # 获取特征重要性
        importance = model.feature_importances_

        # 转换为 DataFrame 并排序
        feature_names = X_train.columns  # 如果使用 pandas 数据集
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        })
        importance_df = importance_df.sort_values(
            by="Importance", ascending=False)
        importance_df.reset_index(drop=True, inplace=True)

        self.importance_df = importance_df

        # 绘制比例条形图
        plt.figure(figsize=(10, 12), dpi=150)
        plt.barh(importance_df["Feature"], importance_df["Importance"],
                 color="skyblue", edgecolor="black")
        for i in range(len(importance_df)):
            plt.text(importance_df["Importance"][i], i, f"{importance_df['Importance'][i]:.2f}", ha='right', va='center')
        plt.xlabel("Feature Importance")
        plt.title("XGBoost Feature Importance")
        plt.gca().invert_yaxis()  # 反转Y轴
        plt.show()

        self.model = model
        self.standard_columns_with_ratio = X.columns.tolist()

        # shutdown()





    def preprocess(self, X):
        X = X.copy()
        if self.with_texture:
            # print(X["textural position"])
            X["textural position"] = self.textural_position_encoder.transform(X["textural position"])
            textural_position = X["textural position"]

        try:
            X = X[self.standard_columns]
            if self.with_texture:
                X["textural position"] = textural_position
        except KeyError:
            raise KeyError(
                'Input columns do not match the expected standard columns.')
        
        if self.with_ratio:
            X  = add_ratio_keep_nan_keep_inverse(X, self.standard_columns)
            # X = X[self.standard_columns_with_ratio] add nan when key error
            for col in self.standard_columns_with_ratio:
                if col not in X.columns:
                    X[col] = np.nan
            X = X[self.standard_columns_with_ratio]

        return X





    def count_data_number(self, data):
        '''
        Count the number of data points in the input data.

        '''
        import matplotlib.pyplot as plt
        import seaborn as sns

        unit_label = data[['Eruption']]
        unit_label['count'] = 1
        unit_label = unit_label.groupby('Eruption').count().reset_index()

        # plot
        plt.figure(figsize=(10, 6), dpi=150)
        # add number of data points
        sns.barplot(x='Eruption', y='count', data=unit_label)
        for i in range(len(unit_label)):
            plt.text(i, unit_label['count'][i],
                     unit_label['count'][i], ha='center')
        plt.xticks(rotation=45)
        plt.ylabel('Number of data points')
        plt.xlabel('eruption')
        plt.title('Number of data points in each class')

        plt.xticks(rotation=45)
        plt.show()

        return unit_label

    def confusion_matrix(self, data, features_order = None):
        '''
        Plot the confusion matrix of the model.

        Parameters:
            data (pd.DataFrame):
                The input data.
            features_order (list, optional):
                The order of the features to plot. Defaults to None, which uses the default order.

        '''
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        y = data['Eruption']
        # map y
        y = self.label_encoder.transform(y)

        y_pred = self.predict(data)
        cm = confusion_matrix(y, y_pred)  # Use the provided order

        # Get default class order from the label encoder
        default_order = list(self.label_encoder.classes_)

        if features_order is None:
            features_order = default_order

        # Ensure the custom order is valid
        if not set(features_order).issubset(set(default_order)):
            raise ValueError("features_order contains invalid labels that are not in the default class order.")


        # Map the custom order to the indices in the default order
        order_indices = [default_order.index(label) for label in features_order]

        # Reorder the confusion matrix rows and columns
        cm = cm[np.ix_(order_indices, order_indices)]

        # Plot the confusion matrix
        plt.figure(figsize=(10, 8), dpi=150)
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', cbar=True,
                    xticklabels=features_order, yticklabels=features_order)
        # color bar
        # plt.colorbar(label='Count', ax=plt.gca())
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

        return cm



    def features_trend(self, data, x, y):
        '''
        Plot the trend of the features in the input data.


        '''
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 6), dpi=300)
        for eruption in data['Eruption'].unique():
            df_eruption = data[data['Eruption'] == eruption]

            plt.scatter(df_eruption[x], df_eruption[y],
                        label=eruption, s=50, alpha=0.6, edgecolors='black')

        plt.xlabel(fontsize=12, xlabel=f"{x}")
        plt.ylabel(fontsize=12, ylabel=f"{y}")

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)

        plt.show()


    def features_trend_compare_prediction(self, data, x, y):
        """
        Plot the trend of the features in the input data, showing actual Eruption categories 
        and whether predictions are correct or not. Colors are consistent for the same Eruption.

        Parameters:
            data (pd.DataFrame): The input data with features and actual labels.
            x (str): Feature name for x-axis.
            y (str): Feature name for y-axis.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import matplotlib.cm as cm
        import numpy as np

        plt.figure(figsize=(10, 6), dpi=300)

        # Generate predictions and map them
        predictions = self.predict(data)
        predictions = self.label_encoder.inverse_transform(predictions)

        # Add prediction correctness column
        data['Prediction'] = predictions
        data['Correct'] = data['Eruption'] == data['Prediction']

        # Define a consistent color map for actual Eruption categories
        eruption_categories = data['Eruption'].unique()
        colors = cm.get_cmap('viridis', len(eruption_categories))  # Use a colormap
        color_map = {eruption: f'C{i % 10}' for i, eruption in enumerate(eruption_categories)}

        # Plot data points
        for eruption in eruption_categories:
            df_eruption = data[data['Eruption'] == eruption]
            for correct in [True, False]:
                subset = df_eruption[df_eruption['Correct'] == correct]
                plt.scatter(
                    subset[x], subset[y],
                    label=f"{eruption} - {'Correct prediction' if correct else 'Incorrect prediction'}",
                    s=50, alpha=0.8, edgecolors='black',
                    c=[color_map[eruption]] * len(subset),
                    marker='o' if correct else 'x'
                )

        # Set plot details
        plt.xlabel(x, fontsize=12)
        plt.ylabel(y, fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=10, loc='best', ncol=2)
        # plt.title(f"Feature Trend: {x} vs {y}", fontsize=14)
        plt.tight_layout()
        plt.show()


    
    def plot_feature_trends(self, data):
        '''
        Plot the trends of the features in the input data.

        Parameters:
            data (pd.DataFrame):
                The input data.
            features (list):
                List of feature names to plot.
            target (str):
                The target column name.

        '''
        import matplotlib.pyplot as plt
        import seaborn as sns

        features = self.standard_columns
        if self.with_texture:
            features.append('textural position')
        
        target = 'Eruption'

        for feature in features:
            plt.figure(figsize=(10, 6), dpi=150)
            sns.lineplot(data=data, x=target, y=feature, marker='o')
            plt.title(f'Trend of {feature} for different {target} classes')
            plt.xlabel(target)
            plt.ylabel(feature)
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.show()

    def report(self, data, searching_times=100, random_state=42, **kwargs):
        '''
        Generate a report for the model.

        Parameters:
            data (pd.DataFrame):
                The input data.

        '''
        print("Model Report")
        print('------------------------------------------------------')
        print("Data Overview")
        self.count_data_number(data)
        print('------------------------------------------------------')
        print("training...")
        self.train(data, searching_times=searching_times, random_state=random_state, **kwargs)
        print('------------------------------------------------------')
        print("Analysis: all data")
        self.confusion_matrix(data)
        self.plot_feature_trends(data)
        print('------------------------------------------------------')
        print(self)
        print("End of Report")
        print("note: use features_trend_compare_prediction to see the trend of features and predictions")


def custom_trial_dirname(trial):
    # 只保留 trial 的 ID 或者简短信息
    return f"trial_{trial.trial_id}"