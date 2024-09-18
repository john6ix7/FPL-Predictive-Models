#-- Auther: John Wilson --
#-- version 1.0 --
# Import necessary libraries for data processing, modeling, evaluation, and optimization
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value

class PlayerModel:
	"""
    Class for creating and training a model to predict player performance.
    """
    
    def __init__(self, df, features, target='points_per_game'):
        """
        Initializes the PlayerModel class.

        Args:
            df (pd.DataFrame): Dataframe containing player data.
            features (list): List of feature columns used for training the model.
            target (str): Target column to predict (default is 'points_per_game').
        """
        self.df = df
        self.features = features
        self.target = target
        self.model = None
        self.scaler = StandardScaler()
        
    def train_model(self, test_size=0.1, random_state=42):
        """
        Trains the model using a linear regression algorithm.
        
        Args:
            test_size (float): Proportion of data to use for testing.
            random_state (int): Random seed for reproducibility.

        Returns:
            tuple: Evaluation metrics (MAE, MSE, R2, MAPE, CV MSE).
        """
        X = self.df[self.features]
        y = self.df[self.target]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Scale the features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Train a linear regression model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = self.model.predict(X_test)

        return self.evaluate_model(y_test, y_pred)