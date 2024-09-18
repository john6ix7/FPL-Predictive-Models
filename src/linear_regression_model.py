#-- Auther: John Wilson -- #
#-- version 1.0 -- #
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

    def evaluate_model(self, y_test, y_pred):
        """
        Evaluates the trained model using different metrics.

        Args:
            y_test (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            tuple: Evaluation metrics (MAE, MSE, R2, MAPE, CV MSE).
        """
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Calculate Mean Absolute Percentage Error (MAPE)
        def calculate_mape(y_true, y_pred):
            non_zero_indices = y_true != 0
            if np.any(non_zero_indices):
                return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100
            else:
                return np.inf

        mape = calculate_mape(y_test, y_pred)

        print("Model Evaluation Metrics:")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"R^2 Score: {r2}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

        # Cross-validation score
        scores = cross_val_score(self.model, self.scaler.transform(self.df[self.features]), self.df[self.target], cv=5, scoring='neg_mean_squared_error')
        cv_mse = -scores.mean()
        print(f"Cross-Validation MSE: {cv_mse}")

        # Print feature coefficients
        print("\nFeature Coefficients:")
        for feature, coef in zip(self.features, self.model.coef_):
            print(f"{feature}: {coef:.4f}")

        # Plot feature importance
        self.plot_feature_importance()

        return mae, mse, r2, mape, cv_mse

    def plot_feature_importance(self):
        """
        Plots the feature importance based on the model's coefficients.
        """
        coefficients = self.model.coef_
        feature_importance = np.abs(coefficients)
        importance_df = pd.DataFrame({'Feature': self.features, 'Importance': feature_importance})
        importance_df.sort_values(by='Importance', ascending=False, inplace=True)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.show()

    def add_predictions(self):
        """
        Adds the predicted performance to the dataframe.

        Returns:
            pd.DataFrame: Dataframe with an additional 'predicted_performance' column.
        """
        self.df['predicted_performance'] = self.model.predict(self.scaler.transform(self.df[self.features]))
        return self.df
    
class TeamSelector:
    """
    Class to select the optimal team based on constraints using linear programming.
    """
    
    def __init__(self, df, budget_constraint, max_players, exact_goalkeepers=None, exact_defenders=None):
        """
        Initializes the TeamSelector class.

        Args:
            df (pd.DataFrame): Dataframe containing player data.
            budget_constraint (float): Maximum budget for selecting players.
            max_players (int): Maximum number of players to select.
            exact_goalkeepers (int): Exact number of goalkeepers to select (optional).
            exact_defenders (int): Exact number of defenders to select (optional).
        """
        self.df = df
        self.budget_constraint = budget_constraint
        self.max_players = max_players
        self.exact_goalkeepers = exact_goalkeepers
        self.exact_defenders = exact_defenders

    def select_players(self):
        """
        Solves the linear programming problem to select the optimal team based on performance.

        Returns:
            pd.DataFrame: Dataframe of selected players.
        """
        # Define the optimization problem
        prob = LpProblem("Player_Selection_Problem", LpMaximize)

        # Define decision variables for each player
        player_vars = {index: LpVariable(f"Player_{index}", cat='Binary') for index in self.df.index}

        # Objective function: maximize the sum of predicted performance of selected players
        prob += lpSum(player_vars[i] * self.df.loc[i, 'predicted_performance'] for i in self.df.index)

        # Budget constraint: sum of player costs must be within budget
        prob += lpSum(player_vars[i] * self.df.loc[i, 'now_cost'] for i in self.df.index) <= self.budget_constraint

        # Max players constraint: limit the number of selected players
        prob += lpSum(player_vars[i] for i in self.df.index) <= self.max_players

        # Position-specific constraints
        if self.exact_goalkeepers is not None:
            prob += lpSum(player_vars[i] for i in self.df.index if self.df.loc[i, 'element_type'] == 1) == self.exact_goalkeepers
        if self.exact_defenders is not None:
            prob += lpSum(player_vars[i] for i in self.df.index if self.df.loc[i, 'element_type'] == 2) == self.exact_defenders

        # Solve the optimization problem
        prob.solve()

        # Get selected players based on the solution
        selected_indices = [i for i in self.df.index if value(player_vars[i]) == 1]
        selected_team = self.df.loc[selected_indices]

        return selected_team

def load_data(file_path):
    """
    Loads data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe or an empty dataframe if the file is not found.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Initialize dictionaries to store evaluation metrics and feature coefficients for each position
    metrics_dict = {
        'Attacker': [],
        'Midfielder': [],
        'Defender/Goalkeeper': []
    }

    coefficients_dict = {
        'Attacker': [],
        'Midfielder': [],
        'Defender/Goalkeeper': []
    }