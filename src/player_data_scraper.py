#-- Auther: John Wilson --
#-- version 1.0 --
import requests
import pandas as pd
import os

# Base URL for the Fantasy Premier League (FPL) API
api_base_url = "https://fantasy.premierleague.com/api/"

# Function to make a GET request to a specified FPL API endpoint
def fetch_data_from_api(endpoint):
    full_url = api_base_url + endpoint
    response = requests.get(full_url)
    
    # Check if the response is successful (status code 200)
    if response.status_code == 200:
        return response.json()  # Parse and return the JSON data
    else:
        # Log an error message if the request failed
        print(f"Error: Unable to retrieve data from {full_url}. Status code: {response.status_code}")
        return None

# Function to save a DataFrame as a CSV file
def export_dataframe_to_csv(dataframe, file_path):
    dataframe.to_csv(file_path, index=False)  # Save DataFrame to CSV without the index
    print(f"Data successfully saved to {file_path}")
    
# Function to fetch and process player data from the FPL API
def retrieve_player_data(): 
    # Fetch comprehensive FPL data
    fpl_data = fetch_data_from_api("bootstrap-static/")
    if fpl_data:
        # Extract the relevant 'elements' section containing player details
        player_info = fpl_data.get("elements", [])
        # Convert the player data into a DataFrame
        player_df = pd.DataFrame(player_info)
        return player_df
    else:
        print("Error: No player data available.")
        return None
    
# Define the directory path to save the data files
output_directory = r"C:\Users\johns\Programming\Python\Projects\FPL-Predictive-Models\data"

# Create the directory if it doesn't already exist
os.makedirs(output_directory, exist_ok=True)

# Specify the file path for saving player data
player_data_csv_path = os.path.join(output_directory, "FPL_Player_Data_GW4_24-25.csv")


# Retrieve player data and save it to a CSV file
player_data = retrieve_player_data()
if player_data is not None:
    export_dataframe_to_csv(player_data, player_data_csv_path)