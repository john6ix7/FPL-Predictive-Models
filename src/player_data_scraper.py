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
