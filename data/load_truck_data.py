"""
Load and process truck routing data
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_truck_routing_data(file_path, max_locations=None):
    """
    Load truck routing data from CSV file
    
    Args:
        file_path: Path to the CSV file
        max_locations: Maximum number of locations to return (default: None)
    
    Returns:
        coordinates: List of (lat, lon) tuples
        location_names: List of location names
        weights: List of weights for each location
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Drop duplicates based on City, Latitude, Longitude
    df = df.drop_duplicates(subset=['City', 'Latitude', 'Longitude'])
    
    # Sort by weight (descending)
    df = df.sort_values('Weight', ascending=False)
    
    # Limit the number of locations if specified
    if max_locations:
        df = df.head(max_locations)
    
    # Extract the coordinates and location names
    coordinates = list(zip(df['Latitude'], df['Longitude']))
    location_names = df['City'].tolist()
    weights = df['Weight'].tolist()
    
    return coordinates, location_names, weights

def create_weighted_data_model(distance_matrix, num_vehicles, weights=None, depot=0):
    """
    Creates a data model dictionary for the VRP solvers with weighted locations.
    
    Args:
        distance_matrix: 2D array of distances between locations
        num_vehicles: Number of vehicles available
        weights: Weights for each location (e.g., delivery size)
        depot: Index of the depot location (default: 0)
    
    Returns:
        data_model: Dictionary containing the problem data
    """
    data_model = {}
    data_model['distance_matrix'] = distance_matrix
    data_model['num_vehicles'] = num_vehicles
    data_model['depot'] = depot
    
    if weights:
        data_model['weights'] = weights
    
    return data_model

def calculate_center_of_mass(coordinates, weights):
    """
    Calculate the center of mass of the locations based on weights
    
    Args:
        coordinates: List of (lat, lon) tuples
        weights: List of weights for each location
    
    Returns:
        center: (lat, lon) tuple for the center of mass
    """
    if not weights or len(weights) != len(coordinates):
        # If no weights or incorrect size, use simple average
        center_lat = sum(c[0] for c in coordinates) / len(coordinates)
        center_lon = sum(c[1] for c in coordinates) / len(coordinates)
        return (center_lat, center_lon)
    
    # Calculate weighted center
    total_weight = sum(weights)
    weighted_lat = sum(c[0] * w for c, w in zip(coordinates, weights)) / total_weight
    weighted_lon = sum(c[1] * w for c, w in zip(coordinates, weights)) / total_weight
    
    return (weighted_lat, weighted_lon)

def find_optimal_depot(coordinates, weights=None):
    """
    Find the location closest to the center of mass to use as a depot
    
    Args:
        coordinates: List of (lat, lon) tuples
        weights: List of weights for each location
    
    Returns:
        depot_index: Index of the location to use as a depot
    """
    center = calculate_center_of_mass(coordinates, weights)
    
    # Find the closest location to the center
    min_dist = float('inf')
    depot_index = 0
    
    for i, coord in enumerate(coordinates):
        # Simple Euclidean distance (for quick estimation)
        dist = ((coord[0] - center[0])**2 + (coord[1] - center[1])**2)**0.5
        if dist < min_dist:
            min_dist = dist
            depot_index = i
    
    return depot_index