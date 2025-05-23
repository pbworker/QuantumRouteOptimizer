import streamlit as st                        ##used for the web application
import numpy as np                            ##used for the numercial operations( arrays, martices)
import pandas as pd                           ## pd fr the dataframes
import time                                   ##introduce delays, measure execution time
import matplotlib.pyplot as plt               ## plotting the graphs
import folium                                 ## visualize geographic data
from streamlit_folium import folium_static    ## getting the visualize geographic data in the streamlit
import plotly.express as px                   ##Quick plotting
import plotly.graph_objects as go             ## ful control on the plotting
import io                                     ## input and output streams

from utils.classical_vrp import solve_vrp_ortools, solve_vrp_genetic_algorithm ##uses Google OR-Tools
from utils.quantum_vrp import solve_vrp_qaoa ### Solves VRP using Quantum Approximate Optimization Algorithm (QAOA)

###Custom Visualization Helpers
from utils.visualization import (
    plot_route_on_map,
    plot_performance_comparison,
    generate_route_visualizations
)

## Data Handling Utilities
from utils.data_handler import (
    generate_distance_matrix,
    get_random_coordinates,
    create_data_model
)

###Sample Data & Truck Routing Helpers
from data.sample_data import sample_locations
from data.load_truck_data import (
    load_truck_routing_data,
    create_weighted_data_model,
    find_optimal_depot
)

# Set the page configuration
st.set_page_config(
    page_title="Quantum-Based Vehicle Routing Optimization",
    page_icon="🚚",
    layout="wide"
)

# Title and introduction
st.title("Optimized Quantum-Based Vehicle Routing Protocol")
st.markdown("""
This application demonstrates the power of quantum computing applied to the Vehicle Routing Problem (VRP).
Compare classical optimization methods against quantum approaches and visualize the results.
""")

# Sidebar for configuration parameters
st.sidebar.header("Configuration")

# Number of locations selection
num_locations = st.sidebar.slider(
    "Number of Locations",
    min_value=3,
    max_value=30,
    value=5,
    help="Number of locations to visit (including depot)"
)

# Number of vehicles
num_vehicles = st.sidebar.slider(
    "Number of Vehicles",
    min_value=1,
    max_value=5,
    value=2,
    help="Number of vehicles in the fleet"
)
# User input for the depot
index = st.sidebar.number_input(
     "Enter Depot Index",
    min_value=0,max_value=num_locations-1,
     value=0, 
     help="Enter the index of the depot location (0-based)"
)

# Method selection
methods = st.sidebar.multiselect(
    "Solution Methods",
    ["OR-Tools (Classical)", "Genetic Algorithm (Classical)", "QAOA (Quantum)"],
    default=["OR-Tools (Classical)", "QAOA (Quantum)"],
    help="Select which algorithms to use for solving the VRP"
)

# Check if at least one method is selected
if not methods:
    st.warning("Please select at least one solution method.")
    st.stop()

# Random seed for reproducibility
random_seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    max_value=10000,
    value=42,
    help="Seed for random number generation (for reproducibility)"
)

np.random.seed(random_seed)

# Data source selection
data_source = st.sidebar.radio(
    "Select Data Source",
    ["Sample Dataset", "Random Data", "Upload Custom Data"],
    index=0,
    help="Choose where to get location data from"
)

# Main content
st.header("Vehicle Routing Problem Setup")

# Variables for data
coords = None
location_names = None
weights = None

# Handle the different data sources
if data_source == "Sample Dataset":
    st.info("Using sample locations dataset")
    
    # Use sample dataset but limit to selected number of locations
    locations = sample_locations[:num_locations]
    location_names = [loc['name'] for loc in locations]
    coords = [(loc['lat'], loc['lon']) for loc in locations]
    
    df_locations = pd.DataFrame({
        'Location': location_names,
        'Latitude': [c[0] for c in coords],
        'Longitude': [c[1] for c in coords]
    })
    
elif data_source == "Random Data":
    st.info("Using randomly generated locations")
    
    # Generate random coordinates
    coords = get_random_coordinates(num_locations, seed=random_seed)
    location_names = [f"Location {i}" if i > 0 else "Depot" for i in range(num_locations)]
    
    df_locations = pd.DataFrame({
        'Location': location_names,
        'Latitude': [c[0] for c in coords],
        'Longitude': [c[1] for c in coords]
    })
    
else:  # Upload Custom Data
    st.info("Upload your custom truck routing data (CSV format)")
    
    # File uploader for CSV
    uploaded_file = st.file_uploader(
        "Upload truck routing data CSV", 
        type="csv",
        help="CSV file should contain columns: City, Latitude, Longitude, Weight"
    )
    
    if uploaded_file is not None:
        try:
            # Load the data
            file_contents = uploaded_file.read()
            
            # Create a StringIO object to reuse the content
            string_data = io.StringIO(file_contents.decode("utf-8"))
            
            # Preview the uploaded CSV
            df_preview = pd.read_csv(string_data)
            st.write("Preview of uploaded data:")
            st.dataframe(df_preview.head())
            
            # Reset the string pointer to the beginning for reuse
            string_data.seek(0)
            
            # Process the data
            coords, location_names, weights = load_truck_routing_data(
                string_data, 
                max_locations=num_locations
            )
            
            # Create dataframe for display
            df_locations = pd.DataFrame({
                'Location': location_names,
                'Latitude': [c[0] for c in coords],
                'Longitude': [c[1] for c in coords],
                'Weight': weights
            })
            
            # Find optimal depot based on center of mass
            depot_index = find_optimal_depot(coords, weights)
            st.success(f"Optimal depot location calculated: {location_names[depot_index]}")
            
        except Exception as e:
            st.error(f"Error processing the uploaded file: {str(e)}")
            st.stop()
    else:
        st.warning("Please upload a CSV file with location data to continue.")
        st.stop()

# Display the locations table
st.subheader("Locations Data")
st.dataframe(df_locations)

# Calculate distance matrix
distance_matrix = generate_distance_matrix(coords)

# Create data model for the VRP solvers
if coords is None or len(coords) < 2:
    st.error("Not enough location data to create a routing problem. Please select a different data source or upload valid data.")
    st.stop()

# Create data model based on data source
if data_source == "Upload Custom Data" and weights is not None:
    # For custom data with weights, use depot index from center of mass calculation
    depot_idx = find_optimal_depot(coords, weights)
    st.info(f"Using {location_names[depot_idx]} as depot based on weight distribution.")
    data_model = create_weighted_data_model(distance_matrix, num_vehicles, weights, depot=depot_idx)
else:
    # Default data model for sample or random data
    data_model = create_data_model(distance_matrix, num_vehicles)

# Display distance matrix
with st.expander("View Distance Matrix"):
    st.dataframe(pd.DataFrame(
        data_model['distance_matrix'],
        columns=location_names,
        index=location_names
    ))

# Show map with locations
st.subheader("Map View of Locations")
m = folium.Map()

# Check if we have data to display
if coords and location_names:
    # Display markers with appropriate information
    for i, (lat, lon) in enumerate(coords):
        # Determine if this is a depot (either index 0 for sample/random data or the calculated depot index)
        is_depot = False
        if data_source == "Upload Custom Data" and weights is not None:
            # When using custom data with weights, we use center of mass calculation for depot
            depot_index = find_optimal_depot(coords, weights)
            is_depot = (i == depot_index)
        else:
            is_depot = (i == 0)  # Default for sample/random data
            
        # Set marker color
        color = 'red' if is_depot else 'blue'
        
        # Create popup text
        if data_source == "Upload Custom Data" and weights:
            popup_text = f"{location_names[i]}<br>Weight: {weights[i]:.2f}"
        else:
            popup_text = f"{location_names[i]}"
            
        # Add marker
        folium.Marker(
            [lat, lon],
            popup=popup_text,
            tooltip=location_names[i],
            icon=folium.Icon(color=color, icon="info-sign" if is_depot else None)
        ).add_to(m)

    # Fit map to location bounds
    location_data = pd.DataFrame(coords, columns=['Latitude', 'Longitude'])
    sw = location_data.min().values.tolist()
    ne = location_data.max().values.tolist()
    m.fit_bounds([sw, ne])

folium_static(m)

# Add a button to trigger the optimization
if st.button("Solve Vehicle Routing Problem"):
    with st.spinner("Computing optimal routes..."):
        results = {}
        execution_times = {}
        
        # Solve using selected methods
        if "OR-Tools (Classical)" in methods:
            start_time = time.time()
            ortools_routes, ortools_obj_val = solve_vrp_ortools(data_model)
            end_time = time.time()
            results["OR-Tools (Classical)"] = {
                "routes": ortools_routes,
                "objective_value": ortools_obj_val
            }
            execution_times["OR-Tools (Classical)"] = end_time - start_time
            
        if "Genetic Algorithm (Classical)" in methods:
            start_time = time.time()
            ga_routes, ga_obj_val = solve_vrp_genetic_algorithm(data_model)
            end_time = time.time()
            results["Genetic Algorithm (Classical)"] = {
                "routes": ga_routes,
                "objective_value": ga_obj_val
            }
            execution_times["Genetic Algorithm (Classical)"] = end_time - start_time
            
        if "QAOA (Quantum)" in methods:
            start_time = time.time()
            qaoa_routes, qaoa_obj_val = solve_vrp_qaoa(data_model)
            end_time = time.time()
            results["QAOA (Quantum)"] = {
                "routes": qaoa_routes,
                "objective_value": qaoa_obj_val
            }
            execution_times["QAOA (Quantum)"] = end_time - start_time
        
        # Display results
        st.header("Optimization Results")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Route Maps", "Performance Comparison", "Detailed Results"])
        
        with tab1:
            st.subheader("Optimized Routes")
            
            for method_name, result in results.items():
                st.markdown(f"### {method_name}")
                route_map = plot_route_on_map(coords, result["routes"], location_names)
                folium_static(route_map)
                
                # Generate route description
                for i, route in enumerate(result["routes"]):
                    if route:
                        # Create route description with weights if available
                        if data_source == "Upload Custom Data" and weights:
                            locations_with_weights = []
                            total_weight = 0
                            for loc in route:
                                if loc < len(location_names) and loc < len(weights):
                                    locations_with_weights.append(f"{location_names[loc]} ({weights[loc]:.2f})")
                                    total_weight += weights[loc]
                                else:
                                    locations_with_weights.append(f"{location_names[loc]}")
                            
                            route_desc = " → ".join(locations_with_weights)
                            st.markdown(f"**Vehicle {i+1}**: {route_desc} (Total Weight: {total_weight:.2f})")
                        else:
                            route_desc = " → ".join([location_names[loc] for loc in route])
                            st.markdown(f"**Vehicle {i+1}**: {route_desc}")
        
        with tab2:
            st.subheader("Performance Comparison")
            
            # Create comparison metrics
            comparison_data = {
                "Method": [],
                "Total Distance": [],
                "Execution Time (seconds)": []
            }
            
            for method_name, result in results.items():
                comparison_data["Method"].append(method_name)
                comparison_data["Total Distance"].append(result["objective_value"])
                comparison_data["Execution Time (seconds)"].append(execution_times[method_name])
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df)
            
            # Plot comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.bar(
                    comparison_df,
                    x="Method",
                    y="Total Distance",
                    title="Total Distance Comparison",
                    color="Method"
                )
                st.plotly_chart(fig1)
                
            with col2:
                fig2 = px.bar(
                    comparison_df,
                    x="Method",
                    y="Execution Time (seconds)",
                    title="Execution Time Comparison",
                    color="Method",
                    log_y=True
                )
                st.plotly_chart(fig2)
                
            # If quantum method was used, show potential quantum advantage
            if "QAOA (Quantum)" in methods:
                st.subheader("Quantum vs Classical Analysis")
                
                classical_methods = [m for m in methods if "Classical" in m]
                if classical_methods:
                    # Filter for classical methods only
                    classical_df = comparison_df[comparison_df["Method"].isin(classical_methods)]
                    quantum_df = comparison_df[comparison_df["Method"] == "QAOA (Quantum)"]
                    
                    best_classical_distance = classical_df["Total Distance"].min()
                    quantum_distance = quantum_df["Total Distance"].iloc[0]
                    
                    st.markdown(f"""
                    ### Quantum Advantage Analysis
                    
                    - Best classical solution distance: **{best_classical_distance:.2f}**
                    - Quantum solution distance: **{quantum_distance:.2f}**
                    - Difference: **{(best_classical_distance - quantum_distance):.2f}** ({(1 - quantum_distance/best_classical_distance)*100:.1f}% improvement)
                    
                    Note: For small problem instances, classical methods often perform well. Quantum advantage typically becomes 
                    more significant for larger, more complex problems where classical methods face exponential scaling challenges.
                    """)
        
        with tab3:
            st.subheader("Detailed Results")
            
            for method_name, result in results.items():
                with st.expander(f"{method_name} Details"):
                    st.markdown(f"**Objective Value (Total Distance)**: {result['objective_value']:.2f}")
                    st.markdown(f"**Execution Time**: {execution_times[method_name]:.4f} seconds")
                    
                    # Show routes
                    st.markdown("**Routes:**")
                    for i, route in enumerate(result["routes"]):
                        if route:
                            if data_source == "Upload Custom Data" and weights:
                                # Calculate total weight for the route
                                route_weight = sum(weights[loc] for loc in route if loc < len(weights))
                                st.markdown(f"Vehicle {i+1}: {route} (Total Weight: {route_weight:.2f})")
                            else:
                                st.markdown(f"Vehicle {i+1}: {route}")

# Add information section about the algorithms
with st.expander("About the Algorithms"):
    st.markdown("""
    ## Vehicle Routing Problem (VRP)
    
    The Vehicle Routing Problem involves finding optimal routes for a fleet of vehicles to deliver goods to a set of customers.
    
    ### Classical Approaches
    
    - **OR-Tools**: Google's operations research tools that use heuristic algorithms to find near-optimal solutions.
    - **Genetic Algorithm**: A metaheuristic inspired by natural selection, evolving a population of solutions.
    
    ### Quantum Approaches
    
    - **Quantum Approximate Optimization Algorithm (QAOA)**: A hybrid quantum-classical algorithm designed for combinatorial optimization problems.
    
    QAOA works by encoding the problem into a cost Hamiltonian and applying a series of quantum operations to find high-quality solutions.
    
    ## Quantum Advantage
    
    Quantum computers have the potential to solve certain optimization problems more efficiently than classical computers.
    For VRP with many constraints and locations, quantum approaches could potentially find better solutions faster.
    
    Current quantum computers are still limited by noise and qubit count (NISQ era), so hybrid approaches are often used.
    """)

# Add footer with resources
st.markdown("---")
st.markdown("""
**Resources:**
* [Qiskit Documentation](https://qiskit.org/documentation/)
* [Google OR-Tools](https://developers.google.com/optimization)
* [Vehicle Routing Problem](https://en.wikipedia.org/wiki/Vehicle_routing_problem)
""")
