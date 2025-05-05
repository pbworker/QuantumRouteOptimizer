"""
Visualization utilities for the Vehicle Routing Problem
"""

import folium
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from itertools import cycle

def plot_route_on_map(coordinates, routes, location_names=None):
    """
    Plots the optimized routes on a folium map.
    
    Args:
        coordinates: List of (lat, lon) tuples for each location
        routes: List of routes, where each route is a list of location indices
        location_names: Optional list of location names
    
    Returns:
        folium.Map: The map with plotted routes
    """
    # Create a map centered on the mean of coordinates
    center_lat = sum(c[0] for c in coordinates) / len(coordinates)
    center_lon = sum(c[1] for c in coordinates) / len(coordinates)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # Define colors for the routes
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple']
    
    # Add markers for each location
    for i, (lat, lon) in enumerate(coordinates):
        popup_text = f"{location_names[i] if location_names else f'Location {i}'}"
        icon_color = 'red' if i == 0 else 'blue'  # Depot in red
        
        folium.Marker(
            [lat, lon],
            popup=popup_text,
            tooltip=popup_text,
            icon=folium.Icon(color=icon_color, icon="info-sign" if i == 0 else None)
        ).add_to(m)
    
    # Add polylines for each route
    for i, route in enumerate(routes):
        color = colors[i % len(colors)]
        route_points = []
        
        for location_idx in route:
            route_points.append(coordinates[location_idx])
        
        if route_points:
            folium.PolyLine(
                route_points,
                color=color,
                weight=2,
                opacity=0.7,
                tooltip=f"Vehicle {i+1}"
            ).add_to(m)
    
    # Fit the map to the bounds of all coordinates
    if coordinates:
        sw = min(c[0] for c in coordinates), min(c[1] for c in coordinates)
        ne = max(c[0] for c in coordinates), max(c[1] for c in coordinates)
        m.fit_bounds([sw, ne])
    
    return m

def plot_performance_comparison(results, execution_times):
    """
    Creates a performance comparison plot between different VRP solvers.
    
    Args:
        results: Dictionary mapping method names to their results
        execution_times: Dictionary mapping method names to execution times
    
    Returns:
        fig: The matplotlib figure object
    """
    methods = list(results.keys())
    objective_values = [results[method]["objective_value"] for method in methods]
    times = [execution_times[method] for method in methods]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot objective values
    bars1 = ax1.bar(methods, objective_values, color='skyblue')
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Total Distance')
    ax1.set_title('Objective Value Comparison')
    ax1.set_ylim(bottom=0)
    
    # Add values on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom')
    
    # Plot execution times
    bars2 = ax2.bar(methods, times, color='salmon')
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Execution Time Comparison')
    ax2.set_yscale('log')  # Log scale for better visibility
    
    # Add values on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                 f'{height:.4f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def generate_route_visualizations(data_model, routes, location_names=None):
    """
    Generates a network visualization of the routes.
    
    Args:
        data_model: The VRP data model
        routes: List of routes, where each route is a list of location indices
        location_names: Optional list of location names
    
    Returns:
        fig: The matplotlib figure object
    """
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(len(data_model['distance_matrix'])):
        name = location_names[i] if location_names else f"Location {i}"
        is_depot = (i == data_model['depot'])
        G.add_node(i, name=name, is_depot=is_depot)
    
    # Define colors for the routes
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    color_cycle = cycle(colors)
    edge_colors = []
    edges = []
    
    # Add edges for each route
    for route in routes:
        route_color = next(color_cycle)
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i+1]
            G.add_edge(from_node, to_node)
            edges.append((from_node, to_node))
            edge_colors.append(route_color)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define node positions
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    node_colors = ['red' if G.nodes[n]['is_depot'] else 'skyblue' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, 
                          width=2, arrows=True, arrowsize=20, ax=ax)
    
    # Draw labels
    labels = {n: G.nodes[n]['name'] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold', ax=ax)
    
    plt.axis('off')
    plt.title('Vehicle Routing Network Visualization')
    
    return fig

def create_vrp_network_graph(data_model, solution=None, location_names=None):
    """
    Creates a Plotly network graph for the VRP problem and solution.
    
    Args:
        data_model: The VRP data model
        solution: Optional solution as list of routes
        location_names: Optional list of location names
    
    Returns:
        fig: Plotly Figure object
    """
    # Initialize graph
    G = nx.Graph()
    
    # Add nodes
    num_locations = len(data_model['distance_matrix'])
    depot = data_model['depot']
    
    for i in range(num_locations):
        name = location_names[i] if location_names else f"Location {i}"
        G.add_node(i, name=name, is_depot=(i == depot))
    
    # Add all edges with distances
    for i in range(num_locations):
        for j in range(i+1, num_locations):
            G.add_edge(i, j, weight=data_model['distance_matrix'][i][j])
    
    # Get node positions
    pos = nx.spring_layout(G, seed=42)
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(G.nodes[node]['name'])
        node_colors.append('red' if G.nodes[node]['is_depot'] else 'blue')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        hoverinfo='text',
        marker=dict(
            showscale=False,
            size=15,
            color=node_colors,
            line=dict(width=2, color='black')
        ),
        textposition='top center'
    )
    
    # Create edge traces (all edges in light gray)
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#cccccc'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create solution edge traces (if solution provided)
    solution_traces = []
    
    if solution:
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        for i, route in enumerate(solution):
            route_x = []
            route_y = []
            route_text = []
            
            for j in range(len(route) - 1):
                from_node = route[j]
                to_node = route[j+1]
                
                x0, y0 = pos[from_node]
                x1, y1 = pos[to_node]
                
                route_x.extend([x0, x1, None])
                route_y.extend([y0, y1, None])
                
                dist = G[from_node][to_node]['weight']
                route_text.append(f"Distance: {dist:.2f}")
            
            color = colors[i % len(colors)]
            
            solution_trace = go.Scatter(
                x=route_x, y=route_y,
                line=dict(width=2, color=color),
                hoverinfo='text',
                name=f"Vehicle {i+1}",
                text=route_text,
                mode='lines'
            )
            
            solution_traces.append(solution_trace)
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace] + solution_traces)
    
    # Update layout
    fig.update_layout(
        title='Vehicle Routing Problem Network',
        titlefont_size=16,
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(x=1.05, y=1, traceorder='normal')
    )
    
    return fig
