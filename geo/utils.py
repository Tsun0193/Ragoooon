import requests
import folium
import json
import os
from geopy.geocoders import Nominatim
from dotenv import load_dotenv

load_dotenv("../.env")
os.chdir("../")

assert os.getenv("ORS_TOKEN") is not None, "OpenRouteService API token must be provided."

def get_current_location():
    """
    Attempts to get the current location based on the user's IP address.
    Falls back to manual input if automatic detection fails.
    """
    try:
        response = requests.get('https://ipinfo.io/json')
        data = response.json()
        loc = data['loc'].split(',')
        latitude = float(loc[0])
        longitude = float(loc[1])
        print(f"Detected current location: {data['city']}, {data['region']}, {data['country']}")
        return (latitude, longitude)
    except Exception as e:
        print("Could not automatically detect location.")
        latitude = float(input("Enter your current latitude: "))
        longitude = float(input("Enter your current longitude: "))
        return (latitude, longitude)
    
def get_destination(destination: str = None, **kwargs):
    """
    Prompts the user to enter the destination address and returns its coordinates.
    """
    assert destination is not None, "Destination address must be provided."

    geolocator = Nominatim(user_agent="route_planner")
    try:
        location = geolocator.geocode(destination)
        print(f"Destination location: {location.address}")
        return (location.latitude, location.longitude)
    except Exception as e:
        print("Could not find the destination address.")
        return None
    
def calculate_route(start_coords, end_coords, api_key = os.getenv("ORS_TOKEN")):
    """
    Uses OpenRouteService API to calculate the route between two coordinates.
    Returns the route geometry and the distance in kilometers.
    """
    url = 'https://api.openrouteservice.org/v2/directions/driving-car/geojson'
    headers = {
        'Authorization': api_key,
        'Content-Type': 'application/json'
    }
    body = {
        "coordinates": [
            [start_coords[1], start_coords[0]],  # [lng, lat]
            [end_coords[1], end_coords[0]]
        ]
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        data = response.json()
        distance = data['features'][0]['properties']['segments'][0]['distance'] / 1000
        geometry = data['features'][0]['geometry']['coordinates']
        route = [(coord[1], coord[0]) for coord in geometry]
        return route, distance
    else:
        print(f"Error fetching route: {response.status_code} - {response.text}")
        return None, None
    
def plot_route(route, start_coords, end_coords, **kwargs):
    """
    Plots the route on an interactive map and saves it as an HTML file.
    """
    # Initialize the map at the starting point
    m = folium.Map(location=start_coords, zoom_start=13)

    # Add the route as a PolyLine
    folium.PolyLine(route, color="blue", weight=5, opacity=0.7).add_to(m)

    # Add markers for start and end
    folium.Marker(location=start_coords, popup="Start", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(location=end_coords, popup="Destination", icon=folium.Icon(color='red')).add_to(m)

    # Save the map as an HTML file if specified
    filename = kwargs.get("filename", "route_map.html")
    path = kwargs.get("path", "assets/routes/")
    if not os.path.exists(path):
        os.makedirs(path)

    m.save(f"{path}{filename}")
    print(f"Route map saved to {path}{filename}")

    return m

if __name__ == "__main__":
    pass