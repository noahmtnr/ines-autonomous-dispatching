import warnings

import pandas as pd

warnings.filterwarnings("ignore")
import folium

hubs_df = pd.read_csv("data/hubs/new_hubs.CSV")
print(hubs_df)

hubs_np = hubs_df.to_numpy()

# plot hubs:
boulder_coords = location = [40.778, -73.953]
# Create the map
map_hubs = folium.Map(location=boulder_coords, zoom_start=12)

# Add 70 hubs to the map
for i in range(70):
    folium.Marker([hubs_np[i, 0], hubs_np[i, 1]], popup=f"hub {i}").add_to(map_hubs)

# Display the map
import webbrowser

map_hubs.save("cur_hubs_map.html")
webbrowser.open("cur_hubs_map.html")
