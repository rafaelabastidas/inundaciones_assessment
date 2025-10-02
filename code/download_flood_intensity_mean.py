import ee
import geemap
import os

from utils import norm


out_dir='outputs'

os.makedirs(out_dir, exist_ok=True)
out_png = os.path.join(out_dir, "mex_flood_intensity_mean_por_estado.png")
out_html = os.path.join(out_dir, "mex_flood_intensity_mean_por_estado.html")

# Fronteras estatales desde GEE (GAUL nivel 1)
ee.Initialize(project='earth-engine-rafaela')



gaul_l1 = (ee.FeatureCollection("FAO/GAUL/2015/level1")
           .filter(ee.Filter.eq('ADM0_NAME', 'Mexico')))

# Convertir a GeoDataFrame
try:
    gdf_states = geemap.ee_to_geopandas(gaul_l1)
except AttributeError:
    gdf_states = geemap.ee_to_gdf(gaul_l1)


gdf_states = gdf_states[['ADM1_NAME', 'geometry']].copy()
gdf_states['ADM1_NAME'] = gdf_states['ADM1_NAME'].astype(str)
gdf_states['key'] = gdf_states['ADM1_NAME'].map(norm)

gdf_states.to_file("data_raw/gdf_states.geojson", driver='GeoJSON')












