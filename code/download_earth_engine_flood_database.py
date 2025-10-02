

print("Starting script...")


import ee
import re
import os
import pandas as pd
import glob
import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


###################################################
################## HAZARD
##################################################

ee.Initialize(project='earth-engine-rafaela')

# Cargar Global Floods
gfd = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")

# México: GAUL nivel 1 (ESTADOS)
gaul_l1 = (ee.FeatureCollection("FAO/GAUL/2015/level1")
           .filter(ee.Filter.eq('ADM0_NAME', 'Mexico')))

# Área total del estado (km²) como propiedad persistente
gaul_l1 = gaul_l1.map(lambda f: f.set({
    'area_total_km2': ee.Number(f.geometry().area(maxError=1)).divide(1e6)
}))

# Agua permanente (JRC GSW) para excluir (>=10 meses/año con agua)
gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
perm_water_mask = gsw.select('seasonality').gte(10)

def monthly_flood_image(year, month, region_geom):
    """Imagen binaria de inundación (>0) para (year, month) con fallback a 0 y sin agua permanente."""
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, 'month')

    coll = (gfd
            .filterBounds(region_geom)
            .filterDate(start, end)
            .select('flooded'))

    flooded_sum = ee.Image(ee.Algorithms.If(
        coll.size().gt(0),
        coll.sum().rename('flooded'),
        ee.Image(0).rename('flooded')
    )).clip(region_geom)

    return flooded_sum.gt(0).updateMask(perm_water_mask.Not())

def floods_by_estado_month(year):
    """FeatureCollection con área_inundada_km2 y %_inundada por ESTADO (level1) para los 12 meses del año."""
    months = ee.List.sequence(1, 12)

    def per_month(m):
        m = ee.Number(m)
        flooded_bin = monthly_flood_image(year, m, gaul_l1.geometry())

        # Área inundada en km2 para el mes m
        area_inundada_km2_img = flooded_bin.multiply(ee.Image.pixelArea()).divide(1e6).rename('area_inundada_km2')

        # Reducir por estados
        stats = area_inundada_km2_img.reduceRegions(
            collection=gaul_l1,
            reducer=ee.Reducer.sum(),
            scale=500
        ).map(lambda f: f.set({
            'year': year,
            'month': m,
            'estado_name': f.get('ADM1_NAME'),
            'estado_code': f.get('ADM1_CODE'),
            'area_inundada_km2': ee.Number(f.get('sum')),
            'pct_inundada': ee.Algorithms.If(
                ee.Number(f.get('area_total_km2')).gt(0),
                ee.Number(f.get('sum')).divide(ee.Number(f.get('area_total_km2'))).multiply(100),
                None
            )
        }))

        return stats

    fc = ee.FeatureCollection(months.map(per_month)).flatten()
    fc = fc.map(lambda f: f.setGeometry(None))
    return fc.select([
        'year','month',
        'estado_name','estado_code',
        'area_total_km2','area_inundada_km2','pct_inundada'
    ])

# Exportar por año a Google Drive
years = list(range(2010, 2025))
for y in years:
    fc_y = floods_by_estado_month(y)
    task = ee.batch.Export.table.toDrive(
        collection=fc_y,
        description=f'mexico_floods_estado_month_{y}',
        folder='GEE_Floods',
        fileNamePrefix=f'mexico_floods_estado_month_{y}',
        fileFormat='CSV'
    )
    task.start()
    print(f'Export iniciado para {y}: {task.id}')
