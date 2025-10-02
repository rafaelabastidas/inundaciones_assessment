

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
################## EXPOSICION
##################################################

# Población usando WorldPop 100m

def population_image(year, region_geom):
    return (ee.ImageCollection("WorldPop/GP/100m/pop")
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .mosaic()
            .unmask(0)
            .clip(region_geom))

def poblacion_afectada_por_estado_mes(year, scale=250):
    """
    Devuelve FeatureCollection con:
      - year, month
      - estado_name, estado_code
      - poblacion_afectada
    """
    months = ee.List.sequence(1, 12)
    region_geom = gaul_l1.geometry()
    pop_img = population_image(year, region_geom).rename('pop')

    def per_month(m):
        m = ee.Number(m)
        flooded_bin = monthly_flood_image(year, m, region_geom)

        # Población afectada: población en pixeles inundados
        pop_afectada_img = pop_img.updateMask(flooded_bin)

        # Suma por estado
        stats = pop_afectada_img.reduceRegions(
            collection=gaul_l1,
            reducer=ee.Reducer.sum(),
            scale=scale
        ).map(lambda f: f.set({
            'year': year,
            'month': m,
            'estado_name': f.get('ADM1_NAME'),
            'estado_code': f.get('ADM1_CODE'),
            'poblacion_afectada': ee.Algorithms.If(
                f.get('sum'), ee.Number(f.get('sum')), ee.Number(0)
            )
        }))
        return stats

    fc = ee.FeatureCollection(months.map(per_month)).flatten()
    fc = fc.map(lambda f: f.setGeometry(None))
    return fc.select(['year','month','estado_name','estado_code','poblacion_afectada'])

# Exportar por año a Google Drive

years = list(range(2010, 2025))
for y in years:
    fc_y = poblacion_afectada_por_estado_mes(y)
    task = ee.batch.Export.table.toDrive(
        collection=fc_y,
        description=f'mexico_poblacion_afectada_estado_mes_{y}',
        folder='GEE_Floods',
        fileNamePrefix=f'mexico_poblacion_afectada_estado_mes_{y}',
        fileFormat='CSV'
    )
    task.start()
    print(f'Export iniciado para {y}: {task.id}')
