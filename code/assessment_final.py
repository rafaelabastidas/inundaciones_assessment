import os
import glob
import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import geopandas as gpd


from utils import norm

out_dir='outputs'


###################### Descargar los archivos a local

output_dir = "data_raw/mex_20_24"

# Cargar todos los CSV exportados
files = glob.glob(os.path.join(output_dir, "mexico_floods_estado_month_*.csv"))
df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


###################### Descargar los archivos a local

output_dir = "data_raw/afectacion"

# Cargar todos los CSV exportados
files = glob.glob(os.path.join(output_dir, "mexico_poblacion_afectada_estado_mes_*.csv"))
df_a = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

###################### Unificar bases (afectación floods por área y personas afectadas)

df = pd.merge(
    df,
    df_a[['year','month','estado_code', 'poblacion_afectada']],
    on=['year','month','estado_code'],
    how='outer',
    suffixes=('', '_dup')
)

###################################################
################## MORTALIDAD INFANTIL
##################################################

# Cargar base de INEGUI de defunciones registradas (Defunciones registradas) Anual
path="data_raw/Indicadores20250930124251_menores.csv"
mort = pd.read_csv(path, encoding="latin-1")
mort= mort[mort["Área geográfica"]!='00 Estados Unidos Mexicanos']

year_cols = [col for col in mort.columns if col.isdigit()]

# Pasar a formato largo
long = mort.melt(id_vars=["estado_name"], value_vars=year_cols,
               var_name="year", value_name="mortality")

long["year"] = long["year"].astype(int)
long["mortality"] = pd.to_numeric(long["mortality"], errors="coerce")

###################### Unificar bases

annual_exp = (df
              .groupby(['estado_name','year'], as_index=False)
              .agg(
                  #flood_area_km2_sum=('area_inundada_km2','sum'),
                  area_total_km2_max=('area_total_km2','max'),
                  flood_area_km2_max=('area_inundada_km2','max'),   # valor máximo mensual (evento pico)
                  flood_area_km2_sum=('area_inundada_km2', 'sum'),
                  flood_months_any=('area_inundada_km2', lambda x: (x>0).sum()), # meses con inundación
                  flood_afect_sum=('poblacion_afectada','sum')
              ))


merged = long.merge(annual_exp, on=["estado_name", "year"], how="inner", suffixes=("_long", "_df"))

###################################################
################## POBLACIÓN
##################################################

path="data_raw/Poblacion.xlsx"
pob = pd.read_excel(path, skiprows=5)

# Reshape wide a long
pob = pob.melt(
    id_vars=["estado_name"],
    var_name="year",
    value_name="population"
)

# Asegurar tipo numérico en year
pob["year"] = pob["year"].astype(int)

# Crear la "census_year"
merged['census_year'] = np.where(merged['year'] <= 2015, 2010, 2020)

# Preparar df_long para unir por census_year
pob = pob.rename(columns={'year': 'census_year', 'population': 'population_census'})

# Hacer el merge
merged = merged.merge(
    pob[['estado_name', 'census_year', 'population_census']],
    on=['estado_name', 'census_year'],
    how='left'
)

# Dejar columna final como 'population' y limpiar
merged = merged.rename(columns={'population_census': 'population'}).drop(columns=['census_year'])

###################################################
################## MODELO ECONOMÉTRICO
##################################################

# Mantener información hasta 2018
merged=merged[merged['year']<=2018]

#Tasa de mortalidad
merged['death_rate'] = (merged['mortality'] / merged['population']) * 100000

# Flood intensity
merged['flood_intensity_area'] = (
    merged['flood_area_km2_max'] / merged['area_total_km2_max']
) * 100   # % del territorio inundado

# Flood dummy adjusted
merged['flood_dummy'] = (merged['flood_area_km2_max'] > 0).astype(int)

merged['flood_dummy_corr'] = np.where(
    merged['flood_intensity_area'] > 0.5,
    merged['flood_dummy'],
    0
)

# Preparación: orden y variables auxiliares
merged = merged.sort_values(["estado_name", "year"]).copy()

# Diferencias (variación absoluta)
merged["death_rate_diff"] = merged.groupby("estado_name")["death_rate"].diff()
merged["flood_dummy_diff"] = merged.groupby("estado_name")["flood_dummy_corr"].diff()

# Crecimiento (variación %)
merged["death_rate_growth"] = (
    merged.groupby("estado_name")["death_rate"]
          .pct_change()
          .replace([np.inf, -np.inf], np.nan)
)

###################### MODELO 1: NIVELES: FE con clusters alineados
form_levels = "death_rate ~ flood_dummy_corr + C(estado_name) + C(year)"
cols_levels = ["death_rate", "flood_dummy_corr", "estado_name", "year"]
df_levels = merged[cols_levels].dropna().copy()
df_levels["cluster_id"] = df_levels["estado_name"].astype("category").cat.codes

m_levels = smf.ols(form_levels, data=df_levels).fit(
    cov_type="cluster",
    cov_kwds={"groups": df_levels["cluster_id"]}
)
print("\n=== FE OLS (niveles) ===")
print(m_levels.summary())

###################### MODELO 2: DIFERENCIAS
form_diff = "death_rate_diff ~ flood_dummy_diff + C(year)"
cols_diff = ["death_rate_diff", "flood_dummy_diff", "estado_name", "year"]
df_diff = merged[cols_diff].dropna().copy()
df_diff["cluster_id"] = df_diff["estado_name"].astype("category").cat.codes

m_diff = smf.ols(form_diff, data=df_diff).fit(
    cov_type="cluster",
    cov_kwds={"groups": df_diff["cluster_id"]}
)
print("\n=== FE OLS (primeras diferencias) ===")
print(m_diff.summary())

###################### MODELO 3: CRECIMIENTO
form_growth = "death_rate_growth ~ flood_dummy_corr + C(estado_name) + C(year)"
cols_growth = ["death_rate_growth", "flood_dummy_corr", "estado_name", "year"]
df_growth = merged[cols_growth].dropna().copy()
df_growth["cluster_id"] = df_growth["estado_name"].astype("category").cat.codes

m_growth = smf.ols(form_growth, data=df_growth).fit(
    cov_type="cluster",
    cov_kwds={"groups": df_growth["cluster_id"]}
)
print("\n=== FE OLS (crecimiento %) ===")
print(m_growth.summary())


###################################################
################## GRÁFICOS
##################################################

###################### Mapa por área afectada según intensidad

# Columna con nombre del estado y con la intensidad
col_estado = "estado_name"
col_intensity = "flood_intensity_area"


# Promedio por estado
df_mean = (merged
           .dropna(subset=[col_estado, col_intensity])
           .groupby(col_estado, as_index=False)[col_intensity]
           .mean()
           .rename(columns={col_intensity: "flood_intensity_mean"}))

df_mean['key'] = df_mean[col_estado].map(norm)


gdf_states = gpd.read_file("data_raw/gdf_states.geojson")


gdf_plot = gdf_states.merge(df_mean[['key','flood_intensity_mean']], on='key', how='left')

# Mapa estático
fig, ax = plt.subplots(figsize=(9, 7))
gdf_plot.plot(
    column='flood_intensity_mean',
    legend=True,
    linewidth=0.8,
    edgecolor='#333333',
    cmap='Blues',
    ax=ax
)
ax.set_title("México: Intensidad media de inundación por estado (promedio anual 2010-18)")
ax.axis('off')
plt.tight_layout()

plt.savefig(os.path.join(out_dir, "mex_flood_intensity_mean_por_estado.png"), dpi=300)
plt.show()


###################### Gráfico de barras de población afectada

def plot_total_affected_by_state(
    merged: pd.DataFrame,
    start_year: int = 2010,
    end_year: int = 2018,
    out_path: str | None = None,
    annotate: bool = True
):
    """
    Grafica barras (horizontal) con el TOTAL del período de población afectada por inundaciones
    para cada estado, usando 'flood_afect_sum'

    Parámetros:
      - start_year, end_year
      - out_path
      - annotate
    """
    df = merged.copy()

    # Chequeos
    needed = {"estado_name", "year", "flood_afect_sum"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Faltan columnas necesarias: {needed - set(df.columns)}")

    # Filtro temporal
    dfp = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()
    if dfp.empty:
        raise ValueError(f"No hay datos en el rango {start_year}-{end_year}.")

    # Reemplazar NaNs por 0 en la métrica a sumar
    dfp["flood_afect_sum"] = pd.to_numeric(dfp["flood_afect_sum"], errors="coerce").fillna(0)

    # Total período por estado
    tot = (dfp.groupby("estado_name", as_index=False)["flood_afect_sum"]
              .sum()
              .rename(columns={"flood_afect_sum": "affected_total"}))

    if tot.empty:
        raise ValueError("No se pudo computar el total por estado.")

    tot = tot.sort_values("affected_total", ascending=False)

    # Suma nacional
    nat_total = tot["affected_total"].sum()

    # Plot
    plt.figure(figsize=(10, max(6, 0.35 * len(tot))))
    bars = plt.barh(tot["estado_name"], tot["affected_total"])
    plt.gca().invert_yaxis()

    # Línea de referencia (mediana)
    median_val = float(tot["affected_total"].median())
    plt.axvline(median_val, linestyle="--", linewidth=1.2,
                label=f"Mediana estados: {median_val:,.0f}")

    plt.title(
        f"Población afectada directamente por inundaciones — Total {start_year}–{end_year}\n"
        f"(suma anual por estado, personas)",
        pad=12
    )
    plt.xlabel("Personas afectadas (total del período)")
    plt.ylabel("Estado")
    plt.legend(loc="lower right", frameon=False)
    plt.grid(False)

    # Anotaciones
    if annotate:
        ax = plt.gca()
        for rect, val in zip(bars, tot["affected_total"]):
            ax.text(
                rect.get_width() * 1.01, rect.get_y() + rect.get_height()/2,
                f"{val:,.0f}",
                va="center", ha="left"
            )

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


out_png = os.path.join(out_dir, "afectados_total_2010_2018.png")
plot_total_affected_by_state(merged, start_year=2010, end_year=2018,
                              out_path=out_png)


###################### Boxplot para resultados

def twoway_demean(s: pd.Series, g1: pd.Series, g2: pd.Series) -> pd.Series:
    """
    Two-way demeaning
    """
    overall = s.mean()
    m1 = s.groupby(g1).transform("mean")
    m2 = s.groupby(g2).transform("mean")
    return s - m1 - m2 + overall

def within_plot_growth_by_treatment(df_growth: pd.DataFrame, out_path: str | None = None):

    req = {"death_rate_growth", "flood_dummy_corr", "estado_name", "year"}
    if not req.issubset(df_growth.columns):
        raise ValueError(f"Faltan columnas: {req - set(df_growth.columns)}")

    d = df_growth.copy()
    # Residualizar outcome con FE (two-way demeaning)
    d["y_within"] = twoway_demean(d["death_rate_growth"], d["estado_name"], d["year"])

    # Split por tratamiento
    y0 = d.loc[d["flood_dummy_corr"] == 0, "y_within"].dropna()
    y1 = d.loc[d["flood_dummy_corr"] == 1, "y_within"].dropna()

    # Boxplot lado a lado
    plt.figure(figsize=(6.5, 4.5))
    bp = plt.boxplot([y0, y1], labels=["Sin inundación", "Con inundación"], vert=True, patch_artist=False)
    # Medias anotadas
    m0, m1 = y0.mean(), y1.mean()
    for i, m in enumerate([m0, m1], start=1):
        plt.text(i, m, f"μ={m:.3f}", ha="center", va="bottom")

    plt.title("Incremento de mortalidad anual infantil — residualizado por FE (estado & año)")
    plt.ylabel("Variación de mortalidad infantil (residualizado)")
    plt.grid(False)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


out_png = os.path.join(out_dir, "within_growth_boxplot.png")
within_plot_growth_by_treatment(df_growth, out_png)
