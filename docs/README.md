# IADB - Data Scientist Assessment

## Project Information

**File:** Assessment_final.py
**Written by:** Rafaela Bastidas Ripalda
**Created:** October 2025

## Purpose

To generate a script to process data and generate results of floods in infant mortality, at state and year level in Mexico.

## Data Files

- `mexico_floods_estado_month_*.csv`
- `mexico_poblacion_afectada_estado_mes_*.csv`
- `mortalidad_INEGI.csv`
- `Poblacion.xlsx`

## Installation

Install required packages:

```bash
pip3 install -r requirements.txt
```

## Running the main script

This runs the main script that exports the charts

```bash
python code/assessment_final.py
```


## Downloading additional datasets

This step exports Google Earth data to Google Drive, requires a google account and a google earth project.

This can be skipped since I already downloaded the data, and added it in data_raw for convenience. 

To run it, and export data to data_raw:

```bash
earthengine authenticate
python code/download_earth_engine_flood_database.py
python code/download_earth_global_population_data.py
```



## Outputs

Images are exported to /outputs.