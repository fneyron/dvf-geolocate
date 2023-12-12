import collections
import gzip
import json
import multiprocessing
import os
import re
from tenacity import retry, stop_after_attempt, wait_fixed

import sys
import time

import numpy as np
import pandas as pd
import requests

MILESIME = "2023-01-01"
# INPUT_FILE = "input/dvf/valeursfoncieres-test.txt"
OUTPUT_FOLDER = "output/dvf_geolocated/"
COLUMNS = {
    "Date mutation": "date_mutation",
    "No disposition": "numero_disposition",
    "Nature mutation": "nature_mutation",
    "Valeur fonciere": "valeur_fonciere",
    "No voie": "adresse_numero",
    "B/T/Q": "adresse_suffixe",
    "Type de voie": "adresse_nom_voie",
    "Code voie": "adresse_code_voie",
    "Code postal": "code_postal",
    "Code commune": "code_commune",
    "Commune": "nom_commune",
    "Code departement": "code_departement",
    "Prefixe de section": "prefixe_section",
    "Section": "section",
    "No plan": "no_plan",
    "No Volume": "no_volume",
    "1er lot": "lot1_numero",
    "Surface Carrez du 1er lot": "lot1_surface_carrez",
    "2eme lot": "lot2_numero",
    "Surface Carrez du 2eme lot": "lot2_surface_carrez",
    "3eme lot": "lot3_numero",
    "Surface Carrez du 3eme lot": "lot3_surface_carrez",
    "4eme lot": "lot4_numero",
    "Surface Carrez du 4eme lot": "lot4_surface_carrez",
    "5eme lot": "lot5_numero",
    "Surface Carrez du 5eme lot": "lot5_surface_carrez",
    "Nombre de lots": "nb_lots",
    "Code type local": "code_type_local",
    "Type local": "type_local",
    "Surface reelle bati": "surface_reelle_bati",
    "Nombre pieces principales": "nb_pieces_principales",
    "Nature culture": "nature_culture",
    "Nature culture speciale": "nature_culture_speciale",
    "Surface terrain": "surface_terrain",
}

COLUMNS_TYPE = collections.defaultdict(lambda: 'str', {
    'Valeur fonciere': 'float64',
    'Surface Carrez du 1er lot': 'float64',
    'Surface Carrez du 2eme lot': 'float64',
    'Surface Carrez du 3eme lot': 'float64',
    'Surface Carrez du 4eme lot': 'float64',
    'Surface Carrez du 5eme lot': 'float64',
    'Surface terrain': 'float64',
    'surface_reelle_bati': 'float64',
    'code_postal': 'str',
    'nb_pieces_principales': 'int64',
    'nb_lots': 'int64'

})


def calculate_centroid(coordinates):
    if not coordinates is np.nan and coordinates:
        lats = np.array([coord[1] for coord in coordinates[0]])
        lons = np.array([coord[0] for coord in coordinates[0]])
        lat = np.mean(lats)
        lon = np.mean(lons)
        return lat, lon
    else:
        return (np.nan, np.nan)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def process_commune(group, code_departement, code_commune):
    url = (
        f"https://cadastre.data.gouv.fr/data/etalab-cadastre/{MILESIME}/geojson/communes/{code_departement}/{code_commune}/cadastre-{code_commune}-parcelles.json.gz")
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            data = json.loads(gzip.decompress(resp.content))
            id_parcelle_to_coordinates = {d["id"]: d["geometry"]["coordinates"] for d in data["features"]}

            # calculate the centroid coordinates and create new latitude and longitude columns
            centroids = pd.DataFrame(
                group["id_parcelle"].map(id_parcelle_to_coordinates).apply(calculate_centroid).tolist(),
                index=group.index, columns=["latitude", "longitude"])

            # return a new dataframe that includes the original columns and the new latitude and longitude columns
            return pd.concat([group, centroids], axis=1)[list(group.columns) + ["latitude", "longitude"]]

    except Exception as err:
        print("Timeout url:%s, error:%s" % (url, err))


if __name__ == "__main__":
    start = time.time()

    if len(sys.argv) != 2:
        print("Usage: python3 {sys.argv[0]} argument")
        sys.exit(1)

    file = sys.argv[1]
    if not os.path.isfile(file):
        print(f"Error: file {file} does not exist.")
        sys.exit(1)

    file_name = os.path.basename(file)
    output_file = re.sub(r"(\.[^.]+)$", ".csv", file_name)
    output_file = f"{os.path.splitext(output_file)[0]}-geolocated.csv"

    # Read CSV File, rename columns and keep only interesting ones
    df = pd.read_csv(file, sep="|", decimal=",", dtype=COLUMNS_TYPE, usecols=COLUMNS.keys())
    df = df.rename(columns=COLUMNS).loc[:, COLUMNS.values()]

    # Fill empty properties
    df["prefixe_section"] = df["prefixe_section"].fillna(0)
    df['surface_terrain'] = df['surface_terrain'].fillna(0)
    df['surface_reelle_bati'] = df['surface_reelle_bati'].fillna(0)
    df['nb_pieces_principales'] = df['nb_pieces_principales'].fillna(0)
    df['nb_lots'] = df['nb_lots'].fillna(0)

    # Calculate Parcelle ID in order to dl the right parcelle file
    df["id_parcelle"] = df.apply(
        lambda x: f"{x['code_departement']:0>2}" + (f"{x['code_commune']:0>2}" if x[
            'code_departement'].startswith(
            "97") else f"{x['code_commune']:0>3}") + f"{x['prefixe_section']:0>3}" + f"{x['section']:0>2}" + f"{x['no_plan']:0>4}",
        axis=1)

    results = []
    with multiprocessing.Pool() as pool:
        for _, commune in df.groupby(["code_departement", "code_commune"]):
            code_departement = f"{commune['code_departement'].iloc[0]:0>2}"
            if code_departement.startswith("97"):
                code_commune = code_departement + f"{commune['code_commune'].iloc[0]:0>2}"
            else:
                code_commune = code_departement + f"{commune['code_commune'].iloc[0]:0>3}"
            result = pool.apply_async(process_commune, args=(commune, code_departement, code_commune))
            results.append(result)
        pool.close()
        pool.join()

    results = [r.get() for r in results if r.get() is not None]
    result_df = pd.concat(results)
    print("Pourcentage de geoloc: %s" % (result_df["latitude"].count() / df["date_mutation"].count() * 100))

    result_df.to_csv(os.path.join(OUTPUT_FOLDER, output_file), index=False)

    print("Execution: %s secondes" %(time.time() - start))
