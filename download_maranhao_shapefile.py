"""
Download Maranhão state boundary shapefile from IBGE.
"""

import os
import zipfile
import urllib.request
import geopandas as gpd


def download_maranhao_shapefile(output_dir):
    """
    Download and extract Maranhão state boundary from IBGE.
    Returns path to the shapefile.
    """
    os.makedirs(output_dir, exist_ok=True)

    shapefile_path = os.path.join(output_dir, 'maranhao.shp')
    if os.path.exists(shapefile_path):
        print(f"Shapefile already exists: {shapefile_path}")
        return shapefile_path

    # IBGE provides state boundaries
    # Using simplified boundaries for efficiency
    url = "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/UFs/MA/MA_UF_2022.zip"

    zip_path = os.path.join(output_dir, 'ma_uf.zip')

    print("Downloading Maranhão shapefile from IBGE...")
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(output_dir)

    # Find the extracted shapefile
    for f in os.listdir(output_dir):
        if f.endswith('.shp') and 'UF' in f:
            extracted_shp = os.path.join(output_dir, f)
            # Rename to standard name
            gdf = gpd.read_file(extracted_shp)
            gdf.to_file(shapefile_path)
            print(f"Saved to {shapefile_path}")
            break

    # Cleanup
    os.remove(zip_path)

    return shapefile_path


def create_maranhao_mask(shapefile_path, lats, lons, resolution=0.25):
    """
    Create a boolean mask array for Maranhão from the shapefile.
    """
    import numpy as np
    from shapely.geometry import Point

    gdf = gpd.read_file(shapefile_path)
    geometry = gdf.unary_union

    lat_grid = np.arange(lats[0], lats[1], resolution)
    lon_grid = np.arange(lons[0], lons[1], resolution)

    mask = np.zeros((len(lat_grid), len(lon_grid)), dtype=bool)

    for i, lat in enumerate(lat_grid):
        for j, lon in enumerate(lon_grid):
            point = Point(lon, lat)
            if geometry.contains(point):
                mask[i, j] = True

    return mask, lat_grid, lon_grid


if __name__ == "__main__":
    output_dir = os.path.join(os.getenv('TELNET_DATADIR', 'data'), 'shapefiles')
    download_maranhao_shapefile(output_dir)
