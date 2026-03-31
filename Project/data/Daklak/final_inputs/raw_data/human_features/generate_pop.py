import requests, os
from pathlib import Path

Path("worldpop").mkdir(parents=True, exist_ok=True)

# Vietnam 1km population density — các năm có sẵn: 2000–2020
for year in [2018, 2019, 2020]:
    url = (
        "https://data.worldpop.org/GIS/Population_Density/"
        f"Global_2000_2020_1km/{year}/VNM/"
        f"vnm_pd_{year}_1km.tif"
    )
    out = f"worldpop/vnm_pd_{year}_1km.tif"

    if os.path.exists(out):
        print(f"Already exists: {out}")
        continue

    print(f"Downloading {year}...")
    r = requests.get(url, stream=True)
    with open(out, "wb") as f:
        for chunk in r.iter_content(1024*512):
            f.write(chunk)
    print(f"Done: {out}")

# File ~80MB mỗi năm