import cdsapi
c = cdsapi.Client()

years = [str(y) for y in range(2015, 2025)]
months = [f"{m:02d}" for m in range(1, 13)]
days = [f"{d:02d}" for d in range(1, 32)]
hours = [f"{h:02d}:00" for h in range(24)]

for y in years:
    for m in months:
        print(f"Downloading {y}-{m}")
        c.retrieve(
            'reanalysis-era5-land',
            {
                'variable': [
                    '2m_temperature',
                    '2m_dewpoint_temperature',
                    '10m_u_component_of_wind',
                    '10m_v_component_of_wind',
                    'total_precipitation',
                ],
                'year': y,
                'month': m,
                'day': days,
                'time': hours,
                'area': [13.8,107.4,12.1,109.5],  # Daklak bbox
                'format': 'netcdf',
            },
            f'era5_{y}_{m}.nc')
