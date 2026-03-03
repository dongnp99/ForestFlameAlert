import xarray as xr

ds = xr.open_dataset("raw_nc/era5_2015_01.nc", engine="netcdf4")
print(ds.dims)
print(ds.coords)
print(ds.data_vars)

