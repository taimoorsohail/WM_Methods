import xesmf
import xarray as xr
import numpy as np
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def remap_3D(T,S, partitions, depth = 1, zonal_int = False, depth_int = False, **kwargs):
    '''
    A function to map watermass locations back into 1x1 degree geographical space.
    Inputs:
    T: a 4-dimensional array in the format (time, latitude, longitude, depth)
    S: a 4-dimensional array in the format (time, latitude, longitude, depth)
    partitions: An array with the T and S limits for a given watermass bin, in the format
    (time, bin#, 4) - where the 4 indices correspond to -
    0: S_min
    1: S_max
    2: T_min
    3: T_max
    OPTIONAL:
    zonal_int: Boolean flag, if True the zonally integrated mask is produced (False by default)
    depth_int: Boolean flag, if True the depth-integrated mask is produced (False by default) 
    dims: A list of strings representing the dimension names in the order 
    ['time', 'latitude', 'longitude', 'depth']

    Outputs: 
    A 3D or 2D time-integrated set of geographical locations where the water mass is present.
    Note that we regrid the horizontal grid to a 1x1 grid here for ease of plotting and data management (after masking).
    '''

    ## Load dimension names if specified
    dimensions = np.array(list(kwargs.values())).flatten()
    ## If no names were specified, throw up a warning letting the user know the default names will be used. 
    if len(dimensions)==0:
        print('WARNING: No list of dimensions provided, assuming dimensions are named [time, latitude, longitude, depth]')
        dimensions = ['time', 'latitude', 'longitude', 'depth']
    
    ## Initialise an empty DataArray with dimensions ['tree_depth', 'time', 'latitude', 'longitude', 'depth']. 
    da_fuzz = xr.zeros_like(S).expand_dims({'tree_depth':2**depth}).assign_coords({'tree_depth':np.arange(0,2**depth)})
    ## Create the xarray regridder which will regrid the end-product into 1x1 grid for ease of visualisation and analysis
    ds_out = xesmf.util.grid_global(1, 1)
    ds_out = ds_out.drop({'lon_b', 'lat_b'})
    ds_out = ds_out.assign_coords({'lat': ds_out.lat})
    ds_out = ds_out.assign_coords({'lon': ds_out.lon})
    ## Ensure the xesmf regridding function is calling the same horizontal dimension names as the original data
    ds_out = ds_out.rename({'lon': str(dimensions[2]), 'lat': str(dimensions[1])})
    ## Looping through time and tree depth, create a mask which is 1 in all Eulerian grid cells that satisfy the WM grid boundaries
    for i in tqdm(range(T.shape[0])):
        for j in (range(partitions.shape[1])):
            da_fuzz[j,i] = xr.where((S.isel(time=i)>partitions[i,j,0])&\
                                    (T.isel(time=i)>partitions[i,j,2])&\
                                    (S.isel(time=i)<=partitions[i,j,1])&\
                                    (T.isel(time=i)<=partitions[i,j,3]),\
                                     1, 0)
            ## Regrid output to a 1x1 grid using bilinear interpolation
            regridder_da_fuzz = xesmf.Regridder(da_fuzz, ds_out, 'bilinear', periodic=True)
            da_fuzz_regridded = regridder_da_fuzz(da_fuzz)
    
    ## Depending on the flags, output a 3D mask file, or a zonally or depth-integrated mask file, for all bins. 

    if not (zonal_int or depth_int):
        return da_fuzz_regridded.sum(str(dimensions[0]))/da_fuzz_regridded.sum(str(dimensions[0])).sum('tree_depth')
    if zonal_int and not depth_int:
        return (da_fuzz_regridded.sum('x').sum(str(dimensions[0])))/(da_fuzz_regridded.sum('x').sum(str(dimensions[0]))).sum('tree_depth')
    if (zonal_int & depth_int):
        return (da_fuzz_regridded.sum('x').sum(str(dimensions[0])))/(da_fuzz_regridded.sum('x').sum(str(dimensions[0]))).sum('tree_depth'), \
            (da_fuzz_regridded.sum(str(dimensions[-1])).sum(str(dimensions[0])))/(da_fuzz_regridded.sum(str(dimensions[-1])).sum(str(dimensions[0]))).sum('tree_depth')
    if depth_int and not zonal_int:
        return (da_fuzz_regridded.sum(str(dimensions[-1])).sum(str(dimensions[0])))/(da_fuzz_regridded.sum(str(dimensions[-1])).sum(str(dimensions[0]))).sum('tree_depth')
