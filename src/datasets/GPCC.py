'''
Created on 2013-09-09

This module contains meta data and access functions for the GPCC climatology and time-series. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import netCDF4 as nc # netcdf python module
import os # check if files are present
#import types # to add precip conversion fct. to datasets
#from importlib import import_module
# internal imports
from geodata.base import Variable, Axis
from geodata.netcdf import DatasetNetCDF
from geodata.gdal import addGDALtoDataset, GridDefinition, loadPickledGridDef, addGeoLocator
from geodata.misc import DatasetError
from utils.nctools import writeNetCDF
from datasets.common import getRootFolder, grid_folder, transformPrecip, timeSlice
from datasets.common import translateVarNames, loadObservations, addLandMask, addLengthAndNamesOfMonth, getFileName
from processing.process import CentralProcessingUnit

## GPCC Meta-data

dataset_name = 'GPCC'
root_folder = getRootFolder(dataset_name=dataset_name) # get dataset root folder based on environment variables

# GPCC grid definition           
geotransform_025 = (-180.0, 0.25, 0.0, -90.0, 0.0, 0.25)
size_025 = (1440,720) # (x,y) map size
geotransform_05 = (-180.0, 0.5, 0.0, -90.0, 0.0, 0.5)
size_05 = (720,360) # (x,y) map size
geotransform_10 = (-180.0, 1.0, 0.0, -90.0, 0.0, 1.0)
size_10 = (360,180) # (x,y) map size
geotransform_25 = (-180.0, 2.5, 0.0, -90.0, 0.0, 2.5)
size_25 = (144,72) # (x,y) map size

# make GridDefinition instance
GPCC_025_grid = GridDefinition(name='GPCC_025',projection=None, geotransform=geotransform_025, size=size_025)
GPCC_05_grid = GridDefinition(name='GPCC_05',projection=None, geotransform=geotransform_05, size=size_05)
GPCC_10_grid = GridDefinition(name='GPCC_10',projection=None, geotransform=geotransform_10, size=size_10)
GPCC_25_grid = GridDefinition(name='GPCC_25',projection=None, geotransform=geotransform_25, size=size_25)


# variable attributes and name
varatts = dict(p    = dict(name='precip', units='mm/month', transform=transformPrecip), # total precipitation rate
               s    = dict(name='stations', units='#'), # number of gauges for observation
               # axes (don't have their own file; listed in axes)
               lon  = dict(name='lon', units='deg E'), # geographic longitude field
               lat  = dict(name='lat', units='deg N')) # geographic latitude field
#                time = dict(name='time', units='days', offset=1)) # time coordinate
# attributes of the time axis depend on type of dataset 
ltmvaratts = dict(time=dict(name='time', units='month', offset=1), **varatts) 
tsvaratts = dict(time=dict(name='time', units='day', offset=-28854), **varatts)
# N.B.: the time-series time offset is chose such that 1979 begins with the origin (time=0)
# list of variables to load
varlist = varatts.keys() # also includes coordinate fields    


## Functions to load different types of GPCC datasets 

# climatology
ltmfolder = root_folder + 'climatology/' # climatology subfolder
def loadGPCC_LTM(name=dataset_name, varlist=None, resolution='025', varatts=ltmvaratts, filelist=None, 
                 folder=ltmfolder):
  ''' Get a properly formatted dataset the monthly accumulated GPCC precipitation climatology. '''
  # prepare input
  if resolution not in ('025','05', '10', '25'): raise DatasetError, "Selected resolution '%s' is not available!"%resolution
  # translate varlist
  if varlist is None: varlist = varatts.keys()
  if varlist and varatts: varlist = translateVarNames(varlist, varatts)
  # load variables separately
  if 'p' in varlist:
    dataset = DatasetNetCDF(name=name, folder=folder, filelist=['normals_v2011_%s.nc'%resolution], varlist=['p'], 
                            varatts=varatts, ncformat='NETCDF4_CLASSIC')
  if 's' in varlist: 
    gauges = nc.Dataset(folder+'normals_gauges_v2011_%s.nc'%resolution, mode='r', format='NETCDF4_CLASSIC')
    stations = Variable(data=gauges.variables['p'][0,:,:], axes=(dataset.lat,dataset.lon), **varatts['s'])
    # consolidate dataset
    dataset.addVariable(stations, asNC=False, copy=True)  
  dataset = addGDALtoDataset(dataset, projection=None, geotransform=None, gridfolder=grid_folder)
  # N.B.: projection should be auto-detected as geographic
  # return formatted dataset
  return dataset


def checkGridRes(grid, resolution, period=None, lclim=False):
  ''' helper function to verify grid/resoluton selection ''' 
  # prepare input
  if grid is not None and grid[0:5].lower() == 'gpcc_': 
    resolution = grid[5:]
    grid = None
  elif resolution is None: 
    resolution = '025' if period is None and lclim else '05'
  # check for valid resolution 
  if resolution not in ('025','05', '10', '25'): 
    raise DatasetError, "Selected resolution '%s' is not available!"%resolution  
  if resolution == '025' and period is not None: 
    raise DatasetError, "The highest resolution is only available for the long-term mean!"
  # return
  return grid, resolution

# Time-series (monthly)
orig_ts_folder = root_folder + 'full_data_1900-2010/' # climatology subfolder
orig_ts_file = 'full_data_v6_{0:s}_{1:s}.nc' # extend by variable name and resolution
tsfile = 'gpcc{0:s}_monthly.nc' # extend with grid type only
def loadGPCC_TS(name=dataset_name, grid=None, varlist=None, resolution='25', varatts=None, filelist=None, 
                folder=None, lautoregrid=None):
  ''' Get a properly formatted dataset with the monthly GPCC time-series. '''
  if grid is None:
    # load from original time-series files 
    if folder is None: folder = orig_ts_folder
    # prepare input  
    if resolution not in ('05', '10', '25'): raise DatasetError, "Selected resolution '%s' is not available!"%resolution
    # translate varlist
    if varatts is None: varatts = tsvaratts.copy()
    if varlist is None: varlist = varatts.keys()
    if varlist and varatts: varlist = translateVarNames(varlist, varatts)
    if filelist is None: # generate default filelist
      filelist = []
      if 'p' in varlist: filelist.append(orig_ts_file.format('precip',resolution))
      if 's' in varlist: filelist.append(orig_ts_file.format('statio',resolution))
    # load dataset
    dataset = DatasetNetCDF(name=name, folder=folder, filelist=filelist, varlist=varlist, varatts=varatts, multifile=False, ncformat='NETCDF4_CLASSIC')
    # replace time axis with number of month since Jan 1979 
    data = np.arange(0,len(dataset.time),1, dtype='int16') + (1901-1979)*12 # month since 1979 (Jan 1979 = 0)
    timeAxis = Axis(name='time', units='month', coord=data, atts=dict(long_name='Month since 1979-01'))
    dataset.replaceAxis(dataset.time, timeAxis, asNC=False, deepcopy=False)
    # add GDAL info
    dataset = addGDALtoDataset(dataset, projection=None, geotransform=None)
    # N.B.: projection should be auto-detected as geographic
  else:
    # load from neatly formatted and regridded time-series files
    if folder is None: folder = avgfolder
    grid, resolution = checkGridRes(grid, resolution, period=None, lclim=False)
    dataset = loadObservations(name=name, folder=folder, projection=None, resolution=resolution, grid=grid, 
                               period=None, varlist=varlist, varatts=varatts, filepattern=tsfile, 
                               filelist=filelist, lautoregrid=lautoregrid, mode='time-series')
  # return formatted dataset
  return dataset

# pre-processed climatology files (varatts etc. should not be necessary)
avgfolder = root_folder + 'gpccavg/' 
avgfile = 'gpcc{0:s}_clim{1:s}.nc' # the filename needs to be extended by %('_'+resolution,'_'+period)
# function to load these files...
def loadGPCC(name=dataset_name, resolution=None, period=None, grid=None, varlist=None, varatts=None, 
             folder=avgfolder, filelist=None, lautoregrid=True):
  ''' Get the pre-processed monthly GPCC climatology as a DatasetNetCDF. '''
  grid, resolution = checkGridRes(grid, resolution, period=period, lclim=True)
  # load standardized climatology dataset with GPCC-specific parameters
  dataset = loadObservations(name=name, folder=folder, projection=None, resolution=resolution, period=period, 
                             grid=grid, varlist=varlist, varatts=varatts, filepattern=avgfile, 
                             filelist=filelist, lautoregrid=lautoregrid, mode='climatology')
  # return formatted dataset
  return dataset

# function to load station climatologies
def loadGPCC_Stn(name=dataset_name, period=None, station=None, resolution=None, varlist=None, varatts=None, 
                folder=avgfolder, filelist=None, lautoregrid=True):
  ''' Get the pre-processed monthly GPCC climatology as a DatasetNetCDF at station locations. '''
  grid, resolution = checkGridRes(None, resolution, period=period, lclim=True); del grid
  # load standardized climatology dataset with GPCC-specific parameters
  dataset = loadObservations(name=name, folder=folder, projection=None, period=period, station=station, 
                             varlist=varlist, varatts=varatts, filepattern=avgfile, filelist=filelist, 
                             resolution=resolution, lautoregrid=False, mode='climatology')
  # return formatted dataset
  return dataset

# function to load station time-series
def loadGPCC_StnTS(name=dataset_name, station=None, resolution=None, varlist=None, varatts=None, 
                  folder=avgfolder, filelist=None, lautoregrid=True):
  ''' Get the pre-processed monthly GPCC time-series as a DatasetNetCDF at station locations. '''
  grid, resolution = checkGridRes(None, resolution, period=None, lclim=False); del grid
  # load standardized time-series dataset with GPCC-specific parameters
  dataset = loadObservations(name=name, folder=folder, projection=None, period=None, station=station, 
                             varlist=varlist, varatts=varatts, filepattern=tsfile, filelist=filelist, 
                             resolution=resolution, lautoregrid=False, mode='time-series')
  # return formatted dataset
  return dataset

# function to load regionally averaged climatologies
def loadGPCC_Shp(name=dataset_name, period=None, shape=None, resolution=None, varlist=None, varatts=None, 
                folder=avgfolder, filelist=None, lautoregrid=True, lencl=False):
  ''' Get the pre-processed monthly GPCC climatology as a DatasetNetCDF averaged over regions. '''
  grid, resolution = checkGridRes(None, resolution, period=period, lclim=True); del grid
  # load standardized climatology dataset with GPCC-specific parameters
  dataset = loadObservations(name=name, folder=folder, projection=None, period=period, shape=shape, lencl=lencl,
                             station=None, varlist=varlist, varatts=varatts, filepattern=avgfile, 
                             filelist=filelist, resolution=resolution, lautoregrid=False, mode='climatology')
  # return formatted dataset
  return dataset

# function to load regional/shape time-series
def loadGPCC_ShpTS(name=dataset_name, shape=None, resolution=None, varlist=None, varatts=None, 
                  folder=avgfolder, filelist=None, lautoregrid=True, lencl=False):
  ''' Get the pre-processed monthly GPCC time-series as a DatasetNetCDF averaged over regions. '''
  grid, resolution = checkGridRes(None, resolution, period=None, lclim=False); del grid
  # load standardized time-series dataset with GPCC-specific parameters
  dataset = loadObservations(name=name, folder=folder, projection=None, shape=shape, station=None, lencl=lencl, 
                             varlist=varlist, varatts=varatts, filepattern=tsfile, filelist=filelist, 
                             resolution=resolution, lautoregrid=False, mode='time-series', period=None)
  # return formatted dataset
  return dataset


## Dataset API

dataset_name # dataset name
root_folder # root folder of the dataset
orig_file_pattern = orig_ts_file # filename pattern: variable name and resolution
ts_file_pattern = tsfile # filename pattern: grid
clim_file_pattern = avgfile # filename pattern: grid, and period
data_folder = avgfolder # folder for user data
grid_def = {'025':GPCC_025_grid, '05':GPCC_05_grid, '10':GPCC_10_grid, '25':GPCC_25_grid}
LTM_grids = ['025','05','10','25'] # grids that have long-term mean data 
TS_grids = ['05','10','25'] # grids that have time-series data
grid_res = {'025':0.25, '05':0.5, '10':1.0, '25':2.5}
default_grid = GPCC_025_grid
# grid_def = {0.25:GPCC_025_grid, 0.5:GPCC_05_grid, 1.0:GPCC_10_grid, 2.5:GPCC_25_grid}
# grid_tag = {0.25:'025', 0.5:'05', 1.0:'10', 2.5:'25'}
# functions to access specific datasets
loadLongTermMean = loadGPCC_LTM # climatology provided by publisher
loadTimeSeries = loadGPCC_TS # time-series data
loadClimatology = loadGPCC # pre-processed, standardized climatology
loadStationClimatology = loadGPCC_Stn # climatologies without associated grid (e.g. stations or basins) 
loadStationTimeSeries = loadGPCC_StnTS # time-series without associated grid (e.g. stations or basins)
loadShapeClimatology = loadGPCC_Shp # climatologies without associated grid (e.g. provinces or basins) 
loadShapeTimeSeries = loadGPCC_ShpTS # time-series without associated grid (e.g. provinces or basins)


## (ab)use main execution for quick test
if __name__ == '__main__':
  
  mode = 'test_climatology'; reses = ('025',); period = None
#   mode = 'test_timeseries'; reses = ('25',)
#   mode = 'test_point_climatology'; reses = ('025',); period = None
#   mode = 'test_point_timeseries'; reses = ('05',)  
#   mode = 'convert_climatology'; reses = ('25',); period = None
#   reses = ('025','05', '10', '25'); period = None  
  mode = 'average_timeseries'; reses = ('25',) # for testing
  reses = ('05', '10', '25')
#   reses = ('25',)
  period = (1970,2000)
#   period = (1979,1982)
#   period = (1979,1983)
#   period = (1979,1984)
#   period = (1979,1989)
#   period = (1979,1991)
#   period = (1979,1994)
#   period = (1984,1994)
#   period = (1989,1994)
#   period = (1979,2009)
#   period = (1949,2009)
#   period = (1997,1998)
#   period = (1979,1980)
#   period = (2010,2011)
  grid = 'GPCC' # 'arb2_d02'
  pntset = 'shpavg' # 'ecprecip'
  
  # generate averaged climatology
  for res in reses:    
    
    if mode == 'test_climatology':
            
      # load averaged climatology file
      print('')
      dataset = loadGPCC(grid=grid,resolution=res,period=period)
      print(dataset)
      print('')
      print(dataset.geotransform)
      print(dataset.precip.getArray().mean())
      print(dataset.precip.masked)
      
      # print time coordinate
      print
      print dataset.time.atts
      print
      print dataset.time.data_array
          
    elif mode == 'test_timeseries':
      
      # load time-series file
      print('')
      dataset = loadGPCC_TS(grid=grid,resolution=res)
      print(dataset)
      print('')
      print(dataset.time)
      print(dataset.time.coord)
      print(dataset.time.coord[78*12]) # Jan 1979
#       print(dataset.time.masked)
#       print('')
#       print(dataset.geotransform)
#       print(dataset.precip.getArray().mean())
#       print(dataset.precip.masked)
      
          
    if mode == 'test_point_climatology':
            
      # load averaged climatology file
      print('')
      if pntset in ('shpavg',): 
        dataset = loadGPCC_Shp(shape=pntset, resolution=res, period=period)
        print(dataset.shp_area.mean())
        print('')
      else: dataset = loadGPCC_Stn(station=pntset, resolution=res, period=period)
      dataset.load()
      print(dataset)
      print('')
      print(dataset.precip.mean())
      print(dataset.precip.masked)
      
      # print time coordinate
      print
      print dataset.time.atts
      print
      print dataset.time.data_array

    elif mode == 'test_point_timeseries':
    
      # load station time-series file
      print('') 
      if pntset in ('shpavg',): dataset = loadGPCC_ShpTS(shape=pntset, resolution=res)
      else: dataset = loadGPCC_StnTS(station=pntset, resolution=res)
      print(dataset)
      print('')
      print(dataset.time)
      print(dataset.time.coord)
      assert dataset.time.coord[78*12] == 0 # Jan 1979
      assert dataset.shape[0] == 1

        
    elif mode == 'convert_climatology':      
      
      
      # load dataset
      dataset = loadGPCC_LTM(varlist=['stations','precip'],resolution=res)
      # change meta-data
      dataset.name = 'GPCC'
      dataset.title = 'GPCC Long-term Climatology'
      dataset.atts.resolution = res      
      # load data into memory
      dataset.load()

      # add landmask
      addLandMask(dataset) # create landmask from precip mask
      dataset.mask(dataset.landmask) # mask all fields using the new landmask      
      # add length and names of month
      addLengthAndNamesOfMonth(dataset, noleap=False) 
      
      # figure out a different filename
      filename = getFileName(grid=res, period=None, name='GPCC', filepattern=avgfile)
      print('\n'+filename+'\n')      
      if os.path.exists(avgfolder+filename): os.remove(avgfolder+filename)      
      # write data and some annotation
      ncset = writeNetCDF(dataset, avgfolder+filename, close=False)
#       add_var(ncset,'name_of_month', name_of_month, 'time', # add names of month
#                  atts=dict(name='name_of_month', units='', long_name='Name of the Month')) 
       
      # close...
      ncset.close()
      dataset.close()
      # print dataset before
      print(dataset)
      print('')           
      
      
    elif mode == 'average_timeseries':
      
      
      # load source
      periodstr = 'Climatology' if period is None else '{0:4d}-{1:4d}'.format(*period)
      print('\n\n   ***   Processing Resolution %s from %s   ***   \n\n'%(res,periodstr))
      if period is None: source = loadGPCC_LTM(varlist=None,resolution=res) # ['stations','precip']
      else: source = loadGPCC_TS(varlist=None,resolution=res)
      source = source(time=timeSlice(period))
      #source.load()
      print(source)
      print('\n')
            
      # prepare sink
      gridstr = res if grid == 'GPCC' else grid
      filename = getFileName(grid=gridstr, period=period, name='GPCC', filepattern=avgfile)
      if os.path.exists(avgfolder+filename): os.remove(avgfolder+filename)
      atts =dict(period=periodstr, name='GPCC', title='GPCC Climatology') 
      sink = DatasetNetCDF(name='GPCC Climatology', folder=avgfolder, filelist=[filename], atts=source.atts, mode='w')
#       sink = addGDALtoDataset(sink, griddef=source.griddef)
      
      # initialize processing
      CPU = CentralProcessingUnit(source, sink, tmp=True)

      if period is not None:
        # determine averaging interval
        offset = source.time.getIndex(period[0]-1979)/12 # origin of monthly time-series is at January 1979 
        # start processing climatology
        CPU.Climatology(period=period[1]-period[0], offset=offset, flush=False)
#         CPU.sync(flush=True)

      # get NARR coordinates
      if grid is not 'GPCC':
        griddef = loadPickledGridDef(grid=grid, res=None, folder=grid_folder)
        #new_grid = import_module(grid[0:4]).__dict__[grid+'_grid']
#       if grid == 'NARR':
#         from datasets.NARR import NARR_grid
        # reproject and resample (regrid) dataset
        CPU.Regrid(griddef=griddef, flush=False)
            
#       # shift longitude axis by 180 degrees  left (i.e. -180 - 180 -> 0 - 360)
#       print('')
#       print('   +++   processing shift longitude   +++   ') 
#       CPU.Shift(lon=-180, flush=True)
#       print('\n')
#
#       # shift longitude axis by 180 degrees  left (i.e. -180 - 180 -> 0 - 360)
#       print('')
#       print('   +++   processing shift/roll   +++   ') 
#       CPU.Shift(shift=72, axis='lon', byteShift=True, flush=False)
#       print('\n')      

      # get results
      CPU.sync(flush=True)
      
      
#       print('')
#       print(sink)
#       print('')

      # add geolocators
      if grid is not 'GPCC':
        sink = addGeoLocator(sink, griddef=griddef, lgdal=True, lcheck=True)
        
      # add landmask
      #sink.mask(sink.landmask)
      #print sink.dataset
      addLandMask(sink) # create landmask from precip mask
      #sink.stations.mask(sink.landmask) # mask all fields using the new landmask
      # add length and names of month
      addLengthAndNamesOfMonth(sink, noleap=False) 
                    
#       newvar = sink.precip
#       print
#       print newvar.name, newvar.masked
#       print newvar.fillValue
#       print newvar.data_array.__class__
#       print
      
      # close...
      sink.sync()
      sink.close()
      # print dataset
      print('')
      print(sink)
      del sink     
      print

#       # print time coordinate
#       dataset = loadGPCC(grid=grid,resolution=res,period=period)      
#       print dataset
#       print
#       print dataset.time
#       print
#       print dataset.time.data_array
    
