'''
Created on 2017-12-24

This module contains common meta data and access functions for CMIP5 model output. 

@author: Andre R. Erler, GPL v3
'''

# external imports
import numpy as np
import os, pickle
# from atmdyn.properties import variablePlotatts
from geodata.base import Variable, Axis, concatDatasets, monthlyUnitsList
from geodata.netcdf import DatasetNetCDF, VarNC
from geodata.gdal import addGDALtoDataset, GDALError
from geodata.misc import DatasetError, AxisError, DateError, ArgumentError, isNumber, isInt
from datasets.common import ( translateVarNames, data_root, grid_folder, default_varatts, 
                              addLengthAndNamesOfMonth, selectElements, stn_params, shp_params )
from geodata.gdal import loadPickledGridDef, griddef_pickle
from datasets.WRF import Exp as WRF_Exp
from processing.process import CentralProcessingUnit

# some meta data (needed for defaults)
root_folder = data_root + '/CMIP5/' # long-term mean folder
#outfolder = root_folder + 'cesmout/' # WRF output folder
avgfolder = root_folder + 'cmip5avg/' # monthly averages and climatologies
#cvdpfolder = root_folder + 'cvdp/' # CVDP output (netcdf files and HTML tree)
#diagfolder = root_folder + 'diag/' # output from AMWG diagnostic package (climatologies and HTML tree) 

## list of experiments
class Exp(WRF_Exp): 
  parameters = WRF_Exp.parameters.copy()
  defaults = WRF_Exp.defaults.copy()
  # special CESM parameters
  #parameters['cvdpfolder'] = dict(type=basestring,req=True) # new parameters need to be registered
  #parameters['diagfolder'] = dict(type=basestring,req=True) # new parameters need to be registered
  # and defaults
  defaults['avgfolder'] = lambda atts: '{0:s}/{1:s}/'.format(avgfolder,atts['name'])
  #defaults['cvdpfolder'] = lambda atts: '{0:s}/{1:s}/'.format(cvdpfolder,atts['name'])
  #defaults['diagfolder'] = lambda atts: '{0:s}/{1:s}/'.format(diagfolder,atts['name'])
  defaults['domains'] = None # not applicable here
  defaults['parent'] = None # not applicable here
  

# return name and folder
def getFolderName(name=None, experiment=None, folder=None, mode='avg', cvdp_mode=None, exps=None):
  ''' Convenience function to infer and type-check the name and folder of an experiment based on various input. '''
  # N.B.: 'experiment' can be a string name or an Exp instance
  # figure out experiment name
  if experiment is None and name not in exps:
    if cvdp_mode is None: cvdp_mode = 'ensemble' # backwards-compatibility
    if not isinstance(folder,basestring):
      if mode == 'cvdp' and ( cvdp_mode == 'observations' or cvdp_mode == 'grand-ensemble' ): 
        folder = "{:s}/grand-ensemble/".format(cvdpfolder)              
      else: raise IOError, "Need to specify an experiment folder in order to load data."    
  else:
    # load experiment meta data
    if isinstance(experiment,Exp): pass # preferred option
    elif exps is None: raise DatasetError, 'No dictionary of Exp instances specified.'
    elif isinstance(experiment,basestring): experiment = exps[experiment] 
    elif isinstance(name,basestring) and name in exps: experiment = exps[name]
    else: raise DatasetError, 'Dataset of name \'{0:s}\' not found!'.format(name or experiment)
    if cvdp_mode is None:
      cvdp_mode = 'ensemble' if experiment.ensemble else ''  
    # root folder
    if folder is None: 
      if mode == 'avg': folder = experiment.avgfolder
      elif mode == 'cvdp': 
        if cvdp_mode == 'ensemble': 
          expfolder = experiment.ensemble or experiment.name 
          folder = "{:s}/{:s}/".format(cvdpfolder,expfolder)
        elif cvdp_mode == 'grand-ensemble': folder = "{:s}/grand-ensemble/".format(cvdpfolder)
        else: folder = experiment.cvdpfolder
      elif mode == 'diag': folder = experiment.diagfolder
      else: raise NotImplementedError,"Unsupported mode: '{:s}'".format(mode)
    elif not isinstance(folder,basestring): raise TypeError
    # name
    if name is None: name = experiment.shortname
    if not isinstance(name,basestring): raise TypeError      
  # check if folder exists
  if not os.path.exists(folder): raise IOError, 'Dataset folder does not exist: {0:s}'.format(folder)
  # return name and folder
  return folder, experiment, name


## variable attributes and name
class FileType(object): 
  ''' Container class for all attributes of of the constants files. '''
  atts = NotImplemented
  vars = NotImplemented
  climfile = None
  tsfile = None
  cvdpfile = None
  diagfile = None

### NOTE THE UNITS AND FILLVALUES ARE NOT ALL CHECKED YET!!!!

#class CMIP53D(FileType):
#  ''' Variables and attributes of the atmospheric pressure level files. '''
#  def __init__(self):
#    #self.name = 'plev3d'
#    self.atts = dict(hur     = dict(name='RH', units='\%',  fillValue=-999, atts=dict(long_name='Relative Humidity')), # Relative Humidity
#                     hus     = dict(name='??', units='??',  fillValue=-999, atts=dict(long_name='Specific Humidity')), # Specific Humidity
#                     ta      = dict(name='T',  units='K',   fillValue=-999, atts=dict(long_name='Air Temperature')), # Temperature
#                     ua      = dict(name='u',  units='m/s', fillValue=-999, atts=dict(long_name='Eastward Wind')), # Zonal Wind Speed
#                     va      = dict(name='v',  units='m/s', fillValue=-999, atts=dict(long_name='Northward Wind')),) # Meridional Wind Speed
#                     zg      = dict(name='Z',  units='m',   fillValue=-999, atts=dict(long_name='Geopotential Height ')), # Geopotential Height 
                     
                     


class CMIP52D(FileType):
  ''' Variables and attributes of the atmospheric surface files. '''
  def __init__(self):
    self.atts = dict(clt         = dict(name='clt', units='\%'), # Total Cloud Fraction (Not recorded in WRF/CESM)
                     evspsbl     = dict(name='evap', units='kg/m^2/s'), # surface evaporation
                     hfls        = dict(name='lhfx', units='W/m^2'), # surface latent heat flux 
                     hfss        = dict(name='hfx', units='W/m^2'), # surface sensible heat flux 
                     huss        = dict(name='q2', units='kg/kg'), # 2m specific humidity                  
                     pr          = dict(name='precip', units='kg/m^2/s'), # total precipitation rate (kg/m^2/s)  
                     prc         = dict(name='preccu', units='kg/m^2/s'), # convective precipitation rate (kg/m^2/s)
                     prsn        = dict(name='solprec', units='kg/m^2/s'), # solid precipitation rate
                     prw         = dict(name='prw', units='kg/m^2'), # atmospheric water vapor content (Not recorded in WRF/CESM)
                     ps          = dict(name='ps', units='Pa'), # surface air pressure
                     psl         = dict(name='pmsl', units='Pa'), # mean sea level pressure (Not recorded in WRF)
                     rlds        = dict(name='LWDNB', units='W/m^2'), # Downwelling Longwave Radiation at Surface
                     rlus        = dict(name='LWUPB', units='W/m^2'), # Upwelling Longwave Radiation at Surface
                     rlut        = dict(name='LWUPT', units='W/m^2'), # Outgoing Longwave Radiation at Top of Atmosphere
                     rsds        = dict(name='SWDNB', units='W/m^2'), # Downwelling Shortwave Radiation at Surface    
                     rsdt        = dict(name='SWDNT', units='W/m^2'), # Incident Shortwave Radiation at Top of Atmosphere
                     rsus        = dict(name='SWUPB', units='W/m^2'), # Upwelling Shortwave Radiation at Surface
                     rsut        = dict(name='SWUPT', units='W/m^2'), # Outgoing Shortwave Radiation at Top of Atmosphere
                     tas         = dict(name='T2', units='K'), # 2m Temperature
                     tasmax      = dict(name='MaxTmax', units='K'),   # Daily Maximum Temperature (monthly mean at 2m)                     
                     tasmin      = dict(name='MinTmin', units='K'),   # Daily Minimum Temperature (monthly mean at 2m)
                     ts          = dict(name='Ts', units='K'), # Skin Temperature (SST)
                     uas         = dict(name='u10',  units='m/s'), # Surface Zonal Wind Speed (at 10m)
                     vas         = dict(name='v10',  units='m/s'), # Surface Meridional Wind Speed (at 10m)
                     #TS         = dict(name='SST', units='K'), # Skin Temperature (SST)
                     )

#class LMON(FileType): # Fractional data
#  ''' Variables and attributes of the land surface files. '''
#  def __init__(self):
#    self.atts = dict(mrros    = dict(name='sfroff', units='kg/m^2/s'), # surface run-off
#                     mrro  = dict(name='runoff', units='kg/m^2/s'), # total surface and sub-surface run-off
#                     mrso  = dict(name='runoff', units='kg/m^2/s'), # Total soil moisture content
#                     )
#
#class LIMON(FileType):
#  ''' Variables and attributes of the landIce surface files. '''
#  def __init__(self):
#    self.atts = dict(snc     = dict(name='snwcvr', units=''), # snow cover (fractional)
#                     snm    = dict(name='snwmlt', units='kg/m^2/s'), # snow melting rate
#                     )
  

class fx(FileType):
  ''' Variables and attributes of the time invariant files. '''
  def __init__(self):
    self.name = 'const' 
    self.atts = dict(orog    = dict(name='zs', units='m'), # surface altitude
    
# axes (don't have their own file)
class Axes(FileType):
  ''' A mock-filetype for axes. '''
  def __init__(self):
    self.atts = dict(time        = dict(name='time', units='days', offset=-47116, atts=dict(long_name='Month since 1979')), # time coordinate (days since 1979-01-01)
                     # NOTE THAT THE CMIP5 DATASET HAVE DIFFERENT TIME OFFSETS BETWEEN MEMBERS !!!
                     # N.B.: the time coordinate is only used for the monthly time-series data, not the LTM
                     #       the time offset is chose such that 1979 begins with the origin (time=0)
                     lon           = dict(name='lon', units='deg E'), # west-east coordinate
                     lat           = dict(name='lat', units='deg N'), # south-north coordinate
                     plev = dict(name='lev', units='')) # hybrid pressure coordinate
    self.vars = self.atts.keys()

# Time-Series (monthly)
def loadCMIP5_TS(experiment=None, name=None, grid=None, filetypes=None, varlist=None, varatts=None,  
                translateVars=None, lautoregrid=None, load3D=False, ignore_list=None, lcheckExp=True,
                lreplaceTime=True, lwrite=False, exps=None):
  ''' Get a properly formatted CESM dataset with a monthly time-series. (wrapper for loadCESM)'''
  return loadCMIP5_All(experiment=experiment, name=name, grid=grid, period=None, station=None, 
                      filetypes=filetypes, varlist=varlist, varatts=varatts, translateVars=translateVars, 
                      lautoregrid=lautoregrid, load3D=load3D, ignore_list=ignore_list, mode='time-series', 
                      lcheckExp=lcheckExp, lreplaceTime=lreplaceTime, lwrite=lwrite, exps=exps)

# load minimally pre-processed CESM climatology files 
def loadCMIP5(experiment=None, name=None, grid=None, period=None, filetypes=None, varlist=None, 
             varatts=None, translateVars=None, lautoregrid=None, load3D=False, ignore_list=None, 
             lcheckExp=True, lreplaceTime=True, lencl=False, lwrite=False, exps=None):
  ''' Get a properly formatted monthly CESM climatology as NetCDFDataset. '''
  return loadCMIP5_All(experiment=experiment, name=name, grid=grid, period=period, station=None, 
                      filetypes=filetypes, varlist=varlist, varatts=varatts, translateVars=translateVars, 
                      lautoregrid=lautoregrid, load3D=load3D, ignore_list=ignore_list, exps=exps, 
                      mode='climatology', lcheckExp=lcheckExp, lreplaceTime=lreplaceTime, lwrite=lwrite)


# load any of the various pre-processed CESM climatology and time-series files 
def loadCMIP5_All(experiment=None, name=None, grid=None, station=None, shape=None, period=None, 
                 varlist=None, varatts=None, translateVars=None, lautoregrid=None, load3D=False, 
                 ignore_list=None, mode='climatology', cvdp_mode=None, lcheckExp=True, exps=None,
                 lreplaceTime=True, filetypes=None, lencl=False, lwrite=False, check_vars=None):
  ''' Get any of the monthly CESM files as a properly formatted NetCDFDataset. '''
  # period
  if isinstance(period,(tuple,list)):
    if not all(isNumber(period)): raise ValueError
  elif isinstance(period,basestring): period = [int(prd) for prd in period.split('-')]
  elif isinstance(period,(int,np.integer)) or period is None : pass # handled later
  else: raise DateError, "Illegal period definition: {:s}".format(str(period))
  # prepare input  
  lclim = False; lts = False; lcvdp = False; ldiag = False # mode switches
  if mode.lower() == 'climatology': # post-processed climatology files
    lclim = True
    folder,experiment,name = getFolderName(name=name, experiment=experiment, folder=None, mode='avg', exps=exps)    
    if period is None: raise DateError, 'Currently CESM Climatologies have to be loaded with the period explicitly specified.'
  elif mode.lower() in ('time-series','timeseries'): # concatenated time-series files
    lts = True
    folder,experiment,name = getFolderName(name=name, experiment=experiment, folder=None, mode='avg', exps=exps)
    lclim = False; period = None; periodstr = None # to indicate time-series (but for safety, the input must be more explicit)
    if lautoregrid is None: lautoregrid = False # this can take very long!
  elif mode.lower() == 'cvdp': # concatenated time-series files
    lcvdp = True
    folder,experiment,name = getFolderName(name=name, experiment=experiment, folder=None, mode='cvdp', 
                                           cvdp_mode=cvdp_mode, exps=exps)
    if period is None:
      if not isinstance(experiment,Exp): raise DatasetError, 'Periods can only be inferred for registered datasets.'
      period = (experiment.beginyear, experiment.endyear)  
  elif mode.lower() == 'diag': # concatenated time-series files
    ldiag = True
    folder,experiment,name = getFolderName(name=name, experiment=experiment, folder=None, mode='diag', exps=exps)
    raise NotImplementedError, "Loading AMWG diagnostic files is not supported yet."
  else: raise NotImplementedError,"Unsupported mode: '{:s}'".format(mode)  
  # cast/copy varlist
  if isinstance(varlist,basestring): varlist = [varlist] # cast as list
  elif varlist is not None: varlist = list(varlist) # make copy to avoid interference
  # handle stations and shapes
  if station and shape: raise ArgumentError
  elif station or shape: 
    if grid is not None: raise NotImplementedError, 'Currently CESM station data can only be loaded from the native grid.'
    if lcvdp: raise NotImplementedError, 'CVDP data is not available as station data.'
    if lautoregrid: raise GDALError, 'Station data can not be regridded, since it is not map data.'   
    lstation = bool(station); lshape = bool(shape)
    # add station/shape parameters
    if varlist:
      params = stn_params if lstation else shp_params
      for param in params:
        if param not in varlist: varlist.append(param)
  else:
    lstation = False; lshape = False
  # period  
  if isinstance(period,(int,np.integer)):
    if not isinstance(experiment,Exp): raise DatasetError, 'Integer periods are only supported for registered datasets.'
    period = (experiment.beginyear, experiment.beginyear+period)
  if lclim: periodstr = '_{0:4d}-{1:4d}'.format(*period)
  elif lcvdp: periodstr = '{0:4d}-{1:4d}'.format(period[0],period[1]-1)
  else: periodstr = ''
  # N.B.: the period convention in CVDP is that the end year is included
  # generate filelist and attributes based on filetypes and domain
  if filetypes is None: filetypes = ['atm','lnd']
  elif isinstance(filetypes,(list,tuple,set,basestring)):
    if isinstance(filetypes,basestring): filetypes = [filetypes]
    else: filetypes = list(filetypes)
    # interprete/replace WRF filetypes (for convenience)
    tmp = []
    for ft in filetypes:
      if ft in ('const','drydyn3d','moist3d','rad','plev3d','srfc','xtrm','hydro'):
        if 'atm' not in tmp: tmp.append('atm')
      elif ft in ('lsm','snow'):
        if 'lnd' not in tmp: tmp.append('lnd')
      elif ft in ('aux'): pass # currently not supported
#       elif ft in (,):
#         if 'atm' not in tmp: tmp.append('atm')
#         if 'lnd' not in tmp: tmp.append('lnd')        
      else: tmp.append(ft)
    filetypes = tmp; del tmp
    if 'axes' not in filetypes: filetypes.append('axes')    
  else: raise TypeError  
  atts = dict(); filelist = []; typelist = []
  for filetype in filetypes:
    fileclass = fileclasses[filetype]
    if lclim and fileclass.climfile is not None: filelist.append(fileclass.climfile)
    elif lts and fileclass.tsfile is not None: filelist.append(fileclass.tsfile)
    elif lcvdp and fileclass.cvdpfile is not None: filelist.append(fileclass.cvdpfile)
    elif ldiag and fileclass.diagfile is not None: filelist.append(fileclass.diagfile)
    typelist.append(filetype)
    atts.update(fileclass.atts) 
  # figure out ignore list  
  if ignore_list is None: ignore_list = set(ignore_list_2D)
  elif isinstance(ignore_list,(list,tuple)): ignore_list = set(ignore_list)
  elif not isinstance(ignore_list,set): raise TypeError
  if not load3D: ignore_list.update(ignore_list_3D)
  if lautoregrid is None: lautoregrid = not load3D # don't auto-regrid 3D variables - takes too long!
  # translate varlist
  if varatts is not None: atts.update(varatts)
  lSST = False
  if varlist is not None:
    varlist = list(varlist) 
    if 'SST' in varlist: # special handling of name SST variable, as it is part of Ts
      varlist.remove('SST')
      if not 'Ts' in varlist: varlist.append('Ts')
      lSST = True # Ts is renamed to SST below
    if translateVars is None: varlist = list(varlist) + translateVarNames(varlist, atts) # also aff translations, just in case
    elif translateVars is True: varlist = translateVarNames(varlist, atts) 
    # N.B.: DatasetNetCDF does never apply translation!
  # NetCDF file mode
  ncmode = 'rw' if lwrite else 'r'   
  # get grid or station-set name
  if lstation:
    # the station name can be inserted as the grid name
    gridstr = '_'+station.lower(); # only use lower case for filenames
    griddef = None
  elif lshape:
    # the station name can be inserted as the grid name
    gridstr = '_'+shape.lower(); # only use lower case for filenames
    griddef = None
  else:
    if grid is None or grid == experiment.grid: 
      gridstr = ''; griddef = None
    else: 
      gridstr = '_'+grid.lower() # only use lower case for filenames
      griddef = loadPickledGridDef(grid=grid, res=None, filename=None, folder=grid_folder, check=True)
  # insert grid name and period
  filenames = []
  for filetype,fileformat in zip(typelist,filelist):
    if lclim: filename = fileformat.format(gridstr,periodstr) # put together specfic filename for climatology
    elif lts: filename = fileformat.format(gridstr) # or for time-series
    elif lcvdp: filename = fileformat.format(experiment.name if experiment else name,periodstr) # not implemented: gridstr
    elif ldiag: raise NotImplementedError
    else: raise DatasetError
    filenames.append(filename) # append to list (passed to DatasetNetCDF later)
    # check existance
    filepath = '{:s}/{:s}'.format(folder,filename)
    if not os.path.exists(filepath):
      nativename = fileformat.format('',periodstr) # original filename (before regridding)
      nativepath = '{:s}/{:s}'.format(folder,nativename)
      if os.path.exists(nativepath):
        if lautoregrid: 
          from processing.regrid import performRegridding # causes circular reference if imported earlier
          griddef = loadPickledGridDef(grid=grid, res=None, folder=grid_folder)
          dataargs = dict(experiment=experiment, filetypes=[filetype], period=period)
          print("The '{:s}' (CESM) dataset for the grid ('{:s}') is not available:\n Attempting regridding on-the-fly.".format(name,filename,grid))
          if performRegridding('CESM','climatology' if lclim else 'time-series', griddef, dataargs): # default kwargs
            raise IOError, "Automatic regridding failed!"
          print("Output: '{:s}'".format(name,filename,grid,filepath))            
        else: raise IOError, "The '{:s}' (CESM) dataset '{:s}' for the selected grid ('{:s}') is not available - use the regrid module to generate it.".format(name,filename,grid) 
      else: raise IOError, "The '{:s}' (CESM) dataset file '{:s}' does not exits!\n({:s})".format(name,filename,folder)
   
  # load dataset
  #print varlist, filenames
  if experiment: title = experiment.title
  else: title = name
  dataset = DatasetNetCDF(name=name, folder=folder, filelist=filenames, varlist=varlist, axes=None, 
                          varatts=atts, title=title, multifile=False, ignore_list=ignore_list, 
                          ncformat='NETCDF4', squeeze=True, mode=ncmode, check_vars=check_vars)
  # replace time axis
  if lreplaceTime:
    if lts or lcvdp:
      # check time axis and center at 1979-01 (zero-based)
      if experiment is None: ys = period[0]; ms = 1
      else: ys,ms,ds = [int(t) for t in experiment.begindate.split('-')]; assert ds == 1
      if dataset.hasAxis('time'):
        ts = (ys-1979)*12 + (ms-1); te = ts+len(dataset.time) # month since 1979 (Jan 1979 = 0)
        atts = dict(long_name='Month since 1979-01')
        timeAxis = Axis(name='time', units='month', coord=np.arange(ts,te,1, dtype='int16'), atts=atts)
        dataset.replaceAxis(dataset.time, timeAxis, asNC=False, deepcopy=False)
      if dataset.hasAxis('year'):
        ts = ys-1979; te = ts+len(dataset.year) # month since 1979 (Jan 1979 = 0)
        atts = dict(long_name='Years since 1979-01')
        yearAxis = Axis(name='year', units='year', coord=np.arange(ts,te,1, dtype='int16'), atts=atts)
        dataset.replaceAxis(dataset.year, yearAxis, asNC=False, deepcopy=False)
    elif lclim:
      if dataset.hasAxis('time') and not dataset.time.units.lower() in monthlyUnitsList:
        atts = dict(long_name='Month of the Year')
        timeAxis = Axis(name='time', units='month', coord=np.arange(1,13, dtype='int16'), atts=atts)
        assert len(dataset.time) == len(timeAxis), dataset.time
        dataset.replaceAxis(dataset.time, timeAxis, asNC=False, deepcopy=False)
      elif dataset.hasAxis('year'): raise NotImplementedError, dataset
  # rename SST
  if lSST: dataset['SST'] = dataset.Ts
  # correct ordinal number of shape (should start at 1, not 0)
  if lshape:
    # mask all shapes that are incomplete in dataset
    if lencl and 'shp_encl' in dataset: dataset.mask(mask='shp_encl', invert=True)   
    if dataset.hasAxis('shapes'): raise AxisError, "Axis 'shapes' should be renamed to 'shape'!"
    if not dataset.hasAxis('shape'): raise AxisError
    if dataset.shape.coord[0] == 0: dataset.shape.coord += 1
  # check
  if len(dataset) == 0: raise DatasetError, 'Dataset is empty - check source file or variable list!'
  # add projection, if applicable
  if not ( lstation or lshape ):
    dataset = addGDALtoDataset(dataset, griddef=griddef, gridfolder=grid_folder, lwrap360=True, geolocator=True)
  # return formatted dataset
  return dataset

## Dataset API

dataset_name = 'CMIP5' # dataset name
root_folder # root folder of the dataset
avgfolder # root folder for monthly averages
outfolder # root folder for direct WRF output
ts_file_pattern = 'cmip5{0:s}{1:s}_monthly.nc' # filename pattern: filetype, grid
clim_file_pattern = 'cmip5{0:s}{1:s}_clim{2:s}.nc' # filename pattern: filetype, grid, period
data_folder = root_folder # folder for user data
grid_def = {'':None} # there are too many... 
grid_res = {'':1.} # approximate grid resolution at 45 degrees latitude
default_grid = None 
# functions to access specific datasets
loadLongTermMean = None # WRF doesn't have that...
loadClimatology = loadCESM # pre-processed, standardized climatology
loadTimeSeries = loadCESM_TS # time-series data
#loadStationClimatology = loadCESM_Stn # pre-processed, standardized climatology at stations
#loadStationTimeSeries = loadCESM_StnTS # time-series data at stations
#loadShapeClimatology = loadCESM_Shp # climatologies without associated grid (e.g. provinces or basins) 
#loadShapeTimeSeries = loadCESM_ShpTS # time-series without associated grid (e.g. provinces or basins)


## (ab)use main execution for quick test
if __name__ == '__main__':
  
  # set mode/parameters
#   mode = 'test_climatology'
#   mode = 'test_timeseries'
#   mode = 'test_ensemble'
#   mode = 'test_point_climatology'
#   mode = 'test_point_timeseries'
#   mode = 'test_point_ensemble'
#   mode = 'test_cvdp'
  mode = 'pickle_grid'
#     mode = 'shift_lon'
#   experiments = ['Ctrl-1', 'Ctrl-A', 'Ctrl-B', 'Ctrl-C']
#   experiments += ['Ctrl-2050', 'Ctrl-A-2050', 'Ctrl-B-2050', 'Ctrl-C-2050']
  experiments = ('Ctrl-1',)
  periods = (15,)
  filetypes = ('atm',) # ['atm','lnd','ice']
  grids = ('cesm1x1',)*len(experiments) # grb1_d01
#   pntset = 'shpavg'
  pntset = 'ecprecip'

  from projects.CESM_experiments import Exp, CESM_exps, ensembles
  # N.B.: importing Exp through CESM_experiments is necessary, otherwise some isinstance() calls fail

  # pickle grid definition
  if mode == 'pickle_grid':
    
    for grid,experiment in zip(grids,experiments):
      
      print('')
      print('   ***   Pickling Grid Definition for {0:s}   ***   '.format(grid))
      print('')
      
      # load GridDefinition
      dataset = loadCESM(experiment=CESM_exps[experiment], grid=None, filetypes=['lnd'], period=(1979,1989))
      griddef = dataset.griddef
      #del griddef.xlon, griddef.ylat      
      print griddef
      griddef.name = grid
      print('   Loading Definition from \'{0:s}\''.format(dataset.name))
      # save pickle
      filename = '{0:s}/{1:s}'.format(grid_folder,griddef_pickle.format(grid))
      if os.path.exists(filename): os.remove(filename) # overwrite
      filehandle = open(filename, 'w')
      pickle.dump(griddef, filehandle)
      filehandle.close()
      
      print('   Saving Pickle to \'{0:s}\''.format(filename))
      print('')
      
      # load pickle to make sure it is right
      del griddef
      griddef = loadPickledGridDef(grid, res=None, folder=grid_folder)
      print(griddef)
      print('')
      print griddef.wrap360
      