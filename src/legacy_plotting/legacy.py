'''
Created on Jan 20, 2015

A retirement home for old plotting functions are still used, but should not be developed further.
These functions are mainly used by the 'areastats' and 'multimap' modules. 

@author: Andre R. Erler, GPL v3
'''

import numpy as np
from types import ModuleType
from warnings import warn
from geodata.misc import isInt, DatasetError


## functions to load datasets for areastats and multimap

# helper function for loadDatasets (see below)
def loadDataset(exp, prd, dom, grd, res, filetypes=None, varlist=None, lbackground=True, lWRFnative=True, 
                lautoregrid=False, WRF_exps=None, CESM_exps=None):
  ''' A function that loads a dataset, based on specified parameters '''
  if not isinstance(exp,basestring): raise TypeError
  if exp[0].isupper():
    if exp == 'Unity': 
      from datasets.Unity import loadUnity
      ext = loadUnity(resolution=res, period=prd, grid=grd, varlist=varlist, lautoregrid=lautoregrid)
      axt = 'Merged Observations'        
    elif exp == 'GPCC': 
      from datasets.GPCC import loadGPCC
      ext = loadGPCC(resolution=res, period=prd, grid=grd, varlist=varlist, lautoregrid=lautoregrid)
      axt = 'GPCC Observations'
    elif exp == 'CRU': 
      from datasets.CRU import loadCRU
      ext = loadCRU(period=prd, grid=grd, varlist=varlist, lautoregrid=lautoregrid)
      axt = 'CRU Observations' 
    elif exp == 'PCIC': # PCIC with some background field
      from datasets.PCIC import loadPCIC
      if lbackground:
        if all(var in ('precip','stations','lon2D','lat2D','landmask','landfrac') for var in varlist): 
          from datasets.GPCC import loadGPCC
          from datasets.PRISM import loadPRISM
          ext = (loadGPCC(grid=grd, varlist=varlist, lautoregrid=lautoregrid), 
                 loadPRISM(grid=grd, varlist=varlist, lautoregrid=lautoregrid),
                 loadPCIC(grid=grd, varlist=varlist, lautoregrid=lautoregrid),)
          axt = 'PCIC PRISM (and GPCC)'
        else:
          from datasets.PRISM import loadPRISM 
          from datasets.CRU import loadCRU
          ext = (loadCRU(period='1971-2001', grid=grd, varlist=varlist, lautoregrid=lautoregrid),
                 loadPRISM(grid=grd, varlist=varlist, lautoregrid=lautoregrid), 
                 loadPCIC(grid=grd, varlist=varlist, lautoregrid=lautoregrid)) 
          axt = 'PCIC PRISM (and CRU)'
      else:
        ext = loadPCIC(grid=grd, varlist=varlist, lautoregrid=lautoregrid); axt = 'PCIC PRISM'
    elif exp == 'PRISM': # PRISM with some background field
      from datasets.PRISM import loadPRISM
      if lbackground:
        if all(var in ('precip','stations','lon2D','lat2D','landmask','landfrac') for var in varlist): 
          from datasets.GPCC import loadGPCC
          ext = (loadGPCC(grid=grd, varlist=varlist, lautoregrid=lautoregrid), 
                 loadPRISM(grid=grd, varlist=varlist, lautoregrid=lautoregrid),)
          axt = 'PRISM (and GPCC)'
        else: 
          from datasets.CRU import loadCRU
          ext = (loadCRU(period='1979-2009', grid=grd, varlist=varlist, lautoregrid=lautoregrid), 
                 loadPRISM(grid=grd, varlist=varlist, lautoregrid=lautoregrid)) 
          axt = 'PRISM (and CRU)'
      else:
        ext = loadPRISM(grid=grd, varlist=varlist, lautoregrid=lautoregrid); axt = 'PRISM'
    elif exp == 'CFSR': 
      from datasets.CFSR import loadCFSR
      ext = loadCFSR(period=prd, grid=grd, varlist=varlist, lautoregrid=lautoregrid)
      axt = 'CFSR Reanalysis' 
    elif exp == 'NARR': 
      from datasets.NARR import loadNARR
      ext = loadNARR(period=prd, grid=grd, varlist=varlist, lautoregrid=lautoregrid)
      axt = 'NARR Reanalysis'
    elif exp[-5:] == '_CVDP':
      from datasets.CESM import loadCVDP, loadCVDP_Obs
      # load data generated by CVDP
      exp = exp[:-5]
      if exp in CESM_exps: # CESM experiments/ensembles
        exp = CESM_exps[exp]
        ext = loadCVDP(experiment=exp, period=prd, grid=grd, varlist=varlist, 
                       lautoregrid=lautoregrid, exps=CESM_exps)        
      else: # try observations
        ext = loadCVDP_Obs(name=exp, grid=grd, varlist=varlist, lautoregrid=lautoregrid)
      axt = ext.title
    else: # all other uppercase names are CESM runs
      from datasets.CESM import loadCESM
      exp = CESM_exps[exp]
      #print exp.name, exp.title
      ext = loadCESM(experiment=exp, period=prd, grid=grd, varlist=varlist, filetypes=filetypes, 
                     lautoregrid=lautoregrid, exps=CESM_exps)
      axt = exp.title
  else: 
    # WRF runs are all in lower case
    from datasets.WRF import loadWRF
    exp = WRF_exps[exp]      
    parent = None
    if isinstance(dom,(list,tuple)):
      if 0 == dom[0]:
        dom = dom[1:]
        parent, tmp = loadDataset(exp.parent, prd, dom, grd, res, varlist=varlist, lbackground=False, 
                                  lautoregrid=lautoregrid, WRF_exps=WRF_exps, CESM_exps=CESM_exps); del tmp    
    if lWRFnative: grd = None
    ext = loadWRF(experiment=exp, period=prd, grid=grd, domains=dom, filetypes=filetypes, 
                  varlist=varlist, varatts=None, lautoregrid=lautoregrid, exps=WRF_exps)
    if parent is not None: ext = (parent,) + tuple(ext)
    axt = exp.title # defaults to name...
  # return values
  return ext, axt    


def checkItemList(itemlist, length, dtype, default=NotImplemented, iterable=False, trim=True):
  ''' return a list based on item and check type '''
  # N.B.: default=None is not possible, because None may be a valid default...
  if itemlist is None: itemlist = []
  if iterable:
    # if elements are lists or tuples etc.
    if not isinstance(itemlist,(list,tuple,set)): raise TypeError, str(itemlist)
    if not isinstance(default,(list,tuple,set)) and not default is None: # here the default has to be a list of items
      raise TypeError, "Default for iterable items needs to be iterable." 
    if len(itemlist) == 0: 
      itemlist = [default]*length # make default list
    elif all([not isinstance(item,(list,tuple,set)) for item in itemlist]):
      # list which is actually an item and needs to be put into a list of its own
      itemlist = [itemlist]*length
    else:
      # a list with (eventually) iterable elements     
      if trim:
        if len(itemlist) > length: del itemlist[length:]
        elif len(itemlist) < length: itemlist += [default]*(length-len(itemlist))
      else:
        if len(itemlist) == 1: itemlist *= length # extend to desired length
        elif len(itemlist) != length: 
          raise TypeError, "Item list {:s} must be of length {:d} or 1.".format(str(itemlist),len(itemlist))
      if dtype is not None:
        for item in itemlist:
          if item != default: # only checks the non-default values
            if not isinstance(itemlist,dtype):
              raise TypeError, "Item {:s} must be of type {:s}".format(str(item),dtype.__name__)
            # don't check length of sublists: that would cause problems with some code
    # type checking, but only the iterables which are items, not their items
    for item in itemlist: # check types 
      if item is not None and not isinstance(item,dtype): 
        raise TypeError, "Item {:s} must be of type {:s}".format(str(item),dtype.__name__)
  else:
    if isinstance(itemlist,(list,tuple,set)): # still want to exclude strings
      itemlist = list(itemlist)
      if default is NotImplemented: 
        if len(itemlist) > 0: default = itemlist[-1] # use last item
        else: default = None   
      if trim:
        if len(itemlist) > length: del itemlist[length:] # delete superflous items
        elif len(itemlist) < length:
          itemlist += [default]*(length-len(itemlist)) # extend with default or last item
      else:
        if len(itemlist) == 1: itemlist *= length # extend to desired length
        elif len(itemlist) != length: 
          raise TypeError, "Item list {:s} must be of length {:d} or 1.".format(str(itemlist),len(itemlist))    
      if dtype is not None:
        for item in itemlist:
          if not isinstance(item,dtype) and item != default: # only checks the non-default values
            raise TypeError, "Item {:s} must be of type {:s}".format(str(item),dtype.__name__)        
    else:
      if not isinstance(itemlist,dtype): 
        raise TypeError, "Item {:s} must be of type {:s}".format(str(itemlist),dtype.__name__)
      itemlist = [itemlist]*length
  return itemlist

    
# function to load a list of datasets/experiments based on names and other common parameters
def loadDatasets(explist, n=None, varlist=None, titles=None, periods=None, domains=None, grids=None,
                 resolutions='025', filetypes=None, lbackground=True, lWRFnative=True, ltuple=True, 
                 lautoregrid=False, WRF_exps=None, CESM_exps=None):
  ''' function to load a list of datasets/experiments based on names and other common parameters '''
  # for load function (below)
  if lbackground and not ltuple: raise ValueError
  # check and expand lists
  if n is None: n = len(explist)
  elif not isinstance(n, (int,np.integer)): raise TypeError
  explist = checkItemList(explist, n, (basestring,tuple))
  titles = checkItemList(titles, n, basestring, default=None)
  periods  = checkItemList(periods, n, (basestring,int,np.integer), default=None, iterable=False)
  if isinstance(domains,tuple): ltpl = ltuple
  else: ltpl = False # otherwise this causes problems with expanding this  
  domains  = checkItemList(domains, n, (int,np.integer,tuple), default=None, iterable=ltpl) # to return a tuple, give a tuple of domains
  grids  = checkItemList(grids, n, basestring, default=None)
  resolutions  = checkItemList(resolutions, n, basestring, default=None)
  # expand variable list
  varlists = []
  for i in xrange(len(explist)):
    vl = set()
    for var in varlist: 
      if isinstance(var,(tuple,list)): 
        if isinstance(var[i], dict): vl.update(var[i].values())
        else: vl.add(var[i])
      elif isinstance(var, dict): vl.update(var.values())
      else: vl.add(var)
    vl.update(('lon2D','lat2D','landmask','landfrac')) # landfrac is needed for CESM landmask
    varlists.append(vl)

  def addPeriodExt(exp, prd):
    if exp[-5:] not in ('-2050','-2100'):
      if prd[:5] in ('2045-','2050-'): exp = exp + '-2050'
-      elif prd[:5] in ('2085-','2090-'): exp = exp + '-2100' 
-    return exp

  # resolve experiment list
  print("Loading Datasets:")
  dslist = []; axtitles = []
  for exp,vl,tit,prd,dom,grd,res in zip(explist,varlists,titles,periods,domains,grids,resolutions): 
    if isinstance(exp,tuple):
      print("  - " + ','.join(exp))  
      if lbackground: raise ValueError, 'Adding Background is not supported in combination with experiment tuples!'
      if not isinstance(dom,(list,tuple)): dom =(dom,)*len(exp)
      if len(dom) != len(exp): raise ValueError, 'Only one domain is is not supported for each experiment!'          
      ext = []; axt = []        
      for ex,dm in zip(exp,dom):
        ex = addPeriodExt(ex, prd)
        et, at = loadDataset(ex, prd, dm, grd, res, filetypes=filetypes, varlist=vl, 
                             lbackground=False, lWRFnative=lWRFnative, lautoregrid=lautoregrid,
                             WRF_exps=WRF_exps, CESM_exps=CESM_exps)
        for var in vl: 
          if var not in et and var not in ('lon2D','lat2D','landmask','landfrac'): 
            raise DatasetError, "Variable '{:s}' not found in Dataset '{:s}!".format(var,et.name) 
        #if isinstance(et,(list,tuple)): ext += list(et); else: 
        ext.append(et)
        #if isinstance(at,(list,tuple)): axt += list(at); else: 
        axt.append(at)
      ext = tuple(ext); axt = tuple(axt)
    else:
      print("  - " + exp)
      exp = addPeriodExt(exp, prd)
      ext, axt = loadDataset(exp, prd, dom, grd, res, filetypes=filetypes, varlist=vl, 
                             lbackground=lbackground, lWRFnative=lWRFnative, lautoregrid=lautoregrid,
                             WRF_exps=WRF_exps, CESM_exps=CESM_exps)
      for exp in ext if isinstance(ext,(tuple,list)) else (ext,):
        for var in vl: 
          if var not in exp and var not in ('lon2D','lat2D','landmask','landfrac'): 
            print var, exp
            raise DatasetError, "Variable '{:s}' not found in Dataset '{:s}'!".format(var,exp.name)         
    dslist.append(ext) 
    if tit is not None: axtitles.append(tit)
    else: axtitles.append(axt)
  # count experiment tuples (layers per panel)
  if ltuple:
    nlist = [] # list of length for each element (tuple)
    for n in xrange(len(dslist)):
      if not isinstance(dslist[n],(tuple,list)): # should not be necessary
        dslist[n] = (dslist[n],)
      elif isinstance(dslist[n],list): # should not be necessary
        dslist[n] = tuple(dslist[n])
      nlist.append(len(dslist[n])) # layer counter for each panel  
  # return list with datasets and plot titles
  if ltuple:
    return dslist, axtitles, nlist
  else:
    return dslist, axtitles
  

## more legacy functions for plotting

# function to expand level lists and colorbar ticks
def expandLevelList(levels, data=None):  
  ''' figure out level list based on level parameters and actual data '''
  # trivial case: already numpy array
  if isinstance(levels,np.ndarray):
    return levels 
  # tuple with three or two elements: use as argument to linspace 
  elif isinstance(levels,tuple) and (len(levels)==3 or len(levels)==2):
    return np.linspace(*levels)
  # list or long tuple: recast as array
  elif isinstance(levels,(list,tuple)):
    return np.asarray(levels)
  # use additional info in data to determine limits
  else:
    # figure out vector limits
    # use first two elements, third is number of levels
    if isinstance(data,(tuple,list)) and len(data)==3:  
      minVec = min(data[:2]); maxVec = max(data[:2])
    # just treat as level list
    else: 
      minVec = min(data); maxVec = max(data)
    # interpret levels as number of levels in given interval
    # only one element: just number of levels
    if isinstance(levels,(tuple,list,np.ndarray)) and len(levels)==1: 
      return np.linspace(minVec,maxVec,levels[0])
    # numerical value: use as number of levels
    elif isinstance(levels,(int,float)):
      return np.linspace(minVec,maxVec,levels)        


# load matplotlib with some custom defaults
def loadMPL(linewidth=None, mplrc=None, backend='QT4Agg', lion=False):
  import matplotlib as mpl
  mpl.use(backend) # enforce QT4
  import matplotlib.pylab as pyl
  # some custom defaults  
  if linewidth is not None:
    mpl.rc('lines', linewidth=linewidth)
    if linewidth == 1.5: mpl.rc('font', size=12)
    elif linewidth == .75: mpl.rc('font', size=8)
    else: mpl.rc('font', size=10)
  # apply rc-parameters from dictionary (override custom defaults)
  if (mplrc is not None) and isinstance(mplrc,dict):
    # loop over parameter groups
    for (key,value) in mplrc.iteritems():
      mpl.rc(key,**value)  # apply parameters
  # prevent figures from closing: don't run in interactive mode, or pyl.show() will not block
  if lion: pyl.ion()
  else: pyl.ioff()
  # return matplotlib instance with new parameters
  return mpl, pyl


# method to return a figure and an array of ImageGrid axes
def getFigAx(subplot, name=None, title=None, figsize=None,  mpl=None, margins=None,
             sharex=None, sharey=None, AxesGrid=False, ngrids=None, direction='row',
             axes_pad = None, add_all=True, share_all=None, aspect=False,
             label_mode='L', cbar_mode=None, cbar_location='right',
             cbar_pad=None, cbar_size='5%', axes_class=None, lreduce=True): 
  # configure matplotlib
  warn('Deprecated function: use Figure or Axes class methods.')
  if mpl is None: import matplotlib as mpl
  elif isinstance(mpl,dict): mpl = loadMPL(**mpl) # there can be a mplrc, but also others
  elif not isinstance(mpl,ModuleType): raise TypeError
  from plotting.figure import MyFigure # prevent circular reference
  # figure out subplots
  if isinstance(subplot,(np.integer,int)):
    if subplot == 1: subplot = (1,1)
    elif subplot == 2: subplot = (1,2)
    elif subplot == 3: subplot = (1,3)
    elif subplot == 4: subplot = (2,2)
    elif subplot == 6: subplot = (2,3)
    elif subplot == 9: subplot = (3,3)
    else: raise NotImplementedError
  elif not (isinstance(subplot,(tuple,list)) and len(subplot) == 2) and all(isInt(subplot)): raise TypeError    
  # create figure
  if figsize is None: 
    if subplot == (1,1): figsize = (3.75,3.75)
    elif subplot == (1,2) or subplot == (1,3): figsize = (6.25,3.75)
    elif subplot == (2,1) or subplot == (3,1): figsize = (3.75,6.25)
    else: figsize = (6.25,6.25)
    #elif subplot == (2,2) or subplot == (3,3): figsize = (6.25,6.25)
    #else: raise NotImplementedError
  # figure out margins
  if margins is None:
    # N.B.: the rectangle definition is presumably left, bottom, width, height
    if subplot == (1,1): margins = (0.09,0.09,0.88,0.88)
    elif subplot == (1,2) or subplot == (1,3): margins = (0.06,0.1,0.92,0.87)
    elif subplot == (2,1) or subplot == (3,1): margins = (0.09,0.11,0.88,0.82)
    elif subplot == (2,2) or subplot == (3,3): margins = (0.055,0.055,0.925,0.925)
    else: margins = (0.09,0.11,0.88,0.82)
    #elif subplot == (2,2) or subplot == (3,3): margins = (0.09,0.11,0.88,0.82)
    #else: raise NotImplementedError    
    if title is not None: margins = margins[:3]+(margins[3]-0.03,) # make room for title
  if AxesGrid:
    if share_all is None: share_all = True
    if axes_pad is None: axes_pad = 0.05
    # create axes using the Axes Grid package
    fig = mpl.pylab.figure(facecolor='white', figsize=figsize, FigureClass=MyFigure)
    if axes_class is None:
      from plotting.axes import MyLocatableAxes  
      axes_class=(MyLocatableAxes,{})
    from mpl_toolkits.axes_grid1 import ImageGrid
    # AxesGrid: http://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html
    grid = ImageGrid(fig, margins, nrows_ncols = subplot, ngrids=ngrids, direction=direction, 
                     axes_pad=axes_pad, add_all=add_all, share_all=share_all, aspect=aspect, 
                     label_mode=label_mode, cbar_mode=cbar_mode, cbar_location=cbar_location, 
                     cbar_pad=cbar_pad, cbar_size=cbar_size, axes_class=axes_class)
    # return figure and axes
    axes = tuple([ax for ax in grid]) # this is already flattened
    if lreduce and len(axes) == 1: axes = axes[0] # return a bare axes instance, if there is only one axes    
  else:
    # create axes using normal subplot routine
    if axes_pad is None: axes_pad = 0.03
    wspace = hspace = axes_pad
    if share_all: 
      sharex='all'; sharey='all'
    if sharex is True or sharex is None: sharex = 'col' # default
    if sharey is True or sharey is None: sharey = 'row'
    if sharex: hspace -= 0.015
    if sharey: wspace -= 0.015
    # create figure
    from matplotlib.pyplot import subplots    
    # GridSpec: http://matplotlib.org/users/gridspec.html 
    fig, axes = subplots(subplot[0], subplot[1], sharex=sharex, sharey=sharey,
                         squeeze=lreduce, facecolor='white', figsize=figsize, FigureClass=MyFigure)    
    # there is also a subplot_kw=dict() and fig_kw=dict()
    # just adjust margins
    margin_dict = dict(left=margins[0], bottom=margins[1], right=margins[0]+margins[2], 
                       top=margins[1]+margins[3], wspace=wspace, hspace=hspace)
    fig.subplots_adjust(**margin_dict)
  # add figure title
  if name is not None: fig.canvas.set_window_title(name) # window title
  if title is not None: fig.suptitle(title) # title on figure (printable)
  # return Figure/ImageGrid and tuple of axes
  #if AxesGrid: fig = grid # return ImageGrid instead of figure
  return fig, axes


# function to adjust subplot parameters
def updateSubplots(fig, mode='shift', **kwargs):
  ''' simple helper function to move (relocate), shift, or scale subplot margins '''
  warn('Deprecated function: use Figure or Axes class methods.')
  pos = fig.subplotpars
  margins = dict() # original plot margins
  margins['left'] = pos.left; margins['right'] = pos.right 
  margins['top'] = pos.top; margins['bottom'] = pos.bottom
  margins['wspace'] = pos.wspace; margins['hspace'] = pos.hspace
  # update subplot margins
  if mode == 'move': margins.update(kwargs)
  else: 
    for key,val in kwargs.iteritems():
      if key in margins:
        if mode == 'shift': margins[key] += val
        elif mode == 'scale': margins[key] *= val
  # finally, actually update figure
  fig.subplots_adjust(**margins)
  # and now repair damage: restore axes
  for ax in fig.axes:
    if ax.get_title():
      pos = ax.get_position()
      pos = pos.from_bounds(x0=pos.x0, y0=pos.y0, width=pos.width, height=pos.height-0.03)    
      ax.set_position(pos)


# add subplot/axes label
def addLabel(ax, label=None, loc=1, stroke=False, size=None, prop=None, **kwargs):
  from matplotlib.offsetbox import AnchoredText 
  from matplotlib.patheffects import withStroke
  from string import lowercase
  warn('Deprecated function: use Figure or Axes class methods.')    
  # expand list
  if not isinstance(ax,(list,tuple)): ax = [ax] 
  l = len(ax)
  if not isinstance(label,(list,tuple)): label = [label]*l
  if not isinstance(loc,(list,tuple)): loc = [loc]*l
  if not isinstance(stroke,(list,tuple)): stroke = [stroke]*l
  # settings
  if prop is None:
    prop = dict()
  if not size: prop['size'] = 18
  args = dict(pad=0., borderpad=1.5, frameon=False)
  args.update(kwargs)
  # cycle over axes
  at = [] # list of texts
  for i in xrange(l):
    if label[i] is None:
      label[i] = '('+lowercase[i]+')'
    elif isinstance(label[i],int):
      label[i] = '('+lowercase[label[i]]+')'
    # create label    
    at.append(AnchoredText(label[i], loc=loc[i], prop=prop, **args))
    ax[i].add_artist(at[i]) # add to axes
    if stroke[i]: 
      at[i].txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
  return at


# function to place (shared) colorbars at a specified figure margins
def sharedColorbar(fig, cf, clevs, colorbar, cbls, subplot, margins):
  loc = colorbar.pop('location','bottom')      
  # determine size and spacing
  if loc=='top' or loc=='bottom':
    orient = colorbar.pop('orientation','horizontal') # colorbar orientation
    je = subplot[1] # number of colorbars: number of rows
    ie = subplot[0] # number of plots per colorbar: number of columns
    cbwd = colorbar.pop('cbwd',0.025) # colorbar height
    sp = margins['wspace']
    wd = (margins['right']-margins['left'] - sp*(je-1))/je # width of each colorbar axis 
  else:
    orient = colorbar.pop('orientation','vertical') # colorbar orientation
    je = subplot[0] # number of colorbars: number of columns
    ie = subplot[1] # number of plots per colorbar: number of rows
    cbwd = colorbar.pop('cbwd',0.025) # colorbar width
    sp = margins['hspace']
    wd = (margins['top']-margins['bottom'] - sp*(je-1))/je # width of each colorbar axis
  shrink = colorbar.pop('shrinkFactor',1)
  # shift existing subplots
  if loc=='top': newMargin = margins['top']-margins['hspace'] -cbwd
  elif loc=='right': newMargin = margins['right']-margins['left']/2 -cbwd
  else: newMargin = 2*margins[loc] + cbwd    
  fig.subplots_adjust(**{loc:newMargin})
  # loop over variables (one colorbar for each)
  for i in range(je):
    if dir=='vertical': ii = je-i-1
    else: ii = i
    offset = (wd+sp)*float(ii) + wd*(1-shrink)/2 # offset due to previous colorbars
    # horizontal colorbar(s) at the top
    if loc == 'top': ci = i; cax = [margins['left']+offset, newMargin+margins['hspace'], shrink*wd, cbwd]             
    # horizontal colorbar(s) at the bottom
    elif loc == 'bottom': ci = i; cax = [margins['left']+offset, margins[loc], shrink*wd, cbwd]        
    # vertical colorbar(s) to the left (get axes reference right!)
    elif loc == 'left': ci = i*ie; cax = [margins[loc], margins['bottom']+offset, cbwd, shrink*wd]        
    # vertical colorbar(s) to the right (get axes reference right!)
    elif loc == 'right': ci = i*ie; cax = [newMargin+margins['wspace'], margins['bottom']+offset, cbwd, shrink*wd]
    # make colorbar 
    fig.colorbar(mappable=cf[ci],cax=fig.add_axes(cax),ticks=expandLevelList(cbls[i],clevs[i]),
                 orientation=orient,**colorbar)
  # return figure with colorbar (just for the sake of returning something) 
  return fig
