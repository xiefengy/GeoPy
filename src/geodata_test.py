'''
Created on 2013-08-24 

Unittest for the PyGeoDat main package geodata.

@author: Andre R. Erler, GPL v3
'''

import unittest
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import os

# import modules to be tested
from geodata.nctools import writeNetCDF
from geodata.misc import isZero, isOne, isEqual
from geodata.base import Variable, Axis, Dataset, Ensemble, concatVars, concatDatasets
from datasets.common import data_root

class BaseVarTest(unittest.TestCase):  
  
  # some test parameters (TestCase does not take any arguments)  
  plot = False # whether or not to display plots 
  stats = False # whether or not to compute stats on data
  
  def setUp(self):
    ''' create Axis and a Variable instance for testing '''
    self.dataset_name = 'TEST'
    # some setting that will be saved for comparison
    self.size = (48,2,4) # size of the data array and axes
    # the 4-year time-axis is for testing some time-series analysis functions
    te, ye, xe = self.size
    self.atts = dict(name = 'test',units = 'n/a',FillValue=-9999)
    data = np.arange(self.size[0], dtype='int8').reshape(self.size[:1]+(1,))%12 +1
    data = data.repeat(np.prod(self.size[1:]),axis=1,).reshape(self.size)
    #print data
    self.data = data
    # create axis instances
    t = Axis(name='time', units='month', coord=(1,te,te))
    y = Axis(name='y', units='none', coord=(1,ye,ye))
    x = Axis(name='x', units='none', coord=(1,xe,xe))
    self.axes = (t,y,x)
    # create axis and variable instances (make *copies* of data and attributes!)
    self.var = Variable(name=self.atts['name'],units=self.atts['units'],axes=self.axes,
                        data=self.data.copy(),atts=self.atts.copy())
    self.rav = Variable(name=self.atts['name'],units=self.atts['units'],axes=self.axes,
                        data=self.data.copy(),atts=self.atts.copy())
    # check if data is loaded (future subclasses may initialize without loading data by default)                  
    if not self.var.data: self.var.load(self.data.copy()) # again, use copy!
    if not self.rav.data: self.rav.load(self.data.copy()) # again, use copy!
        
  def tearDown(self):
    ''' clean up '''     
    self.var.unload() # just to do something... free memory
    self.rav.unload()
    
  ## basic tests every variable class should pass

  def testAttributes(self):
    ''' test handling of attributes '''
    # get test objects
    var = self.var; atts = self.atts
    # test getattr
    assert (atts['name'],atts['units']) == (var.name,var.units)
    # test setattr
    var.atts.comments = 'test'; var.plot['comments'] = 'test'
    assert var.plot.comments == var.atts['comments']      
    #     print 'Name: %s, Units: %s, Missing Values: %s'%(var.name, var.units, var._FillValue)
    #     print 'Comments: %s, Plot Comments: %s'%(var.Comments,var.plotatts['plotComments'])
    
  def testAxis(self):
    ''' test stuff related to axes '''
    # get test objects
    var = self.var
    # test contains 
    for ax,n in zip(self.axes,self.size):
      assert ax in var.axes
      assert len(ax) == n
    #if ax in var: print '%s is the %i. axis and has length %i'%(ax.name,var[ax]+1,len(ax))
    # replace axis
    oldax = var.axes[-1]    
    newax = Axis(name='z', units='none', coord=(1,len(oldax),len(oldax)))
    var.replaceAxis(oldax,newax)
    assert var.hasAxis(newax) and not var.hasAxis(oldax)
  
  def testBinaryArithmetic(self):
    ''' test binary arithmetic functions '''
    # get test objects
    var = self.var
    rav = self.rav
    # arithmetic test
    a = var + rav
    assert isEqual(self.data*2, a.data_array)
    s = var - rav
    assert isZero(s.data_array)
    m = var * rav
    assert isEqual(self.data**2, m.data_array)
    if (rav.data_array == 0).any(): # can't divide by zero!
      if (rav.data_array != 0).any():  # test masking: mask zeros
        rav.mask(np.logical_not(rav.data_array), fillValue=rav.fillValue, merge=True)
      else: raise TypeError, 'Cannot divide by all-zero field!' 
    d = var / rav
    assert isOne(d.data_array)
    # test results
    #     print (self.data.filled() - var.data_array.filled()).max()
#     assert isEqual(np.ones_like(self.data), d.data_array)
#     assert isOne(d.data_array)  
  
  def testBroadcast(self):
    ''' test reordering, reshaping, and broadcasting '''
    # get test objects
    var = self.var
    z = Axis(name='z', units='none', coord=(1,5,5)) # new axis    
    new_axes = var.axes[0:1] + (z,) + var.axes[-1:0:-1] # dataset independent
    new_axes_names = tuple([ax.name for ax in new_axes])
    # test reordering and reshaping/extending (using axis names)
    new_shape = tuple([var.shape[var.axisIndex(ax)] if var.hasAxis(ax) else 1 for ax in new_axes]) 
    data = var.getArray(axes=new_axes_names, broadcast=False, copy=True)
    #print var.shape # this is what it was
    #print data.shape # this is what it is
    #print new_shape 
    assert data.shape == new_shape 
    # test broadcasting to a new shape (using Axis instances) 
    new_shape = tuple([len(ax) for ax in new_axes]) # this is the shape we should get
    data = var.getArray(axes=new_axes, broadcast=True, copy=True)
    #print var.shape # this is what it was
    #print data.shape # this is what it is
    #print new_shape # this is what it should be
    assert data.shape == new_shape 
    
  def testConcatVars(self):
    ''' test concatenation of variables '''
    # get copy of variable
    var = self.var
    copy = self.var.copy()
    lckax = self.dataset_name not in ('GPCC','NARR') # will fail with GPCC and NARR, due to sub-monthly time units
    # simple test
    concat_data = concatVars([var,copy], axis='time', asVar=False, lcheckAxis=lckax)
    # N.B.: some datasets have tiem units in days or hours, which is not uniform 
    shape = list(var.shape); 
    tax = var.axisIndex('time')
    shape[tax] = var.shape[tax] + copy.shape[tax]
    assert concat_data.shape == tuple(shape)
    # advanced test
    concat_var = concatVars([var,copy], axis='time', asVar=True, lcheckAxis=lckax, 
                            idxlim=(0,12), offset=1000, name='concatVar')
    shape[tax] = 2 * 12
    assert concat_var.shape == tuple(shape)
    assert len(concat_var.time) == 24 and max(concat_var.time.coord) > 1000
    assert concat_var.name == 'concatVar'
    assert isEqual(concat_var[:].take(xrange(12)),concat_data.take(xrange(12)))
    tlen = var.shape[tax]
    assert isEqual(concat_var[:].take(xrange(12,24), axis=tax),concat_data.take(xrange(tlen,tlen+12), axis=tax))    
        
  def testCopy(self):
    ''' test copy and deepcopy of variables (and axes) '''
    # get copy of variable
    var = self.var.deepcopy(name='different') # deepcopy calls copy
    # check identity
    assert var != self.var
    assert var.name == 'different' and self.var.name != 'different'      
    assert (var.units == self.var.units) and (var.units is self.var.units) # strings are immutable...
    assert (var.atts is not self.var.atts) and (var.atts != self.var.atts) # ...dictionaries are not
    # N.B.: note that due to the name change, their atts are different!
    for key,value in var.atts.iteritems():
      if key == 'name': assert np.any(value != self.var.atts[key]) 
      else: assert np.all(value == self.var.atts[key])
    assert isEqual(var.data_array,self.var.data_array) 
    # change array
    var.data_array += 1 # test if we have a true copy and not just a reference 
    assert not isEqual(var.data_array,self.var.data_array)

  def testEnsemble(self):
    ''' test the Ensemble container class '''
    # test object
    var = self.var
    # make a copy
    copy = var.copy()
    copy.name = 'copy of {}'.format(var.name)
    yacov = var.copy()
    yacov.name = 'yacod' # used later    
    # instantiate ensemble
    ens = Ensemble(var, copy, name='ensemble', title='Test Ensemble')
    # basic functionality
    assert len(ens.members) == len(ens)
    # these var/ax names are specific to the test dataset...
    if all(ens.hasAxis('time')):
      print ens.time 
      assert ens.time == [var.time , copy.time]
    # collective add/remove
    # test adding a new member
    ens += yacov # this is an ensemble operation
    print('')
    print(ens)
    print('')    
    ens -= yacov # this is a dataset operation
    assert not ens.hasMember(yacov)
    # perform a variable operation
    ens.mean(axis='time')
    print(ens.prettyPrint(short=True))
    ens -= 'test' # subtract by name
    print('')
    print(ens)
    print('')    
    assert not ens.hasMember('test')
      
  def testIndexing(self):
    ''' test indexing and slicing '''
    # get test objects
    var = self.var
    # indexing (getitem) test  
    if var.ndim >= 3:  
      # standard indexing
      assert isEqual(self.data[0,1,1], var[0,1,1], masked_equal=True)
      assert isEqual(self.data[0,:,1:-1], var[0,:,1:-1], masked_equal=True)
      # range and value indexing
      ax0 = var.axes[0]; ax1 = var.axes[1]; ax2 = var.axes[2]
      co0 = ax0.coord; co1 = ax1.coord; co2 = ax2.coord
      axes = {ax2.name:(co2[1],co2[-1]), ax0.name:co0[-1]}
      slcvar = var(**axes)
      assert slcvar.ndim == var.ndim-1
      assert slcvar.shape == (var.shape[1],var.shape[2]-1)
      for slcax,ax in zip(slcvar.axes,var.axes[1:]):
        assert slcax.name == ax.name
        assert slcax.units == ax.units
      assert isEqual(slcvar[:], var[-1,:,1:], masked_equal=True)
      # list indexing
      l0 = [0,-1]*3; l1 = [-1,0]*3; l2 = [-1,0]*3 
      axes = {ax0.name:(co0[1],co0[-1]), ax1.name:co1[l1], ax2.name:co2[l2], }
      slcvar = var(**axes)
      assert slcvar.ndim == 2
      assert len(slcvar.axes[0]) == var.shape[0]-1
      assert len(slcvar.axes[1]) == len(l0) 
      assert isEqual(slcvar[:], var[1:,l1,l2], masked_equal=True)
      # integer index indexing
      axes = {ax0.name:(1,-1), ax1.name:l1, ax2.name:l2}
      slcvar = var(lidx=True, **axes)
      assert slcvar.ndim == 2
      assert len(slcvar.axes[0]) == var.shape[0]-1
      assert len(slcvar.axes[1]) == len(l0) 
      assert isEqual(slcvar[:], var[1:,l1,l2], masked_equal=True)
      
  
  def testLoad(self):
    ''' test data loading and unloading '''
    # get test objects
    var = self.var
    # unload and load test
    var.unload()
    var.load(self.data.copy())
    assert self.size == var.shape
    assert isEqual(self.data, var.data_array)
    
  def testMask(self):
    ''' test masking and unmasking of data '''
    # get test objects
    var = self.var; rav = self.rav
    masked = var.masked
    mask = var.getMask()
    data = var.getArray(unmask=True, fillValue=-9999)
    # test unmasking and masking again
    var.unmask(fillValue=-9999)
    assert isEqual(data, var[:]) # trivial
    var.mask(mask=mask)
    assert isEqual(self.data, var.getArray(unmask=(not masked)))
    # test masking with a variable
    var.unmask(fillValue=-9999)
    assert isEqual(data, var[:]) # trivial
    var.mask(mask=rav.data_array> 6)
    #print ma.array(self.data,mask=(rav.data_array>0)), var.getArray(unmask=False)
    assert isEqual(ma.array(self.data,mask=(rav.data_array>6)), var.getArray(unmask=False)) 
    
  def testPrint(self):
    ''' just print the string representation '''
    assert self.var.prettyPrint()
    print('')
    s = str(self.var)
    print s
    print('')
    
  def testReductionArithmetic(self):
    ''' test reducing arithmetic functions '''
    # get test objects
    var = self.var; t,x,y = self.axes # for upwards compatibility!
#     print self.data.mean(), var.mean()
    print self.data.std(ddof=3), var.std(ddof=3)
    assert isEqual(self.data.mean(), var.mean())
    assert isEqual(self.data.std(ddof=1), var.std(ddof=1))
    assert isEqual(self.data.max(), var.max())
    assert isEqual(self.data.min(), var.min())
    assert isEqual(self.data.mean(axis=var.axisIndex(t.name)), var.mean(**{t.name:None}).getArray())
    assert isEqual(self.data.std(axis=var.axisIndex(t.name),ddof=3), var.std(ddof=3, **{t.name:None}).getArray())
    assert isEqual(self.data.max(axis=var.axisIndex(x.name)), var.max(**{x.name:None}).getArray())
    assert isEqual(self.data.min(axis=var.axisIndex(y.name)), var.min(**{y.name:None}).getArray())
    
  def testSeasonalReduction(self):
    ''' test functions that reduce monthly data to yearly data '''
    # get test objects
    var = self.var
    assert var.axisIndex('time') == 0 and len(var.time) == self.data.shape[0]
    assert len(var.time)%12 == 0, "Need full years to test seasonal mean/min/max!"
    tax = var.axisIndex('time')
    #print self.data.mean(), var.mean().getArray()
    if var.time.units.lower()[:5] in 'month':
      yvar = var.seasonalMean('jj', asVar=True)
      assert yvar.hasAxis('year')
      assert yvar.shape == var.shape[:tax]+(var.shape[0]/12,)+var.shape[tax+1:]
      cvar = var.climMean()
      assert len(cvar.getAxis('time')) == 12
      assert cvar.shape == var.shape[:tax]+(12,)+var.shape[tax+1:]      
    if self.__class__ is BaseVarTest:
      # this only works with a specially prepared data field
      yfake = np.ones((var.shape[0]/12,)+var.shape[1:])
      assert yvar.shape == yfake.shape
      assert isEqual(yvar.getArray(), yfake*6.5)
      yfake = np.ones((var.shape[0]/12,)+var.shape[1:], dtype=var.dtype)
      # N.B.: the data increases linearly in time and is constant in space (see setup fct.)
      assert isEqual(var.seasonalMax('mam',asVar=False), yfake*5)
      assert isEqual(var.seasonalMin('mam',asVar=False), yfake*3)
      # test climatology
      assert tax == 0      
      cdata = self.data.reshape((4,12,)+var.shape[1:]).mean(axis=0)
      assert isEqual(cvar.getArray(), cdata)

  def testSqueeze(self):
    ''' test removal of singleton dimensions '''
    var = self.var
    ndim = var.ndim
    sdim = 0
    for dim in var.shape: 
      if dim == 1: sdim += 1
    # squeeze
    var.squeeze()
    # test
    assert var.ndim == ndim - sdim
    assert all([dim > 1 for dim in var.shape]) 
    
  def testUnaryArithmetic(self):
    ''' test unary arithmetic functions '''
    # get test objects
    var = self.var
    # arithmetic test
    var += 2.
    var -= 2.
    var *= 2.
    var /= 2.
    # test results
    #     print (self.data.filled() - var.data_array.filled()).max()
    assert isEqual(self.data, var.data_array)  
    

class BaseDatasetTest(unittest.TestCase):  
  
  # some test parameters (TestCase does not take any arguments)
  plot = False # whether or not to display plots 
  stats = False # whether or not to compute stats on data
  
  def setUp(self):
    ''' create Dataset with Axes and a Variables for testing '''
    if RAM: self.folder = ramdisk
    else: self.folder = os.path.expanduser('~') # just use home directory (will be removed)
    self.dataset_name = 'TEST'
    # some setting that will be saved for comparison
    self.size = (3,3,3) # size of the data array and axes
    te, ye, xe = self.size
    self.atts = dict(name = 'var',units = 'n/a',FillValue=-9999)
    self.data = np.random.random(self.size)   
    # create axis instances
    t = Axis(name='time', units='none', coord=(1,te,te))
    y = Axis(name='y', units='none', coord=(1,ye,ye))
    x = Axis(name='x', units='none', coord=(1,xe,xe))
    self.axes = (t,y,x)
    # create axis and variable instances (make *copies* of data and attributes!)
    var = Variable(name='var',units=self.atts['units'],axes=self.axes,
                        data=self.data.copy(),atts=self.atts.copy())
    lar = Variable(name='lar',units=self.atts['units'],axes=self.axes[1:],
                        data=self.data[0,:].copy(),atts=self.atts.copy())    
    rav = Variable(name='rav',units=self.atts['units'],axes=self.axes,
                        data=self.data.copy(),atts=self.atts.copy())
    self.var = var; self.lar =lar; self.rav = rav 
    # make dataset
    self.dataset = Dataset(varlist=[var, lar, rav], name='test')
    # check if data is loaded (future subclasses may initialize without loading data by default)
    if not self.var.data: self.var.load(self.data.copy()) # again, use copy!
    if not self.rav.data: self.rav.load(self.data.copy()) # again, use copy!
        
  def tearDown(self):
    ''' clean up '''     
    self.var.unload() # just to do something... free memory
    self.rav.unload()
    
  ## basic tests every variable class should pass

  def testAddRemove(self):
    ''' test adding and removing variables '''
    # test objects: var and ax
    name='test'
    ax = Axis(name='ax', units='none')
    var = Variable(name=name,units='none',axes=(ax,))
    dataset = self.dataset
    le = len(dataset)
    # add/remove axes
    dataset.addVariable(var, copy=False) # add variables as is
    assert dataset.hasVariable(var)
    assert dataset.hasAxis(ax)
    assert len(dataset) == le + 1
    dataset.removeAxis(ax) # should not work now
    assert dataset.hasAxis(ax)    
    dataset.removeVariable(var)
    assert not dataset.hasVariable(name)
    assert len(dataset) == le
    dataset.removeAxis(ax)
    assert not dataset.hasAxis(ax)
    # replace variable
    oldvar = dataset.variables.values()[-1]
    newvar = Variable(name='another_test', units='none', axes=oldvar.axes, data=np.zeros_like(oldvar.getArray()))
#     print oldvar.name, oldvar.data
#     print oldvar.shape    
#     print newvar.name, newvar.data
#     print newvar.shape
    dataset.replaceVariable(oldvar,newvar)
    print dataset
    assert dataset.hasVariable(newvar, strict=False)
    assert not dataset.hasVariable(oldvar, strict=False)  
    # replace axis
    oldax = dataset.axes.values()[-1]
    newax = Axis(name='z', units='none', coord=(1,len(oldax),len(oldax)))
#     print oldax.name, oldax.data
#     print oldax.data_array    
#     print newax.name, newax.data
#     print newax.data_array
    dataset.replaceAxis(oldax,newax)
    assert dataset.hasAxis(newax) and not dataset.hasAxis(oldax)  
    assert not any([var.hasAxis(oldax) for var in dataset])
    
  def testConcatDatasets(self):
    ''' test concatenation of datasets '''
    # get copy of dataset
    self.dataset.load() # need to load first!
    ds = self.dataset
    cp = self.dataset.copy()
    nocat = self.lar
    if nocat is not None: ncname = nocat.name
    catvar = self.var
    varname = catvar.name
    catax = self.axes[0]
    axname = catax.name
    lckax = self.dataset_name not in ('GPCC','NARR') # will fail with GPCC and NARR, due to sub-monthly time units
    # generate test data
    concat_data = concatVars([ds[varname],cp[varname]], axis=catax, asVar=False, lcheckAxis=lckax) # should be time
    shape = list(catvar.shape); 
    shape[0] = catvar.shape[0]*2
    shape = tuple(shape)
    assert concat_data.shape == shape # this just tests concatVars
    # simple test
    ccds = concatDatasets([ds, cp], axis=axname, coordlim=None, idxlim=None, offset=0, lcheckAxis=lckax)
    print ccds
    ccvar = ccds[varname] # test concatenated variable 
    assert ccvar.shape == shape
    assert isEqual(ccvar.data_array, concat_data) # masked_equal = True
    if nocat is not None: 
      ccnc = ccds[ncname] # test other variable (should be the same) 
      assert ccnc.shape == nocat.shape
    
  def testContainer(self):
    ''' test basic container functionality '''
    # test objects: vars and axes
    dataset = self.dataset
    # check container properties 
    assert len(dataset.variables) == len(dataset)
    for varname,varobj in dataset.variables.iteritems():
      assert varname in dataset
      assert varobj in dataset
    # test get, del, set
    varname = dataset.variables.keys()[0]
    var = dataset[varname]
    assert isinstance(var,Variable) and var.name == varname
    del dataset[varname]
    assert not dataset.hasVariable(varname)
    dataset[varname] = var
    assert dataset.hasVariable(varname)
    
  def testCopy(self):
    ''' test copying the entire dataset '''
    # test object
    dataset = self.dataset
    # make a copy
    copy = dataset.copy()
    copy.name = 'copy of {}'.format(dataset.name)
    # test
    assert copy is not dataset # should not be the same
    assert isinstance(copy,Dataset) and not isinstance(copy,DatasetNetCDF)
    assert all([copy.hasAxis(ax.name) for ax in dataset.axes.values()])
    assert all([copy.hasVariable(var.name) for var in dataset.variables.values()])

  def testEnsemble(self):
    ''' test the Ensemble container class '''
    # test object
    dataset = self.dataset
    dataset.load()
    # make a copy
    copy = dataset.copy()
    copy.name = 'copy of {}'.format(dataset.name)
    yacod = dataset.copy()
    yacod.name = 'yacod' # used later    
    # instantiate ensemble
    ens = Ensemble(dataset, copy, name='ensemble', title='Test Ensemble')
    # basic functionality
    assert len(ens.members) == len(ens)
    # these var/ax names are specific to the test dataset...
    if all(ens.hasVariable('var')):      
      assert isinstance(ens.var,Ensemble) and ens.var.basetype == Variable
      #assert ens.var == Ensemble(dataset.var, copy.var, basetype=Variable, idkey='dataset_name')
      assert ens.var.members == [dataset.var, copy.var]
      #print ens.var
      #print Ensemble(dataset.var, copy.var, basetype=Variable, idkey='dataset_name')
    #print(''); print(ens); print('')        
    #print ens.time
    assert ens.time == [dataset.time , copy.time]
    # Axis ensembles are not supported anymore, since they are often shared.
    #assert isinstance(ens.time,Ensemble) and ens.time.basetype == Variable
    # collective add/remove
    ax = Axis(name='ax', units='none')
    var = Variable(name='new',units='none',axes=(ax,))
    ens += var # this is a dataset operation
    assert all(ens.hasVariable('new'))
    # test adding a new member
    ens += yacod # this is an ensemble operation
    #print(''); print(ens); print('')    
    ens -= var # this is a dataset operation
    assert not any(ens.hasVariable('new'))
    ens -= 'test'
    # fancy test of Variable and Dataset integration
#     print ens[self.var.name][0]
#     print ens[self.var.name].mean(axis='time')
    assert not any(ens[self.var.name].mean(axis='time').hasAxis('time'))
    print(ens.prettyPrint(short=True))

  def testIndexing(self):
    ''' test collective slicing and coordinate/point extraction  '''
    # get test objects
    self.dataset.load() # sometimes we just need to load
    dataset = self.dataset
    # select variables
    var2 = self.lar; var3 = self.var
    if len(dataset.axes) == 3:
      # get axis that is not in var2 first
      ax0, ax1, ax2 = var3.axes
      co0 = ax0.coord; co1 = ax1.coord; co2 = ax2.coord
      # range and value indexing    
      axes = {ax0.name:(co0[1],co0[-1])}
      # apply function under test
      slcds = dataset(**axes)
      # verify results
      print slcds
      slcvar = slcds[var3.name]
      assert slcvar.ndim == var3.ndim
      assert slcvar.shape == (var3.shape[0]-1,)+var3.shape[1:]
      for slcax,ax in zip(slcvar.axes,var3.axes):
        assert slcax.name == ax.name
        assert slcax.units == ax.units
      assert isEqual(slcvar[:], var3[1:,:,:], masked_equal=True)
      if var2 is not None:
        oldvar = slcds[var2.name]
        assert oldvar.shape == var2.shape
        for oldax,ax in zip(oldvar.axes,var2.axes):
          assert oldax.name == ax.name
          assert oldax.units == ax.units
        assert isEqual(oldvar[:], var2[:], masked_equal=True)      
      # list indexing
      l1 = [-1,0]*3; l2 = [0,-1]*3 
      axes = {ax1.name:co1[l1], ax2.name:co2[l2], }
      # apply function under test
      slcds = dataset(**axes)
      # verify results
      tvar =slcds[var3.name]
      assert tvar.ndim == var3.ndim-1
      assert tvar.shape == (var3.shape[0],len(l1))
      assert isEqual(tvar[:], var3[:,l1,l2], masked_equal=True)
      if var2 is not None:
        lvar = slcds[var2.name]
        assert lvar.shape == (len(l1),)
        assert isEqual(lvar[:], var2[l1,l2], masked_equal=True)      
      # integer index indexing
      axes = {ax0.name:(1,-1), ax1.name:l1, ax2.name:l2}
      # apply function under test
      slcds = dataset(lidx=True, **axes)
      print slcds
      # verify results
      slcvar =slcds[var3.name]
      assert slcvar.ndim == var3.ndim-1
      assert slcvar.shape == (var3.shape[0]-1,len(l1))
      assert isEqual(slcvar[:], var3[1:,l1,l2], masked_equal=True)
    else: raise AssertionError

  def testPrint(self):
    ''' just print the string representation '''
    assert self.dataset.__str__()
    print('')
    print(self.dataset)
    print('')
    
  def testWrite(self):
    ''' write test dataset to a netcdf file '''    
    filename = self.folder + '/test.nc'
    if os.path.exists(filename): os.remove(filename)
    # test object
    dataset = self.dataset
    # add non-conforming attribute
    dataset.atts['test'] = [1,'test',3]
    # write file
#     print dataset.y
#     print dataset.y.getArray(), len(dataset.y)
    writeNetCDF(dataset,filename,writeData=True)
    # check that it is OK
    assert os.path.exists(filename)
    ncfile = nc.Dataset(filename)
    assert ncfile
    print(ncfile)
    ncfile.close()
    if os.path.exists(filename): os.remove(filename)
  

# import modules to be tested
from geodata.netcdf import VarNC, AxisNC, DatasetNetCDF

class NetCDFVarTest(BaseVarTest):  
  
  # some test parameters (TestCase does not take any arguments)
  dataset = 'NARR' # dataset to use (also the folder name)
  plot = False # whether or not to display plots 
  stats = False # whether or not to compute stats on data
  
  def setUp(self):
    self.dataset_name = self.dataset
    if RAM: folder = ramdisk
    else: folder = '/{:s}/{:s}/'.format(data_root,self.dataset) # dataset name is also in folder name
    # select dataset
    if self.dataset == 'GPCC': # single file
      filelist = ['gpcc_test/full_data_v6_precip_25.nc'] # variable to test
      varlist = ['p']
      ncfile = filelist[0]; ncvar = varlist[0]      
    elif self.dataset == 'NARR': # multiple files
      filelist = ['narr_test/air.2m.mon.ltm.nc', 'narr_test/prate.mon.ltm.nc', 'narr_test/prmsl.mon.ltm.nc'] # variable to test
      varlist = ['air','prate','prmsl','lon','lat']
      ncfile = filelist[0]; ncvar = varlist[0]
    # load a netcdf dataset, so that we have something to play with     
    if os.path.exists(folder+ncfile): self.ncdata = nc.Dataset(folder+ncfile,mode='r')
    else: raise IOError, folder+ncfile
    # load variable
    ncvar = self.ncdata.variables[ncvar]      
    # get dimensions and coordinate variables
    size = tuple([len(self.ncdata.dimensions[dim]) for dim in ncvar.dimensions])
    axes = tuple([AxisNC(self.ncdata.variables[dim], length=le) for dim,le in zip(ncvar.dimensions,size)]) 
    # initialize netcdf variable 
    self.ncvar = ncvar; self.axes = axes
    self.var = VarNC(ncvar, axes=axes, load=True)    
    self.rav = VarNC(ncvar, axes=axes, load=True) # second variable for binary operations    
    # save the original netcdf data
    self.data = ncvar[:].copy() #.filled(0)
    self.size = tuple([len(ax) for ax in axes])
    # construct attributes dictionary from netcdf attributes
    self.atts = { key : self.ncvar.getncattr(key) for key in self.ncvar.ncattrs() }
    self.atts['name'] = self.ncvar._name
    if 'units' not in self.atts: self.atts['units'] = '' 
      
  def tearDown(self):  
    self.var.unload()   
    self.ncdata.close()
  
  ## specific NetCDF test cases

  def testFileAccess(self):
    ''' test access to data without loading '''
    # get test objects
    var = self.var
    var.unload()
    # access data
    data = var[:]
    assert data.shape == self.data.shape
    assert isEqual(self.data[:], data)
    # assert no data
    assert not var.data
    assert var.data_array is None

  def testIndexing(self):
    ''' test indexing and slicing '''
    # get test objects
    var = self.var
    # indexing (getitem) test    
    if var.ndim == 3:
      assert isEqual(self.data[1,1,1], var[1,1,1])
      assert isEqual(self.data[1,:,1:-1], var[1,:,1:-1])
    # test axes

  def testLoadSlice(self):
    ''' test loading of slices '''
    # get test objects
    var = self.var
    var.unload()
    # load slice
    if var.ndim == 3:
      sl = (slice(0,12,1),slice(20,50,5),slice(70,140,15))
      var.load(sl)
      assert (12,6,5) == var.shape
      if var.masked:
        assert isEqual(self.data.__getitem__(sl), var.data_array)
      else:
        assert isEqual(self.data.__getitem__(sl).filled(var.fillValue), var.data_array)
    else: raise AssertionError

  def testScaling(self):
    ''' test scale and offset operations '''
    # get test objects
    var = self.var
    # unload and change scale factors    
    var.unload()
    var.scalefactor = 2.
    var.offset = 100.
    # load data with new scaling
    var.load()
    assert self.size == var.shape
    assert isEqual((self.data+100.)*2, var.data_array)
  

class DatasetNetCDFTest(BaseDatasetTest):  
  
  # some test parameters (TestCase does not take any arguments)
  dataset_name = 'GPCC' # dataset to use (also the folder name)
  plot = False # whether or not to display plots 
  stats = False # whether or not to compute stats on data
  
  def setUp(self):
    
    if RAM: folder = ramdisk
    else: folder = '/{:s}/{:s}/'.format(data_root,self.dataset_name) # dataset name is also in folder name
    self.folder = folder
    # select dataset
    name = self.dataset_name
    if self.dataset_name == 'GPCC': # single file      
      filelist = ['gpcc_test/full_data_v6_precip_25.nc'] # variable to test
      varlist = ['p']; varatts = dict(p=dict(name='precip'))
      ncfile = filelist[0]; ncvar = varlist[0]      
      self.dataset = DatasetNetCDF(name=name,folder=folder,filelist=filelist,varlist=varlist,varatts=varatts)
    elif self.dataset_name == 'NARR': # multiple files
      filelist = ['narr_test/air.2m.mon.ltm.nc', 'narr_test/prate.mon.ltm.nc', 'narr_test/prmsl.mon.ltm.nc'] # variable to test
      varlist = ['air','prate','prmsl','lon','lat'] # not necessary with ignore_list = ('nbnds',)
      varatts = dict(air=dict(name='T2'),prmsl=dict(name='pmsl'))
      ncfile = filelist[0]; ncvar = varlist[0]
      self.dataset = DatasetNetCDF(name=name,folder=folder,filelist=filelist,varlist=None,varatts=varatts, ignore_list=('nbnds',))
    # load a netcdf dataset, so that we have something to play with      
    self.ncdata = nc.Dataset(folder+ncfile,mode='r')
    # load a sample variable directly
    self.ncvarname = ncvar
    ncvar = self.ncdata.variables[ncvar]
    # get dimensions and coordinate variables
    size = tuple([len(self.ncdata.dimensions[dim]) for dim in ncvar.dimensions])
    axes = tuple([AxisNC(self.ncdata.variables[dim], length=le) for dim,le in zip(ncvar.dimensions,size)]) 
    # initialize netcdf variable 
    self.ncvar = ncvar; self.axes = axes
    self.var = VarNC(ncvar, name='T2' if name is 'NARR' else 'precip', axes=axes, load=True)
    if name is 'NARR': self.lar = VarNC(self.ncdata.variables['lon'], name='lon', axes=axes[1:], load=True)
    else: self.lar = None
    self.rav = VarNC(ncvar, name='T2' if name is 'NARR' else 'precip', axes=axes, load=True)
    # save the original netcdf data
    self.data = ncvar[:].copy() #.filled(0)
    self.size = tuple([len(ax) for ax in axes])
    # construct attributes dictionary from netcdf attributes
    self.atts = { key : self.ncvar.getncattr(key) for key in self.ncvar.ncattrs() }
    self.atts['name'] = self.ncvar._name
    if 'units' not in self.atts: self.atts['units'] = '' 
      
  def tearDown(self):  
    self.var.unload()   
    self.ncdata.close()
  
  ## specific NetCDF test cases
  
  def testCopy(self):
    ''' test copying the entire dataset '''    
    filename = self.folder + 'test.nc'
    if os.path.exists(filename): os.remove(filename)
    # test object
    dataset = self.dataset
    # make a copy
    copy = dataset.copy(asNC=True, filename=filename)
    # test
    assert copy is not dataset # should not be the same
    assert isinstance(copy,DatasetNetCDF)
    assert all([copy.hasAxis(ax.name) for ax in dataset.axes.values()])
    assert all([copy.hasVariable(var.name) for var in dataset.variables.values()])
    copy.close()
    assert os.path.exists(filename) # check for file
      
  def testCreate(self):
    ''' test creation of a new NetCDF dataset and file '''
    filename = self.folder + 'test.nc'
    if os.path.exists(filename): os.remove(filename)
    # create NetCDF Dataset
    dataset = DatasetNetCDF(filelist=[filename],mode='w')
#     print(dataset)
    # add an axis
    ax = Axis(name='t', units='', coord=np.arange(10))
    dataset.addAxis(ax, asNC=True)
    # add a random variable
    var = Variable(name='test', units='', axes=(ax,), data=np.zeros((10,)))
    dataset.addVariable(var, asNC=True)
    # add some attribute
    dataset.atts.test = 'test'
    # synchronize with disk and close
    dataset.sync()     
#     print(dataset)
    dataset.close()
    # check that it is OK
    assert os.path.exists(filename)
    ncfile = nc.Dataset(filename)
    assert ncfile
    ncfile.close()
    dataset = DatasetNetCDF(filelist=[filename],mode='r')
    print(dataset)
    dataset.close()

  def testStringVar(self):
    ''' test behavior of string variables in a netcdf dataset '''
    filename = self.folder + 'test.nc'
    if os.path.exists(filename): os.remove(filename)
    # create NetCDF Dataset
    dataset = DatasetNetCDF(filelist=[filename],mode='w')
    # add an axis
    ax = Axis(name='t', units='', coord=np.arange(3))
    dataset.addAxis(ax, asNC=True)
    # add a string variable
    test_string = ['This','is a','string']
    strarray = np.array(test_string)
    strvar = Variable(name='string', units='', axes=(ax,), data=strarray)
    dataset.addVariable(strvar, asNC=True)
    # add some attribute
    dataset.atts.test = 'test'
    # synchronize with disk and close
    dataset.sync()     
#     print(dataset)
    dataset.close()
    # check that it is OK
    assert os.path.exists(filename)
    ncfile = nc.Dataset(filename)
#     print(ncfile)
    assert ncfile
    ncfile.close()
    dataset = DatasetNetCDF(filelist=[filename],mode='r',load=True)
    print(dataset)
    assert all(dataset.string.data_array == np.array(test_string))
    dataset.close()

  def testLoad(self):
    ''' test loading and unloading of data '''
    # test objects: vars and axes
    dataset = self.dataset
    # load data
    dataset.load()
    assert all([var.data for var in dataset])
    # unload data
    dataset.unload()
    assert all([not var.data for var in dataset])


# import modules to be tested
from geodata.gdal import addGDALtoVar, addGDALtoDataset
from datasets.NARR import projdict

class GDALVarTest(NetCDFVarTest):  
  
  # some test parameters (TestCase does not take any arguments)
  dataset = 'GPCC' # dataset to use (also the folder name)
  plot = False # whether or not to display plots 
  stats = False # whether or not to compute stats on data
  # some projection settings for tests
  projection = ''
  
  def setUp(self):
    super(GDALVarTest,self).setUp()
    # add GDAL functionality to variable
    if self.dataset == 'NARR':
      self.var = addGDALtoVar(self.var, projection=projdict)
    else: 
      self.var = addGDALtoVar(self.var)
      
  def tearDown(self):  
    super(GDALVarTest,self).tearDown()
  
  ## specific GDAL test cases

  def testAddProjection(self):
    ''' test function that adds projection features '''
    # get test objects
    var = self.var # NCVar object
#     print var.xlon[:]
#     print var.ylat[:]
    print var.geotransform # need to subtract false easting and northing!
    # trivial tests
    assert var.gdal
    if self.dataset == 'NARR': assert var.isProjected == True
    if self.dataset == 'GPCC': assert var.isProjected == False
    assert var.geotransform
    data = var.getGDAL()
    assert data is not None
    assert data.ReadAsArray()[:,:,:].shape == (var.bands,)+var.mapSize 


class DatasetGDALTest(DatasetNetCDFTest):  
  
  # some test parameters (TestCase does not take any arguments)
  dataset_name = 'NARR' # dataset to use (also the folder name)
  plot = False # whether or not to display plots 
  stats = False # whether or not to compute stats on data
  
  def setUp(self):
    super(DatasetGDALTest,self).setUp()
    # add GDAL functionality to variable
    if self.dataset.name == 'NARR':
      self.dataset = addGDALtoDataset(self.dataset, projection=projdict) # projected
    else: 
      self.dataset = addGDALtoDataset(self.dataset) # not projected
      
  def tearDown(self):  
    super(DatasetGDALTest,self).tearDown()
  
  ## specific GDAL test cases

  def testAddProjection(self):
    ''' test function that adds projection features '''
    # get test objects
    dataset = self.dataset # dataset object
#     print var.xlon[:]
#     print var.ylat[:]
    # trivial tests
    assert dataset.gdal
    assert dataset.projection
    assert dataset.geotransform
    assert len(dataset.geotransform) == 6 # need to subtract false easting and northing!
    if self.dataset.name == 'NARR': 
      assert dataset.isProjected == True
      assert dataset.xlon == dataset.x and dataset.ylat == dataset.y    
    if self.dataset.name == 'GPCC': 
      assert dataset.isProjected == False
      assert dataset.xlon == dataset.lon and dataset.ylat == dataset.lat
    # check variables
    for var in dataset.variables.values():
      assert (var.ndim >= 2 and var.hasAxis(dataset.xlon) and var.hasAxis(dataset.ylat)) == var.gdal              
    
    
if __name__ == "__main__":

    # construct dictionary of test classes defined above
    test_classes = dict()
    local_values = locals().copy()
    for key,val in local_values.iteritems():
      if key[-4:] == 'Test':
        test_classes[key[:-4]] = val

    # list of tests to be performed
    tests = [] 
    # list of variable tests
    tests += ['BaseVar'] 
    tests += ['NetCDFVar']
    tests += ['GDALVar']
    # list of dataset tests
    tests += ['BaseDataset']
    tests += ['DatasetNetCDF']
    tests += ['DatasetGDAL']
    
    # RAM disk settings ("global" variable)
    RAM = False # whether or not to use a RAM disk
    ramdisk = '/media/tmp/' # folder where RAM disk is mounted
    
    # run tests
    report = []
    for test in tests:
      s = unittest.TestLoader().loadTestsFromTestCase(test_classes[test])
      #s = unittest.TestLoader().loadTestsFromName('DatasetGDALTest.testEnsemble')
      report.append(unittest.TextTestRunner(verbosity=2).run(s))
      
    # print summary
    runs = 0; errs = 0; fails = 0
    for name,test in zip(tests,report):
      #print test, dir(test)
      runs += test.testsRun
      e = len(test.errors)
      errs += e
      f = len(test.failures)
      fails += f
      if e+ f != 0: print("\nErrors in '{:s}' Tests: {:s}".format(name,str(test)))
    if errs + fails == 0:
      print("\n   ***   All {:d} Test(s) successfull!!!   ***   \n".format(runs))
    else:
      print("\n   ###   Test Summary:   Ran {:d} Test(s), encountered {:d} Failure(s) and {:d} Error(s)   ###   \n".format(runs,errs,fails))
    