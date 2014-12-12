'''
Created on Dec 11, 2014

A custom Figure class that provides some specialized functions and uses a custom Axes class.

@author: Andre R. Erler, GPL v3
'''

# external imports
from types import NoneType, ModuleType
import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure, SubplotBase, subplot_class_factory
# internal imports
from geodata.misc import isInt 
from plotting.axes import MyAxes, MyLocatableAxes, Axes
from plotting.utils import loadMPL


## my new figure class
class MyFigure(Figure):
  ''' 
    A custom Figure class that provides some specialized functions and uses a custom Axes class.
    This is achieved by overloading add_axes and add_subplot.
    (This class does not support built-in projections; use the Basemap functionality instead.)  
  '''
  # some default parameters
  title_height = 0.03
  shared_legend = None
  legend_axes = None
  shared_colorbar = None
  colorbar_axes = None
  
  def __init__(self, *args, **kwargs):
    ''' constructor that accepts custom axes_class as keyword argument '''
    # parse arguments
    if 'axes_class' in kwargs:
      axes_class = kwargs.pop('axes_class')
      if not issubclass(axes_class, Axes): raise TypeError
    else: axes_class = MyAxes # default
    if 'axes_args' in kwargs:
      axes_args = kwargs.pop('axes_args')
      if axes_args is not None and not isinstance(axes_args, dict): raise TypeError
    else: axes_args = None # default
    # call parent constructor
    super(MyFigure,self).__init__(*args, **kwargs)
    # save axes class for later
    self.axes_class = axes_class   
    self.axes_args = axes_args 
    
# N.B.: using the built-in mechanism to choose Axes seems to cause more problems
#     from matplotlib.projections import register_projection
#     # register custom class with mpl
#     register_projection(axes_class)
#   def add_axes(self, *args, **kwargs):
#     ''' overloading original add_subplot in order to use custom Axes (adapted from parent) '''
#     if 'projection' not in kwargs:
#       kwargs['projection'] = 'my'
#     super(MyFigure,self).__init__(*args, **kwargs)
  
  def add_axes(self, *args, **kwargs):
    ''' overloading original add_subplot in order to use custom Axes (adapted from parent) '''
    if not len(args):
        return
    # shortcut the projection "key" modifications later on, if an axes
    # with the exact args/kwargs exists, return it immediately.
    key = self._make_key(*args, **kwargs)
    ax = self._axstack.get(key)
    if ax is not None:
        self.sca(ax)
        return ax
    if isinstance(args[0], Axes): # allow all Axes, if passed explicitly
      a = args[0]
      assert(a.get_figure() is self)
    else:
      rect = args[0]
      # by registering the new Axes class as a projection, it may be possible 
      # to use the old axes creation mechanism, but it doesn't work this way...
      #       from matplotlib.figure import process_projection_requirements 
      #       if 'projection' not in kwargs: kwargs['projection'] = 'my'
      #       axes_class, kwargs, key = process_projection_requirements(
      #           self, *args, **kwargs)
      axes_class = self.axes_class # defaults to my new custom axes (MyAxes)
      key = self._make_key(*args, **kwargs)
      # check that an axes of this type doesn't already exist, if it
      # does, set it as active and return it
      ax = self._axstack.get(key)
      if ax is not None and isinstance(ax, axes_class):
          self.sca(ax)
          return ax
      # create the new axes using the axes class given
      # add default axes arguments
      if self.axes_args is not None:
        axes_args = self.axes_args.copy()
        axes_args.update(kwargs)
      else: axes_args = kwargs
      a = axes_class(self, rect, **axes_args)
    self._axstack.add(key, a)
    self.sca(a)
    return a
  
  def add_subplot(self, *args, **kwargs):
    ''' overloading original add_subplot in order to use custom Axes (adapted from parent) '''
    if not len(args):
        return
    if len(args) == 1 and isinstance(args[0], int):
        args = tuple([int(c) for c in str(args[0])])
    if isinstance(args[0], SubplotBase):
      # I'm not sure what this does...
      a = args[0]
      assert(a.get_figure() is self)
      # make a key for the subplot (which includes the axes object id
      # in the hash)
      key = self._make_key(*args, **kwargs)
    else:
      #       if 'projection' not in kwargs: kwargs['projection'] = 'my'
      #       axes_class, kwargs, key = process_projection_requirements(
      #           self, *args, **kwargs)      
      axes_class = self.axes_class # defaults to my new custom axes (MyAxes)
      key = self._make_key(*args, **kwargs)
      # try to find the axes with this key in the stack
      ax = self._axstack.get(key)
      if ax is not None:
        if isinstance(ax, axes_class):
          # the axes already existed, so set it as active & return
          self.sca(ax)
          return ax
        else:
          # Undocumented convenience behavior:
          # subplot(111); subplot(111, projection='polar')
          # will replace the first with the second.
          # Without this, add_subplot would be simpler and
          # more similar to add_axes.
          self._axstack.remove(ax)
      # add default axes arguments
      if self.axes_args is not None:
        axes_args = self.axes_args.copy()
        axes_args.update(kwargs)      
      else: axes_args = kwargs
      # generate subplot class and create axes instance
      a = subplot_class_factory(axes_class)(self, *args, **axes_args)
    self._axstack.add(key, a)
    self.sca(a)
    return a
  
  # function to adjust subplot parameters
  def updateSubplots(self, mode='shift', **kwargs):
    ''' simple helper function to move (relocate), shift, or scale subplot margins '''
    pos = self.subplotpars
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
    self.subplots_adjust(**margins)
    # and now repair damage: restore axes
    for ax in self.axes:
      if ax.get_title():
        pos = ax.get_position()
        pos = pos.from_bounds(x0=pos.x0, y0=pos.y0, width=pos.width, height=pos.height-self.title_height)    
        ax.set_position(pos)

  # add common/shared legend to a multi-panel plot
  def addSharedLegend(self, plts=None, legs=None, fontsize=None, **kwargs):
    ''' add a common/shared legend to a multi-panel plot '''
    # complete input
    if legs is None: legs = [plt.get_label() for plt in plts]
    elif not isinstance(legs, (list,tuple)): raise TypeError
    if not isinstance(plts, (list,tuple,NoneType)): raise TypeError
    # selfure out fontsize and row numbers  
    fontsize = fontsize or self.axes[0].get_yaxis().get_label().get_fontsize() # or fig._suptitle.get_fontsize()
    nlen = len(plts) if plts else len(legs)
    if fontsize > 11: ncols = 2 if nlen == 4 else 3
    else: ncols = 3 if nlen == 6 else 4              
    # make room for legend
    leghgt = np.ceil(nlen/ncols) * fontsize + 0.055
    ax = self.add_axes([0, 0, 1,leghgt]) # new axes to hold legend, with some attributes
    ax.set_frame_on(False); ax.axes.get_yaxis().set_visible(False); ax.axes.get_xaxis().set_visible(False)
    self.updateSubplots(mode='shift', bottom=leghgt) # shift bottom upwards
    # define legend parameters
    legargs = dict(loc=10, ncol=ncols, borderaxespad=0., fontsize=fontsize, frameon=True,
                   labelspacing=0.1, handlelength=1.3, handletextpad=0.3, fancybox=True)
    legargs.update(kwargs)
    # create legend and return handle
    if plts: legend = ax.legend(plts, legs, **legargs)
    else: legend = ax.legend(legs, **legargs)
    # store axes handle and legend
    self.legend_axes = ax
    self.shared_legend = legend
    return legend
    
  # add subplot/axes labels
  def addLabels(self, labels=None, loc=1, lstroke=False, lalphabet=True, size=None, prop=None, **kwargs):
    # expand list
    axes = self.axes
    n = len(axes)
    if not isinstance(labels,(list,tuple)): labels = [labels]*n
    if not isinstance(loc,(list,tuple)): loc = [loc]*n
    if not isinstance(lstroke,(list,tuple)): lstroke = [lstroke]*n
    # settings
    if prop is None: prop = dict()
    if not size: prop['size'] = 'large'
    args = dict(pad=0., borderpad=1.5, frameon=False)
    args.update(kwargs)
    # cycle over axes
    ats = [] # list of texts
    for i,ax in enumerate(axes):
      # skip shared legend or colorbar
      if ax is not self.legend_axes and ax is not self.colorbar_axes:
        # default label
        label = labels[i]
        if label is None: 
          label = i
          if not lalphabet: label += 1
        # create label artist
        ats.append(ax.addLabel(label, loc=loc[i], lstroke=lstroke[i], lalphabet=lalphabet, 
                               prop=prop, **args))      
    return ats
  

## convenience function to return a figure and an array of ImageGrid axes
def getFigAx(subplot, name=None, title=None, figsize=None,  mpl=None, margins=None,
             sharex=None, sharey=None, AxesGrid=False, ngrids=None, direction='row',
             axes_pad = None, add_all=True, share_all=None, aspect=False,
             label_mode='L', cbar_mode=None, cbar_location='right', lreduce=True,
             cbar_pad=None, cbar_size='5%', axes_class=None, axes_args=None,
             figure_class=None, figure_args=None): 
  # configure matplotlib
  if mpl is None: import matplotlib as mpl
  elif isinstance(mpl,dict): mpl = loadMPL(**mpl) # there can be a mplrc, but also others
  elif not isinstance(mpl,ModuleType): raise TypeError
  # default figure class
  if figure_class is None: figure_class = MyFigure
  elif not issubclass(figure_class, Figure): raise TypeError 
  if figure_args is None: figure_args = dict() 
  elif not isinstance(figure_args, dict): raise TypeError
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
    title_height = getattr(figure_class, 'title_height', 0.03) # use default from figure
    if title is not None: margins = margins[:3]+(margins[3]-title_height,) # make room for title
  if AxesGrid:
    if share_all is None: share_all = True
    if axes_pad is None: axes_pad = 0.05
    # create axes using the Axes Grid package
    if axes_class is None: axes_class=MyLocatableAxes
    fig = mpl.pylab.figure(facecolor='white', figsize=figsize, axes_class=axes_class, 
                           FigureClass=MyFigure, **figure_args)
    if axes_args is None: axes_class = (axes_class,{})
    elif isinstance(axes_args,dict): axes_class = (axes_class,axes_args)
    else: raise TypeError
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
    # other axes arguments
    if axes_class is None: axes_class=MyAxes
    if axes_args is not None and not isinstance(axes_args,dict): raise TypeError
    # create figure
    from matplotlib.pyplot import subplots    
    # GridSpec: http://matplotlib.org/users/gridspec.html 
    fig, axes = subplots(subplot[0], subplot[1], sharex=sharex, sharey=sharey,squeeze=lreduce, 
                         facecolor='white', figsize=figsize, FigureClass=MyFigure, 
                         subplot_kw=axes_args, axes_class=axes_class, **figure_args)    
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

if __name__ == '__main__':
    pass