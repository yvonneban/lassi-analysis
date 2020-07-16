#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 18:15:46 2020

@author: yvonne
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter


class tls(object):
    """
    Class to handle simulation and computation of TLS scans and data. Error 
    terms taken from Mularikrishnan 2015, values taken from Holst 2017.
    """
    def __init__(self, backface=False, # boolean
                 offset=(0,0,0), # m, metres
                 x1n=-0.9e-3, # m, metres
                 dx1n=0.04e-3, # m, metres
                 x1z=-0.2e-3, # m, metres
                 dx1z=0.02e-3, # m, metres
                 x2=-0.9e-3, # m, metres
                 dx2=0.04e-3, # m, metres
                 x3=-0.3e-3, # m, metres
                 dx3=0.02e-3, # m, metres
                 x4=8.29031e-5, # rad, radians
                 dx4=4.36332e-7, # rad, radians
                 x5n=5.09054e-5, # rad, radians
                 dx5n=3.78155e-6, # rad, radians
                 x5z=4.489375e-4, # rad, radians
                 dx5z=1.89077e-6, # rad, radians
                 x6=2.5695e-5, # rad, radians
                 dx6=9.6963e-7, # rad, radians
                 x7=4.489375e-4, # rad, radians
                 dx7=1.89077e-6, # rad, radians
                 x8x=0, # none
                 dx8x=0, # none
                 x8y=0, # none
                 dx8y=0, # none
                 x9n=0, # none
                 dx9n=0, # none
                 x9z=0, # none
                 dx9z=0, # none
                 x10=0, # m, metres
                 dx10=0, # m, metres
                 x11a=0, # rad, radians
                 dx11a=0, # rad, radians
                 x11b=0, # rad, radians
                 dx11b=0, # rad, radians
                 x12a=0, # rad, radians
                 dx12a=0, # rad, radians
                 x12b=0, # rad, radians
                 dx12b=0 # rad, radians
                 ):
        """
        Initialises a tls object with the instrument errors.
        Input:
            backface (boolean, default=False)
            offset (metres, 3-tuple of float in x,y,z, default=(0,0,0))
            x1n (metres, float, default=-0.9e-3)
            dx1n (metres, float, default=0.04e-3)
            x1z (metres, float, default=-0.2e-3)
            dx1z (metres, float, default=0.02e-3)
            x2 (metres, float, default=-0.9e-3)
            dx2 (metres, float, default=0.04e-3)
            x3 (metres, float, default=-0.3e-3)
            dx3 (metres, float, default=0.02e-3)
            x4 (radians, float, default=8.29031e-5)
            dx4 (radians, float, default=4.36332e-7)
            x5n (radians, float, default=5.09054e-5)
            dx5n (radians, float, default=3.78155e-6)
            x5z (radians, float, default=4.489375e-4)
            dx5z (radians, float, default=1.89077e-6)
            x6 (radians, float, default=2.5695e-5)
            dx6 (radians, float, default=9.6963e-7)
            x7 (radians, float, default=4.489375e-4)
            dx7 (radians, float, default=1.89077e-6)
            x8x (none, float, default=0)
            dx8x (none, float, default=0)
            x8y (none, float, default=0)
            dx8y (none, float, default=0)
            x9n (none, float, default=0)
            dx9n (none, float, default=0)
            x9z (none, float, default=0)
            dx9z (none, float, default=0)
            x10 (metres, float, default=0)
            dx10 (metres, float, default=0)
            x11a (radians, float, default=0)
            dx11a (radians, float, default=0)
            x11b (radians, float, default=0)
            dx11b (radians, float, default=0)
            x12a (radians, float, default=0)
            dx12a (radians, float, default=0)
            x12b (radians, float, default=0)
            dx12b (radians, float, default=0)
        Output:
            none
        """
        self.backface=backface
        self.offset=offset # m, metres
        self.x1n = x1n # m, metres
        self.dx1n = dx1n # m, metres
        self.x1z = x1z # m, metres
        self.dx1z = dx1z # m, metres
        self.x2 = x2 # m, metres
        self.dx2 = dx2 # m, metres
        self.x3 = x3 # m, metres
        self.dx3 = dx3 # m, metres
        self.x4 = x4 # rad, radians
        self.dx4 = dx4 # rad, radians
        self.x5n = x5n # rad, radians
        self.dx5n = dx5n # rad, radians
        self.x5z = x5z # rad, radians
        self.dx5z = dx5z # rad, radians
        self.x6 = x6 # rad, radians
        self.dx6 = dx6 # rad, radians
        self.x7 = x7 # rad, radians
        self.dx7 = dx7 # rad, radians
        self.x8x = x8x # none
        self.dx8x = dx8x # none
        self.x8y = x8y # none
        self.dx8y = dx8y # none
        self.x9n = x9n # none
        self.dx9n = dx9n # none
        self.x9z = x9z # none
        self.dx9z = dx9z # none
        self.x10 = x10 # m, metres
        self.dx10 = dx10 # m, metres
        self.x11a = x11a # rad, radians
        self.dx11a = dx11a # rad, radians
        self.x11b = x11b # rad, radians
        self.dx11b = dx11b # rad, radians
        self.x12a = x12a # rad, radians
        self.dx12a = dx12a # rad, radians
        self.x12b = x12b # rad, radians
        self.dx12b = dx12b # rad, radians
        
    def Cart_to_Sph(self, xdata, ydata, zdata):
        """
        Returns the R, Hangle, and Vangle data in spherical coordinates
        measured by the TLS from xdata, ydata, and zdata, in Cartesian 
        coordinates ignoring offsets. Assigns values to xdata, ydata, and 
        zdata arrays.
        Input:
            xdata (metres, np array of float)
            ydata (metres, np array of float)
            zdata (metres, np array of float)
        Output:
            Rarray (metres, np array of float)
            Harray (radians, np array of float)
            Varray (radians, np array of float)
        """
        if xdata.shape != ydata.shape or xdata.shape != zdata.shape or ydata.shape != zdata.shape:
            raise ValueError('data arrays do not have the same dimensions')
        
        Rarray = np.sqrt(self.xdata**2 + self.ydata**2 + self.zdata**2) # m, metres
        Harray = np.arctan2(self.ydata, self.xdata) # rad, radians
        Varray = np.arccos(self.zdata/Rarray) # rad, radians
        return Rarray, Harray, Varray
    
    def scan(self, xdata, ydata, zdata):
        """
        Returns the R, Hangle, and Vangle data measured by the TLS from xdata,
        ydata, and zdata. Assigns values to R, Hangle, and Vangle offsets and 
        errors.
        Input:
            xdata (metres, np array of float)
            ydata (metres, np array of float)
            zdata (metres, np array of float)
        Output:
            Rdata (metres, np array of float)
            Hdata (radians, np array of float)
            Vdata (radians, np array of float)
        """
        self.xdata = xdata - self.offset[0] # m, metres
        self.ydata = ydata - self.offset[1] # m, metres
        self.zdata = zdata - self.offset[2] # m, metres
        
        self.xerr = 0 # m, metres
        self.yerr = 0 # m, metres
        self.zerr = 0 # m, metres
        
        Rarray, Harray, Varray = self.Cart_to_Sph(self.xdata, self.ydata, self.zdata)
        
        if self.backface:
            k=-1
        else:
            k=1
        
        self.Roffset = k*self.x2*np.sin(Varray) + self.x10 # m, metres
        self.Hoffset = k*(self.x1z/(Rarray*np.tan(Varray)) + self.x3/(Rarray*np.sin(Varray)) + self.x5z/np.tan(Varray) + 2*self.x6/np.sin(Varray) - self.x7/np.tan(Varray) - self.x8x*np.sin(Harray) + self.x8y*np.cos(Harray)) + (self.x1n/Rarray + self.x5n + self.x11a*np.cos(2*Harray) + self.x11b*np.sin(2*Harray)) # rad, radians
        self.Voffset = k*(self.x1n*np.cos(Varray)/Rarray + self.x2*np.cos(Varray)/Rarray + self.x4 + self.x5n*np.cos(Varray) + self.x9n*np.cos(Varray)) + (-self.x1z*np.sin(Varray)/Rarray - self.x5z*np.sin(Varray) - self.x9z*np.sin(Varray) + self.x12a*np.cos(2*Varray) + self.x12b*np.sin(2*Varray)) # rad, radians
        
        self.Rerr = np.abs( k*self.dx2*np.sin(Varray) + self.dx10 ) # m, metres
        self.Herr = np.abs( k*(self.dx1z/(Rarray*np.tan(Varray)) + self.dx3/(Rarray*np.sin(Varray)) + self.dx5z/np.tan(Varray) + 2*self.dx6/np.sin(Varray) - self.dx7/np.tan(Varray) - self.dx8x*np.sin(Harray) + self.dx8y*np.cos(Harray)) + (self.dx1n/Rarray + self.dx5n + self.dx11a*np.cos(2*Harray) + self.dx11b*np.sin(2*Harray)) ) # rad, radians
        self.Verr = np.abs( k*(self.dx1n*np.cos(Varray)/Rarray + self.dx2*np.cos(Varray)/Rarray + self.dx4 + self.dx5n*np.cos(Varray) + self.dx9n*np.cos(Varray)) + (-self.dx1z*np.sin(Varray)/Rarray - self.dx5z*np.sin(Varray) - self.dx9z*np.sin(Varray) + self.dx12a*np.cos(2*Varray) + self.dx12b*np.sin(2*Varray)) ) # rad, radians
        
        self.Rdata = Rarray + self.Roffset # m, metres
        self.Hdata = Harray + self.Hoffset # rad, radians
        self.Vdata = Varray + self.Voffset # rad, radians
        
        return self.Rdata, self.Hdata, self.Vdata
    
    def error(self):
        """
        Returns randomly generated uncertainties drawn from a normal
        distribution of widths given by Rerr, Herr, and Verr.
        Note: must be run after self.scan().
        Input:
            none
        Output:
            Rsigma (metres, np array of float)
            Hsigma (radians, np array of float)
            Vsigma (radians, np array of float)
        """
        Rsigma = np.random.normal(scale=self.Rerr) # m, metres
        Hsigma = np.random.normal(scale=self.Herr) # rad, radians
        Vsigma = np.random.normal(scale=self.Verr) # rad, radians
        return Rsigma, Hsigma, Vsigma
    
    def Sph_to_Cart(self, Rarray, Harray, Varray):
        """
        Assigns the xdata, ydata, and zdata in Cartesian coordinatess derived
        from Rarray, Harray, and Varray in spherical coordinates.
        Input:
            Rarray (metres, np array of float)
            Harray (radians, np array of float)
            Varray (radians, np array of float)
        Output:
            xarray (metres, np array of float)
            yarray (metres, np array of float)
            zarray (metres, np array of float)
        """
        if Rarray.shape != Harray.shape or Rarray.shape != Varray.shape or Harray.shape != Varray.shape:
            raise ValueError('data arrays do not have the same dimensions')
        
        xarray = Rarray*np.sin(Varray)*np.cos(Harray) + self.offset[0] # m, metres
        yarray = Rarray*np.sin(Varray)*np.sin(Harray) + self.offset[1] # m, metres
        zarray = Rarray*np.cos(Varray) + self.offset[2] # m, metres
        return xarray, yarray, zarray
    
    def unscan(self, Rdata, Hdata, Vdata, Rerr=0, Herr=0, Verr=0):
        """
        Returns the x, y, and z data derived from Rdata, Hdata, and Vdata.
        Input:
            Rdata (metres, np array of float)
            Hdata (radians, np array of float)
            Vdata (radians, np array of float)
            Rerr (metres, np array of float, default=0)
            Herr (radians, np array of float, default=0)
            Verr (radians, np array of float, default=0)
        Output:
            xdata (metres, np array of float)
            ydata (metres, np array of float)
            zdata (metres, np array of float)
        """
        self.Rdata = Rdata # m, metres
        self.Hdata = Hdata # rad, radians
        self.Vdata = Vdata # rad, radians
        self.Rerr = Rerr # m, metres
        self.Herr = Herr # rad, radians
        self.Verr = Verr # rad, radians
        
        self.xdata, self.ydata, self.zdata = self.Sph_to_Cart(self.Rdata, self.Hdata, self.Vdata)
        
        self.xerr = self.Rerr*np.sin(self.Vdata)*np.cos(self.Hdata) + self.Verr*self.Rdata*np.cos(self.Vdata)*np.cos(self.Hdata) + self.Herr*self.Rdata*np.sin(self.Vdata)*np.sin(self.Hdata) # m, metres
        self.yerr = self.Rerr*np.sin(self.Vdata)*np.sin(self.Hdata) + self.Verr*self.Rdata*np.cos(self.Vdata)*np.sin(self.Hdata) + self.Herr*self.Rdata*np.sin(self.Vdata)*np.cos(self.Hdata) # m, metres
        self.zerr = self.Rerr*np.cos(self.Vdata) + self.Verr*self.Rdata*np.sin(self.Vdata) # m, metres
        
        return self.xdata, self.ydata, self.zdata, self.xerr, self.yerr, self.zerr


class parabola(object):
    """
    Class that describes a parabola shape with axis of symmetry parallel to 
    vertical.
    """
    def __init__(self, focus=10, vertex=(0,0,0)):
        """
        Initialises a parabola object with focus and vertex given by inputs.
        Input:
            focus (metres, float, default=10)
            vertex (metres, 3-tuple of float in (x,y,z), default=(0,0,0))
        Output:
            none
        """
        self.focus = focus # m, metres
        self.vertex = vertex # m, metres
    
    def section(self, xdata, ydata):
        """
        Returns the section of parabola determined by x- and y-coordinates 
        given by xdata and ydata.
        Input:
            xdata (metres, np array of float)
            ydata (metres, np array of float)
        Output:
            zarray (metres, np array of float)
        """
        if xdata.shape != ydata.shape:
            raise ValueError('xdata and ydata do not have the same dimensions')
        
        self.xdata = xdata # m, metres
        self.ydata = ydata # m, metres
        self.zarray = ((self.xdata - self.vertex[0])**2 + (self.ydata - self.vertex[1])**2) / (4.*self.focus) + self.vertex[2] # m, metres
        return self.zarray


class dish(parabola):
    """
    Class that describes a parabolic dish with axis of symmetry parallel to 
    vertical.
    """
    def __init__(self, focus=60, vertex=(0,0,0)):
        """
        Initialises a dish object with focus and vertex given by inputs.
        Input:
            focus (metres, float, default=60)
            vertex (metres, 3-tuple of float in (x,y,z), default=(0,0,0))
        Output:
            none
        """
        parabola.__init__(self, focus=focus, vertex=vertex)
        
    def circsection(self, radius=50, centre=(0,25), res=0.1):
        """
        Returns a circular section of parabola centred at centre (x,y) with 
        radius given by radius and resolution given by res.
        Input:
            radius (metres, float, default=50)
            centre (metres, 2-tuple of float in (x,y), default=(50,50))
            res (metres, float, default=0.1)
        Output:
            zarray (metres, np array of float)
        """
        self.radius=radius
        self.centre=centre
        self.xdata, self.ydata = np.meshgrid(np.arange(centre[0]-radius, centre[0]+radius, res), np.arange(centre[1]-radius, centre[1]+radius, res)) # m, metres
        parabola.section(self, self.xdata, self.ydata)
        self.zarray[((self.xdata-centre[0])**2 + (self.ydata-centre[1])**2)>radius**2] = np.nan # m, metres
        return self.zarray
        

class wall(object):
    """
    Class that describes a plane object.
    """
    def __init__(self, offset=(0,0,0), size=(10,10), res=0.1):
        """
        Initialises a plane object with offset, size, and resolution given by
        inputs.
        Input:
            offset (metres, 3-tuple of float in (x,y,z), default=(0,0,0))
            size (metres, 2-tuple of float in (x,y), default=(0,0))
            res (metres, float, default=0.1)
        Output:
            none
        """
        self.offset = offset
        self.size = size
        self.xdata, self.ydata = np.meshgrid(np.arange(offset[0]-size[0]/2, offset[0]+size[0]/2, res), np.arange(offset[1]-size[1]/2, offset[1]+size[1]/2, res))
        self.zdata = np.ones_like(self.xdata)*offset[2]


class ball(object):
    """
    Class that describes an ellipsoid shape.
    """
    def __init__(self, centre=(0,0,0), radius=1):
        """
        Initialises a ellipsoid object with centre and axes given by inputs.
        Input:
            centre (metres, 3-tuple of float in (x,y,z), default=(0,0,0))
            radius (metres, float or 3-tuple of float in (x,y,z), default=1)
        Output:
            none
        """
        self.centre = centre # m, metres
        if type(radius) == int or float:
            self.radius = (radius, radius, radius) # m, metres
        elif len(radius) == 3:
            self.radius = radius # m, metres
        else:
            raise ValueError('radius does not have right dimensions')
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        self.xdata = self.radius[0] * np.outer(np.cos(u), np.sin(v))
        self.ydata = self.radius[1] * np.outer(np.sin(u), np.sin(v))
        self.zdata = self.radius[2] * np.outer(np.ones(np.size(u)), np.cos(v))


def rotate(data_vector, rotation_vector=[0,0,0]):
    """
    Returns rotated data array.
    Input:
        data_vector (metres, np array of float of shape (n,3))
        rotation_vector (radians, list of float of shape (n,3), 
            default=[0,0,0])
    Output:
        r.apply(data_vector) (metres, np array of float of shape (n,3), 
            rotated data vector)
    """
    r = R.from_rotvec(rotation_vector) # m, metres
    return r.apply(data_vector) # m, metres

def rotatexyz(xdata, ydata, zdata, rotation_vector=[0,0,0]):
    """
    Returns rotated data array with data entered in separate x-, y-, z-arrays.
    Input:
        xdata (metres, np array of float)
        ydata (metres, np array of float)
        zdata (metres, np array of float)
        rotation_vector (radians, list of float of shape (n,3), 
            default=[0,0,0])
    Output:
        r.apply(data_vector).reshape(vector.shape) (metres, np array of float 
            of shape (x,y,3), rotated data array)
    """
    if xdata.shape != ydata.shape or xdata.shape != zdata.shape or ydata.shape != zdata.shape:
        raise ValueError('data arrays do not have the same dimensions')
    
    r = R.from_rotvec(rotation_vector)
    vector = np.dstack((xdata, ydata, zdata)) # m, metres
    data_vector = vector.reshape((-1,3)) # m, metres
    return r.apply(data_vector).reshape(vector.shape) # m, metres

def smooth(data_array, sigma=10, downsamp=2):
    """
    Returns the 2D data array convolved with a symmetric Gaussian smoothing 
    function of width given by sigma and downsampled by factor given by 
    downsamp.
    Input:
        data_array (np array of float of shape x,y)
        sigma (int or float, default=10)
        downsamp (int, default=2)
    Output:
        gaussian_filter(data_array, sigma=sigma)[::2**downsamp,::2**downsamp] 
            (np array of float)
    """
    return gaussian_filter(data_array, sigma=sigma)[::2**downsamp,::2**downsamp]
