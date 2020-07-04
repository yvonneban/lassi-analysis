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
    def __init__(self, offset=(0,0,0), # m, metres
                 x1n=-0.9e3, # m, metres
                 x1z=-0.2e-3, # m, metres
                 x2=-0.9e3, # m, metres
                 x3=-0.3e-3, # m, metres
                 x4=8.29031e-5, # rad, radians
                 x5n=5.09054e-5, # rad, radians
                 x5z=4.489375e-4, # rad, radians
                 x6=2.5695e-5, # rad, radians
                 x7=4.489375e-4, # rad, radians
                 x8x=0, # none
                 x8y=0, # none
                 x9n=0, # none
                 x9z=0, # none
                 x10=0, # m, metres
                 x11a=0, # rad, radians
                 x11b=0, # rad, radians
                 x12a=0, # rad, radians
                 x12b=0 # rad, radians
                 ):
        """
        Initialises a tls object with the instrument errors.
        Input:
            offset (metres, 3-tuple of float in x,y,z, default=(0,0,0))
            x1n (metres, float, default=-0.9e3)
            x1z (metres, float, default=-0.2e-3)
            x2 (metres, float, default=-0.9e3)
            x3 (metres, float, default=-0.3e-3)
            x4 (radians, float, default=8.29031e-5)
            x5n (radians, float, default=5.09054e-5)
            x5z (radians, float, default=4.489375e-4)
            x6 (radians, float, default=2.5695e-5)
            x7 (radians, float, default=4.489375e-4)
            x8x (none, float, default=0)
            x8y (none, float, default=0)
            x9n (none, float, default=0)
            x9z (none, float, default=0)
            x10 (metres, float, default=0)
            x11a (radians, float, default=0)
            x11b (radians, float, default=0)
            x12a (radians, float, default=0)
            x12b (radians, float, default=0)
        Output:
            none
        """
        self.offset=offset # m, metres
        self.x1n = x1n # m, metres
        self.x1z = x1z # m, metres
        self.x2 = x2 # m, metres
        self.x3 = x3 # m, metres
        self.x4 = x4 # rad, radians
        self.x5n = x5n # rad, radians
        self.x5z = x5z # rad, radians
        self.x6 = x6 # rad, radians
        self.x7 = x7 # rad, radians
        self.x8x = x8x # none
        self.x8y = x8y # none
        self.x9n = x9n # none
        self.x9z = x9z # none
        self.x10 = x10 # m, metres
        self.x11a = x11a # rad, radians
        self.x11b = x11b # rad, radians
        self.x12a = x12a # rad, radians
        self.x12b = x12b # rad, radians
        
    def scan(self, xdata, ydata, zdata):
        """
        Returns the R, Hangle, and Vangle data measured by the TLS from xdata,
        ydata, and zdata. Assigns values to R, Hangle, and Vangle error arrays.
        Input:
            xdata (metres, np array of float)
            ydata (metres, np array of float)
            zdata (metres, np array of float)
        Output:
            Rdata (metres, np array of float)
            Hdata (radians, np array of float)
            Vdata (radians, np array of float)
        """
        if xdata.shape != ydata.shape or xdata.shape != zdata.shape or ydata.shape != zdata.shape:
            raise ValueError('data arrays do not have the same dimensions')
        
        self.xdata = xdata - self.offset[0] # m, metres
        self.ydata = ydata - self.offset[1] # m, metres
        self.zdata = zdata - self.offset[2] # m, metres
        
        self.xerr = 0 # m, metres
        self.yerr = 0 # m, metres
        self.zerr = 0 # m, metres
        
        self.Rdata = np.sqrt(self.xdata**2 + self.ydata**2 + self.zdata**2) # m, metres
        self.Hdata = np.arctan2(self.ydata, self.xdata) # rad, radians
        self.Vdata = np.arccos(self.zdata/self.Rdata) # rad, radians
        
        self.Rerr = self.x2*np.sin(self.Vdata) + self.x10 # m, metres
        self.Herr = (self.x1z/(self.Rdata*np.tan(self.Vdata)) + self.x3/(self.Rdata*np.sin(self.Vdata)) + self.x5z/np.tan(self.Vdata) + 2*self.x6/np.sin(self.Vdata) - self.x7/np.tan(self.Vdata) - self.x8x*np.sin(self.Hdata) + self.x8y*np.cos(self.Hdata)) + (self.x1n/self.Rdata + self.x5n + self.x11a*np.cos(2*self.Hdata) + self.x11b*np.sin(2*self.Hdata)) # rad, radians
        self.Verr = (self.x1n*np.cos(self.Vdata)/self.Rdata + self.x2*np.cos(self.Vdata)/self.Rdata + self.x4 + self.x5n*np.cos(self.Vdata) + self.x9n*np.cos(self.Vdata)) + (-self.x1z*np.sin(self.Vdata)/self.Rdata - self.x5z*np.sin(self.Vdata) - self.x9z*np.sin(self.Vdata) + self.x12a*np.cos(2*self.Vdata) + self.x12b*np.sin(2*self.Vdata)) # rad, radians
        
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
    
    def unscan(self, Rdata, Hdata, Vdata, Rerr=0, Herr=0, Verr=0):
        """
        Returns the x, y , and z data derived from Rdata, Hdata, and Vdata.
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
        if Rdata.shape != Hdata.shape or Rdata.shape != Vdata.shape or Hdata.shape != Vdata.shape:
            raise ValueError('data arrays do not have the same dimensions')
        
        self.Rdata = Rdata # m, metres
        self.Hdata = Hdata # rad, radians
        self.Vdata = Vdata # rad, radians
        
        self.Rerr = Rerr # m, metres
        self.Herr = Herr # rad, radians
        self.Verr = Verr # rad, radians
        
        self.xdata = self.Rdata*np.sin(self.Vdata)*np.cos(self.Hdata) + self.offset[0] # m, metres
        self.ydata = self.Rdata*np.sin(self.Vdata)*np.sin(self.Hdata) + self.offset[1] # m, metres
        self.zdata = self.Rdata*np.cos(self.Vdata) + self.offset[2] # m, metres
        
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
        
    def circsection(self, radius=50, centre=(50,50), res=0.1):
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
        self.zarray[((self.xdata-radius)**2 + (self.ydata-radius)**2)>radius**2] = np.nan # m, metres
        return self.zarray
        

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
    
    def section(self, xdata, ydata):
        """
        Returns the section of ellipsoid determined by x- and y-coordinates 
        given by xdata and ydata.
        Input:
            xdata (metres, np array of float)
            ydata (metres, np array of float)
        Output:
            top (metres, np array of float)
            bottom (metres, np array of float)
        """
        if xdata.shape != ydata.shape:
            raise ValueError('xdata and ydata do not have the same dimensions')
        
        self.xdata = xdata # m, metres
        self.ydata = ydata # m, metres
        self.dome = self.radius[2] * np.sqrt((1 - ((self.xdata - self.centre[0])/self.radius[0])**2 - ((self.ydata - self.centre[1])/self.radius[1])**2)) # m, metres
        self.top = self.centre[2] + self.dome # m, metres
        self.bottom =  self.centre[2] - self.dome # m, metres
        return self.top, self.bottom     


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
