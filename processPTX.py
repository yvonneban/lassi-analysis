import sys
import random

# import matplotlib
# matplotlib.use("agg")

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
from astropy import units as u
from astropy.coordinates import CartesianRepresentation
from astropy.coordinates.matrix_utilities import rotation_matrix

from parabolas import scatter3dPlot

def rotateXYaboutZ(xyz, rotDegrees):

    # define it as cartesian
    rep = CartesianRepresentation(xyz[:, 0], xyz[:, 1], xyz[:, 2])

    # define our rotation about z
    # TBF: why must we negate this to match Mathematica results?
    rot = rotation_matrix(-rotDegrees * u.deg, axis='z')

    # apply the rotation
    new_rep = rep.transform(rot)

    new_xyz = new_rep.get_xyz().value

    # get it back into original form
    # TBF: must be a faster method!
    xyzNew = []
    x = new_xyz[0]
    y = new_xyz[1]
    z = new_xyz[2]
    for i in range(len(x)):
        xyzNew.append((x[i], y[i], z[i]))
        
    return np.array(xyzNew)    
        
    
def processPTX(fpath, rotationAboutZdegrees=None, searchRadius=None, rFilter=True):


    # import ipdb; ipdb.set_trace() 
    if rotationAboutZdegrees is None:
        rotationAboutZdegrees = 150.
    if searchRadius is None:    
        searchRadius = 47
   
    with open(fpath, 'r') as f:
        ls = f.readlines()
    
    xyz = processPTXdata(ls, rotationAboutZdegrees, searchRadius)

    # write to CSV file
    outf = fpath + ".csv"
    np.savetxt(outf, xyz, delimiter=",")

def processPTXdata(lines, rotationAboutZdegrees, searchRadius, quiet=True, sampleSize=None, preview=True):

    ls = lines

    if not quiet:
        print "Original file num lines: ", len(ls)

    #badLine = '0 0 0 0.500000\r\n'
    badLine = '0 0 0 0.500000\n'
    # reject the '0 0 0 0.500000' lines - TBF: what are those?
    ls = [l for l in ls if l != badLine]
    #for l in ls:
    #    print "'%s' vs '%s'" % (l, badLine)

    if not quiet:
        print "File minus '0 0 0 0.50000' lines: ", len(ls)

    #print "beginning of file: "
    #print ls[:12]

    # remove the header
    # ls = ls[10:]
    ls = ls[11:]
    # import ipdb; ipdb.set_trace()
    #print "now beginning of file w/ out header: "
    #print ls[:12]

    # parse strings so we can ignore the fourth value
    sample = 1000
    xyz = []
    for i, l in enumerate(ls):
        # print i, l
        if "Line" in l:
            # print l
            continue
        ll = l.split(' ')
        x = float(ll[0])
        y = float(ll[1])
        z = float(ll[2])
        xyz.append((x, y, z))

    xyz = np.array(xyz)    

    # print "Skipping rotation and filtering!"
    # return xyz

    # rotation!  This takes a long time.  TBF: why? *)
    #rot=AffineTransform[{RotationMatrix[150Degree,{0,0,1}],{0,0,0}}];
    #lall=Map[rot,lall];
    xyz = rotateXYaboutZ(xyz, rotationAboutZdegrees)

    # and we only use those parts that are within this radius TBF? *)
    # ls=Select[ls,Norm[Take[#,2]-{-54,0}]<47&]
    # Here's what's going on:
    # Select will go through and apply the predicat # < 47 to each element of 'ls'
    # but, in this case each element, #, is passed to Norm[Take[#,2]-{-54, 0}]
    # What's happening there is that the x, y from each element in xyz above is taken,
    # and 54 is added to each x.  Norm == Sqrt(x**2 + y**2); that looks like a radius to me.
    # so, if the radius is less the 47, this data element is kept.


    if preview:
        # a preview of what our processed data looks like
        sampleSize = 10000
        lsIdx = random.sample(range(len(xyz)), sampleSize)
        xyzr = xyz[lsIdx]
        xr, yr, zr = splitXYZ(xyzr)
        xlim = (-100, 100)
        ylim = (-100, 100)
        scatter3dPlot(xr, yr, zr, "Sampling of processed data", xlim=xlim, ylim=ylim)
        f = plt.figure()
        ax = f.gca()
        # take a look at the x y orientation
        ax.plot(xr, yr, 'bo', [0], [0], '*')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("X Y orientation of data")

    return filterOutRadius(xyz, searchRadius=searchRadius)
    
def filterOutRadius(xyz, searchRadius=None, mysteryX=None):
    "return only those xyz points where sqrt(x**2 + y**2) is within a limit"

    if mysteryX is None:
        mysteryX = 54
    if searchRadius is None:    
        searchRadius = 47

    xyzInside = []
    for x, y, z in xyz:
        x2 = x + mysteryX
        r = np.sqrt(x2**2 + y**2)
        # if r < searchRadius and r > 20:
        if r < searchRadius:
        #if z > 0. and z < 60.:
            xyzInside.append((x, y, z))
    xyzInside = np.array(xyzInside)

    return xyzInside

def getRawXYZsample(fn, sampleRate):
    with open(fn, 'r') as f:
        ls = f.readlines()

    # skip the header
    ls = ls[11:]

    xs = []
    ys = []
    zs = []

    for i, l in enumerate(ls):
        # print i, l
        if "Line" in l:
            # print l
            continue
        # TBF: just take a sample
        if i % sampleRate != 0:
            continue    
        ll = l.split(' ')
        x = float(ll[0])
        y = float(ll[1])
        z = float(ll[2])
        xs.append(x)
        ys.append(y)
        zs.append(z)

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    return xs, ys, zs

def getRawXYZ(ls, sampleSize=None):
    # with open(fn, 'r') as f:
        # ls = f.readlines()

    # skip the header
    ls = ls[11:]

    numLines = len(ls)

    if sampleSize is not None:
        print "Picking %d random data points from a total of %d" % (sampleSize, numLines)
        lsIdx = random.sample(range(numLines), sampleSize)
    else:
        lsIdx = range(numLines)

    xs = []
    ys = []
    zs = []
    it = []

    numLines = 0

    # for i, l in enumerate(ls):
    for i in lsIdx:
        l = ls[i]

        # print i, l
        if "Line" in l:
            # print l
            numLines += 1
            continue

        ll = l.split(' ')
        x = float(ll[0])
        y = float(ll[1])
        z = float(ll[2])
        i = float(ll[3])
        xs.append(x)
        ys.append(y)
        zs.append(z)
        it.append(i)

    print "Skipped %d non-data lines" % numLines

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    it = np.array(it)

    return xs, ys, zs, it

def testOffsets(lines, xOffset, yOffset, radius):
    "creates plots to make sure we are centered"

    sampleSize = 10000
    x, y, z, _ = getRawXYZ(lines, sampleSize=sampleSize)

    # plot the random sample of xyz data
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Random sample of %d points" % sampleSize)

    # plot the raw data including where the origin is
    f = plt.figure()
    ax = f.gca()
    ax.plot(x, y, 'bo', [0], [0], '*')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("X Y orientation of data")

    # translate data
    xo = x - xOffset
    yo = y - yOffset

    xr = []
    yr = []
    xro = []
    yro = []
    for i in range(len(xo)):
        thisX = xo[i]
        thisY = yo[i]
        r = np.sqrt(thisX**2 + thisY**2)
        if r < radius:
            xr.append(thisX)
            yr.append(thisY)
        else:
            xro.append(thisX)
            yro.append(thisY)
    xr = np.array(xr)
    yr = np.array(yr)
    xro = np.array(xro)
    yro = np.array(yro)

    f = plt.figure()
    ax = f.gca()
    ax.plot(xr, yr, 'o', xro, yro, 'ro', [0], [0], 'r*')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("XY centered and below radus: %5.2f" % radius)

def radialFilter(x, y, z, xOffset, yOffset, radius):
    "returns only those points within the radius"
    xr = []
    yr = []
    zr = []
    for i in range(len(x)):
        r = np.sqrt((x[i]-xOffset)**2 + (y[i]-yOffset)**2)
        if r < radius:
            xr.append(x[i])
            yr.append(y[i])
            zr.append(z[i])
    xr = np.array(xr)        
    yr = np.array(yr)        
    zr = np.array(zr)        
    return xr, yr, zr

def processNewPTXData(lines,
                      xOffset=None,
                      yOffset=None,
                      plotTest=True,
                      rot=None,
                      sampleSize=None,
                      iFilter=False,
                      rFilter=True):
    "this is the processing we see works with 2019 data"

    if rot is None:
        rot = -90.0
    if xOffset is None:
        xOffset = -8.0
    if yOffset is None:    
        yOffset = 50.0

    radius = 47.

    if plotTest:
        # make some plots that ensure how we are doing
        # our radial filtering
        testOffsets(lines, xOffset, yOffset, radius)

    # get the actual float values from the file contents
    x, y, z, i = getRawXYZ(lines, sampleSize=sampleSize)

    print "Starting with %d lines of data" % len(x)


    # lets first just remove all the zero data
    mask = i != 0.0
    i = i[mask]

    numFilteredOut = len(x) - len(i)
    percent = (float(numFilteredOut) / float(len(x))) * 100.
    print "Filtered out %d points of %d (%5.2f%%) intensity equal to zero" % (numFilteredOut, len(x), percent)

    x = x[mask]
    y = y[mask]
    z = z[mask]

    print "Now we have %d lines of data" % len(x)

    # we only want the data that has a decent intesity
    meanI = np.mean(i)
    stdI = np.std(i)
    print "Intensity: max=%5.2f, min=%5.2f, mean=%5.2f, std=%5.2f" % (np.max(i),
                                                                      np.min(i),
                                                                      meanI,
                                                                      stdI)

    if iFilter:    
        #lowestI = meanI # - stdI
        #mask = i > lowestI
        #highest = 0.8
        #mask = i < highest
        mask = np.logical_and(i > 0.75, i < 0.85)
        i = i[mask]

        numFilteredOut = len(x) - len(i)
        percent = (float(numFilteredOut) / float(len(x))) * 100.
        #print "Filtered out %d points of %d (%5.2f%%) below intensity %5.2f" % (numFilteredOut, len(x), percent, lowestI)
        #print "Filtered out %d points of %d (%5.2f%%) higher intensity %5.2f" % (numFilteredOut, len(x), percent, highest)
        print "Filtered out %d points of %d (%5.2f%%) via intensity" % (numFilteredOut, len(x), percent)

        x = x[mask]
        y = y[mask]
        z = z[mask]

        print "Now we have %d lines of data" % len(x)

    assert len(x) == len(y)
    assert len(y) == len(z)
    assert len(z) == len(i)

    # we only want the inner 90% or so of the dish
    if rFilter:
        orgNum = len(x)
        x, y, z =  radialFilter(x, y, z, xOffset, yOffset, radius)
        newNum = len(x)
        print "radial limit filtered out %d points outside radius %5.2f" % ((orgNum - newNum), radius)
        print "Now we have %d lines of data" % len(x)

    # TBF: why must we do this?  No idea, but this,
    # along with a rotation of -90. gets our data to
    # look just like the 2016 data at the same stage
    z = -z

    # z - filter: at this point we should have the
    # dish, but with some things the radial filter didn't
    # get rid of above or below the dish
    zLimit = -80
    mask = z > -80
    orgNum = len(z)
    x = x[mask]
    y = y[mask]
    z = z[mask]
    newNum = len(z)
    print "z - limit filtered out %d points below %5.2f" % ((orgNum - newNum), zLimit)

    # x, y, z -> [(x, y, z)]
    # for rotation phase
    xyz = []
    for i in range(len(x)):
        xyz.append((x[i], y[i], z[i]))
    xyz = np.array(xyz)

    if rot is not None or rot != 0.0:
        print "Rotating about Z by %5.2f degrees" % rot
        rotationAboutZdegrees = rot
        xyz = rotateXYaboutZ(xyz, rotationAboutZdegrees)
    
    print "Now we have %d lines of data" % len(xyz)

    if plotTest:
        # we plotted stuff earlier, so let's get
        # a preview of what our processed data looks like
        if sampleSize is None:
            sampleSize = 10000
            lsIdx = random.sample(range(len(xyz)), sampleSize)
            xyzr = xyz[lsIdx]
            xr, yr, zr = splitXYZ(xyzr)
        else:
            xr, yr, zr = splitXYZ(xyz)    
        xlim = (-100, 100)
        ylim = (-100, 100)
        scatter3dPlot(xr, yr, zr, "Sampling of processed data", xlim=xlim, ylim=ylim)
        f = plt.figure()
        ax = f.gca()
        # take a look at the x y orientation
        ax.plot(xr, yr, 'bo', [0], [0], '*')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("X Y orientation of data")

    return xyz

def aggregateXYZ(x, y, z):
    xyz = []
    for i in range(len(x)):
        xyz.append((x[i], y[i], z[i]))
    return np.array(xyz)

def splitXYZ(xyz):
    x = []
    y = []
    z = []

    for xi, yi, zi in xyz:
        x.append(xi); y.append(yi); z.append(zi)
        
    x = np.array(x)
    z = np.array(z) 
    y = np.array(y)    
    return x, y, z

def processNewPTX(fpath,
                  rot=None,
                  sampleSize=None,
                  iFilter=False,
                  rFilter=True):

    with open(fpath, 'r') as f:
        ls = f.readlines()
    
    xyz = processNewPTXData(ls,
                            rot=rot,
                            sampleSize=sampleSize,
                            iFilter=iFilter,
                            rFilter=rFilter)

    # TBF: the old interface expects this 
    # in a different format
    # xyz = []
    # for i in range(len(x)):
    #     xyz.append((x[i], y[i], z[i]))
    # xyz = np.array(xyz)

    # write to CSV file
    outf = fpath + ".csv"
    np.savetxt(outf, xyz, delimiter=",")

if __name__ == "__main__":
    import os
    dataPath = "/home/sandboxes/jbrandt/Telescope27Mar2019"
    refFn = "Scan-9_5100x5028_20190327_1145_ReIoNo_ReMxNo_ColorNo_.ptx"
    fn = os.path.join(dataPath, refFn)

    #fpath = "/home/sandboxes/pmargani/LASSI/data/LeicaDropbox/PTX/Test1_STA14_Bump1_High-02_METERS.ptx"
    #fpath = sys.argv[1]
    #testRotation(90.)
    #fpath = "data/Test1_STA14_Bump1_Hig-02_METERS_SAMPLE.ptx"
    processPTX(fn)

