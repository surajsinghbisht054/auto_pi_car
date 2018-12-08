#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si


def bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
                  False - Curve is open
    """

    # If periodic, extend the point array by count+degree+1
    cv = np.asarray(cv)
    count = len(cv)

    if periodic:
        factor, fraction = divmod(count+degree+1, count)
        cv = np.concatenate((cv,) * factor + (cv[:fraction],))
        count = len(cv)
        degree = np.clip(degree,1,degree)

    # If opened, prevent degree from exceeding count-1
    else:
        degree = np.clip(degree,1,count-1)


    # Calculate knot vector
    kv = None
    if periodic:
        kv = np.arange(0-degree,count+degree+degree-1)
    else:
        kv = np.clip(np.arange(count+degree+1)-degree,0,count-degree)

    # Calculate query range
    u = np.linspace(periodic,(count-degree),n)


    # Calculate result
    return np.array(si.splev(u, (kv,cv.T,degree))).T.astype(int)


ary = np.load('img.npy')


part1 = ary[:, :640]
part2 = ary[:, 640:]

#print part1.shape
#print part2.shape
co1 = np.argwhere(part1==1)
co2 = np.argwhere(part2==1)

#co1 = bspline(co1, degree=5)
#print co1[:, 0]

#b1 = np.ones(part1.shape)
#b2 = np.ones(part2.shape)
#for a,b in co1:
#	b1[a:a+5][b:b+5]=1


#f = plt.figure()
#f.add_subplot(2,2,1)
#plt.imshow(b1)

#f.add_subplot(2,2,2)
#plt.imshow(b2)

#f.add_subplot(2,2,3)
#plt.imshow(part1)
#
#f.add_subplot(2,2,4)
#plt.imshow(part2)
#plt.show()

colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

cv = np.array([[ 50.,  25.],
   [ 59.,  12.],
   [ 50.,  10.],
   [ 57.,   2.],
   [ 40.,   4.],
   [ 40.,   14.]])

plt.plot(co1[:,0],co1[:,1], 'o-', label='Control Points')

for d in range(20, 5):
    p = bspline(co1,n=100,degree=d,periodic=True)
    x,y = p.T
    plt.plot(x,y,'k-',label='Degree %s'%d,color=colors[d%len(colors)])

plt.minorticks_on()
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
#plt.xlim(35, 70)
#plt.ylim(0, 30)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
