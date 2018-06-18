#### 1. Import the numpy package under the name `np` (★☆☆)

import numpy as np

#### 2. Print the numpy version and the configuration (★☆☆)

print(np.__version__,np.show_config)

#### 3. Create a null vector of size 10 (★☆☆)

np.zeros(10)

#### 4.  How to find the memory size of any array (★☆☆)

Z = np.zeros((10,10))
print("%d bytes" % (Z.size * Z.itemsize))

#### 5.  How to get the documentation of the numpy add function from the command line? (★☆☆)

help(np.add) #Not from cmd

#or

%run `python -c "import numpy; numpy.info(numpy.add)"`


#### 6.  Create a null vector of size 10 but the fifth value which is 1 (★☆☆)

cero = np.zeros(10)
cero[4] = 1
cero

#### 7.  Create a vector with values ranging from 10 to 49 (★☆☆)

v = np.arange(10,50)
v

#### 8.  Reverse a vector (first element becomes last) (★☆☆)

rv = v[::-1]
rv

#### 9.  Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)

m = np.arange(9).reshape(3,3)
m

#### 10. Find indices of non-zero elements from \[1,2,0,0,4,0\] (★☆☆)

nz = np.nonzero([1,2,0,0,4,0])
nz

#### 11. Create a 3x3 identity matrix (★☆☆)
i = np.eye(3)
i

#### 12. Create a 3x3x3 array with random values (★☆☆)

np.random.rand(3,3,3)

#### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)

t = np.random.rand(10,10)
t

np.amax(t)
np.amin(t)

#or
t.max()
t.min()

#### 14. Create a random vector of size 30 and find the mean value (★☆☆)

p = np.random.rand(30)
p.mean()


#### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)

o = np.ones((3,3))
o
o[1,1] = 0
o

#or

o[1:-1,1:-1] = 0
o

#### 16. How to add a border (filled with 0's) around an existing array? (★☆☆)

np.pad(o,1,'constant', constant_values=0)

#### 17. What is the result of the following expression? (★☆☆)


```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
0.3 == 3 * 0.1
```

#False X nan
#True X False
#True X False
#Error X nan
#True X Cause floats aren't exactly the number they represent


#### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)

x = np.ones((5,5))
x
help(range)
for i in range(1,5):
    for j in range(4):
        if i==j+1:
            x[i,j] = i
x

#or better

Z = np.diag(1+np.arange(4),k=-1)
print(Z)
help(np.diag)


#### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)

#doesn't work
help(np.arange)
help(np.random.randint)
Z=np.random.randint(2,11,(8,8))
Z
for i in np.arange(-7,7):
    if i % 2 == 0:
        Z = np.diag((0,0),i)
    else:
        Z = np.diag((1,1),i)
Z

#works
Z = np.zeros((8,8),dtype=int)
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(Z)

#### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?

print(np.unravel_index(100,(6,7,8)))


#### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)

help(np.tile)
c=np.array([[0,1],[1,0]])
c
np.tile(c, (4,4))

#### 22. Normalize a 5x5 random matrix (★☆☆)

Z=np.random.randint(2,11,(5,5))
z = (Z - np.min(Z))/(np.max(Z)-np.min(Z))
z

#or

Zmax, Zmin = Z.max(), Z.min()
Z = (Z - Zmin)/(Zmax - Zmin)
print(Z)

#### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)

help(np.dtype)

#still don't understand that

color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)])

#### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)

Z = np.random.rand(5,3)
X = np.random.rand(3,2)

help(np.matmul)

Y = np.matmul(Z,X)
Y

#### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)

#my solution
Z = np.random.randint(1,10, (5,1))
Z

Z = np.where((Z<=8) & (Z>=3),-Z,Z)
Z

#or their solution

Z = np.arange(11)
Z[(3 < Z) & (Z <= 8)] *= -1
print(Z)

#### 26. What is the output of the following script? (★☆☆)

help(np.sum)

```python
# Author: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```
#Lesson: DO NOT import *

#### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)

Z = np.arange(10)
```python
Z**Z
2 << Z >> 2
Z << 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```

#### 28. What are the result of the following expressions?

np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)


#### 29. How to round away from zero a float array ? (★☆☆)

Z = np.random.uniform(-10,+10,10)
Z

#False
Y = np.ceil(Z)
Y

help(np.info(np.copysign)) #Change the sign of x1 to that of x2, elementwise.

print (np.copysign(np.ceil(np.abs(Z)), Z)) # takes the ceiling of the abs value
#and gives it the sign from the original number.

#### 30. How to find common values between two arrays? (★☆☆)

help(np.intersect1d)
Z1 = np.random.randint(0,10,10)
Z2 = np.random.randint(0,10,10)
print(np.intersect1d(Z1,Z2))

#### 31. How to ignore all numpy warnings (not recommended)? (★☆☆)

# Suicide mode on
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0
Z
# Back to sanity
_ = np.seterr(**defaults)

#### 32. Is the following expressions true? (★☆☆)

help(np.emath)
np.sqrt(-1) == np.emath.sqrt(-1) #No, emath gives an answer with complex numbers,
#whereas sqrt gives NaN.

#### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)


import datetime

current = np.datetime64(datetime.datetime.now())
tomorrow = current + np.timedelta64(1, 'D')
yesterday = current - np.timedelta64(1, 'D')

current
tomorrow
yesterday

#or

```python

yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today     = np.datetime64('today', 'D')
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')

#### 34. How to get all the dates corresponding to the month of July 2016? (★★☆)

C = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
C

#### 35. How to compute ((A+B)\*(-A/2)) in place (without copy)? (★★☆)

A = np.ones(3)*1
B = np.ones(3)*2
C = np.ones(3)*3

np.add(A, B, out=B)
np.divide(A, 2, out=A)
np.negative(A, out=A)
np.multiply(A,B, out=A)

#### 36. Extract the integer part of a random array using 5 different methods (★★☆)

help(np.random.rand)
C = np.random.rand(3,2)
C

#1
C1 = np.floor(C)
C1

#2
C2 = C-(C%1)
C2

#3
C3 = np.ceil(C-1)
C3

#4
np.info(np.trunc)
C4 = np.trunc(C)
C4

#5
C5 = C.astype(int)
C5

#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)

#my sol
help(np.array)
help(np.ogrid)
mgrid = np.lib.index_tricks.nd_grid()
Y = mgrid[0:5,0:5]
Z = Y[1]
Z

#the smart solution
Z = np.zeros((5,5))
Z
Z += np.arange(5)
print(Z)

#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)

help(np.fromiter)
iterable = (x for x in range(10))
A = np.fromiter(iterable, int)
A

#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)

#isn't the question poorly formulated or does ranging imply equally spaced out?
v = np.random.rand(10,1)
v

help(np.linspace)
#I don't know how to exclude the startpoint and still have it be div in 10
v = np.linspace(0,1,11,endpoint=False)[1:] #well, that makes sense, they're still
#equally spaced out though not div 10
print(Z)
Z[0]
1 - Z[9]

#### 40. Create a random vector of size 10 and sort it (★★☆)

v = np.random.rand(10)
help(np.sort)
v = np.sort(v)
v

#or

Z = np.random.random(10)
Z.sort()
print(Z)

#### 41. How to sum a small array faster than np.sum? (★★☆)

import time

J = np.arange(10)

help(np.add.reduce)

import time

start = time.clock()
F = np.add.reduce(J)
print (time.clock() - start)

start = time.clock()
S = np.sum(J)
print (time.clock() - start)


#### 42. Consider two random array A and B, check if they are equal (★★☆)

A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)

(A==B).all()

#or

# Assuming identical shape of the arrays and a tolerance for the comparison of values

help(np.allclose)
equal = np.allclose(A,B)
print(equal)

# Checking both the shape and the element values, no tolerance (values have to be exactly equal)

equal = np.array_equal(A,B)
print(equal)


#### 43. Make an array immutable (read-only) (★★☆)

Y = np.arange(10)
Y.flags.writeable = False
Y[0] = 1

#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)

import math

C = np.random.randint(0, 11, (10,2))
C

start = time.clock()
P = np.zeros((10,2))
for i in range(10):
    P[i,0] = np.sqrt((C[i,0])**2+(C[i,1])**2)
    P[i,1] = math.atan2(C[i,1],C[i,0])
P
print(time.clock() - start)

#or their solution, mine is faster in this case, but I suspect it might not
#scale so well.

start = time.clock()
X,Y = C[:,0], C[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print(R)
print(T)
print(time.clock() - start)

#### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)

T = np.random.rand(10)
T
T[T.argmax()] = 0
T

#### 46. Create a structured array with `x` and `y` coordinates covering the \[0,1\]x\[0,1\] area (★★☆)

help(np.meshgrid)
Z = np.zeros((5,5), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
Z

####  47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))

X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(A, B)
print(np.linalg.det(C))
C

#### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)

help(np.iinfo)

for dtype in [np.int16, np.int32, np.int64]:
    print(np.iinfo(dtype).min)
    print(np.iinfo(dtype).max)

for dtype in [np.float16, np.float32, np.float64]:
    print(np.finfo(dtype).min)
    print(np.finfo(dtype).max)
    print(np.finfo(dtype).eps)

#### 49. How to print all the values of an array? (★★☆)

help(np.set_printoptions)

Y = np.arange(np.random.randint(0, 10, (1,1))) #array with random length
Y

np.set_printoptions(threshold=len(Y))

#or their solution

np.set_printoptions(threshold=np.nan)
Z = np.zeros((16,16))
print(Z)

#### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)

np.set_printoptions(threshold = 1000)

help(np.argmin)

Z = np.arange(100)
v = np.random.uniform(0,100)
v
index = (np.abs(Z-v)).argmin()
print(Z[index])

#### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)



#### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)



#### 53. How to convert a float (32 bits) array into an integer (32 bits) in place?



#### 54. How to read the following file? (★★☆)


```
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```

#### 55. What is the equivalent of enumerate for numpy arrays? (★★☆)



#### 56. Generate a generic 2D Gaussian-like array (★★☆)



#### 57. How to randomly place p elements in a 2D array? (★★☆)



#### 58. Subtract the mean of each row of a matrix (★★☆)



#### 59. How to sort an array by the nth column? (★★☆)



#### 60. How to tell if a given 2D array has null columns? (★★☆)
