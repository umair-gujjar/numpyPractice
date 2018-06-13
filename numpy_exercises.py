#### 1. Import the numpy package under the name `np` (★☆☆)

import numpy as np
import timeit

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



#### 36. Extract the integer part of a random array using 5 different methods (★★☆)



#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)



#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)



#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)



#### 40. Create a random vector of size 10 and sort it (★★☆)



#### 41. How to sum a small array faster than np.sum? (★★☆)



#### 42. Consider two random array A and B, check if they are equal (★★☆)



#### 43. Make an array immutable (read-only) (★★☆)



#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)



#### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)



#### 46. Create a structured array with `x` and `y` coordinates covering the \[0,1\]x\[0,1\] area (★★☆)



####  47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))



#### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)



#### 49. How to print all the values of an array? (★★☆)



#### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)



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
