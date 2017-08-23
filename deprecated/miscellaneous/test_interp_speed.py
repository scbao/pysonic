from scipy.interpolate import interp1d
import numpy as np
import timeit


x = np.linspace(1, 10000, 5)
f = lambda x: x**2
y = f(x)

xnew = 1234
u = timeit.timeit(lambda: np.interp(xnew, x, y))
print(u)

u = timeit.timeit(lambda: interp1d(x, y))
print(u)

h = interp1d(x, y)

u = timeit.timeit(lambda: h(xnew))
print(u)