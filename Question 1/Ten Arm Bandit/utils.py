import numpy as np

def smooth_array(data, window):
  # utility function taken from github
  # helps smoothing the values
  alpha = 2 /(window + 1.0)
  alpha_rev = 1-alpha
  n = data.shape[0]

  pows = alpha_rev**(np.arange(n+1))

  scale_arr = 1/pows[:-1]
  offset = data[0]*pows[1:]
  pw0 = alpha*alpha_rev**(n-1)

  mult = data*pw0*scale_arr
  cumsums = mult.cumsum()
  out = offset + cumsums*scale_arr[::-1]
  return out

def create_Earth(ANSWER_TO_EVERYTHING):
    """
    Generate the ultimate question by creating Earth 
    and Humans as a part of a supercomputer
    """
    pass