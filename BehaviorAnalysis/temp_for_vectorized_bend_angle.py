

Vectorized calc_bend_angle() below, and a prompt to run later for vectorized
fit_y_eq_Bx_simple() 



def calc_bend_angle(x, y, M=None):
    """
    Calculate the best-fit line from points (x[M], y[M]) to (x[0], y[0]) 
    and the best-fit line from points (x[M], y[M]) to (x[-1], y[-1]), 
    Return the angle between these points in the range [0, pi].
    x and y are multidimensional -- see inputs below
    
    Bend angle defined as pi minus opening angle, 
    so that a straight fish has angle 0.
    
    inputs
    (x, y) each of shape (Nframes, N, Nfish); calculate angle for each
        frame and fish,
    M : the index of the midpoint; will use N/2 if not input
    
    output
    angle : in range [0, pi]; shape (Nframes, Nfish)
    """
    
    Nframes, N, Nfish = x.shape
    if M is None:
        M = round(N/2)
        
    # First segment: from M to 0
    x1 = x[:, M::-1, :]  # shape (Nframes, M+1, Nfish)
    y1 = -1.0 * y[:, M::-1, :]  # shape (Nframes, M+1, Nfish)
    
    # Shift coordinates to start at origin
    x1_shifted = x1 - x1[:, 0:1, :]  # broadcasting over middle dimension
    y1_shifted = y1 - y1[:, 0:1, :]
    
    # Calculate slopes for first segment
    slope1 = fit_y_eq_Bx_simple(x1_shifted, y1_shifted)  # shape (Nframes, Nfish)
    
    # Calculate direction signs using mean of differences
    signx1 = np.sign(np.mean(np.diff(x1, axis=1), axis=1))  # shape (Nframes, Nfish)
    signy1 = np.sign(np.mean(np.diff(y1, axis=1), axis=1))
    
    # Create direction vectors for first segment
    norm1 = np.sqrt(1 + slope1**2)
    vector1 = np.stack([signx1, signy1 * np.abs(slope1)], axis=1) / norm1[:, np.newaxis, :]
    
    # Second segment: from M to end
    x2 = x[:, M:, :]  # shape (Nframes, N-M, Nfish)
    y2 = -1.0 * y[:, M:, :]
    
    # Shift coordinates to start at origin
    x2_shifted = x2 - x2[:, 0:1, :]
    y2_shifted = y2 - y2[:, 0:1, :]
    
    # Calculate slopes for second segment
    slope2 = fit_y_eq_Bx_simple(x2_shifted, y2_shifted)
    
    # Calculate direction signs using mean of differences
    signx2 = np.sign(np.mean(np.diff(x2, axis=1), axis=1))
    signy2 = np.sign(np.mean(np.diff(y2, axis=1), axis=1))
    
    # Create direction vectors for second segment
    norm2 = np.sqrt(1 + slope2**2)
    vector2 = np.stack([signx2, signy2 * np.abs(slope2)], axis=1) / norm2[:, np.newaxis, :]
    
    # Calculate angle between vectors using dot product
    # Sum along axis 1 (the vector components)
    dot_product = np.sum(vector1 * vector2, axis=1)
    angle = np.arccos(dot_product)
    
    # Convert to bend angle
    angle = np.pi - angle
    
    return angle


Here is fit_y_eq_Bx_simple(). Modify it to act on the appropriate dimension.

def fit_y_eq_Bx_simple(x, y):

    """

    function to fit the equation y = Bx to arrays [x], [y] 

    (i.e. a line with intercept at the origin)

    Inputs:

        x : x array; acts on dimension 0

        y : y array

    Outputs

        B : slope

    """

    sxx = np.sum(x**2, axis=0)

    sxy = np.sum(x*y, axis=0)

    return np.where(sxx != 0, sxy / sxx, np.inf)


Claude Returns::
    
def fit_y_eq_Bx_simple(x, y):
    """ 
    function to fit the equation y = Bx to arrays [x], [y] 
    (i.e. a line with intercept at the origin)
    
    Inputs:
    x : x array; shape (Nframes, N, Nfish)
    y : y array; same shape as x
    
    Outputs
    B : slope; shape (Nframes, Nfish)
    """
    sxx = np.sum(x**2, axis=1)  # sum over N dimension
    sxy = np.sum(x*y, axis=1)   # sum over N dimension
    return np.where(sxx != 0, sxy / sxx, np.inf)
