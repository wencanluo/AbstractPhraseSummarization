import numpy as np
from scipy import spatial

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi
    return angle

def cosine_similarity(v1, v2):
    c = spatial.distance.cosine(v1, v2)
    if np.isnan(c).any():
        return 1.0
    return 1 - c

if __name__ == '__main__':
    print cosine_similarity([1.0, 0], [0, 1.0])
    print cosine_similarity([0, 0], [0, 0])
    print cosine_similarity([1, 1], [1, 1])
    print cosine_similarity([1, 0], [1, 0])
    