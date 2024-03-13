import numpy as np

# based on https://en.wikipedia.org/wiki/Polyomino
POLYOMINOES = [2, 3, 4, 5, 6]
ADJUSTED_PROB = [0.1, 0.1, 0.2, 0.3, 0.3]

def random_polyomino(polyominoes, adjusted_prob=None):
    '''
    Generate a random polyomino.

    Parameters
    ----------
        `polyominoes` : list of integers
            the possible sizes of the random polyomino.
        `adjusted_prob` : list of floats, optional
            the probability of each polyomino size.
    
    Returns
    -------
        `A` : 3-dimensional matrix of size `(n x n x n)`
            a randomly generated polyomino.
    '''

    # create an empty n x n x n matrix
    n = np.random.choice(polyominoes, p=adjusted_prob)
    A = np.zeros((n, n, n))

    # for generating random directions
    x, y, z = 0, 0, 0
    #             x+     x-    y+     y-    z+,    z-
    dir_avail = [True, False, True, False, True, False]

    # place n blocks
    for _ in range(n):
        # mark position
        A[x, y, z] = 1

        # get random direction
        dir = np.random.choice(np.where(dir_avail)[0])

        # move in direction
        if dir == 0:
            x += 1
        elif dir == 1:
            x -= 1
        elif dir == 2:
            y += 1
        elif dir == 3:
            y -= 1
        elif dir == 4:
            z += 1
        elif dir == 5:
            z -= 1
        
        # update available directions
        if x + 1 >= n or A[x + 1, y, z] != 0:
            dir_avail[0] = False
        else:
            dir_avail[0] = True
        if x - 1 < 0 or A[x - 1, y, z] != 0:
            dir_avail[1] = False
        else:
            dir_avail[1] = True
        if y + 1 >= n or A[x, y + 1, z] != 0:
            dir_avail[2] = False
        else:
            dir_avail[2] = True
        if y - 1 < 0 or A[x, y - 1, z] != 0:
            dir_avail[3] = False
        else:
            dir_avail[3] = True
        if z + 1 >= n or A[x, y, z + 1] != 0:
            dir_avail[4] = False
        else:
            dir_avail[4] = True
        if z - 1 < 0 or A[x, y, z - 1] != 0:
            dir_avail[5] = False
        else:
            dir_avail[5] = True

    # return random shape
    return A
