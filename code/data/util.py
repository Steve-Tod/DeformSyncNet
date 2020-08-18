import torch
import random
import numpy as np

# Normalization

def center_bounding_box(points):
    # input : Numpy Tensor N_pts, D_dim
    # ouput : Numpy Tensor N_pts, D_dim
    # Center bounding box of first 3 dimensions
    if isinstance(points, torch.Tensor):
        min_vals = torch.min(points, 0)[0]
        max_vals = torch.max(points, 0)[0]
        points = points - (min_vals + max_vals) / 2
        return points, (min_vals + max_vals) / 2, (max_vals - min_vals) / 2
    elif isinstance(points, np.ndarray):
        min_vals = np.min(points, 0)
        max_vals = np.max(points, 0)
        points = points - (min_vals + max_vals) / 2
        return points, (min_vals + max_vals) / 2, (max_vals - min_vals) / 2
    else:
        print(type(points))
        print("Pierre-Alain was right.")
        
def BoundingBox(points):
    points, _, bounding_box = center_bounding_box(points)
    diameter = torch.max(bounding_box)
    points = points / diameter
    return points

############
# re-sample
############

class Walkerrandom:
    """ Walker's alias method for random objects with different probablities
    http://code.activestate.com/recipes/576564-walkers-alias-method-for-random-objects-with-diffe/
    """

    def __init__(self, weights, keys=None):
        """ builds the Walker tables prob and inx for calls to random().
        The weights (a list or tuple or iterable) can be in any order;
        they need not sum to 1.
        """

        n = self.n = len(weights)
        self.keys = keys
        sumw = sum(weights)
        prob = [w * n / sumw for w in weights]  # av 1
        inx = [-1] * n
        short = [j for j, p in enumerate(prob) if p < 1]
        long = [j for j, p in enumerate(prob) if p > 1]
        while short and long:
            j = short.pop()
            k = long[-1]
            # assert prob[j] <= 1 <= prob[k]
            inx[j] = k
            prob[k] -= (1 - prob[j])  # -= residual weight
            if prob[k] < 1:
                short.append(k)
                long.pop()
            #if Test:
            #    print "test Walkerrandom: j k pk: %d %d %.2g" % (j, k, prob[k])
        self.prob = prob
        self.inx = inx
        #if Test:
        #    print "test", self


    def __str__(self):
        """ e.g. "Walkerrandom prob: 0.4 0.8 1 0.8  inx: 3 3 -1 2" """

        probstr = " ".join(["%.2g" % x for x in self.prob])
        inxstr = " ".join(["%.2g" % x for x in self.inx])
        return "Walkerrandom prob: %s  inx: %s" % (probstr, inxstr)


    def random(self):
        """ each call -> a random int or key with the given probability
        fast: 1 randint(), 1 random.uniform(), table lookup
        """

        u = random.uniform(0, 1)
        j = random.randint(0, self.n - 1)  # or low bits of u
        randint = j if u <= self.prob[j] else self.inx[j]
        return self.keys[randint] if self.keys else randint


def resample_points(points, scales, n_out_points=None):
    assert (points.ndim == 3)
    assert (scales.ndim == 1)
    assert (points.shape[0] == scales.size)

    n_sets = points.shape[0]
    n_in_points = points.shape[1]
    n_dim = points.shape[2]
    if n_out_points is None:
        n_out_points = points.shape[1]

    wrand = Walkerrandom(scales)
    set_idxs = np.empty(n_out_points, dtype='int')
    index_list = [[] for _ in range(points.shape[0])]
    for i in range(n_out_points):
        set_idxs[i] = wrand.random()
        index_list[set_idxs[i]].append(i)

    point_rand_idxs_in_sets = np.random.randint(n_in_points, size=n_out_points)

    sample_points = points[set_idxs, point_rand_idxs_in_sets, :]
    return sample_points, set_idxs, index_list