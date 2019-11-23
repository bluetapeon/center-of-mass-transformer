import numpy as np

def prepare_weights(y):
    rval = np.zeros_like(y, dtype=np.float)
    for key, count in zip(*np.unique(y, return_counts=True)):
        np.place(rval, y == key, (1 - np.divide(count, y.shape[0])))
    return rval


def calc_center_of_mass(points, weights=None):
    if weights is None:
        weights = np.ones(points.shape[0])

    assert points.shape[0] == weights.shape[0]

    return np.divide(np.sum((points.T * weights).T), np.sum(weights))


def transform_from_cartesian_to_spherical(c_coords):
    r = np.sqrt(np.sum(np.power(c_coords, 2), axis=1))
    phi_v = []

    for _ in range(c_coords.shape[1] - 1):
        a = c_coords[:, _]
        b = np.sqrt(np.sum(np.power(c_coords[:, _:], 2), axis=1))
        phi = np.arccos(np.divide(a, b, out=np.zeros(a.shape), where=(b != 0)))
        phi_v.append(phi)

    np.subtract(np.pi * 2, phi_v[-1], out=phi_v[-1], where=(c_coords[:, -1] < 0))

    return np.stack((r, *phi_v)).T
