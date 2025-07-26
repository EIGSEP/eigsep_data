import numpy as np


def get_rotation_matrix(v1, v2):
    """
    Get the rotation matrix that rotates vector v1 to vector v2 with
    Rodrigues' rotation formula.

    Parameters
    ----------
    v1 : np.ndarray
    v2 : np.ndarray

    Returns
    -------
    R : np.ndarray
        The rotation matrix that rotates v1 to v2, i.e., R @ v1 = v2.

    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    v = np.cross(v1, v2)
    s = np.linalg.norm(v)
    c = np.dot(v1, v2)

    if s < 1e-8:  # vectors are parallel or anti-parallel
        if c > 0:  # parallel
            return np.eye(3)
        # anti-parallel, 180 degree rotation around any axis orthogonal to v1
        axis = np.array([1, 0, 0])  # arbitrary axis
        v = np.cross(v1, axis)
        v /= np.linalg.norm(v)
        Kmat = _get_skew_symmetric_matrix(v)
        R = np.eye(3) + 2 * Kmat @ Kmat  # sin(pi) = 0, cos(pi) = -1
        return R

    Kmat = _get_skew_symmetric_matrix(v)
    R = np.eye(3) + Kmat + Kmat @ Kmat * (1 - c) / (s**2)
    return R


def _get_skew_symmetric_matrix(v):
    """
    Get the skew-symmetric matrix of a vector v.

    Parameters
    ----------
    v : np.ndarray
        A 3-element vector.

    Returns
    -------
    K : np.ndarray
        The skew-symmetric matrix of v.

    """
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class Imu:
    """
    Class for handling IMU data. Input data are raw readings from the
    IMU (see EIGSEP/pico-firmware for the expected format). The class
    allows defining the coordinate system and provides methods to get
    orientation in this coordinate system based on different IMU sensor
    readings (gravity, magnetic field, etc.).
    """

    def __init__(self, imu_data):
        """
        Parameters
        ----------
        imu_data : dict
            See EIGSEP/pico-firmware for the expected format.

        """
        self.data = imu_data
        # rotation matrix for coordinate system conversion
        self._coord_Rmat = np.eye(3)
        # XXX need way to set coordinate system
        # ie initial coord conversion (this vector is z-axis, etc.)

    @property
    def coord_Rmat(self):
        """Return the rotation matrix for coordinate system conversion."""
        return self._coord_Rmat

    @coord_Rmat.setter
    def coord_Rmat(self, value):
        """
        Set the rotation matrix for coordinate system conversion.

        Parameters
        ----------
        value : np.ndarray
            A 3x3 rotation matrix.

        """
        if isinstance(value, np.ndarray) and value.shape == (3, 3):
            self._coord_Rmat = value
        else:
            raise ValueError("Rotation matrix must be a 3x3 numpy array.")

    def rot_vector(self, v):
        return self.coord_Rmat @ v

    def set_reference(self, grav_ref, mag_ref=None):
        """
        Set the reference vectors for the coordinate system.

        Parameters
        ----------
        grav_ref : np.ndarray
            The direction of the gravity vector in the coordinate system.
        mag_ref : np.ndarray
            The direction of the magnetic field vector (x, north) in the
            coordinate system.

        """
        # get the matrix that rotates the gravity vector to the z-axis
        R1 = get_rotation_matrix(grav_ref, np.array([0, 0, -1]))

        if mag_ref is None:
            # default to native x-axis
            mag_ref = np.array([1, 0, 0])

        xp = R1 @ mag_ref
        # project xp onto the x-y plane
        xp = xp - np.array([0, 0, xp[2]])
        xp /= np.linalg.norm(xp)
        R2 = get_rotation_matrix(xp, np.array([1, 0, 0]))

        self.coord_Rmat = R2 @ R1

    @property
    def acceleration(self):
        """Return the acceleration vector in m/s^2."""
        return np.array(
            [self.data["accel_x"], self.data["accel_y"], self.data["accel_z"]]
        )

    @property
    def linear_acceleration(self):
        """Return the linear acceleration vector in m/s^2."""
        return np.array(
            [
                self.data["lin_accel_x"],
                self.data["lin_accel_y"],
                self.data["lin_accel_z"],
            ]
        )

    @property
    def gravity(self):
        """Return the gravity vector in m/s^2."""
        return self.acceleration - self.linear_acceleration

    def get_tilt_from_gravity(self):
        g_norm = self.gravity / np.linalg.norm(self.gravity)
        return np.arccos(-g_norm[2])  # angle with respect to z-axis
