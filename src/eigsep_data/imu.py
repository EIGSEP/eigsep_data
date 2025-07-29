from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R


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


@dataclass(frozen=True)
class ImuCalibrator:
    """
    Class for calibrating IMU data. It contains the rotation matrix
    that transforms the IMU data from the native coordinate system to
    the desired coordinate system.

    """
    Rmat: np.ndarray

    @classmethod
    def from_ref(cls, grav_ref, mag_ref=None):
        """
        Create an ImuCalibrator instance from the reference vectors
        specifying the direction of the gravity vector and the
        magnetic field vector in the desired coordinate system.

        Parameters
        ----------
        grav_ref : np.ndarray
            The direction of the gravity vector in the coordinate system.
        mag_ref : np.ndarray
            The direction of the magnetic field vector (x, north) in the
            coordinate system.

        Returns
        -------
        ImuCalibrator

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
        return cls(R2 @ R1)

    @classmethod
    def from_imu_data(cls, imu_data : dict):
        """
        Create an ImuCalibrator instance from an IMU reading at the
        ``home`` position, i.e., the position where the IMU is aligned
        with the desired coordinate system.

        Parameters
        ----------
        imu_data : dict
            See EIGSEP/pico-firmware for the expected format.

        Returns
        -------
        ImuCalibrator

        """ 
        grav_ref = np.array(
            [
                imu_data["accel_x"] - imu_data["lin_accel_x"],
                imu_data["accel_y"] - imu_data["lin_accel_y"],
                imu_data["accel_z"] - imu_data["lin_accel_z"],
            ]
        )
        mag_ref = np.array(
            [imu_data["mag_x"], imu_data["mag_y"], imu_data["mag_z"]]
        )
        return cls.from_ref(grav_ref, mag_ref)

    def apply(self, vec: np.ndarray) -> np.ndarray:
        """
        Apply the rotation matrix to a vector.

        Parameters
        ----------
        vec : np.ndarray
            A 3-element vector in the native coordinate system.

        Returns
        -------
        np.ndarray
            The vector in the desired coordinate system.

        """
        return self.Rmat @ vec


@dataclass
class ImuSnapshot:
    """
    Class for handling IMU data. Input data are raw readings from the
    IMU (see EIGSEP/pico-firmware for the expected format). The class
    allows using ImuCalibrator to define the coordinate system and 
    provides methods to get orientation in this coordinate system based
    on different IMU sensor readings (gravity, magnetic field, etc.).
    """

    accel : np.ndarray
    lin_accel : np.ndarray
    quat : np.ndarray
    calibrator : ImuCalibrator = ImuCalibrator(Rmat=np.eye(3))

    @classmethod
    def from_imu_data(
        cls,
        imu_data: dict,
        calibrator: ImuCalibrator = ImuCalibrator(Rmat=np.eye(3))
    ) -> "ImuSnapshot":
        """
        Create an ImuSnapshot instance from IMU data.

        Parameters
        ----------
        imu_data : dict
            See EIGSEP/pico-firmware for the expected format.

        Returns
        -------
        ImuSnapshot

        """
        return cls(
            accel=np.array(
                [imu_data["accel_x"], imu_data["accel_y"], imu_data["accel_z"]]
            ),
            lin_accel=np.array(
                [
                    imu_data["lin_accel_x"],
                    imu_data["lin_accel_y"],
                    imu_data["lin_accel_z"],
                ]
            ),
            quat=np.array(
                [
                    imu_data["quat_real"],
                    imu_data["quat_i"],
                    imu_data["quat_j"],
                    imu_data["quat_k"],
                ]
            ),
            calibrator=calibrator,
        )

    @property
    def gravity(self):
        """Return the gravity vector in m/s^2."""
        return self.accel - self.lin_accel

    def get_tilt_from_gravity(self): 
        g_norm = self.gravity / np.linalg.norm(self.gravity)
        g_norm = self.calibrator.apply(g_norm)
        th = np.arccos(-g_norm[2])  # angle with respect to z-axis
        return th
        
    def get_tilt_from_quat(self):
        Rmat = R.from_quat(self.quat).as_matrix()
        z_axis = np.array([0, 0, 1])
        g_norm = Rmat @ z_axis
        g_norm /= np.linalg.norm(g_norm)
        g_norm = self.calibrator.apply(g_norm)
        return np.arccos(-g_norm[2])
