from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import numpy.linalg as la
from scipy.linalg import expm


class InvalidR3VectorException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.msg = "Invalid vector!"


def gravity(latitude):
    # Calculate the gravity of Atlanta
    g_e = 9.780327
    beta_1 = 5.30244e-3
    beta_2 = -5.8e-6
    return g_e * (1 + beta_1 * np.sin(latitude) ** 2 + beta_2 * np.sin(2 * latitude) ** 2)


class Model:

    def __init__(self, linear=True, control_affine=True):
        self.linear = linear
        self.control_affine = control_affine
        self.jacobian = lambda: None  # df / dx
        self.input_gradient = lambda: None  # dg / dx
        self.A = None
        self.B = None
        self.C = None  # optional
        self.shape = (0, 0)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def A_(self, A):
        if isinstance(A, np.ndarray) and self.linear:
            self.A = A
            self.shape_(A.shape)

    def B_(self, B):
        if isinstance(B, np.ndarray) and self.control_affine:
            self.B = B

    def C_(self, C):
        if isinstance(C, np.ndarray):
            self.C = C

    def shape_(self, shape):
        self.shape = shape

    def jacobian_(self, jac):
        self.jacobian = jac

    def jacobian(self, x):
        ...

    def predict(self, *args, **kwargs) -> Any:
        """ Child classes should implement prediction logic. """
        ...


class BlimpModel(Model):
    state_index_to_name_map = {
        0: '$v_x$',
        1: '$v_y$',
        2: '$v_z$',
        3: '$\omega_{\\theta}$',
        4: '$\omega_{\phi}$',
        5: '$\omega_{\psi}$',
        6: '$x$',
        7: '$y$',
        8: '$z$',
        9: '$\\theta$',    # Roll
        10: '$\phi$',      # Pitch
        11: '$\psi$',      # Yaw
    }
    state_name_to_index_map = {v: k for k, v in state_index_to_name_map.items()}

    def __init__(self, clip_grad=True):
        super()
        self.clip_grad = clip_grad
        self.linear = False
        self.control_affine = True
        self.shape_((12, 12))  # six degrees of freedom

        atl_latitude = 33.74888889 * np.pi / 180
        self.g = gravity(atl_latitude)  # WSG2 model. Standard g=9.80665 m/s^2

        # Setup constants
        self.r_b_z_tb = None
        self.M_inv = None
        self.HDH = None
        self.I = None
        self.mass_matrix = None
        self.r_b_z_gb = None
        self.m_RB = None

        # export extra values for the swing reducing flight controller.
        self.theta_setpoint_coefficient = None
        self.inner_loop_coefficient = None

        self._load_constants()
        self.jacobian_(self.jacobian)
        self._setup_B()

    @staticmethod
    def skew(v):  # valid for vectors in R3.
        if v.size == 3:
            return np.array([[0, -v[2, 0], v[1, 0]],
                             [v[2, 0], 0, -v[0, 0]],
                             [-v[1, 0], v[0, 0], 0]])
        else:
            raise InvalidR3VectorException

    @classmethod
    def H(cls, v):
        return np.block([[np.eye(3), BlimpModel.skew(v).T],
                         [np.zeros((3, 3)), np.eye(3)]])

    def _load_constants(self):
        m_RB = 0.1249  # kg
        m_Ax = 0.0466  # kg
        m_Az = 0.0545  # kg
        r_b_z_gb = 0.09705  # m
        r_b_z_tb = 0.12  # m, TODO MEASURE.
        D_CB_vx = 0.0125  # unitless
        D_CB_vz = 0.0480  # unitless
        D_CB_wy = 0.000980  # N * m * s / rad
        I_CG_y = 0.005821  # kg * m^2

        I_b__gA = np.zeros((3, 3))
        I_b__gRB = np.diag([I_CG_y, I_CG_y, I_CG_y])
        m_A = np.diag([m_Ax, m_Ax, m_Az])
        r_b__bg = np.array([[0],
                            [0],
                            [r_b_z_gb]])
        D_CB = np.diag([D_CB_vx, D_CB_vx, D_CB_vz, D_CB_wy, D_CB_wy, D_CB_wy])

        M1 = np.block([[m_RB * np.eye(3), np.zeros((3, 3))],
                       [np.zeros((3, 3)), I_b__gRB]])
        M2 = np.block([[m_A, np.zeros((3, 3))],
                       [np.zeros((3, 3)), I_b__gA]])

        # export whatever we need
        self.r_b_z_tb = r_b_z_tb
        self.M_inv = np.linalg.inv(M1 + M2)
        self.HDH = self.H(-r_b__bg).T @ D_CB @ self.H(-r_b__bg)
        self.I = np.diag(I_b__gA + I_b__gRB)
        self.mass_matrix = m_RB * np.eye(3) + m_A

        self.r_b_z_gb = r_b_z_gb
        self.m_RB = m_RB

        # export extra values for the swing reducing flight controller.
        self.theta_setpoint_coefficient = (D_CB_vx * self.r_b_z_tb) / (r_b_z_gb * m_RB * self.g)  # exported.
        self.inner_loop_coefficient = (r_b_z_gb / self.r_b_z_tb) * m_RB * self.g  # exported.

    def _setup_B(self):
        # This sets up the input matrix
        to_4 = np.array([[1, 0, 0, 0, self.r_b_z_tb, 0],
                         [0, 1, 0, self.r_b_z_tb, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1]]).T
        zeros_padding = np.zeros((6, 4))
        M_inv_blocked = np.block([[self.M_inv, np.zeros((6, 6))],
                                  [np.zeros((6, 12))]])
        self.B_(M_inv_blocked @ np.block([[to_4], [zeros_padding]]))  # -self.M_inv @
        return

    def discretize(self, dT):
        # Change the Jacobian and B matrix appropriately.
        self.jacobian_(lambda x: expm(dT * self.jacobian(x)))
        self.B_(self.B * dT)

    def get_grad(self, x):
        v = x[0:3].squeeze(1)
        w = x[3:6].squeeze(1)
        theta = x[9:12].squeeze(1)

        # inertial_gradient_matrix = np.diag([self.I[2]-self.I[1], self.I[0]-self.I[2], self.I[1]-self.I[0]])

        # grad_Iw is in R 3x3
        grad_v_M = self.mass_matrix @ self.skew(np.array([w]).T)
        grad_w_M = -self.mass_matrix @ self.skew(np.array([v]).T)
        grad_v_I = np.zeros((3, 3))
        grad_w_I = np.zeros((3, 3))  # inertial_gradient_matrix @ np.abs(self.skew(np.array([w]).T))

        temp_grad = self.HDH + np.block([[grad_v_M, grad_w_M],
                                         [grad_v_I, grad_w_I]])

        free_jacobian = -self.M_inv @ temp_grad

        angle_jacobian = np.array([[-np.cos(theta[0]) * np.cos(theta[1]),
                                    np.sin(theta[0]) * np.sin(theta[1]), 0],
                                   [0, -np.cos(theta[1]), 0],
                                   [0, 0, 0]]) * self.r_b_z_gb * self.m_RB * self.g

        # Integrators portion
        integrator_jacobian = self.M_inv @ np.block([[np.zeros((3, 6))],
                                                     [np.zeros((3, 3)), angle_jacobian]])

        return np.block([[free_jacobian, integrator_jacobian],
                         [np.eye(6), np.zeros((6, 6))]])

    def jacobian(self, x: np.ndarray):
        grad = self.get_grad(x)
        if self.clip_grad:
            grad = np.clip(grad, -0.99, +0.99)   # TODO: What should these bounds be
        return grad

    def predict(self, X, U, dT):
        I = np.eye(12)
        X_next = (I + dT * self.jacobian(X)) @ X + dT * self.B @ U
        return X_next


class BlimpSim:
    model = BlimpModel(clip_grad=True)

    @classmethod
    def step(cls, X, U, dT):
        if not X.ndim == 2:
            X = X.reshape(-1, 1)
        if not U.ndim == 2:
            U = U.reshape(-1, 1)
        return cls.model.predict(X, U, dT).reshape(-1)


if __name__ == """__main__""":
    # Quick test
    sim = BlimpSim()
    X = np.zeros(12)
    U = np.array([1, 0, 0, 0])
    dT = 0.01
    for i in range(100):
        X = sim.step(X, U, dT)
        print(f"[{i}] X: {np.array2string(X, precision=2, separator=',', suppress_small=True)}")
