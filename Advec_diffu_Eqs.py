import numpy as np
import tensorflow as tf


def get_infos2Advection_1D(x_left=0.0, x_right=1.0, t_init=0.0, ws=0.1, ds=0.1, eqs_name=None):
    if eqs_name == 'Advection1':
        Utrue = lambda x, t: (0.025 / np.sqrt(0.000625 + 0.02 * t)) * np.exp(-(x + 0.5 - t) * (x + 0.5 - t) / (0.00125 + 0.04 * t))
        Uleft = lambda x, t: (0.025 / np.sqrt(0.000625 + 0.02 * t)) * np.exp(-(x_left + 0.5 - t) * (x_left + 0.5 - t) / (0.00125 + 0.04 * t))
        Uright = lambda x, t: (0.025 / np.sqrt(0.000625 + 0.02 * t)) * np.exp(-(x_right + 0.5 - t) * (x_right + 0.5 - t) / (0.00125 + 0.04 * t))
        Uinit = lambda x, t: (0.025 / np.sqrt(0.000625 + 0.02 * t_init)) * np.exp(-(x + 0.5 - t_init) * (x + 0.5 - t_init) / (0.00125 + 0.04 * t_init))
        return Utrue, Uleft, Uright, Uinit
    elif eqs_name == 'Advection2':
        Utrue = lambda x, t: 1.0 / np.sqrt(4.0 * t + 1) * np.exp(-(x - 1.0 - ws*t) * (x - 1.0 - ws*t) / (ds*4.0 * t + ds))
        Uleft = lambda x, t: 1.0 / np.sqrt(4.0 * t + 1) * np.exp(-(x_left - 1.0 - ws*t) * (x_left - 1.0 - ws*t) / (ds*4.0 * t + ds))
        Uright = lambda x, t: 1.0 / np.sqrt(4.0 * t + 1) * np.exp(-(x_right - 1.0 - ws*t) * (x_right - 1.0 - ws*t) / (ds*4.0 * t + ds))
        Uinit = lambda x, t: 1.0 / np.sqrt(4.0 * t_init + 1) * np.exp(-(x - 1.0 - ws*t_init) * (x - 1.0 - ws*t_init) / (ds*4.0 * t_init + ds))
        return Utrue, Uleft, Uright, Uinit


def get_infos2Advection_2D(x_left=0.0, x_right=1.0, y_bottom=0.0, y_top=1.0, t_init=0.0, ws=0.1, ds=0.1, eqs_name=None):
    return 0


def get_infos2Convection_3D(input_dim=1, out_dim=1, mesh_number=2, intervalL=0.0, intervalR=1.0, equa_name=None):
    if equa_name == 'Convection3D':
        # mu1= 2*np.pi
        # mu2 = 4*np.pi
        # mu3 = 8*np.pi
        # mu1 = np.pi
        # mu2 = 5 * np.pi
        # mu3 = 10 * np.pi
        mu1 = np.pi
        mu2 = 10 * np.pi
        mu3 = 20 * np.pi
        f = lambda x, y, z: (mu1*mu1+mu2*mu2+mu3*mu3+x*x+2*y*y+3*z*z)*tf.sin(mu1*x)*tf.sin(mu2*y)*tf.sin(mu3*z)
        A_eps = lambda x, y, z: 1.0*tf.ones_like(x)
        kappa = lambda x, y, z: x*x+2*y*y+3*z*z
        u = lambda x, y, z: tf.sin(mu1*x)*tf.sin(mu2*y)*tf.sin(mu3*z)
        u_00 = lambda x, y, z: tf.sin(mu1*intervalL)*tf.sin(mu2*y)*tf.sin(mu3*z)
        u_01 = lambda x, y, z: tf.sin(mu1*intervalR)*tf.sin(mu2*y)*tf.sin(mu3*z)
        u_10 = lambda x, y, z: tf.sin(mu1*x)*tf.sin(mu2*intervalL)*tf.sin(mu3*z)
        u_11 = lambda x, y, z: tf.sin(mu1*x)*tf.sin(mu2*intervalR)*tf.sin(mu3*z)
        u_20 = lambda x, y, z: tf.sin(mu1*x)*tf.sin(mu2*y)*tf.sin(mu3*intervalL)
        u_21 = lambda x, y, z: tf.sin(mu1*x)*tf.sin(mu2*y)*tf.sin(mu3*intervalR)

    return A_eps, kappa, f, u, u_00, u_01, u_10, u_11, u_20, u_21


def get_infos2Convection_5D(input_dim=1, out_dim=1, mesh_number=2, intervalL=0.0, intervalR=1.0, equa_name=None):
    if equa_name == 'Boltzmann1':
        lam = 2
        mu = 30
        f = lambda x, y: (lam*lam+mu*mu)*(tf.sin(mu*x) + tf.sin(mu*y))
        A_eps = lambda x, y: 1.0*tf.ones_like(x)
        kappa = lambda x, y: lam*lam*tf.ones_like(x)
        u = lambda x, y: -1.0*(np.sin(mu)/np.sinh(lam))*tf.sinh(lam*x) + tf.sin(mu*x) -1.0*(np.sin(mu)/np.sinh(lam))*tf.sinh(lam*y) + tf.sin(mu*y)
        u_00 = lambda x, y, z, s, t: tf.zeros_like(x)
        u_01 = lambda x, y, z, s, t: tf.zeros_like(x)
        u_10 = lambda x, y, z, s, t: tf.zeros_like(x)
        u_11 = lambda x, y, z, s, t: tf.zeros_like(x)
        u_20 = lambda x, y, z, s, t: tf.zeros_like(x)
        u_21 = lambda x, y, z, s, t: tf.zeros_like(x)
        u_30 = lambda x, y, z, s, t: tf.zeros_like(x)
        u_31 = lambda x, y, z, s, t: tf.zeros_like(x)
        u_40 = lambda x, y, z, s, t: tf.zeros_like(x)
        u_41 = lambda x, y, z, s, t: tf.zeros_like(x)

    return A_eps, kappa, u, f, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41