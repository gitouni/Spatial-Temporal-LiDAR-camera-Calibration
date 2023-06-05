from GP_reg import get_sklearn_gpr
import argparse
from functools import partial
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial.transform import Rotation

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x0", type=float, nargs=3, default=[0,0,2])
    parser.add_argument("--n0", type=float, nargs=3, default=[1.5,-0.6,2.7])
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--test_x", type=float, nargs=2, default=[0.3,-0.3])
    parser.add_argument("--sample_radius",type=float, default=1)
    parser.add_argument("--n_samples",type=int, default=50)
    parser.add_argument("--view_npoints",type=int, default=10)
    parser.add_argument("--seed",default=0)
    parser.add_argument("--rot", type=float, nargs=3, default=[0.2,-0.2,0.15])
    parser.add_argument("--trans",type=float,nargs=3,default=[0.5,0.3,0.55])
    return parser.parse_args()

def mf(X:np.ndarray):
    """Ax+By+Cz = D0

    Args:
        X (np.ndarray): x & y

    Returns:
        np.ndarray: z
    """
    return 1-np.sin(0.2 * np.linalg.norm(X - 0.5, axis=-1))

def cartesian_to_sphereical(x:np.ndarray):
    y = np.zeros_like(x)
    y[:,0] = np.sqrt(np.sum(x**2, axis=1))
    valid_rev = y[:,0]>0  # if r = 0, assign theta = phi = 0
    y[valid_rev,1] = np.arccos(x[valid_rev,2]/y[valid_rev,0])  # arcose(z/r)
    y[valid_rev,2] = np.arctan2(x[valid_rev,1], x[valid_rev,0])  # arctan(y,x)
    return y

def spherical_to_cartesian(x:np.ndarray):
    y = np.zeros_like(x)
    y[:,0] = x[:,0] * np.sin(x[:,1]) * np.cos(x[:,2])
    y[:,1] = x[:,0] * np.sin(x[:,1]) * np.sin(x[:,2])
    y[:,2] = x[:,0] * np.cos(x[:,1])
    return y

def transfer_points(points:np.ndarray, rotmat:np.ndarray, trans:np.ndarray):
    if np.ndim(trans) == 1:
        return np.transpose(rotmat @ points.T + trans[:,None])
    else:
        return np.transpose(rotmat @ points.T + trans)

def print_params(title:str, params:dict):
    print(title)
    for key, value in params.items():
        print("{}: {}".format(key, value))

if __name__ == "__main__":
    args = options()
    np.random.seed(args.seed)
    cart_gpr = get_sklearn_gpr(length_scale_bounds=[0.1,100])
    sph_gpr = get_sklearn_gpr(length_scale_bounds=[0.1,100])
    trans = np.array(args.trans)
    rotvec = np.array(args.rot)
    rotmat = Rotation.from_rotvec(rotvec).as_matrix()
    tf_points = partial(transfer_points, rotmat=rotmat, trans=trans)
    train_x = np.random.uniform(-args.sample_radius, args.sample_radius, (args.n_samples, 2))
    if args.noise > 0:
        noise = np.random.uniform(-args.noise , args.noise , (args.n_samples,))
    else:
        noise = 0
    train_y = mf(train_x)  # (N,)
    train_xy = np.hstack((train_x, train_y[:, None]))  # (N,2) , (N,1) -> (N,3)
    sph_tr_xy = cartesian_to_sphereical(train_xy)  # (x,y,z) -> (r,th,phi)
    sph_tr_x, sph_tr_y = sph_tr_xy[:,1:] , sph_tr_xy[:,0]  # give phi and theta, predict r
    test_x = np.array(args.test_x)
    test_y = mf(test_x)
    grid_d = np.linspace(-args.sample_radius, args.sample_radius, args.view_npoints)
    grid_x1, grid_x2 = np.meshgrid(grid_d, grid_d)  # [n,n]
    grid_X = np.stack([grid_x1, grid_x2], axis=-1)
    grid_z = mf(np.stack([grid_x1, grid_x2], axis=-1))
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.plot_surface(grid_x1, grid_x2, grid_z, cmap=cm.get_cmap("coolwarm"), linewidth=0, alpha=0.2, antialiased=False)
    grid_xyz = np.stack((grid_x1.reshape(-1), grid_x2.reshape(-1), grid_z.reshape(-1)),axis=-1)
    roted_grid_xyz = transfer_points(grid_xyz, rotmat, np.zeros(3)).reshape(args.view_npoints, args.view_npoints, -1)
    ax.plot_surface(roted_grid_xyz[...,0], roted_grid_xyz[...,1], roted_grid_xyz[...,2], cmap=cm.get_cmap("Greys"), linewidth=0, alpha=0.4, antialiased=False)
    traned_grid_xyz = tf_points(grid_xyz).reshape(args.view_npoints, args.view_npoints, -1)
    ax.plot_surface(traned_grid_xyz[...,0], traned_grid_xyz[...,1], traned_grid_xyz[...,2], cmap=cm.get_cmap("Blues"), linewidth=0, alpha=0.4, antialiased=False)
    
    cart_gpr.fit(train_x, train_y)
    sph_gpr.fit(sph_tr_x, sph_tr_y)
    print_params("cart_gpr:",cart_gpr.kernel_.get_params())
    print_params("sph_gpr:",sph_gpr.kernel_.get_params())
    raw_z = cart_gpr.predict(test_x[None,:]).item()
    raw_xyz = np.array([test_x[0], test_x[1], raw_z])  # (3,)
    ax.scatter(raw_xyz[0], raw_xyz[1], raw_xyz[2], c='g')
    roted_xyz = transfer_points(raw_xyz[None,:], rotmat.T, np.zeros(3))  # (1,3) unreal x y z
    roted_sph = cartesian_to_sphereical(roted_xyz)  # (1,3) unreal range, real phi and theta
    # dth = roted_sph[0,1] - raw_sph[0,1]
    # dphi = roted_sph[0,2] - raw_sph[0,2]
    roted_r = sph_gpr.predict(roted_sph[:,1:]).item()
    print("roted_r:{}".format(roted_r))
    roted_sph[0,0] = roted_r  # real range
    roted_xyz = spherical_to_cartesian(roted_sph) # (1,3) , corresponding roted xyz in raw manifold
    ax.scatter(roted_xyz[0,0], roted_xyz[0,1], roted_xyz[0,2], c='red')
    corr_xyz = np.array([test_x[0], test_x[1], roted_r])
    ax.scatter(corr_xyz[0], corr_xyz[1], corr_xyz[2], c='m')
    # pre_trans = rotmat.T @ trans
    # traned_z = cart_gpr.predict(roted_xyz[:,:2] - pre_trans[:2]).item() + pre_trans[2]
    # final_xyz = np.array([test_x[0], test_x[1], traned_z])
    # print(final_xyz)
    # ax.scatter(final_xyz[0], final_xyz[1], final_xyz[2], c='k')
    plt.show()