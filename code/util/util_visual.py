import os
import h5py
import subprocess
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

#########
# Plot
#########

def plot_3d_point_cloud(x,
                        y,
                        z,
                        show=True,
                        show_axis=True,
                        in_u_sphere=False,
                        marker='.',
                        s=8,
                        alpha=.8,
                        figsize=(5, 5),
                        elev=10,
                        azim=240,
                        axis=None,
                        title=None,
                        lim=None,
                        *args,
                        **kwargs):

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if lim:
        ax.set_xlim3d(*lim[0])
        ax.set_ylim3d(*lim[1])
        ax.set_zlim3d(*lim[2])
    elif in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        lim = (min(np.min(x), np.min(y), np.min(z)),
                max(np.max(x), np.max(y), np.max(z)))
        ax.set_xlim(1.3 * lim[0], 1.3 * lim[1])
        ax.set_ylim(1.3 * lim[0], 1.3 * lim[1])
        ax.set_zlim(1.3 * lim[0], 1.3 * lim[1])
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if show:
        plt.show()

    return fig

def plot_3d_point_cloud_dict(name_dict, lim, size=2):
    num_plots = len(name_dict)
    fig = plt.figure(figsize=(size*num_plots, size))
    ax = {}
    for i, (k, v) in enumerate(name_dict.items()):
        ax[k] = fig.add_subplot(1, num_plots, i + 1, projection='3d')
        plot_3d_point_cloud(v[2], -v[0], v[1], axis=ax[k], show=False, lim=lim)
        ax[k].set_title(k)
    plt.tight_layout()
    return fig

#########
# render
#########

def get_rendered_image_of_param(param, ver_mat, ver_vec, triangles, orig_ids, face_labels, renderer_path, tmp_dir):
    if ver_vec is None:
        vertices = np.dot(ver_mat, param)
    else:
        vertices = np.dot(ver_mat, param) + ver_vec
    os.mkdir(tmp_dir)
    vertices = vertices.reshape((-1, 3)).astype(np.float32)
    mesh_path = os.path.join(tmp_dir, 'mesh.obj')
    face_path = os.path.join(tmp_dir, 'face_ids.txt')
    img_path = os.path.join(tmp_dir, 'mesh.png')
    with open(mesh_path, 'w') as obj_f:
        vert_str = ''
        for item in vertices:
            vert_str += 'v %f %f %f \n' % (item[0],item[1],item[2])
        obj_f.write(vert_str)
        tri_str = ''
        for item in triangles:
            tri_str += 'f {0} {1} {2}\n'.format(item[0] + 1,item[1] + 1,item[2] + 1)
        obj_f.write(tri_str)
        
    with open(face_path, 'w') as obj_f:
        face_str = ''
        for item in face_labels:
            face_str += '%d\n' % orig_ids[item]
        obj_f.write(face_str)
    p = subprocess.Popen(('bash', renderer_path), cwd=tmp_dir)
    p.wait()
    with Image.open(img_path) as img:
        img = Image.fromarray(np.array(img))
    # clear
    p = subprocess.Popen(('rm', '-r', tmp_dir))
    p.wait()
    return img

def render(param, param_mask, path, renderer_path, tmp_dir):
    
    with h5py.File(path, 'r') as f:
        ver_mat = np.array(f['vertices_mat'])
        ver_vec = np.zeros(ver_mat.shape[0])
        triangles = np.array(f['faces'])
        orig_ids = np.array(f['orig_ids'])
        face_labels = np.array(f['face_labels'])
        default_param = np.array(f['default_param'])
        
    if param is None:
        param_tmp = default_param
    else:
        num_param = int(param_mask.sum().item())
        param_tmp = param[:num_param].cpu()
    return get_rendered_image_of_param(param_tmp, ver_mat, None, triangles,
                                       orig_ids, face_labels,
                                       renderer_path, tmp_dir)