from typing import Union, Any

import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt


# 这个函数有问题，需要研究一下
def reorder_index(ind_in, N):
    # This function is used to reorder the index of element for a given index-vector, then return the reorder index
    # N is a number, it represents the number of row and column for matrix
    if N % 2 == 0:
        M = N*N
    else:
        M = N*N -1

    # 相邻位置交换，如[1,2,3,4,5,6,7,8,9,10]-->[2,1,4,3,6,5,8,7,10,9]
    ind1 = np.zeros_like(ind_in)
    ind2 = np.zeros_like(ind_in)
    ind3 = np.zeros_like(ind_in)
    ind4 = np.zeros_like(ind_in)
    for i in range(1, M):
        if i % 2 == 0:
            ind1[i] = ind_in[i-1]
        else:
            ind1[i] = ind_in[i+1]

    # 前半段的偶数位置与后半段奇数位置交换，[2,1,4,3,6,   5,8,7,10,9]-->[2,10,4,8,6    5,3,7,1,9]
    ind2 = ind1
    for i in range(1, M/2):
        if i % 2 == 1:
            ind2[i] = ind1[M-i+1]
            ind2[M-i+1] = ind1[i]

    # 将数组分为四段，第一三段镜像倒换
    ind3 = ind2
    for i in range(1, M/4):
        ind3[i] = ind2[3*M/4-i+1]
        ind3[3*M/4-i+1] = ind2[i]

    # %转换为矩阵，然后转置，再变为向量
    ind_mat = np.reshape(ind3, [N, N])
    trans_ind_mat = np.transpose(ind_mat, (1, 0))
    ind4 = np.reshape(trans_ind_mat, [1, N*N])

    # 将数组分为两段，前后偶数位置互换
    ind5 = ind4
    for i in range(1, M/2):
        if i % 2 == 0:
            ind5[i] = ind4[i+M/2]
            ind5[i+M/2] = ind4[i]

    index = ind5
    # 将数组分为四段，第二四段镜像倒换
    for i in range(1, M/4):
        index[i + M/4] = ind5[M - i + 1]
        index[M - i + 1] = ind5[i + M/4]

    sort_index = np.sort(index)
    print(sort_index)
    print('xxxxxx')


# 这个文件用于生成数据，包括边界、内部、以及初始点（在这里未设置）
def fun1(z, t, ws, ds):  # 标准化变量
    stand1 = (z - ws * t) / np.sqrt(4 * ds * t)
    return stand1


def ip(z, t, ws, ds):  # 计算脉冲函数
    out = ((np.exp(-(fun1(z, t, ws, ds)) ** 2)) / (np.sqrt(np.pi * ds * t))) + erfc(fun1(z, t, ws, ds)) * ws * 0.5 / ds
    return out


def gen_label(z1, t1, ws, ds, split, p1):
    t = np.linspace(1, t1, split).reshape(1, split)
    z = np.ones_like(t) * z1
    p = np.ones_like(t).reshape(split, 1) * p1
    temp1 = (z - ws * t) / np.sqrt(4 * ds * t)
    exp1 = np.exp(-(np.square(temp1)))
    temp2 = exp1 / np.sqrt(np.pi * ds * t)
    temp3 = erfc(temp1)
    i = temp2 + (0.5 * ws / ds) * temp3
    u = np.matmul(i, p)
    return u.item()


def dt(z):
    return 1 / z


def rand_it(batch_size, variable_dim, region_a, region_b):
    # 随机生成树
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    x_it = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    x_it = x_it.astype(np.float32)
    return x_it


class DataGen2:
    def __init__(self, zs, ze, ts, te, zsteps, tsteps, ws=0.001, ds=0.0002, p=0.0001):
        super(DataGen2, self).__init__()
        self.zs = zs  # 默认值
        self.ze = ze
        self.ts = ts
        self.te = te
        self.zsteps = zsteps
        self.tsteps = tsteps
        self.ws = ws
        self.ds = ds
        self.p = p

    def gen_inter(self, batch_size):
        # generate interior points using loops
        x_input = rand_it(batch_size, 1, self.zs, self.ze)  # 取的内部点的为随机生成
        t_input = rand_it(batch_size, 1, self.ts, self.te)
        label = np.zeros((batch_size, 1))
        for i in range(batch_size):
            split = 100
            p = np.ones(split) * self.p
            t = np.linspace(0.1, t_input[i], split)
            y = np.zeros(split)
            pp = 0
            for j in range(split):
                temp = ip(x_input[i], t[j], self.ws, self.ds)
                y[j] = temp
            for nn in range(split):
                pp = pp + y[nn] * p[nn]
            label[i] = pp
        x_collocation = x_input
        t_collocation = t_input
        label_rand = label
        return x_collocation, t_collocation, label_rand

    def gen_inter_m(self, batch_size):
        # generate interior points using matrix method
        x_input = rand_it(batch_size, 1, self.zs, self.ze)  # 取的内部点的为随机生成
        t_input = rand_it(batch_size, 1, self.ts, self.te)
        label = np.zeros((batch_size, 1))
        split = 100
        for i in range(batch_size):
            temp = gen_label(x_input[i], t_input[i], self.ws, self.ds, split, self.p)
            label[i] = temp
        x_collocation = x_input
        t_collocation = t_input
        label_rand = label
        return x_collocation, t_collocation, label_rand

    def gen_bound(self, batch_size):
        x_input = rand_it(batch_size, 1, 0, 0)
        t_input = rand_it(batch_size, 1, self.ts, self.te)
        label = np.ones((batch_size, 1)) * self.p
        x_collocation = x_input
        t_collocation = t_input
        label_rand = label
        return x_collocation, t_collocation, label_rand

    def gen_init(self, batch_size):
        x_input = rand_it(batch_size, 1, self.zs, self.ze)
        t_input = rand_it(batch_size, 1, 0, 0)
        label = np.zeros((batch_size, 1))
        x_collocation = x_input
        t_collocation = t_input
        label_rand = label
        return x_collocation, t_collocation, label_rand

    def gen_labels_all(self):
        # 生成全部网格点的标签
        c = np.zeros((self.zsteps, self.tsteps))
        z = np.linspace(self.zs, self.ze, self.zsteps)
        p = np.ones(self.tsteps) * self.p
        time = np.linspace(self.ts, self.te, self.tsteps)
        for i in range(self.zsteps):  # 生成内部点
            y = np.zeros(self.tsteps)  # 这里生成一个脉冲函数的缓存数组，对于每个高度重新设置一个0向量组
            for t in range(self.tsteps):
                temp = ip(z[i], time[t], self.ws, self.ds)
                y[t] = temp
            for t in range(self.tsteps):
                pp = 0
                for nn in range(0, t + 1):
                    pp = pp + y[nn] * p[nn]
                c[i][t] = pp  # sh生成内部
        c_all = c.reshape(-1, 1)
        label1 = c_all
        return label1

    def gen_input(self):
        x_collocation = np.linspace(self.zs, self.ze, self.zsteps).reshape(self.zsteps, 1)
        t_collocation = np.linspace(self.ts, self.te, self.tsteps).reshape(self.tsteps, 1)
        x_repeat = np.repeat(x_collocation, self.tsteps).reshape(self.zsteps * self.tsteps, 1)
        t2 = list(t_collocation)
        t1 = list(t_collocation)
        for i in range(self.zsteps - 1):
            t2.extend(t1)
        t_repeat = np.array(t2)
        pt_x_collocation = Variable(torch.from_numpy(x_repeat).float(), requires_grad=True)
        pt_t_collocation = Variable(torch.from_numpy(t_repeat).float(), requires_grad=True)
        inputs = torch.cat([pt_t_collocation, pt_x_collocation], 1)
        return inputs

    # 内部输入点生成
    def gen_boundary(self):
        # 边界输入点生成 z(高度)取0，时间分为steps步
        x_boundary_co = np.zeros((self.tsteps, 1))
        t_collocation = np.linspace(self.ts, self.te, self.tsteps).reshape(self.tsteps, 1)
        pt_x_boundary = Variable(torch.from_numpy(x_boundary_co).float())
        pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True)
        boundary_input = torch.cat([pt_t_collocation, pt_x_boundary], 1)
        return boundary_input

    def gen_boundary_label(self):
        p = np.ones(self.tsteps) * self.p

    # 当Dt是与z相关的函数
    def gen_inter_dz(self, batch_size):
        x_input = rand_it(batch_size, 1, self.zs, self.ze)
        t_input = rand_it(batch_size, 1, self.ts, self.te)
        label = np.zeros((batch_size, 1))
        for i in range(batch_size):
            split = 100
            p = np.ones(split) * self.p
            t = np.linspace(0.1, t_input[i], split)
            y = np.zeros(split)
            temp = 0
            pp = 0
            for j in range(split):
                temp = ip(x_input[i], t[j], self.ws, self.ds_function(x_input[i]))
                y[j] = temp
            for nn in range(split):
                pp = pp + y[nn] * p[nn]
            label[i] = pp
        pt_x_collocation = Variable(torch.from_numpy(x_input).float(), requires_grad=True)
        pt_t_collocation = Variable(torch.from_numpy(t_input).float(), requires_grad=True)
        inputs_rand = torch.cat([pt_t_collocation, pt_x_collocation], 1)
        label_rand = Variable(torch.from_numpy(label).float(), requires_grad=True)
        return inputs_rand, label_rand

    def gen_labels_all_dz(self):
        # 生成全部网格点的标签
        c = np.zeros((self.zsteps, self.tsteps))
        z = np.linspace(self.zs, self.ze, self.zsteps)
        p = np.ones(self.tsteps) * self.p
        time = np.linspace(self.ts, self.te, self.tsteps)
        for i in range(self.zsteps):  # 生成内部点
            y = np.zeros(self.tsteps)  # 这里生成一个脉冲函数的缓存数组，对于每个高度重新设置一个0向量组
            for t in range(self.tsteps):
                temp = ip(z[i], time[t], self.ws, self.ds_function(z[i]))
                y[t] = temp
            for t in range(self.tsteps):
                pp = 0
                for nn in range(0, t + 1):
                    pp = pp + y[nn] * p[nn]
                c[i][t] = pp  # sh生成内部
        c_all = c.reshape(-1, 1)
        label1 = Variable(torch.from_numpy(c_all.copy()).float(), requires_grad=False)
        return label1

    def ds_function(self, num, mode='1'):
        if mode == '1':
            return (num - self.zs) / (self.ze - self.zs) * self.ds
        elif mode == '2':
            return 1 / num
        elif mode == '3':
            return (num - self.zs) * 0.001

    def gen_mesh(self, batch_size2mesh=50):
        x_coord = np.linspace(self.zs, self.ze, num=batch_size2mesh)
        t_coord = np.linspace(self.ts, self.te, num=batch_size2mesh)
        mesh_x, mesh_t = np.meshgrid(x_coord, t_coord)
        x_points = np.reshape(mesh_x, [-1, 1])
        t_points = np.reshape(mesh_t, [-1, 1])
        return x_points, t_points

    def gen_mesh2scatter(self, batch_size2mesh=50):
        x_coord = np.linspace(self.zs, self.ze, num=batch_size2mesh)
        t_coord = np.linspace(self.ts, self.te, num=batch_size2mesh)
        mesh_x, mesh_t = np.meshgrid(x_coord, t_coord)
        x_points = np.reshape(mesh_x, [-1, 1])
        t_points = np.reshape(mesh_t, [-1, 1])
        np.random.shuffle(x_points)
        np.random.shuffle(t_points)
        return x_points, t_points


def test1():
    # 问题设置
    zs = 0.1
    ze = 2
    ts = 1
    te = 3000
    zsteps = 10
    tsteps = 300
    ws = 0.001
    ds = 0.0002
    p = 0.0001

    # 生成数据
    data = DataGen2(zs, ze, ts, te, zsteps=zsteps, tsteps=tsteps, ws=ws, ds=ds, p=p)
    x_points, t_points = data.gen_mesh(batch_size2mesh=60)
    xt = np.concatenate([x_points, t_points], axis=-1)
    x_mesh = np.reshape(x_points, newshape=[60, 60])
    t_mesh = np.reshape(t_points, newshape=[60, 60])

    plt.plot(x_mesh, t_mesh,
             color='red',  # 全部点设置为红色
             marker='.',  # 点的形状为圆点
             linestyle='')  # 线型为空，也即点与点之间不用线连接
    plt.grid(True)
    plt.show()
    print('XXXXXXXXXXXXXXXXXXX')


def test2():
    # 问题设置
    zs = 0.1
    ze = 2
    ts = 1
    te = 500
    zsteps = 10
    tsteps = 300
    ws = 0.001
    ds = 0.0002
    p = 0.0001

    # 生成数据
    data = DataGen2(zs, ze, ts, te, zsteps=zsteps, tsteps=tsteps, ws=ws, ds=ds, p=p)
    x_points, t_points = data.gen_mesh2scatter(batch_size2mesh=100)

    plt.plot(x_points, t_points,
             color='red',  # 全部点设置为红色
             marker='.',  # 点的形状为圆点
             linestyle='')  # 线型为空，也即点与点之间不用线连接
    plt.grid(True)
    plt.show()
    print('XXXXXXXXXXXXXXXXXXX')


if __name__ == '__main__':
    test2()
