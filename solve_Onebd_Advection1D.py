import os
import sys
import platform
import shutil
import time
import matplotlib
import tensorflow as tf
import data_gen2
from data_gen2 import DataGen2
import numpy as np
import saveData
import plotData
import DNN_Class_base
import DNN_Log_Print
import DNN_tools
import Advec_diffu_Eqs
import DNN_data


class PDE_DNN(object):
    def __init__(self, input_dim=2, out_dim=1, hidden_layer=None, Model_name='DNN', name2actIn='tanh',
                 name2actHidden='tanh', name2actOut='linear', opt2regular_WB='L2', type2numeric='float32',
                 factor2freq=None, sFourier=1.0, ws=0.001, ds=0.0002, HighFreq=False):
        super(PDE_DNN, self).__init__()
        if 'DNN' == str.upper(Model_name):
            self.DNN = DNN_Class_base.Pure_Dense_Net(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name, actName2in=name2actIn,
                actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric)
        elif 'SCALE_DNN' == str.upper(Model_name) or 'DNN_SCALE' == str.upper(Model_name):
            self.DNN = DNN_Class_base.Dense_ScaleNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name, actName2in=name2actIn,
                actName=name2actHidden, actName2out=name2actOut, repeat_high_freq=HighFreq, type2float=type2numeric)
        elif 'FOURIER_DNN' == str.upper(Model_name) or 'DNN_FOURIERBASE' == str.upper(Model_name):
            self.DNN = DNN_Class_base.Dense_FourierNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name, actName2in=name2actIn,
                actName=name2actHidden, actName2out=name2actOut, repeat_high_freq=HighFreq, type2float=type2numeric)

        if type2numeric == 'float32':
            self.float_type = tf.float32
        elif type2numeric == 'float64':
            self.float_type = tf.float64
        elif type2numeric == 'float16':
            self.float_type = tf.float16

        self.factor2freq = factor2freq
        self.opt2regular_WB = opt2regular_WB
        self.sFourier = sFourier

        self.ws = ws
        self.ds = ds

        self.mat2X = tf.constant([[1, 0]], dtype=self.float_type)  # 1 行 2 列
        self.mat2T = tf.constant([[0, 1]], dtype=self.float_type)  # 1 行 2 列

    def loss2PDE(self, X=None, t=None, loss_type='l2_loss'):
        assert (X is not None)
        assert (t is not None)

        shape2X = X.get_shape().as_list()
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        XT = tf.matmul(X, self.mat2X) + tf.matmul(t, self.mat2T)

        UNN = self.DNN(XT, scale=self.factor2freq, sFourier=self.sFourier)
        dUNN2X = tf.gradients(UNN, X)[0]
        dUNN2t = tf.gradients(UNN, t)[0]
        dUNNxx = tf.gradients(dUNN2X, X)[0]

        loss_pde = dUNN2t + self.ws * dUNN2X - self.ds * dUNNxx
        loss_it = tf.reduce_mean(tf.square(loss_pde))
        return UNN, loss_it, dUNN2X, dUNN2t, dUNNxx

    def loss2bd(self, X_bd=None, t=None, Ubd_exact=None, if_lambda2Ubd=True):
        assert (X_bd is not None)
        assert (Ubd_exact is not None)

        shape2X = np.shape(X_bd)
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        if if_lambda2Ubd:
            Ubd = Ubd_exact(X_bd, t)
        else:
            Ubd = Ubd_exact

        XT = tf.matmul(X_bd, self.mat2X) + tf.matmul(t, self.mat2T)
        UNN_bd = self.DNN(XT, scale=self.factor2freq, sFourier=self.sFourier)
        loss_bd_square = tf.square(UNN_bd - Ubd)
        loss_bd = tf.reduce_mean(loss_bd_square)
        return loss_bd

    def loss2Init(self, X=None, tinit=None, Uinit_exact=None, if_lambda2Uinit=True):
        assert (X is not None)
        assert (tinit is not None)
        assert (Uinit_exact is not None)

        shape2X = X.get_shape().as_list()
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        if if_lambda2Uinit:
            Uinit = Uinit_exact(X, tinit)
        else:
            Uinit = Uinit_exact

        XT = tf.matmul(X, self.mat2X) + tf.matmul(tinit, self.mat2T)
        UNN_init = self.DNN(XT, scale=self.factor2freq, sFourier=self.sFourier)
        loss_init_square = tf.square(UNN_init - Uinit)
        loss_init = tf.reduce_mean(loss_init_square)
        return loss_init

    def get_regularSum2WB(self):
        sum2WB = self.DNN.get_regular_sum2WB(self.opt2regular_WB)
        return sum2WB

    def evaluate_PDE_DNN(self, X_points=None, t_points=None):
        assert (X_points is not None)
        assert (t_points is not None)
        shape2X = X_points.get_shape().as_list()
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        XT = tf.matmul(X_points, self.mat2X) + tf.matmul(t_points, self.mat2T)
        UNN = self.DNN(XT, scale=self.factor2freq, sFourier=self.sFourier)
        return UNN


def solve_Advection_diffusion(R):
    log_out_path = R['FolderName']            # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):      # 判断路径是否已经存在
        os.mkdir(log_out_path)                # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s_%s.txt' % ('log', '_train')
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    DNN_Log_Print.dictionary2space_time(R, log_fileout)

    if R['equa_name'] == 'Advection1':
        ws = 1.0
        ds = 0.01
        region_l = 0.0
        region_r = 1.0
        init_time = 0.0
        end_time = 2.0
        U_true, U_left, U_right, U_init = Advec_diffu_Eqs.get_infos2Advection_1D(
            x_left=region_l, x_right=region_r, t_init=init_time, ws=ws, ds=ds, eqs_name=R['equa_name'])
    elif R['equa_name'] == 'Advection2':
        ws = 0.8
        ds = 0.005
        region_l = 1.0
        region_r = 5
        init_time = 0.0
        end_time = 10
        U_true, U_left, U_right, U_init = Advec_diffu_Eqs.get_infos2Advection_1D(
            x_left=region_l, x_right=region_r, t_init=init_time, ws=ws, ds=ds, eqs_name=R['equa_name'])

    # 问题需要的设置
    batchsize_in = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']
    batchsize_init = R['batch_size2init']
    batchsize2mesh_test = R['batchsize2mesh_test']
    batchsize_test = batchsize2mesh_test * batchsize2mesh_test

    bd_penalty_init = R['init_boundary_penalty']  # Regularization parameter for boundary conditions
    init_penalty_init = R['init_init_penalty']    # Regularization parameter for init conditions
    penalty2WB = R['penalty2weight_biases']       # Regularization parameter for weights and biases
    lr_decay = R['learning_rate_decay']
    learning_rate = R['learning_rate']
    act_func = R['name2act_hidden']

    input_dim = R['input_dim']

    model = PDE_DNN(input_dim=R['input_dim']+1, out_dim=R['output_dim'], hidden_layer=R['hidden_layers'],
                    Model_name=R['model2NN'], name2actIn=R['name2act_in'], name2actHidden=R['name2act_hidden'],
                    name2actOut=R['name2act_out'], opt2regular_WB=R['regular_wb_model'], type2numeric='float32',
                    factor2freq=R['freq'], sFourier=R['sfourier'], ws=ws, ds=ds, HighFreq=False)

    global_steps = tf.compat.v1.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.compat.v1.variable_scope('vscope', reuse=tf.compat.v1.AUTO_REUSE):
            X_in = tf.compat.v1.placeholder(tf.float32, name='X_in', shape=[batchsize_in, input_dim])  # * 行 1 列
            t_in = tf.compat.v1.placeholder(tf.float32, name='t_in', shape=[batchsize_in, 1])          # * 行 1 列
            U_in2train = tf.compat.v1.placeholder(tf.float32, name='U_in2train', shape=[batchsize_in, 1])  # * 行 1 列

            Xbd_left = tf.compat.v1.placeholder(tf.float32, name='Xbd_left', shape=[batchsize_bd, input_dim])    # *行1列
            # Xbd_right = tf.compat.v1.placeholder(tf.float32, name='Xbd_right', shape=[batchsize_bd, input_dim])  # *行1列
            t_bd = tf.compat.v1.placeholder(tf.float32, name='t_bd', shape=[batchsize_bd, 1])

            Ubd_left = tf.compat.v1.placeholder(tf.float32, name='Ubd_left', shape=[batchsize_bd, 1])
            Ubd_right = tf.compat.v1.placeholder(tf.float32, name='Ubd_right', shape=[batchsize_bd, 1])

            Xinit = tf.compat.v1.placeholder(tf.float32, name='Xinit', shape=[batchsize_init, input_dim])        # *行1列
            tinit = tf.compat.v1.placeholder(tf.float32, name='tinit', shape=[batchsize_init, 1])
            Uinit = tf.compat.v1.placeholder(tf.float32, name='Uinit', shape=[batchsize_init, 1])

            boundary_penalty = tf.compat.v1.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            init_penalty = tf.compat.v1.placeholder_with_default(input=1e2, shape=[], name='init_p')
            in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')

            X_in2test = tf.compat.v1.placeholder(tf.float32, name='X_in2test', shape=[batchsize_test, input_dim])
            t_in2test = tf.compat.v1.placeholder(tf.float32, name='t_in2test', shape=[batchsize_test, 1])

            UNN2train, loss_it, dUNN2dX, dUNN2dt, dUNN2dxx = model.loss2PDE(X=X_in, t=t_in, loss_type=R['loss_type'])

            loss_bd = model.loss2bd(X_bd=Xbd_left, t=t_bd, Ubd_exact=Ubd_left, if_lambda2Ubd=False)
            # loss_right = model.loss2bd(X_bd=Xbd_right, t=t_bd, Ubd_exact=Ubd_right, if_lambda2Ubd=False)
            # loss_bd = loss_left + loss_right

            loss_init = model.loss2Init(X=Xinit, tinit=tinit, Uinit_exact=Uinit, if_lambda2Uinit=False)

            regularSum2WB = model.get_regularSum2WB()
            PWB = penalty2WB * regularSum2WB

            loss = loss_it + boundary_penalty * loss_bd + init_penalty * loss_init + PWB  # 要优化的loss function

            my_optimizer = tf.compat.v1.train.AdamOptimizer(in_learning_rate)
            train_my_loss = my_optimizer.minimize(loss, global_step=global_steps)

            # 训练上的真解值和训练结果的误差
            mean_square_error = tf.reduce_mean(tf.square(U_in2train - UNN2train))
            residual_error = mean_square_error / tf.reduce_mean(tf.square(U_in2train))

            UNN2test = model.evaluate_PDE_DNN(X_points=X_in2test, t_points=t_in2test)

    t0 = time.time()
    loss_in_all, loss_bd_all, loss_init_all, loss_all, train_mse_all, train_rel_all = [], [], [], [], [], []
    test_mse_all, test_rel_all = [], []                                                    # 空列表, 使用 append() 添加元素
    test_epoch = []

    x_test = DNN_data.rand_it(batchsize_test, input_dim, region_a=region_l, region_b=region_r)
    t_test = DNN_data.rand_it(batchsize_test, input_dim, region_a=init_time, region_b=end_time)
    # t_test = np.ones(shape=[batchsize_test, 1], dtype=np.float32) * 0.5
    u_true2test = U_true(x_test, t_test)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True                        # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                            # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        tmp_lr = learning_rate
        for i_epoch in range(R['max_epoch'] + 1):
            x_in_batch = DNN_data.rand_it(batchsize_in, input_dim, region_a=region_l, region_b=region_r)
            t_in_batch = DNN_data.rand_it(batchsize_in, 1, region_a=init_time, region_b=end_time)
            Uin_numpy = U_true(x_in_batch, t_in_batch)

            xl_bd_batch, xr_bd_batch = DNN_data.rand_bd_1D(batchsize_bd, input_dim,
                                                           region_a=region_l, region_b=region_r)
            t_bd_batch = DNN_data.rand_it(batchsize_bd, 1, region_a=init_time, region_b=end_time)

            Ubd2left_numpy = U_left(xl_bd_batch, t_bd_batch)
            # Ubd2right_numpy = U_right(xr_bd_batch, t_bd_batch)

            x_init_batch = DNN_data.rand_it(batchsize_init, input_dim, region_a=region_l, region_b=region_r)
            t_init_batch = np.ones(shape=[batchsize_init, 1], dtype=np.float32) * init_time

            Uinit_numpy = U_init(x_init_batch, t_init_batch)

            tmp_lr = tmp_lr * (1 - lr_decay)
            if R['activate_penalty2bd_increase'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_bd = bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_bd = 10 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_bd = 20 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_bd = 50 * bd_penalty_init
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_bd = 100 * bd_penalty_init
                else:
                    temp_penalty_bd = 200 * bd_penalty_init
            else:
                temp_penalty_bd = bd_penalty_init

            if R['activate_penalty2init_increase'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_init = bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_init = 10 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_init = 50 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_init = 100 * bd_penalty_init
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_init = 200 * bd_penalty_init
                else:
                    temp_penalty_init = 500 * bd_penalty_init
            else:
                temp_penalty_init = init_penalty_init

            _, loss_it_tmp, loss_bd_tmp, loss_init_tmp, loss_tmp, train_mse_tmp, train_res_tmp, pwb = sess.run(
                [train_my_loss, loss_it, loss_bd, loss_init, loss, mean_square_error, residual_error, PWB],
                feed_dict={X_in: x_in_batch, t_in: t_in_batch, U_in2train: Uin_numpy, Xbd_left: xl_bd_batch,
                           t_bd: t_bd_batch, Ubd_left: Ubd2left_numpy, Xinit: x_init_batch, tinit: t_init_batch,
                           Uinit: Uinit_numpy, in_learning_rate: tmp_lr, boundary_penalty: temp_penalty_bd,
                           init_penalty: temp_penalty_init})

            # du2dx, du2dt, du2dxx = sess.run([dUNN2dX, dUNN2dt, dUNN2dxx],
            #                                 feed_dict={X_in: x_in_batch, t_in: t_in_batch, U_in2train: in_labels,
            #                                 Xbd_left: x_bd_batch, t_bd: t_bd_batch, Ubd_left: bd_labels,
            #                                 Xinit: x_init_batch, tinit: t_init_batch, Uinit: init_labels,
            #                                 in_learning_rate: tmp_lr, boundary_penalty: temp_penalty_bd,
            #                                 init_penalty: temp_penalty_init})

            loss_in_all.append(loss_it_tmp)
            loss_bd_all.append(loss_bd_tmp)
            loss_init_all.append(loss_init_tmp)
            loss_all.append(loss_tmp)
            train_mse_all.append(train_mse_tmp)
            train_rel_all.append(train_res_tmp)
            if i_epoch % 100 == 0:
                run_times = time.time() - t0
                DNN_Log_Print.print_and_log_train_one_epoch2space_time(
                    i_epoch, run_times, tmp_lr, temp_penalty_bd, temp_penalty_init, pwb, loss_it_tmp, loss_bd_tmp, loss_init_tmp, loss_tmp,
                    train_mse_tmp, train_res_tmp, log_out=log_fileout)

                # # ---------------------------   test network ----------------------------------------------
                test_epoch.append(i_epoch / 100)
                unn2test = sess.run(UNN2test, feed_dict={X_in2test: x_test, t_in2test: t_test})
                mse2test = np.mean(np.square(u_true2test - unn2test))
                test_mse_all.append(mse2test)
                res2test = mse2test / np.mean(np.square(u_true2test))
                test_rel_all.append(res2test)

                DNN_Log_Print.print_and_log_test_one_epoch(mse2test, res2test, log_out=log_fileout)
                # DNN_tools.log_string(str(du2dx), log_fileout)
                # DNN_tools.log_string('\n', log_fileout)
                # DNN_tools.log_string(str(du2dt), log_fileout)
                # DNN_tools.log_string('\n', log_fileout)
                # DNN_tools.log_string(str(du2dxx), log_fileout)
                # DNN_tools.log_string('\n', log_fileout)

    # ------------------- save the training results into mat file and plot them -------------------------
    saveData.save_trainLosses2mat_SpaceTime(loss_in_all, loss_bd_all, loss_init_all, loss_all, name2in='loss2PDE',
                                            name2bd='loss2bd', name2init='loss2init', name2all='total_loss',
                                            actName=act_func, outPath=R['FolderName'])
    plotData.plotTrain_loss_1act_func(loss_in_all, lossType='loss_pde', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_init_all, lossType='loss_init', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)

    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=act_func, outPath=R['FolderName'])
    plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=act_func, seedNo=R['seed'],
                                         outPath=R['FolderName'], yaxis_scale=True)

    # ------------------- save the testing results into mat file and plot them -------------------------
    saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=act_func, outPath=R['FolderName'])
    plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=act_func,
                              seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)

    saveData.save_2testSolus2mat(u_true2test, unn2test, actName='Utrue', actName1=act_func, outPath=R['FolderName'])
    plotData.plot_scatter_solu2test(unn2test, x_test, t_test, actName=act_func, seedNo=R['seed'],
                                    outPath=R['FolderName'])


if __name__ == "__main__":
    R = {}
    R['gpuNo'] = 0
    if platform.system() == 'Windows':
        os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
    else:
        print('-------------------------------------- linux -----------------------------------------------')
        # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
        matplotlib.use('Agg')

    # 文件保存路径设置
    store_file = 'AdvecEq'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])  # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  复制并保存当前文件 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if platform.system() == 'Windows':
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # ---------------------------- Setup of laplace equation ------------------------------
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    # step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    # R['activate_stop'] = int(step_stop_flag)

    R['activate_stop'] = int(0)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    # R['max_epoch'] = 10000
    # R['max_epoch'] = 20000
    R['max_epoch'] = 50000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    R['PDE_type'] = 'Advection diffusion'
    R['equa_name'] = 'Advection1'
    # R['equa_name'] = 'Advection2'
    R['input_dim'] = 1                   # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                  # 输出维数

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of DNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 训练集的设置(内部和边界)
    # R['batch_size2interior'] = 3000  # 内部训练数据的批大小
    R['batch_size2interior'] = 5000  # 内部训练数据的批大小
    R['batch_size2boundary'] = 1000  # 边界训练数据大小
    R['batch_size2init'] = 1000

    R['batchsize2mesh_test'] = 80

    R['loss_type'] = 'L2_loss'

    R['optimizer_name'] = 'Adam'     # 优化器
    # R['learning_rate'] = 1e-2        # 学习率
    # R['learning_rate_decay'] = 2.5e-4  # 学习率 decay

    R['learning_rate'] = 5e-3        # 学习率
    R['learning_rate_decay'] = 2e-4  # 学习率 decay

    # R['learning_rate'] = 2e-4        # 学习率
    # R['learning_rate_decay'] = 5e-5  # 学习率 decay

    R['regular_wb_model'] = 'L0'
    # R['regular_wb_model'] = 'L1'
    # R['regular_wb_model'] = 'L2'
    # R['penalty2weight_biases'] = 0.000  # Regularization parameter for weights
    R['penalty2weight_biases'] = 0.00001  # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0001  # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.001                  # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0025                 # Regularization parameter for weights

    # 边界的惩罚处理方式,以及边界的惩罚因子
    R['activate_penalty2bd_increase'] = 0
    # R['activate_penalty2bd_increase'] = 1

    if R['activate_penalty2bd_increase'] == 0:
        R['init_boundary_penalty'] = 100  # Regularization parameter for boundary conditions
    else:
        R['init_boundary_penalty'] = 10  # Regularization parameter for boundary conditions

    R['activate_penalty2init_increase'] = 0
    # R['activate_penalty2init_increase'] = 1

    if R['activate_penalty2init_increase'] == 0:
        R['init_init_penalty'] = 100
    else:
        R['init_init_penalty'] = 10

    # 网络的频率范围设置
    R['freq'] = np.concatenate(([1], np.arange(1, 30 - 1)), axis=0)

    # &&&&&&&&&&&&&&&&&&& 使用的网络模型 &&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['model2NN'] = 'DNN'
    # R['model2NN'] = 'Scale_DNN'
    R['model2NN'] = 'Fourier_DNN'

    # &&&&&&&&&&&&&&&&&&&&&& 隐藏层的层数和每层神经元数目 &&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if R['model2NN'] == 'Fourier_DNN':
        R['hidden_layers'] = (50, 80, 60, 60, 40)  # 1*125+250*100+100*80+80*80+80*60+60*1= 44385 个参数
    else:
        R['hidden_layers'] = (100, 80, 60, 60, 40)  # 1*250+250*100+100*80+80*80+80*60+60*1= 44510 个参数

    # &&&&&&&&&&&&&&&&&&& 激活函数的选择 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    R['name2act_in'] = 'tanh'
    # R['name2act_in'] = 's2relu'
    # R['name2act_in'] = 'sin'
    # R['name2act_in'] = 'gelu'
    # R['name2act_in'] = 'mgelu'
    # R['name2act_in'] = 'sinADDcos'

    # R['name2act_hidden'] = 'relu'
    R['name2act_hidden'] = 'tanh'
    # R['name2act_hidden'] = 'srelu'
    # R['name2act_hidden'] = 's2relu'
    # R['name2act_hidden'] = 'sin'
    # R['name2act_hidden'] = 'gelu'
    # R['name2act_hidden'] = 'mgelu'
    # R['name2act_hidden'] = 'sinADDcos'
    # R['name2act_hidden'] = 'elu'
    # R['name2act_hidden'] = 'phi'

    R['name2act_out'] = 'linear'

    R['sfourier'] = 1.0
    if R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 'tanh':
        R['sfourier'] = 1.0
    elif R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 's2relu':
        R['sfourier'] = 1.0
    else:
        R['sfourier'] = 1.0

    solve_Advection_diffusion(R)