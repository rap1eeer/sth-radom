import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')


def load_data():
    # 从文件导入数据
    datafile = 'housing.data'
    data = np.fromfile(datafile, sep=' ')
    print(data.shape)
    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                     'MEDV']
    feature_num = len(feature_names)

    # 将原始数据进行reshape, 变为[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])
    print(data.shape)

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    data_slice = data[:offset]

    # 计算train数据集的最大值、最小值和平均值
    maxinums, mininums, avgs = data_slice.max(axis=0), data_slice.min(axis=0), data_slice.sum(axis=0) / \
                               data_slice.shape[0]

    # 对数据进行归一化处理
    for i in range(feature_num):
        # print(maxinums[i], mininums[i], avgs[i])
        data[:, i] = (data[:, i] - avgs[i]) / (maxinums[i] - mininums[i])

    # 训练集和测试集的划分比例
    # ratio = 0.8
    train_data = data[:offset]
    test_data = data[offset:]

    return train_data, test_data


class NetWork(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置了固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0

    def forward(self, x):
        z = np.dot(x, self.w) + self.b

        return z

    def loss(self, z, y):
        error = z - y
        cost = error * error
        cost = np.mean(cost)

        return cost

    def predict(self, x):
        z = self.forward(x)
        return z

    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z - y) * x
        gradient_w = np.mean(gradient_w, axis=0)  # axis=0表示把每一行做相加然后再除以总的行数
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)
        # 此处b是一个数值，所以可以直接用np.mean得到一个标量（scalar）
        return gradient_w, gradient_b

    def update(self, gradient_w, gradient_b, eta=0.01):  # eta代表学习率，是控制每次参数值变动的大小，即移动步长，又称为学习率
        self.w = self.w - eta * gradient_w  # 相减: 参数向梯度的反方向移动
        self.b = self.b - eta * gradient_b

    def train(self, x, y, iterations=1000, eta=0.01):
        losses = []
        for i in range(iterations):
            # 四步法
            z = self.forward(x)
            L = self.loss(z, y)
            gradient_w, gradient_b = self.gradient(x, y)
            self.update(gradient_w, gradient_b, eta)
            losses.append(L)
            if (i + 1) % 10 == 0:
                print('iter {}, loss {}'.format(i, L))
        return losses


# 获取数据
train_data, test_data = load_data()
print(train_data.shape)
x = train_data[:, :-1]
y = train_data[:, -1:]

# 创建网络
net = NetWork(13)
num_iterations = 2000
# 启动训练
losses = net.train(x, y, iterations=num_iterations, eta=0.01)

# 画出损失函数的变化趋势
plot_x = np.arange(num_iterations)
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)

x_test = test_data[:, :-1]
y_test = test_data[:, -1:]

# 使用训练好的模型进行预测
y_pred = net.predict(x_test)

# 计算均方根误差（RMSE）作为评估指标
mse = np.mean((y_pred - y_test) ** 2)
rmse = np.sqrt(mse)
print('模型在测试集上的均方根误差为:', rmse)
plt.show()
