import numpy as np


class Layer:
    """
    神经网络层基类

    param:
        next_layer: 下一层
    """

    def __init__(self):
        pass

    def set_next_layer(self, layer):
        self.next_layer: Layer = layer

    def forward(self):
        pass

    def backward(self):
        pass

    def update(self):
        self.next_layer.update()


class Linear(Layer):
    """
    线性全连接层

    W 矩阵示例：
        w0, w1, w2
        ... ... ...
        ... ... ...

    X :
         1   ...
        x1   ...
        x2   ...


    W.dot(X):
        w0+w1*x1+w2*x2   ...
            ...          ...
            ...          ...


    param:
        learning_rate: 学习率
        input_dim: 输入维度
        output_dim: 输出维度
        input_mat: 输入矩阵
        output_mat: 输出矩阵
        grad: 梯度矩阵

    """

    def __init__(self, w: np.ndarray):
        self.w = w
        self.input_dim = w.shape[1] - 1
        self.output_dim = w.shape[0]
        self.learning_rate = 1

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def forward(self, input_mat: np.ndarray):
        ones_row = np.ones((1, input_mat.shape[1]))
        self.input_mat = np.vstack((ones_row, input_mat))  # 添加常量输入行
        self.output_mat = self.w.dot(self.input_mat)

        self.next_layer.forward(self.output_mat)

    def backward(self):

        grad_to_pass = self.next_layer.backward()
        self.grad_w = np.einsum("ij,kj->ik", grad_to_pass, self.input_mat)
        self.grad_x = np.einsum("ij,ik->jk", self.w, grad_to_pass)[1:, :]

        return self.grad_x

    def update(self):

        self.w -= self.grad_w * self.learning_rate
        self.next_layer.update()


class Sigmoid(Layer):

    def forward(self, input_mat):
        self.input_mat = input_mat
        self.output_mat = 1 / (1 + np.exp(-input_mat))
        self.next_layer.forward(self.output_mat)

    def backward(self):
        grad_to_pass = self.next_layer.backward()
        return self.output_mat * (1 - self.output_mat) * grad_to_pass


class ReLU(Layer):

    def forward(self, input_mat):
        self.input_mat = input_mat
        self.output_mat = np.where(input_mat > 0, input_mat, 0)
        self.next_layer.forward(self.output_mat)

    def backward(self):
        grad_to_pass = self.next_layer.backward()
        return np.where(self.input_mat > 0, 1, 0) * grad_to_pass


class Leak_ReLU(Layer):

    def forward(self, input_mat):
        self.input_mat = input_mat
        self.output_mat = np.where(input_mat > 0, input_mat, 0.01 * input_mat)
        self.next_layer.forward(self.output_mat)

    def backward(self):
        grad_to_pass = self.next_layer.backward()
        return np.where(self.input_mat > 0, 1, 0) * grad_to_pass


class Loss(Layer):
    """
    损失层

    param:
        loss_history: 损失历史
        target_mat: 目标矩阵

    """

    def __init__(self):
        self.loss_history = []

    def update(self):
        pass

    def set_target_mat(self, target_mat: np.ndarray):
        self.target_mat = target_mat
        self.batch_size = target_mat.shape[1]

    def clear_loss_history(self):
        self.loss_history = []


class Softmax_MCELoss(Loss):
    """
    softmax层和平均交叉熵损失
    """

    def forward(self, input_mat: np.ndarray):
        self.input_mat = input_mat

        x_max = input_mat.max(axis=0)
        exp = np.exp(input_mat - x_max)
        self.output_mat = exp / exp.sum(axis=0)

    def backward(self):
        loss = np.where(self.target_mat > 0, np.log(self.output_mat), 0).sum() / (
            -self.batch_size
        )
        self.loss_history.append(loss)

        grad_to_pass = (self.target_mat - self.output_mat) / (-self.batch_size)
        self.grad = grad_to_pass
        return grad_to_pass

    def get_loss(self):
        return self.loss_history[-1]

    def get_output(self):
        return self.output_mat


class MSL(Loss):
    """
    平均平方误差
    """

    pass


class NN:
    """
    神经网络类，组合各层
    """

    def __init__(self, *args: Layer):
        self.layers = args
        self.first_layer = self.layers[0]
        self.loss_layer = self.layers[-1]
        for i in range(len(args) - 1):
            args[i].set_next_layer(args[i + 1])

    def predict(self, x):
        self.first_layer.forward(x)
        return self.loss_layer.get_output()

    def train(self, input_mat, target_mat, epoch):
        self.loss_layer.clear_loss_history()
        self.loss_layer.set_target_mat(target_mat)
        for i in range(epoch):
            self.first_layer.forward(input_mat)
            self.first_layer.backward()
            self.first_layer.update()

    def show_loss_history(self):
        return self.loss_layer.loss_history
