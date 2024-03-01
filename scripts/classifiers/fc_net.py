from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # Два слоя, запомнили размерности слоев, запомнили их и зарандомили веса
        self.num_layers = 2

        D_H_C_dims = [input_dim] + [hidden_dim] + [num_classes]

        for idx in range(self.num_layers):
            nrows = D_H_C_dims[idx]
            ncols = D_H_C_dims[idx+1]

            layer_name = "%d" % (idx+1)
            weight_name = "W" + layer_name
            bias_name = "b" + layer_name
            self.params[weight_name] = weight_scale * np.random.randn(nrows, ncols)
            self.params[bias_name] = np.zeros(ncols)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        caches = {}
        scores = X
        # реализуем прямой проход для случая, если слой последний и не последний
        for i in range(1, self.num_layers + 1):
            layer_name = "%d" % i
            W_name = "W" + layer_name
            b_name = "b" + layer_name
            cache_name = "cache" + layer_name

            if self.num_layers == i:
                scores, cache = affine_forward(scores, self.params[W_name],
                                               self.params[b_name])
            else:
                scores, cache = affine_relu_forward(
                scores, self.params[W_name], self.params[b_name])
                
            caches[cache_name] = cache


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # считаем ошибку и градиант при обратном проходе также для двух случаев
        loss, dx = softmax_loss(scores, y)
        
        for i in range(self.num_layers, 0, -1):
            layer_name = "%d" % i
            W_name = "W" + layer_name
            b_name = "b" + layer_name
            cache_name = "cache" + layer_name

            loss += 0.5 * self.reg * (self.params[W_name] ** 2).sum()
            
            if self.num_layers == i:
                dx, grads[W_name], grads[b_name] = affine_backward(dx, caches[cache_name])
                
            else:
                (der, grads[W_name], grads[b_name]) = affine_relu_backward(dx, caches[cache_name])

            grads[W_name] += self.reg * self.params[W_name]
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Обучение этой нейронной сети с использованием стохастического градиентного спуска.

        Входы:
        - X: массив numpy формы (N, D), содержащий обучающие данные.
        - y: массив numpy формы (N,), содержащий метки обучения; y[i] = c означает,
          что X[i] имеет метку c, где 0 <= c < C.
        - X_val: массив numpy формы (N_val, D), содержащий данные для валидации.
        - y_val: массив numpy формы (N_val,), содержащий метки валидации.
        - learning_rate: Скаляр, задающий скорость обучения для оптимизации.
        - learning_rate_decay: Скаляр, задающий коэффициент, используемый для затухания
          скорости обучения после каждой эпохи.
        - reg: Скаляр, задающий степень регуляризации.
        - num_iters: Количество шагов для оптимизации.
        - batch_size: Количество примеров обучения, используемых за один шаг.
        - verbose: логическое значение; если True, вывод прогресса во время оптимизации.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        # Используем SGD для оптимизации параметров в этой модели
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            # Создание случайной минипакетной выборки обучающих данных и меток
            batch_idxes = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_idxes, :]
            y_batch = y[batch_idxes]

            # Вычисление потерь и градиентов с использованием текущей минипакетной выборки
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            # Обновление параметров сети с использованием градиентов
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']

            if verbose and it % 100 == 0:
                print('итерация %d / %d: потери %f' % (it, num_iters, loss))

            # Каждую эпоху проверяем точность на обучающем и валидационном наборах
            # и уменьшаем скорость обучения.
            if it % iterations_per_epoch == 0:
                # Проверка точности
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Уменьшение скорости обучения
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Использование обученных весов этой двухслойной сети для прогнозирования меток
        для данных. Для каждой точки данных мы прогнозируем оценки для каждого из C
        классов и присваиваем каждой точке данных класс с наивысшей оценкой.

        Входы:
        - X: массив numpy формы (N, D), содержащий N D-мерных точек данных для классификации.

        Возвращает:
        - y_pred: массив numpy формы (N,), содержащий предсказанные метки для каждого из
          элементов X. Для всех i y_pred[i] = c означает, что X[i] предсказывается
          иметь класс c, где 0 <= c < C.
        """
        y_pred = None

        # Прямой проход через сеть
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # Прогнозирование оценок для каждого класса для входа
        scores = np.maximum(X.dot(W1) + b1, 0).dot(W2) + b2

        # Выбор класса с наивысшей оценкой для каждой точки данных
        y_pred = np.argmax(scores, axis=1)

        return y_pred


import numpy as np

class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10, 
                 dropout=1, normalization=None, reg=0.0, weight_scale=1e-2, dtype=np.float32, seed=None,):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # реализовали инициализацию, также как и 2-слойной
        D_H_C_dims = [input_dim] + hidden_dims + [num_classes]
        
        for idx in range(self.num_layers):
            nrows = D_H_C_dims[idx]
            ncols = D_H_C_dims[idx+1]

            layer_name = "%d" % (idx+1)
            weight_name = "W" + layer_name
            bias_name = "b" + layer_name
            self.params[weight_name] = weight_scale * np.random.randn(nrows, ncols)
            self.params[bias_name] = np.zeros(ncols)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        caches = {}
        scores = X
        # прямой проход реализовали...
        for i in range(1, self.num_layers + 1):
            layer_name = "%d" % i
            W_name = "W" + layer_name
            b_name = "b" + layer_name
            cache_name = "cache" + layer_name
      
            if self.num_layers == i:
                scores, cache = affine_forward(scores, self.params[W_name], self.params[b_name])

            else:
                scores, cache = affine_relu_forward(scores, self.params[W_name], self.params[b_name])
  
            caches[cache_name] = cache

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        grads = {}
        loss, dx = softmax_loss(scores, y)
        # обратный проход
        for i in range(self.num_layers, 0, -1):
            layer_name = "%d" % i
            W_name = "W" + layer_name
            b_name = "b" + layer_name
            cache_name = "cache" + layer_name
      
            loss += 0.5 * self.reg * (self.params[W_name] ** 2).sum()
      
            if self.num_layers == i:
                dx, grads[W_name], grads[b_name] = affine_backward(dx, caches[cache_name])
                
            else:
                (dx, grads[W_name], grads[b_name]) = affine_relu_backward(dx, caches[cache_name])

            grads[W_name] += self.reg * self.params[W_name]


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


