from file_io import Dataset
import numpy as np
import math

class Perceptron(object):

    def __init__(self, num_input):
        self.num_input = num_input


    def init_model(self,
                   learning_rate,
                   target_label,
                   random_data_order=True,
                   random_weight=True):
        '''Initialize the model of Perceptron

        Args:
            learning_rate (float): if > 0: fixed learning_rate, if < 0, change
                with epoch (1 / epoch).
            target_label (int): the label you want train on
            random_data_order (bool): if you want train the model with random
                data input, then set this flag to true.
            random_weight (boolean): if True, then the initial value of weight
                is random(uniform distribution); if False, then the initial
                value of weight is zero.
        '''
        self.learning_rate = learning_rate
        if (self.learning_rate > 0):
            self.dec_lr = False
        else:
            self.dec_lr = True

        self.target_label = target_label

        if (random_weight):
            self.weight = np.random.rand(self.num_input+1, 1)
        else:
            self.weight = np.zeros((self.num_input+1, 1))

        self.random_data_order = random_data_order


    def predict(self,data):
        '''Predict the label of a given data

        Args:
            data (numpy array): data you want to predict

        Returns:
            int: the predicted label of the data
            numpy array: the confidence of a predict
        '''
        confidence = np.dot(data, self.weight)

        if (confidence > 0):
            # return 1, np.absolute(confidence)
            return 1, confidence
        elif (confidence == 0):
            # return 0, np.absolute(confidence)
            return 0, confidence
        else:
            # return -1, np.absolute(confidence)
            return -1, confidence


    def train(self, dataset, epoch, plot_error=True, print_mesg=True):
        '''Train the model with give data and epoch

        Args:
            dataset (numpy array): the dataset used to train the model
            epoch (int): numuber of iterations you want train the data
            plot_error (bool): if you want to plot the error curve of training
                result in each epoch, then set this flag to true
            print_mesg (bool): if you do not want print message, set this to False
        Todo:
            implement plotting function
        '''
        for i in range(epoch):
            if (self.dec_lr):
                if (i == 0):
                    self.learning_rate = 1
                else:
                    self.learning_rate = 1.0 / i

            if (self.random_data_order):
                dataset.shuffle_dataset()

            sum_error = 0.0
            sum_delta = 0.0
            num_correct = 0.0
            while (dataset.have_next()):
                error = 0.0
                data, label = dataset.next_data_set(self.target_label)
                predict_label, _ = self.predict(data)
                error = label - predict_label

                if (error == 0):
                    num_correct += 1.0
                # print('[epoch] error: {0:f}'.format(error))
                sum_error += error ** 2
                # print('[epoch] weight_update: {0}'.format(error * data.T))
                # self.weight += self.learning_rate * error * data.T
                sum_delta += self.learning_rate * error * data.T

            self.weight += sum_delta

            if (print_mesg):
                print('epoch: {0:d}, lr: {1:f}, error: {2:f}, accuracy: {3}'.format(
                    i, self.learning_rate, math.sqrt(sum_error), num_correct/dataset.num_data()))




    def test(self, dataset, plot_error=True, print_mesg=True):
        '''Test the model with give data and epoch

        Args:
            dataset (numpy array): the dataset used to train the model
            plot_error (bool): if you want to plot the error curve of training
                result in each epoch, then set this flag to true
            print_mesg (bool): if you do not want print message, set this to False
        Todo:
            implement plotting function
        '''
        num_correct = 0.0
        sum_error = 0.0
        while (dataset.have_next()):
            error = 0.0
            data, label = dataset.next_data_set(self.target_label)
            predict_label, _ = self.predict(data)
            error = label - predict_label

            if (error == 0):
                num_correct += 1.0

            sum_error += error ** 2


        if (print_mesg):
            print('error: {0:f}, accuracy: {1}'.format(
                math.sqrt(sum_error), num_correct/dataset.num_data()))


if __name__ == "__main__":
    perceptron = Perceptron(784)
    # perceptron.init_model(0.5, False)
    perceptron.init_model(0.005, target_label=1, random_weight=True)
    print(perceptron.weight)
    print(np.shape(perceptron.weight))

    dataset = Dataset("./data/train.p")
    # data, label = dataset.next_data_set()

    perceptron.train(dataset, 100)
    perceptron.test(dataset)

    # predict, confidence = perceptron.predict(data)
    # print(type(predict))
    # print(confidence)
    # print(type(confidence))

    # print(1.0/5)
