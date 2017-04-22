import numpy as np
import pickle

class Dataset(object):

    def __init__(self, data_file):
        self.data_set = pickle.load(open(data_file, "rb"))
        # self.data_set = np.array(self.data_set)
        self.current = 0


    def num_data(self):
        return np.shape(self.data_set)[0]

    def shuffle_dataset(self):
        '''Shuffle the dataset for random input'''
        np.random.shuffle(self.data_set)


    def have_next(self):
        '''Check whether there is still data left

        Returns:
            bool: if not reach the end of the dataset, return True; else return
                False
        '''
        if (self.current < len(self.data_set)):
            return True
        else:
            self.current = 0
            return False


    def next_data_set(self, target_label=-2):
        '''Return next data and label in the dataset

        Args:
            target_label (int): the target label you want train

        Returns:
            numpy array: the data returned
            int: the label of the data. If the label is the target_label, then
                is 1, otherwise, -1
        '''
        ret_data = np.array(self.data_set[self.current][0])
        ret_data = np.append(ret_data, [1])
        shape = np.shape(ret_data)
        ret_data = ret_data.reshape((1, shape[0]))
        ret_label = self.data_set[self.current][1]

        if (target_label != -2):
            if (ret_label == target_label):
                ret_label = 1
            else:
                ret_label = -1

        self.current += 1

        # return np.array(ret_data), ret_label
        return ret_data, ret_label


if __name__ == "__main__":
    data_set = Dataset("./data/train.p")

    # data_set.shuffle_dataset()

    # if (data_set.have_next()):
        # # data, label = data_set.next_data_set(target_label=0)
        # data, label = data_set.next_data_set()

    # print(data)
    # print(np.shape(data))
    while (data_set.have_next()):
        _, label = data_set.next_data_set()
        print(label)
    # train_data = pickle.load(open("./data/train.p", "rb"))

    print(data_set.num_data())

    # print(train_data[1][0]);
