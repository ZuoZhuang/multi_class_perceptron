from file_io import Dataset
from perceptron import Perceptron
# import numpy as np
# import math


num_input = 784

perceptron0 = Perceptron(num_input=num_input)
perceptron1 = Perceptron(num_input=num_input)
perceptron2 = Perceptron(num_input=num_input)
perceptron3 = Perceptron(num_input=num_input)
perceptron4 = Perceptron(num_input=num_input)
perceptron5 = Perceptron(num_input=num_input)
perceptron6 = Perceptron(num_input=num_input)
perceptron7 = Perceptron(num_input=num_input)
perceptron8 = Perceptron(num_input=num_input)
perceptron9 = Perceptron(num_input=num_input)

def multi_label_test(dataset):
    num_correct = 0.0
    while (dataset.have_next()):
        data, label = dataset.next_data_set()
        predict_label0, confidence0 = perceptron0.predict(data)
        predict_label1, confidence1 = perceptron1.predict(data)
        predict_label2, confidence2 = perceptron2.predict(data)
        predict_label3, confidence3 = perceptron3.predict(data)
        predict_label4, confidence4 = perceptron4.predict(data)
        predict_label5, confidence5 = perceptron5.predict(data)
        predict_label6, confidence6 = perceptron6.predict(data)
        predict_label7, confidence7 = perceptron7.predict(data)
        predict_label8, confidence8 = perceptron8.predict(data)
        predict_label9, confidence9 = perceptron9.predict(data)

        predict_label = 0
        max_confidence = confidence0

        if (confidence1 > max_confidence):
            predict_label = 1
            max_confidence = confidence1
        if (confidence2 > max_confidence):
            predict_label = 2
            max_confidence = confidence2
        if (confidence3 > max_confidence):
            predict_label = 3
            max_confidence = confidence3
        if (confidence4 > max_confidence):
            predict_label = 4
            max_confidence = confidence4
        if (confidence5 > max_confidence):
            predict_label = 5
            max_confidence = confidence5
        if (confidence6 > max_confidence):
            predict_label = 6
            max_confidence = confidence6
        if (confidence7 > max_confidence):
            predict_label = 7
            max_confidence = confidence7
        if (confidence8 > max_confidence):
            predict_label = 8
            max_confidence = confidence8
        if (confidence9 > max_confidence):
            predict_label = 9
            max_confidence = confidence9

        if (predict_label == label):
            num_correct += 1

    print('accuracy = {0}'.format(num_correct/dataset.num_data()))


def multi_class_perceptron(epoch, learning_rate, random_data_order, random_weight, print_mesg):
    '''The multi-class model of Perceptron

    Args:
        epoch (int): number of iterations you want train the model
        learning_rate (float): if > 0: fixed learning_rate; if < 0, change
            with epoch (1 / epoch).
        random_data_order (bool): if you want train the model with random
            data input, then set this flag to true.
        random_weight (boolean): if True, then the initial value of weight
            is random(uniform distribution); if False, then the initial
            value of weight is zero.
        print_mesg (bool): if you do not want print message, set this to False
    '''
    train_dataset = Dataset("./data/train.p")
    test_dataset = Dataset("./data/test.p")

    # perceptron for label 0
    # perceptron0 = Perceptron(num_input=num_input)
    perceptron0.init_model(learning_rate=learning_rate, target_label=0,
                           random_data_order=random_data_order,
                           random_weight=random_weight)
    print('[perceptron 0] Training...')
    perceptron0.train(train_dataset, epoch=epoch, print_mesg=print_mesg)
    print('[perceptron 0] training result: ', end='')
    perceptron0.test(train_dataset)
    print('[perceptron 0] test result: ', end='')
    perceptron0.test(test_dataset)
    print('-----------')

    # perceptron for label 1
    # perceptron1 = Perceptron(num_input=num_input)
    perceptron1.init_model(learning_rate=learning_rate, target_label=1,
                           random_data_order=random_data_order,
                           random_weight=random_weight)
    print('[perceptron 1] Training...')
    perceptron1.train(train_dataset, epoch=epoch, print_mesg=print_mesg)
    print('[perceptron 1] training result: ', end='')
    perceptron1.test(train_dataset)
    print('[perceptron 1] test result: ', end='')
    perceptron1.test(test_dataset)
    print('-----------')

    # perceptron for label 2
    # perceptron2 = Perceptron(num_input=num_input)
    perceptron2.init_model(learning_rate=learning_rate, target_label=2,
                           random_data_order=random_data_order,
                           random_weight=random_weight)
    print('[perceptron 2] Training...')
    perceptron2.train(train_dataset, epoch=epoch, print_mesg=print_mesg)
    print('[perceptron 2] training result: ', end='')
    perceptron2.test(train_dataset)
    print('[perceptron 2] test result: ', end='')
    perceptron2.test(test_dataset)
    print('-----------')

    # perceptron for label 3
    # perceptron3 = Perceptron(num_input=num_input)
    perceptron3.init_model(learning_rate=learning_rate, target_label=3,
                           random_data_order=random_data_order,
                           random_weight=random_weight)
    print('[perceptron 3] Training...')
    perceptron3.train(train_dataset, epoch=epoch, print_mesg=print_mesg)
    print('[perceptron 3] training result: ', end='')
    perceptron3.test(train_dataset)
    print('[perceptron 3] test result: ', end='')
    perceptron3.test(test_dataset)
    print('-----------')

    # perceptron for label 4
    # perceptron4 = Perceptron(num_input=num_input)
    perceptron4.init_model(learning_rate=learning_rate, target_label=4,
                           random_data_order=random_data_order,
                           random_weight=random_weight)
    print('[perceptron 4] Training...')
    perceptron4.train(train_dataset, epoch=epoch, print_mesg=print_mesg)
    print('[perceptron 4] training result: ', end='')
    perceptron4.test(train_dataset)
    print('[perceptron 4] test result: ', end='')
    perceptron4.test(test_dataset)
    print('-----------')

    # perceptron for label 5
    # perceptron5 = Perceptron(num_input=num_input)
    perceptron5.init_model(learning_rate=learning_rate, target_label=5,
                           random_data_order=random_data_order,
                           random_weight=random_weight)
    print('[perceptron 5] Training...')
    perceptron5.train(train_dataset, epoch=epoch, print_mesg=print_mesg)
    print('[perceptron 5] training result: ', end='')
    perceptron5.test(train_dataset)
    print('[perceptron 5] test result: ', end='')
    perceptron5.test(test_dataset)
    print('-----------')

    # perceptron for label 6
    # perceptron6 = Perceptron(num_input=num_input)
    perceptron6.init_model(learning_rate=learning_rate, target_label=6,
                           random_data_order=random_data_order,
                           random_weight=random_weight)
    print('[perceptron 6] Training...')
    perceptron6.train(train_dataset, epoch=epoch, print_mesg=print_mesg)
    print('[perceptron 6] training result: ', end='')
    perceptron6.test(train_dataset)
    print('[perceptron 6] test result: ', end='')
    perceptron6.test(test_dataset)
    print('-----------')

    # perceptron for label 7
    # perceptron7 = Perceptron(num_input=num_input)
    perceptron7.init_model(learning_rate=learning_rate, target_label=7,
                           random_data_order=random_data_order,
                           random_weight=random_weight)
    print('[perceptron 7] Training...')
    perceptron7.train(train_dataset, epoch=epoch, print_mesg=print_mesg)
    print('[perceptron 7] training result: ', end='')
    perceptron7.test(train_dataset)
    print('[perceptron 7] test result: ', end='')
    perceptron7.test(test_dataset)
    print('-----------')

    # perceptron for label 8
    # perceptron8 = Perceptron(num_input=num_input)
    perceptron8.init_model(learning_rate=learning_rate, target_label=8,
                           random_data_order=random_data_order,
                           random_weight=random_weight)
    print('[perceptron 8] Training...')
    perceptron8.train(train_dataset, epoch=epoch, print_mesg=print_mesg)
    print('[perceptron 8] training result: ', end='')
    perceptron8.test(train_dataset)
    print('[perceptron 8] test result: ', end='')
    perceptron8.test(test_dataset)
    print('-----------')

    # perceptron for label 9
    # perceptron9 = Perceptron(num_input=num_input)
    perceptron9.init_model(learning_rate=learning_rate, target_label=9,
                           random_data_order=random_data_order,
                           random_weight=random_weight)
    print('[perceptron 9] Training...')
    perceptron9.train(train_dataset, epoch=epoch, print_mesg=print_mesg)
    print('[perceptron 9] training result: ', end='')
    perceptron9.test(train_dataset)
    print('[perceptron 9] test result: ', end='')
    perceptron9.test(test_dataset)
    print('-----------')

    print('All model train completed!')
    print('============================')

    print('[Training Result] ', end='')
    multi_label_test(train_dataset)

    print('[Test Result] ', end='')
    multi_label_test(test_dataset)


if __name__ == "__main__":
    '''The multi-class model of Perceptron

    Args:
        epoch (int): number of iterations you want train the model
        learning_rate (float): if > 0: fixed learning_rate; if < 0, change
            with epoch (1 / epoch).
        random_data_order (bool): if you want train the model with random
            data input, then set this flag to true.
        random_weight (boolean): if True, then the initial value of weight
            is random(uniform distribution); if False, then the initial
            value of weight is zero.
        print_mesg (bool): if you do not want print message, set this to False
    '''

    multi_class_perceptron(epoch=1000, learning_rate=0.001, random_data_order=True, random_weight=True, print_mesg=False)
