Usage: cd to folder with main.py, then run python main.py

Further editing:

Edit following function to adjust the model:
	multi_class_perceptron(epoch=1000, learning_rate=0.001, random_data_order=True, random_weight=True, print_mesg=False)

Arguements:

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