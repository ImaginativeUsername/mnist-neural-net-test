import mnist_loader
import network

#OPTIONS

epochs = 10
mini_batch_size = 10
eta = 3.0

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

def run():
    
    net = network.Network([784, 30, 10])
    net.SGD(training_data, epochs, mini_batch_size, eta)
