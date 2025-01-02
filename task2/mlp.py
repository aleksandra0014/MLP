import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# activation function 
def sigmoid(y):
    return 1 / (1 + np.exp(-y))

# activation derivative
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Funkcja ReLU
def relu(x):
    return np.maximum(0, x)

# Pochodna funkcji ReLU
def relu_derivative(x):
    return np.where(x > 0, 1, 0)


# step function for prediction 
def step_function(y, threshold=0.5):
    return np.where(y >= threshold, 1, 0)



num_epoch = 10000
lr = 0.1
momentum = 0 
INPUT_LAYER = 16
HIDDEN_LAYER = 8
OUTPUT_LAYER = 2

class MLP_momentum:
    def __init__(self, activation_func=sigmoid, activation_derivative=sigmoid_derivative, num_epoch=num_epoch, learning_rate=lr, momentum=momentum, input_layer=INPUT_LAYER,
                 hidden_layer=HIDDEN_LAYER, output_layer=OUTPUT_LAYER, init_weight_value=None, patience=None, if_print=False):
        
        np.random.seed(45)

        self.epochs = num_epoch
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.activation_func = activation_func
        self.activation_derivative = activation_derivative

        self.num = 0
        self.if_print = if_print
        self.patience = patience
        if patience is not None:
            self.best_train_loss = np.inf
            self.no_impr_count = 0 

        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer

        if init_weight_value is not None:
            self.weights_1 = np.full((self.input_layer, self.hidden_layer), init_weight_value, dtype=np.float64)
            self.weights_2 = np.full((self.hidden_layer, self.output_layer), init_weight_value, dtype=np.float64)
            self.bias_1 = np.full((1, hidden_layer), init_weight_value, dtype=np.float64)
            self.bias_2 = np.full((1, output_layer), init_weight_value, dtype=np.float64)
        else:
            self.weights_1 = np.random.uniform(-1, 1, (self.input_layer, self.hidden_layer)).astype(np.float64)
            self.weights_2 = np.random.uniform(-1, 1, (self.hidden_layer, self.output_layer)).astype(np.float64)
            self.bias_1 = np.random.uniform(-1, 1, (1, self.hidden_layer)).astype(np.float64)
            self.bias_2 = np.random.uniform(-1, 1, (1, self.output_layer)).astype(np.float64)



        self.costs = []

        self.start_w1 = self.weights_1
        self.start_w2 = self.weights_2
        self.start_wb1 = self.bias_1
        self.start_wb2 = self.bias_2

        self.weights_1_history = [self.start_w1]
        self.weights_2_history = [self.start_w2]

        self.v_w1 = np.zeros_like(self.weights_1)
        self.v_w2 = np.zeros_like(self.weights_2)
        self.v_b1 = np.zeros_like(self.bias_1)
        self.v_b2 = np.zeros_like(self.bias_2)

        self.best_w1 = self.start_w1
        self.best_w2 = self.start_w2
        self.best_b1 = self.start_wb1
        self.best_b2 = self.start_wb2

    def forward_propagation(self, X):
        z = np.dot(X, self.weights_1) + self.bias_1
        h = self.activation_func(z)
        p = np.dot(h, self.weights_2) + self.bias_2
        return z, h, p

    def cost(self, y, p):
        return np.mean((y - p) ** 2)

    def back_propagation(self, X, y, z, h, p):

        loss_gradient = 2 * (p - y) / len(y)
        loss_gradient = np.array(loss_gradient)
        self.weights_2 = np.array(self.weights_2)


        dw2 = np.dot(h.T, loss_gradient)
        d1 = np.dot(loss_gradient, self.weights_2.T)
        d2 = np.multiply(d1, self.activation_derivative(z))
        dw1 = np.dot(X.T, d2)
        dwb2 = np.sum(loss_gradient, axis=0, keepdims=True)
        dwb1 = np.sum(d2, axis=0, keepdims=True)
        return dw2, dw1, dwb2, dwb1

    def train(self, X, y):
        for epoch in range(self.epochs):
            self.num += 1 
            z, h, p = self.forward_propagation(X)
            dw2, dw1, dwb2, dwb1= self.back_propagation(X, y, z, h, p)

            # momentum update
            self.v_w2 = self.momentum * self.v_w2 - self.learning_rate * dw2
            self.v_w1 = self.momentum * self.v_w1 - self.learning_rate * dw1
            self.v_b2 = self.momentum * self.v_b2 - self.learning_rate * dwb2
            self.v_b1 = self.momentum * self.v_b1 - self.learning_rate * dwb1

            # weights update
            self.weights_2 += self.v_w2
            self.weights_1 += self.v_w1
            self.bias_2 += self.v_b2
            self.bias_1 += self.v_b1

            # adding weights to the history to store the best weights 
            self.weights_1_history.append(self.weights_1.copy())
            self.weights_2_history.append(self.weights_2.copy())

            # computing loss and adding to history 
            train_loss = self.cost(y, p)
            self.costs.append(train_loss)

            # checking early stopping and updating the best weights if so 
            if self.patience:
                if train_loss < self.best_train_loss:
                    self.best_train_loss = train_loss
                    self.best_w1 = self.weights_1
                    self.best_w2 = self.weights_2
                    self.best_b1 = self.bias_1
                    self.best_b2 = self.bias_2
                    self.no_impr_count = 0
                else:
                    self.no_impr_count += 1

            if self.patience:
                if self.no_impr_count >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1} - Best Train Loss: {self.best_train_loss:.4f}")
                    break
                
            if self.if_print:
            # Wyświetlanie postępu co 100 epok
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}/{self.epochs}, Loss: {self.cost(y,p):.6f}")

        # print('Najlepsze wagi:')
        # print(f'w1: {self.weights_1}')
        # print(f'w2: {self.weights_2}')
        # print(f'b1: {self.bias_1}')
        # print(f'b2: {self.bias_2}')


    def predict(self, x):
        z, h, f = self.forward_propagation(x)
        return f

    def plot_cost_func(self, ax=None, label=None):
        if ax is None:
            ax = plt.gca()  # Jeśli nie podano, używamy domyślnego obszaru
        ax.plot(range(self.num), self.costs, label=label)
        ax.set_xlabel('Epoka')
        ax.set_ylabel('Koszt')
        ax.set_title('Wartość funkcji kosztu')
        ax.legend()

    def avg_cost(self):
        # print('Początkowe wagi: ')
        # print(f'w1: {self.start_w1}, \nw2:{self.start_w2}, \nwb1:{self.start_wb1}, \nwb2:{self.start_wb2}')
        print(f'Średni koszt po wszystkich epokach: {np.mean(self.costs)}')
        print(f'Ostatni koszt: {self.costs[-1]}')
    
    def plot_weights(self):
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        weights_1 = np.array(self.weights_1_history)
        for i in range(self.input_layer):
            for j in range(self.hidden_layer):
                ax[0].plot(range(len(weights_1)), weights_1[:, i, j], label=f"W1[{i},{j}]")
        ax[0].set_title("Zmiany wag: Wejście -> Ukryta")
        ax[0].set_xlabel("Epoka")
        ax[0].set_ylabel("Wartość wag")
        ax[0].legend()

        weights_2 = np.array(self.weights_2_history)
        for i in range(self.hidden_layer):
            for j in range(self.output_layer):
                ax[1].plot(range(len(weights_2)), weights_2[:, i, j], label=f"W2[{i},{j}]")
        ax[1].set_title("Zmiany wag: Ukryta -> Wyjście")
        ax[1].set_xlabel("Epoka")
        ax[1].set_ylabel("Wartość wag")
        ax[1].legend()

        plt.tight_layout()
        plt.show()

def cost_per_lr(X, Y, m, input_layer, hidden_layer, output_layer):
    lr_list = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5]
    num_epochs = [100, 1000, 10000]
    for epoch in num_epochs:
        costs = []
        for lr in lr_list:
            mlp = MLP_momentum(sigmoid, sigmoid_derivative, num_epoch=epoch, learning_rate=lr, momentum=m, input_layer=input_layer, hidden_layer=hidden_layer, output_layer=output_layer)
            mlp.train(X,Y)
            costs.append(np.mean(mlp.costs))
        plt.plot(lr_list, costs, marker='o', label=f'Epoka = {epoch}')
        # for i, cost in enumerate(costs):
        #     plt.text(lr_list[i], cost, f'{cost:.4f}', ha='center', va='bottom', fontsize=9, color='red')
    
    plt.grid()
    plt.xlabel('Współczynnik uczenia')
    plt.ylabel('Koszt')
    plt.title('Koszt funkcji a wsp uczenia dla różnej liczby epok')
    plt.legend()  
    plt.show() 

def cost_per_m(X, Y, lr, input_layer, hidden_layer, output_layer):
    momentum_values = [0.0, 0.3, 0.5, 0.9]
    num_epochs = [100, 1000, 10000]
    
    plt.figure(figsize=(8, 6))  
    for epoch in num_epochs:
        costs = []
        for m in momentum_values:
            mlp = MLP_momentum(sigmoid, sigmoid_derivative, num_epoch=epoch, momentum=m, learning_rate=lr , input_layer=input_layer, hidden_layer=hidden_layer, output_layer=output_layer)
            mlp.train(X, Y)
            costs.append(np.mean(mlp.costs))
        plt.plot(momentum_values, costs, marker='o', label=f'Epoka = {epoch}')
        # for i, cost in enumerate(costs):
        #     plt.text(momentum_values[i], cost, f'{cost:.4f}', ha='center', va='bottom', fontsize=9, color='red')
    
    plt.grid()
    plt.xlabel('Momentum')
    plt.ylabel('Koszt')
    plt.title('Koszt funkcji a momentum dla różnej liczby epok')
    plt.legend()  
    plt.show()


def cost_per_epoch(X, Y, m, lr, input_layer, hidden_layer, output_layer):
    num_epochs = [100, 1000, 10000, 100000]
    
    plt.figure(figsize=(8, 6))  
    costs = []
    
    for epoch in num_epochs:    
        mlp = MLP_momentum(sigmoid, sigmoid_derivative, num_epoch=epoch, momentum=m, learning_rate=lr, input_layer=input_layer, hidden_layer=hidden_layer, output_layer=output_layer)
        mlp.train(X, Y)
        costs.append(np.mean(mlp.costs))    

    plt.plot(num_epochs, costs, marker='o', linestyle='-', color='b', label="Koszt")

    # for i, cost in enumerate(costs):
    #     plt.text(num_epochs[i], cost, f'{cost:.4f}', ha='center', va='bottom', fontsize=11, color='red')
    
    plt.grid()
    plt.xlabel('Liczba epok')
    plt.ylabel('Koszt')
    plt.title('Koszt funkcji a liczba epok')
    plt.legend()  
    plt.show()


def check_func_act(X,Y, lr, m, input_layer, hidden_layer, output_layer):
    mlp1 = MLP_momentum(activation_func=sigmoid, activation_derivative=sigmoid_derivative, num_epoch=1000, learning_rate=lr, momentum=m, input_layer=input_layer, hidden_layer=hidden_layer, output_layer=output_layer)
    mlp2 = MLP_momentum(activation_func=relu, activation_derivative=relu_derivative, num_epoch=1000, learning_rate=lr, momentum=m, input_layer=input_layer, hidden_layer=hidden_layer, output_layer=output_layer)
    mlp1.train(X,Y)
    mlp2.train(X,Y)
    fig, ax = plt.subplots(1, 2, figsize=(10,6))
    ax[0] = mlp1.plot_cost_func(ax[0], label='Sigmoid')
    ax[1] = mlp2.plot_cost_func(ax[1], label='Relu')
    plt.show()
    return mlp1.avg_cost(), mlp2.avg_cost()

    

def check_stop(X,Y, lr, m, input_layer, hidden_layer, output_layer):
    mlp1= MLP_momentum(activation_func=sigmoid, activation_derivative=sigmoid_derivative, num_epoch=10000, learning_rate=lr, momentum=m, patience=5, input_layer=input_layer, hidden_layer=hidden_layer, output_layer=output_layer)
    mlp2 = MLP_momentum(activation_func=sigmoid, activation_derivative=sigmoid_derivative, num_epoch=10000, learning_rate=lr, momentum=m, patience=10, input_layer=input_layer, hidden_layer=hidden_layer, output_layer=output_layer)
    mlp3 = MLP_momentum(activation_func=sigmoid, activation_derivative=sigmoid_derivative, num_epoch=10000, learning_rate=lr, momentum=m, patience=50, input_layer=input_layer, hidden_layer=hidden_layer, output_layer=output_layer)

    mlp1.train(X,Y)
    mlp2.train(X,Y)
    mlp3.train(X,Y)

    fig, ax = plt.subplots(1, 3, figsize=(15,6))
    ax[0] = mlp1.plot_cost_func(ax[0], label='5')
    ax[1] = mlp2.plot_cost_func(ax[1], label='10')
    ax[2] = mlp3.plot_cost_func(ax[2], label='50')
    plt.show()

    return mlp1.avg_cost(), mlp2.avg_cost(), mlp3.avg_cost()
    


def run_simulation(X,Y, input_layer, hidden_layer, output_layer):
    learning_rates = [0.01, 0.1, 0.3, 0.5]
    momentum_values = [0.0, 0.5, 0.9]

    results = []
    fig, axes = plt.subplots(len(learning_rates), len(momentum_values), figsize=(15, 10), sharex=True, sharey=True)

    for i, lr in enumerate(learning_rates):
        for j, momentum in enumerate(momentum_values):
            ax = axes[i, j]  # Wybór odpowiedniego subplotu

            print(f"Testing: lr={lr}, momentum={momentum}")
            mlp = MLP_momentum(
                activation_func=sigmoid,
                activation_derivative=sigmoid_derivative,
                num_epoch=10000,  
                learning_rate=lr,
                input_layer=input_layer,
                hidden_layer=hidden_layer,
                output_layer=output_layer,
                momentum=momentum,
            )
            mlp.train(X, Y)

            ax.plot(mlp.costs, label="epochs=10000")

            last_avg_cost = np.mean(mlp.costs)  # Średni koszt z ostatnich 50 epok

            ax.text(0.5, 0.5, f'Avg Cost: {last_avg_cost:.4f}', ha='center', va='center', fontsize=12, transform=ax.transAxes)
            
            ax.set_title(f"LR={lr}, Momentum={momentum}")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.legend()

    plt.tight_layout()
    plt.show()
    return results

