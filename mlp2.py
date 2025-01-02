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
HIDDEN_LAYER2 = 4
OUTPUT_LAYER = 2

class MLP_momentum2:
    def __init__(self, activation_func=sigmoid, activation_derivative=sigmoid_derivative, num_epoch=num_epoch, learning_rate=lr, momentum=momentum, input_layer=INPUT_LAYER,
                 hidden_layer=HIDDEN_LAYER, hidden_layer2=HIDDEN_LAYER2, output_layer=OUTPUT_LAYER, init_weight_value=None, patience=10, if_print=False):
        
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
        self.hidden_layer2 = hidden_layer2
        self.output_layer = output_layer

        if init_weight_value is not None:
            self.weights_1 = np.full((self.input_layer, self.hidden_layer), init_weight_value, dtype=np.float64)
            self.weights_2 = np.full((self.hidden_layer, self.hidden_layer2), init_weight_value, dtype=np.float64)
            self.weights_3 = np.full((self.hidden_layer2, self.output_layer), init_weight_value, dtype=np.float64)
            self.bias_1 = np.full((1, hidden_layer), init_weight_value, dtype=np.float64)
            self.bias_2 = np.full((1, hidden_layer2), init_weight_value, dtype=np.float64)
            self.bias_3 = np.full((1, output_layer), init_weight_value, dtype=np.float64)
        else:
            self.weights_1 = np.random.uniform(-1, 1, (self.input_layer, self.hidden_layer)).astype(np.float64)
            self.weights_2 = np.random.uniform(-1, 1, (self.hidden_layer, self.hidden_layer2)).astype(np.float64)
            self.weights_3 = np.random.uniform(-1, 1, (self.hidden_layer2, self.output_layer)).astype(np.float64)
            self.bias_1 = np.random.uniform(-1, 1, (1, self.hidden_layer)).astype(np.float64)
            self.bias_2 = np.random.uniform(-1, 1, (1, self.hidden_layer2)).astype(np.float64)
            self.bias_3 = np.random.uniform(-1, 1, (1, self.output_layer)).astype(np.float64)

        self.costs = []

        self.start_w1 = self.weights_1
        self.start_w2 = self.weights_2
        self.start_w3 = self.weights_3
        self.start_wb1 = self.bias_1
        self.start_wb2 = self.bias_2
        self.start_wb3 = self.bias_3

        self.weights_1_history = [self.start_w1]
        self.weights_2_history = [self.start_w2]
        self.weights_3_history = [self.start_w3]

        self.v_w1 = np.zeros_like(self.weights_1)
        self.v_w2 = np.zeros_like(self.weights_2)
        self.v_w3 = np.zeros_like(self.weights_3)
        self.v_b1 = np.zeros_like(self.bias_1)
        self.v_b2 = np.zeros_like(self.bias_2)
        self.v_b3 = np.zeros_like(self.bias_3)

        self.best_w1 = self.start_w1
        self.best_w2 = self.start_w2
        self.best_w3 = self.start_w3
        self.best_b1 = self.start_wb1
        self.best_b2 = self.start_wb2
        self.best_b3 = self.start_wb3

    def forward_propagation(self, X):
        z = np.dot(X, self.weights_1) + self.bias_1
        h = self.activation_func(z)
        z2 = np.dot(h, self.weights_2) + self.bias_2
        h2 = self.activation_func(z2)
        p = np.dot(h2, self.weights_3) + self.bias_3
        return z, h, z2, h2, p

    def cost(self, y, p):
        return np.mean((y - p) ** 2)

    def back_propagation(self, X, y, z1, h1, z2, h2, p):
        loss_gradient = 2 * (p - y) / len(y)  

        # Gradients for weights_3 and bias_3 (output layer)
        dw3 = np.dot(h2.T, loss_gradient)
        db3 = np.sum(loss_gradient, axis=0, keepdims=True)

        # Propagacja błędu przez drugą warstwę ukrytą (h2 -> z2)
        dh2 = np.dot(loss_gradient, self.weights_3.T)
        dz2 = dh2 * self.activation_derivative(z2)

        # Gradients for weights_2 and bias_2 (second hidden layer)
        dw2 = np.dot(h1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Propagacja błędu przez pierwszą warstwę ukrytą (h1 -> z1)
        dh1 = np.dot(dz2, self.weights_2.T)
        dz1 = dh1 * self.activation_derivative(z1)

        # Gradients for weights_1 and bias_1 (first hidden layer)
        dw1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return dw3, dw2, dw1, db3, db2, db1


    def train(self, X, y):
        for epoch in range(self.epochs):
            self.num += 1 
            z, h, z2, h2, p = self.forward_propagation(X)
            dw3, dw2, dw1, db3, db2, db1 = self.back_propagation(X, y, z, h, z2, h2, p)

            # momentum update
            self.v_w3 = self.momentum * self.v_w3 - self.learning_rate * dw3
            self.v_w2 = self.momentum * self.v_w2 - self.learning_rate * dw2
            self.v_w1 = self.momentum * self.v_w1 - self.learning_rate * dw1
            self.v_b3 = self.momentum * self.v_b3 - self.learning_rate * db3
            self.v_b2 = self.momentum * self.v_b2 - self.learning_rate * db2
            self.v_b1 = self.momentum * self.v_b1 - self.learning_rate * db1

            # weights update
            self.weights_3 += self.v_w3
            self.weights_2 += self.v_w2
            self.weights_1 += self.v_w1
            self.bias_3 += self.v_b3
            self.bias_2 += self.v_b2
            self.bias_1 += self.v_b1

            # adding weights to the history to store the best weights 
            self.weights_1_history.append(self.weights_1.copy())
            self.weights_2_history.append(self.weights_2.copy())
            self.weights_3_history.append(self.weights_3.copy())

            # computing loss and adding to history 
            train_loss = self.cost(y, p)
            self.costs.append(train_loss)

            # checking early stopping and updating the best weights if so 
            if self.patience:
                if train_loss < self.best_train_loss:
                    self.best_train_loss = train_loss
                    self.best_w1 = self.weights_1
                    self.best_w2 = self.weights_2
                    self.best_w3 = self.weights_3
                    self.best_b1 = self.bias_1
                    self.best_b2 = self.bias_2
                    self.best_b3 = self.bias_3
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
        z, h, z2, h2, p = self.forward_propagation(x)
        return p

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
    

def cost_per_lr2(X,Y):
    lr_list = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5]
    num_epochs = [1000, 10000, 100000]
    for epoch in num_epochs:
        costs = []
        for lr in lr_list:
            mlp = MLP_momentum2(sigmoid, sigmoid_derivative, num_epoch=epoch, learning_rate=lr, momentum=0)
            mlp.train(X,Y)
            costs.append(np.mean(mlp.costs))
        plt.plot(lr_list, costs, marker='o', label=f'Epoka = {epoch}')
        for i, cost in enumerate(costs):
            plt.text(lr_list[i], cost, f'{cost:.4f}', ha='center', va='bottom', fontsize=9, color='red')
    
    plt.grid()
    plt.xlabel('Współczynnik uczenia')
    plt.ylabel('Koszt')
    plt.title('Koszt funkcji vs wsp uczenia dla różej liczby epok')
    plt.legend()  
    plt.show() 

def cost_per_m2(X, Y):
    momentum_values = [0.0, 0.3, 0.5, 0.9]
    num_epochs = [1000, 10000, 100000]
    
    plt.figure(figsize=(8, 6))  
    for epoch in num_epochs:
        costs = []
        for m in momentum_values:
            mlp = MLP_momentum2(sigmoid, sigmoid_derivative, num_epoch=epoch, momentum=m, learning_rate=0.1)
            mlp.train(X, Y)
            costs.append(np.mean(mlp.costs))
        plt.plot(momentum_values, costs, marker='o', label=f'Epoka = {epoch}')
        for i, cost in enumerate(costs):
            plt.text(momentum_values[i], cost, f'{cost:.4f}', ha='center', va='bottom', fontsize=9, color='red')
    
    plt.grid()
    plt.xlabel('Momentum')
    plt.ylabel('Koszt')
    plt.title('Koszt funkcji vs momentum dla różej liczby epok')
    plt.legend()  
    plt.show()


def cost_per_epoch2(X, Y):
    num_epochs = [100, 1000, 10000, 100000]
    
    plt.figure(figsize=(8, 6))  
    costs = []
    
    for epoch in num_epochs:    
        mlp = MLP_momentum2(sigmoid, sigmoid_derivative, num_epoch=epoch, momentum=0.9, learning_rate=0.5)
        mlp.train(X, Y)
        costs.append(np.mean(mlp.costs))    

    plt.plot(num_epochs, costs, marker='o', linestyle='-', color='b', label="Koszt")

    for i, cost in enumerate(costs):
        plt.text(num_epochs[i], cost, f'{cost:.4f}', ha='center', va='bottom', fontsize=11, color='red')
    
    plt.grid()
    plt.xlabel('Liczba epok')
    plt.ylabel('Koszt')
    plt.title('Koszt funkcji vs liczba epok')
    plt.legend()  
    plt.show()


def check_func_act2(X,Y):
    mlp1 = MLP_momentum2(activation_func=sigmoid, activation_derivative=sigmoid_derivative, num_epoch=1000, learning_rate=0.1, momentum=0.9)
    mlp2 = MLP_momentum2(activation_func=relu, activation_derivative=relu_derivative, num_epoch=1000, learning_rate=0.1, momentum=0.9)
    mlp1.train(X,Y)
    mlp2.train(X,Y)

    fig, ax = plt.subplots(1, 2, figsize=(10,6))
    ax[0] = mlp1.plot_cost_func(ax[0], label='Sigmoid')
    ax[1] = mlp2.plot_cost_func(ax[1], label='Relu')

    plt.show()

def check_stop2(X,Y):
    mlp1= MLP_momentum2(activation_func=sigmoid, activation_derivative=sigmoid_derivative, num_epoch=10000, learning_rate=0.1, momentum=0.9, patience=5)
    mlp2 = MLP_momentum2(activation_func=sigmoid, activation_derivative=sigmoid_derivative, num_epoch=10000, learning_rate=0.1, momentum=0.9, patience=10)
    mlp3 = MLP_momentum2(activation_func=sigmoid, activation_derivative=sigmoid_derivative, num_epoch=10000, learning_rate=0.1, momentum=0.9, patience=10)

    mlp1.train(X,Y)
    mlp2.train(X,Y)
    mlp3.train(X,Y)

    fig, ax = plt.subplots(1, 3, figsize=(15,6))
    ax[0] = mlp1.plot_cost_func(ax[0], label='5')
    ax[1] = mlp2.plot_cost_func(ax[1], label='10')
    ax[2] = mlp3.plot_cost_func(ax[2], label='50')
    plt.show()


def run_simulation2():
    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])  # Przykład funkcji XOR
    Y = np.array([[1], [1], [0], [0]])  # Oczekiwane wyniki

    learning_rates = [0.01, 0.1, 0.5]
    momentum_values = [0.0, 0.5, 0.9]

    results = []
    fig, axes = plt.subplots(len(learning_rates), len(momentum_values), figsize=(15, 10), sharex=True, sharey=True)

    for i, lr in enumerate(learning_rates):
        for j, momentum in enumerate(momentum_values):
            ax = axes[i, j]  # Wybór odpowiedniego subplotu

            print(f"Testing: lr={lr}, momentum={momentum}")
            mlp = MLP_momentum2(
                activation_func=sigmoid,
                activation_derivative=sigmoid_derivative,
                num_epoch=10000,  
                learning_rate=lr,
                input_layer=2,
                hidden_layer=2,
                output_layer=1,
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

if __name__ == '__main__':
    mlp = MLP_momentum2()
    X = np.array(pd.read_csv('signals/input_data.csv', header=None))
    Y = (pd.read_csv('signals/target_data.csv', header=None))
    Y = np.array(Y[[0,1]])
    mlp.train(X,Y)