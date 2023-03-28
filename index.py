import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Linear activation function
def linear(x):
    return x


# ReLU activation function
def relu(x):
    return np.maximum(0, x)


# Tanh activation function
def tanh(x):
    return np.tanh(x)


@app.route('/calculate', methods=['POST'])
def calculate():
    global hidden_output, output, formula
    input1 = float(request.form['input1'])
    input2 = float(request.form['input2'])
    weight1_1 = float(request.form['weight1_1'])
    weight1_2 = float(request.form['weight1_2'])
    weight2_1 = float(request.form['weight2_1'])
    weight2_2 = float(request.form['weight2_2'])
    h_weight1 = float(request.form['h_weight1'])
    h_weight2 = float(request.form['h_weight2'])
    bias1 = float(request.form['bias1'])
    activation = request.form['activation']

    inputs = np.array([input1, input2])
    hidden_weights = np.array([[weight1_1, weight1_2], [weight2_1, weight2_2]])
    hidden_biases = np.array([bias1, bias1])
    output_weights = np.array([h_weight1, h_weight2])
    output_biases = np.array([bias1])

    if activation == 'sigmoid':
        hidden_output = sigmoid(np.dot(inputs, hidden_weights) + hidden_biases)
        output = sigmoid(np.dot(hidden_output, output_weights) + output_biases)
        formula = 'sigmoid(h1*w11 + h2*w21 + b1), sigmoid(h1*w12 + h2*w22 + b2)'
    elif activation == 'tanh':
        hidden_output = tanh(np.dot(inputs, hidden_weights) + hidden_biases)
        output = tanh(np.dot(hidden_output, output_weights) + output_biases)
        formula = 'tanh(h1*w11 + h2*w21 + b1), tanh(h1*w12 + h2*w22 + b2)'
    elif activation == 'relu':
        hidden_output = relu(np.dot(inputs, hidden_weights) + hidden_biases)
        output = relu(np.dot(hidden_output, output_weights) + output_biases)
        formula = 'relu(h1*w11 + h2*w21 + b1), relu(h1*w12 + h2*w22 + b2)'
    elif activation == 'linear':
        hidden_output = linear(np.dot(inputs, hidden_weights) + hidden_biases)
        output = linear(np.dot(hidden_output, output_weights) + output_biases)
        formula = 'linear(h1*w11 + h2*w21 + b1), linear(h1*w12 + h2*w22 + b2)'

    h_weight1 = float(request.form['h_weight1'])
    h_weight2 = float(request.form['h_weight2'])
    data = {
        'input1': input1,
        'input2': input2,
        'weight1_1': weight1_1,
        'weight1_2': weight1_2,
        'weight2_1': weight2_1,
        'weight2_2': weight2_2,
        'bias1': bias1,
        'h_weight1': h_weight1,
        'h_weight2': h_weight2,
        'activation': activation,
        'dot_product_hidden': np.dot(inputs, hidden_weights),
        'sum_with_hidden_bias': np.dot(inputs, hidden_weights) + hidden_biases,
        # 'hidden_output': hidden_output.numpy(),
        'hidden1': hidden_output[0],
        'hidden2': hidden_output[1],
        'dot_product_output': np.dot(hidden_output, output_weights),
        'sum_with_output_bias': float(np.dot(hidden_output, output_weights) + output_biases),
        'output': output[0],
        'formula': formula
    }

    return render_template('index.html', data=data)


if __name__ == '__main__':
    app.run(debug=True)
