 # Stochastic Gradient Descent Neural Network in Pure C

A lightweight, efficient neural network implementation using **stochastic gradient descent (SGD)**, written entirely in **pure C**. This project focuses on performance, simplicity, and a low-level understanding of neural network mechanics without the overhead of external libraries.

## Features

- Fully connected feedforward neural network architecture
- Stochastic Gradient Descent (SGD) for training
- Dynamic network topology (adjustable layers and neurons)
- No dependencies—written in pure C

## How It Works

The core of this neural network relies on:

1. **Forward Propagation:** Computes outputs layer-by-layer using weighted sums and activation functions.
2. **Backpropagation:** Calculates gradients for each weight to minimize error.
3. **Stochastic Gradient Descent:** Updates weights using gradients from random subsets (mini-batches) of data.


## 📝 Usage

### Training the Network

```c
#include "include/math.h"
#include "include/structure.h"
#include "include/neural_network.h"
//include additonal header for handling strings and io
int main(){
	//linear input matrix of size 4 x 2
	Matrix * inputs = initMatrix(4,2);
	float inputsf[8] ={1,0,	1,1,	 0,1,	0,0};
	memcpy(inputs->matrix, inputsf,sizeof(float)*8);
	
	//linear input matrix of size 4 x 1
	Matrix * outputs = initMatrix(4,1);
	float outputf[4] = {1,	0,	 1,	0};
	memcpy(outputs->matrix, outputf,sizeof(float)*4);

	//layers Vector {2,2,1} 2 neurons at the input 2 inputs at the hidden layer 1 output
	Vector * layersV = initVector(3);
	float layers[3] = {2,2,1};
	memcpy(layersV->vector, layers,sizeof(float)*3);
	
	int layers_size=3;
	int learning_rate=1;
	neural_network nn = {
		.layers_size=layers_size,
		.layer_size = layersV,
		.learning_rate = learning_rate,
		.input =inputs,
		.output =outputs

	};
	construct_network(&nn);
	train_network(&nn,2000);
	nn.test_mode = 1;
	test_network(&nn);

}
```

### Configuration Options

- **Learning Rate:** Adjustable to control the step size during optimization.
- **Layers and neurons:** Adjustable Layers and Neurons size.
- **Epochs:** Number of iterations over the training dataset.
- **Activation Functions:** Only Sigmoid avaliable now.

## ⚙️ Build & Run

```bash
make              # Compiles the program
make run  # Runs the neural network
```

## 📊 Example Dataset

Sample datasets for classification and regression tasks are provided in main.c(XOR) predicting.

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository, open issues, or submit pull requests.

## 📄 License

This project is licensed under the MIT License.

---

**Note:** This project is intended for educational purposes, demonstrating the fundamental principles of neural networks and SGD in C.
