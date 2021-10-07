# deep_learning_framework

A small deep learning framework, with an uninsprational name... PyDeep.

## Implemented/Planned Features
### Layers
- [x] Linear Layers
- [x] Conv2D
- [x] MaxPool2D
- [ ] rnn
- [ ] lstm
### Activations
- [x] Sigmoid
- [x] ReLU
- [x] LeakyReLu
- [x] SoftMax

### Loss functions
- [x] MSELoss
- [ ] L1Loss
- [x] CrossEntropyLoss
- [ ] BinaryCrossEntropyLoss

### Utilities
- [ ] Dataloader
- [ ] CustomDataset
- [ ] Train
- [ ] CrossValidate
- [ ] EarlyStopping
- [ ] Save Model
- [ ] Load Model
- [ ] Pretrainded Model
- [ ] CUDA Support

## Example Use Case

### Classification MLP
```python
# Training a regular MLP with relu acitvations for classification tasks
import pydeep.nn as nn
import numpy as np



layers = [nn.Linear(784, 100), 
          nn.Sigmoid(), 
          nn.Linear(100, 30), 
          nn.Sigmoid(), 
          nn.Linear(30, 10)]


learning_rate = 0.001 # set learning rate
epochs = 10
batch_size = 10

# initialise model
loss = nn.CrossEntropyLoss() # init. loss function
optim = nn.SGD(lr = learning_rate)  # init. optimiser
model = nn.Sequential(layers=layers, loss=loss, optim=optim) # init model

# training loop
for e in range(epochs):
    epoch_loss = 0
    # currently not supporting dataloader, so we have to diy
    perm = np.random.permutation(len(trainx))
    for j in range(len(trainx) // batch_size):
        batch_idx = perm[j: (j+1) * batch_size]
        
        batchx, batchy = trainx[batch_idx], trainy[batch_idx]
        
        preds = model(batchx)
        loss = model.loss(preds, batchy)
        
        # required steps to backpropgate the gradients and then update the weights
        model.backward()
        model.step()
        
        # due to the implementation, model.zero_grad() is not required to reset the gradients, 
        # but it is implemented for future extensions on the package
        # model.zero_grad()
        epoch_loss += loss
        
    print(f"Epoch {i+1} Loss: {epoch_loss:.4f}") # display the epoch loss of current epoch
    

# to generata predictions
preds = model(testx)

```

## Acknowledgement
This project is built on top of the source code from this amazing [tutorial](https://towardsdatascience.com/how-to-build-a-diy-deep-learning-framework-in-numpy-59b5b618f9b7).






