# Examples on LeNet300
We demonstrate the flexibility of our framework by easily exploring multiple compression schemes with minimal effort. As an example, say we are tasked with compressing the storage bits of the LeNet300 neural network trained on MNIST dataset. The LeNet300 is a three-layer neural network with 300, 100, and 10 neurons respectively on every layer and has an error of 2.13% on the test set.
The full code is in [lenet300.py](lenet300.py). You can run all examples using [provided script](run.sh):
```bash
chmod +x run.sh && ./run.sh
```
In order to run the LC algorithm, we need to provide an L step implementation and compression tasks to an instance of the `LCAlgorithm`:
```python
lc_alg = lc.Algorithm(
    model=net,                            # model to compress
    compression_tasks=compression_tasks,  # specifications of compression
    l_step_optimization=my_l_step,        # implementation of the L-step
    mu_schedule=mu_s,                     # schedule of mu values
    evaluation_func=train_test_acc_eval_f # evaluation function
)
lc_alg.run()     

```  
## Setting up the compression
We implement the L step below:
```python
def my_l_step(model, lc_penalty, step):
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    lr = lr_base*(0.98**step) # we use a fixed decayed learning rate for each L step
    optimizer = optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    for epoch in range(epochs_per_step):
        for x, target in train_loader: # loop over the dataset
            optimizer.zero_grad()
            loss = model.loss(model(x), target) + lc_penalty() # loss + LC penalty
            loss.backward()
            optimizer.step()
```
Notice that it is not different from the implementation of regular training of the neural network, the only difference is for L step, we need to optimize the loss + lc_penalty. 

Now, having the L step implementation, we can formulate the compression tasks. Say, we would like to know what would be the test error if the model is optimally quantized with a separate codebook on each layer:
```python
import lc
from lc.torch import ParameterTorch as Param, AsVector
from lc.compression_types import AdaptiveQuantization

compression_tasks = {
    Param(l1.weight): (AsVector, AdaptiveQuantization(k=2)),
    Param(l2.weight): (AsVector, AdaptiveQuantization(k=2)),
    Param(l3.weight): (AsVector, AdaptiveQuantization(k=2))
}          
```
That's it, the result of the compression is a network with quantized values, and the test error of 2.56%. 
What would be the performance of the model if we prune the 1st layer, apply low-rank to 2nd, and quantize the third layer? We can do it by a simple modification of the compression tasks structure:
```python
compression_tasks = {
    Param(l1.weight): (AsVector, ConstraintL0Pruning(kappa=5000)),
    Param(l2.weight): (AsIs,     LowRank(target_rank=10))
    Param(l3.weight): (AsVector, AdaptiveQuantization(k=2))
}
```
In such case, the test error is 2.51\%.

Our framework allows to apply a single compression to multiple layer, e.g., prune 5\% of all weights:
```python
compression_tasks = {
    Param([l1.weight, l2.weight, l3.weights]): 
        (AsVector, ConstraintL0Pruning(kappa=13310)) # 13310 = 5%
}
```
or apply multiple compressions additively:
```python
compression_tasks = {
    Param([l1.weight, l2.weight, l3.weights]): [
        (AsVector, ConstraintL0Pruning(kappa=2662)), # 2662 = 1%
        (AsVector, AdaptiveQuantization(k=2))
    ]
}
```
## More examples
The only requirement to compress using different scheme is to provide a new compression tasks. The table below shows some possible combinations.
<table>
<tr>
<th>Semantics</th></th><th>Compression Structure</th><th>Test Error</th>
</tr>

<tr><td>reference model</td> <td>n/a</td> <td>2.13%</td></tr>
<tr><td>quantize all layers</td> <td>
<pre lang="python">
compression_tasks = {
    Param(l1.weight): (AsVector, AdaptiveQuantization(k=2)),
    Param(l2.weight): (AsVector, AdaptiveQuantization(k=2)),
    Param(l3.weight): (AsVector, AdaptiveQuantization(k=2))
}          
</pre>
</td> <td>2.56%</td></tr>
<tr><td>quantize first and third layers</td> <td>
<pre lang="python">
compression_tasks = {
    Param(l1.weight): (AsVector, AdaptiveQuantization(k=2)),
    Param(l3.weight): (AsVector, AdaptiveQuantization(k=2))
}         
</pre>
</td> <td>2.26%</td></tr>
<tr><td>prune all but 5\%</td> <td>
<pre lang="python">
compression_tasks = {
    Param([l1.weight, l2.weight, l3.weights]): 
        (AsVector, ConstraintL0Pruning(kappa=13310)) # 13310 = 5%
}         
</pre>
</td> <td>2.18%</td></tr>
<tr><td>single codebook quantization with 1% correction (pruning)</td> <td>
<pre lang="python">
compression_tasks = {
    Param([l1.weight, l2.weight, l3.weights]): [
        (AsVector, ConstraintL0Pruning(kappa=2662)), # 2662 = 1%
        (AsVector, AdaptiveQuantization(k=2))
    ]
}     
</pre>
</td> <td>2.17%</td></tr>
<tr><td>prune first layer, low-rank to second, quantize third</td> <td>
<pre lang="python">
compression_tasks = {
    Param(l1.weight): (AsVector, ConstraintL0Pruning(kappa=5000)),
    Param(l2.weight): (AsIs,     LowRank(target_rank=10))
    Param(l3.weight): (AsVector, AdaptiveQuantization(k=2))
}         
</pre>
</td> <td>2.13%</td></tr>
<tr><td>rank selection with alpha=1e-6</td> <td>
<pre lang="python">
compression_tasks = {
    Param(l1.weight): (AsIs,     RankSelection(alpha=1e-6))
    Param(l2.weight): (AsIs,     RankSelection(alpha=1e-6))
    Param(l3.weight): (AsIs,     RankSelection(alpha=1e-6))
}      
</pre>
</td> <td>1.90%</td></tr>
</table>