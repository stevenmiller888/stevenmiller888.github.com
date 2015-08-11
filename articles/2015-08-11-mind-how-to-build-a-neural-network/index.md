
# Mind: How to Build a Neural Network

[Artificial neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network) are statistical learning models, inspired by biological neural networks (central nervous systems, such as the brain), that are used in [machine learning](https://en.wikipedia.org/wiki/List_of_machine_learning_concepts). These networks are represented as systems of interconnected "neurons", which send messages to each other. The connections within the network can be systematically adjusted based on inputs and outputs, making them ideal for supervised learning.

Neural networks can be intimidating, especially for people with little experience in machine learning and cognitive science! However, through code, this tutorial will explain how neural networks operate. By the end, you will know how to build your own flexible, learning network, similar to [Mind](https://www.github.com/stevenmiller888/mind).

The only prerequisites are having a basic understanding of JavaScript, high-school Calculus, and simple matrix operations. Other than that, you don't need to know anything. Have fun!

## Understanding the Mind

A neural network is a collection of "neurons" with "synapses" connecting them. The collection is organized into three main parts: the input layer, the hidden layer, and the output layer. Note that you can have _n_ hidden layers, with the term "deep" learning implying multiple hidden layers.

![](https://cldup.com/ytEwlOfrRZ-2000x2000.png)

*Screenshot taken from [this great introductory video](https://www.youtube.com/watch?v=bxe2T-V8XRs), which trains a neural network to predict a test score based on hours spent studying and sleeping the night before.*

Hidden layers are necessary when the neural network has to make sense of something really complicated, contextual, or non obvious, like image recognition. The term "deep" learning came from having many hidden layers. These layers are known as "hidden", since they are not visible as a network output. Read more about hidden layers [here](http://stats.stackexchange.com/questions/63152/what-does-the-hidden-layer-in-a-neural-network-compute) and [here](http://www.cs.cmu.edu/~dst/pubs/byte-hiddenlayer-1989.pdf).

The circles represent neurons and lines represent synapses. Synapses take the input and multiply it by a "weight" (the "strength" of the input in determining the output). Neurons add the outputs from all synapses and apply an activation function.

Training a neural network basically means calibrating all of the "weights" by repeating two key steps, forward propagation and back propagation.

Since neural networks are great for regression, the best input data are numbers (as opposed to discrete values, like colors or movie genres, whose data is better for statistical classification models). The output data will be a number within a range like 0 and 1 (this ultimately depends on the activation function—more on this below).

In **forward propagation**, we apply a set of weights to the input data and calculate an output. For the first forward propagation, the set of weights is selected randomly.

In **back propagation**, we measure the margin of error of the output and adjust the weights accordingly to decrease the error.

Neural networks repeat both forward and back propagation until the weights are calibrated to accurately predict an output.

Next, we'll walk through a simple example of training a neural network to function as an ["Exclusive or" ("XOR") operation](https://en.wikipedia.org/wiki/Exclusive_or) to illustrate each step in the training process.

### Forward Propagation

*Note that all calculations will show figures truncated to the thousandths place.*

The XOR function can be represented by the mapping of the below inputs and outputs, which we'll use as training data. It should provide a correct output given any input acceptable by the XOR function.

```
input | output
--------------
0, 0  | 0
0, 1  | 1
1, 0  | 1
1, 1  | 0
```

Let's use the last row from the above table, `(1, 1) => 0`, to demonstrate forward propagation:

![](http://i.imgur.com/l2ljT1F.png)

*Note that we use a single hidden layer with only three neurons for this example.*

We now assign weights to all of the synapses. Note that these weights are selected randomly (based on Gaussian distribution) since it is the first time we're forward propagating. The initial weights will be between 0 and 1, but note that the final weights don't need to be.

![](http://i.imgur.com/RRn0pgb.png)

We sum the product of the inputs with their corresponding set of weights to arrive at the first values for the hidden layer. You can think of the weights as measures of influence the input nodes have on the output.

```
1 * 0.8 + 1 * 0.2 = 1
1 * 0.4 + 1 * 0.9 = 1.3
1 * 0.3 + 1 * 0.5 = 0.8
```

We put these sums smaller in the circle, because they're not the final value:

![](http://i.imgur.com/tVBcyZz.png)

To get the final value, we apply the [activation function](https://en.wikipedia.org/wiki/Activation_function) to the hidden layer sums. The purpose of the activation function is to transform the input signal into an output signal and are necessary for neural networks to model complex non-linear patterns that simpler models might miss.

There are many types of activation functions—linear, sigmoid, hyperbolic tangent, even step-wise. To be honest, I don't know why one function is better than another.

![](https://cldup.com/hxmGABAI7Y.png)

*Table taken from [this paper](http://www.asprs.org/a/publications/pers/2003journal/november/2003_nov_1225-1234.pdf).*

For our example, let's use the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function)* for activation:

![](http://i.imgur.com/xgQmN9r.png)

The sigmoid function looks like this, graphically:

![](http://i.imgur.com/RVbqJsg.jpg)

And applying S(x) to the three hidden layer _sums_, we get:

```
S(1.0) = 0.26894142137
S(1.3) = 0.78583498304
S(0.8) = 0.31002551887
```

We add that to our neural network as hidden layer _results_:

![](http://i.imgur.com/GjQuLls.png)

Then, we sum the product of the hidden layer results with the second set of weights (also determined at random the first time around) to determine the output sum.

```
0.27 * 0.3 + 0.79 * 0.5 + 0.31 * 0.9 = 0.755
```

..finally we apply the activation function to get the final output result.

```
S(0.755) = 0.6802671966986485
```

This is our full diagram:

![](http://i.imgur.com/GIWpAJ6.png)

Because we used a random set of initial weights, the value of the output neuron is off the mark; in this case by +0.68 (since the target is 0). If we stopped here, this set of weights would be a great neural network for inaccurately representing the XOR operation.

Let's fix that by using back propagation to adjust the weights to improve the network!

### Back Propagation

To improve our model, we first have to quantify just how wrong our predictions are. Then, we adjust the weights accordingly so that the margin of errors are decreased.

Similar to forward propagation, back propagation calculations occur at each "layer". We begin by changing the weights between the hidden layer and the output layer.

![](http://i.imgur.com/pYhOMXJ.png)

Calculating the incremental change to these weights happens in two steps: 1) we find the margin of error of the output result (what we get after applying the activation function) to back out the necessary change in the output sum (we call this `delta output sum`) and 2) we extract the change in weights by multiplying `delta output sum` by the hidden layer results.

The `output sum margin of error` is the target output result minus the calculated output result:

![](http://i.imgur.com/IAddjWL.png)

And doing the math:

```
Target = 0
Calculated = 0.68
Target - calculated = -0.68
```

To calculate the necessary change in the output sum, or `delta output sum`, we take the derivative of the activation function and apply it to the output sum. In our example, the activation function is the sigmoid function.

To refresh your memory, the activation function, sigmoid, takes the sum and returns the result:

![](http://i.imgur.com/rKHEE51.png)

So the derivative of sigmoid, also known as sigmoid prime, will give us the rate of change (or "slope") of the activation function at the output sum:

![](http://i.imgur.com/8xQ6TiU.png)

Since the `output sum margin of error` is the difference in the result, we can simply multiply that with the rate of change to give us the `delta output sum`:

![](http://i.imgur.com/4qnVb6S.png)

Conceptually, this means that the change in the output sum is the same as the sigmoid prime of the output result. Doing the actual math, we get:

```
Delta output sum = S'(sum) * (output sum margin of error)
Delta output sum = S'(0.755) * (-0.68)
Delta output sum = -0.1479
```

Here is a graph of the Sigmoid function to give you an idea of how we are using the derivative to move the input towards the right direction. Note that this graph is not to scale.

![](http://i.imgur.com/ByyQIJ8.png)

Now that we have the proposed change in the output layer sum (-0.14), let's use this in the derivative of the output sum function to determine the new change in weights.

As a reminder, the mathematical definition of the `output sum` is the product of the hidden layer result and the weights between the hidden and output layer:

![](http://i.imgur.com/ITudruR.png)

The derivative of the `output sum` is:

![](http://i.imgur.com/57mJyOe.png)

..which can also be represented as:

![](http://i.imgur.com/TR7FS2S.png)

This relationship suggests that a greater change in output sum yields a greater change in the weights; input neurons with the biggest contribution (higher weight to output neuron) should experience more change in the connecting synapse.

Let's do the math:

```
hidden result 1 = 0.2689
hidden result 2 = 0.7858
hidden result 3 = 0.3100

Delta weights = delta output sum / hidden layer results
Delta weights = -0.1479 / [0.2689, 0.7858, 0.3100]
Delta weights = [-0.5500, -0.1882, -0.4771]

old w7 = 0.3
old w8 = 0.5
old w9 = 0.9

new w7 = -0.25
new w8 = 0.3118
new w9 = 0.4229
```

To determine the change in the weights between the _input and hidden_ layers, we perform the similar, but notably different, set of calculations. Note that in the following calculations, we use the initial weights instead of the recently adjusted weights from the first part of the backward propagation.

Remember that the relationship between the hidden result, the weights between the hidden and output layer, and the output sum is:

![](http://i.imgur.com/ITudruR.png)

Instead of deriving for `output sum`, let's derive for `hidden result` as a function of `output sum` to ultimately find out `delta hidden sum`:

![](http://i.imgur.com/25TS8NU.png)
![](http://i.imgur.com/iQIR1MD.png)

Also, remember that the change in the `hidden result` can also be defined as:

![](http://i.imgur.com/ZquX1pv.png)

Let's multiply both sides by sigmoid prime of the hidden sum:

![](http://i.imgur.com/X0wvirh.png)
![](http://i.imgur.com/msHbhQl.png)

All of the pieces in the above equation can be calculated, so we can determine the `delta hidden sum`:

```
Delta hidden sum = delta output sum / hidden-to-outer weights * S'(hidden sum)
Delta hidden sum = -0.1479 / [0.3, 0.5, 0.9] * S'([1, 1.3, 0.8])
Delta hidden sum = [-0.4930, -0.2958, -0.1643] * [0.1966, 0.1683, 0.2139]
Delta hidden sum = [-0.0969, -0.0498, -0.0351]
```

Once we get the `delta hidden sum`, we calculate the change in weights between the input and hidden layer by dividing it with the input data, `(1, 1)`. The input data here is equivalent to the `hidden results` in the earlier back propagation process to determine the change in the hidden-to-output weights. Here is the derivation of that relationship, similar to the one before:

![](http://i.imgur.com/7NmXWSh.png)
![](http://i.imgur.com/1SDxECJ.png)
![](http://i.imgur.com/KYuSAgw.png)

Let's do the math:

```
input 1 = 1
input 2 = 1

Delta weights = delta hidden sum / input data
Delta weights = [-0.0969, -0.0498, -0.0351] / [1, 1]
Delta weights = [-0.0969, -0.0498, -0.0351, -0.0969, -0.0498, -0.0351]

old w1 = 0.8
old w2 = 0.4
old w3 = 0.3
old w4 = 0.2
old w5 = 0.9
old w6 = 0.5

new w1 = 0.7031
new w2 = 0.3502
new w3 = 0.2649
new w4 = 0.1031
new w5 = 0.8502
new w6 = 0.4649
```

Here are the new weights, right next to the initial random starting weights as comparison:

```
old         new
-----------------
w1: 0.8     w1: 0.7031
w2: 0.4     w2: 0.3502
w3: 0.3     w3: 0.2649
w4: 0.2     w4: 0.1031
w5: 0.9     w5: 0.8502
w6: 0.5     w6: 0.4649
w7: 0.3     w7: -0.25
w8: 0.5     w8: 0.3118
w9: 0.9     w9: 0.4229
```

Once we arrive at the adjusted weights, we start again with forward propagation. When training a neural network, it is common to repeat both these processes thousands of times (by default, Mind iterates 10,000 times).

And doing a quick forward propagation, we can see that the final output here is a little closer to the expected output:

![](http://i.imgur.com/76yrfwb.png)

Through just one iteration of forward and back propagation, we've already improved the network!!

*Check out [this short video](https://www.youtube.com/watch?v=GlcnxUlrtek) for a great explanation of identifying global minima in a cost function as a way to determine necessary weight changes.*

## Building the Mind

Building a complete neural network library requires more than just understanding forward and back propagation. We also need to think about how a user of the network will want to configure it (e.g. set total number of learning iterations) and other API-level design considerations.

To simplify our explanation of neural networks via code, the code snippets below build a neural network, `Mind`, with a single hidden layer. The actual [Mind](https://github.com/stevenmiller888/mind) library, however, provides the flexibility to build a network with multiple hidden layers.

### Initialization

First, we need to set up our constructor function. Let's give the option to use the sigmoid activation or the hyperbolic tangent activation function. Additionally, we'll allow our users to set the learning rate, number of iterations, and number of units in the hidden layer, while providing sane defaults for each. Here's our constructor:

```javascript
function Mind(opts) {
  if (!(this instanceof Mind)) return new Mind(opts);
  opts = opts || {};

  opts.activator === 'sigmoid'
    ? (this.activate = sigmoid, this.activatePrime = sigmoidPrime)
    : (this.activate = htan, this.activatePrime = htanPrime);

  // hyperparameters
  this.learningRate = opts.learningRate || 0.7;
  this.iterations = opts.iterations || 10000;
  this.hiddenUnits = opts.hiddenUnits || 3;
}
```

> Note that here we use the [`sigmoid`](https://www.npmjs.com/package/sigmoid), [`sigmoid-prime`](https://www.npmjs.com/package/sigmoid-prime), [`htan`](https://www.npmjs.com/package/htan), and [`htan-prime`](https://www.npmjs.com/package/htan-prime) npm modules.

### Forward Propagation

The forward propagation process is a series of sum products and transformations. Let's calculate the first hidden sum with all four input data:

![](http://i.imgur.com/ZhO0Nj2.png)

This can be represented as such:

![](http://i.imgur.com/XcSZgTk.png)

To get the result from the sum, we apply the activation function, sigmoid, to each element:

![](http://i.imgur.com/rhnNQZW.png)

Then, we do this again with the hidden result as the new input to get to the final output result. The entire forward propagation code looks like:

```javascript
Mind.prototype.forward = function(examples) {
  var activate = this.activate;
  var weights = this.weights;
  var ret = {};

  ret.hiddenSum = multiply(weights.inputHidden, examples.input);
  ret.hiddenResult = ret.hiddenSum.transform(activate);
  ret.outputSum = multiply(weights.hiddenOutput, ret.hiddenResult);
  ret.outputResult = ret.outputSum.transform(activate);

  return ret;
};
```

Note that `this.activate` and `this.weights` are set at the initialization of a new `Mind` via [passing an `opts` object](https://github.com/stevenmiller888/mind/blob/master/lib/index.js#L40).

### Back Propagation

Back propagation is a bit more complicated. Let's look at the last layer first. We calculate the `output error` (same equation as before):

![](http://i.imgur.com/IAddjWL.png)

And the equivalent in code:

```javascript
var errorOutputLayer = subtract(examples.output, results.outputResult);
```

Then, we determine the change in the output layer sum, or `delta output sum`:

![](http://i.imgur.com/4qnVb6S.png)

And the code:

```javascript
var deltaOutputLayer = dot(results.outputSum.transform(activatePrime), errorOutputLayer);
```

Then, we figure out the hidden output changes. We use this formula:

![](http://i.imgur.com/TR7FS2S.png)

Here is the code:

```javascript
var hiddenOutputChanges = scalar(multiply(deltaOutputLayer, results.hiddenResult.transpose()), learningRate);
```

Note that we scale the change by a magnitude, `learningRate`, which is from 0 to 1. The learning rate applies a greater or lesser portion of the respective adjustment to the old weight. If there is a large variability in the input (there is little relationship among the training data) and the rate was set high, then the network may not learn well or at all. Setting the rate too high also introduces the risk of ['overfitting'](https://en.wikipedia.org/wiki/Overfitting), or training the network to generate a relationship from noise instead of the actual underlying function.

Since we're dealing with matrices, we handle the division by multiplying the `delta output sum` with the hidden results matrices' transpose.

Then, we do this process [again](https://github.com/stevenmiller888/mind/blob/master/lib/index.js#L200) for the input to hidden layer.

The code for the back propagation function is below. Note that we're passing what is returned by the `forward` function as the second argument:

```javascript
Mind.prototype.back = function(examples, results) {
  var activatePrime = this.activatePrime;
  var learningRate = this.learningRate;
  var weights = this.weights;

  // compute weight adjustments
  var errorOutputLayer = subtract(examples.output, results.outputResult);
  var deltaOutputLayer = dot(results.outputSum.transform(activatePrime), errorOutputLayer);
  var hiddenOutputChanges = scalar(multiply(deltaOutputLayer, results.hiddenResult.transpose()), learningRate);
  var deltaHiddenLayer = dot(multiply(weights.hiddenOutput.transpose(), deltaOutputLayer), results.hiddenSum.transform(activatePrime));
  var inputHiddenChanges = scalar(multiply(deltaHiddenLayer, examples.input.transpose()), learningRate);

  // adjust weights
  weights.inputHidden = add(weights.inputHidden, inputHiddenChanges);
  weights.hiddenOutput = add(weights.hiddenOutput, hiddenOutputChanges);

  return errorOutputLayer;
};
```

### Putting both together

Now that we have both the forward and back propagation, we can define the function `learn` that will put them together. The `learn` function will accept training data (`examples`) as an array of matrices. Then, we assign random samples to the initial weights (via [`sample`](https://github.com/stevenmiller888/sample)). Lastly, we use a `for` loop and repeat `this.iterations` to do both forward and backward propagation.

```javascript
Mind.prototype.learn = function(examples) {
  examples = normalize(examples);

  this.weights = {
    inputHidden: Matrix({
      columns: this.hiddenUnits,
      rows: examples.input[0].length,
      values: sample
    }),
    hiddenOutput: Matrix({
      columns: examples.output[0].length,
      rows: this.hiddenUnits,
      values: sample
    })
  };

  for (var i = 0; i < this.iterations; i++) {
    var results = this.forward(examples);
    var errors = this.back(examples, results);
  }

  return this;
};
```

*More information about the Mind API [here](https://github.com/stevenmiller888/mind).*

Now you have a basic understanding of how neural networks operate, how to train them, and also how to build your own!

If you have any questions or comments, don't hesitate to find me on [twitter](https://www.twitter.com/stevenmiller888). Shout out to [Andy](https://www.twitter.com/andyjiang) for his help on reviewing this.

## Additional Resources

[Neural Networks Demystified](https://www.youtube.com/watch?v=bxe2T-V8XRs), by [Stephen Welch](https://www.twitter.com/stephencwelch)

[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap3.html), by [Michael Nielsen](http://michaelnielsen.org/)

[The Nature of Code, Neural Networks](http://natureofcode.com/book/chapter-10-neural-networks/), by [Daniel Shiffman](https://twitter.com/shiffman)

[Artificial Neural Networks](https://en.wikipedia.org/wiki/Artificial_neural_network), Wikipedia

[Basic Concepts for Neural Networks](http://www.cheshireeng.com/Neuralyst/nnbg.htm), by Ross Berteig

[Artificial Neural Networks](http://www.saedsayad.com/artificial_neural_network.htm), by [Saed Sayad](http://www.saedsayad.com/author.htm)

[How to Decide the Number of Hidden Layers and Nodes in a Hidden Layer](http://www.researchgate.net/post/How_to_decide_the_number_of_hidden_layers_and_nodes_in_a_hidden_layer)

[How to Decide size of Neural Network like number of neurons in a hidden layer & Number of hidden layers?](http://in.mathworks.com/matlabcentral/answers/72654-how-to-decide-size-of-neural-network-like-number-of-neurons-in-a-hidden-layer-number-of-hidden-lay)
