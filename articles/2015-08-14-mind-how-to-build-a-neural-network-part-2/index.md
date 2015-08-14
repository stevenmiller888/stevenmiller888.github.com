
# Mind: How to Build a Neural Network (Part Two)

*In this second part on learning how to build a neural network, we will dive into the implementation of a flexible library in JavaScript. In case you missed it, here is [Part One](/mind-how-to-build-a-neural-network), which goes over what neural networks are and how they operate.*

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

> Note that `this.activate` and `this.weights` are set at the initialization of a new `Mind` via [passing an `opts` object](https://github.com/stevenmiller888/mind/blob/master/lib/index.js#L40). `multiply` and `transform` come from an npm [module](https://www.npmjs.com/package/node-matrix) for performing basic matrix operations.

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

> Note that `subtract`, `dot` , `scalar`, `multiply`, and `add` come from the same npm [module](https://www.npmjs.com/package/node-matrix) we used before for performing matrix operations.

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
