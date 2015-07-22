"""
This script is a starting point for new Blocks users already familiar with
 Machine Learning and Theano.

We demonstrate how to use blocks to train a generic set of parameters (theano
 shared variables) that influence some arbitrary cost function (theano
 symbolic variable), so you can start using blocks features (e.g. monitoring,
 extensions, training algorithms) with your Theano code today.

To run an experiment, we simply construct a main_loop.MainLoop and call its
 run() method.  It suffices to pass the MainLoop a blocks.model.Model
 (which needs the cost), a blocks.algorithms.TrainingAlgorithm (which needs the
 cost and parameters), and a fuel.streams.DataStream*

As it is the script will run indefinitely, with no output.  You can interrupt
 training training anytime with Ctrl+C, or termination conditions can be added
 via extensions.

*The DataStream object is part of the partner library Fuel
 (https://github.com/mila-udem/fuel).
"""

import numpy
np = numpy
import theano
import theano.tensor as T

# (Here we make a toy dataset of two 2D gaussians with different means.)
num_examples = 1000
batch_size = 100
means = np.array([[-1., -1.], [1, 1]])
std = 0.5
labels = np.random.randint(size=num_examples, low=0, high=1)
features = means[labels, :] + std * np.random.normal(size=(num_examples, 2))
labels = labels.reshape((num_examples, 1)).astype(theano.config.floatX)
features = features.astype(theano.config.floatX)

# Define "data_stream"
from collections import OrderedDict
from fuel.datasets import IndexableDataset
# The names here (e.g. 'name1') need to match the names of the variables which
#  are the leaves of the computational graph for the cost.
dataset = IndexableDataset(
              OrderedDict([('name1', features), ('name2', labels)]))
from fuel.streams import DataStream, ForceFloatX
from fuel.schemes import SequentialScheme
data_stream = ForceFloatX(DataStream(dataset,
                                     iteration_scheme=SequentialScheme(
                                       dataset.num_examples, batch_size)))


# Define "cost" and "parameters"
# (We use logistic regression to classify points by distribution)
inputs = T.matrix('name1')
targets = T.matrix('name2')
ninp, nout = 2, 1
W = theano.shared(.01*np.random.uniform(
                        size=((ninp, nout))).astype(theano.config.floatX))
b = theano.shared(np.zeros(nout).astype(theano.config.floatX))
output = T.nnet.sigmoid(T.dot(inputs, W) + b)
# a theano symbolic expression
cost = T.mean(T.nnet.binary_crossentropy(output, targets))
# a list of theano.shared variables
parameters = [W, b]


# wrap everything in Blocks objects and run!
from blocks.model import Model
model = Model([cost])
from blocks.algorithms import GradientDescent, Scale
algorithm = GradientDescent(cost=cost,
                            parameters=parameters,
                            step_rule=Scale(learning_rate=.01))
from blocks.main_loop import MainLoop
my_loop = MainLoop(model=model,
                   data_stream=data_stream,
                   algorithm=algorithm)
my_loop.run()
