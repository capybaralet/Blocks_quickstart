"""
This is a fuel.streams.DataStream that takes a 2d-numpy.array design_matrix
(and optionally a batch_size) and wraps it for a blocks.main_loop.MainLoop
with a ForceFloatX for good measure.
"""
#from collections import OrderedDict
from fuel.datasets import IndexableDataset
from fuel.streams import DataStream, ForceFloatX
from fuel.schemes import SequentialScheme

def design_matrix_data_stream(design_matrix, batch_size=128):
    dataset = IndexableDataset(design_matrix)
    return ForceFloatX(DataStream(dataset,
                                  iteration_scheme=SequentialScheme(
                                    dataset.num_examples, batch_size)))
