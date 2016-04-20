from __future__ import print_function
from copy import copy
import sys

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

class BreakChainException(Exception):
    pass

class Chainable(object):
    '''
    The core of the Chain Framework.
    This class verifies input before calling self._apply

    If the Object only takes a single input it will be redirected directly.
    '''
    inputs = tuple()
    outputs = tuple()

    def __call__(self, data={}):
        if self._isvalid(data):
            data = self._apply(copy(data))
        else:
            raise ValueError(
                    "Inputs required are: {}\n Data provides only: {}".format(
                        str(self.inputs),
                        str(data)
                        ))
        return data

    def _isvalid(self, data):
        valid = False
        try:
            inputs = data.keys()
        except AttributeError:
            #pass
            input_count = len(self.inputs)
            valid = input_count == 1# and isinstance(self, DynamicInput)
        else:
            valid = set(self.inputs).issubset(inputs)
        finally:
            return valid

class Link(Chainable):

    def _apply(self, input_dict):
        inputs = self._prepare_input(input_dict)
        output = self.calculate(*inputs)
        # output_dict = copy(input_dict)
        output = self._prepare_output(output)
        input_dict.update(output)
        return input_dict

    def _prepare_input(self, input_dict):
        output = []
        for i in self.inputs:
            try:
                output.append(input_dict.get(i))
            except AttributeError:
                raise ValueError("Link requires a dictionary-like object as input.")
        return output

    def _prepare_output(self, output):
        '''
        Possible inputs:
        - None : Link doesn't output
        - One element : Link has one output
        - Tuple
        '''
        if len(self.outputs) == 0:
            return {}
        if len(self.outputs) == 1:
            return { self.outputs[0]: output }
        if len(output) == len(self.outputs):
            return dict(zip(self.outputs, output))
        else:
            raise ValueError(
                    "{} is expected to return {} but actual value was {}".format(
                        self.__class__.__name__, (self.outputs), str(output)))


class Chain(Chainable):

    def __init__(self, *args, **kwargs):
        try:
            self.breakable = kwargs.pop('breakable')
        except KeyError:
            self.breakable = False
        self._links = list()
        self.inputs = set()
        self.outputs = set()
        for arg in args:
            self._links.append(arg)
            self.inputs.update(set(arg.inputs).difference(self.outputs))
            self.outputs.update(arg.outputs)
        self.outputs.update(self.inputs)
        super(Chain, self).__init__()

    def _apply(self, input_dict):
        for link in self._links:
            try:
                input_dict = link(input_dict)
            except BreakChainException as e:
                if self.breakable:
                    input_dict = self.cleanup(input_dict)
                    break
                else:
                    raise e
        return input_dict

    def cleanup(self, input_dict):
        output_dict = dict.fromkeys(self.outputs)
        output_dict.update(input_dict)
        return output_dict


class DynamicInput(Chainable):
    '''
    Transforms the input into the dictionary {name: input }
    '''

    def __init__(self, name):
        self.outputs = (name,)
        self.inputs = (name,)
        self._name = name

    def _apply(self, array):
        return {self._name: array}


class DynamicOutput(Chainable):
    '''
    This will return the value of `name`.
    '''

    def __init__(self, name):
        self.inputs = (name,)
        self._name = name

    def _apply(self, input_dict):
        return input_dict[self._name]
