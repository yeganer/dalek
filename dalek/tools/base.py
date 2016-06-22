from __future__ import print_function
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

    def __call__(self, *args, **kwargs):
        return self.full(*args, **kwargs)[1]

    def full(self, *args, **kwargs):
        return self._apply(*self._prepare_args(args, kwargs))

    def _prepare_args(self, args, kwargs):
        '''
        Returns:
        (Tuple) (args, kwargs) where
        args: a list of all the inputs required for the current Chainable
        kwargs: input data passed as kwargs or args[0]
        '''
        try:
            for arg in args:
                # Assume data is passed as keywords
                kwargs.update(arg)
        except TypeError:
            # some arg not iterable
            if len(self.inputs) == len(args) and arg is args[0]:
                kwargs.update(
                        {k:v for k,v in zip(self.inputs, args)})
            else:
                raise TypeError(
                        "expected {} arguments, got {} \n{}\n{}".format(
                            len(self.inputs),
                            len(args),
                            self.inputs,
                            args))

        try:
            args = [kwargs[k] for k in self.inputs]
        except KeyError:
                raise KeyError(
                        "expected {} as input, got {}".format(
                            str(self.inputs),
                            str(kwargs.keys())
                            ))

        return args, kwargs

    def _apply(self, args, kwargs):
        raise NotImplementedError('This has to be implemented by the subclass.')


class Link(Chainable):

    def _apply(self, args, kwargs):
        return self._prepare_output(self.calculate(*args), **kwargs)

    def _prepare_output(self, args, **kwargs):
        '''
        Possible inputs:
        - None : Link doesn't output
        - One element : Link has one output
        - Tuple
        '''


        if len(self.outputs) == 1:
            kwargs[self.outputs[0]] = args
        elif len(self.outputs) > 1 and len(args) == len(self.outputs):
            kwargs.update(dict(zip(self.outputs, args)))
        #if self.outpus
        #else:
        #    raise ValueError(
        #            "{} is expected to return {} but actual value was {}".format(
        #                self.__class__.__name__, (self.outputs), str(output)))
        return args, kwargs


class Chain(Link):

    def __init__(self, *args, **kwargs):
        self.breakable = kwargs.pop('breakable', False)
        self._links = args
        self.inputs = set()
        self.outputs = set()
        for arg in args:
            self.inputs.update(set(arg.inputs).difference(self.outputs))
            self.outputs.update(arg.outputs)
        self.outputs.update(self.inputs)
        super(Chain, self).__init__()

    def _apply(self, args, kwargs):
        for link in self._links:
            try:
                args, kwargs = link.full([], **kwargs)
            except BreakChainException as e:
                if self.breakable:
                    kwargs = self.cleanup(**kwargs)
                    break
                else:
                    raise e
        return args, kwargs

    def cleanup(self, **kwargs):
        output_dict = dict.fromkeys(self.outputs)
        output_dict.update(kwargs)
        return output_dict


class DynamicOutput(Chainable):
    '''
    This will return the value of `name`.
    '''

    def __init__(self, name):
        self.inputs = (name,)
        self._name = name

    def _apply(self, args, kwargs):
        return kwargs[self._name], kwargs
