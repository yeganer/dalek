from dalek.tools.base import Link


class Posterior(Link):
    inputs = ('logprior', 'loglikelihood',)
    outputs = ('posterior',)

    def calculate(self, logprior, loglikelihood):
        try:
            return logprior + loglikelihood.value
        except (TypeError, AttributeError) as e:
            if loglikelihood is None:
                return logprior
            else:
                raise e
