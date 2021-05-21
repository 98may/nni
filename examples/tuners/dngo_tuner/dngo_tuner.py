import random
import torch
import numpy as np

from nni.tuner import Tuner
import nni.parameter_expressions as parameter_expressions
from torch.distributions import Normal
from pybnn import DNGO

def random_archi_generator(nas_ss, random_state):
    '''random
    '''
    chosen_arch = {}
    for key, val in nas_ss.items():
        if val['_type'] == 'choice':
            choices = val['_value']
            index = random_state.randint(len(choices))
            # check values type
            if type(choices[0]) == int or type(choices[0]) == float:
                chosen_arch[key] = choices[index]
            else:
                chosen_arch[key] = index
        elif val['_type'] == 'uniform':
            chosen_arch[key] = random.uniform(val['_value'][0], val['_value'][1])
        elif val['_type'] == 'randint':
            chosen_arch[key] = random_state.randint(
                    val['_value'][0], val['_value'][1])
        elif val['_type'] == 'quniform':
            chosen_arch[key] = parameter_expressions.quniform(
                val['_value'][0], val['_value'][1], val['_value'][2], random_state)
        elif val['_type'] == 'loguniform':
            chosen_arch[key] = parameter_expressions.loguniform(
                val['_value'][0], val['_value'][1], random_state)
        elif val['_type'] == 'qloguniform':
            chosen_arch[key] = parameter_expressions.qloguniform(
                val['_value'][0], val['_value'][1], val['_value'][2], random_state)

        else:
            raise ValueError('Unknown key %s and value %s' % (key, val))
    return chosen_arch

class DngoTuner(Tuner):

    def __init__(self):

        self.searchspace_json = None
        self.random_state = None
        self.model = DNGO(do_mcmc=False)
        self.first_flag = True
        self.x = []
        self.y = []


    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        '''
        Receive trial's final result.
        parameter_id: int
        parameters: object created by 'generate_parameters()'
        value: final metrics of the trial, including default metric
        '''
        # update DNGO model
        self.y.append(value)

    def generate_parameters(self, parameter_id, **kwargs):
        '''
        Returns a set of trial (hyper-)parameters, as a serializable object
        parameter_id: int
        '''
        if self.first_flag:
            self.first_flag = False
            first_x = random_archi_generator(self.searchspace_json, self.random_state)
            self.x.append(list(first_x.values()))
            return first_x
        
        self.model.train(np.array(self.x), np.array(self.y), do_optimize=True)
        # random samples
        candidate_x = []
        for _ in range(1000):
            a = random_archi_generator(self.searchspace_json, self.random_state)
            candidate_x.append(a)
        
        x_test = np.array([np.array(list(xi.values())) for xi in candidate_x])
        m, v = self.model.predict(x_test)
        mean = torch.Tensor(m)
        sigma = torch.Tensor(v)
        # u = (mean - torch.Tensor([args.objective]).expand_as(mean)) / sigma
        u = (mean - torch.Tensor([0.95]).expand_as(mean)) / sigma
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        
        indices = torch.argsort(ei)
        rev_indices = reversed(indices)
        ind = rev_indices[0].item()
        new_x = candidate_x[ind]
        self.x.append(list(new_x.values()))

        return new_x
    

    def update_search_space(self, search_space):
        '''
        Tuners are advised to support updating search space at run-time.
        If a tuner can only set search space once before generating first hyper-parameters,
        it should explicitly document this behaviour.
        search_space: JSON object created by experiment owner
        '''
        # your code implements here.
        self.searchspace_json = search_space
        self.random_state = np.random.RandomState()

        

