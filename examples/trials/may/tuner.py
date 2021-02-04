from nni.tuner import Tuner

import torch
import argparse
from collections import defaultdict

from dngo_darts import load_arch2vec
from dngo_darts import get_init_samples





class MayTuner(Tuner):
    
    def __init__(self):

        # parser = argparse.ArgumentParser(description="arch2vec-DNGO")
        # parser.add_argument("--seed", type=int, default=3, help="random seed")
        # parser.add_argument('--cfg', type=int, default=4, help='configuration (default: 4)')
        # parser.add_argument('--dim', type=int, default=16, help='feature dimension')
        # parser.add_argument('--objective', type=float, default=0.95, help='ei objective')
        # parser.add_argument('--init_size', type=int, default=16, help='init samples')
        # parser.add_argument('--batch_size', type=int, default=5, help='acquisition samples')
        # parser.add_argument('--inner_epochs', type=int, default=50, help='inner loop epochs')
        # parser.add_argument('--train_portion', type=float, default=0.9, help='inner loop train/val split')
        # parser.add_argument('--max_budgets', type=int, default=100, help='max number of trials')
        # parser.add_argument('--output_path', type=str, default='saved_logs/bo', help='bo')
        # parser.add_argument('--logging_path', type=str, default='', help='search logging path')
        # args = parser.parse_args()
        # torch.manual_seed(args.seed)

        self.proposed_geno = []
        self.i_geno = 1

        self.proposed_val_acc = []
        self.proposed_test_acc = []

        self.CURR_BEST_VALID = 0.
        self.CURR_BEST_TEST = 0.
        self.CURR_BEST_GENOTYPE = None
        # self.MAX_BUDGET = args.max_budgets
        self.MAX_BUDGET = 100
        self.window_size = 200
        self.counter = 0
        self.visited = {}
        self.best_trace = defaultdict(list)

        self.embedding_path = "/home/v-ayanmao/msra/nni/examples/trials/may/arch2vec-darts.pt"
        self.features, self.genotype = load_arch2vec(self.embedding_path)
        self.features, self.genotype = self.features.cpu().detach(), self.genotype
        feat_samples, geno_samples, valid_label_samples, test_label_samples, visited = get_init_samples(self.features, self.genotype, self.visited)

        for feat, geno, acc_valid, acc_test in zip(feat_samples, geno_samples, valid_label_samples, test_label_samples):
            counter += 1
            if acc_valid > CURR_BEST_VALID:
                CURR_BEST_VALID = acc_valid
                CURR_BEST_TEST = acc_test
                CURR_BEST_GENOTYPE = geno
            best_trace['validation_acc'].append(float(CURR_BEST_VALID))
            best_trace['test_acc'].append(float(CURR_BEST_TEST))
            best_trace['genotype'].append(CURR_BEST_GENOTYPE)
            best_trace['counter'].append(counter)
            # np.random.seed(args.seed)
            # init_inds = np.random.permutation(list(range(features.shape[0])))[
            #     :args.init_size]
            # init_inds = torch.Tensor(init_inds).long()
            # print('init index: {}'.format(init_inds))
            # init_feat_samples = features[init_inds]
            # self.init_geno_samples = [self.genotype[i.item()] for i in init_inds]
            # self.init_valid_label_samples = []
            # self.init_test_label_samples = []

            # self.proposed_geno = []


    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        '''
        Receive trial's final result.
        parameter_id: int
        parameters: object created by 'generate_parameters()'
        value: final metrics of the trial, including default metric
        '''
        # your code implements here.

        if self.i_geno == 1:
            self.proposed_val_acc = []
            self.proposed_test_acc = []
        
        val_acc = value[0]
        test_acc = value[1]
        proposed_val_acc.append(val_acc)
        proposed_test_acc.append(test_acc)

        label_next_valid = torch.Tensor(proposed_val_acc)
        label_next_test = torch.Tensor(proposed_test_acc)


        # add proposed networks to the pool
        for feat, geno, acc_valid, acc_test in zip(feat_next, geno_next, label_next_valid, label_next_test):
            feat_samples = torch.cat((feat_samples, feat.view(1, -1)), dim=0)
            geno_samples.append(geno)
            valid_label_samples = torch.cat((valid_label_samples.view(-1, 1), acc_valid.view(1, 1)), dim=0)
            test_label_samples = torch.cat((test_label_samples.view(-1, 1), acc_test.view(1, 1)), dim=0)
            counter += 1
            if acc_valid.item() > CURR_BEST_VALID:
                CURR_BEST_VALID = acc_valid.item()
                CURR_BEST_TEST = acc_test.item()
                CURR_BEST_GENOTYPE = geno

            best_trace['validation_acc'].append(float(CURR_BEST_VALID))
            best_trace['test_acc'].append(float(CURR_BEST_TEST))
            best_trace['genotype'].append(CURR_BEST_GENOTYPE)
            best_trace['counter'].append(counter)
    

    def generate_parameters(self, parameter_id, **kwargs):
        '''
        Returns a set of trial (hyper-)parameters, as a serializable object
        parameter_id: int
        '''
        self.counter += 1
        if self.i_geno != 0 and self.i_geno < len(self.proposed_geno):
            next_geno = self.proposed_geno[self.i_geno]
            # print("@next_geno==self.proposed_geno[self.i_geno] : ", next_geno)
            self.i_geno += 1
            if self.i_geno == len(self.proposed_geno):
                self.i_geno = 1
            # return next_geno
        else:
            print("feat_samples:", feat_samples.shape)
            print("length of genotypes:", len(geno_samples))
            print("valid label_samples:", valid_label_samples.shape)
            print("test label samples:", test_label_samples.shape)
            print("current best validation: {}".format(CURR_BEST_VALID))
            print("current best test: {}".format(CURR_BEST_TEST))
            print("counter: {}".format(self.counter))
            print(feat_samples.shape)
            print(valid_label_samples.shape)
            model = DNGO(num_epochs=100, n_units=128, do_mcmc=False, normalize_output=False)
            model.train(X=feat_samples.numpy(), y=valid_label_samples.view(-1).numpy(), do_optimize=True)
            print(model.network)
            m = []
            v = []
            chunks = int(features.shape[0] / window_size)
            if features.shape[0] % window_size > 0:
                chunks += 1
            features_split = torch.split(features, window_size, dim=0)
            for i in range(chunks):
                m_split, v_split = model.predict(features_split[i].numpy())
                m.extend(list(m_split))
                v.extend(list(v_split))
            mean = torch.Tensor(m)
            sigma = torch.Tensor(v)
            # u = (mean - torch.Tensor([args.objective]).expand_as(mean)) / sigma
            u = (mean - torch.Tensor([0.95]).expand_as(mean)) / sigma
            normal = Normal(torch.zeros_like(u), torch.ones_like(u))
            ucdf = normal.cdf(u)
            updf = torch.exp(normal.log_prob(u))
            ei = sigma * (updf + u * ucdf)

            # feat_next, geno_next, label_next_valid, label_next_test, visited = propose_location(ei, features, genotype, visited, counter)
            count = counter
            # k = args.batch_size
            k = 5
            c = 0
            print('remaining length of indices set:', len(features) - len(visited))
            indices = torch.argsort(ei)
            ind_dedup = []
            # remove random sampled indices at each step
            for idx in reversed(indices):
                if c == k:
                    break
                if idx.item() not in visited:
                    visited[idx.item()] = True
                    ind_dedup.append(idx.item())
                    c += 1
            ind_dedup = torch.Tensor(ind_dedup).long()
            print('proposed index: {}'.format(ind_dedup))
            proposed_x = features[ind_dedup]
            self.proposed_geno = [genotype[i.item()] for i in ind_dedup]
            next_geno = self.proposed_geno[0]
            # print("@next_geno==self.proposed_geno[0] : ", next_geno)




        return next_geno
    

    def update_search_space(self, search_space):
        '''
        Tuners are advised to support updating search space at run-time.
        If a tuner can only set search space once before generating first hyper-parameters,
        it should explicitly document this behaviour.
        search_space: JSON object created by experiment owner
        '''
        # your code implements here.
