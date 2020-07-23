from lc import Algorithm
from lc.torch import ParameterTorch as LCParameterTorch
from torch import nn
import torch
import time
import asyncio
import numpy as np


class RankSelectionLcAlg(Algorithm):
    def __init__(self, model, compression_tasks, lc_config, l_step_optimization, evaluation_func, l_step_config):
        mu_schedule = [lc_config['mu_init'] * (lc_config['mu_inc'] ** step) for step in range(lc_config['steps']) for _ in range(lc_config['mu_rep'])]
        print("Used mu-schedule:", mu_schedule)
        l_step_config['all_start_time'] = time.time()
        super(RankSelectionLcAlg, self).__init__(model, compression_tasks, mu_schedule,
                                                 l_step_optimization, evaluation_func)
        # All view-related data structures should be attached to model parameters for easier retrieval
        all_torch_nn_parameters = []
        for lc_param in self.lc_parameters:
            if isinstance(lc_param, LCParameterTorch):
                all_torch_nn_parameters.extend(lc_param.d_theta_list)
            else:
                raise Exception("RankSelection contains non-torch LCparameters, critical error")
        print(f"Total of {len(all_torch_nn_parameters)} additional parameters will be added to model_parameters.")
        model.lc_param_list = nn.ParameterList(getter() for getter in all_torch_nn_parameters)
        self.l_step_config = l_step_config
        self.lstep_info = {}
        self.eval_info = {}
        self.step_number = -1
        self.lc_config = lc_config
        self.model_state = None
        self.best_model_state = None
        self.best_compression_tasks_state = None
        self.best_compression_tasks_info = None
        self.best_compression_tasks_lambdas = None

    def save(self, step_number, setup_name, tag):
        # I'm making changes to saving routine to only save last and best  lc iteration model state
        self.model_state = self.model.state_dict()

        for key, value in self.model_state.items():
            # this line is done to remove item from gpu memory
            self.model_state[key] = value.cpu()

        to_save = {}
        to_save['model_states'] = self.model_state
        to_save['last_step_number'] = step_number
        to_save['l_step_config'] = self.l_step_config
        to_save['l_step_info'] = self.lstep_info
        to_save['lc_config'] = self.lc_config
        to_save['mu_schedule'] = self.mu_schedule
        to_save['eval_info'] = self.eval_info

        compression_tasks_state = {}
        compression_tasks_info = {}
        compression_tasks_lambdas = {}
        # Since we are not storing model states, we are going to store compressed C-step solutions
        for param in self.compression_tasks.keys():
            (view, compression, task_name) = self.compression_tasks[param]
            compression_tasks_state[task_name] = compression.state_dict
            compression_tasks_info[task_name] = compression.info
            compression_tasks_lambdas[task_name] = param.lambda_

        to_save['compression_tasks_state'] = compression_tasks_state
        to_save['compression_tasks_info'] = compression_tasks_info
        to_save['compression_tasks_lambdas'] = compression_tasks_lambdas

        if self.eval_info['best']['step_number'] == step_number:
            self.best_compression_tasks_state = compression_tasks_state
            self.best_compression_tasks_info = compression_tasks_info
            self.best_compression_tasks_lambdas = compression_tasks_lambdas
            self.best_model_state = self.model_state

        to_save['best_compression_tasks_state'] = self.best_compression_tasks_state
        to_save['best_compression_tasks_info'] = self.best_compression_tasks_info
        to_save['best_compression_tasks_lambdas'] = self.best_compression_tasks_lambdas
        to_save['best_model_state'] = self.best_model_state

        return to_save

    def restore(self, setup_name, tag, continue_with_original_config=False, restore_from_best=False):
        checkpoint = torch.load(f'results/{setup_name}_lc_{tag}.th', map_location='cpu')
        step_number = checkpoint['last_step_number']
        self.best_model_state = checkpoint['best_model_state']

        if restore_from_best:
            self.model.load_state_dict(self.best_model_state)
        else:
            self.model.load_state_dict(checkpoint['model_states'])
        self.model = self.model.cuda()

        self.model_states = checkpoint['model_states']
        if continue_with_original_config:
            self.l_step_config = checkpoint['l_step_config']
            self.lc_config = checkpoint['lc_config']
            self.mu_schedule = checkpoint['mu_schedule']
        print("L_STEP CONFIG:")
        print(self.l_step_config)
        print("LC CONFIG:")
        print(self.lc_config)
        print("MU schedule:")
        print(self.mu_schedule)

        self.lstep_info = checkpoint['l_step_info']
        self.eval_info = checkpoint['eval_info']

        compression_tasks_state = checkpoint['compression_tasks_state']
        compression_tasks_info = checkpoint['compression_tasks_info']
        compression_tasks_lambdas = checkpoint['compression_tasks_lambdas']

        if restore_from_best:
            step_number = self.eval_info['best']['step_number']
            compression_tasks_state = checkpoint['best_compression_tasks_state']
            compression_tasks_info = checkpoint['best_compression_tasks_info']
            compression_tasks_lambdas = checkpoint['best_compression_tasks_lambdas']


        self.best_compression_tasks_lambdas = checkpoint['best_compression_tasks_lambdas']
        self.best_compression_tasks_info = checkpoint['best_compression_tasks_info']
        self.best_compression_tasks_state = checkpoint['best_compression_tasks_state']
        for param in self.compression_tasks.keys():
            (view, compression, task_name) = self.compression_tasks[param]
            compression_state = compression_tasks_state[task_name]
            if "init_shape" not in compression_state:
                # we need to manually set init_shape since old version didn't save it
                x=param.vector_to_compression_view(param.w, view)
                compression_state["init_shape"] = x.shape
            compression.load_state_dict(compression_state)
            compression.info = compression_tasks_info[task_name]

            if isinstance(param, LCParameterTorch):
                param.retrieve(full=True)
                param.lambda_ = compression_tasks_lambdas[task_name]
                param.delta_theta = param.compression_view_to_vector(compression.uncompress_state(), view)
                print(param.delta_theta, compression, task_name)
            else:
                raise Exception("RankSelection contains non-torch LCparameters, critical error")

        return step_number

    def multipliers_step(self):
        """
        Updates Lagrange multipliers estimates in one step procedure.
        """
        for param in self.lc_parameters:
            if param.bn_after and self.compression_tasks[param][1].selected_rank == 0:
                continue
            param.lambda_ -= self.mu * (param.w - param.delta_theta)

    def lc_penalty(self):
        """
        Computes total penalty among all lc.Parameters for this compression run.
        """
        loss_ = 0.0

        for lc_parameter in self.lc_parameters:
            if lc_parameter.bn_after and self.compression_tasks[lc_parameter][1].selected_rank == 0:
                continue
            loss_ += self.mu*0.5*lc_parameter.lc_penalty()

        return loss_

    def l_step(self, step):

        l_step_needed = False

        for lc_parameter in self.lc_parameters:
            if not lc_parameter.bn_after:
                l_step_needed = True
                break
            elif self.compression_tasks[lc_parameter][1].selected_rank != 0:
                l_step_needed = True
                break

        if not l_step_needed:
            print(f"With current compression structure, L-step for μ={self.mu} is skipped.")
            return

        for param in self.lc_parameters:
            param.target = param.delta_theta + (0 if self.mu == 0 else 1 / self.mu * param.lambda_)

        self.release()
        info = self.l_step_optimization(self.model, self.lc_penalty, step, self.l_step_config)
        self.lstep_info[step] = info
        self.retrieve()

    def evaluate(self):
        # first make regular evaluation of LC objective: loss + lc_penalty
        # self.evaluation_func()
        # then compressed evaluation of objective: loss
        self.compression_eval()
        info = self.evaluation_func(self.model)
        self.eval_info[self.step_number] = info

        if 'best' in self.eval_info:
            # if we have best, try to update
            best_info = self.eval_info['best']
            if info['nested_train_loss'] < best_info['nested_train_loss']:
                # copy results
                self.eval_info['best'] = dict(info)
                self.eval_info['best']['step_number'] = self.step_number
        else:
            # if there is no best, current info is best by def
            self.eval_info['best'] = dict(info)
            self.eval_info['best']['step_number'] = self.step_number

        # output best info
        best_info = self.eval_info['best']
        print('------------------>--------------------')
        print('\tBEST TRAIN LOSS was encountered at step :', best_info['step_number'])
        print('\tBEST nested train loss: {:.6f}, accuracy: {:.4f}'
              .format(best_info['nested_train_loss'], best_info['nested_train_acc']))
        print('\tCORR nested test  loss: {:.6f}, accuracy: {:.4f}'
              .format(best_info['nested_test_loss'], best_info['nested_acc']))
        print('------------------>--------------------')
        self.compression_train()

    def run(self, name=None, tag=None, restore=False):
        # set mu=0 to perform direct compression
        start_step = 0
        if not restore:
            self.mu = 0
            self.c_step(step_number=-1)
            print("Direct compression has been performed.")

            for param in self.lc_parameters:
                param.lambda_ *= 0
                # this will effectively set Lagrange multipliers to 0

            self.evaluate() # evaluates train/test acc-s
        else:
            restore_step_number = self.restore(name, tag)
            print(f"We are continuing previous training from checkpoint={restore_step_number}")
            start_step = restore_step_number + 1

        from threading import Thread
        import asyncio

        def start_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        new_loop = asyncio.new_event_loop()
        t = Thread(target=start_loop, args=(new_loop,))
        t.start()

        for step_number in range(start_step, len(self.mu_schedule)):
            mu = self.mu_schedule[step_number]
            self.mu = mu
            self.step_number = step_number
            print(f"Current LC step is {step_number}, μ={self.mu:.3e}")
            if step_number != 0:
                self.l_step(step_number)
            else:
                print('skipping initial l_step')
            print(f"L-step #{step_number} has finished.")
            self.c_step(step_number)
            print(f"C-step #{step_number} has finished.")
            if step_number+1 < len(self.mu_schedule) and mu == self.mu_schedule[step_number+1]:
                print("NOT UPDATING Lagrange multipliers")
            else:
                self.multipliers_step()
                print("Lagrange multipliers have been updated.")
            self.evaluate()
            current_state = self.save(step_number, name, tag)

            async def actual_save():
                torch.save(current_state, f'results/{name}_lc_{tag}.th')

            asyncio.run_coroutine_threadsafe(actual_save(), new_loop)

            # this is an implementation of early abort
            # if selected ranks are not changing for the last 5 iterations and non-zero, abort
            # def last_five_ranks_same():
            #     for task_name, task_info in current_state['compression_tasks_info'].items():
            #         current_selected_rank = task_info[step_number]['selected_rank']
            #         for old_step in range(step_number-5, step_number):
            #             if old_step not in task_info:
            #                 return False
            #             if current_selected_rank != task_info[old_step]['selected_rank'] or current_selected_rank == 0:
            #                 return False
            #     return True
            #
            # if last_five_ranks_same():
            #     print("No rank changes have been detected in the last 5 LC steps. LC terminated early.")
            #     return

        async def last_task():
            print("Async file saving has been finished.")
            new_loop.stop()
        asyncio.run_coroutine_threadsafe(last_task(), new_loop)
