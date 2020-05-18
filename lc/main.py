#!/usr/bin/env python3
from .compression_types import CompressionTypeBase
import numpy as np


class Algorithm():
    """
    This class implements model compression using LC framework. See [1] for more details on the algorithm.

    To compress a model, user needs to construct lc.Algorithm object and provide:
        1) a model to be compressed,
        2) the associated compression tasks,
        3) an implementation of the L-step, i.e. the function to optimize the loss + a penalty
        4) a schedule of μ values.

    To start the compression .run() method should be invoked.

    References:
    1. Miguel Á. Carreira-Perpiñán
       Model compression as constrained optimization, with application to neural nets.Part I: general framework.
       https://arxiv.org/abs/1707.01209

    """
    def __init__(self, model, compression_tasks, mu_schedule, l_step_optimization, evaluation_func, c_step_reps=50):
        """
        Constructs lc.Algorithm object

        :param model: model to compress
        :param compression_tasks: a dictionary mapping lc.Parameter => (compression view, compression type, task_name),
        :param mu_schedule: a list of μ values (doubles)
        :param l_step_optimization:
        :param evaluation_func:
        :param c_step_reps: How many times the alternation between c-steps will happen in additive compression regime.
            Has no effect otherwise.
        """
        super(Algorithm, self).__init__()
        self.model = model
        self.mu = None
        self.mu_schedule = mu_schedule
        self.lc_parameters = compression_tasks.keys()
        self.c_step_reps = c_step_reps

        # user provides a constructor for a view, but it must be converted to an actual view
        # which happens here:
        for param in compression_tasks.keys():
            one_or_multiple_tasks = compression_tasks[param]

            if isinstance(one_or_multiple_tasks, tuple):
                # single compression per task
                (compression_view_constructor, compression, task_name) = compression_tasks[param]
                if isinstance(compression, CompressionTypeBase):
                    compression_tasks[param] = (compression_view_constructor(param.w_view.original_form), compression, task_name)
                else:
                    raise Exception("The given compression is not an instance of CompressionTypeBase")
            elif isinstance(one_or_multiple_tasks, list) and len(one_or_multiple_tasks) == 1:
                (compression_view_constructor, compression, task_name) = compression_tasks[param][0]
                if isinstance(compression, CompressionTypeBase):
                    compression_tasks[param] = (compression_view_constructor(param.w_view.original_form), compression, task_name)
                else:
                    raise Exception("The given compression is not an instance of CompressionTypeBase")
            elif isinstance(one_or_multiple_tasks, list):
                # additive combinations
                for param in compression_tasks.keys():
                    list_ = []
                    for (compression_view_constructor, compression, task_name) in compression_tasks[param]:
                        list_.append((compression_view_constructor(param.w_view.original_form), compression, task_name))
                        compression.current_sol = 0
                    compression_tasks[param] = list_
            else:
                raise Exception("The given compression tasks are not supported or contain an error:"
                                "Given compression is not a child of CompressionTypeBase nor a list of CompressionTypeBase-s")

        self.compression_tasks = compression_tasks
        # self.additive_compression_tasks = additive_compression_tasks
        self.l_step_optimization = l_step_optimization
        self.evaluation_func = evaluation_func
        self.retrieve()

    def lc_penalty(self):
        """
        Computes total penalty among all lc.Parameters for this compression run.
        """
        loss_ = 0.0
        for lc_parameter in self.lc_parameters:
            loss_ += self.mu*0.5*lc_parameter.lc_penalty()

        return loss_

    def release(self):
        for param in self.lc_parameters:
            param.release()

    def retrieve(self):
        for param in self.lc_parameters:
            param.retrieve()

    def c_step(self, step_number):
        """
        This methods performs actual C-steps on compression tasks. For every task given, it runs extracts model
        parameters associated with the task, reshapes them according to compression view and compresses them. The result
        of compression will be put into param.delta_theta
        """

        for param, task_details in self.compression_tasks.items():
            if isinstance(task_details, tuple):
                view, compression, _name = task_details
                # The c-steps for regular compression tasks (one compression per task). We need to solve
                #   min ‖w-Δ(Θ)-λ/μ‖²
                # which is given by C-step of Δ for (w-λ/μ).

                # offset weight vector: w_lambda_mu =  w - lambda/mu (w-λ/μ)
                w_lambda_mu = param.w - (0 if self.mu == 0 else 1 / self.mu * param.lambda_)
                compression.mu = self.mu
                compression.step_number = step_number
                w_lambda_mu_compression_view = param.vector_to_compression_view(w_lambda_mu, view)
                delta_theta_compression_view = compression.compress(w_lambda_mu_compression_view)
                # updated the current delta_theta, i.e. Δ(Θ)
                param.delta_theta = param.compression_view_to_vector(delta_theta_compression_view, view)

            elif isinstance(task_details, list):
                # Additive compressions regime
                # In this setting we have multiple compressions for ever parameter, which are additively combined.
                # Thus, the compression problem is given as:
                #       min ‖w-\sum_{i}Δᵢ(Θᵢ)‖²
                # To solve it we use the alternating optimization over Θᵢ-s, which turns out to be a C-step of
                # corresponding compression Δᵢ. For example, for the k-th term we are solving:
                #       min_{Θₖ} ‖w-\sum_{i≠k}Δᵢ(Θᵢ) - Δₖ(Θₖ)‖²  <===> C_step of Δₖ for (w-\sum_{i≠ₖ}Δᵢ(Θᵢ))

                # To compute the term w-\sum_{i≠k}Δᵢ(Θᵢ), we compute the sum of all Δᵢ, which is current delta_theta:
                current_sol = param.delta_theta

                for j in range(self.c_step_reps):
                    # The alternation is repeated c_step_reps times.
                    for (view, compression, _name) in task_details:
                        # we compute \sum_{i≠k}Δᵢ(Θᵢ) first
                        other_deltas = current_sol - compression.current_sol

                        # offset the weight vector: w_lambda_mu =  w - \sum_{i≠ₖ}Δᵢ(Θᵢ) - λ/μ
                        w_lambda_mu = param.w - other_deltas - (0 if self.mu == 0 else 1 / self.mu * param.lambda_)
                        compression.mu = self.mu
                        compression.step_number = step_number
                        w_lambda_mu_compression_view = param.vector_to_compression_view(w_lambda_mu, view)

                        # the C-step corresponding to a particular compression:
                        delta_theta_compression_view = compression.compress(w_lambda_mu_compression_view)
                        compression.current_sol = param.compression_view_to_vector(delta_theta_compression_view, view)
                        current_sol = other_deltas + compression.current_sol
                        current_err = np.sum(
                            (param.w - current_sol - (0 if self.mu == 0 else 1 / self.mu * param.lambda_)) ** 2)
                        print(f"C-step for additive task {_name}, "
                              f"alt-opt iteration:{j}, err=‖w-Δ(Θ)‖²= {current_err:.6f}")

                # updated the current delta_theta, i.e., \sum_{i}Δᵢ(Θᵢ)
                delta_theta = np.sum([compression.current_sol for (_, compression, _) in task_details], axis=0)
                param.delta_theta = delta_theta

    def multipliers_step(self):
        """
        Updates Lagrange multipliers estimates in one step procedure.
        """
        for param in self.lc_parameters:
            param.lambda_ -= self.mu * (param.w - param.delta_theta)

    def compression_eval(self):
        """
        Sets w=Δ(ϴ), i.e. making sure that model weights are coming from compression.
        :return:
        """
        for param in self.lc_parameters:
            param.eval()

    def compression_train(self):
        for param in self.lc_parameters:
            param.train()

    def evaluate(self):
        # first make regular evaluation of LC objective: loss + lc_penalty
        # self.evaluation_func()
        # then compressed evaluation of objective: loss
        self.compression_eval()
        self.evaluation_func(self.model)
        self.compression_train()

    def l_step(self, step):
        """
        This method sets a new target for the LC penalty function and invokes user supplied L_step_optimization function.
        Since L-step optimization interacts with outside DL frameworks, everything should be packed so these frameworks
        can work with new input properly.
        """
        # before releasing we combine Lagrange multipliers with quadratic penalty values to update target for penalty
        # aka offset compressed weight vector, which will be target for the penalty
        for param in self.lc_parameters:
            param.target = param.delta_theta + (0 if self.mu == 0 else 1 / self.mu * param.lambda_)

        self.release()
        self.l_step_optimization(self.model, self.lc_penalty, step)
        self.retrieve()

    def run(self):
        """
        This runs the LC algorithm proper.
        """
        # set mu=0 to perform direct compression
        self.mu = 0
        self.c_step(step_number=0)
        print("Direct compression has been performed.")

        for param in self.lc_parameters:
            param.lambda_ *= 0
            # this will effectively set Lagrange multipliers to 0

        self.evaluate() # evaluates train/test acc-s
        for step_number, mu in enumerate(self.mu_schedule):
            self.mu = mu
            print(self.mu)
            self.l_step(step_number)
            print(f"L-step #{step_number} has finished.")
            self.c_step(step_number)
            print(f"C-step #{step_number} has finished.")
            self.multipliers_step()
            print("Lagrange multipliers have been updated.")
            self.evaluate()