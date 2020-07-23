from lc.models.torch.lenet5 import lenet5_drop
from lc.models.torch.lenet300 import lenet300_classic
from lc_for_rank_selection_cvpr2020.utils import AverageMeter, Recorder, format_time, data_loader, compute_acc_loss
import torch
import time
from lc.torch import ParameterTorch as LCParameterTorch, AsIs
from torch import nn
from lc.compression_types.low_rank import RankSelection
from lc_for_rank_selection_cvpr2020.utils import add_flops_counting_methods
from lc_for_rank_selection_cvpr2020.new_finetune import reparametrize_low_rank
from lc.models.torch.utils import count_params


__all__ = ['lenet300_all', 'lenet5_all']


class lenet_all():
  def __init__(self, name, model, stored_ref):
    self.device = torch.device('cuda')
    self.name = name
    self.model = model.to(self.device)
    self.train_loader, self.test_loader = data_loader(batch_size=256, n_workers=4, dataset="MNIST")
    pretrained_model = torch.load('references/{}.th'.format(stored_ref))
    self.model.load_state_dict(pretrained_model['model_state'])

  def lc_setup(self):
    def l_step_optimization(model, lc_penalty, step, config):

      all_start_time = config['all_start_time']

      lr_scheduler = None
      my_params = filter(lambda p: p.requires_grad, model.parameters())
      learning_rate = config['lr']

      if config['lr_decay_mode'] == 'after_l':
        learning_rate *= (config['lr_decay'] ** step)
        print(f"Current LR={learning_rate}")

      def constract_my_forward_lc_eval(lc_penalty):
        pen = lc_penalty()

        def my_forward_lc_eval(x, target):
          out_ = model.forward(x)
          return out_, model.loss(out_, target) + pen

        return my_forward_lc_eval

      optimizer = torch.optim.SGD(my_params, learning_rate,
                                  momentum=config['momentum'], nesterov=True)

      if config['lr_decay_mode'] == 'restart_on_l':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_decay'])

      if 'lr_trick' in config:
        l_trick_value = 0.1
        print('LR trick in play. first epoch is trained with LR of {:.4e}'.format(config['lr'] * l_trick_value))
        for param_group in optimizer.param_groups:
          param_group['lr'] = config['lr'] * l_trick_value
        # TODO: revert back the lr_trick?

      epochs_in_this_it = config['epochs'] if step > 0 else \
        config['first_mu_epochs'] if 'first_mu_epochs' in config else config['epochs']
      print('Epochs in this iteration is :', epochs_in_this_it)

      print('Epochs in this iteration is :', epochs_in_this_it)
      model.eval()

      lc_evaluator = constract_my_forward_lc_eval(lc_penalty)
      accuracy, ave_loss = compute_acc_loss(lc_evaluator, self.train_loader)
      print('\ttrain loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss, accuracy))
      accuracy, ave_loss = compute_acc_loss(lc_evaluator, self.test_loader)
      print('\ttest  loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss, accuracy))
      model.train()
      epoch_time = AverageMeter()
      rec = Recorder()

      # avg_epoch_losses = []
      for epoch in range(epochs_in_this_it):
        start_time = time.time()
        avg_loss_ = AverageMeter()
        for x, target in self.train_loader:
          optimizer.zero_grad()
          x, target = x.cuda(), target.cuda(non_blocking=True)
          loss = model.loss(model(x), target) + lc_penalty()
          avg_loss_.update(loss.item())
          loss.backward()
          optimizer.step()
        end_time = time.time()
        training_time = end_time - all_start_time
        epoch_time.update(end_time - start_time)

        print("LC step {0}, Epoch {1} finished in {2.val:.3f}s (avg: {2.avg:.3f}s). Training for {3}"
              .format(step, epoch, epoch_time, format_time(end_time - all_start_time)))
        print('AVG train loss {0.avg:.6f}'.format(avg_loss_))
        rec.record('average_loss_per_epoch', avg_loss_)

        if (epoch + 1) % config['print_freq'] == 0:
          model.eval()
          lc_evaluator = constract_my_forward_lc_eval(lc_penalty)

          accuracy, ave_loss = compute_acc_loss(lc_evaluator, self.train_loader)
          rec.record('train', [ave_loss, accuracy, training_time, step + 1, epoch + 1])
          print('\ttrain loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss, accuracy))

          accuracy, ave_loss = compute_acc_loss(lc_evaluator, self.test_loader)
          rec.record('test', [ave_loss, accuracy, training_time, step + 1, epoch + 1])
          print('\ttest  loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss, accuracy))
          model.train()

        if config['lr_decay_mode'] == 'restart_on_l':
          print("\told LR: {:.4e}".format(optimizer.param_groups[0]['lr']))
          lr_scheduler.step()
          print("\tnew LR: {:.4e}".format(optimizer.param_groups[0]['lr']))

        else:
          print("\tLR: {:.4e}".format(learning_rate))

      info = {'train': rec.train, 'test': rec.test, 'average_loss_per_train_epoch': rec.average_loss_per_epoch}
      return info

    def evaluation(model):
      def my_forward_eval(x, target):
        out_ = model.forward(x)
        return out_, model.loss(out_, target)

      model.eval()

      accuracy_train, ave_loss_train = compute_acc_loss(my_forward_eval, self.train_loader)
      print('\tnested train loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss_train, accuracy_train))
      # rec.record('train_nested', [ave_loss, accuracy, training_time, step + 1])
      accuracy_test, ave_loss_test = compute_acc_loss(my_forward_eval, self.test_loader)
      model.train()
      print('\tnested test  loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss_test, accuracy_test))
      # rec.record('test_nested', [ave_loss, accuracy, training_time, step + 1])
      # model.lc_train()
      return {
        'nested_train_loss': ave_loss_train,
        'nested_train_acc': accuracy_train,
        'nested_test_loss': ave_loss_test,
        'nested_acc': accuracy_test
      }

    def create_lc_compression_task(config_):

      if config_['criterion'] == "flops":
        model = add_flops_counting_methods(self.model)
        model.start_flops_count()

        for x, target in self.train_loader:
          _ = model(x[None, 0].cuda())
          break
        uncompressed_flops = model.compute_average_flops_cost()
        print('The number of FLOPS in this model', uncompressed_flops)
        model.stop_flops_count()

      compression_tasks = {}
      for i, (w_get, module) in enumerate([((lambda x=x: getattr(x, 'weight')), x) for x in self.model.modules() if
                                           isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear)]):
        compression_tasks[LCParameterTorch(w_get, self.device)] \
          = (AsIs,
             RankSelection(conv_scheme=config_['conv_scheme'], alpha=config_["alpha"], criterion=config_["criterion"],
                           normalize=False, module=module), f"task_{i}")

      return compression_tasks

    return l_step_optimization, evaluation, create_lc_compression_task

  def finetune_setup(self, tag_of_lc_model, c_step_config):
    exp_run_details = torch.load(f"results/{self.name}_lc_{tag_of_lc_model}.th", map_location="cpu")
    # despite the 's' at the end, there is only one model state, the last
    model_state_to_load = exp_run_details['model_states']
    last_lc_it = exp_run_details['last_step_number']

    model = add_flops_counting_methods(self.model)
    model.start_flops_count()

    for x, target in self.train_loader:
      _ = model(x[None, 0].cuda())
      break
    all_flops = model.compute_average_flops_cost()
    model.stop_flops_count()
    self.model = self.model.cpu()
    all_params = count_params(self.model)

    compression_info = {}
    for task_name, infos in exp_run_details['compression_tasks_info'].items():
      compression_info[task_name] = infos[last_lc_it]
    # compression_infos = exp_run_details['compression_tasks_info'][last_lc_it]

    print(model_state_to_load.keys())

    for key in list(model_state_to_load.keys()):
      if key.startswith("lc_param_list"):
        del model_state_to_load[key]

    del exp_run_details
    import gc
    gc.collect()

    print(model_state_to_load.keys())

    self.model.load_state_dict(model_state_to_load)
    print("model has been sucessfully loaded")

    for i, module in enumerate(
        [x for x in self.model.modules() if isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear)]):
      module.selected_rank_ = compression_info[f"task_{i}"]['selected_rank']
      print(module.selected_rank_)

    reparametrize_low_rank(self.model)
    print(self.model)

    self.model = self.model.to(self.device)
    print("Low rank layers of the model has been successfully reparameterized with sequence of full-rank matrices.")

    def my_forward_eval(x, target):
      out_ = self.model.forward(x)
      return out_, self.model.loss(out_, target)

    self.model.eval()
    accuracy_train, ave_loss_train = compute_acc_loss(my_forward_eval, self.train_loader)
    print('\tBefore finetuning, the train loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss_train, accuracy_train))
    # rec.record('train_nested', [ave_loss, accuracy, training_time, step + 1])
    accuracy_test, ave_loss_test = compute_acc_loss(my_forward_eval, self.test_loader)
    self.model.train()
    print('\tBefore finetuning, the test loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss_test, accuracy_test))

    model = add_flops_counting_methods(self.model)
    model.start_flops_count()

    for x, target in self.train_loader:
      _ = model(x[None, 0].cuda())
      break
    compressed_flops = model.compute_average_flops_cost()
    model.stop_flops_count()
    self.model = self.model.cpu()
    compressed_params = count_params(model)
    self.model = self.model.to(self.device)
    print('The number of FLOPS in original model', all_flops)
    print('The number of params in original model:', all_params)
    print('The number of FLOPS in this model', compressed_flops)
    print('The number of params in this model:', compressed_params)
    flops_rho = all_flops[0] / compressed_flops[0]
    storage_rho = all_params / compressed_params
    print(f'FLOPS ρ={flops_rho:.3f}; STORAGE ρ={storage_rho:.3f};')

    compression_stats = {'original_flops': all_flops, 'compressed_flops': compressed_flops,
                         'flops_rho': flops_rho, 'storage_rho': storage_rho}

    def finetuning(config):
      all_start_time = time.time()
      my_params = filter(lambda p: p.requires_grad, self.model.parameters())
      optimizer = torch.optim.SGD(my_params, config['lr'],
                                  momentum=config['momentum'], nesterov=True)
      epoch_time = AverageMeter()
      lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_decay'])

      train_info = {}
      test_info = {}

      from threading import Thread
      import asyncio

      def start_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

      new_loop = asyncio.new_event_loop()
      t = Thread(target=start_loop, args=(new_loop,))
      t.start()

      for epoch in range(0, config['epochs']):
        start_time = time.time()
        avg_loss_ = AverageMeter()
        for x, target in self.train_loader:
          optimizer.zero_grad()
          x, target = x.cuda(), target.cuda()
          loss = self.model.loss(self.model(x), target)
          loss.backward()
          avg_loss_.update(loss.item())
          optimizer.step()
        end_time = time.time()
        training_time = end_time - all_start_time
        epoch_time.update(end_time - start_time)

        print("Epoch {0} finished in {1.val:.3f}s (avg: {1.avg:.3f}s). Training for {2}"
              .format(epoch, epoch_time, format_time(end_time - all_start_time)))
        print('AVG train loss {0.avg:.6f}'.format(avg_loss_))

        print("\tLR: {:.4e}".format(lr_scheduler.get_lr()[0]))
        lr_scheduler.step()
        if (epoch + 1) % config['print_freq'] == 0:
          self.model.eval()
          accuracy, ave_loss = compute_acc_loss(my_forward_eval, self.train_loader)
          train_info[epoch + 1] = [ave_loss, accuracy, training_time]
          print('\ttrain loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss, accuracy))

          accuracy, ave_loss = compute_acc_loss(my_forward_eval, self.test_loader)
          test_info[epoch + 1] = [ave_loss, accuracy, training_time]
          print('\ttest  loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss, accuracy))
          self.model.train()

          to_save = {}
          to_save['config'] = config
          to_save['optimizer_state'] = optimizer.state_dict()
          to_save['model_state'] = self.model.state_dict()
          to_save['training_time'] = training_time
          to_save['traing_info'] = train_info
          to_save['test_info'] = test_info
          to_save['current_epoch'] = epoch
          to_save['compression_stats'] = compression_stats

          async def actual_save():
            # TODO: make better saves, 1) mv file as backup, 2) save new data 3) delte bk
            torch.save(to_save, f'results/{self.name}_ft_{config["tag"]}.th')
          asyncio.run_coroutine_threadsafe(actual_save(), new_loop)

      async def last_task():
        print("Async file saving has been finished.")
        new_loop.stop()

      asyncio.run_coroutine_threadsafe(last_task(), new_loop)

    return finetuning


class lenet300_all(lenet_all):
  def __init__(self):
    super(lenet300_all, self).__init__("lenet300_all", lenet300_classic(), 'lenet300_classic')



class lenet_all_normalized():
  def __init__(self, name, model, stored_ref):
    self.device = torch.device('cuda')
    self.name = name
    self.model = model.to(self.device)
    self.train_loader, self.test_loader = data_loader(batch_size=256, n_workers=4, dataset="MNIST")
    pretrained_model = torch.load('references/{}.th'.format(stored_ref))
    self.model.load_state_dict(pretrained_model['model_state'])

  def lc_setup(self):
    def l_step_optimization(model, lc_penalty, step, config):

      all_start_time = config['all_start_time']

      lr_scheduler = None
      my_params = filter(lambda p: p.requires_grad, model.parameters())
      learning_rate = config['lr']

      if config['lr_decay_mode'] == 'after_l':
        learning_rate *= (config['lr_decay'] ** step)
        print(f"Current LR={learning_rate}")

      def constract_my_forward_lc_eval(lc_penalty):
        pen = lc_penalty()

        def my_forward_lc_eval(x, target):
          out_ = model.forward(x)
          return out_, model.loss(out_, target) + pen

        return my_forward_lc_eval

      optimizer = torch.optim.SGD(my_params, learning_rate,
                                  momentum=config['momentum'], nesterov=True)

      if config['lr_decay_mode'] == 'restart_on_l':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_decay'])

      if 'lr_trick' in config:
        l_trick_value = 0.1
        print('LR trick in play. first epoch is trained with LR of {:.4e}'.format(config['lr'] * l_trick_value))
        for param_group in optimizer.param_groups:
          param_group['lr'] = config['lr'] * l_trick_value
        # TODO: revert back the lr_trick?

      epochs_in_this_it = config['epochs'] if step > 0 else \
        config['first_mu_epochs'] if 'first_mu_epochs' in config else config['epochs']
      print('Epochs in this iteration is :', epochs_in_this_it)

      print('Epochs in this iteration is :', epochs_in_this_it)
      model.eval()

      lc_evaluator = constract_my_forward_lc_eval(lc_penalty)
      accuracy, ave_loss = compute_acc_loss(lc_evaluator, self.train_loader)
      print('\ttrain loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss, accuracy))
      accuracy, ave_loss = compute_acc_loss(lc_evaluator, self.test_loader)
      print('\ttest  loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss, accuracy))
      model.train()
      epoch_time = AverageMeter()
      rec = Recorder()

      # avg_epoch_losses = []
      for epoch in range(epochs_in_this_it):
        start_time = time.time()
        avg_loss_ = AverageMeter()
        for x, target in self.train_loader:
          optimizer.zero_grad()
          x, target = x.cuda(), target.cuda(non_blocking=True)
          loss = model.loss(model(x), target) + lc_penalty()
          avg_loss_.update(loss.item())
          loss.backward()
          optimizer.step()
        end_time = time.time()
        training_time = end_time - all_start_time
        epoch_time.update(end_time - start_time)

        print("LC step {0}, Epoch {1} finished in {2.val:.3f}s (avg: {2.avg:.3f}s). Training for {3}"
              .format(step, epoch, epoch_time, format_time(end_time - all_start_time)))
        print('AVG train loss {0.avg:.6f}'.format(avg_loss_))
        rec.record('average_loss_per_epoch', avg_loss_)

        if (epoch + 1) % config['print_freq'] == 0:
          model.eval()
          lc_evaluator = constract_my_forward_lc_eval(lc_penalty)

          accuracy, ave_loss = compute_acc_loss(lc_evaluator, self.train_loader)
          rec.record('train', [ave_loss, accuracy, training_time, step + 1, epoch + 1])
          print('\ttrain loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss, accuracy))

          accuracy, ave_loss = compute_acc_loss(lc_evaluator, self.test_loader)
          rec.record('test', [ave_loss, accuracy, training_time, step + 1, epoch + 1])
          print('\ttest  loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss, accuracy))
          model.train()

        if config['lr_decay_mode'] == 'restart_on_l':
          print("\told LR: {:.4e}".format(optimizer.param_groups[0]['lr']))
          lr_scheduler.step()
          print("\tnew LR: {:.4e}".format(optimizer.param_groups[0]['lr']))

        else:
          print("\tLR: {:.4e}".format(learning_rate))

      info = {'train': rec.train, 'test': rec.test, 'average_loss_per_train_epoch': rec.average_loss_per_epoch}
      return info

    def evaluation(model):
      def my_forward_eval(x, target):
        out_ = model.forward(x)
        return out_, model.loss(out_, target)

      model.eval()

      accuracy_train, ave_loss_train = compute_acc_loss(my_forward_eval, self.train_loader)
      print('\tnested train loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss_train, accuracy_train))
      # rec.record('train_nested', [ave_loss, accuracy, training_time, step + 1])
      accuracy_test, ave_loss_test = compute_acc_loss(my_forward_eval, self.test_loader)
      model.train()
      print('\tnested test  loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss_test, accuracy_test))
      # rec.record('test_nested', [ave_loss, accuracy, training_time, step + 1])
      # model.lc_train()
      return {
        'nested_train_loss': ave_loss_train,
        'nested_train_acc': accuracy_train,
        'nested_test_loss': ave_loss_test,
        'nested_acc': accuracy_test
      }

    def create_lc_compression_task(config_):

      if config_['criterion'] == "flops":
        model = add_flops_counting_methods(self.model)
        model.start_flops_count()

        for x, target in self.train_loader:
          _ = model(x[None, 0].cuda())
          break
        uncompressed_flops = model.compute_average_flops_cost()
        print('The number of FLOPS in this model', uncompressed_flops)
        model.stop_flops_count()

      compression_tasks = {}
      for i, (w_get, module) in enumerate([((lambda x=x: getattr(x, 'weight')), x) for x in self.model.modules() if
                                           isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear)]):
        compression_tasks[LCParameterTorch(w_get, self.device)] \
          = (AsIs,
             RankSelection(conv_scheme=config_['conv_scheme'], alpha=config_["alpha"], criterion=config_["criterion"],
                           normalize=True, module=module), f"task_{i}")

      return compression_tasks

    return l_step_optimization, evaluation, create_lc_compression_task

  def finetune_setup(self, tag_of_lc_model, c_step_config):
    exp_run_details = torch.load(f"results/{self.name}_lc_{tag_of_lc_model}.th", map_location="cpu")
    # despite the 's' at the end, there is only one model state, the last
    model_state_to_load = exp_run_details['model_states']
    last_lc_it = exp_run_details['last_step_number']

    model = add_flops_counting_methods(self.model)
    model.start_flops_count()

    for x, target in self.train_loader:
      _ = model(x[None, 0].cuda())
      break
    all_flops = model.compute_average_flops_cost()
    model.stop_flops_count()
    self.model = self.model.cpu()
    all_params = count_params(self.model)

    compression_info = {}
    for task_name, infos in exp_run_details['compression_tasks_info'].items():
      compression_info[task_name] = infos[last_lc_it]
    # compression_infos = exp_run_details['compression_tasks_info'][last_lc_it]

    print(model_state_to_load.keys())

    for key in list(model_state_to_load.keys()):
      if key.startswith("lc_param_list"):
        del model_state_to_load[key]

    del exp_run_details
    import gc
    gc.collect()

    print(model_state_to_load.keys())

    self.model.load_state_dict(model_state_to_load)
    print("model has been sucessfully loaded")

    for i, module in enumerate(
        [x for x in self.model.modules() if isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear)]):
      module.selected_rank_ = compression_info[f"task_{i}"]['selected_rank']
      print(module.selected_rank_)

    reparametrize_low_rank(self.model)
    print(self.model)

    self.model = self.model.to(self.device)
    print("Low rank layers of the model has been successfully reparameterized with sequence of full-rank matrices.")

    def my_forward_eval(x, target):
      out_ = self.model.forward(x)
      return out_, self.model.loss(out_, target)

    self.model.eval()
    accuracy_train, ave_loss_train = compute_acc_loss(my_forward_eval, self.train_loader)
    print('\tBefore finetuning, the train loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss_train, accuracy_train))
    # rec.record('train_nested', [ave_loss, accuracy, training_time, step + 1])
    accuracy_test, ave_loss_test = compute_acc_loss(my_forward_eval, self.test_loader)
    self.model.train()
    print('\tBefore finetuning, the test loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss_test, accuracy_test))

    model = add_flops_counting_methods(self.model)
    model.start_flops_count()

    for x, target in self.train_loader:
      _ = model(x[None, 0].cuda())
      break
    compressed_flops = model.compute_average_flops_cost()
    model.stop_flops_count()
    self.model = self.model.cpu()
    compressed_params = count_params(model)
    self.model = self.model.to(self.device)
    print('The number of FLOPS in original model', all_flops)
    print('The number of params in original model:', all_params)
    print('The number of FLOPS in this model', compressed_flops)
    print('The number of params in this model:', compressed_params)
    flops_rho = all_flops[0] / compressed_flops[0]
    storage_rho = all_params / compressed_params
    print(f'FLOPS ρ={flops_rho:.3f}; STORAGE ρ={storage_rho:.3f};')

    compression_stats = {'original_flops': all_flops, 'compressed_flops': compressed_flops,
                         'flops_rho': flops_rho, 'storage_rho': storage_rho}

    def finetuning(config):
      all_start_time = time.time()
      my_params = filter(lambda p: p.requires_grad, self.model.parameters())
      optimizer = torch.optim.SGD(my_params, config['lr'],
                                  momentum=config['momentum'], nesterov=True)
      epoch_time = AverageMeter()
      lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_decay'])

      train_info = {}
      test_info = {}

      from threading import Thread
      import asyncio

      def start_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

      new_loop = asyncio.new_event_loop()
      t = Thread(target=start_loop, args=(new_loop,))
      t.start()

      for epoch in range(0, config['epochs']):
        start_time = time.time()
        avg_loss_ = AverageMeter()
        for x, target in self.train_loader:
          optimizer.zero_grad()
          x, target = x.cuda(), target.cuda()
          loss = self.model.loss(self.model(x), target)
          loss.backward()
          avg_loss_.update(loss.item())
          optimizer.step()
        end_time = time.time()
        training_time = end_time - all_start_time
        epoch_time.update(end_time - start_time)

        print("Epoch {0} finished in {1.val:.3f}s (avg: {1.avg:.3f}s). Training for {2}"
              .format(epoch, epoch_time, format_time(end_time - all_start_time)))
        print('AVG train loss {0.avg:.6f}'.format(avg_loss_))

        print("\tLR: {:.4e}".format(lr_scheduler.get_lr()[0]))
        lr_scheduler.step()
        if (epoch + 1) % config['print_freq'] == 0:
          self.model.eval()
          accuracy, ave_loss = compute_acc_loss(my_forward_eval, self.train_loader)
          train_info[epoch + 1] = [ave_loss, accuracy, training_time]
          print('\ttrain loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss, accuracy))

          accuracy, ave_loss = compute_acc_loss(my_forward_eval, self.test_loader)
          test_info[epoch + 1] = [ave_loss, accuracy, training_time]
          print('\ttest  loss: {:.6f}, accuracy: {:.4f}'.format(ave_loss, accuracy))
          self.model.train()

          to_save = {}
          to_save['config'] = config
          to_save['optimizer_state'] = optimizer.state_dict()
          to_save['model_state'] = self.model.state_dict()
          to_save['training_time'] = training_time
          to_save['traing_info'] = train_info
          to_save['test_info'] = test_info
          to_save['current_epoch'] = epoch
          to_save['compression_stats'] = compression_stats

          async def actual_save():
            # TODO: make better saves, 1) mv file as backup, 2) save new data 3) delte bk
            torch.save(to_save, f'results/{self.name}_ft_{config["tag"]}.th')
          asyncio.run_coroutine_threadsafe(actual_save(), new_loop)

      async def last_task():
        print("Async file saving has been finished.")
        new_loop.stop()

      asyncio.run_coroutine_threadsafe(last_task(), new_loop)

    return finetuning


class lenet5_all(lenet_all_normalized):
  def __init__(self):
    super(lenet5_all, self).__init__("lenet5_all", lenet5_drop(), 'lenet5_drop')