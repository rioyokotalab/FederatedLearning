from collections.abc import Mapping
import torch
from data_manager import get_dm_class
from model import create_model  # Utility function to create a model
from utils import get_optimizer, get_scheduler, seed_worker  # Utility functions

class Client:
    def __init__(self, id, config, model=None):
        self.id = id
        self.config = config
        if len(config.dataset) == 1:
            self.dataset_name = config.dataset[0]
        else:
            # If multiple datasets are specified, assign one task to each client
            assert len(config.dataset) == config.num_clients
            self.dataset_name = config.dataset[self.id]
        self.state = dict(step=0)
        self.latest_loss = None
        if self.config.aggregation_algorithm == "regmean":
            self.covs = {}
            self.handles = []

        # Set up the dataset manager for the client
        self.dm = get_dm_class(self.dataset_name)(self.config)
        self.train_dataset, self.eval_dataset, self.test_dataset = self.dm.load_dataset(self.dataset_name)

        # Set up the model for the client
        self.model = model
        if self.model is None:
            self.model = self.initialize_model()

        # Set up the optimizer
        self.optimizer = get_optimizer(self.model.parameters(), config)

        # Set up the learning rate scheduler if specified
        self.scheduler = get_scheduler(self.optimizer, config) if config.lr_scheduler else None

    def initialize_model(self):
        # Initialize the model
        model = create_model(self.config, self.dm.num_labels)
        return model

    def train(self, inner_loop):
        # Training logic for the node
        self.model.train()
        g = torch.Generator()
        g.manual_seed(self.config.seed)
        data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True,
                                                  collate_fn=self.dm.collate_fn, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        running_loss = 0.0
        
        for inputs in data_loader:
            inputs = self._prepare_inputs(inputs)

            # Perform any startup operations before the forward pass
            self._startup()

            # Forward pass
            outputs = self.model(**inputs)
            loss = outputs.loss
            running_loss += loss

            # Perform any teardown operations after the forward pass
            self._teardown()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients if specified
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            # Update parameters
            self.optimizer.step()

            # Update learning rate if scheduler is set
            if self.scheduler:
                self.scheduler.step()
            
            self.state["step"] += 1
            if self.state["step"] % inner_loop == 0:
                self.latest_loss = running_loss / inner_loop
                break

        # Return the state of the model and the loss for logging
        return self.model.state_dict(), loss.item()

    def _startup(self):
        # Perform any startup operations before the forward pass
        if self.config.aggregation_algorithm == "fedavg":
            pass
        elif self.config.aggregation_algorithm == "regmean":
            self._regmean_startup()
    
    def _regmean_startup(self):
        # Perform any startup operations before the forward pass for regmean
        current_step = self.state["step"]
        do_extend = ((self.config.regmean_cov_interval is not None and 
                      ((self.config.regmean_cov_interval == 1 and current_step == 0) or 
                       (self.config.regmean_cov_interval != 1 and (current_step + 1) % self.config.regmean_cov_interval == 0))) or
                       (self.config.regmean_cov_interval is None and
                        ((self.config.regmean_update_before_aggregate >= self.config.inner_loop and current_step == 0) or
                         (self.config.regmean_update_before_aggregate < self.config.inner_loop and
                          self.config.inner_loop - (current_step % self.config.inner_loop) == self.config.regmean_update_before_aggregate))))
        if do_extend:
            linear_modules = {}
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    linear_modules[name] = module
            for name, module in linear_modules.items():
                handle = module.register_forward_hook(self.get_grams(name))
                self.handles.append(handle)

    def get_grams(self, name):
        def hook(module, input, output):
            """
            Note: adhere to signature of hook functions
            """
            x = input[0].detach()  # $[b,t,h]
            x = x.view(-1, x.size(-1))
            xtx = torch.matmul(x.transpose(0, 1), x)  # [h,h]
            if name not in self.covs:
                self.covs[name] = xtx / x.size(0)
            else:
                self.covs[name] = self.covs[name] * self.config.regmean_ema_decay + xtx / x.size(0) * (1 - self.config.regmean_ema_decay)

        return hook
    
    def _teardown(self):
        # Perform any teardown operations after the forward pass
        if self.config.aggregation_algorithm == "fedavg":
            return
        elif self.config.aggregation_algorithm == "regmean":
            self._regmean_teardown()
    
    def _regmean_teardown(self):
        # Perform any teardown operations after the forward pass for regmean
        current_step = self.state["step"]
        remove_handles = ((self.config.regmean_cov_interval is not None and
                           ((self.config.regmean_cov_interval == 1 and
                             (current_step + 1) == self.config.communication_rounds * self.config.inner_loop) or
                             (self.config.regmean_cov_interval != 1 and len(self.handles) > 0))) or
                             (self.config.regmean_cov_interval is None and
                              ((self.config.regmean_update_before_aggregate >= self.config.inner_loop and
                                (current_step + 1) == self.config.communication_rounds * self.config.inner_loop) or
                                (self.config.regmean_update_before_aggregate < self.config.inner_loop and
                                 (current_step + 1) % self.config.inner_loop == 0))))
        if remove_handles:
            for handle in self.handles:
                handle.remove()
            self.handles = []

    def _prepare_input(self, data):
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            return data.to(next(self.model.parameters()).device)
        return data
    
    def _prepare_inputs(self, inputs):
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError("The batch received was empty, your model won't be able to train on it.")
        if self.config.model == "roberta-base":
            new_inputs = {}
            for k in ["input_ids", "attention_mask", "labels", "dataset"]:
                if k in inputs:
                    new_inputs[k] = inputs[k]
            return new_inputs
        else:
            return inputs

    def update_model(self, new_model):
        # Update the model parameters with the aggregated parameters
        for n, p in self.model.named_parameters():
            if n in new_model:
                p.data.copy_(new_model[n].data)

    def evaluate(self):
        # Evaluation logic for the client
        self.model.eval()
        metrics = {}
        if self.eval_dataset is not None and ({"label", "labels", "labels_ids"} & set(self.eval_dataset.features)):
            dataset = self.eval_dataset
            prefix = "val_"
        elif self.test_dataset is not None and ({"label", "labels", "labels_ids"} & set(self.test_dataset.features)):
            dataset = self.test_dataset
            prefix = "test_"
        else:
            print(f"Warning: no evaluation dataset found for client {self.id}")
            dataset = None
        if dataset is not None:
            g = torch.Generator()
            g.manual_seed(self.config.seed)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.config.eval_batch_size,
                                                      collate_fn=self.dm.collate_fn, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        
            total_loss = 0.0
            num_data = 0
            all_preds = None
            all_labels = None
            with torch.no_grad():
                for inputs in data_loader:
                    inputs = self._prepare_inputs(inputs)
                    outputs = self.model(**inputs)
                    total_loss += outputs.loss * inputs["labels"].numel()
                    num_data += inputs["labels"].numel()
                    if all_preds is None:
                        all_preds = outputs.logits
                    else:
                        all_preds = torch.cat((all_preds, outputs.logits))
                    if all_labels is None:
                        all_labels = inputs["labels"]
                    else:
                        all_labels = torch.cat((all_labels, inputs["labels"]))
            metrics.update(self.dm.compute_metrics(self.dataset_name, all_preds, all_labels))
            for k in list(metrics):
                metrics[prefix+k] = metrics[k]
                del metrics[k]
            metrics[prefix+"loss"] = total_loss / num_data
        if self.latest_loss is not None:
            metrics["train_loss"] = self.latest_loss
        
        # Return metrics for logging
        return metrics
