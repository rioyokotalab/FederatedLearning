from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

from utils import reduce_non_diag

class Server:
    def __init__(self, clients, config):
        self.clients = clients
        self.config = config
        self.num_gpus = torch.cuda.device_count()

    def select_participants(self, num_participants):
        # Randomly select 'num_participants' nodes to participate in the training round
        return sorted(torch.randperm(len(self.clients))[:num_participants])

    def train_client(self, client, inner_loop, gpu_index):
        device = torch.device(f'cuda:{gpu_index}') if torch.cuda.is_available() else torch.device('cpu')
        client.model.to(device)
        client.train(inner_loop)
        if self.num_gpus > self.config.num_participants:
            # If there are more GPUs than participants, we need to move the model to the CPU before returning
            client.model.to(torch.device('cpu'))

    def train_round(self, participants_indices, inner_loop):
        if self.num_gpus > 0:
            # Use a ThreadPool to simulate GPU parallelism
            with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
                for i, idx in enumerate(participants_indices):
                    executor.submit(self.train_client, self.clients[idx], inner_loop, i % self.num_gpus)
        else:
            # If no GPUs are available, train sequentially on CPU
            for idx in participants_indices:
                self.train_client(self.clients[idx], inner_loop, None)

    def aggregate_and_distribute(self, participants_indices):
        aggregated_model = self.aggregate(participants_indices)
        for client in self.clients:
            client.update_model(aggregated_model)

    def aggregate(self, participants_indices):
        # Aggregate the models of the participants
        if self.config.aggregation_algorithm == "fedavg":
            return self.simple_average(participants_indices)
        elif self.config.aggregation_algorithm == "regmean":
            return self.regmean(participants_indices)
    
    def simple_average(self, participants_indices):
        device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        aggregated_model = {}
        state_dicts = [self.clients[idx].model.state_dict() for idx in participants_indices]
        for key in state_dicts[0].keys():
            if (all([key in state_dict for state_dict in state_dicts[1:]]) and
                all([state_dicts[0][key].shape == state_dict[key].shape for state_dict in state_dicts[1:]])):
                aggregated_model[key] = sum([state_dict[key].to(device) for state_dict in state_dicts]) / len(state_dicts)
        return aggregated_model
    
    def regmean(self, participants_indices):
        device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        aggregated_model = {}
        state_dicts = [self.clients[idx].model.state_dict() for idx in participants_indices]
        all_covs = [self.clients[idx].covs for idx in participants_indices]
        for key in state_dicts[0].keys():
            h_avged = False
            valid = (all([key in state_dict for state_dict in state_dicts[1:]]) and
                     all([state_dicts[0][key].shape == state_dict[key].shape for state_dict in state_dicts[1:]]))
            if valid and key.endswith(".weight"):
                module_name = key[:-len(".weight")]
                if module_name in all_covs[0]:
                    cov_m_ws, covs = [], []
                    for i, cov in enumerate(all_covs):
                        param_cov = cov[module_name]
                        if self.config.regmean_reduce_nondiag >= 0:
                            param_cov = reduce_non_diag(param_cov, a=self.config.regmean_reduce_nondiag)
                        param = state_dicts[i][key]
                        cov_m_ws.append(torch.matmul(param_cov, param.transpose(0, 1)).to(device))
                        covs.append(param_cov.to(device))
                    sum_cov = sum(covs)
                    sum_cov_m_ws = sum(cov_m_ws)
                    try:
                        u = torch.linalg.cholesky(sum_cov)
                        sum_cov_inv = torch.cholesky_inverse(u)
                    except:
                        sum_cov_inv = torch.inverse(sum_cov)
                    aggregated_model[key] = torch.matmul(sum_cov_inv, sum_cov_m_ws).transpose(0, 1)
                    h_avged = True
            if not h_avged and valid:
                aggregated_model[key] = sum([state_dict[key].to(device) for state_dict in state_dicts]) / len(state_dicts)
        return aggregated_model
    
    def eval_client(self, client, gpu_index):
        device = torch.device(f'cuda:{gpu_index}') if torch.cuda.is_available() else torch.device('cpu')
        client.model.to(device)
        metrics = client.evaluate()
        if self.num_gpus > self.config.num_participants:
            # If there are more GPUs than participants, we need to move the model to the CPU before returning
            client.model.to(torch.device('cpu'))
        return metrics

    def evaluate(self):
        metrics = [{} for _ in range(len(self.clients))]
        if self.num_gpus > 0:
            # Use a ThreadPool to simulate GPU parallelism
            with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
                futures = {executor.submit(self.eval_client, client, i % self.num_gpus): client.id for i, client in enumerate(self.clients)}
                for future in as_completed(futures):
                    metrics[futures[future]] = future.result()
        else:
            # If no GPUs are available, train sequentially on CPU
            for client in self.clients:
                metrics[client.id] = self.eval_client(client, None)
        if "train_loss" in metrics[0]:
            ret = {"train_loss_avg": 0.0}
        else:
            ret = {}
        for i, metric in enumerate(metrics):
            for key, value in metric.items():
                ret[f"Client{i+1}/{key}"] = value
            if "val_key_score" in metric:
                if "val_key_score_avg" not in ret:
                    ret["val_key_score_avg"] = metric["val_key_score"] / len(metrics)
                else:
                    ret["val_key_score_avg"] += metric["val_key_score"] / len(metrics)
            if "test_key_score" in metric:
                if "test_key_score_avg" not in ret:
                    ret["test_key_score_avg"] = metric["test_key_score"] / len(metrics)
                else:
                    ret["test_key_score_avg"] += metric["test_key_score"] / len(metrics)
            if "train_loss" in metric:
                ret["train_loss_avg"] += metric["train_loss"].item() / len(metrics)
            if "val_loss" in metric:
                if "val_loss_avg" not in ret:
                    ret["val_loss_avg"] = metric["val_loss"].item() / len(metrics)
                else:
                    ret["val_loss_avg"] += metric["val_loss"].item() / len(metrics)
            if "test_loss" in metric:
                if "test_loss_avg" not in ret:
                    ret["test_loss_avg"] = metric["test_loss"].item() / len(metrics)
                else:
                    ret["test_loss_avg"] += metric["test_loss"].item() / len(metrics)
        return ret
