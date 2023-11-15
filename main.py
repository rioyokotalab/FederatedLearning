import argparse
import os
import time
import wandb

from client import Client
from server import Server
from utils import set_random_seeds


def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning System')

    parser.add_argument('--num_clients', type=int, required=True, help='Total number of clients (nodes)')
    parser.add_argument('--num_participants', type=int, required=True, help='Number of clients participating in a communication round')
    parser.add_argument('--communication_rounds', type=int, required=True, help='Number of communications to perform')
    parser.add_argument('--inner_loop', type=int, required=True, help='Number of steps between communications')
    parser.add_argument('--model', type=str, required=True, choices=["roberta-base"], help='Model architecture')
    parser.add_argument('--dataset', nargs='+', type=str, required=True, help='Datasets to train on')
    parser.add_argument('--optimizer', type=str, required=True, help='Optimizer')
    parser.add_argument('--aggregation_algorithm', type=str, required=True, help='Aggregation algorithm')
    parser.add_argument('--regmean_cov_interval', type=int, help='Interval for computing the covs in regmean')
    parser.add_argument('--regmean_update_before_aggregate', type=int, default=10,
                        help='Number of update steps immidiately before the aggregation for regmean')
    parser.add_argument('--regmean_ema_decay', type=float, default=0.95, help='EMA decay for regmean')
    parser.add_argument('--regmean_reduce_nondiag', type=float, default=0.9, help='EMA decay for regmean')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--eval_interval', type=int, default=1, help='Interval for evaluation')
    parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--lr_scheduler', type=str, help='Learning rate scheduler')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm')
    parser.add_argument('--warmup_steps', type=int, help='Warmup steps')
    parser.add_argument('--warmup_ratio', type=float, help='Warmup ratio')
    parser.add_argument('--momentum', type=float, default=0.0, help='Momentum')
    parser.add_argument('--dampening', type=float, default=0.0, help='Dampening')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--nesterov', type=bool, default=True, action=argparse.BooleanOptionalAction, help='Nesterov momentum')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--adam_eps', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--adam_amsgrad', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Adam amsgrad')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')
    parser.add_argument('--enable_full_determinism', action='store_true', help='Make operations deterministic')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable Weights & Biases')

    args = parser.parse_args()
    if args.nesterov and (args.momentum <= 0.0 or args.dampening != 0.0):
        args.nesterov = False
    return args

def main():
    args = parse_args()
    
    # Initialize Weights & Biases
    project = os.getenv("WANDB_PROJECT", None)
    entity = os.getenv("WANDB_ENTITY", None)
    wandb.init(project=project, entity=entity, config=args, mode="disabled" if args.disable_wandb else None)

    # Set random seeds
    set_random_seeds(args.seed, args.enable_full_determinism)

    # Create clients
    clients = [Client(id=i, config=args) for i in range(args.num_clients)]

    # Create the server
    server = Server(clients, args)

    # Start the timer
    start = time.perf_counter()

    # Evaluate the initial model
    metrics = server.evaluate()
    metrics["communication_round"] = 0
    s = "[0]"
    if "val_key_score_avg" in metrics:
        s += f" val_key_score_avg: {metrics['val_key_score_avg']:.4f}, val_loss_avg: {metrics['val_loss_avg']:.4f}"
    if "test_key_score_avg" in metrics:
        s += f", test_key_score_avg: {metrics['test_key_score_avg']:.4f}, test_loss_avg: {metrics['test_loss_avg']:.4f}"
    s += f", time: {time.perf_counter() - start:.2f}s, {metrics=}"
    print(s)
    wandb.log(metrics)
    start = time.perf_counter()
    # Start the training process
    for communication_round in range(args.communication_rounds):
        participants = server.select_participants(args.num_participants)
        server.train_round(participants, args.inner_loop)
        server.aggregate_and_distribute(participants)
        
        # Log the metrics after each evaluation interval
        if (communication_round + 1) % args.eval_interval == 0 or communication_round + 1 == args.communication_rounds:
            metrics = server.evaluate()
            metrics["communication_round"] = communication_round + 1
            s = f"[{communication_round+1}] train_loss_avg: {metrics['train_loss_avg']:.4f}"
            if "val_key_score_avg" in metrics:
                s += f", val_key_score_avg: {metrics['val_key_score_avg']:.4f}, val_loss_avg: {metrics['val_loss_avg']:.4f}"
            if "test_key_score_avg" in metrics:
                s += f", test_key_score_avg: {metrics['test_key_score_avg']:.4f}, test_loss_avg: {metrics['test_loss_avg']:.4f}"
            s += f", time: {time.perf_counter() - start:.2f}s, {metrics=}"
            print(s)
            wandb.log(metrics)
            start = time.perf_counter()
    
    wandb.finish()

if __name__ == '__main__':
    main()
