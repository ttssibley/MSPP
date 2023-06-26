import os.path
import torch
from copy import deepcopy
import pickle

from args import Arguments
from datasets import get_dataloaders
from query import QuerySelector
from train import train
from eval import evaluate
from UNet import UNetVgg16


def pixelpick_train(args):
    n_rounds = args.max_pixels // args.n_pixels_per_round
    train_loader, _, _ = get_dataloaders(args, queries=None, is_train=False)
    # initialize queries
    qs = QuerySelector(args, train_loader)
    queries = qs.gen_init_queries()
    args_query0 = deepcopy(args)
    args_query0.experim_name = 'query_0'
    args_query0.checkpoints_dir = f"{args.root}/checkpoints/{args.dataset}/{args_query0.experim_name}"
    Arguments.update_checkpoints_dir(args_query0, args_query0.checkpoints_dir)
    with open(args_query0.query_file_path, 'wb') as f:
        pickle.dump(queries, f)
    n_queried_pixels = args.n_init_pixels_per_class * args.n_classes
    print(f"Initialization: n_pixels_queried = {n_queried_pixels}")
    # training for round 0
    if not os.path.exists(f"{args_query0.checkpoints_dir}/model.pth"):
        os.makedirs(args_query0.checkpoints_dir, exist_ok=True)
        train(args_query0, queries)
    else:
        print("Model already exists. Skip training.")
    evaluate(args_query0, 'val', save_pred=False)
    evaluate(args_query0, 'test', save_pred=False)

    model = UNetVgg16(n_classes=args.n_classes).to(args.device)
    curr_args = args_query0
    for i in range(n_rounds):
        print(f"Round {i+1}/{n_rounds}: n_pixels_queried = {n_queried_pixels + (i+1)*args.n_pixels_per_round}")
        model.load_state_dict(torch.load(curr_args.model_path)['model_state_dict'])
        queries = qs(model)
        curr_args = deepcopy(args)
        Arguments.update_checkpoints_dir(curr_args, f"{args.checkpoints_dir}/query_{i+1}")
        with open(curr_args.query_file_path, 'wb') as f:
            pickle.dump(queries, f)
        train(curr_args, queries, model)
        evaluate(curr_args, 'val', save_pred=False)
        evaluate(curr_args, 'test', save_pred=False)


def pixelpick_eval(args, mode='val'):
    n_rounds = args.max_pixels // args.n_pixels_per_round
    args_query0 = deepcopy(args)
    args_query0.experim_name = 'query_0'
    args_query0.checkpoints_dir = f"{args.root}/checkpoints/{args.dataset}/{args_query0.experim_name}"
    Arguments.update_checkpoints_dir(args_query0, args_query0.checkpoints_dir)
    scores = evaluate(args_query0, mode, save_pred=True)
    score_track = [scores]

    curr_args = args_query0
    for i in range(n_rounds):
        Arguments.update_checkpoints_dir(curr_args, f"{args.checkpoints_dir}/query_{i + 1}")
        scores = evaluate(curr_args, mode, save_pred=True)
        score_track.append(scores)
    return score_track


if __name__ == '__main__':
    arg_parser = Arguments()
    args = arg_parser.parse_args(verbose=True)
    if args.debug:
        args.max_pixels = 20
        args.n_pixels_per_round = 10
        args.n_train_iters = 20
        args.val_interval = 10

    pixelpick_train(args)
