
""" argparse configuration""" 

import torch
import os
import argparse


def get_args():
    """Get all the args"""
    parser = argparse.ArgumentParser(description="CS546 Project: Trasnformer Generalization to Arbitary Context Lengths")
    parser.add_argument(
            "--net",
            type=str,
            choices=['tf', 'cnn', 'gru', 'lstm', 'xlnet', 'ibert', 'ibertpos', 'noor', 'lan', 'ibert2', 'bruno', 'nam'],
            default='tf',
            help='network choices')
    parser.add_argument(
            "--epochs",
            type=int,
            default='50',
            help='number of epochs')
    parser.add_argument(
            "--train_size",
            type=int,
            default='25600',
            help='number of training examples per epoch')
    parser.add_argument(
            "--validation_size",
            type=int,
            default='1536',
            help='number of validation examples')
    parser.add_argument(
            "--batch_size",
            type=int,
            default='64',
            help='batch size')
    parser.add_argument(
            "--model_size",
            type=int,
            default='256',
            help='internal channel dimension')
    parser.add_argument(
            "--num_heads",
            type=int,
            default='4',
            help='number of heads in TF-based models')
    parser.add_argument(
            "--digits",
            type=int,
            default='12',
            help='Max number of digits')
    parser.add_argument(
            "--num_layers",
            type=int,
            default='4',
            help='Number of layers in the model')
    parser.add_argument(
            "--seq_type",
            type=str,
            choices= ['fib', 'arith', 'palin', 'copy', 'ptbc', 'ptbw'],
            default='fib',
            help='fib: fibonacci / arith: arithmetic / palin: palindrome / ptbc: ptb char / ptbw: ptb word')
    parser.add_argument(
            "--lr",
            type=float,
            default=1e-4,
            help='Default learning rate')
    parser.add_argument(
            "--log",
            type=str,
            choices= ['true', 'false'],
            default='false',
            help='Save result to file')
    parser.add_argument(
            "--exp",
            type=int,
            default=0,
            help='Experiment number')
    return parser.parse_args()

