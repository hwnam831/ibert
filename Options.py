
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
            choices=['tf', 'cnn', 'nam', 'noor', 'lan', 'vikram', 'bruno'],
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
            default='256',
            help='batch size')
    parser.add_argument(
            "--model_size",
            type=int,
            default='512',
            help='internal channel dimension')
    parser.add_argument(
            "--digits",
            type=int,
            default='5',
            help='Max number of digits')
    parser.add_argument(
            "--seq_type",
            type=str,
            choices= ['fib', 'arith', 'palin'],
            default='fib',
            help='fib: fibonacci / arith: arithmetic / palin: palindrome')
    parser.add_argument(
            "--lr",
            type=float,
            default=1e-4,
            help='Default learning rate')

    return parser.parse_args()

