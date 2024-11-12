from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import time
import json

import numpy as np
import torch
import torch.nn.functional as F
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from ate_asc_modeling import BertForSequenceLabeling
from optimization import BertAdam    # , CustomAdamW
from tokenization import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from ate_asc_features import ATEASCProcessor, convert_examples_to_features, get_labels
from utils import get_logger, get_aspect_chunks, get_polaity_chunks

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return max(0, (1.0 - x) / (1.0 - warmup))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_input_parameter():
    global logger
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, bert-large-uncased.")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--task_name", default="ate_asc", type=str, required=False, help="The name of the task to train.")
    parser.add_argument("--data_name", default="", type=str, required=False, help="The name of the task to train.")
    parser.add_argument("--train_file", default=None, type=str, required=False)
    parser.add_argument("--valid_file", default=None, type=str, required=False)
    parser.add_argument("--test_file", default=None, type=str, required=False)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \nSequences longer than this will be truncated, and sequences shorter \nthan this will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--num_thread_reader', type=int, default=0, help='')
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n0 (default value): dynamic loss scaling.\nPositive power of 2: static loss scaling value.\n")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. A number of warnings are expected for a normal CoQA evaluation.")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.");
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    parser.add_argument("--use_ghl", action='store_true', help="Whether use weighted cross entropy to decoder.")
    parser.add_argument("--use_vat", action='store_true', help="Whether use vat to encoder.")

    parser.add_argument("--use_decoder", default=True, type=str2bool, help="Whether use decoder to asc.")
    parser.add_argument("--num_decoder_layer", default=2, type=int, help="When `use_decoder' is True, set the number of decoder.")
    parser.add_argument("--decoder_shared_layer", default=3, type=int, help="When `use_decoder' is True, set the number of shared encoder.")

    # Christina: added training step number as argument
    parser.add_argument("--training_step", type=int, default=0, help="which of the three training steps are being executed")
    # christina
    parser.add_argument("--training_name", type=str, default="", help="a name for the training for logging")
    # christina
    parser.add_argument("--classification_head", type=str, default="", help="a name for the classification_head for logging")

    args = parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  {}: {}".format(key, args.__dict__[key]))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    task_config = {
        "use_ghl": args.use_ghl,
        "use_vat": args.use_vat,
        "num_decoder_layer": args.num_decoder_layer,
        "decoder_shared_layer": args.decoder_shared_layer,
    }

    return args, task_config

def init_device(args):
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    return device, n_gpu


def init_model(args, num_tp_labels, task_config, device, n_gpu):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
        if args.num_decoder_layer != 0:
            layer_num_list = [layer_num for layer_num in range(args.num_decoder_layer)]
            new_model_state_dict = {}
            model_state_dict_exsit_keys = model_state_dict.keys()
            max_bert_layer = max([int(k_str.split(".")[3]) for k_str in model_state_dict_exsit_keys if "bert.encoder.layer" in k_str])
            for k_str in model_state_dict_exsit_keys:
                new_model_state_dict[k_str] = model_state_dict[k_str]
                for layer_num in layer_num_list:
                    bert_key_name = "bert.encoder.layer.{}".format(max_bert_layer - args.num_decoder_layer + 1 + layer_num)
                    mirror_key_name = "bert.encoder.layer.{}".format(layer_num)
                    if k_str.find(bert_key_name) == 0:
                        new_key_name = k_str.replace(bert_key_name, mirror_key_name).replace("bert.encoder", "decoder.decoder")
                        if "attention.self" in new_key_name:
                            new_key_name_sufx = new_key_name.replace("attention.self", "slf_attn.att")
                            new_model_state_dict[new_key_name_sufx] = model_state_dict[k_str].clone()
                            new_key_name_sufx = new_key_name.replace("attention.self", "enc_attn.att")
                            new_model_state_dict[new_key_name_sufx] = model_state_dict[k_str].clone()
                        elif "attention.output" in new_key_name:
                            new_key_name_sufx = new_key_name.replace("attention.output", "slf_attn.output")
                            new_model_state_dict[new_key_name_sufx] = model_state_dict[k_str].clone()
                            new_key_name_sufx = new_key_name.replace("attention.output", "enc_attn.output")
                            new_model_state_dict[new_key_name_sufx] = model_state_dict[k_str].clone()
                        else:
                            new_model_state_dict[new_key_name] = model_state_dict[k_str].clone()

                if k_str.find("bert.embeddings") == 0:
                    new_key_name = k_str.replace("bert.embeddings", "decoder.embeddings")
                    new_model_state_dict[new_key_name] = model_state_dict[k_str].clone()

            model_state_dict = new_model_state_dict
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else \
        os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = BertForSequenceLabeling.from_pretrained(args.bert_model, cache_dir=cache_dir, state_dict=model_state_dict,
                                                    num_tp_labels=num_tp_labels, task_config=task_config)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    return model


def prep_optimizer(args, model, num_train_optimization_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    if args.use_decoder:
        no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
        decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

        no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." in n]
        no_decay_nobert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." not in n]

        decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." in n]
        decay_nobert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." not in n]

        coef_lr = 10.
        optimizer_grouped_parameters = [
            {'params': [p for n, p in no_decay_bert_param_tp], 'weight_decay': 0.01},
            {'params': [p for n, p in no_decay_nobert_param_tp], 'weight_decay': 0.01, 'lr': args.learning_rate * coef_lr},
            {'params': [p for n, p in decay_bert_param_tp], 'weight_decay': 0.0},
            {'params': [p for n, p in decay_nobert_param_tp], 'weight_decay': 0.0, 'lr': args.learning_rate * coef_lr}
        ]
    else:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters, lr=args.learning_rate, bias_correction=False, max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
        # optimizer = CustomAdamW(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup_proportion,
        #                     t_total=num_train_optimization_steps)
    return optimizer


def dataloader_train(args, tokenizer, file_path):
    dataset = ATEASCProcessor(file_path=file_path, set_type="train")
    logger.info("Loaded train file: {}".format(file_path))
    at_labels, as_labels = get_labels(dataset.label_tp_list)

    features = convert_examples_to_features(dataset.examples, (at_labels, as_labels),
                                            args.max_seq_length, tokenizer, verbose_logging=args.verbose_logging)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_at_label_ids = torch.tensor([f.at_label_id for f in features], dtype=torch.long)
    all_as_label_ids = torch.tensor([f.as_label_id for f in features], dtype=torch.long)

    all_label_mask = torch.tensor([f.label_mask for f in features], dtype=torch.long)
    all_label_mask_X = torch.tensor([f.label_mask_X for f in features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_at_label_ids, all_as_label_ids,
                               all_label_mask, all_label_mask_X)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.num_thread_reader)

    return dataloader, train_data, (at_labels, as_labels)


def dataloader_val(args, tokenizer, file_path, label_tp_list, set_type="val"):

    dataset = ATEASCProcessor(file_path=file_path, set_type=set_type)
    logger.info("Loaded val file: {}".format(file_path))

    eval_features = convert_examples_to_features(dataset.examples, label_tp_list,
                                                 args.max_seq_length, tokenizer, verbose_logging=args.verbose_logging)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_at_label_ids = torch.tensor([f.at_label_id for f in eval_features], dtype=torch.long)
    all_as_label_ids = torch.tensor([f.as_label_id for f in eval_features], dtype=torch.long)

    all_label_mask = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
    all_label_mask_X = torch.tensor([f.label_mask_X for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_at_label_ids, all_as_label_ids,
                              all_label_mask, all_label_mask_X)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    return eval_dataloader, eval_data


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, tokenizer, optimizer, global_step, num_train_optimization_steps):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = 100
    start_time = time.time()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    weight_gradient = None  # Init in model: [bin_num]
    weight_gradient_labels = None  # Init in model:
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, at_label_ids, as_label_ids, label_mask, label_mask_X = batch
        loss, acc_sum, weight_gradient, weight_gradient_labels = model(input_ids, segment_ids, input_mask, label_mask_X,
                                                                       at_label_ids, as_label_ids,
                                                                       weight_gradient, weight_gradient_labels)
        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        tr_loss += float(loss.item())
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                # modify learning rate with special warm up BERT uses
                # if args.fp16 is False, BertAdam is used that handles this automatically
                lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % log_step == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.num_train_epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss.item()), (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    tr_loss = tr_loss / len(train_dataloader)
    return tr_loss, global_step


def cal_f1_at(y_true, y_pred):
    correct_pred, total_ground, total_pred, total_correct = 0., 0., 0., 0.
    total_predictions = 0  # To keep track of total predictions for accuracy    
    lab_chunks_list, lab_pred_chunks_list = [], []

    for ground_seq, pred_seq in zip(y_true, y_pred):
        lab_chunks = get_aspect_chunks(ground_seq, default="O")
        lab_pred_chunks = get_aspect_chunks(pred_seq, default="O")
        lab_chunks_list.append(lab_chunks)
        lab_pred_chunks_list.append(lab_pred_chunks)

        lab_chunks = set(lab_chunks)
        lab_pred_chunks = set(lab_pred_chunks)

        correct_pred += len(lab_chunks & lab_pred_chunks)
        total_pred += len(lab_pred_chunks)
        total_ground += len(lab_chunks)

        # Increment total_correct for every correct prediction in the sequence
        total_correct += sum(g == p for g, p in zip(ground_seq, pred_seq))
        total_predictions += len(ground_seq)  # Total number of tokens in the sequence

    p = correct_pred / total_pred if total_pred > 0 else 0.
    r = correct_pred / total_ground if total_ground > 0 else 0.
    f1 = 2 * p * r / (p + r) if p > 0 and r > 0 else 0.

    # Calculate accuracy
    accuracy = total_correct / total_predictions if total_predictions > 0 else 0.

    return p, r, f1, accuracy, lab_chunks_list, lab_pred_chunks_list


def cal_f1_as(y_true, y_pred, at_lab_chunks_list, at_lab_pred_chunks_list, must_predict=False):
    correct_pred, total_ground, total_pred, total_correct_aspect_sentiments = 0., 0., 0., 0.
    total_aspects = 0  # To keep track of the total number of aspect terms for accuracy calculation
    lab_chunks_list, lab_pred_chunks_list = [], []
    
    for ground_seq, pred_seq, at_lab_chunks, at_lab_pred_chunks in zip(y_true, y_pred, at_lab_chunks_list, at_lab_pred_chunks_list):
        lab_chunks = get_polaity_chunks(ground_seq, at_lab_chunks, default="O", must_predict=must_predict)
        lab_pred_chunks = get_polaity_chunks(pred_seq, at_lab_pred_chunks, default="O", must_predict=must_predict)
        lab_chunks_list.append(lab_chunks)
        lab_pred_chunks_list.append(lab_pred_chunks)

        lab_chunks = set(lab_chunks)
        lab_pred_chunks = set(lab_pred_chunks)

        correct_pred += len(lab_chunks & lab_pred_chunks)
        total_pred += len(lab_pred_chunks)
        total_ground += len(lab_chunks)

        # For accuracy, only consider the correctly identified aspects (intersection of lab_chunks and lab_pred_chunks)
        correct_aspects = lab_chunks & lab_pred_chunks

        # CALCULATION FOR ACCURACY ISN'T WORKING CORRECTLY - WILL NEED REVISION IN THE FUTURE!!! 
        # Increment total_correct_aspect_sentiments for every correct sentiment prediction for the identified aspect terms
        for aspect in correct_aspects:
            # Aspect is structured as (sentiment_label, start_index, end_index)
            _, start_index, end_index = aspect
            if end_index < len(ground_seq):  # Ensure end_index is within bounds
                aspect_ground_tokens = tuple([ground_seq[index] for index in range(start_index, end_index + 1)])
                aspect_pred_tokens = tuple([pred_seq[index] for index in range(start_index, end_index + 1)])
                if aspect_ground_tokens == aspect_pred_tokens:
                    total_correct_aspect_sentiments += 1
            # else:
            #     logger.warning(f"Aspect term indices out of range: {aspect}")


        total_aspects += len(correct_aspects)  # Count the number of correctly identified aspect terms

    p = correct_pred / total_pred if total_pred > 0 else 0.
    r = correct_pred / total_ground if total_ground > 0 else 0.
    f1 = 2 * p * r / (p + r) if p > 0 and r > 0 else 0.

    # Calculate accuracy specifically for the sentiment of identified aspect terms
    accuracy = total_correct_aspect_sentiments / total_aspects if total_aspects > 0 else 0.

    return p, r, f1, accuracy, lab_chunks_list, lab_pred_chunks_list


# def cal_f1_as(y_true, y_pred, at_lab_chunks_list, at_lab_pred_chunks_list, must_predict=False):
#     correct_pred, total_ground, total_pred, total_correct = 0., 0., 0., 0.
#     total_predictions = 0  # To keep track of total predictions for accuracy    
#     lab_chunks_list, lab_pred_chunks_list = [], []
#     for ground_seq, pred_seq, lab_chunks, lab_pred_chunks in zip(y_true, y_pred, at_lab_chunks_list, at_lab_pred_chunks_list):

#         lab_chunks = get_polaity_chunks(ground_seq, lab_chunks, default="O", must_predict=must_predict)
#         lab_pred_chunks = get_polaity_chunks(pred_seq, lab_pred_chunks, default="O", must_predict=must_predict)
#         lab_chunks_list.append(lab_chunks)
#         lab_pred_chunks_list.append(lab_pred_chunks)

#         lab_chunks = set(lab_chunks)
#         lab_pred_chunks = set(lab_pred_chunks)

#         correct_pred += len(lab_chunks & lab_pred_chunks)
#         total_pred += len(lab_pred_chunks)
#         total_ground += len(lab_chunks)

#         # Increment total_correct for every correct prediction in the sequence
#         total_correct += sum(g == p for g, p in zip(ground_seq, pred_seq))
#         total_predictions += len(ground_seq)  # Total number of tokens in the sequence

#     p = correct_pred / total_pred if total_pred > 0 else 0.
#     r = correct_pred / total_ground if total_ground > 0 else 0.
#     f1 = 2 * p * r / (p + r) if p > 0 and r > 0 else 0.

#     # Calculate accuracy
#     accuracy = total_correct / total_predictions if total_predictions > 0 else 0.

#     return p, r, f1, accuracy, lab_chunks_list, lab_pred_chunks_list


def eval_epoch(model, eval_dataloader, label_tp_list, device):
    if hasattr(model, 'module'):
        model = model.module
    model.eval()

    y_true_at = []
    y_pred_at = []
    y_true_as = []
    y_pred_as = []
    at_label_list, as_label_list = label_tp_list
    at_label_map = {i: label for i, label in enumerate(at_label_list)}
    as_label_map = {i: label for i, label in enumerate(as_label_list)}
    for input_ids, input_mask, segment_ids, at_label_ids, as_label_ids, label_mask, label_mask_X in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        at_label_ids = at_label_ids.to(device)
        as_label_ids = as_label_ids.to(device)
        label_mask = label_mask.to(device)
        label_mask_X = label_mask_X.to(device)

        with torch.no_grad():
            # logits, decoder_logits = model(input_ids, segment_ids, input_mask)
            logits, sequence_output, encoder_output = model.get_encoder_logits(input_ids, segment_ids, input_mask)
            pred_dec_ids = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            decoder_logits = model.get_decoder_logits(encoder_output, input_mask, label_mask_X, pred_dec_ids)
            logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            decoder_logits = torch.argmax(F.log_softmax(decoder_logits, dim=2), dim=2)
            logits = logits.detach().cpu().numpy()
            decoder_logits = decoder_logits.detach().cpu().numpy()

        at_label_ids = at_label_ids.to('cpu').numpy()
        as_label_ids = as_label_ids.to('cpu').numpy()
        label_mask = label_mask.to('cpu').numpy()

        for i, mask_i in enumerate(label_mask):
            temp_11 = []
            temp_12 = []
            temp_21 = []
            temp_22 = []
            for j, l in enumerate(mask_i):
                if l > -1:
                    temp_11.append(at_label_map[at_label_ids[i][j]])
                    temp_12.append(at_label_map[logits[i][j]])
                    temp_21.append(as_label_map[as_label_ids[i][j]])
                    temp_22.append(as_label_map[decoder_logits[i][j]])

            y_true_at.append(temp_11)
            y_pred_at.append(temp_12)
            y_true_as.append(temp_21)
            y_pred_as.append(temp_22)

    p, r, f1, a, at_lab_chunks_list, at_lab_pred_chunks_list = cal_f1_at(y_true_at, y_pred_at)
    # logger.info("AT a:{:.4f}\tAT p:{:.4f}\tr:{:.4f}\tf1:{:.4f}".format(a, p, r, f1))

    as_lab_p, as_lab_r, as_lab_f1, as_lab_a, _, _ = cal_f1_as(y_true_as, y_pred_as, at_lab_chunks_list, at_lab_pred_chunks_list)
    # logger.info("AS a:{:.4f{\tAS p:{:.4f}\tr:{:.4f}\tf1:{:.4f}".format(as_lab_a, as_lab_p, as_lab_r, as_lab_f1))

    return p, r, f1, a, as_lab_p, as_lab_r, as_lab_f1, as_lab_a


def save_model(epoch, args, model):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}".format(epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file


def load_model(epoch, args, num_tp_labels, task_config, device):
    model_file = os.path.join(
        args.output_dir,
        "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        logger.info("Model loaded from %s", model_file)
        model = BertForSequenceLabeling.from_pretrained(args.bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
                                                        state_dict=model_state_dict, num_tp_labels=num_tp_labels,
                                                        task_config=task_config)
        model.to(device)
    else:
        model = None
    return model


# Christina
def initialize_or_update_structure(global_settings, log_file='training_performance.json'):
    # Check for 'name', 'bert_model', and 'data_name' in global_settings
    check_fields = ['name', 'bert_model', 'data_name', 'classification_head']
    simplified_global_settings = {k: global_settings[k] for k in check_fields if k in global_settings}

    # If the log file already exists, we need to check for existing entries
    if os.path.isfile(log_file):
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        # Convert settings to a comparable format (e.g., a sorted tuple of items)
        simplified_settings_tuple = tuple(sorted(simplified_global_settings.items()))
        
        # Check if the simplified global settings already exist in the file
        existing_entry_index = None
        for i, entry in enumerate(data):
            existing_simplified_settings_tuple = tuple(sorted(entry["settings"].items()))
            if existing_simplified_settings_tuple == simplified_settings_tuple:
                existing_entry_index = i
                break
        
        # If settings exist, return the index of the existing entry
        if existing_entry_index is not None:
            return existing_entry_index
        else:
            # If settings don't exist, add a new entry
            data.append({"settings": simplified_global_settings, "training_steps": {}})
            with open(log_file, 'w') as f:
                json.dump(data, f, indent=4)
            return len(data) - 1  # Return the index of the new entry
    else:
        # If the file doesn't exist, initialize it with the current global settings
        with open(log_file, 'w') as f:
            json.dump([{"settings": simplified_global_settings, "training_steps": {}}], f, indent=4)
        return 0  # Return the index of the new entry

    
# Christina - added logging of model settings and performance metrics to a file 
def log_metrics(entry_index, training_step, step_settings, epoch, epoch_metrics, log_file='training_performance.json'):
    with open(log_file, 'r') as f:
        data = json.load(f)

    # Access the specific entry based on entry_index
    entry = data[entry_index]

    # If the training step doesn't exist, initialize it with step_settings
    if str(training_step) not in entry["training_steps"]:
        entry["training_steps"][str(training_step)] = step_settings
        entry["training_steps"][str(training_step)]["epochs"] = {}

    # Update the epochs dictionary with the new epoch metrics
    entry["training_steps"][str(training_step)]["epochs"][str(epoch)] = epoch_metrics

    with open(log_file, 'w') as f:
        json.dump(data, f, indent=4)


# Christina - added test result logging
def log_test_metrics(entry_index, training_step, test_metrics, log_file='training_performance.json'):
    with open(log_file, 'r') as f:
        data = json.load(f)

    # Access the specific entry based on entry_index
    entry = data[entry_index]

    # Ensure the training step exists
    if str(training_step) not in entry["training_steps"]:
        print(f"Training step {training_step} not found. Ensure training metrics are logged before logging test metrics.")
        return

    # Add the test metrics to the training step
    entry["training_steps"][str(training_step)]["test_metrics"] = test_metrics

    with open(log_file, 'w') as f:
        json.dump(data, f, indent=4)



DATALOADER_DICT = {}
DATALOADER_DICT["ate_asc"] = {"train":dataloader_train, "eval":dataloader_val}

DATASET_DICT={}
DATASET_DICT["lap"] = {"train_file":"laptops_2014_train.txt", "valid_file":"laptops_2014_trial.txt", "test_file":"laptops_2014_test.gold.txt"}
DATASET_DICT["res"] = {"train_file":"restaurants_union_train.txt", "valid_file":"restaurants_union_trial.txt", "test_file":"restaurants_union_test.gold.txt"}
# Christina  - added information for abea training dataset (NOTE: the trial and train files are apparently always the same in these examples)
DATASET_DICT["abea"] = {"train_file":"abea_train.txt", "valid_file":"abea_train.txt", "test_file":"abea_test.gold.txt"}
for i in ["2014", "2015", "2016"]:
    DATASET_DICT["res{}".format(i)] = {"train_file": "restaurants_{}_train.txt".format(i), "valid_file": "restaurants_{}_trial.txt".format(i), "test_file": "restaurants_{}_test.gold.txt".format(i)}
for i in range(10):
    DATASET_DICT["twt{}".format(i+1)] = {"train_file":"twitter_{}_train.txt".format(i+1), "valid_file":"twitter_{}_test.gold.txt".format(i+1), "test_file":"twitter_{}_test.gold.txt".format(i+1)}
    
#Christina
# ADDED TWEMLAB GOLDSTANDARD
DATASET_DICT["twemlab_all"] = {"train_file":"twemlab_all_train.txt", "valid_file":"twemlab_all_trial.txt", "test_file":"twemlab_all_test.gold.txt"}
DATASET_DICT["subset1"] = {"train_file":"subset1_train.txt", "valid_file":"subset1_trial.txt", "test_file":"subset1_test.gold.txt"}
DATASET_DICT["subset2"] = {"train_file":"subset2_train.txt", "valid_file":"subset2_trial.txt", "test_file":"subset2_test.gold.txt"}
DATASET_DICT["subset3"] = {"train_file":"subset3_train.txt", "valid_file":"subset3_trial.txt", "test_file":"subset3_test.gold.txt"}
DATASET_DICT["subset4"] = {"train_file":"subset4_train.txt", "valid_file":"subset4_trial.txt", "test_file":"subset4_test.gold.txt"}
DATASET_DICT["subset5"] = {"train_file":"subset5_train.txt", "valid_file":"subset5_trial.txt", "test_file":"subset5_test.gold.txt"}
DATASET_DICT["subset6"] = {"train_file":"subset6_train.txt", "valid_file":"subset6_trial.txt", "test_file":"subset6_test.gold.txt"}

# Christina - 10 fold cross validation for abea dataset
for i in range(10):
    DATASET_DICT["abea{}".format(i+1)] = {"train_file":"abea_{}_train.txt".format(i+1), "valid_file":"abea_{}_trial.txt".format(i+1), "test_file":"abea_{}_test.gold.txt".format(i+1)}


def main():
    global logger
    args, task_config = parse_input_parameter()

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device, n_gpu = init_device(args)

    data_name = args.data_name.lower()
    if data_name in DATASET_DICT:
        args.train_file = DATASET_DICT[data_name]["train_file"]
        args.valid_file = DATASET_DICT[data_name]["valid_file"]
        args.test_file = DATASET_DICT[data_name]["test_file"]
    else:
        assert args.train_file is not None
        assert args.valid_file is not None
        assert args.test_file is not None

    task_name = args.task_name.lower()
    if task_name not in DATALOADER_DICT:
        raise ValueError("Task not found: %s" % (task_name))

    if n_gpu > 1 and (args.use_ghl):
        logger.warning("Multi-GPU make the results not reproduce.")

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Generate label list from training dataset
    file_path = os.path.join(args.data_dir, args.train_file)
    train_dataloader, train_examples, label_tp_list = DATALOADER_DICT[task_name]["train"](args, tokenizer, file_path)

    logging.info("AT Labels are = %s:", "["+", ".join(label_tp_list[0])+"]")
    logging.info("AS Labels are = %s:", "["+", ".join(label_tp_list[1])+"]")
    at_num_labels = len(label_tp_list[0])
    as_num_labels = len(label_tp_list[1])
    num_tp_labels = (at_num_labels, as_num_labels)

    task_config["at_labels"] = label_tp_list[0]
    model = init_model(args, num_tp_labels, task_config, device, n_gpu)

    # Generate test dataset
    file_path = os.path.join(args.data_dir, args.test_file)
    test_dataloader, test_examples = DATALOADER_DICT[task_name]["eval"](args, tokenizer, file_path,
                                                                        label_tp_list=label_tp_list, set_type="test")
    logger.info("***** Loaded Test Dataset *****")
    logger.info("  Num examples = %d", len(test_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    if args.do_train:
        num_train_optimization_steps = (int(len(
            train_dataloader) + args.gradient_accumulation_steps - 1) / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        optimizer = prep_optimizer(args, model, num_train_optimization_steps)

        logger.info("***** Loaded Training Dataset *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        file_path = os.path.join(args.data_dir, args.valid_file)
        eval_dataloader, eval_examples = DATALOADER_DICT[task_name]["eval"](args, tokenizer, file_path,
                                                                            label_tp_list=label_tp_list, set_type="val")
        logger.info("***** Loaded Validation Dataset *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)


        # Christina - Define the training settings you want to log, for example:
        training_step= args.training_step

        glob_settings = {
            'name': args.training_name,
            'bert_model': args.bert_model,
            'data_name': args.data_name,
            'classification_head': args.classification_head
        }
        entry_index = initialize_or_update_structure(glob_settings)

        step_settings = {
            'learning_rate': args.learning_rate,
            'train_batch_size': args.train_batch_size,
            'optimizer': type(optimizer).__name__,
            'bert_model': args.bert_model,
            'max_seq_length': args.max_seq_length,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'warmup_proportion': args.warmup_proportion,
            'fp16': args.fp16,
            'loss_scale': args.loss_scale,
            'use_ghl': args.use_ghl,
            'use_vat': args.use_vat,
            'task_name': args.task_name,
            'data_name': args.data_name,
            'num_train_epochs': args.num_train_epochs,
            'name': args.training_name,
            'num_decoder_layer': args.num_decoder_layer,
            'decoder_shared_layer': args.decoder_shared_layer
        }

        global_step = 0

        overall_p = 0 
        overall_r = 0 
        overall_f1 = 0 

        for epoch in range(args.num_train_epochs):
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, tokenizer,
                                               optimizer, global_step, num_train_optimization_steps)
            logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.num_train_epochs, tr_loss)
            # save_model(epoch, args, model)
            # eval_epoch(model, eval_dataloader, label_tp_list, device)

            # Christina- TRAINING metrics
            at_p, at_r, at_f1, at_a, as_p, as_r, as_f1, as_a = eval_epoch(model, train_dataloader, label_tp_list, device)
            # log the results
            logger.info("*** Performance Metrics on Training Data ***")
            logger.info("AT a:{:.4f}\tAT p:{:.4f}\tr:{:.4f}\tf1:{:.4f}".format(at_a, at_p, at_r, at_f1))
            logger.info("AS a:{:.4f}\tAS p:{:.4f}\tr:{:.4f}\tf1:{:.4f}".format(as_a, as_p, as_r, as_f1))



            # Christina- VALIDATION metrics
            at_p, at_r, at_f1, at_a, as_p, as_r, as_f1, as_a = eval_epoch(model, eval_dataloader, label_tp_list, device)
            # log the results
            logger.info("*** Performance Metrics on Validation Data ***")
            logger.info("AT a:{:.4f}\tAT p:{:.4f}\tr:{:.4f}\tf1:{:.4f}".format(at_a, at_p, at_r, at_f1))
            logger.info("AS a:{:.4f}\tAS p:{:.4f}\tr:{:.4f}\tf1:{:.4f}".format(as_a, as_p, as_r, as_f1))

            epoch_metrics = {
                "AT": {
                    "precision": at_p,
                    "recall": at_r,
                    "f1_score": at_f1,
                    "accuracy": at_a
                    },
                "AS": {
                    "precision": as_p,
                    "recall": as_r,
                    "f1_score": as_f1,
                    "accuracy": as_a
                    }
            }
            # Log the metrics for the current epoch
            log_metrics(entry_index, training_step, step_settings, epoch, epoch_metrics)

            # save_model if a new highest f1 score is found
            if as_f1 > overall_f1: 
                save_model(epoch, args, model) 
                overall_f1 = as_f1


        logger.info("*** Performance Metrics on Test Data***")
        at_p, at_r, at_f1, at_a, as_p, as_r, as_f1, as_a = eval_epoch(model, test_dataloader, label_tp_list, device)
        logger.info("AT a:{:.4f}\tAT p:{:.4f}\tr:{:.4f}\tf1:{:.4f}".format(at_a, at_p, at_r, at_f1))
        logger.info("AS a:{:.4f}\tAS p:{:.4f}\tr:{:.4f}\tf1:{:.4f}".format(as_a, as_p, as_r, as_f1))
        test_metrics = {
            "AT": {
                "precision": at_p,
                "recall": at_r,
                "f1_score": at_f1,
                "accuracy": at_a
                },
            "AS": {
                "precision": as_p,
                "recall": as_r,
                "f1_score": as_f1,
                "accuracy": as_a
                }
        }
        log_test_metrics(entry_index, training_step, test_metrics)
        if as_f1 > 0.5:
            save_model(epoch, args, model)

    elif args.do_eval:
        eval_epoch(model, test_dataloader, label_tp_list, device)
    else:
        if args.init_model:
            eval_epoch(model, test_dataloader, label_tp_list, device)
        else:
            for epoch in range(args.num_train_epochs):
                # Load a trained model that you have fine-tuned
                model = load_model(epoch, args, num_tp_labels, task_config, device)
                if not model:
                    break
                eval_epoch(model, test_dataloader, label_tp_list, device)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard break~")