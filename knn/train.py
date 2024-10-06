import argparse
import torch

from transformers import HfArgumentParser

from loguru import logger
from torch.utils.data import DataLoader

from component import MyCollator as Collator
from component import DataArguments
from component import CustomDataset

from model_src import DualTowerBert as MyModel

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

from safetensors.torch import load_file
from accelerate.utils import gather_object
from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from sklearn.metrics import classification_report
import numpy as np
import torch.nn.functional as F
import sys
import random
sys.path.append("/home/yaoxingzhi1/JD_Young/bert-qp-models")


fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
)


def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='/home/yaoxingzhi1/knn/args/train_dual_tower.json', help="")
    parser.add_argument("--local_rank", type=int, help="")
    args = parser.parse_args()
    train_args_file = args.train_args_file
    parser = HfArgumentParser((DataArguments))
    data_config = parser.parse_json_file(train_args_file, allow_extra_keys=True)[0]
    feature_config = data_config.FeatureConfig 
    query_model_config = data_config.query_model_config
    doc_model_config = data_config.doc_model_config
    logger.info(data_config)
    return data_config, feature_config, query_model_config, doc_model_config


def main():
    data_config, feature_config, query_model_config, doc_model_config = setup_everything()
    accelerator = Accelerator(dispatch_batches=True, split_batches=True, log_with=["tensorboard"], project_dir=data_config.output_dir)
    # split_batches: If True, each gpu got batch_size/num_gpus
    # dispatch_batches: If set to True, the datalaoder prepared is only iterated through on the main process and then the batches are split and broadcast to each process

    # 创建ProjectConfiguration对象，设定其他数据保存的目录
    accelerator.init_trackers("logs", config=query_model_config)

    model = MyModel(query_model_config, doc_model_config)

    # missing_keys, unexpected_keys = model.load_state_dict(load_file('/home/qiuyiming3/officials/bert-base-chinese/model.safetensors'), strict=False)
    # missing_keys, unexpected_keys = model.load_state_dict(load_file('/home/qiuyiming3/torch/upload/outputs.train.l12/model.safetensors'), strict=False)
    if data_config.warm_up_model != '':
        ckpt = load_file(data_config.warm_up_model)
        ckpt = {k: v for k, v in ckpt.items() if 'cid_input_ids' not in k}
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        logger.info(f">>>>>>>>> missing_keys: {len(missing_keys)}, unexpected_keys:\
            {len(unexpected_keys)}")
        logger.info(missing_keys)
        logger.info(unexpected_keys)

    # 计算模型参数量
    total = sum(p.numel() for p in model.parameters())
    logger.info("Total model params: %.2fM" % (total / 1e6))
    
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_untrainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params}, nottrainable: {total_untrainable_params}")

    train_qp_dataset = CustomDataset(
        feature_config = feature_config,
        data_path = data_config.train_file,
        sample_num = data_config.sample_num
        )
    
    train_dataloader = DataLoader(train_qp_dataset,
        batch_size = data_config.total_train_batch_size,
        pin_memory = True,
        num_workers = 4,
        prefetch_factor = 4,
        drop_last = True,
        collate_fn=Collator(feature_config))

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=data_config.learning_rate)

    num_training_steps = data_config.sample_num * data_config.num_train_epochs/data_config.total_train_batch_size
    warmup_proportion = 0.1
    num_warmup_steps =  int(warmup_proportion * num_training_steps)
    logger.info(f">>>>>>>>> num_warmup_steps: {num_warmup_steps}")
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps = num_warmup_steps,
    #     num_training_steps = num_training_steps
    # )
    scheduler = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps = num_warmup_steps
    )
    
    model, optimizer, train_dataloader, _, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, None, scheduler)

    # vocab = open('/home/qiuyiming3/front_cls/dict/cid_vocab.txt', 'r', encoding='utf8').read().strip().split('\n')
    # idx2vocab = dict([(i, vocab) for i, vocab in enumerate(vocab)])

    # vocab2 = open('/home/qiuyiming3/cot_task/front_cid_vocab.txt.add100000', 'r', encoding='utf-8').read().split('\n')
    # idx2name = dict([(vocab2[i].split('\t')[:2]) for i in range(len(vocab2)) if len(vocab2[i].split('\t'))==9])

    device = accelerator.device
    model.to(device)
    model.zero_grad()
    step = 0

    # def eval(accelerator, step):
    #     logger.info('start eval')
    #     with torch.no_grad():
    #         sample_f1_all = []
    #         pred_logits = []
    #         for batch in eval_dataloader:
    #             target = batch['label'].to(device).float()
    #             res = model(**batch)
    #             logits = torch.sigmoid(res.logits)

    #             # 收集并转换数据
    #             gathered_logits = accelerator.gather(logits.float()).cpu().numpy()
    #             gathered_targets = accelerator.gather(target.float()).cpu().numpy()
                

    #             # 二值化logits
    #             pred_logits = np.where(gathered_logits > 0.5, torch.tensor(1), torch.tensor(0))

    #             # 对当前批次进行分类报告
    #             report = classification_report(gathered_targets, pred_logits, digits=4, output_dict=True)
    #             if "samples avg" in report:
    #                 sample_f1_all.append(report["samples avg"]["f1-score"])
            # pred_logits = np.concatenate(pred_logits, 0)
            # pred_logits = np.where(pred_logits > 0.5, torch.tensor(1), torch.tensor(0))
            # labels_all = np.concatenate(labels_all, 0)

            # if sample_f1_all:
            #     avg_sample_f1 = np.mean(sample_f1_all)

            # else:
            #     avg_sample_f1 = 0


            # if accelerator.is_main_process:
            #     print("Average Sample F1_score:", avg_sample_f1)
            #     accelerator.log({"average_sample/f1": avg_sample_f1}, step=step)
                # report = classification_report(labels_all, pred_logits, digits=4, output_dict=True)
                # f1 = lambda i: report[i]['f1-score']
                # prec = lambda i: report[i]['precision']
                # rec = lambda i: report[i]['recall']
                # print('samples', report['samples avg'])
                # print('macro', report['macro avg'])
                # print('micro', report['micro avg'])
                # accelerator.log({"sample/f1": f1('samples avg')}, step=step)
                # accelerator.log({"sample/rec": rec('samples avg')}, step=step)
                # accelerator.log({"sample/prec": prec('samples avg')}, step=step)
                # accelerator.log({"macro/f1": f1('macro avg')}, step=step)
                # accelerator.log({"macro/rec": rec('macro avg')}, step=step)
                # accelerator.log({"macro/prec": prec('macro avg')}, step=step)
                # accelerator.log({"micro/f1": f1('micro avg')}, step=step)
                # accelerator.log({"micro/rec": rec('micro avg')}, step=step)
                # accelerator.log({"micro/prec": prec('micro avg')}, step=step)

    step = 1
    for e in range(data_config.num_train_epochs):
        logger.info(f'\n\nEpoch {e}')
        model.train()
        for step, batch in enumerate(train_dataloader, start=step):
            optimizer.zero_grad()
            res = model(**batch)
            loss = res.loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            if accelerator.is_main_process and step % 10 ==0:
                current_lr = scheduler.get_lr()[0]
                batch_size = batch['query'].shape[0]
                accelerator.log({"lr": current_lr}, step=step)
                accelerator.log({"training_loss": loss}, step=step)
                print(f'step: {step}, loss: {loss}, lr_rate: {current_lr}, batch_size: {batch_size}')

            # if step % 5000 == 0:
            #     model.eval()
            #     accelerator.wait_for_everyone()
            #     eval(accelerator=accelerator, step=step)
            #     model.train()

            if step % 10000 == 0:
                accelerator.wait_for_everyone()
                accelerator.save_model(model, data_config.output_dir)

        accelerator.wait_for_everyone()
        accelerator.save_model(model, data_config.output_dir)
        accelerator.end_training()

if __name__ == "__main__":
    main()
