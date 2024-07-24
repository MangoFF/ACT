# 0. imports
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from loguru import logger
import argparse
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments,set_seed
from dataset import Knowledge_DPODataset
from trl import DPOTrainer
from os.path import join
from transformers import TrainerCallback
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel
from datasets import Dataset
import os
from trl.trainer.dpo_config import DPOConfig

@dataclass
class Arguments:
    """
    一些自定义参数
    """
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    train_file: str = field(metadata={"help": "train data"})
    eval_file: Optional[str] = field(default="", metadata={"help": "eval data"})
    

def convert_to_hf_dataset(pytorch_dataset):
    # 取出所有数据
    data = [pytorch_dataset[i] for i in range(len(pytorch_dataset))]
    # 若数据集中的每一项是字典，直接转换即可
    if isinstance(data[0], dict):
        return Dataset.from_dict({k: [dic[k] for dic in data] for k in data[0]})
    # 若数据集中的每一项是元组，假设每个元素代表一个特性
    else:
        feature_names = [f'feature_{i}' for i in range(len(data[0]))]
        return Dataset.from_dict({name: [item[i] for item in data] for i, name in enumerate(feature_names)})
    
    
class EarlyStoppingCallback(TrainerCallback):
    "A callback that stops the training when a certain training loss is reached"
    def __init__(self, early_stopping_threshold):
        self.early_stopping_threshold = early_stopping_threshold
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        This callback function stops the training when the training loss goes below the given threshold
        """
        if state.log_history:
            #print(state.log_history)
            last_loss = state.log_history[-1].get(
                "loss",
                state.log_history[-1].get("train_loss",1)
                )
            if last_loss <= self.early_stopping_threshold:
                control.should_training_stop = True
                
def setup_everything():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_args_file", type=str, default='dpo.json', help="")
    parser.add_argument("--local_rank", type=int, help="")
    args = parser.parse_args()
    
    
    train_args_file = args.train_args_file
    
    # 读取训练的参数配置
    parser = HfArgumentParser((Arguments, DPOConfig))
    # 解析得到自定义参数，以及自带参数
    args, training_args = parser.parse_json_file(json_file=train_args_file)
    
    # 创建输出目录
    try:
        if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)
    except:
        pass
    
    logger.add(join(training_args.output_dir, 'train.log'))
    logger.info("train_args:{}".format(training_args))
    # 设置随机种子
    set_seed(training_args.seed)
    return args, training_args


if __name__ == "__main__":
    
    
    args, training_args = setup_everything()
    
    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.train()
    model.config.use_cache = False
    
    # 2. Load the Tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    
    # 2. Load the Stack-exchange paired dataset
    train_dataset = Knowledge_DPODataset(data_file_dir=args.train_file)
    train_dataset = convert_to_hf_dataset(train_dataset)
    
    train_test_ratio = 0.99
    n_train_examples = int(train_test_ratio * len(train_dataset))

    # Split dataset
    split_datasets = train_dataset.train_test_split(test_size=(1-train_test_ratio))
    
    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        reference_free = True,
        args=training_args,
        train_dataset=split_datasets['train'],
        eval_dataset=split_datasets['test'],
        tokenizer=tokenizer,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(join(training_args.output_dir))

    # 7. save
    output_dir = os.path.join(join(training_args.output_dir), "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)