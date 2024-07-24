import json
from loguru import logger
import ast
import astunparse
from typing import Dict
from torch.utils.data import Dataset
import os
from os.path import join
import pandas as pd
from tqdm import tqdm
import pickle
from torch.utils.data import IterableDataset
from datasets import load_dataset
import random


class Knowledge_DPODataset(Dataset):
    """
    Firefly项目默认的数据组织格式
    """
    def __init__(self, data_file_dir, few_shot_num = 0,example_file_dir = None):
        
        self.few_shot_num = few_shot_num
        
        logger.info('Loading train data: {}'.format(data_file_dir))
        data_list = []
        for root, dirs, files in os.walk(data_file_dir):
            for file_name in files:
                print(file_name)
                with open(os.path.join(root, file_name), 'r', encoding='utf8') as f:
                    data_lines = f.readlines()
                json_strings = ''.join(data_lines)
                new_data = json.loads(json_strings)
                if isinstance(new_data,list):
                    data_list.extend(new_data)
                elif isinstance(new_data,dict):
                    data_list.extend(list(new_data.values()))
                else:
                    raise ValueError("Unknown data format")
                
        self.dpo_list = []    
        max_prompt_lenth = 0
        max_answer_lenth = 0
        for i,data in enumerate(data_list):
            # 分装正确和错误答案
            correct = []
            wrong = []
            for answer in data["answer_list"]:
                max_answer_lenth = max(max_answer_lenth, len(answer["answer"]))
                if answer["correct"]:
                    correct.append(answer["answer"])
                else:
                    wrong.append(answer["answer"])
            
            # 没有正确或者没有错误，我们认为信息量为 0，不学习
            if not wrong or not correct:
                continue
            
            
            max_prompt_lenth = max(max_prompt_lenth,len(data["question"] + data.get("middle_answer","")))
            
            # 为了保证样本的平衡，我们每次选择样本数目较少的那一类（以防出现极端情况，只有一道错误的题目）
           
            if len(correct) < len(wrong):
                # 这个时候说明掌握的不好，着重学习,我们把样本量翻一倍
                for i in range(len(correct)):
                    for _ in range(2):
                        item = {
                                "prompt":data["question"] + data.get("middle_answer",""),
                                "chosen":correct[i],
                                "rejected":random.choice(wrong)
                        }
                        self.dpo_list.append(item)
            else:
                # 这个时候说明已经基本掌握，我们不着重
                for i in range(len(wrong)):
                    item = {
                            "prompt":data["question"] + data.get("middle_answer",""),
                            "chosen":correct[i],
                            "rejected":wrong[i]
                    }
                    self.dpo_list.append(item)
                                   
        random.shuffle(self.dpo_list)
               
        logger.info("there are {} data in dataset".format(len(self.dpo_list)))
        logger.info(f"Max_prompt_len : {max_prompt_lenth}  in dataset")
        logger.info(f"Max_answer_len  :{max_answer_lenth} data in dataset")
        
        logger.info('Loading example data: {}'.format(example_file_dir))
        
        self.example_data_list = []
        if example_file_dir:
            for root, dirs, files in os.walk(data_file_dir):
                for file_name in files:
                    print(file_name)
                    with open(os.path.join(root, file_name), 'r', encoding='utf8') as f:
                        data_lines = f.readlines()
                    json_strings = ''.join(data_lines)
                    new_data = json.loads(json_strings)
                    if isinstance(new_data,list):
                        self.example_data_list.extend(new_data)
                    elif isinstance(new_data,dict):
                        self.example_data_list.extend(list(new_data.values()))
                    else:
                        raise ValueError("Unknown data format")
                    
            assert few_shot_num <= len(self.example_data_list)
        
            

    def __len__(self):
        return len(self.dpo_list)

    def __getitem__(self, index):
        # 每条数据格式为: <s>input1</s>target1</s>input2</s>target2</s>...
        data = self.dpo_list[index]
        
        few_shot_prompt = "<s>"
        if self.example_data_list:
            example_shots = random.sample(self.example_data_list, self.few_shot_num)            
            
            for shot in example_shots:
                few_shot_prompt = few_shot_prompt + "Question:" + shot['question'] 
                few_shot_prompt = few_shot_prompt + "Answer:" + shot['answer'] + '</s>'
            
        questino_prompt = 'Question:' + data["prompt"] + 'Answer:'
        return {
            "prompt": few_shot_prompt + questino_prompt ,
            "chosen":data["chosen"],
            "rejected":data["rejected"]
        }