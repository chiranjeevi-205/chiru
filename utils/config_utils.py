import yaml
import os
import shutil
import argparse

def load_yaml(args, yml):
    with open(yml, 'r', encoding='utf-8') as fyml:
        dic = yaml.load(fyml.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])

def build_record_folder(args):

    if not os.path.isdir("./records/"):
        os.mkdir("./records/")
    
    args.save_dir = "./records/" + args.project_name + "/" + args.exp_name + "/"
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_dir + "backup/", exist_ok=True)
    shutil.copy(args.c, args.save_dir+"config.yaml")

def get_args(with_deepspeed: bool=False):

    parser = argparse.ArgumentParser("Fine-Grained Visual Classification")
    parser.add_argument("--c", default="", type=str, help="config file path")
    args = parser.parse_args()

    load_yaml(args, args.c)
    build_record_folder(args)

    return args
def pred_mapper(pred):
    name_map = {
        0:"Bird 1",
        1:"Bird 2",
        # 2:"Bird 1",
        # 3: "Bird 2",
        # 4: "Bird 1",
        # 5: "Bird 2",
        # 6: "Bird 1",
        # 7: "Bird 2",
        # 8: "Bird 1",
        # 9: "Bird 2",
        # 10: "Bird 1",
        # 11: "Bird 2",
        # 12: "Bird 1",
        # 13: "Bird 2"
        # 0: "monkeyA",
        # 1: "monkeyB",
        # 2: "monkeyC",
        # 0:"LR1_Bird_1",
        # 1:"LR1_Bird_2",
        # 2:"LR2 Bird1",
        # 3: "LR2_Bird2",
        # 4: "LR4_Bird1",
        # 5: "LR4_Bird2",
        # 6: "LR6_Bird1",
        # 7: "LR6_Bird2",
        # 8: "AMAZON_10_Bird1",
        # 9: "AMAZON_10_Bird2",
        # 10: "LR12_Bird1",
        # 11: "LR12_Bird2",
        # 12: "AMAZON_15_Bird1",
        # 13: "AMAZON_15_Bird2"
    }
    return name_map.get(pred, "")

