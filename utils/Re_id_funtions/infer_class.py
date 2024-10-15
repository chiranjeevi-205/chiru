import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")


class InferClass:
    def __init__(self):
        self.model = None

    def build_model(self, pretrainewd_path: str,
                    img_size: int, 
                    fpn_size: int, 
                    num_classes: int,
                    num_selects: dict,
                    use_fpn: bool = True, 
                    use_selection: bool = True,
                    use_combiner: bool = True, 
                    comb_proj_size: int = None):
        from models.pim_module.pim_module_eval import PluginMoodel

        self.model = PluginMoodel(img_size=img_size,
                                  use_fpn=use_fpn,
                                  fpn_size=fpn_size,
                                  proj_type="Linear",
                                  upsample_type="Conv",
                                  use_selection=use_selection,
                                  num_classes=num_classes,
                                  num_selects=num_selects, 
                                  use_combiner=use_combiner,
                                  comb_proj_size=comb_proj_size)

        if pretrainewd_path != "":
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ckpt = torch.load(pretrainewd_path, map_location=device)
            self.model.load_state_dict(ckpt['model'])
            self.model.to(device)

        self.model.eval()

    def inference(self, out, sum_type: str = "softmax", use_label: bool = False, label: int = 0):
        assert sum_type in ["none", "softmax"]

        target_layer_names = ['layer1', 'layer2', 'layer3', 'layer4',
                              'FPN1_layer1', 'FPN1_layer2', 'FPN1_layer3', 'FPN1_layer4', 'comb_outs']

        sum_out = None
        for name in target_layer_names:
            if name != "comb_outs":
                tmp_out = out[name].mean(1)
            else:
                tmp_out = out[name]

            if sum_type == "softmax":
                tmp_out = torch.softmax(tmp_out, dim=-1)

            if sum_out is None:
                sum_out = tmp_out
            else:
                sum_out = sum_out + tmp_out  # note that use '+=' would cause inplace error

        with torch.no_grad():
            if use_label:
                print("use label as target class")
                pred_score = torch.softmax(sum_out, dim=-1)[0][label]
                pred_cls = label
            else:
                pred_probabilities = torch.softmax(sum_out, dim=-1)
                pred_score, pred_cls = torch.max(pred_probabilities, dim=-1)
                pred_score = pred_score[0]
                pred_cls = pred_cls[0]

        print(sum_out.size())
        print("pred: {}, gt: {}, score:{}".format(pred_cls, label, pred_score))

        return pred_cls, pred_score
