import time
import torch
from utils.Re_id_funtions.infer_class import InferClass
from utils.config_utils import load_yaml, pred_mapper
from utils.Re_id_funtions.vis_utils import ImgLoader

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageInference:
    # Class-level variable to ensure the model is only built once
    model_instance = None

    def __init__(self, pretrain_path, config_path):
        self.pretrain_path = pretrain_path
        self.config_path = config_path
        
        # Check if the model is already built, if not, build the model
        if ImageInference.model_instance is None:
            self.rec = InferClass()
            load_yaml(self, self.config_path)
            
            # Build the model once and store it in a class-level variable
            self.rec.build_model(pretrainewd_path=self.pretrain_path,  # Fixed typo here
                                 img_size=self.data_size,
                                 fpn_size=self.fpn_size,
                                 num_classes=self.num_classes,
                                 num_selects=self.num_selects)

            # Move the model to GPU if available
            self.rec.model.to(device)
            
            # Store the model instance for future reuse
            ImageInference.model_instance = self.rec
        else:
            # Reuse the previously built model
            self.rec = ImageInference.model_instance

        # Load image loader with the specified image size
        self.img_loader = ImgLoader(img_size=self.data_size)

    def infer(self, image_path, label=None, use_label=False, sum_features_type="softmax"):
        start_time = time.time()

        # Load the image
        img, ori_img = self.img_loader.load(image_path)

        # Add batch size dimension and move image to the device
        img = img.unsqueeze(0).to(device)
        
        # Forward pass through the model
        out = self.rec.model(img)
        pred, pred_score = self.rec.inference(out, sum_type=sum_features_type, use_label=use_label, label=label)
        
        # Mapping prediction to the corresponding name
        pred_name = pred_mapper(pred.item())
        score = round(pred_score.item(), 3)

        # End the timer and calculate the elapsed time
        elapsed_time = time.time() - start_time
        print(f"Execution time: {elapsed_time} seconds")

        return pred_name,pred_score

