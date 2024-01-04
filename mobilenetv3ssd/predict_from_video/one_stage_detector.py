import os 
import torch
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large as mobilenetv3ssd

from gestures import GESTURES


TORCH_PATH = os.path.join("rgb.pt")
MAX_BBOX_DETECTABLE = 1


class OneStageDetector:
    '''FaceDetector recognized all the faces in an image.

    Only the images which area is larger than the threshold are returned

    # Arguments
        conf_thresh: float
            The minimum confidence to recognize a face - `default 0.3`
        size_threhsold: float 
            The minimum area for a face to be published - `default None`
        
    For more details refer to opencv docs.
    '''

    def __init__(self, conf_thresh=0.3, size_thresh=None, n_gestures=len(GESTURES)):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.n_gestures = n_gestures
        self.net = mobilenetv3ssd(pretrained_backbone=True, num_classes=self.n_gestures)
        self.net.load_state_dict(torch.load(TORCH_PATH, map_location=self.device)["model_state_dict"])
        self.net.to(self.device)
        self.net.eval()
        self.confidence_threshold = conf_thresh
        self.size_threshold = size_thresh
    
    def detect(self, image):
        image = [torch.tensor(image, device=self.device)]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        with torch.no_grad():
            outputs = self.net(image)
        return self.post_processing(outputs=outputs)
        
    def post_processing(self, outputs):
        outputs = outputs[0]     # take only one hand (the one with higher confidence)
        pred_detections = torch.count_nonzero(outputs["scores"] > self.confidence_threshold).item()
        if pred_detections > 0:
            outputs = self.reduce_size_tensor(outputs, pred_detections)
            index_argmin = torch.argmin(outputs["labels"]).item()  #check if a valid gesture is detected
            if  index_argmin != self.n_gestures-1:
                #If valid gesture is detected, we take the index of valid gesture [1,12] with higher scores
                index_valid_gesture = torch.argmax((torch.where(outputs["labels"]< self.n_gestures-1, outputs["scores"], torch.tensor(0., dtype=torch.float32, device=self.device))))
                outputs = {"boxes":outputs["boxes"][index_valid_gesture], "labels": outputs["labels"][index_valid_gesture], "scores": outputs["scores"][index_valid_gesture]}
            #if argmin == 13, only no gesture are detected
            else:
                outputs = self.reduce_size_tensor(outputs, MAX_BBOX_DETECTABLE) #take the no gesture with higher score

        #no prediction, skip frame e set "no-gesture"
        else:
            outputs = self.reduce_size_tensor(outputs, MAX_BBOX_DETECTABLE)
            #print("NO GESTURE DETECTED")
            return {
                            'roi': [0.,0.,0.,0.],
                            'type': 'hand',
                            'label': self.n_gestures-1,
                            'confidence' : 0.,
                            # 'rects': rects
                        }

        out_label_for_classification = outputs["labels"].item()
        out_bbox_for_classification = outputs["boxes"].tolist()
        out_score_for_classification = float(outputs["scores"].item())
        hand_results = {
                            'roi': out_bbox_for_classification,
                            'type': 'hand',
                            'label': out_label_for_classification,
                            'confidence' : out_score_for_classification,
                            # 'rects': rects
                        }
        return hand_results


    def reduce_size_tensor (self, tensor, size):
        return {k: v[:size] for k, v in tensor.items()}