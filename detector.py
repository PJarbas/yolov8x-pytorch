import torch
import numpy as np
import cv2
import pafy
import pandas as pd
import random
from time import time
import norfair
from norfair import Detection, Tracker, Video
import os
import uuid

# adapted from: https://github.com/akash-agni/Real-Time-Object-Detection/blob/main/Object_Detection_Youtube.py


class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """

    def __init__(self, out_file="Output_Video.avi"):
        """
        Initializes the class with youtube url and output file.
        :param out_file: A valid output file name.
        """
        self.model = self.load_model()
        self.classes = self.model.names
        self.class_colors = self.bbox_class_colors()
        
        self.tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=30,
        )
        
        self.out_file = f"{str(uuid.uuid4()).split('-')[0]}_{out_file}"
        self.rm_output_file()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def rm_output_file(self):
        if os.path.exists(self.out_file):
            os.remove(self.out_file)

    def get_video_from_url(self, url):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
        play = pafy.new(url).streams[-1]
        assert play is not None
        return cv2.VideoCapture(play.url)

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        # https://pytorch.org/hub/ultralytics_yolov5/
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
        return model

    def score_frame(self, frame, threshold=0.6):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        return results
        
        # results.pandas().xyxy[0]  # im1 predictions (pandas)
        #      xmin    ymin    xmax   ymax  confidence  class    name
        # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
        # 1  433.50  433.50   517.5  714.5    0.687988     27     tie
        # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
        # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
        
        # out = results.pandas().xyxy[0]
        
        # return out[out['confidence'] > threshold]

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]
    
    def bbox_class_colors(self):
        """
        Create a dict with one random BGR color for each
        class name
        """
        return {name: (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)) for name in self.classes}
    
    def tracking(self, results, frame):
        
        detections_as_xywh = results.xywh[0]
        norfair_detections = []
        
        for detection_as_xywh in detections_as_xywh:

            centroid = np.array(
                [detection_as_xywh[0].item(), detection_as_xywh[1].item()]
            )
            
            scores = np.array([detection_as_xywh[4].item()])
            
            norfair_detections.append(
                Detection(
                    points=centroid,
                    scores=scores,
                    label=self.class_to_label(detection_as_xywh[-1].item()),
                )
            )
            
        tracked_objects = self.tracker.update(detections=norfair_detections)
        norfair.draw_points(frame, tracked_objects)
        
        return frame
            

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        
        for _, data in results.iterrows():
            
            confidence = data["confidence"]
            label = data["name"]
            class_id = data["class"]
            
            xmin = int(data["xmin"])
            xmax = int(data["xmax"])
            ymin = int(data["ymin"])
            ymax = int(data["ymax"])
            
            bgr = self.class_colors[class_id]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), bgr, 1)
            cv2.putText(frame, "{} {:.2f}".format(label, float(confidence)), (xmin, ymin),
                        cv2.FONT_HERSHEY_SIMPLEX, lineType=cv2.FILLED,
                        fontScale=0.5, color=bgr, thickness=2)
        
        return frame

    def detect(self, video_url):
        """
        This function is called when class is executed, it runs the loop to read the video frame
        :param video_url: Has to be as youtube URL, on which prediction is made.
        :return: void
        """
        
        player = self.get_video_from_url(video_url)
        
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Save the output video file
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))
        while player.isOpened():
            
            start_time = time()
            ret, frame = player.read()
            
            assert ret
            
            results = self.score_frame(frame)
            
            frame = self.tracking(results, frame)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)
            
            print(f"Frames Per Second : {fps}")
            out.write(frame)
            
        player.release()

    