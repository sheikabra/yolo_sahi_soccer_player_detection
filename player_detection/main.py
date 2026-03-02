import os

from binstar_client.utils.detect import Detector
from nltk.cluster import euclidean_distance

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
from norfair import Detection, Tracker
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import supervision as sv
import pickle
import os
from sahi.utils.ultralytics import download_yolo11n_model
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
from tqdm import tqdm
from sahi import AutoDetectionModel
from PIL import Image


class OffsideDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self,frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch

        return detections

    def get_object_tracks(self, frames, read_from_stub = False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks
        detections = self.detect_frames(frames)
        objects_tracked = []
        tracks = {
            "player":[],
            "referee":[],
            "ball":[],
            "goalkeeper":[]
        }

        detections = self.detect_frames(frames)
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            tracks["player"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})
            tracks["goalkeeper"].append({})
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                #if cls_id == cls_names_inv['player']:
                    #tracks["player"][frame_num][track_id] = {"bbox":bbox}
                #if cls_id == cls_names_inv["referee"]:
                    #tracks["referee"][frame_num][track_id] = {"bbox":bbox}
                #if cls_id == cls_names_inv['goalkeeper']:
                    #tracks["goalkeeper"][frame_num][track_id] = {"bbox":bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)
        return tracks

    def draw_rectangle(self, frame, bbox, color, track_id=None):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if track_id is not None:
            cv2.putText(frame, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            #player_dict = tracks["player"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            #referee_dict = tracks["referee"][frame_num]
            #goalkeeper_dict = tracks["goalkeeper"][frame_num]

            for track_id, player in ball_dict.items():
                frame = self.draw_rectangle(frame, player['bbox'],(0,0,255), track_id)
            output_video_frames.append(frame)
        return output_video_frames

    def show_cropped_detections(self,frames, detections, target_cls=0):

        for idx, (frame, result) in enumerate(zip(frames, detections)):
            if result.boxes is None:
                continue
            count = 0
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id != target_cls:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cropped = frame[y1:y2, x1:x2]

                if cropped.size == 0:
                    continue

                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                plt.figure()
                plt.title(f"Frame {idx + 1} - Object {count + 1}")
                plt.imshow(cropped_rgb)
                plt.axis("off")
                plt.show()
                count += 1

    def interpolate_ball_positions(self, ball_tracks, save_stub_path=None):

        centers = []
        for frame_data in ball_tracks:
            if 1 in frame_data:
                bbox = frame_data[1]['bbox']
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2
                centers.append((x_center, y_center))
            else:
                centers.append((np.nan, np.nan))

        df = pd.DataFrame(centers, columns=["x", "y"])
        df = df.interpolate(method="linear", limit_direction="both")


        # df["x"] = df["x"].rolling(window=3, min_periods=1, center=True).mean()
        # df["y"] = df["y"].rolling(window=3, min_periods=1, center=True).mean()

        # Rebuild interpolated bounding boxes with fixed size (e.g., 20x20 box around center)
        box_size = 20
        new_tracks = []
        for x, y in df.to_numpy():
            x1 = x - box_size / 2
            y1 = y - box_size / 2
            x2 = x + box_size / 2
            y2 = y + box_size / 2
            new_tracks.append({1: {"bbox": [x1, y1, x2, y2]}})

        if save_stub_path:
            with open(save_stub_path, 'wb') as f:
                pickle.dump(new_tracks, f)

        return new_tracks


def read_vid_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames



def save_video(frames, output_path, fps=30):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Video saved to {output_path}")

def run_yolo_and_save(video_path, model_path, output_path):
    # Load YOLO model
    model = YOLO(model_path)

    # Read video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video at {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO
        results = model(frame)[0]

        # Draw detections
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{results.names[cls_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Output saved to {output_path}")

def sahi_slices(video_path, output_video_path,model):

    cap, fps, width, height, frame_count = get_video_info(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for _ in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB,)
        annotated_frame = sahi_prediction(model, frame)
        bgr_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    cap.release()
    out.release()
    print("Video Saved with Sahi Predictions")




def sahi_prediction(model, frame):
    result = get_sliced_prediction(
        image=frame,
        detection_model=model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        postprocess_type='GREEDYNMM'
    )

    result.object_prediction_list = [
        d for d in result.object_prediction_list
    ]

    annotated_frame_dict = visualize_object_predictions(
        image=frame,
        object_prediction_list=result.object_prediction_list
    )
    annotated_frame = annotated_frame_dict.get("image")

    if isinstance(annotated_frame, Image.Image):
        annotated_frame = np.array(annotated_frame)
    return annotated_frame

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, width, height, frame_count
















if __name__ == '__main__':
    video_path = 'Images/offside/bri_south.mp4'
    output_path = 'Images/output/bri_south_eleven_net_ball.mp4'
    model_path = 'models/best_eleven_net.pt'
    frames = read_vid_frames(video_path)
    detector = OffsideDetector(model_path)
    tracks = detector.get_object_tracks(frames, read_from_stub = False, stub_path='stubs/track_stubs.pkl')
    #tracks["ball"] = detector.interpolate_ball_positions(tracks["ball"], save_stub_path="stubs/interpolated_ball.pkl")

    output_video = detector.draw_annotations(frames, tracks)
    save_video(output_video, output_path)
    #run_yolo_and_save(video_path, model_path, output_path)
    #detection_model = AutoDetectionModel.from_pretrained(
        #model_type="ultralytics",
        #model_path=model_path,
        #confidence_threshold=0.7,
        #device="cpu",
    #)
    #download_yolo11n_model(model_path)
    #sahi_slices(video_path, output_path, detection_model)











# See PyCharm help at https://www.jetbrains.com/help/pycharm/
