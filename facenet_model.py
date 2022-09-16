from facenet_pytorch import MTCNN
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)

#video = mmcv.VideoReader('video_2.mp4')
video = cv2.VideoCapture(0)
frames_tracked = []

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = video.read()

    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  
    # Display the resulting frame
    # cv2.imshow('frame', frame)
      
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Detect faces
    boxes, _ = mtcnn.detect(frame)
    #print(len(boxes))

    # Draw faces
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    if boxes is not None:
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
    frame_draw = frame_draw.resize((640, 360), Image.BILINEAR)
    frame = cv2.cvtColor(np.array(frame_draw), cv2.COLOR_BGR2RGB)
    # Add to frame list
    cv2.imshow("frame", frame)

video.release()