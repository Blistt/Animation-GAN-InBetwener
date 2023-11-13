'''
This script extracts line drawings from colored 2d animated frames using an unofficial
implementation of sketchKeras (https://github.com/higumax/sketchKeras-pytorch)
'''

import numpy as np
import torch
import cv2
from model import SketchKeras
import os
from pathlib import Path

def preprocess(frame):
    h, w, c = frame.shape
    blurred = cv2.GaussianBlur(frame, (0, 0), 3)
    highpass = frame.astype(int) - blurred.astype(int)
    highpass = highpass.astype(np.float64) / 128.0
    highpass /= np.max(highpass)

    ret = np.zeros((512, 512, 3), dtype=np.float64)
    ret[0:h,0:w,0:c] = highpass
    return ret

def postprocess(pred, thresh=0.18, smooth=False):
    assert thresh <= 1.0 and thresh >= 0.0

    pred = np.amax(pred, 0)
    pred[pred < thresh] = 0
    pred = 1 - pred
    pred *= 255
    pred = np.clip(pred, 0, 255).astype(np.uint8)
    if smooth:
        pred = cv2.medianBlur(pred, 3)
    return pred

def extract(frame, device='cuda:1', model=None):
    if model is None:
        model = SketchKeras().to(device)
        model.load_state_dict(torch.load('_line_extractor/weights/model.pth'))
        print('_line_extractor/weights/model.pth loaded')

    # resize
    height, width = float(frame.shape[0]), float(frame.shape[1])
    if width > height:
        new_width, new_height = (512, int(512 / width * height))
    else:
        new_width, new_height = (int(512 / height * width), 512)
    frame = cv2.resize(frame, (new_width, new_height))
    
    # preprocess
    frame = preprocess(frame)
    x = frame.reshape(1, *frame.shape).transpose(3, 0, 1, 2)
    x = torch.tensor(x).float()
    
    # feed into the network
    with torch.no_grad():
        pred = model(x.to(device))
    pred = pred.squeeze()
    
    # postprocess
    output = pred.cpu().detach().numpy()
    output = postprocess(output, thresh=0.18) 
    output = output[:new_height, :new_width]

    return output


def extract_from_video(video_path, model=None, device='cuda:1'):
    if model is None:
        model = SketchKeras().to(device)
        model.load_state_dict(torch.load('_line_extractor/weights/model.pth'))
        print('_line_extractor/weights/model.pth loaded')

    line_frames = []
    # Read in the video
    vidcap = cv2.VideoCapture(video_path)
    # Extract video's frame rate
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print('fps', fps)
    # Extract video's frame count
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('frame_count', frame_count)
    shape = 0

    success, frame = vidcap.read()
    count = 0
    success = True
    while success:
        success, frame = vidcap.read()
        if success:
            line_frame = extract(frame, device=device, model=model)
            shape = line_frame.shape
            if count % 100 == 0:
                print('save sample frame')
                cv2.imwrite('_line_extractor/frame%d.png' % count, line_frame)
            line_frames.append(line_frame)
            print('frame {} extracted'.format(count))
        count += 1
    
    # create video with the same frame rate as input video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    output_video_path = Path(video_path).stem + '_line.mp4'
    print('passed shape', shape)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (512, 288))
    for frame in line_frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        print(frame.shape)
        out.write(frame)
    out.release()
    print('Video saved at {}'.format(output_video_path))


if __name__ == '__main__':
    video_path = '/home/farriaga/gan-interpolator/_notebooks/Horimiya_1_Clip.mp4'
    device = 'cuda:0'
    model = SketchKeras().to(device)
    print('working dir', os.getcwd())
    model.load_state_dict(torch.load('_line_extractor/weights/model.pth'))
    print('weights/model.pth loaded')
    extract_from_video(video_path, model, device)

    
    
    
    


