'''
This script extracts line drawings from colored 2d animated frames using an unofficial
implementation of sketchKeras (https://github.com/higumax/sketchKeras-pytorch)
'''

import numpy as np
import torch
import cv2
from .model import SketchKeras
import os
from pathlib import Path
from matplotlib import pyplot as plt

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
    pred[pred >= thresh] = 1
    pred = 1 - pred
    pred *= 255
    pred = np.clip(pred, 0, 255).astype(np.uint8)
    if smooth:
        pred = cv2.medianBlur(pred, 3)
    return pred

def extract(frame, device='cuda:1', model=None, bin_thresh=0.5, smooth=False):
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
    output = postprocess(output, thresh=bin_thresh, smooth=smooth) 
    output = output[:new_height, :new_width]

    return output


def extract_from_video(video_path, model=None, device='cuda:1'):
    # if model is None:
    #     model = SketchKeras().to(device)
    #     model.load_state_dict(torch.load('_line_extractor/weights/model.pth'))
    #     print('_line_extractor/weights/model.pth loaded')

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
            frame = cv2.resize(frame, (512, 288))
            line_frame = extract(frame, device=device, model=model)
            line_frames.append(line_frame)
        count += 1
    
    # create video with the same frame rate as input video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    output_video_path = f'{video_path[:-4]}_line.mp4'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (512, 288))
    for frame in line_frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)
    out.release()
    print('Video saved at {}'.format(output_video_path))


def extract_from_dir(input_path, model=None, device='cuda:1', bin_thresh=0.5, smooth=False):
    if model is None:
        model = SketchKeras().to(device)
    
    # Iterate over all files and directories in the input path
    for entry in os.scandir(input_path):
        if entry.is_file():
            print('extracting from {}'.format(entry.path))
            frame = cv2.imread(entry.path)
            line_frame = extract(frame, device=device, model=model, bin_thresh=bin_thresh, smooth=smooth)
            # save line frame in parent direcotry of input_path using pathlib
            cv2.imwrite(os.path.join('mini_datasets/to_trace_results', entry.name), line_frame)
            visualize(frame, line_frame, os.path.join('mini_datasets/to_trace_results_comparison', entry.name))
        # If entry is a directory, recursively call extract_from_dir
        elif entry.is_dir():
            extract_from_dir(entry.path, model=model, device=device)

def visualize(img1, img2, filename):
    '''
    Visualizes a pair of compared images
    '''
    # Converts image to RGB
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(121)
    ax1.imshow(img1)
    ax1.set_title('Original image')
    plt.axis('off')
    ax2 = fig.add_subplot(122)
    ax2.imshow(img2, cmap='gray')
    ax2.set_title('sketchKeras')
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(filename)
    plt.close()


# if __name__ == '__main__':
#     # print parent of current working directory with pathlib
#     video_path = f'{str(Path.cwd())}/to_generate/video/video.mp4'
#     device = 'cuda'
#     model = SketchKeras().to(device)
#     model.load_state_dict(torch.load('line_extractor/weights/model.pth', map_location=torch.device('cpu')))
#     print('weights/model.pth loaded')
#     extract_from_video(video_path, device=device, model=model)

 


