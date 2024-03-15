"""
This script generates an image using a chosen model and a checkpoint.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd())) if str(Path.cwd()) not in sys.path else None
import torch
from torchvision import transforms
from generators.generator_padded import UNetPadded
from line_extractor.model import SketchKeras
from line_extractor.extract import extract_from_video
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def generate(model, frame1, frame2, binary_thresh=0.0, crop_shape=None, device='cuda:0'):

    transform=transforms.Compose([transforms.ToTensor(),
                            transforms.Grayscale(num_output_channels=1),
                         ])

    frame1 = transform(frame1).to(device)
    frame3 = transform(frame2).to(device)  

    # Add batch dimension
    frame1 = frame1.unsqueeze(0)
    frame3 = frame3.unsqueeze(0)
    
    # Binarizes frames
    frame1 = (frame1 > 0.75).float()
    frame3 = (frame3 > 0.75).float()


    # Generates in-between frame for a given triplet
    in_between = model(frame1, frame3)

    # performs center crop if specified with torch center crop
    if crop_shape:
        in_between = transforms.CenterCrop(crop_shape)(in_between)
        frame1 = transforms.CenterCrop(crop_shape)(frame1)
        frame2 = transforms.CenterCrop(crop_shape)(frame2)
        frame3 = transforms.CenterCrop(crop_shape)(frame3)

    if binary_thresh > 0.0:
        in_between[in_between >= binary_thresh] = 1.0
        in_between[in_between < binary_thresh] = 0.0

    # if tensor in cuda, move it to cpu
    if in_between.is_cuda:
        in_between = in_between.cpu()
    in_between = in_between.squeeze(0).squeeze(0).detach().numpy()

    return in_between


def generate_video(path, model, thresh=[0.65, 0.97], normalize=True, double_fps=False):
    new_frames = []
    # Read in the video
    vidcap = cv2.VideoCapture(path)
    print('Video has', int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), 'frames')
    success, prev_frame = vidcap.read()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    if normalize:
        prev_frame = prev_frame / np.max(prev_frame)
        prev_frame[prev_frame < 0.75] = 0
        prev_frame[prev_frame >= 0.75] = 1
    unique_frame = prev_frame.copy()
    target_frame = prev_frame.copy()
    i = 0
    num_unique_frames = 0
    num_duplicate_frames = 0
    num_shot_boundaries = 0
    success = True
    generated_frame = None
    while success:
        success,curr_frame = vidcap.read()
        # Display the read image, not save, just display
        if success:
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            curr_frame = curr_frame / np.max(curr_frame)
            curr_frame[curr_frame < 0.75] = 0
            curr_frame[curr_frame >= 0.75] = 1
            ssim_diff = ssim(prev_frame, curr_frame, data_range=curr_frame.max() - curr_frame.min(),
                             channel_axis=None)
            unique = thresh[0] <= ssim_diff <= thresh[1]
            duplicate = ssim_diff > thresh[1]
            shot_boundary = ssim_diff < thresh[0]
            if unique:
              unique_frame = prev_frame.copy()
              num_unique_frames += 1
              target_frame = unique_frame.copy()
            if duplicate:
              num_duplicate_frames += 1
              if not double_fps and i > 0:
                target_frame = generate(model, new_frames[-1], curr_frame)
            if shot_boundary:
              target_frame = prev_frame.copy()
              num_shot_boundaries += 1
            if i == 1:
              target_frame = prev_frame 
            if double_fps and not shot_boundary:
               generated_frame = generate(model, prev_frame, curr_frame)   
        new_frames.append(target_frame)
        if generated_frame is not None:
            new_frames.append(generated_frame)
        prev_frame = curr_frame
        i += 1
    print('Video has', num_unique_frames, 'unique frames')
    print('Video has', num_duplicate_frames, 'duplicate frames')
    print('Video has', num_shot_boundaries, 'shot boundaries')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 24
    output_video_path = f'{path[:-4]}_generated.mp4'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (512, 288))
    for frame in new_frames:
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)
    out.release()

    print('Interpolated video has', len(new_frames), 'frames')


if __name__ == '__main__':

    # Extract line work from video
    video = 'video_dup.mp4'
    path = f'{str(Path.cwd())}/to_generate/video/'
    video_path = f'{path}{video}'
    device = 'cuda'
    model = SketchKeras().to(device)
    model.load_state_dict(torch.load('line_extractor/weights/model.pth', map_location=torch.device('cpu')))
    print('weights/model.pth loaded')
    extract_from_video(video_path, device=device, model=model)

    # Generate in-between frames
    thresh = [0.60, 0.95]
    model = UNetPadded(2, output_channels=1, hidden_channels=64).to('cuda')
    # print datatype of model weights
    checkpoint_path = f'{str(Path.cwd())}/checkpoints/gen_checkpoint.pth'
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    new_frames = generate_video(f'{video_path[:-4]}_line.mp4', model=model, thresh=thresh,
                                double_fps=False, normalize=True)
    
# def replace_frames(input_video_path, output_video_path):
#     # Open the input video
#     input_video = cv2.VideoCapture(input_video_path)

#     # Get the original video's width, height, and frames per second (fps)
#     width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = input_video.get(cv2.CAP_PROP_FPS)

#     # Define the codec using VideoWriter_fourcc and create a VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

#     # Initialize the previous frame
#     ret, prev_frame = input_video.read()

#     # Initialize the frame counter
#     frame_counter = 0

#     while ret:
#         # If the frame counter is even, write the previous frame to the output video
#         if frame_counter % 2 == 0:
#             output_video.write(prev_frame)
#         # generate random variable with 50% of being true
#         elif np.random.rand() > 0.5:
#             # Otherwise, write the previous frame to the output video
#             output_video.write(prev_frame)
        
#         else:
#             # Otherwise, write the current frame to the output video and update the previous frame
#             output_video.write(frame)
#             prev_frame = frame

#         # Read the next frame from the input video
#         ret, frame = input_video.read()

#         # Increment the frame counter
#         frame_counter += 1

#     # Release the VideoCapture and VideoWriter objects
#     input_video.release()
#     output_video.release()

# if __name__ == '__main__':
#         # Call the function to replace frames in the video
#     replace_frames("/home/farriaga/gan-interpolator/to_generate/video/video.mp4", '/home/farriaga/gan-interpolator/to_generate/video/video_dup.mp4')