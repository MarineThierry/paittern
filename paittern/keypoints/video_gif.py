from asyncore import file_dispatcher
import cv2
import glob
from PIL import Image
from tensorflow.io import read_file
from tensorflow.image import decode_gif
import os
from pathlib import Path


def getFrame(path_gif_images,sec,vidcap,count):
    ''' extract images as frames from video
    args :
        sec : get frame from sec x of video
        vidcap : video captured by cv2
        count: iteration utils for img name
        '''
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    #create image folder
    # Check whether the specified
    # path exists or not
    if not os.path.exists(path_gif_images):
        Path(path_gif_images).mkdir()

    # résoudre ou storer les images
    if hasFrames:
        cv2.imwrite(path_gif_images+str(count)+".jpg", image)
    return hasFrames

def video_to_gif(
    file_path_video,
    path_gif_images='../images/',
    path_gif_output='../gif/',
    gif_name='output_gif',
    frameRate = 0.5):
    '''turn video into gif
    args :
    -file_path_video : input video path
    - path_gif_output : path to gif output
    - path_gif_images : path to folder for images
    - gif_name : file_name for gif output
    - frameRate : step for framing the video
    '''
    print('video capture begin')
    vidcap = cv2.VideoCapture(file_path_video)
    print('video captured')
    sec = 0
    count=1
    success = getFrame(path_gif_images,sec,vidcap,count)
    still_reading, image = vidcap.read()
    while still_reading:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(path_gif_images,sec,vidcap,count)
        still_reading, image = vidcap.read()
    print('video capture done')
    # Create the frames
    frames = []
    # imgs lists all imgs captured into a sort of list
    imgs = glob.glob(f"{path_gif_images}*.jpg")
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    # Check whether the specified
    # path exists or not
    if not os.path.exists(path_gif_output):
        Path(path_gif_output).mkdir()

    frames[0].save(f'{path_gif_output}{gif_name}.gif', format='GIF',append_images=frames[1:],save_all=True,duration=300, loop=0)

    #url upload cloudinary

    print('gif')
    #gif_path = url cloudinary
    gif_path = f'{path_gif_output}{gif_name}.gif'
    gif = read_file(gif_path)
    gif = decode_gif(gif)
    return gif

#test
# video_to_gif('http://res.cloudinary.com/paittern/video/upload/v1646661019/test.mov')
