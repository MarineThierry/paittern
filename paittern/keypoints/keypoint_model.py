from email.mime import image
from  paittern.keypoints.preprocessing import init_crop_region,determine_torso_and_body_range,determine_crop_region,torso_visible,crop_and_resize
import tensorflow as tf
import tensorflow_hub as hub
from paittern.keypoints.utils import draw_prediction_on_image,progress,np_to_gif
from paittern.keypoints.video_gif import video_to_gif
from PIL import Image
from tensorflow.io import read_file
from tensorflow.image import decode_gif
import numpy as np







def movenet(input_image,module):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """


    model = module.signatures['serving_default']

    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores

def run_inference(movenet, image, crop_region, crop_size):
    """Runs model inferece on the cropped region.

    The function runs the model inference on the cropped region and updates the
    model output to the original image coordinate system.
    """
    image_height, image_width, _ = image.shape
    input_image = crop_and_resize(
    tf.expand_dims(image, axis=0), crop_region, crop_size=crop_size)
      # Run model inference.
    keypoints_with_scores = movenet(input_image)
    # Update the coordinates.
    for idx in range(17):
        keypoints_with_scores[0, 0, idx, 0] = (
            crop_region['y_min'] * image_height +
            crop_region['height'] * image_height *
            keypoints_with_scores[0, 0, idx, 0]) / image_height
        keypoints_with_scores[0, 0, idx, 1] = (
            crop_region['x_min'] * image_width +
            crop_region['width'] * image_width *
            keypoints_with_scores[0, 0, idx, 1]) / image_width
    return keypoints_with_scores




def run_model_gif(image):
    # Load the input image.
    print('Enter the model')
    #load model once
    model_name = "movenet_lightning"
    if "movenet_lightning" in model_name: # use this one for now not until tuning or refining should we use thunder
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        input_size = 192
    elif "movenet_thunder" in model_name:
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        input_size = 256
    else:
        raise ValueError("Unsupported model name: %s" % model_name)

    num_frames, image_height, image_width, _ = image.shape
    print(image.shape)
    crop_region = init_crop_region(image_height, image_width)
    keypoints_sequence = []
    output_images = []


# code to run the model on each image
    for frame_idx in range(num_frames):
        keypoints_with_scores = run_inference(movenet(module), image[frame_idx, :, :, :], crop_region,\
            crop_size=[input_size, input_size]) # run model for image
        output_images.append(draw_prediction_on_image(image[frame_idx, :, :, :].numpy().astype(np.int32),keypoints_with_scores, crop_region=None,close_figure=True, output_image_height=300)) # add image to image sequence

        keypoints_sequence.append(keypoints_with_scores)
        crop_region = determine_crop_region(keypoints_with_scores, image_height, image_width)

    print(len(keypoints_sequence))
    return keypoints_sequence , output_images


# if __name__ == '__main__':

#     gif = video_to_gif('../raw_data/input_video/test_keypoint_v1.mov')

#     keypoints_sequence, output_images= run_model_gif(gif)
#     output_keypoints = np.array(keypoints_sequence).reshape((gif.shape[0],17,3))

#     output = np.stack(output_images, axis=0)

#     np_to_gif(output_images, 'output_gif_kp',fps=10)
