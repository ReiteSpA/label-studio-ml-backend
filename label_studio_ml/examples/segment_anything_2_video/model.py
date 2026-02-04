import torch
import numpy as np
import os
import pathlib
import cv2
import tempfile
import logging

from typing import List, Dict, Optional
from uuid import uuid4
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path
from label_studio_sdk.label_interface.objects import PredictionValue
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor

logger = logging.getLogger(__name__)

# Get environment variables
DEVICE = os.getenv('DEVICE', 'cuda')
SEGMENT_ANYTHING_2_REPO_PATH = os.getenv('SEGMENT_ANYTHING_2_REPO_PATH', 'segment-anything-2')
MODEL_CONFIG = os.getenv('MODEL_CONFIG', 'sam2_hiera_l.yaml')
MODEL_CHECKPOINT = os.getenv('MODEL_CHECKPOINT', 'sam2_hiera_large.pt')
MAX_FRAMES_TO_TRACK = int(os.getenv('MAX_FRAMES_TO_TRACK', 10))
LABEL_STUDIO_ACCESS_TOKEN = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")
LABEL_STUDIO_HOST = os.environ.get("LABEL_STUDIO_HOST")

# If device is CUDA, use bfloat16 for the entire notebook
if DEVICE == 'cuda':
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# Build path to the model checkpoint
sam2_checkpoint = str(pathlib.Path(__file__).parent / SEGMENT_ANYTHING_2_REPO_PATH / "checkpoints" / MODEL_CHECKPOINT)
# Build the SAM2 video predictor
predictor = build_sam2_video_predictor(MODEL_CONFIG, sam2_checkpoint)


# manage cache for inference state
# TODO: make it process-safe and implement cache invalidation
_predictor_state_key = ''
_inference_state = None

def get_inference_state(video_dir):
    global _predictor_state_key, _inference_state
    if _predictor_state_key != video_dir:
        _predictor_state_key = video_dir
        _inference_state = predictor.init_state(video_path=video_dir)
    return _inference_state

# Define the model class
class SegmentAnything2VideoModel(LabelStudioMLBase):
    """Custom ML Backend model for Segment Anything 2 with videos"""
    # Split the video into frames
    def split_frames(self, video_path, temp_dir, start_frame, end_frame):
        # Open the video file
        logger.debug(f'Opening video file: {video_path}')
        video = cv2.VideoCapture(video_path)

        # Check if the video is loaded correctly
        if not video.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        else:
            # Display the number of frames
            logger.debug(f'Number of frames: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))}')
            
        # Log the frames to be split
        logger.debug(f'Splitting frames from {start_frame} to {end_frame}')

        # Initialize the frame count
        frame_count = 0
        # Loop through the frames
        while True:
            # Read a frame from the video file
            success, frame = video.read()
            # Skip frames before the start frame
            if frame_count < start_frame:
                logger.debug(f'Skipping frame {frame_count}')
                frame_count += 1
                continue
            # Stop at the end frame
            if frame_count > end_frame:
                logger.debug(f'Stopping at frame {frame_count}')
                break

            # If the frame is not read correctly, break the loop
            if not success:
                logger.error(f'Failed to read frame {frame_count}')
                break

            # Generate a filename for the frame using the pattern with frame number: '%05d.jpg'
            frame_filename = os.path.join(temp_dir, f'{frame_count:05d}.jpg')
            # If the frame already exists, yield the frame
            if os.path.exists(frame_filename):
                logger.debug(f'Frame {frame_count}: {frame_filename} already exists')
                yield frame_filename, frame
            # If the frame does not exist, save it as an image file
            else:
                # Save the frame as an image file and yield it
                cv2.imwrite(frame_filename, frame)
                logger.debug(f'Frame {frame_count}: {frame_filename}')
                yield frame_filename, frame

            frame_count += 1

        # Release the video file
        video.release()

    # Get the prompts from the context
    def get_prompts(self, context) -> List[Dict]:
        # Log the prompts to be extracted
        logger.debug(f'Extracting keypoints from context: {context}')
        prompts = []
        # If there is no context or the context result is not available, return empty prompts
        if not context or 'result' not in context or context['result'] is None:
            # Log the empty prompts
            logger.debug('No context result available; returning empty prompts')
            return prompts
        # Loop through the context result
        for ctx in context['result']:
            # Get the object id
            # Loop through each video tracking object separately
            obj_id = ctx['id']
            for obj in ctx['value']['sequence']:
                x = obj['x'] / 100
                y = obj['y'] / 100
                box_width = obj['width'] / 100
                box_height = obj['height'] / 100
                frame_idx = obj['frame'] - 1
                
                xmin = x
                ymin = y
                xmax = x + box_width
                ymax = y + box_height

                # SAM2 video works with keypoints - convert the rectangle to the set of keypoints within the rectangle
                # bbox (x, y) is top-left corner
                kps = [
                    # center of the bbox
                    [x + box_width / 2, y + box_height / 2],
                    # half of the bbox width to the left
                    [x + box_width / 4, y + box_height / 2],
                    # half of the bbox width to the right
                    [x + 3 * box_width / 4, y + box_height / 2],
                    # half of the bbox height to the top
                    [x + box_width / 2, y + box_height / 4],
                    # half of the bbox height to the bottom
                    [x + box_width / 2, y + 3 * box_height / 4]
                ]

                bbox = [xmin, ymin, xmax, ymax]

                points = np.array(kps, dtype=np.float32)
                labels = np.array([1] * len(kps), dtype=np.int32)
                prompts.append({
                    'points': points,
                    'labels': labels,
                    'frame_idx': frame_idx,
                    'obj_id': obj_id,
                    'bbox': bbox
                })
        return prompts

    # Get the frames count and duration
    def _get_fps(self, context, task, video_path):
        # Try to get the frames count and duration from the context
        try:
            if context and context.get("result"):
                value = context["result"][0].get("value", {})
                fc = value.get("framesCount")
                dur = value.get("duration")
                if fc and dur:
                    return int(fc), float(dur)
        except Exception:
            pass

        # If the context is not available, read the frames count and duration from the video file
        logger.warning("Reading FPS/frames directly from video file")

        # Open the video file and check if it is loaded correctly
        cap = cv2.VideoCapture(video_path)
        # If the video file is not opened correctly, raise an error
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for metadata: {video_path}")

        # Get the frames count and fps from the video file
        frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        duration = frames_count / fps

        # Release the video file
        cap.release()

        # Return the frames count and duration
        return frames_count, duration
    
    # def convert_mask_to_bbox(self, mask):
    #     # convert mask to bbox
    #     h, w = mask.shape[-2:]
    #     mask_int = mask.reshape(h, w, 1).astype(np.uint8)
    #     contours, _ = cv2.findContours(mask_int, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     if len(contours) == 0:
    #         return None
    #     x, y, w, h = cv2.boundingRect(contours[0])
    #     return {
    #         'x': x,
    #         'y': y,
    #         'width': w,
    #         'height': h
    #     }

    def convert_mask_to_bbox(self, mask):
        # Squeeze the mask to remove the last dimension
        mask = mask.squeeze()

        # Get the y and x indices where the mask is 1
        y_indices, x_indices = np.where(mask == 1)
        # If there are no y or x indices, return None
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None

        # Find the min and max indices
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)

        # Get mask dimensions
        height, width = mask.shape

        # Calculate bounding box dimensions
        box_width = xmax - xmin + 1
        box_height = ymax - ymin + 1

        # Normalize and scale to percentage
        x_pct = (xmin / width) * 100
        y_pct = (ymin / height) * 100
        width_pct = (box_width / width) * 100
        height_pct = (box_height / height) * 100

        return {
            "x": round(x_pct, 2),
            "y": round(y_pct, 2),
            "width": round(width_pct, 2),
            "height": round(height_pct, 2)
        }


    def dump_image_with_mask(self, frame, mask, output_file, obj_id=None, random_color=False):
        from matplotlib import pyplot as plt
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        # create an image file to display image overlayed with mask
        mask_image = (mask_image * 255).astype(np.uint8)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGRA2BGR)
        mask_image = cv2.addWeighted(frame, 1.0, mask_image, 0.8, 0)
        #logger.debug(f'Shapes: frame={frame.shape}, mask={mask.shape}, mask_image={mask_image.shape}')
        # save in file
        #logger.debug(f'Saving image with mask to {output_file}')
        cv2.imwrite(output_file, mask_image)

    def dumped_images_to_video(self, debug_dir):
        # get all images in the debug directory
        images = [f for f in os.listdir(debug_dir) if f.endswith('.jpg')]
        # sort images by name
        images.sort()
        # create a video from the images
        video = cv2.VideoWriter(debug_dir + '/video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
        for image in images:
            video.write(cv2.imread(os.path.join(debug_dir, image)))
        # release the video writer
        video.release()


    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Returns the predicted mask for a smart keypoint that has been placed."""

        from_name, to_name, value = self.get_first_tag_occurence('VideoRectangle', 'Video')

        task = tasks[0]
        task_id = task['id']
        # Get the video URL from the task
        video_url = task['data'][value]

        # cache the video locally
        #video_path = get_local_path(video_url, task_id=task_id)
        video_path = self.get_local_path(
            video_url,
            ls_access_token=LABEL_STUDIO_ACCESS_TOKEN,
            ls_host=LABEL_STUDIO_HOST,
            task_id=task_id
        )
        logger.debug(f'Video path: {video_path}')

        # get prompts from context
        prompts = self.get_prompts(context) 
        if not prompts:
            logger.warning("No prompts provided; skipping SAM2 tracking")
            return ModelResponse(predictions=[])
        all_obj_ids = set(p['obj_id'] for p in prompts)
        # create a map from obj_id to integer
        obj_ids = {obj_id: i for i, obj_id in enumerate(all_obj_ids)}
        # find the last frame index
        first_frame_idx = min(p['frame_idx'] for p in prompts) if prompts else 0
        frames_count, duration = self._get_fps(context, task, video_path)
        fps = frames_count / duration if duration > 0 else 30  # default fps fallback
        frames_to_track = MAX_FRAMES_TO_TRACK

        logger.debug(
            f'Number of prompts: {len(prompts)}, '
            f'first frame index: {first_frame_idx}, '
            f'obj_ids: {obj_ids}',
            f'fps: {fps}',
            f'frames to track: {frames_to_track}')

        # Split the video into frames
        with tempfile.TemporaryDirectory() as temp_dir:

            # # use persisted dir for debug
            # temp_dir = '/tmp/frames'
            # os.makedirs(temp_dir, exist_ok=True)

            # get all frames
            frames = list(self.split_frames(
                video_path, temp_dir,
                start_frame=first_frame_idx,
                end_frame=first_frame_idx + frames_to_track + 1
            ))
            height, width, _ = frames[0][1].shape
            logger.debug(f'Video width={width}, height={height}')

            # get inference state
            inference_state = get_inference_state(temp_dir)
            predictor.reset_state(inference_state)

            for prompt in prompts:
                # multiply points by the frame size
                if 'bbox' in prompt:
                    logger.debug(f"Adding new box: {prompt['bbox']}")
                    prompt['bbox'] = [prompt['bbox'][0] * width, prompt['bbox'][1] * height, prompt['bbox'][2] * width, prompt['bbox'][3] * height]
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=prompt['frame_idx']- first_frame_idx, 
                        obj_id=obj_ids[prompt['obj_id']],
                        box=prompt['bbox']
                    )
                # If there are points, add them to the predictor
                if 'points' in prompt:
                    logger.debug(f"Adding new points: {prompt['points']}")
                    prompt['points'][:, 0] *= width
                    prompt['points'][:, 1] *= height

                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=prompt['frame_idx']-first_frame_idx,
                        obj_id=obj_ids[prompt['obj_id']],
                        points=prompt['points'],
                        labels=prompt['labels']
                    )

            sequence = []

            debug_dir = '/app/debug-frames'
            #os.makedirs(debug_dir, exist_ok=True)

            logger.debug(f'Propagating in video from frame {first_frame_idx} to {first_frame_idx + frames_to_track}')
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=0,
                max_frame_num_to_track=frames_to_track
            ):
                real_frame_idx = out_frame_idx + first_frame_idx
                for i, out_obj_id in enumerate(out_obj_ids):
                    logger.debug(f'out_frame_idx: {out_frame_idx}, out_obj_id: {out_obj_id}, real_frame_idx: {real_frame_idx}')
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy()

                    # to debug, save the mask as an image
                    self.dump_image_with_mask(frames[out_frame_idx][1], mask, f'{debug_dir}/{out_frame_idx:05d}_{out_obj_id}.jpg', obj_id=out_obj_id, random_color=False)

                    bbox = self.convert_mask_to_bbox(mask)
                    if bbox:
                        sequence.append({
                            'frame': real_frame_idx + 1,
                            # 'x': bbox['x'] / width * 100,
                            # 'y': bbox['y'] / height * 100,
                            # 'width': bbox['width'] / width * 100,
                            # 'height': bbox['height'] / height * 100,
                            'x': bbox['x'],
                            'y': bbox['y'],
                            'width': bbox['width'],
                            'height': bbox['height'],
                            'enabled': True,
                            'rotation': 0,
                            'time': out_frame_idx / fps
                        })
                        #logger.debug(f'sequence: {sequence}')
            self.dumped_images_to_video(debug_dir)
            context_result_sequence = context['result'][0]['value']['sequence']
            logger.debug(f'context_result_sequence: {context_result_sequence}')

            prediction = PredictionValue(
                result=[{
                    'value': {
                        'framesCount': frames_count,
                        'duration': duration,
                        'sequence': context_result_sequence + sequence,
                    },
                    'from_name': 'box',
                    'to_name': 'video',
                    'type': 'videorectangle',
                    'origin': 'manual',
                    # TODO: current limitation is tracking only one object
                    'id': list(all_obj_ids)[0]
                }]
            )
            logger.debug(f'Prediction: {prediction.model_dump()}')
            
            # reset state
            predictor.reset_state(inference_state)

            return ModelResponse(predictions=[prediction])
