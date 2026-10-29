#!/usr/bin/python

import time, os, argparse, itertools
import torch
import torch.nn.functional as F

import avsnap_utils as utils
from s3fd import S3FD

# ========== ========== ========== ==========
# # PARSE ARGS
# ========== ========== ========== ==========

parser = argparse.ArgumentParser(description = "FaceTracker");
parser.add_argument('--data_dir', type=str, default='data/', help='Output direcotry');
parser.add_argument('videofile', type=str, default='', help='Input video file');
parser.add_argument('--crop_scale', type=float, default=0.5, help='Scale bounding box');
parser.add_argument('--min_track', type=int, default=30, help='Minimum facetrack duration');
parser.add_argument('--facedet_scale', type=float, default=0.25, help='Scale input image for face detection');
parser.add_argument('--min_conf', type=float, default=0.9, help='Minimum face detection confidence');
opt = parser.parse_args();

base_name = os.path.splitext(os.path.basename(opt.videofile))[0]
setattr(opt,'out_dir', os.path.join(opt.data_dir, 'out', base_name))

# ========== ========== ========== ==========
# # IOU FUNCTION
# ========== ========== ========== ==========

def bb_intersection_over_union(boxA, boxB):

  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  interArea = max(0, xB - xA) * max(0, yB - yA)

  boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

  iou = interArea / float(boxAArea + boxBArea - interArea)

  return iou

# ========== ========== ========== ==========
# # FACE TRACKING
# ========== ========== ========== ==========

def track_shot(opt, scenefaces):

  iouThres  = 0.5     # Minimum IOU between consecutive face detections
  numFail   = 3       # Number of missed detections allowed
  minSize   = 0.05    # Minimum size of faces
  tracks    = []

  while True:
    track     = []
    for faces in scenefaces:
      for face in faces:
        if track == []:
          track.append(face)
          faces.remove(face)
        elif face[0] - track[-1][0] <= numFail:
          iou = bb_intersection_over_union(face[1], track[-1][1])
          if iou > iouThres:
            track.append(face)
            faces.remove(face)
            continue
        else:
          break

    if track == []:
      break
    elif len(track) > opt.min_track:

      framenum    = torch.tensor([ f[0] for f in track ], dtype=torch.float64)
      bboxes    = torch.tensor([ f[1] for f in track ], dtype=torch.float64)

      frame_i   = torch.arange(track[0][0], track[-1][0]+1)

      bboxes_i  = utils.interp(frame_i.double(), framenum, bboxes)

      if torch.mean(bboxes_i[:,3]-bboxes_i[:,1]) > minSize:
        tracks.append([frame_i,bboxes_i])
  return tracks

# ========== ========== ========== ==========
# # VIDEO CROP AND SAVE
# ========== ========== ========== ==========

def pad_span(start, stop, size, pad):
  # slice [start, stop) of an axis of length size zero padded by pad on both sides, in padded coordinates.
  # Returns (zeros before, frame start, frame stop, zeros after).
  start, stop = (min(max(c, 0), size + 2*pad) - pad for c in (start, stop))
  lo, hi = min(max(start, 0), size), min(max(stop, 0), size)
  return max(0, min(stop, 0) - start), lo, max(lo, hi), max(0, stop - max(start, size))

def crop_video(opt,track):

  video_file = os.path.join(opt.out_dir, 'video.avi')
  fw, fh, fps = utils.probe_vid(video_file)
  cropped_file = os.path.join(opt.out_dir, 'cropped.avi')

  dets = [[], [], []]

  for det in track[1].tolist():

    dets[0].append(((det[3]-det[1])*fw+(det[2]-det[0])*fh)/4) # H+W / 4
    dets[1].append((det[1]+det[3])*fw/2) # crop center x
    dets[2].append((det[0]+det[2])*fh/2) # crop center y

  # Smooth detections
  dets[0] = utils.medfilt(torch.tensor(dets[0], dtype=torch.float64),kernel_size=5).tolist()
  dets[1] = utils.medfilt(torch.tensor(dets[1], dtype=torch.float64),kernel_size=5).tolist()
  dets[2] = utils.medfilt(torch.tensor(dets[2], dtype=torch.float64),kernel_size=7).tolist()

  def crop_frames():
    frames = itertools.islice(utils.iter_frames(video_file), track[0][0].item(), None)
    for det, frame in zip(zip(*dets), frames):

      cs  = opt.crop_scale

      bs  = det[0]            # Detection box size
      bsi = int(bs*(1+2*cs))  # Pad videos by this amount

      my  = det[2]+bsi  # BBox center Y
      mx  = det[1]+bsi  # BBox center X

      # same crop as slicing the frame zero padded by bsi on each side, but only the crop itself is padded
      ya, ystart, ystop, yb = pad_span(int(my-bs), int(my+bs*(1+2*cs)), fh, bsi)
      xa, xstart, xstop, xb = pad_span(int(mx-bs*(1+cs)), int(mx+bs*(1+cs)), fw, bsi)
      face = F.pad(frame[ystart:ystop, xstart:xstop].permute(2,0,1), (xa, xb, ya, yb))

      resized = F.interpolate(face[None].float(), size=(224,224), mode='bilinear', antialias=True)[0]
      yield resized.round().clamp(0,255).to(torch.uint8).permute(1,2,0)

  # ========== CROP AUDIO FILE ==========

  audiotmp  = os.path.join(opt.out_dir, 'audio.wav')
  audiostart  = track[0][0].item()/fps
  audioend  = (track[0][-1].item()+1)/fps

  command = ['ffmpeg', '-y', '-i', video_file, '-ac', '1', '-vn', '-acodec', 'pcm_s16le', '-ar', str(utils.SR),
             '-ss', '%.3f' % audiostart, '-to', '%.3f' % audioend, audiotmp]
  utils.run_command(command)

  # ========== WRITE CROPPED VIDEO WITH AUDIO ==========

  utils.write_vid(cropped_file, crop_frames(), fps, audio_path=audiotmp)

  print('Written %s'%cropped_file)


# ========== ========== ========== ==========
# # FACE DETECTION
# ========== ========== ========== ==========

# batching gives no speedup on CPU, and batch 1 keeps the detections bit-identical there
det_batch_size = 16 if utils.device.type == 'cuda' else 1

def inference_video(opt):

  detector = S3FD(os.path.join(utils.data_dir, 'sfd_face.pth'), utils.device)

  video_file = os.path.join(opt.out_dir, 'video.avi')
  print(utils.probe_vid(video_file))

  dets = []
  batch = []

  def detect_batch():
    # Actual detection.
    start_time = time.time()
    boxes = detector.detect_batch([image for _, image in batch], conf_th=opt.min_conf, scale=opt.facedet_scale)
    elapsed_time = time.time() - start_time

    for (frame_num, _), frame_boxes in zip(batch, boxes):
      dets.append([[frame_num, box[:4], box[4]] for box in frame_boxes.tolist()])
      print('%s-%05d; %d dets; %.2f Hz' %
            (video_file, frame_num, len(dets[-1]), (len(batch)/elapsed_time)))
    batch.clear()

  for frame_num, image in enumerate(utils.iter_frames(video_file)):
    batch.append((frame_num, image))
    if len(batch) == det_batch_size:
      detect_batch()
  if batch:
    detect_batch()

  return dets

# ========== ========== ========== ==========
# # EXECUTE DEMO
# ========== ========== ========== ==========

os.makedirs(opt.out_dir, exist_ok=True)

command = ['ffmpeg', '-y', '-i', opt.videofile, '-qscale:v', '4', '-r', str(utils.FPS),
           '-vf', 'yadif=deint=interlaced', '-af', 'aresample=async=1', os.path.join(opt.out_dir, 'video.avi')]
utils.run_command(command)
faces = inference_video(opt)

tracks = track_shot(opt, faces)
crop_video(opt, tracks[0])
