import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, python_speech_features

import gradio as gr
from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from model.faceDetector.s3fd import S3FD
from talkNet import talkNet
from deepface import DeepFace
import threading
import torch
import gc
import uuid
from datetime import datetime
import json

lock = threading.Lock()  

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "TalkNet Demo or Columnbia ASD Evaluation")

parser.add_argument('--videoName',             type=str, default="001",   help='Demo video name')
parser.add_argument('--videoFolder',           type=str, default="demo",  help='Path for inputs, tmps and outputs')
parser.add_argument('--pretrainModel',         type=str, default="pretrain_TalkSet.model",   help='Path for the pretrained TalkNet model')

parser.add_argument('--nDataLoaderThread',     type=int,   default=20,   help='Number of workers')
parser.add_argument('--facedetScale',          type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
parser.add_argument('--minTrack',              type=int,   default=15,   help='Number of min frames for each shot')
parser.add_argument('--numFailedDet',          type=int,   default=10,   help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--minFaceSize',           type=int,   default=1,    help='Minimum face size in pixels')
parser.add_argument('--cropScale',             type=float, default=0.40, help='Scale bounding box')

parser.add_argument('--start',                 type=int, default=0,   help='The start time of the video')
parser.add_argument('--duration',              type=int, default=0,  help='The duration of the video, when set as 0, will extract the whole video')

parser.add_argument('--evalCol',               dest='evalCol', action='store_true', help='Evaluate on Columnbia dataset')
parser.add_argument('--colSavePath',           type=str, default="/data08/col",  help='Path for inputs, tmps and outputs')


PROCESS_FPS = 25
MODEL_PROCESS_FPS = 25
#INTERMIDIATE_FOLDER = '/opt/oss/noiz_models/yonghui/talknet'
INTERMIDIATE_FOLDER = './intermediate_result'

def release_cuda_memory():
    device = torch.device('cuda:0')
    free, total = torch.cuda.mem_get_info(device)
    if (1.0 * free / total) < 0.2:
        gc.collect()
        torch.cuda.empty_cache()
        new_free, total = torch.cuda.mem_get_info(device)
        print("release cuda memory, size(MB):", (new_free - free)/1024.0/1024.0)

class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.result = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self)  # 等待线程执行完毕
        try:
            return self.result
        except Exception:
            return None
 


def init_args():
	args = parser.parse_args()

	if os.path.isfile(args.pretrainModel) == False: # Download the pretrained model
		Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
		cmd = "gdown --id %s -O %s"%(Link, args.pretrainModel)
		subprocess.call(cmd, shell=True, stdout=None)

	if args.evalCol == True:
		# The process is: 1. download video and labels(I have modified the format of labels to make it easiler for using)
		# 	              2. extract audio, extract video frames
		#                 3. scend detection, face detection and face tracking
		#                 4. active speaker detection for the detected face clips
		#                 5. use iou to find the identity of each face clips, compute the F1 results
		# The step 1 to 3 will take some time (That is one-time process). It depends on your cpu and gpu speed. For reference, I used 1.5 hour
		# The step 4 and 5 need less than 10 minutes
		# Need about 20G space finally
		# ```
		args.videoName = 'col'
		args.videoFolder = args.colSavePath
		args.savePath = os.path.join(args.videoFolder, args.videoName)
		args.videoPath = os.path.join(args.videoFolder, args.videoName + '.mp4')
		args.duration = 0
		if os.path.isfile(args.videoPath) == False:  # Download video
			link = 'https://www.youtube.com/watch?v=6GzxbrO0DHM&t=2s'
			cmd = "youtube-dl -f best -o %s '%s'"%(args.videoPath, link)
			output = subprocess.call(cmd, shell=True, stdout=None)
		if os.path.isdir(args.videoFolder + '/col_labels') == False: # Download label
			link = "1Tto5JBt6NsEOLFRWzyZEeV6kCCddc6wv"
			cmd = "gdown --id %s -O %s"%(link, args.videoFolder + '/col_labels.tar.gz')
			subprocess.call(cmd, shell=True, stdout=None)
			cmd = "tar -xzvf %s -C %s"%(args.videoFolder + '/col_labels.tar.gz', args.videoFolder)
			subprocess.call(cmd, shell=True, stdout=None)
			os.remove(args.videoFolder + '/col_labels.tar.gz')	
	else:
		args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + '.*'))[0]
		args.savePath = os.path.join(args.videoFolder, args.videoName)
	return args

def scene_detect(args):
	# CPU: Scene detection, output is the list of each shot's time duration
	videoManager = VideoManager([args.videoFilePath])
	statsManager = StatsManager()
	sceneManager = SceneManager(statsManager)
	sceneManager.add_detector(ContentDetector())
	baseTimecode = videoManager.get_base_timecode()
	videoManager.set_downscale_factor()
	videoManager.start()
	sceneManager.detect_scenes(frame_source = videoManager)
	sceneList = sceneManager.get_scene_list(baseTimecode)
	savePath = os.path.join(args.pyworkPath, 'scene.pckl')
	if sceneList == []:
		sceneList = [(videoManager.get_base_timecode(),videoManager.get_current_timecode())]
	with open(savePath, 'wb') as fil:
		pickle.dump(sceneList, fil)
		sys.stderr.write('%s - scenes detected %d\n'%(args.videoFilePath, len(sceneList)))
	return sceneList

def inference_video(args):
	# GPU: Face detection, output is the list contains the face location and score in this frame
	DET = S3FD(device='cuda')
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	flist.sort()
	dets = []
	for fidx, fname in enumerate(flist):
		image = cv2.imread(fname)
		imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
		dets.append([])
		for bbox in bboxes:
		  dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
		sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))
	savePath = os.path.join(args.pyworkPath,'faces.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(dets, fil)
	return dets


def batch_inference_video(args):
	# GPU: Face detection, output is the list contains the face location and score in this frame
	DET = S3FD(device='cuda')
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	flist.sort()
	dets = []
	input_images = []
	for fidx, fname in enumerate(flist):
		image = cv2.imread(fname)
		imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		input_images.append(imageNumpy)
	bboxes_array = DET.batch_detect_faces(input_images, conf_th=0.9, scale = args.facedetScale)
	for fidx in range(len(bboxes_array)):
		bboxes = bboxes_array[fidx]
		dets.append([])
		for bbox in bboxes:
		  dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
		sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))
	savePath = os.path.join(args.pyworkPath,'faces.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(dets, fil)
	return dets

def bb_intersection_over_union(boxA, boxB, evalCol = False):
	# CPU: IOU Function to calculate overlap between two image
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	if evalCol == True:
		iou = interArea / float(boxAArea)
	else:
		iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def track_shot(args, sceneFaces):
	# CPU: Face tracking
	iouThres  = 0.5     # Minimum IOU between consecutive face detections
	tracks    = []
	while True:
		track     = []
		for frameFaces in sceneFaces:
			for face in frameFaces:
				if track == []:
					track.append(face)
					frameFaces.remove(face)
				elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
					iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
					if iou > iouThres:
						track.append(face)
						frameFaces.remove(face)
						continue
				else:
					break
		if track == []:
			break
		elif len(track) > args.minTrack:
			frameNum    = numpy.array([ f['frame'] for f in track ])
			bboxes      = numpy.array([numpy.array(f['bbox']) for f in track])
			frameI      = numpy.arange(frameNum[0],frameNum[-1]+1)
			bboxesI    = []
			for ij in range(0,4):
				interpfn  = interp1d(frameNum, bboxes[:,ij])
				bboxesI.append(interpfn(frameI))
			bboxesI  = numpy.stack(bboxesI, axis=1)
			if max(numpy.mean(bboxesI[:,2]-bboxesI[:,0]), numpy.mean(bboxesI[:,3]-bboxesI[:,1])) > args.minFaceSize:
				tracks.append({'frame':frameI,'bbox':bboxesI})
	return tracks

def face_similarity(args, tid1, tid2):
	total_sim = 0.0
	models = [
		"VGG-Face", 
		"Facenet", 
		"Facenet512", 
		"OpenFace", 
		"DeepFace", 
		"DeepID", 
		"ArcFace", 
		"Dlib", 
		"SFace",
		"GhostFaceNet",
	]

	for face_id in range(3):
		file_name1 = args.pycropPath + "/" + str(tid1) + "_" + str(face_id) + ".jpg"
		file_name2 = args.pycropPath + "/" + str(tid2) + "_" + str(face_id) + ".jpg"
		result = DeepFace.verify(
			img1_path = file_name1,
			img2_path = file_name2,
			enforce_detection = False,
			model_name = models[8],
		)
		print("result:", result)
		total_sim = total_sim + result['distance']
	return total_sim / 3.0


def crop_video(args, track, cropFile):
	# CPU: crop the face clips
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Read the frames
	flist.sort()
	vOut = cv2.VideoWriter(cropFile + '_t.avi', cv2.VideoWriter_fourcc(*'XVID'), PROCESS_FPS, (224,224))# Write video
	dets = {'x':[], 'y':[], 's':[]}
	for det in track['bbox']: # Read the tracks
		dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
		dets['y'].append((det[1]+det[3])/2) # crop center x 
		dets['x'].append((det[0]+det[2])/2) # crop center y
	dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
	dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
	dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
	total_frames = len(track['frame'])
	tid = track['tid']
	face_id = 0
	pick_step = int(total_frames/5.0)
	for fidx, frame in enumerate(track['frame']):
		cs  = args.cropScale
		bs  = dets['s'][fidx]   # Detection box size
		bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
		image = cv2.imread(flist[frame])
		frame = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
		my  = dets['y'][fidx] + bsi  # BBox center Y
		mx  = dets['x'][fidx] + bsi  # BBox center X
		x1 =  int(my-bs)
		x2 =  int(my+bs*(1+2*cs))
		#if x1 > 10:
		#	x1 = x1 - 10
		y1 = int(mx-bs*(1+cs))
		y2 = int(mx+bs*(1+cs))
		#if y1 > 10:
		#	y1 = y1 - 10
		face = frame[x1:x2,y1:y2]
		if face.size <= 0:
			continue
		vOut.write(cv2.resize(face, (224, 224)))
		if fidx % pick_step == 0:
			file_name = args.pycropPath + "/" + str(tid) + "_" + str(face_id) + ".jpg"
			cv2.imwrite(file_name, face)
			face_id = face_id + 1
	
	audioTmp    = cropFile + '.wav'
	audioStart  = (track['frame'][0]) / PROCESS_FPS
	audioEnd    = (track['frame'][-1]+1) / PROCESS_FPS
	vOut.release()


	command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
		      (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp)) 
	output = subprocess.call(command, shell=True, stdout=None) # Crop audio file

	_, audio = wavfile.read(audioTmp)
	command = ("ffmpeg -y -i %s_t.avi -i %s -threads %d -c:v copy -c:a copy %s_t2.avi -loglevel panic" % \
			  (cropFile, audioTmp, args.nDataLoaderThread, cropFile)) # Combine audio and video file
	output = subprocess.call(command, shell=True, stdout=None)

	command = (f"ffmpeg -i %s_t2.avi -i %s -threads %d  -filter:0 \"minterpolate='fps={MODEL_PROCESS_FPS}'\" %s.avi -loglevel panic" % \
			  (cropFile, audioTmp, args.nDataLoaderThread, cropFile)) # Combine audio and video file
	output = subprocess.call(command, shell=True, stdout=None)


	os.remove(cropFile + '_t.avi')
	os.remove(cropFile + '_t2.avi')
	#os.remove(cropFile + '.avi')
	#print("crop video finished and removed:", cropFile + '.avi')
	return {'track':track, 'proc_track':dets, "tid":tid, "face_count":face_id}

def extract_MFCC(file, outPath):
	# CPU: extract mfcc
	sr, audio = wavfile.read(file)
	mfcc = python_speech_features.mfcc(audio,sr) # (N_frames, 13)   [1s = 100 frames]
	featuresPath = os.path.join(outPath, file.split('/')[-1].replace('.wav', '.npy'))
	numpy.save(featuresPath, mfcc)

def evaluate_network(files, args):
	# GPU: active speaker detection by pretrained TalkNet
	s = talkNet()
	s.loadParameters(args.pretrainModel)
	sys.stderr.write("Model %s loaded from previous state! \r\n"%args.pretrainModel)
	s.eval()
	allScores = []
	# durationSet = {1,2,4,6} # To make the result more reliable
	durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
	for file in tqdm.tqdm(files, total = len(files)):
		fileName = os.path.splitext(file.split('/')[-1])[0] # Load audio and video
		_, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
		audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
		video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
		videoFeature = []
		while video.isOpened():
			ret, frames = video.read()
			if ret == True:
				face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
				face = cv2.resize(face, (224,224))
				face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
				videoFeature.append(face)
			else:
				break
		video.release()
		videoFeature = numpy.array(videoFeature)
		length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0] / MODEL_PROCESS_FPS)
		audioFeature = audioFeature[:int(round(length * 100)),:]
		videoFeature = videoFeature[:int(round(length * MODEL_PROCESS_FPS)),:,:]
		allScore = [] # Evaluation use TalkNet
		for duration in durationSet:
			batchSize = int(math.ceil(length / duration))
			scores = []
			with torch.no_grad():
				for i in range(batchSize):
					inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()
					inputV = torch.FloatTensor(videoFeature[i * duration * MODEL_PROCESS_FPS: (i+1) * duration * MODEL_PROCESS_FPS,:,:]).unsqueeze(0).cuda()
					embedA = s.model.forward_audio_frontend(inputA)
					embedV = s.model.forward_visual_frontend(inputV)	
					embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
					out = s.model.forward_audio_visual_backend(embedA, embedV)
					score = s.lossAV.forward(out, labels = None)
					scores.extend(score)
			allScore.append(scores)
		allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
		allScores.append(allScore)	
	return allScores

def visualization(tracks, scores, args, return_timestamp_only):
	# CPU: visulize the result for video format
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	flist.sort()
	faces = [[] for i in range(len(flist))]
	for tidx, track in enumerate(tracks):
		score = scores[tidx]
		for fidx, frame in enumerate(track['track']['frame'].tolist()):
			s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
			s = numpy.mean(s)
			faces[frame].append({'track':track['tid'], 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
	speaker_track_ids = []
	for frame in faces:
		max_score = -100
		speaker_track_id = -1
		for i in range(len(frame)):
			track = frame[i]
			if track['score'] > max_score:
				max_score = track['score']
				speaker_track_id = track['track']
		speaker_track_ids.append(speaker_track_id)
	speaker_change_info = []
	for i in range(len(speaker_track_ids)):
		speaker_change_ts = {}
		if i == 0:
			speaker_change_ts['ts'] = i / float(PROCESS_FPS)
			speaker_change_ts['tid'] = speaker_track_ids[i]
			speaker_change_info.append(speaker_change_ts)
			continue
		if speaker_track_ids[i] != speaker_track_ids[i - 1]:
			speaker_change_ts['ts'] = i / float(PROCESS_FPS)
			speaker_change_ts['tid'] = speaker_track_ids[i]
			speaker_change_info.append(speaker_change_ts)
	for i in range(len(speaker_change_info)):
		ts = speaker_change_info[i]['ts']
		if i < len(speaker_change_info) - 1:
			next_ts = speaker_change_info[i + 1]['ts']
			speaker_change_info[i]['duration'] = (next_ts - ts)
		else:
			speaker_change_info[i]['duration'] = 100
		speaker_change_info[i]['remove'] = False
	min_unknown_dur = 0.5
	print("speaker_change_info before remove short unknown slice:", speaker_change_info)
	# remove unknwon slice shorter than 0.5 second
	for i in range(len(speaker_change_info) - 1):
		dur = speaker_change_info[i]['duration']
		tid = speaker_change_info[i]['tid']
		if i > 0 and tid == -1 and dur < min_unknown_dur:
			speaker_change_info[i + 1]['ts'] = speaker_change_info[i + 1]['ts'] - dur / 2.0
			speaker_change_info[i + 1]['duration'] = speaker_change_info[i + 1]['duration'] + dur / 2.0
			speaker_change_info[i - 1]['duration'] = speaker_change_info[i - 1]['duration'] +  dur / 2.0
			speaker_change_info[i]['remove'] = True
	filtered_speaker_change_info = []
	for item in speaker_change_info:
		if not item['remove']:
			filtered_speaker_change_info.append(item)
	speaker_change_info = filtered_speaker_change_info
	filtered_speaker_change_info = []
	print("speaker_change_info after remove short unknown slice:", speaker_change_info)
	# combine same speaker track
	for i in range(len(speaker_change_info)):
		prev_item = None
		tid = speaker_change_info[i]['tid']
		if i > 0:
			prev_tid = speaker_change_info[i - 1]['tid']
			if prev_tid != -1 and tid != -1 and face_similarity(args, prev_tid, tid) < 0.4:
				speaker_change_info[i]['remove'] = True
				speaker_change_info[i - 1]['duration'] = speaker_change_info[i - 1]['duration']  + speaker_change_info[i]['duration']
	for item in speaker_change_info:
		if not item['remove']:
			filtered_speaker_change_info.append(item)
	speaker_change_info = filtered_speaker_change_info 
	print("final speaker_change_ts:", speaker_change_info)
	speaker_change_info = json.dumps(speaker_change_info)
	if return_timestamp_only:
		return speaker_change_info
	firstImage = cv2.imread(flist[0])
	fw = firstImage.shape[1]
	fh = firstImage.shape[0]
	vOut = cv2.VideoWriter(os.path.join(args.pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), PROCESS_FPS, (fw,fh))
	colorDict = {0: 0, 1: 255}
	for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
		image = cv2.imread(fname)
		for face in faces[fidx]:
			clr = colorDict[int((face['score'] >= 0))]
			txt = round(face['score'], 1)
			cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])),(0,clr,255-clr),10)
			cv2.putText(image,'%s'%(txt), (int(face['x']-face['s']), int(face['y']-face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
		vOut.write(image)
	vOut.release()
	command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
		(os.path.join(args.pyaviPath, 'video_only.avi'), os.path.join(args.pyaviPath, 'audio.wav'), \
		args.nDataLoaderThread, os.path.join(args.pyaviPath,'video_out.avi'))) 
	output = subprocess.call(command, shell=True, stdout=None)
	return speaker_change_info

# Main function
def main(input_args, return_timestamp_only):
	# This preprocesstion is modified based on this [repository](https://github.com/joonson/syncnet_python).
	# ```
	# .
	# ├── pyavi
	# │   ├── audio.wav (Audio from input video)
	# │   ├── video.avi (Copy of the input video)
	# │   ├── video_only.avi (Output video without audio)
	# │   └── video_out.avi  (Output video with audio)
	# ├── pycrop (The detected face videos and audios)
	# │   ├── 000000.avi
	# │   ├── 000000.wav
	# │   ├── 000001.avi
	# │   ├── 000001.wav
	# │   └── ...
	# ├── pyframes (All the video frames in this video)
	# │   ├── 000001.jpg
	# │   ├── 000002.jpg
	# │   └── ...	
	# └── pywork
	#     ├── faces.pckl (face detection result)
	#     ├── scene.pckl (scene detection result)
	#     ├── scores.pckl (ASD result)
	#     └── tracks.pckl (face tracking result)
	# ```

	# Initialization 
	input_args.pyaviPath = os.path.join(input_args.savePath, 'pyavi')
	input_args.pyframesPath = os.path.join(input_args.savePath, 'pyframes')
	input_args.pyworkPath = os.path.join(input_args.savePath, 'pywork')
	input_args.pycropPath = os.path.join(input_args.savePath, 'pycrop')
	if os.path.exists(input_args.savePath):
		rmtree(input_args.savePath)
	os.makedirs(input_args.pyaviPath, exist_ok = True) # The path for the input video, input audio, output video
	os.makedirs(input_args.pyframesPath, exist_ok = True) # Save all the video frames
	os.makedirs(input_args.pyworkPath, exist_ok = True) # Save the results in this process by the pckl method
	os.makedirs(input_args.pycropPath, exist_ok = True) # Save the detected face clips (audio+video) in this process

	# Extract video
	input_args.videoFilePath = os.path.join(input_args.pyaviPath, 'video.avi')
	# If duration did not set, extract the whole video, otherwise extract the video from 'args.start' to 'args.start + args.duration'
	if input_args.duration == 0:
		command = (f"ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r {PROCESS_FPS} %s -loglevel panic" % \
			(input_args.videoPath, input_args.nDataLoaderThread, input_args.videoFilePath))
	else:
		command = (f"ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r {PROCESS_FPS} %s -loglevel panic" % \
			(input_args.videoPath, input_args.nDataLoaderThread, input_args.start, input_args.start + input_args.duration, input_args.videoFilePath))
	subprocess.call(command, shell=True, stdout=None)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the video and save in %s \r\n" %(input_args.videoFilePath))
	
	# Extract audio
	input_args.audioFilePath = os.path.join(input_args.pyaviPath, 'audio.wav')
	command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % \
		(input_args.videoFilePath, input_args.nDataLoaderThread, input_args.audioFilePath))
	subprocess.call(command, shell=True, stdout=None)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" %(input_args.audioFilePath))

	# Extract the video frames
	command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" % \
		(input_args.videoFilePath, input_args.nDataLoaderThread, os.path.join(input_args.pyframesPath, '%06d.jpg'))) 
	subprocess.call(command, shell=True, stdout=None)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the frames and save in %s \r\n" %(input_args.pyframesPath))

	# Scene detection for the video frames
	scene = scene_detect(input_args)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scene detection and save in %s \r\n" %(input_args.pyworkPath))	

	# Face detection for the video frames
	faces = batch_inference_video(input_args)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" %(input_args.pyworkPath))

	#print("faces:", faces)
	# Face tracking
	allTracks, vidTracks = [], []
	threads = []
	for shot in scene:
		if shot[1].frame_num - shot[0].frame_num >= input_args.minTrack: # Discard the shot frames less than minTrack frames
			#allTracks.extend(track_shot(input_args, faces[shot[0].frame_num:shot[1].frame_num])) # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
			thread = MyThread(track_shot, (input_args, faces[shot[0].frame_num:shot[1].frame_num]))
			thread.start()
			threads.append(thread)
	for i in range(len(threads)):
		if threads[i].is_alive():
			threads[i].join()
		allTracks.extend(threads[i].get_result())

	for shot in scene:
		if shot[1].frame_num - shot[0].frame_num >= input_args.minTrack: # Discard the shot frames less than minTrack frames
			allTracks.extend(track_shot(input_args, faces[shot[0].frame_num:shot[1].frame_num])) # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" %len(allTracks))
	#print("allTracks:", allTracks)
	#格式： frame 编号， bbox： 脸的box 
	# Face clips cropping
	threads = [None] * len(allTracks)
	vidTracks = []
	for i in range(len(allTracks)):
		allTracks[i]['tid'] = i
		threads[i] = MyThread(crop_video, (input_args, allTracks[i], os.path.join(input_args.pycropPath, '%05d'%i)))
		threads[i].start()
	for i in range(len(threads)):
		if threads[i].is_alive():
			threads[i].join()
		vid_track =  threads[i].get_result()
		if vid_track['face_count'] >= 3:
			vidTracks.append(vid_track)

	savePath = os.path.join(input_args.pyworkPath, 'tracks.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(vidTracks, fil)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop and saved in %s tracks \r\n" %input_args.pycropPath)
	fil = open(savePath, 'rb')
	vidTracks = pickle.load(fil)
	#print("vidTracks:", vidTracks)

	# Active Speaker Detection by TalkNet
	files = glob.glob("%s/*.avi"%input_args.pycropPath)
	files.sort()
	print("files:", files)
	scores = evaluate_network(files, input_args)
	savePath = os.path.join(input_args.pyworkPath, 'scores.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(scores, fil)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted and saved in %s \r\n" %input_args.pyworkPath)
	speaker_change_ts = visualization(vidTracks, scores, input_args, return_timestamp_only)	
	return speaker_change_ts

def process(input_video, return_timestamp_only):
	with lock:
		release_cuda_memory()
		new_args = init_args()
		new_args.videoPath = input_video

		temp_folder = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
		lowercase_str = uuid.uuid4().hex[0:6]
		save_path = INTERMIDIATE_FOLDER + "/" + temp_folder + "_" + lowercase_str

		new_args.videoFolder = save_path
		os.makedirs(save_path, exist_ok=True)
		new_args.savePath = new_args.videoFolder #os.path.join(new_args.videoFolder, "save_path")
		print("new_args:", new_args)
		speaker_change_ts = main(new_args, return_timestamp_only)
		output_video = new_args.savePath + "/pyavi/video_out.avi"
		output_video = os.path.abspath(output_video)
		print("output_video:", output_video)
		release_cuda_memory()
		if return_timestamp_only:
			return None, str(speaker_change_ts)
		return output_video, str(speaker_change_ts)

if __name__ == '__main__':
	demo_inputs = [
		gr.Video(
			sources=["upload"],
			label="Video File(5 seconds to 5 minutes)",
			min_length=5,
			max_length=300
		),
		gr.Dropdown(choices=[
						('No', 0), 
						('YES', 1)], 
					value=1, 
					label="only return timestamp, without video"
		),
	]

	demo_outputs = [
		gr.Video(label="Returned Video", show_download_button=True),
		gr.Textbox(
			label="Speaker Change Timestamp",
			type="text",
		)
	]

	demo = gr.Interface(
		fn=process,
		inputs=demo_inputs,
		outputs=demo_outputs,
		title="Speaker Detection",
	)
	demo.launch(share=False, server_name="0.0.0.0", server_port=8087)
