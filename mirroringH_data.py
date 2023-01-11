import json
import os
import sys
import math
import cv2

if len(sys.argv)<2:
    print("You must specify a json/txt file")
    exit()
	
if sys.argv[1].endswith('.mp4'):
	filenames = [sys.argv[1]]
else:
	filenames = [os.path.join(sys.argv[1], f)  for f in os.listdir(sys.argv[1])]


for filename in filenames:
	
	if not filename.endswith('.mp4'):
		continue
	
	print(filename)

	file_path, video_file = os.path.split(filename)
	video_save = file_path + '/mH_'+ video_file
	if 'mH_' in filename:
		continue

	video = cv2.VideoCapture(filename)
	
	if video.isOpened() == False:
		print("Error reading video file", filename)
		continue

	frame_width = int(video.get(3))
	frame_height = int(video.get(4))

	size = (frame_width, frame_height)

	result = cv2.VideoWriter(video_save, cv2.VideoWriter_fourcc(*'MP4V'), 10, size)
	
	while(True):
		ret, frame = video.read()
		flipped_frame = cv2.flip(frame, 0)
		if ret:
			result.write(flipped_frame)
		else:
			break

	video.release()
	result.release()
