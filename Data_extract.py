import glob
from shutil import copyfile
emotions = {"AF": "fear", "AN" : "anger", "DI" : "disgust", "HA": "happy", "NE" : "neutral", "SA" : "sadness", "SU" : "surprise"}
participants = glob.glob("KDEF/*") #Returns a list of all folders with participant numbers
for x in participants:
	for files in glob.glob("%s/*" %x):
		#print(files)
		name = files[10:-4]
		print(name)
		if (len(name) < 6):
			continue
		emotion = emotions[name[4:6]]
		print (emotion)
		dest_emot = "sorted_set/%s/%s" %(emotion, files[10:])
		copyfile(files, dest_emot)