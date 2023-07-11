import pandas as pd
import glob

from pose import Pose


##### get training data size #####
# we have 38 (=2x17) of real data, and a 27 long one hot encoding (space key is to be added later)
path = "data/"
count = 0
for i in range(1,27):
    filepaths = glob.glob("{}/{}/*.png".format(path,str(i)))
    for f in filepaths:
        count += 1
print("There are {} datapoints".format(count))


####### extract pose data from training images #########

pose = Pose()
df = pd.DataFrame(columns=range(38),index=range(count))
labels = pd.DataFrame(columns=range(27), index=range(count))
###### NB: labels 0 column is a space, and the real data has column indices 1 through 26

ct = 0
for i in range(1,27):
    filepaths = glob.glob("{}/{}/*.png".format(path,str(i)))
    label = [0]*27
    label[i] = 1
    for j,f in enumerate(filepaths):
        frame = cv2.imread(f)
        pose.compute_frame(frame)
        df.loc[ct] = pose.return_keypoint_vector()[1]
        labels.loc[ct] = label
        ct += 1
        # print(pose.return_keypoint_vector())
        # pose.show_pose_landmarks()

########## saving data ########
data_path = 'data.csv'
label_path = 'labels.csv'

df.to_csv(data_path)
labels.to_csv(label_path)
