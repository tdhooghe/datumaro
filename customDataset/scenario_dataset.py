# %%
import os
import cv2 as cv
import datumaro as dm

# Opens the Video file
cap = cv.VideoCapture('/home/thomas/Datasets/scenario/scenario.mp4')
stats = dict()
stats['num_frames'] = cap.get(cv.CAP_PROP_FRAME_COUNT)
stats['fps'] = cap.get(cv.CAP_PROP_FPS)
print(f'stats: {stats}')
num_images = stats['num_frames'] / 15
print(f'num_images: {num_images}')

# %%
i = 0

write_folder = '/home/thomas/Datasets/scenario/hd'
if not os.path.exists(write_folder):
    os.makedirs(write_folder)
os.chdir(write_folder)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:  # if frame is read correctly ret is true
        print("Can't receive frame (stream end?). Exiting ...")
        break
    if i % 15 == 0:  # save one frame every 15
        resized = cv.resize(frame, (1280, 720))
        cv.imwrite('frame' + str(i) + '.jpg', resized)
        cv.imshow('frame', resized)
        if cv.waitKey(1) == ord('q'):
            break
    i += 1
    if i == stats['num_frames']:
        break
cap.release()
cv.destroyAllWindows()

# %%
dm.Dataset.import_from('/home/thomas/Datasets/SCENARIO/cvat/4k/', 'cvat') \
    .export('/home/thomas/Datasets/SCENARIO/yolo/4k/', 'yolo', save_images=True)
