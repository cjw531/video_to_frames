import os
import cv2
import numpy as np

def video_to_frames(video_path, save_path):
    os.system('rm -rf ' + save_path)
    os.system('mkdir ' + save_path)
    
    # reference: https://stackoverflow.com/a/47632941
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        # vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*200)) # added this line every 0.5 sec
        cv2.imwrite(save_path + "/frame_%d.jpg" % count, image) # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def blur_detect(path, selected_path, threshold):
    os.system('rm -rf ' + selected_path)
    os.system('mkdir ' + selected_path)

    for imagePath in sorted(os.listdir(path)):
        # load the image, convert it to grayscale, and compute the
        # focus measure of the image using the Variance of Laplacian
        # method
        image = cv2.imread(path + imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        
        # if the focus measure is less than the supplied threshold,
        # then the image should be considered "blurry"
        if fm < threshold: # threshold is by default
            continue
        else:
            os.system('cp ' + path + imagePath + ' ' + selected_path + imagePath) # save if selected
        # show the image
        # cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        # cv2.imshow("Image", image)
        
        # key = cv2.waitKey(0)
        # if key == 49: 

def img_linspace(img_path, linspace_path, sample_factor):
    # use top 20% and apply the linspace algorithm

    os.system('rm -rf ' + linspace_path)
    os.system('mkdir ' + linspace_path)

    files = sorted(os.listdir(img_path))
    indices = np.linspace(0, len(files) - 1, int(len(files) * sample_factor), dtype=int)

    for i in indices:
        os.system('cp ' + img_path + files[i] + ' ' + linspace_path + files[i]) # save if selected

if __name__ == '__main__':
    # 1. video -> get all frames of imgs
    video_path = 'box.mp4'
    save_path = './box_frames/all_frames/'
    # video_to_frames(video_path, save_path)

    # 2. get all images that are above threshold
    frame_path = './box_frames/all_frames/'
    selected_path = './box_frames/threshold/'
    threshold = 572
    # blur_detect(frame_path, selected_path, threshold)

    # 3. evenly sample based on image filename
    linspace_path = './box_frames/images/'
    sample_factor = 0.5
    # img_linspace(selected_path, linspace_path, sample_factor)

    # ====================================
    # linspace first --> threshold
    selected_path = './box_frames/all_frames/'
    linspace_path = './box_frames/linspace/'
    sample_factor = 0.05
    # img_linspace(selected_path, linspace_path, sample_factor)

    frame_path = './box_frames/linspace/'
    selected_path = './box_frames/images/'
    threshold = 338
    blur_detect(frame_path, selected_path, threshold)