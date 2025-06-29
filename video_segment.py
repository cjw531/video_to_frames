import os
import cv2
import numpy as np
import imageio

def video_to_frames(video_path, save_path):
    os.system('rm -rf ' + save_path)
    os.system('mkdir -p ' + save_path)
    
    # reference: https://stackoverflow.com/a/47632941
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        # vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*200)) # added this line every 0.5 sec
        cv2.imwrite(save_path + "/frame_%d.jpg" % count, image) # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1

def frame_to_segment(frame_path, save_path):
    # segmentation reference: https://machinelearningknowledge.ai/image-segmentation-in-python-opencv/

    os.system('rm -rf ' + save_path)
    os.system('mkdir -p ' + save_path)

    for img_file in os.listdir(frame_path):
        image = cv2.imread(frame_path + img_file)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # reshape img into a 2d vector
        twoDimage = img.reshape((-1,3))
        twoDimage = np.float32(twoDimage)

        # define segmentation parameters
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2 # object and the background, 2 segments
        attempts = 10
        
        # apply k-means
        ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((img.shape))

        # mask the background
        mask_img = segment_mask_background(result_image, frame_path, img_file)

        # save img
        cv2.imwrite(save_path + img_file, mask_img)

def segment_mask_background(segment_img, frame_path, img_filename):
    # TODO: make user to select which rgb value is the background in the future

    # segment image determine which r,g,b value belongs to background
    back_rgb = np.unique(segment_img.reshape(-1, segment_img.shape[2]), axis=0)[1]

    frame_img = cv2.imread(frame_path + img_filename)
    height, width, depth = frame_img.shape
    for i in range(height):
        for j in range(width):
            if (back_rgb == segment_img[i, j]).all(): # if determined to be background in segmented img
                frame_img[i, j] = [0, 0, 0] # mask into black

    return frame_img

def segment_to_video(segment_path, save_path):
    # gif creation reference: https://theailearner.com/2021/05/29/creating-gif-from-video-using-opencv-and-imageio/

    gif_imlist = []
    for i in range(len(os.listdir(segment_path))):
        image = cv2.imread(segment_path + 'frame_' + str(i) + '.jpg')
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image resize
        width = int(img.shape[1] / 10)
        height = int(img.shape[0] / 10)
        dim = (width, height)
        
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        gif_imlist.append(resized)

    imageio.mimsave(save_path + 'segment.gif', gif_imlist, fps=60)

def laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F)

def log(image):
    gaussian_blur = cv2.GaussianBlur(image,(5,5),0)
    return laplacian(gaussian_blur)

def filter_selected(img_path, save_path):
    os.system('mkdir -p ' + save_path)

    image = cv2.imread(img_path)
    cv2.imwrite(save_path + 'laplacian_' + img_path.split('_')[1], laplacian(image))

def img_linspace(img_path, linspace_path, sample_factor):
    # use top 20% and apply the linspace algorithm

    os.system('rm -rf ' + linspace_path)
    os.system('mkdir ' + linspace_path)

    files = sorted(os.listdir(img_path))
    indices = np.linspace(0, len(files) - 1, int(len(files) * sample_factor), dtype=int)

    for i in indices:
        os.system('cp ' + img_path + files[i] + ' ' + linspace_path + files[i]) # save if selected

def laplacian_edge(img_path, save_path):
    os.system('rm -rf ' + save_path)
    os.system('mkdir ' + save_path)

    for img in os.listdir(img_path):
        image = cv2.imread(img_path + img)
        cv2.imwrite(save_path + img, laplacian(image))

def edge_variance_threshold(img_path, sample_factor):
    edge_vars = []
    for img in os.listdir(img_path):
        image = cv2.imread(img_path + img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge_vars.append(np.var(gray.flatten()))

    edge_vars.sort(reverse=True) # sort in descending order
    index_top_40_per = int(len(edge_vars) * sample_factor)
    top_40_var = edge_vars[:index_top_40_per]
    
    return min(top_40_var) # return threshold

def select_sharp(frame_path, edge_path, save_path, thresh):
    os.system('rm -rf ' + save_path)
    os.system('mkdir ' + save_path)

    for img in os.listdir(edge_path):
        image = cv2.imread(edge_path + img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if thresh <= np.var(gray.flatten()):
            os.system('cp ' + frame_path + img + ' ' + save_path + img) # save if selected

if __name__ == '__main__':
    # 1. video -> get all frames of imgs
    video_path = 'hand.mp4'
    save_path = './hand/all_frames/'
    # video_to_frames(video_path, save_path)

    # 2. frames -> segment
    frame_path = './hand/all_frames/'
    save_path = './hand/segment/'
    # frame_to_segment(frame_path, save_path)

    # 3. segment to gif
    segment_path = './hand/segment/'
    save_path = './hand/'
    # segment_to_video(segment_path, save_path)

    # ========================================
    # apply filters to selected images
    img_path = './hand/segment/frame_1.jpg'
    save_path = './hand/filter/'
    # filter_selected(img_path, save_path)
    # ========================================

    # 4. linspace first --> threshold variance
    selected_path = './hand/segment/'
    linspace_path = './hand/linspace/'
    sample_factor = 0.2
    img_linspace(selected_path, linspace_path, sample_factor)

    # 5. apply filters to all images
    # the more the edge, the more the count
    # the more the strong edge, the more the variance
    linspace_path = './hand/linspace/'
    save_path = './hand/filter/'
    laplacian_edge(linspace_path, save_path)

    # 6. get threshold based on edge img variance
    edge_path = './hand/filter/'
    thresh = edge_variance_threshold(edge_path, 0.1)

    # 7. based on edge threshold, get sharp original images
    frame_path = './hand/all_frames/'
    edge_path = './hand/filter/'
    save_path = './hand/images/'
    select_sharp(frame_path, edge_path, save_path, thresh)