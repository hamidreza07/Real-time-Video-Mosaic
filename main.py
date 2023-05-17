import cv2
from pathlib import Path
import numpy as np


class VideMosaic:
    def __init__(self, first_image, output_height_times=10, output_width_times=5, detector_type="sift"):
        """This class processes every frame and generates the panorama
        Args:
            first_image (image for the first frame): first image to initialize the output size
            output_height_times (int, optional): determines the output height based on input image height. Defaults to 2.
            output_width_times (int, optional): determines the output width based on input image widqth. Defaults to 4.
            detector_type (str, optional): the detector for feature detection. It can be "sift" or "orb". Defaults to "sift".
        """
        self.detector_type = detector_type
        if detector_type == "sift":
            self.detector = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.001, edgeThreshold=1000, sigma=2.6,nOctaveLayers=5)
        if detector_type == "akaze":
            self.detector =  cv2.AKAZE_create( threshold=0.001, nOctaves=3,  nOctaveLayers=3,  descriptor_channels=3,  descriptor_size=0)

        if detector_type == "BRISK":
            self.detector =  cv2.BRISK_create(octaves=8, patternScale=1.2)

        elif detector_type == "orb":
            self.detector = cv2.ORB_create(100)
        self.bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)

        self.visualize = True

        self.process_first_frame(first_image)

        self.output_img = np.zeros(shape=(int(output_height_times * first_image.shape[0]), int(
            output_width_times*first_image.shape[1]), first_image.shape[2]))
        self.prev_pts = None
        # offset
        # self.w_offset = int(self.output_img.shape[0]/2 - first_image.shape[0]/2)
        # self.h_offset = int(self.output_img.shape[1]/2 - first_image.shape[1]/2)
        self.w_offset = int(1200)
        self.h_offset = int(100)
        # FLANN matcher parameters
        index_params = dict(algorithm=2, trees=50)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # self.output_img[self.w_offset:self.w_offset+first_image.shape[0],
        #                 self.h_offset:self.h_offset+first_image.shape[1], :] = first_image
        self.prev_gray = None
        # self.H_old = np.eye(3)
        # self.H_old[0, 2] = self.h_offset
        # self.H_old[1, 2] = self.w_offset
        self.prev_frame = None
        self.drone_pos = [0, 0]  # [x, y] position
        self.x = 0
        self.y = 0
        self.H_old = np.eye(3)
        self.H_old[0, 2] = self.h_offset + self.x
        self.H_old[1, 2] = self.w_offset + self.y
    def process_first_frame(self, first_image):
        """processes the first frame for feature detection and description
        Args:
            first_image (cv2 image/np array): first image for feature detection
        """
        self.frame_prev = first_image
        frame_gray_prev = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
        self.kp_prev, self.des_prev = self.detector.detectAndCompute(frame_gray_prev, None)

    def match(self, des_cur, des_prev):
        """matches the descriptors
        Args:
            des_cur (np array): current frame descriptor
            des_prev (np arrau): previous frame descriptor
        Returns:
            array: and array of matches between descriptors
        """
        # matching
        if self.detector_type == "sift":
            matches = self.flann.knnMatch(des_cur, des_prev, k=10)
            good_matches = []
            for match_list in matches:
                m = match_list[0]
                n = match_list[1]
                if m.distance < 0.9 * n.distance:
                    good_matches.append(m)
        if self.detector_type == "akaze" or self.detector_type=='BRISK':
            pair_matches = self.bf.knnMatch(des_cur, des_prev, k=2)
            good_matches = []
            for m, n in pair_matches:
                if m.distance < 0.7*n.distance:
                    good_matches.append(m)

        elif self.detector_type == "orb":
            good_matches = self.bf.match(des_cur, des_prev)
        elif self.detector_type == "akaze":
            matches = self.bf.match(des_cur, des_prev)

        # Sort them in the order of their distance.
        good_matches = sorted(good_matches, key=lambda x: x.distance)

        # get the maximum of 20  best matches
        good_matches = good_matches[:min(len(good_matches), 100)]
        # Draw first 10 matches.
        if self.visualize:
            match_img = cv2.drawMatches(self.frame_cur, self.kp_cur, self.frame_prev, self.kp_prev, good_matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('matches', match_img)
        return good_matches

    def process_frame(self, frame_cur):
        """gets an image and processes that image for mosaicing
        Args:
            frame_cur (np array): input of current frame for the mosaicing
        """
        self.frame_cur = frame_cur
        frame_gray_cur = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2GRAY)
        if self.prev_pts is not None:

                # Convert the previous frame to grayscale


            lk_params = dict(winSize=(100, 100), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


            
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray_cur,self.prev_pts, None, **lk_params)

            # Get the good points and draw them
            good_new = p1[st == 1]
            good_old = self.prev_pts[st == 1]
            mean_flow = np.mean(good_new - good_old, axis=0)
            self.x -= mean_flow[0]
            self.y += mean_flow[1]
       
        self.kp_cur, self.des_cur = self.detector.detectAndCompute(frame_gray_cur, None)

        self.matches = self.match(self.des_cur, self.des_prev)
        # Detect good features to track in the current frame
        pts = cv2.goodFeaturesToTrack(frame_gray_cur, 200, 0.05, 10)
                # Update the previous frame and points
        self.prev_gray = frame_gray_cur.copy()
        self.prev_pts = pts

        if len(self.matches) < 4:
            return
    
        self.H = self.findHomography(self.kp_cur, self.kp_prev, self.matches)
        self.H = np.matmul(self.H_old, self.H)
        # TODO: check for bad Homography

        self.warp(self.frame_cur, self.H)

        # loop preparation
        self.H_old = self.H
        self.kp_prev = self.kp_cur
        self.des_prev = self.des_cur
        self.frame_prev = self.frame_cur

    @ staticmethod
    def findHomography(image_1_kp, image_2_kp, matches):
        """gets two matches and calculate the homography between two images
        Args:
            image_1_kp (np array): keypoints of image 1
            image_2_kp (np_array): keypoints of image 2
            matches (np array): matches between keypoints in image 1 and image 2
        Returns:
            np arrat of shape [3,3]: Homography matrix
        """
        # taken from https://github.com/cmcguinness/focusstack/blob/master/FocusStack.py

        image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
        image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
        for i in range(0, len(matches)):
            image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
            image_2_points[i] = image_2_kp[matches[i].trainIdx].pt

        homography, mask = cv2.findHomography(image_1_points, image_2_points, cv2.LMEDS, ransacReprojThreshold=2.0,maxIters=100)
        
        
        return homography

    def warp(self, frame_cur, H):
        """ warps the current frame based of calculated homography H
        Args:
            frame_cur (np array): current frame
            H (np array of shape [3,3]): homography matrix
        Returns:
            np array: image output of mosaicing
        """
        warped_img = cv2.warpPerspective(frame_cur, H, 
                                         (self.output_img.shape[1], self.output_img.shape[0]), 
                                         flags=cv2.INTER_NEAREST
                                         )


        transformed_corners = self.get_transformed_corners(frame_cur, H)
        warped_img = self.draw_border(warped_img, transformed_corners)

        self.output_img[warped_img > 0] = warped_img[warped_img > 0]
        output_temp = np.copy(self.output_img)
        output_temp = self.draw_border(output_temp, transformed_corners, color=(0, 0, 255))
        output_temp = cv2.resize(output_temp, (0, 0), fx=0.5, fy=.5)
        output_temp = cv2.putText(output_temp, f'X: {self.x:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        output_temp = cv2.putText(output_temp, f'Y: {self.y:.2f}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('output',  output_temp/255.)

        return self.output_img

    @ staticmethod
    def get_transformed_corners(frame_cur, H):
        """finds the corner of the current frame after warp
        Args:
            frame_cur (np array): current frame
            H (np array of shape [3,3]): Homography matrix 
        Returns:
            [np array]: a list of 4 corner points after warping
        """
        corner_0 = np.array([0, 0])
        corner_1 = np.array([frame_cur.shape[1], 0])
        corner_2 = np.array([frame_cur.shape[1], frame_cur.shape[0]])
        corner_3 = np.array([0, frame_cur.shape[0]])

        corners = np.array([[corner_0, corner_1, corner_2, corner_3]], dtype=np.float32)
        transformed_corners = cv2.perspectiveTransform(corners, H)

        transformed_corners = np.array(transformed_corners, dtype=np.int32)
        # mask = np.zeros(shape=(output.shape[0], output.shape[1], 1))
        # cv2.fillPoly(mask, transformed_corners, color=(1, 0, 0))
        # cv2.imshow('mask', mask)

        return transformed_corners

    def draw_border(self, image, corners, color=(0, 0, 0)):
        """This functions draw rectancle border
        Args:
            image ([type]): current mosaiced output
            corners (np array): list of corner points
            color (tuple, optional): color of the border lines. Defaults to (0, 0, 0).
        Returns:
            np array: the output image with border
        """
        for i in range(corners.shape[1]-1, -1, -1):
            cv2.line(image, tuple(corners[0, i, :]), tuple(
                corners[0, i-1, :]), thickness=5, color=color)
        return image


def main():

    video_path = 'IMG_6655.MP4'
    cap = cv2.VideoCapture(video_path)
    is_first_frame = True
    
    # for i in range(25):
    #     ret, frame_cur = cap.read()
    #     cv2.imshow('a',frame_cur)
    #     cv2.waitKey(0)

    i = 0
    while True:
        ret, frame_cur = cap.read()
        if not ret:
            if is_first_frame:
                continue
            break
        if i == 3500:
            break   
        frame_cur = cv2.resize(frame_cur, (0, 0), fx=0.3, fy=0.3)

        if is_first_frame:
            video_mosaic = VideMosaic(frame_cur, detector_type="sift")
            is_first_frame = False
            continue
 
        # process each frame
        video_mosaic.process_frame(frame_cur)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i+= 1
        # print(i)

    cv2.imwrite('mosaic.jpg', video_mosaic.output_img)
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()