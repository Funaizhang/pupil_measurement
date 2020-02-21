import numpy as np
import cv2
import time
import os
import math
import argparse
import itertools

class measurePupil(object):

    def __init__(self, extend, circularity, is_full_face, wait_key, show_boxes, save_frames):
        self.extend             = extend
        self.circularity        = circularity
        self.is_full_face       = is_full_face
        self.wait_key           = wait_key
        self.show_boxes         = show_boxes
        self.save_frames        = save_frames
        self.vid_dim            = ()
        self.font               = cv2.FONT_HERSHEY_SIMPLEX
        self.text_position      = ()
        self.font_scale         = 1
        self.font_color         = (255,255,255)
        self.line_type          = 2
        self.count              = 0

    
    def doVideo(self, input_file):

        # load haar cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        # init kernels for filtering image later
        self.kernel_close = np.ones((2,2), np.uint8)
        self.kernel_open = np.ones((2,2), np.uint8)

        cap = cv2.VideoCapture(input_file)

        while(cap.isOpened()):
            # capture frame-by-frame
            ret, frame = cap.read()

            # get video dims
            self.vid_dim = (int(cap.get(3)), int(cap.get(4)))
            self.text_position = (int(cap.get(3)/3), int(cap.get(4) - 50))

            if ret == True:

                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

                # if the frame contains a full face, detect face first, then detect eyes on face
                if self.is_full_face:
                    faces = self.face_cascade.detectMultiScale(frame, 1.3, 5)
                    pupil_ave = self.doFace(frame, faces)
                        
                # if the frame does not contain a full face, detect eyes to start with
                else:
                    eyes = self.eye_cascade.detectMultiScale(frame, 1.3, 5)
                    pupil_ave = self.doEye(frame, eyes)

                # display pupil measurement
                cv2.putText(
                    frame,
                    'Pupil size: {}.'.format(pupil_ave),
                    self.text_position, 
                    self.font, 
                    self.font_scale,
                    self.font_color,
                    self.line_type)
                
                # display the resulting frame
                cv2.imshow('frame', frame)
                if cv2.waitKey(self.wait_key) & 0xFF == ord('q'):
                    break

            else:
                break

        # release everything once job is finished
        cap.release()
        cv2.destroyAllWindows()


    # doFace measures the pupil given the face ROI
    def doFace(self, frame, faces):
        for (x,y,w,h) in faces:
            # outline detected face
            if self.show_boxes:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 1)
            face_frame = frame[y:(y+h), x:(x+w)]
            
            eyes = self.eye_cascade.detectMultiScale(face_frame, 1.3, 5)
            pupil_ave = self.doEye(face_frame, eyes)

        return pupil_ave
            
    
    # doEye measures the pupil given the eye ROI
    def doEye(self, frame, eyes):
        pupil_widths = []

        # clean up the eye frames by removing the larger frames
        eyes = self.removeBiggerFrames(eyes)

        for (ex,ey,ew,eh) in eyes:
            # outline detected eyes
            if self.show_boxes:
                cv2.rectangle(frame, (ex,ey), ((ex+ew),(ey+eh)), (0,0,255), 1)

            # filter the eye ROI for dark contours
            pupil_frame = frame[ey+1:(ey+eh), ex+1:(ex+ew)]
            threshold_value = np.amin(pupil_frame) + 15
            ret, pupil_frame = cv2.threshold(pupil_frame, threshold_value, 255, cv2.THRESH_BINARY)
            # pupil_frame = cv2.adaptiveThreshold(pupil_frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,51,0)
            pupil_frame = cv2.morphologyEx(pupil_frame, cv2.MORPH_CLOSE, self.kernel_close)
            pupil_frame = cv2.morphologyEx(pupil_frame, cv2.MORPH_OPEN, self.kernel_open)
            contours, _ = cv2.findContours(pupil_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(pupil_frame, contours, -1, (255, 0, 0), 1)
            
            ellipses = []
            ellipse_widths = []

            for contour in contours:

                # make contour more circular
                contour = cv2.convexHull(contour)
                area = cv2.contourArea(contour)

                # reject the contours with big extend
                bounding_box = cv2.boundingRect(contour)
                extend = area / (bounding_box[2] * bounding_box[3])
                if extend > self.extend:
                    continue

                # reject if the ellipse is not very circular
                circumference = cv2.arcLength(contour, True)
                circularity = circumference ** 2 / (4 * math.pi * area)
                if circularity > self.circularity:
                    continue

                # fit an ellipse around the contour
                try:
                    ellipse = cv2.fitEllipse(contour)
                    ellipses.append(ellipse)
                    width = max(ellipse[1])
                    ellipse_widths.append(width)
                except:
                    pass

            # find the widest ellipse, which usually corresponds to the pupil
            try:
                ellipse_idx = np.argmax(ellipse_widths)
                pupil = ellipses[ellipse_idx]
                cv2.ellipse(pupil_frame, box=pupil, color=(0, 255, 0))
                # pupil_widths keeps the pupil measurements for all the detected eyes 
                pupil_widths.append(max(ellipse_widths))
            except:
                pass

            # save the relevant gray-scale pupil_frame
            if self.save_frames:
                img_path = os.path.join(save_path, 'frame_{}.jpg'.format(self.count))
                if not cv2.imwrite(img_path, pupil_frame):
                    raise Exception("Could not save frame")
                self.count += 1

        pupil_widths.sort()
        pupil_ave = None
        # if one eye is detected, use its pupil measurement
        if len(pupil_widths) == 1:
            pupil_ave = int(pupil_widths[0])
        # if more than one eye is detected, average the 2 pupil measurements
        elif len(pupil_widths) > 1 and pupil_widths[1] > pupil_widths[0] * 0.75:
            pupil_ave = int(np.average([pupil_widths[0], pupil_widths[1]]))
                
        return pupil_ave


    def removeBiggerFrames(self, frames):

        invalids = []

        # if a frame contains a smaller frame, it is invalid
        for i, j in itertools.permutations(frames, 2):
            (ix,iy,iw,ih) = i
            (jx,jy,jw,jh) = j
            if ix<jx and iy<jy and ix+iw>jx+jw and iy+ih>jy+jh:
                invalids.append(i)
        
        # get rid off the larger frame
        frames = list(set(map(tuple, frames)) - set(map(tuple, invalids)))
        return frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input params
    parser.add_argument('-extend', default=0.8, type=float)
    parser.add_argument('-circularity', default=1.2, type=float)
    parser.add_argument('-is_full_face', default=True, type=bool)
    parser.add_argument('-wait_key', default=100, type=int)
    parser.add_argument('-show_boxes', default=False, action='store_true')
    parser.add_argument('-save_frames', default=False, action='store_true')
    args = parser.parse_args()

    cur_dir = os.path.dirname(__file__)
    save_path = os.path.join(cur_dir, 'saved_frames')
    vid_path = os.path.join(cur_dir, 'test_vid.mov')

    # check save dir exists
    if args.save_frames:
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    # check video file exists
    if not os.path.isfile(vid_path):
        raise Exception("Could not find video file")

    # main function
    my_test = measurePupil(
        args.extend, 
        args.circularity, 
        args.is_full_face, 
        args.wait_key, 
        args.show_boxes, 
        args.save_frames)
    my_test.doVideo(vid_path)