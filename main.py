import cv2
import datetime
from imutils.video import VideoStream
import numpy as np
from object_detection.utils import detector_utils as detector_utils
from datetime import date
import orien_lines
import argparse

lst1=[]
lst2=[]
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())

detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':
    # Detection confidence threshold to draw bounding box
    score_thresh = 0.75

    # Getting the Video
    vs = VideoStream(0).start()
    # vs = cv2.VideoCapture('mask sample.mp4')
    # Used to calculate fps
    start_time = datetime.datetime.now()
    num_frames = 0

    # Oriendtation of machine
    Orientation = 'bt'

    Line_Perc1 = float(15)

    Line_Perc2 = float(30)

    im_height,im_width = (None,None)
    cv2.namedWindow('Mask Detection',cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = vs.read()
            frame = np.array(frame)

            if im_height == None:
                im_height, im_width = frame.shape[:2]

                # Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease
                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except:
                    print("Error converting to RGB")

                # Run image through tensorflow graph
                boxes, scores, classes = detector_utils.detect_objects(
                    frame, detection_graph, sess)

                Line_Position2 = orien_lines.drawsafelines(frame, Orientation, Line_Perc1, Line_Perc2)
                # Draw bounding boxeses and text
                a, b = detector_utils.draw_box_on_image(3,score_thresh, scores, boxes, classes, im_width, im_height, frame)

                # Calculate Frames per second (FPS)
                num_frames += 1
                elapsed_time = (datetime.datetime.now() -
                                start_time).total_seconds()

                fps = num_frames / elapsed_time

                if args['display']:

                    # Display FPS on frame
                    detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)
                    cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        vs.stop()
                        break

    except KeyboardInterrupt:
        no_of_time_hand_detected=count_no_of_times(lst2)
        no_of_time_hand_crossed=count_no_of_times(lst1)
        today = date.today()
        # save_data(no_of_time_hand_detected, no_of_time_hand_crossed)
        print("Average FPS: ", str("{0:.2f}".format(fps)))
