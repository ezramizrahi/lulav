import cv2
import time
import numpy as np

def initialize_camera():
    return cv2.VideoCapture(0)

def initialize_video_writer(video_path='./data/output.mp4', frame_size=(640, 480), fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    return cv2.VideoWriter(video_path, fourcc, fps, frame_size)

def detect_motion(frame, prev_frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    frame_delta = cv2.absdiff(prev_gray, gray)

    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame

def initialize_orb():
    return cv2.ORB.create()

def match_features(orb, frame1, frame2):
    keypoints1, descriptors1 = orb.detectAndCompute(frame1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(frame2, None)

    if descriptors1 is None or descriptors2 is None:
        return frame1

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    matches = sorted(matches, key=lambda x: x.distance)

    frame_with_matches = cv2.drawKeypoints(frame1, [keypoints2[m.trainIdx] for m in matches[:10]], None, color=(0,255,0), flags=0)

    return frame_with_matches

def record_video(cap, out, duration=10):
    orb = initialize_orb()

    start_time = time.time()
    ret, prev_frame = cap.read()
    while True:
        ret, frame = cap.read()
        if ret:
            frame_with_detection = detect_motion(frame.copy(), prev_frame)
            frame_with_tracking = match_features(orb, frame.copy(), prev_frame)
            combined_frame = cv2.addWeighted(frame_with_detection, 0.5, frame_with_tracking, 0.5, 0)
            prev_frame = frame
            out.write(combined_frame)
            # cv2.imshow('Recording', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # else:
            #     print("Failed to record")
            #     break
            if time.time() - start_time >= duration:
                print(f"Stopped recording after {duration} seconds.")
                break
        else:
            print("Failed to record")
            break

def release_resources(cap, out):
    out.release()
    cap.release()
    cv2.destroyAllWindows()