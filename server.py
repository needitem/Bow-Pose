import cv2
import mediapipe as mp
import numpy as np
import pygame
from flask import Flask, Response, request
import io
import threading
import datetime
import time
import os
import logging

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)  # Only log errors and above


# --- Pygame Constants ---
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 980
GREEN = (100, 200, 100)
WHITE = (255, 255, 255)
BLUE = (20, 60, 120)
RED = (250, 0, 0)
FPS = 15

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Image Streaming Variables ---
frame_buffer = None
last_frame_time = datetime.datetime.now()
new_frame_available = threading.Condition()
lock = threading.Lock()

# --- Video Writer Variables ---
video_writer = None
is_recording = False
RECORDING_DURATION = 5  # 5 seconds recording duration
REPLAY_FOLDER = "./replay/"
start_time = None


def initialize_video_writer(frame_width, frame_height):
    global video_writer
    filename = os.path.join(
        REPLAY_FOLDER, f"replay_{time.strftime('%Y%m%d_%H%M%S')}.avi"
    )  # .avi 확장자 사용
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # XVID 코덱 사용 (더 안정적)
    video_writer = cv2.VideoWriter(filename, fourcc, FPS, (frame_width, frame_height))


# --- Webcam Variables ---
video_capture = cv2.VideoCapture(0)
webcam_frame = None
webcam_lock = threading.Lock()

# --- Video Replay Variables ---
is_replaying = False
replay_done = False  # Flag to indicate if replay has completed
replay_video_capture = None

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
TOGGLE_DRAWING = False  # Toggle to enable/disable drawing of landmarks


# --- Angle Calculation Function ---
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# --- Pygame Console Class ---
class Console:
    def __init__(self):
        self.hit_time = (
            datetime.datetime.now()
        )  # Initialize hit time (not used in this code)
        self.pose_result = None

        # Replay button at the bottom right
        self.replay_button_rect = pygame.Rect(
            SCREEN_WIDTH - 210, SCREEN_HEIGHT - 60, 200, 50
        )

        # Replay video area at the bottom left
        self.replay_video_rect = pygame.Rect(10, SCREEN_HEIGHT - 250, 640, 240)

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.replay_button_rect.collidepoint(event.pos):
                    self.trigger_replay()  # Call the replay function
        return False

    def run_logic(self):
        with webcam_lock:
            webcam_image = webcam_frame.copy()

        if webcam_image is not None:
            webcam_image.flags.writeable = False
            webcam_image = cv2.cvtColor(webcam_image, cv2.COLOR_BGR2RGB)
            self.pose_result = pose.process(webcam_image)

    def display_frame(self, screen):
        screen.fill(GREEN)

        global is_replaying, start_time, video_writer, is_recording

        # --- Display Uploaded Image (Left Side) ---
        with new_frame_available:
            new_frame_available.wait()
            with lock:
                image = frame_buffer
        if image is not None:
            image1 = pygame.image.frombuffer(image.tobytes(), image.shape[1::-1], "BGR")
            screen.blit(image1, [0, 0])

        # --- Display Webcam Feed with Pose Estimation (Right Side) ---
        with webcam_lock:
            webcam_image = webcam_frame.copy()

        if webcam_image is not None and self.pose_result.pose_landmarks:
            if TOGGLE_DRAWING:
                mp_drawing.draw_landmarks(
                    webcam_image,
                    self.pose_result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )
            left_shoulder = self.pose_result.pose_landmarks.landmark[
                mp_pose.PoseLandmark.LEFT_SHOULDER
            ]
            left_elbow = self.pose_result.pose_landmarks.landmark[
                mp_pose.PoseLandmark.LEFT_ELBOW
            ]
            left_wrist = self.pose_result.pose_landmarks.landmark[
                mp_pose.PoseLandmark.LEFT_WRIST
            ]
            right_shoulder = self.pose_result.pose_landmarks.landmark[
                mp_pose.PoseLandmark.RIGHT_SHOULDER
            ]
            right_elbow = self.pose_result.pose_landmarks.landmark[
                mp_pose.PoseLandmark.RIGHT_ELBOW
            ]
            right_wrist = self.pose_result.pose_landmarks.landmark[
                mp_pose.PoseLandmark.RIGHT_WRIST
            ]

            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            if (left_angle > 170 and right_angle < 30) or (
                right_angle > 170 and left_angle < 30
            ):
                print(
                    "Bow pose detected!, start recording for", RECORDING_DURATION, "s"
                )
                if not is_recording:
                    is_recording = True
                    start_time = datetime.datetime.now()
                    with webcam_lock:
                        initialize_video_writer(
                            webcam_image.shape[1], webcam_image.shape[0]
                        )

        if is_recording:
            if (
                datetime.datetime.now() - start_time
            ).total_seconds() > RECORDING_DURATION:
                is_recording = False
                print("Recording stopped!")
                video_writer.release()
                video_writer = None
            else:
                with webcam_lock:
                    try:
                        video_writer.write(webcam_image)
                        print("Frame written")
                    except Exception as e:
                        print(f"Error writing frame: {e}")

        # Display the webcam image on the right side of the screen
        if webcam_image is not None:
            webcam_image = cv2.rotate(webcam_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            webcam_image = cv2.cvtColor(webcam_image, cv2.COLOR_BGR2RGB)
            webcam_surface = pygame.surfarray.make_surface(webcam_image)
            screen.blit(webcam_surface, [SCREEN_WIDTH // 2, 0])

        # --- Replay Video ---
        if is_replaying and replay_video_capture.isOpened():
            ret, replay_frame = replay_video_capture.read()
            if ret:
                replay_frame = cv2.cvtColor(replay_frame, cv2.COLOR_BGR2RGB)

                # Rotate the replay frame 90 degrees counter-clockwise
                replay_frame = cv2.rotate(replay_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Calculate dimensions for right-bottom half
                target_width = SCREEN_WIDTH // 2
                target_height = SCREEN_HEIGHT // 2

                # Resize the video frame while maintaining aspect ratio
                aspect_ratio = replay_frame.shape[1] / replay_frame.shape[0]
                if aspect_ratio > target_width / target_height:
                    new_width = target_width
                    new_height = int(target_width / aspect_ratio)
                else:
                    new_height = target_height
                    new_width = int(target_height * aspect_ratio)

                replay_frame = cv2.resize(replay_frame, (new_width, new_height))

                # Center the video in the target area
                x_offset = SCREEN_WIDTH // 2 + (target_width - new_width) // 2
                y_offset = SCREEN_HEIGHT // 2 + (target_height - new_height) // 2
                replay_video_rect = pygame.Rect(
                    x_offset, y_offset, new_width, new_height
                )

                replay_surface = pygame.surfarray.make_surface(replay_frame)
                screen.blit(replay_surface, replay_video_rect)
            else:
                is_replaying = False
                replay_video_capture.release()
                print("Replay finished!")

        # --- Recording / Replaying Indicator ---
        if is_recording or is_replaying:
            status_text = "Recording..." if is_recording else "Replaying..."
            color = RED if is_recording else BLUE
            font = pygame.font.Font(None, 36)
            text = font.render(status_text, True, color)
            screen.blit(text, (10, 10))
        else:
            font = pygame.font.Font(None, 36)
            text = font.render("Not Recording", True, WHITE)
            screen.blit(text, (10, 10))

        # Display the replay button
        pygame.draw.rect(screen, BLUE, self.replay_button_rect)
        font = pygame.font.Font(None, 36)
        text = font.render("Replay Last Video", True, WHITE)
        screen.blit(text, (SCREEN_WIDTH - 200, SCREEN_HEIGHT - 50))

    def trigger_replay(self):
        global is_replaying, replay_video_capture
        if not is_replaying and not is_recording:
            video_files = [f for f in os.listdir(REPLAY_FOLDER) if f.endswith(".avi")]
            if video_files:
                latest_video = max(
                    video_files,
                    key=lambda f: os.path.getmtime(os.path.join(REPLAY_FOLDER, f)),
                )
                filepath = os.path.join(REPLAY_FOLDER, latest_video)
                replay_video_capture = cv2.VideoCapture(filepath)
                replay_video_capture.set(cv2.CAP_PROP_FPS, FPS * 0.3)
                # Set replay speed to 0.3x

                is_replaying = True
                print(f"Replaying {latest_video}")
            else:
                print("No replay videos found.")


# --- Video Feed Endpoint ---
@app.route("/video_feed")
def video_feed():
    def generate():
        global last_frame_time
        while True:
            time_since_last_frame = (
                datetime.datetime.now() - last_frame_time
            ).total_seconds()
            if time_since_last_frame < 1 / FPS:
                time.sleep(1 / FPS - time_since_last_frame)

            with lock:
                frame = frame_buffer
            if frame is not None:
                _, buffer = cv2.imencode(".jpg", frame)
                frame_bytes = io.BytesIO(buffer)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes.getvalue()
                    + b"\r\n"
                )

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# --- Upload Endpoint ---
@app.route("/upload", methods=["POST"])
def upload_files():
    global frame_buffer
    if request.method == "POST":
        try:
            frame_bytes = request.data
            nparr = np.frombuffer(frame_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            with lock:
                frame_buffer = image
            with new_frame_available:
                new_frame_available.notify()
            return "Frame updated"
        except Exception as e:
            print(f"Error processing uploaded image: {e}")
            return "Error processing image"


# New Endpoint to Trigger Replay
@app.route("/replay/<filename>")
def trigger_replay(filename):
    global is_replaying, replay_video_capture
    if not is_replaying:
        filepath = os.path.join(REPLAY_FOLDER, filename)
        if os.path.exists(filepath):
            replay_video_capture = cv2.VideoCapture(filepath)
            # Set replay speed to 0.3x
            replay_video_capture.set(cv2.CAP_PROP_FPS, FPS * 0.3)
            is_replaying = True
            return f"Replaying {filename}"
        else:
            return f"File {filename} not found"
    else:
        return "Already replaying a video"


# Webcam Capture Thread
def webcam_capture_thread():
    global webcam_frame
    while True:
        ret, frame = video_capture.read()
        if ret:
            with webcam_lock:
                webcam_frame = frame


# Pygame Thread Function
def thread_pygame():
    pygame.init()
    pygame.display.set_caption("Bullseye")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    console = Console()
    done = False
    print("pygame start!\n")

    while not done:
        done = console.process_events()
        console.run_logic()
        console.display_frame(screen)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


# --- Start the Threads ---
thread_1 = threading.Thread(target=thread_pygame)
thread_1.start()
webcam_thread = threading.Thread(target=webcam_capture_thread)
webcam_thread.start()

# --- Run the Flask App ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
