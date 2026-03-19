import cv2
import mediapipe as mp
import pyautogui
import time
import threading
import speech_recognition as sr

# Shared state
last_trigger_time = 0.0
cooldown = 2.0
lock = threading.Lock()
running = True

def trigger_page(direction, source="Gesture"):
    global last_trigger_time
    with lock:
        current_time = time.time()
        if current_time - last_trigger_time > cooldown:
            action = "Next" if direction == 'right' else "Previous"
            print(f"[{source}] Detected! Triggering {action} Page ({direction.capitalize()} Arrow)")
            pyautogui.press(direction)
            last_trigger_time = current_time
            return True
    return False

def voice_listener():
    recognizer = sr.Recognizer()
    try:
        microphone = sr.Microphone()
    except Exception as e:
        print(f"Error initializing microphone: {e}")
        return

    # Optimize recognizer for faster response times
    recognizer.pause_threshold = 0.4        # Wait only 0.4 seconds of silence before processing
    recognizer.non_speaking_duration = 0.3  # Shorter non-speaking duration
    recognizer.dynamic_energy_threshold = False # Disable dynamic threshold to save computation
    recognizer.energy_threshold = 300       # Fixed threshold

    # Adjust for ambient noise
    with microphone as source:
        print("Adjusting for ambient noise... Please wait.")
        try:
            recognizer.adjust_for_ambient_noise(source, duration=2)
        except Exception as e:
            print(f"Could not adjust for ambient noise: {e}")
        print("Voice listener ready. Say 'next'/'forward' or 'back'/'previous'.")
        
    while running:
        with microphone as source:
            try:
                # Reduce phrase time limit so it processes short commands instantly
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=2)
                try:
                    text = recognizer.recognize_google(audio).lower()
                    print(f"Heard: '{text}'")
                    if "next" in text or "forward" in text:
                        trigger_page("right", "Voice")
                    elif "back" in text or "previous" in text:
                        trigger_page("left", "Voice")
                except sr.UnknownValueError:
                    pass # Couldn't understand audio
                except sr.RequestError as e:
                    print(f"Could not request results from Google Speech Recognition service; {e}")
            except sr.WaitTimeoutError:
                # Re-check the running flag and loop
                continue
            except Exception as e:
                if running:
                    print(f"Voice listener encountered an error: {e}")
                    time.sleep(1)

def main():
    global running, last_trigger_time

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a hand landmarker instance with the video mode:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1)

    print("Loading MediaPipe model...")
    with HandLandmarker.create_from_options(options) as landmarker:
        # Use DirectShow backend on Windows for lower latency
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # Lower resolution to speed up frame processing
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Minimize buffer size to always get the most recent frame
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Agent running silently. Press Ctrl+C to stop.")

        # Start Voice thread
        voice_thread = threading.Thread(target=voice_listener, daemon=True)
        voice_thread.start()

        try:
            while cap.isOpened() and running:
                success, img = cap.read()
                if not success:
                    print("Failed to grab frame.")
                    break
                    
                img = cv2.flip(img, 1)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                frame_timestamp_ms = int(time.time() * 1000)
                
                # Detect hand landmarks in the frame
                hand_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                
                if hand_landmarker_result.hand_landmarks:
                    for hand_landmarks in hand_landmarker_result.hand_landmarks:
                        
                        # Thumb parameters
                        thumb_is_up = hand_landmarks[4].y < hand_landmarks[3].y
                        
                        # Finger tips vs MCP joints (Is it folded?)
                        index_is_folded = hand_landmarks[8].y > hand_landmarks[5].y
                        middle_is_folded = hand_landmarks[12].y > hand_landmarks[9].y
                        ring_is_folded = hand_landmarks[16].y > hand_landmarks[13].y
                        pinky_is_folded = hand_landmarks[20].y > hand_landmarks[17].y

                        # Finger tips vs PIP joints (Is it strictly raised?)
                        index_is_up = hand_landmarks[8].y < hand_landmarks[6].y
                        middle_is_up = hand_landmarks[12].y < hand_landmarks[10].y
                        
                        # 1. Thumbs Up -> Right Arrow
                        if thumb_is_up and index_is_folded and middle_is_folded and ring_is_folded and pinky_is_folded:
                            trigger_page("right", "Gesture (Thumbs Up)")
                                
                        # 2. Peace Sign (Index and Middle up, others folded) -> Left Arrow
                        elif index_is_up and middle_is_up and ring_is_folded and pinky_is_folded:
                            trigger_page("left", "Gesture (Peace Sign)")

                # Sleep slightly to prevent high CPU utilization in headless loop
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nStopping agent...")
            running = False

    cap.release()
    # Wait briefly for voice thread to exit smoothly
    voice_thread.join(timeout=1.0)
    print("Agent stopped gracefully.")

if __name__ == "__main__":
    main()