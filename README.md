# Hand Finger Counter

A real-time finger counting app using your webcam, built with MediaPipe and OpenCV. Hold your hand up to the camera and it detects how many fingers you're holding up.

This was a learning project built as an introduction to Python, computer vision, and ML APIs for someone coming from a game development background.

---

## Demo

- Opens your webcam
- Detects your hand in real time
- Counts how many fingers are raised
- Displays the count overlaid on the video feed
- Press **Q** to quit

---

## Requirements

- Python 3.13+
- A webcam
- The MediaPipe hand landmark model file: [`hand_landmarker.task`](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models)

Install dependencies:

```bash
pip install opencv-python mediapipe
```

---

## Usage

1. Download `hand_landmarker.task` and place it in the same directory as the script.
2. Run the script:

```bash
python HandDetection.py
```

3. Hold your hand up in front of your webcam.
4. Press **Q** to exit.

---

## How It Works

**Hand detection** is handled by MediaPipe's Hand Landmarker, which identifies 21 specific points (landmarks) on your hand, one for each knuckle joint, fingertip, and the wrist.

**Finger counting** works by comparing landmark positions:
- For the **thumb**, it checks if the tip (`landmark[4]`) is to the left of the knuckle below it (`landmark[3]`) on the X axis — since the hand is mirrored, this catches it pointing outward.
- For the **other four fingers**, it checks if each fingertip is *above* (lower Y value) its second-to-last joint, meaning the finger is extended rather than curled.

**Drawing** is done with OpenCV, each landmark is drawn as a small green dot, and the finger count is rendered as text in the top left corner.

---

## Stuff I Learned

This project was a first dive into Python and ML tooling. Some of the things I learned:

### Python Syntax & Concepts
- Basic script structure
- `import` statements and how Python modules work
- Defining functions with `def`
- List operations: `.append()`, `.count()`, list comprehensions
- `f-strings` for embedding variables in strings (`f'Fingers: {finger_count}'`)
- Tuple unpacking (`h, w, c = frame.shape`)
- The `&` bitwise operator (used for reading keypress input with OpenCV)

### MediaPipe
- What MediaPipe is: a Google ML framework for real time perception tasks (hands, face, pose, etc.)
- The difference between the older `mediapipe.solutions` API and the newer **Tasks API** (`mediapipe.tasks`)
- How to load a `.task` model file using `BaseOptions`
- Configuring a `HandLandmarkerOptions` object with confidence thresholds
- How landmark data is structured: a list of 21 points, each with normalised `x`, `y`, `z` coordinates (0.0–1.0 relative to the image size)
- What the 21 hand landmarks represent and their index numbers

### OpenCV (cv2)
- Capturing webcam input with `VideoCapture`
- Reading frames in a loop with `cap.read()`
- Flipping a frame horizontally to create a mirror view (`cv2.flip`)
- Converting colour spaces — OpenCV uses BGR by default, MediaPipe needs RGB (`cv2.cvtColor`)
- Drawing on frames: `cv2.circle` and `cv2.putText`
- Displaying a live window with `cv2.imshow`
- Properly releasing resources with `cap.release()` and `cv2.destroyAllWindows()`

### Machine Learning Concepts
- What a **model file** is and why you need to download it separately
- The concept of **confidence thresholds** tuning how certain the model needs to be before it acts on a detection
- How ML models work with **normalised coordinates** instead of raw pixel values
- The difference between **detection** (is there a hand?), **presence** (is it still there?), and **tracking** (follow it frame to frame)

### TensorFlow / ML Ecosystem
- MediaPipe is built on top of TensorFlow under the hood, even if you never call TensorFlow directly
- What a `.task` file is: a packaged TensorFlow Lite model bundled with metadata
- The general idea of **TFLite**, a lightweight version of TensorFlow optimised for running inference on-device in real time

### Git & Project Workflow
- Initialising a repo with `git init`
- Tracking files with `git add` and committing with `git commit`
- The importance of a `.gitignore` — e.g. keeping the large `.task` model file out of version control

---

## File Structure

```
hand-detector/
├── hand_detector.py        # Main script
├── hand_landmarker.task    # MediaPipe model (download separately, don't commit)
├── .gitignore
└── README.md
```

---
