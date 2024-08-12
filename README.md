<h1 align="center">
<br>
  Face Mouse
  <br>
</h1>

<h4 align="center">A virtual mouse pointer capable of moving cursor and performing clicks using facial landmarks detected in video stream</h4>


## Key Features

- **Control your cursor by moving your face**-
the predictor tracks the tip of your nose and moves the cursor in the direction the nose moves.
- **Perform click functions by blinking**-
	- Single blink (lasting 1-5 frames @ ~100-200fps video-stream) – *left click*
	- Double blink (lasting 1-5 frames @ ~100-200fps video-stream) – *double left click*
  - Left Eye [Toggle] - *Disable Mouse Control from Nose*
  - Face Up and Down [Mouse_Control=False] - *Page Scroll up and Down*
	- Mouth Open [Toggle] (lasting 1-5 frames @ ~100-200fps video-stream) –*press mouse down* (for scroll and drag)
	- (long) Single blink (lasting >5 frames @ ~100-200fps video-stream) – *right click*


## How To Use

To clone and run this application, you'll need [Python 3](https://www.python.org/) installed on your computer.
From your command line:

```bash
# Clone this repository
$ git clone https://github.com/AkashCS12/FaceMouse-Modified.git

# Go into the repository
$ cd FaceMouse

#Create Virtual Environement
$ python3.8 -m venv venv

# Install dependencies
$ pip install -r requirements.txt

# Run the app
$ python3 face_mouse_visual.py
```

## Credits

This project is a modified version of facemouse by github user [@shivang02]
.
This software mainly uses the following open source packages (among others):

- [OpenCV](https://opencv.org/)
- [Dlib](http://dlib.net/)
- [NumPy](https://numpy.org/)
- [PyAutoGUI](https://pypi.org/project/PyAutoGUI/)
- [Mouse](https://pypi.org/project/mouse/)

The following sources helped with some logic in the code:

- [Eye blink detection with OpenCV, Python, and dlib](https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/)
