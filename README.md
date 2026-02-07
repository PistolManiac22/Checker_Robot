# Checker Robot
Checker Robot is a robotics + computer vision project that plays checkers using a TM5-900 robotic arm with a built-in camera. It detects the board and pieces with OpenCV, validates and plans moves with an Alpha-Beta–based game engine, and executes physical actions through the robot arm. A Gemini-powered commentator provides real-time feedback and strategy tips to make gameplay more engaging.

## Highlights

- **Vision-driven board detection** using OpenCV for robust piece and board recognition.
- **Move validation and planning** via a game engine that uses Alpha-Beta pruning.
- **Robotic execution** of moves through the TM5-900 arm control stack.
- **AI commentary** for play-by-play descriptions and recommendations.

## Repository Layout

- `Game/`: Game logic and rule enforcement.
- `GameEngine.py`: Alpha-Beta–based decision engine.
- `ImageDetector.py`: Board/piece detection utilities.
- `CoordinateTransformer.py`: Pixel-to-world coordinate conversions.
- `Mover.py` / `RobotArm.py`: Motion planning and robot control helpers.
- `MoveCommentator.py` / `CommentReader.py`: Commentary generation and playback.
- `calibrate/`: Calibration utilities and assets.
- `TestImage/`: Example images for detection testing.
- `run_commentator.sh`: Shell helper to run the commentator pipeline.

## Getting Started

> This project is hardware- and environment-dependent. Use the steps below as a starting point, then adapt them to your robot and camera setup.

1. **Create a Python environment** (recommended).
2. **Install dependencies** required for OpenCV, AI APIs, and hardware SDKs.
3. **Calibrate the camera/board** using the tools in `calibrate/`.
4. **Run the game pipeline** using your preferred entry point (e.g., `GameEngine.py`).

## Usage Notes

- Make sure the robot’s coordinate frame is aligned with the calibrated board coordinates.
- Update any API credentials for the commentator module before running.
- Use the images in `TestImage/` to validate detection logic before live play.

## License

MIT License

Copyright (c) 2025 Checker Robot

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
