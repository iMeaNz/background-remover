# Background Remover
This a school project aiming to implement a real time background remover from a live webcam feed.  
This background remover project is the culmination of an end-of-semester assignment for the "Image Processing" course, part of the curriculum of the software major at Chung-Ang University. 

# Table of Contents
- [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Setup Instructions (using cmd)](#setup-instructions-using-cmd)
  * [Setup Instructions (using Visual Studio)](#setup-instructions)
- [Running the Project](#running-the-project)
  * [Basic mask](#basic-mask)
  * [Edge detection](#edge-detection)
  * [Artificial intelligence detection](#artificial-intelligence-detection)
- [Additional Information](#additional-information)
- [Contributors](#contributors)

# Getting Started
Follow these steps to set up and run the project on your local machine.

## Prerequisites
- Python 3.7+ installed on your system (tested with Python 3.12.7).
- pip (Python's package installer) installed.
- Visual Studio 2022 (optional) for an integrated development environment.

## Setup Instructions (using cmd)

> [!NOTE]
> The following instructions are for setting up the project using the command line. If you prefer using Visual Studio, please refer to the next section.

1. **Clone the Repository**
 
Clone the project repository to your local machine:
```bash
git clone git@github.com:iMeaNz/background-remover.git
cd background-remover
```

2. **Create a Virtual Environment**

Create a Python virtual environment to isolate project dependencies:
```bash
python -m venv venv
```

Activate the virtual environment:
- On Linux/macOS:
```bash
source venv/bin/activate
```
- On Windows:
```bash
venv\Scripts\activate
```
You’ll know the virtual environment is active when your shell shows (venv) at the beginning of the prompt.

3. **Install Dependencies**
Install the required Python packages using the requirements.txt file:
```bash
pip install -r requirements.txt
```

## Setup Instructions (using Visual Studio)

> [!WARNING]
> It seems that when running the project using Visual Studio, python takes a few seconds to load the program: if you see an empty terminal, wait a few seconds and the program should start.

To use our project in Visual Studio, follow these steps:

1. **Install the Python workload in VS2022**
Please follow the instructions in the [official documentation](https://learn.microsoft.com/en-us/visualstudio/python/installing-python-support-in-visual-studio?view=vs-2022) to install the Python workload in Visual Studio 2022.

2. **Open the Project in Visual Studio**
Open Visual Studio 2022 and select "Open a project or solution" from the start page. Navigate to the project folder and select the .sln file to open the project in Visual Studio.

3. **Set Up the Virtual Environment**
Let Visual Studio create a virtual environment for the project, you will be prompted for example to install the required packages (requirements.txt) in the virtual environment.
If you are not prompted, you can manually create a virtual environment by following these steps: In the Solution Explorer, right-click on Python Environments and select "Add Environment". Choose the location for the virtual environment and select the Python interpreter 3.12.

4. **Run the Project**
By default, ai_detection.py is set as the startup file. You can change this by right-clicking on the desired file in the Solution Explorer and selecting "Set as Startup File". Press F5 to run the project.
See below all the files and their functionalities.

# Running the Project

Ensure the virtual environment is active.  
There are 3 different programs in this repository

## Basic mask
This program is a very naive implementation.

The main workflow of the algorithm is to capture the first 10 seconds of the live feed without the subject. This is used to store the static background in the program.  

Then, we simply apply a binary mask onto the current frame with the static background we stored previously.  

We realized this is not really usable because of several hardware problems (webcam's obturation speed and auto-focus). But it was still an interesting implementation to do.

**How to run**
```bash
python basic_mask.py
```

## Edge detection
This is a bit more advanced, we use the canny algorithm and p-tile threshold algorithm to convert the current frame to a binary image containing only the edges of the image.  

Then we find the bounding box of these edges to apply our background removing logic.  

This was interesting, but still not really the desired output, because a box does not perfectly encompass the shape of a person, and if the background is not unified and contains several edges, the bounding box detection can be scuffed.

**How to run**
```bash
python edge_detection.py
```

Options:
 - m: Toggle magnitude visualization.
 - g: Toggle gradient visualizations (Gx, Gy).
 - s: Toggle suppressed visualization.
 - [ + / - ]: Adjust the edge detection threshold.

## Artificial intelligence detection
For this one, we use the [DeepLabV3 model](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/) to detect the person in the image.

The model returns a mask that we can use to apply our background remover logic.

This program runs better if you have an NVIDIA graphic card, if not, it will run by default on your CPU.

**How to run**
```bash
python ai_detection.py
```
Options:
1. Segmentation Overlay: Highlights the person, keeping the original background.
2. Custom Solid Color Background: Allows setting a solid color background using color sliders.
3. Gaussian Blurred Background: Blurs the background for a professional effect.
4. Custom Background Image: Replaces the background with an image chosen by the user.

(use keys 1, 2, 3, 4 to switch between the different modes)

## Deactivating the Virtual Environment
When you're done working on the project, deactivate the virtual environment:

```bash
deactivate
```
# Additional Information

If you encounter issues with missing dependencies, ensure they are installed using pip install.

Also make sure that you use recommended Python version (3.12.7 or at least 3.12). We cannot guarantee that the project will work with other versions.

# Contributors

- Guillaume HEIN
- Mehdy MORVAN
- Noah GOSCINIAK
