# Buil-Research-Envirorment-with-Tensorflow-OpenCV-Python
(Version 0.1, Last Update 24/05/2016)

Tutorial on how to build your own research envirorment for Deep Learning with OpenCV, Python, Tensorfow


The Project  follow the below **index**:

1. **Introduction to the Problem;**
2. **Possible Solutions;**
3. **Virtual Environment**
      1. **Virtual Environment & Python;**
      2. **OpenCV;**
      3. **Tensorflow;**
      4. **CUDA;**
      5. **Other Libraries;**
4. **Conclusions;**
5. **References;**
6. **Copyright**



##1. Introduction to the Problem

Nowadays, the Deep Learning Area use lots of  different libraries, dependicies, script languages, some of the most known are:
- Caffè (http://caffe.berkeleyvision.org/);
- Tensorflow (https://www.tensorflow.org/);
- Keras (http://keras.io/);
- OpenCV (http://opencv.org/).

So Develop & Research on Deep Learning, now means struggle with lots of error, compilation, installation, and some times most of them comes from updates, new library dependecies, let's see which are the possible solutions.

##2. Possible Solutions

Mainly we have three solutions:
- Install on own machine;
- Install in a server (Use the environment remotly);
- Install on Virtual Environment.
This last one is the one, covered in this Tutorial.

##3. Virtual Environment

**Why Virtual Environment?**

Why is easy, Virtual Envirorments lets us to have different dependencies, libraries, updates, of the same or different software, without struggling with error due to mislink or new apis.

The tutorial code consider you are the owner and have all grants to run **sudo** , if you  wanna apply the tutorial to a server you have to ask your administrator to install OpenCV & CUDA on the machine and give you the path to export into the bashrc file, for VirtualEnv, Tensorflow, Python nothing change.

Let's see a four step set up installation for our Research Envirorment with Tensorflow, OpenCV, Python, CUDA and all the libraries that we could need:

### i.Virtual Environment & Python

Open up a terminal and update the apt-get package manager followed by upgrading any pre-installed packages:
```
$ sudo apt-get update 
$ sudo apt-get upgrade

```
Now we need to install our developer tools:
```
$ sudo apt-get install build-essential cmake git pkg-config
```

The pkg-config is likely already installed, but be sure to include it just in case. 
We’ll be using git to pull down the OpenCV repositories from GitHub. 
The cmake package is used to configure our build.

OpenCV needs to be able to load various image file formats from disk, including JPEG, PNG, TIFF, etc. In order to load these image formats from disk, we’ll need our image I/O packages:
```
$ sudo apt-get install libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev
```
At this point, we have the ability to load a given image off of disk. But how do we display the actual image to our screen? The answer is the GTK development library, which the highgui  module of OpenCV depends on to guild Graphical User Interfaces (GUIs):
```
$ sudo apt-get install libgtk2.0-dev
```
We can load images using OpenCV, but what about processing video streams and accessing individual frames? We’ve got that covered here:
```
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
```
Install libraries that are used to optimize various routines inside of OpenCV:
```
$ sudo apt-get install libatlas-base-dev gfortran
```
Install pip, a Python package manager:
```
$ wget https://bootstrap.pypa.io/get-pip.py $ sudo python get-pip.py
```

Install virtualenv and virtualenvwrapper.These two packages allow us to create separate Python environments for each project we are working on. While installing virtualenv  and virtualenvwrapper is not a requirement to get OpenCV 3.0 and Python 2.7+ up and running on your Ubuntu system, I highly recommend it and the rest of this tutorial will assume you have them installed!
```
$ sudo pip install virtualenv virtualenvwrapper 
$ sudo rm -rf ~/.cache/pip
```
Now that we have virtualenv  and virtualenvwrapper  installed, we need to update our ~/.bashrc  file:
```
$ export WORKON_HOME=$HOME/.virtualenvs source /usr/local/bin/virtualenvwrapper.sh
```
This quick update will ensure that both virtualenv  and virtualenvwrapper  are loaded each time you login.
To make the changes to our ~/.bashrc  file take effect, you can either (1) logout and log back in, (2) close your current terminal window and open a new one, or preferably, (3) reload the contents of your ~/.bashrc  file:
```
$ source ~/.bashrc
```
Lastly, we can create our cv  virtual environment where we’ll be doing our computer vision development and OpenCV 3.0 + Python 2.7+ installation:
```
$ mkvirtualenv cv
```
As I mentioned above, this tutorial covers how to install OpenCV 3.0 and Python 2.7+, so we’ll need to install our Python 2.7 development tools:
```
$ sudo apt-get install python2.7-dev
```
Since OpenCV represents images as multi-dimensional NumPy arrays, we better install NumPy into our cv  virtual environment:
```
$ pip install numpy
```

### ii.OpenCV
  
### iii.Tensorflow
  
### iv.CUDA
  
### v.Other Libraries
  
##4. Conclusions
##5. References
##6. Copyright
