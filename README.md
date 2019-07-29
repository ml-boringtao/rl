# Udacity Reinforcement Learning
Solutions to the projects Udacity's Deep Reinforcement Learning Nanodegree program.

## Environment installation
```
./Anaconda3-2019.07-Linux-x86_64.sh

conda create --name drlnd python=3.6
source activate drlnd
echo 'source activate drlnd' >> ~/.bashrc 

git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .

python -m ipykernel install --user --name drlnd --display-name "drlnd"

sudo apt-get install swig
pip install box2d
sudo apt-get install xvfb
pip install xvfbwrapper
pip install pyvirtualdisplay
```
