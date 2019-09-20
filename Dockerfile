FROM python:3.6

COPY . /root/hand_tracking

WORKDIR /root/hand_tracking

RUN pip install opencv-python Pillow

RUN pip install tensorflow-1.14.1-cp36-cp36m-linux_x86_64.whl
