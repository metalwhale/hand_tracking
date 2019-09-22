FROM python:3.7

COPY . /root/hand_tracking

WORKDIR /root/hand_tracking

RUN pip install opencv-python Pillow tensorflow

ENTRYPOINT python run.py
