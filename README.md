## Hand tracking

```
$ docker build -t hand_tracking .
$ docker run -dit --device=/dev/video0:/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$IP:0 --name hand_tracking hand_tracking bash
```
