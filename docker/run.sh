xhost +
#docker run --gpus 1 -it adam_visual:0.1 /bin/bash
docker run --rm -ti --net=host --ipc=host \
   -e DISPLAY=$DISPLAY \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   --gpus 1 -it adam_visual:0.1 /bin/bash

