FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
MAINTAINER Mikayel Samvelyan

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo cmake gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git

# Install python3 pip3
RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get -y install python3-dev
RUN pip3 install --upgrade pip

RUN pip3 install numpy scipy pyyaml matplotlib

# Install OpenCV 4.2.0 from source code 
RUN apt-get install -y libjpeg-dev libpng-dev libtiff-dev
RUN apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
RUN apt-get install -y libxvidcore-dev libx264-dev
RUN apt-get install -y libgtk-3-dev

RUN export TZ=America/Los_Angeles && DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

RUN apt-get install -y libatlas-base-dev gfortran unzip

RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip && unzip opencv.zip && mv opencv-4.2.0 opencv
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip && unzip opencv_contrib.zip && mv opencv_contrib-4.2.0 opencv_contrib

RUN cd opencv && mkdir build && cd build && \
	cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D INSTALL_C_EXAMPLES=OFF \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
	-D PYTHON_EXECUTABLE=/usr/bin/python3 \
	-D BUILD_EXAMPLES=ON \
	-D WITH_CUDA=ON \
	-D WITH_CUDNN=ON \
	-D OPENCV_DNN_CUDA=ON \
	-D ENABLE_FAST_MATH=1 \
	-D CUDA_FAST_MATH=1 \
	-D CUDA_ARCH_BIN=7.5 \
	-D WITH_CUBLAS=1 .. && \
	make -j8 && make install && ldconfig

RUN cd /usr/local/lib/python3.6/dist-packages/cv2/python-3.6 && mv cv2.cpython-36m-x86_64-linux-gnu.so cv2.so
RUN rm -rf opencv.zip opencv_contrib.zip

# Install Tensorflow and Keras
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow==2.2.0
RUN pip3 install keras==2.4.2

# Install OpenPose

## Dependencies
RUN apt-get --assume-yes install libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
RUN apt-get --assume-yes install --no-install-recommends libboost-all-dev
RUN apt-get --assume-yes install libgflags-dev libgoogle-glog-dev liblmdb-dev
RUN pip3 install --upgrade numpy protobuf
RUN apt-get update && apt-get --assume-yes install opencl-headers ocl-icd-opencl-dev
RUN apt-get --assume-yes install libviennacl-dev

RUN apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg software-properties-common wget
RUN wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | sudo apt-key add -
RUN apt-add-repository -y 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get install -y cmake

## Main code
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git openpose && cd openpose && \
    cd models && bash getModels.sh && cd .. && \
    mkdir build && cd build && \
    cmake -DOpenCV_CONFIG_FILE=/opencv/build/OpenCVConfig.cmake .. && \
    make -j8

# Install Adam Visual Perception
RUN git clone https://github.com/isi-vista/adam-visual-perception.git adam-visual-perception/ && \
    cd adam-visual-perception && pip3 install .

# Download pre-trained models
RUN cd adam-visual-perception && bash download_models.sh
