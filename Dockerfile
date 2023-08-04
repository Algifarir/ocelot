# Use the NVIDIA CUDA base image with GPU support
FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# Install libGL and libglib2.0 libraries for Debian-based systems
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Install Python 3 and pip
RUN apt-get install -y python3 python3-pip

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app/models

COPY ocelot400.h5 /opt/app/models/

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
# Set the working directory inside the container
WORKDIR /opt/app/

# Set the PYTHONPATH environment variable
ENV PYTHONPATH "${PYTHONPATH}:/opt/app/"

# Install pip-tools globally inside the container
RUN python3 -m pip install pip-tools

# Copy the contents of the current directory into the container working directory
COPY ./ /opt/app/

# RUN pip install numpy opencv-python pandas tensorflow

RUN pip3 install --no-cache-dir --upgrade pip \
  && pip3 install --no-cache-dir -r requirements.txt \
  && rm -rf /tmp/*


COPY --chown=user:user process.py /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]

