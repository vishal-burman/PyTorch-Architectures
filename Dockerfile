FROM continuumio/miniconda3
LABEL maintainer="Vishal Burman"
LABEL repository="PyTorch-Architectures"

RUN apt update && \
    apt install -y apt-file && \
    apt-file update && \
    apt install -y vim


WORKDIR /home/Desktop/
RUN git clone https://github.com/vishal-burman/PyTorch-Architectures.git && \
    pip install -r PyTorch-Architectures/requirements.txt

CMD ["/bin/bash"]
