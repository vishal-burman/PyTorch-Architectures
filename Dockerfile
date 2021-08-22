FROM continuumio/miniconda3
LABEL maintainer="Vishal Burman"
LABEL repository="PyTorch-Architectures"

RUN apt update && \
    apt install -y apt-file && \
    apt-file update && \
    apt install -y vim


COPY . /home/Desktop/PyTorch-Architectures/
WORKDIR /home/Desktop/PyTorch-Architectures/
RUN pip install -r requirements.txt

CMD ["/bin/bash"]
