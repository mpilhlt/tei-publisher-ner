# FROM python:3
FROM nvcr.io/nvidia/ai-workbench/python-cuda122:1.0.6

ARG BRANCH=feature/spancat

RUN apt-get update && apt-get install -y git

WORKDIR /workspace

# Install Cuda?
# On Archlinux, the packages were: cuda cuda-tools python-pycuda python-pytorch-cuda
# We try with a cuda-equipped base image provided by nvidia first, see FROM command above

# Install tei-publisher-ner plus German and English language models
RUN git clone https://github.com/mpilhlt/tei-publisher-ner.git \
    && cd tei-publisher-ner \
    && git checkout ${BRANCH} \
    && pip3 install --no-cache-dir --upgrade -r requirements.txt \
    && python3 -m spacy download en_core_web_trf \
    && python3 -m spacy download en_core_web_md \
    && python3 -m spacy download de_dep_news_trf \
    && python3 -m spacy download de_core_news_md

EXPOSE 8001

WORKDIR /workspace/tei-publisher-ner

CMD [ "python3", "-m", "spacy", "project", "run", "serve" ]
