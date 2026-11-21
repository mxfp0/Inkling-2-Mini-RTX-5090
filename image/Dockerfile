ARG BASE_IMAGE=vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e
FROM ${BASE_IMAGE}
COPY patch_sm120.py /tmp/patch_sm120.py
RUN python3 -I /tmp/patch_sm120.py && rm /tmp/patch_sm120.py
