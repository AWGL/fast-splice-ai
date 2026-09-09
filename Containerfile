# fast-splice-ai HTTP service.
#
# The reference genome and the SpliceAI model weights are NOT baked in: the
# reference is ~3 GB and the weights are Illumina's under CC BY-NC 4.0. Both are
# mounted read-only at run time. The MANE v1.5 transcript tables are included,
# since they are small and public domain.
FROM docker.io/library/python:3.12-slim

RUN useradd --create-home --uid 10001 fastspliceai
WORKDIR /app

# dependencies first, so source edits do not invalidate the layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

USER fastspliceai
EXPOSE 8722
VOLUME ["/cache"]

# Inside the container, bind 0.0.0.0 so podman can publish the port; the
# published port is bound to 127.0.0.1 on the host by the unit file. There is
# still no authentication - see docs.
ENTRYPOINT ["python", "-m", "fastspliceai.server", \
            "--host", "0.0.0.0", "--port", "8722", \
            "--cache", "/cache/scores.db"]
# Cap threads by default: this is meant to sit alongside other services, not to
# take the machine. Override with --cpus / CPUQuota, or FASTSPLICEAI_CPUS.
ENV FASTSPLICEAI_CPUS=4
CMD ["-R", "/refs/genome.fa", \
     "-A", "/app/data/mane_v1.5/annotation.txt.gz", \
     "-t", "/app/data/mane_v1.5/genepred.txt.gz", \
     "-m", "/models", \
     "-D", "500"]
