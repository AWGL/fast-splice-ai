# Running fast-splice-ai as a podman service

A warm engine answers in ~0.5 s where a cold CLI invocation takes ~3 s, and a
cache hit is ~1 ms. This runs that engine as a rootless container managed by
systemd.

## What is and is not in the image

| | where |
|---|---|
| code, dependencies | in the image |
| MANE v1.5 transcript tables | in the image (small, public domain) |
| **reference genome** | mounted read-only — ~3 GB |
| **SpliceAI model weights** | mounted read-only — Illumina's, CC BY-NC 4.0 |
| score cache | named volume, survives rebuilds |

The weights are deliberately not baked in: redistributing them in an image
would breach their licence.

## Build

```sh
podman build -t fastspliceai:latest .
```

## Run it once, by hand

```sh
podman run --rm -p 127.0.0.1:8722:8722 \
  -v /data/refs/hg38.fa:/refs/genome.fa:ro,z \
  -v /data/refs/hg38.fa.fai:/refs/genome.fa.fai:ro,z \
  -v $HOME/spliceai/models:/models:ro,z \
  -v fastspliceai-cache:/cache:z \
  fastspliceai:latest

curl -s localhost:8722/health
curl -s localhost:8722/score -d '{"variants":["chr9-128504906-T-G"]}'
```

## Run it as a service

Quadlet, so systemd manages it and there is no generated unit to regenerate:

```sh
mkdir -p ~/.config/containers/systemd
cp podman/fastspliceai.container podman/fastspliceai-cache.volume \
   ~/.config/containers/systemd/
$EDITOR ~/.config/containers/systemd/fastspliceai.container   # fix the three paths

systemctl --user daemon-reload
systemctl --user start fastspliceai
systemctl --user status fastspliceai
journalctl --user -u fastspliceai -f
```

To keep it running when you are not logged in:

```sh
loginctl enable-linger $USER
```

## Notes

* **The port is published on `127.0.0.1` only.** The service has no
  authentication, no TLS and no audit log. Inside the container it binds
  `0.0.0.0` because that is how podman publishes a port — the host binding is
  what matters. Do not change `PublishPort` to `0.0.0.0` and leave it at that.
* `:z` on the volume mounts relabels for SELinux. Drop it on systems without
  SELinux if it causes trouble.
* `ReadOnly=true` with a tmpfs for `/tmp`; the cache volume is the only writable
  path. The container runs as uid 10001, not root.
* `Memory=6G` is for the mmapped reference. Lower it and the FASTA reads will
  thrash.
* The cache is keyed by a fingerprint of the annotation, model files, `-D` and
  `-M`. Swapping any of them makes old entries stop matching rather than being
  served — but the rows stay until you pass `--purge-stale`.
