# gta-render: GTA-only pod image (owner rule 11-16: GTA never shares Tesla's nc-render image/repo).
# Same base recipe as the news-clone engine needs + the GTA 2.5D parallax model, so pod_gta/agent/setup.sh skips every
# install/download and only runs the NVENC/overlay probe (~10 s instead of ~4-6 min). No keys inside.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive NVIDIA_DRIVER_CAPABILITIES=all NVIDIA_VISIBLE_DEVICES=all HF_HOME=/opt/hf
RUN apt-get update -qq && apt-get install -y -qq unzip curl xz-utils fonts-liberation fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
RUN curl -s https://rclone.org/install.sh | bash && rclone version | head -1
# torch = laptop versions (cu128 needs host driver >= 570; podrun asks RunPod for CUDA 12.8+ hosts only)
RUN pip install -q --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip install -q --no-cache-dir openai-whisper==20250625 open_clip_torch==3.3.0 transformers pillow numpy requests soundfile
# weights baked (parallel jobs would race on the download)
RUN python3 -c "import whisper; whisper.load_model('small.en', device='cpu')" \
 && python3 -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')" \
 && python3 -c "from transformers import AutoImageProcessor as P, AutoModelForDepthEstimation as M; m='depth-anything/Depth-Anything-V2-Small-hf'; P.from_pretrained(m); M.from_pretrained(m)"
# BtbN static ffmpeg builds; setup.sh probes them on the real GPU at boot (NVENC + overlay_cuda) and picks one
RUN mkdir -p /opt/ff && cd /opt/ff && for V in 9.0 8.1 8.0 7.1; do \
      curl -sfL "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n${V}-latest-linux64-gpl-${V}.tar.xz" -o f.tar.xz \
      && mkdir -p x$V && tar -xJf f.tar.xz -C x$V --strip-components=1 && rm -f f.tar.xz || echo "no build n$V"; \
    done; ls /opt/ff
RUN touch /opt/.nc_render_image /opt/.gta_render_image
