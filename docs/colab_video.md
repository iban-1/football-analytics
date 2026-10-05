# Running the video phase on free Google Colab (GPU)

> Written for convenience and **not tested**. The pipeline itself was run and verified on a
> laptop CPU (about 10 minutes for a 2-minute clip). Colab only makes detection faster.

Everything here is free (Colab free tier, no credit card). Do not upload footage you have no
permission to use.

1. Open https://colab.research.google.com and create a new notebook.
2. Turn on a GPU: **Runtime > Change runtime type > T4 GPU**.
3. Put the project in the notebook (pick one):
   - upload a zip of the repo (left sidebar > Files > upload) and run `!unzip -q football-analytics.zip`, or
   - `!git clone <your-repo-url>` if you pushed it to GitHub.
4. Install the packages and upload your clip:
   ```python
   %cd football-analytics
   !pip install -q -r requirements.txt -r requirements-video.txt
   from google.colab import files
   !mkdir -p data/video
   uploaded = files.upload()          # choose your .mp4
   ```
5. Make sure `video.path` in `config.yaml` is the uploaded file name, for example
   `data/video/clip.mp4` (move the uploaded file there with `!mv clip.mp4 data/video/`).
   If you use a different clip you must also re-pick the pitch landmarks for it (see below).
6. Run it:
   ```python
   !python -m scripts.phase7_video
   ```
   Ultralytics uses the GPU automatically when one is available.
7. Download `results/phase7/annotated.mp4`, `radar.mp4` and the CSV/PNG files from the Files panel
   (the session and the uploaded clip are deleted when Colab disconnects).

## Using a different clip

`config.yaml` > `video.mapped_shot` holds the shot's frame range, a reference frame, and eight
landmark pairs (pixel position in the reference frame <-> pitch position in metres). They were
read by hand from one frame of the demo clip and will be wrong for any other clip. For a new
clip, set `start`, `end`, `reference_frame` and re-pick the landmarks on a frame where box
corners, goalposts and the penalty arc are visible. The code checks the result: it overlays
the pitch on the picture and reports an alignment score.
