# Project Page and Qualitative Media

## Purpose and content

The static project page presents the paper *Why Do Video Diffusion Models Violate Physics? Unveiling the Flaws in Attention Mechanisms* at <https://siriuslala.github.io/physics/>. Its reading order is the title and authors, the verbatim abstract, four basketball comparisons, the self-attention mechanism and RoPE intervention, and six VideoPhy comparisons. Author names follow the repository README. The corresponding-author marker follows the author's public publication list. No affiliation or arXiv identifier is inferred.

The method section uses Figure 6 to illustrate spatial anchoring and Figure 8 to show early candidate competition. Figure 8's winner is defined relative to the final generated trajectory, not to a physically correct ground truth. The diagram is available at full resolution through its image link. The intervention reproduces Equation (3) of Section 6.1 using native MathML, without an external equation-rendering service.

`index.html` is the page entry point. Styles and playback behavior are in `assets/project-page.css` and `assets/project-page.js`. The paper links point to `https://arxiv.org` until the public arXiv URL is available. The review manuscript is private and must not be copied into assets or a publication bundle. `assets/candidate_evolution.png` contains Figure 8 cropped from page 7 of the supplied PDF at six pixels per PDF point. The main README uses smaller figures and animated comparisons linked to the project page.

## Media preparation

The media utility is `wan21_t2v_experiments/prepare_project_media.py`; its launcher is `scripts/prepare_project_media.sh`. This is a CPU-only presentation utility, independent of model inference and the experiment dispatcher. Dependencies are Pillow and `imageio-ffmpeg`, which supplies an FFmpeg executable.

```bash
python -m pip install Pillow imageio-ffmpeg
bash scripts/prepare_project_media.sh /work/liyueyan/Interpretability/physics/viz
```

The source contains ten directories, each with `before.mp4` (the original model) and `after.mp4` (the proposed method). The script moves these originals into `assets/videos/<sample>/`, retaining the original bytes. It refuses to overwrite an existing original with different contents. On reruns, existing destination originals can be reused even after their source files have been moved. Sample names and prompts are specified in `EXAMPLES`; no model size or exact intervention variant is inferred from filenames.

To regenerate a comparison after replacing an original in `assets/videos/`, omit `--source` and select the sample ID:

```bash
python wan21_t2v_experiments/prepare_project_media.py --examples basketball-seed-23
```

`--examples` accepts multiple sample IDs. Without it, all examples are regenerated. Existing manifest records for unselected examples are retained. With no `--source`, all originals are read directly from `assets/videos/`.

Each output directory contains:

| File | Purpose |
| --- | --- |
| `before.mp4`, `after.mp4` | Unmodified source videos |
| `comparison.mp4` | Synchronized side-by-side comparison for the website |
| `comparison.gif` | Automatically animated README image and browser fallback |
| `poster.jpg` | Initial website preview and paused GIF representation |

`assets/videos/manifest.json` records each source folder, caption, dimensions, frame rate, duration, and SHA-256 hashes of the originals. Newly generated records also include `comparison_gap_pixels`, the gutter width before GIF resizing. Before composition, the utility requires the two sources to have matching dimensions, frame rates, and durations. The original sequence is neither cropped nor retimed. Each MP4 panel is resized to 416 pixels wide with its aspect ratio preserved; a label strip identifies the baseline on the left and the proposed method on the right. VideoPhy comparisons include a 24-pixel white gutter between panels and their labels (856 pixels total width, approximately 18 gutter pixels after GIF resizing). Basketball comparisons have a total width of 832 pixels. The same composed layout is used by the MP4, GIF and poster. H.264, YUV 4:2:0 and MP4 fast-start metadata support browser playback. The GIF has a total width of 640 pixels, 12 frames per second, a 192-color palette, and an infinite loop. GIF sampling changes temporal sampling density, not playback speed; its duration is quantized to the GIF frame intervals.

The GIF is generated from the synchronized `comparison.mp4` with this FFmpeg filter graph:

```text
fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle
```

The two branches share the same resized frames: one builds an adaptive palette, and the other applies it with ordered Bayer dithering. `-loop 0` enables infinite repetition. The first comparison frame is exported as `poster.jpg`. Regeneration is a local preparation step; commit the generated files to publish them through the existing workflow.

## Playback

GitHub README comparisons are GIF images because README markup does not offer dependable control over MP4 autoplay. The website uses `autoplay muted loop playsinline` on its MP4 elements. A shared timeline in each composite prevents before/after drift. JavaScript pauses clips outside the viewport and when the tab is hidden, and resumes visible clips when appropriate. If video decoding or autoplay fails, the corresponding animated GIF is displayed automatically. A pause/resume control is available for readers who prefer to stop the animations. No click is required to start ordinary playback.

The site uses system fonts, native MathML, local assets, and no external JavaScript libraries. It supports desktop and mobile layouts. At narrow widths the comparison cards form a single column; each card still shows the two synchronized panels together.

## Preview and publication

To preview the repository directly:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Open <http://127.0.0.1:8000/>. To assemble only the public website, without experiment code or unneeded original media:

```bash
bash scripts/build_project_page.sh /tmp/physics-project-page
python -m http.server 8000 --bind 127.0.0.1 --directory /tmp/physics-project-page
```

## Automatic deployment from this repository

The source repository is `Siriuslala/physics`. The workflow `.github/workflows/pages.yml` publishes the static site directly to <https://siriuslala.github.io/physics/>. The personal website repository is not part of the publication pipeline.

One-time repository setup: open **Settings → Pages → Build and deployment → Source** and select **GitHub Actions**. This setting requires repository management permissions. Once enabled, normal publication requires only a commit and push to `main`:

```bash
git add .
git commit -m "Update project page"
git push origin main
```

The push updates the repository immediately. If it changes `index.html`, anything under `assets/`, `scripts/build_project_page.sh`, or the workflow itself, GitHub Actions then builds and deploys the page asynchronously. The website updates after that deployment succeeds. Changes only to experiment code or documentation do not trigger a website deployment. A push to another branch does not update the live page; merge into `main` to publish it. The workflow also supports **Actions → Deploy project page → Run workflow** on `main` for a manual deployment.

The build job runs the static-site assembly script on a GitHub-hosted runner. It copies the website's explicit asset list and prepared comparison media into a temporary build directory; it does not run model inference or regenerate videos. The upload step packages that directory as a Pages artifact. The deployment job has `pages: write` and `id-token: write` permissions and publishes the artifact to the `github-pages` environment. `deployment.json` records the source repository and commit for deployment verification. Deployment jobs are serialized so that concurrent pushes do not interrupt a running release.

Local source files stay in the research checkout. Build directories under `/tmp` are disposable previews or output bundles, not additional source repositories. For new video examples, generate the MP4, GIF and poster before committing, and update the HTML cards as described below.

## The `.nojekyll` marker

`.nojekyll` is an empty file traditionally used by branch-based GitHub Pages publication to skip Jekyll processing and serve prebuilt static files. The build script creates it in the output directory with `touch`; it is not a hand-maintained source file and therefore does not appear in the repository root. A leading dot makes it hidden in ordinary directory listings; use `ls -a` to see it.

This project's custom Actions workflow uploads the already assembled static site and does not invoke Jekyll. It therefore does not rely on `.nojekyll`, but retaining the marker makes the output bundle usable by traditional static branch publishing as well. No local marker needs to be created before committing or pushing source changes.

The header and footer paper links use `https://arxiv.org` as a placeholder. Update both links in `index.html` when the public arXiv URL is available. Never publish the review PDF. The build script removes any legacy `assets/physics.pdf` from its output directory. To add examples, update `EXAMPLES`, regenerate the media, and add the corresponding static cards to `index.html` and the README.
