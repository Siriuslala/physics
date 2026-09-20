#!/usr/bin/env bash
set -euo pipefail

# Assemble the project's static website for local preview or GitHub Pages.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINATION="${1:-/tmp/physics-project-page}"
mkdir -p "${DESTINATION}/assets/videos"
# Remove any review-only manuscript left by a previous build.
rm -f "${DESTINATION}/assets/physics.pdf"
cp "${ROOT_DIR}/index.html" "${DESTINATION}/index.html"
for name in project-page.css project-page.js teaser.png self_attention.png candidate_evolution.png; do
    cp "${ROOT_DIR}/assets/${name}" "${DESTINATION}/assets/${name}"
done
for source_dir in "${ROOT_DIR}"/assets/videos/*/; do
    sample_name="$(basename "${source_dir}")"
    mkdir -p "${DESTINATION}/assets/videos/${sample_name}"
    for name in comparison.mp4 comparison.gif poster.jpg; do
        cp "${source_dir}${name}" "${DESTINATION}/assets/videos/${sample_name}/${name}"
    done
done
touch "${DESTINATION}/.nojekyll"
printf 'Static site ready: %s\n' "${DESTINATION}"
