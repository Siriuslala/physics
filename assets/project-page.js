"use strict";

const comparisons = [...document.querySelectorAll(".comparison")];
const buttons = [...document.querySelectorAll(".playback")];
let paused = false;

/** Show the synchronized GIF if the browser rejects video autoplay or decoding. */
function showFallback(figure) {
  const video = figure.querySelector("video");
  const fallback = figure.querySelector(".fallback");
  video.pause();
  video.hidden = true;
  fallback.src = paused ? video.poster : fallback.dataset.src;
  fallback.hidden = false;
}

/** Start one visible comparison; native autoplay remains available without JS. */
function playComparison(figure) {
  const video = figure.querySelector("video");
  if (paused || document.hidden) return;
  if (video.hidden) {
    figure.querySelector(".fallback").src = figure.querySelector(".fallback").dataset.src;
    return;
  }
  video.muted = true;
  video.play().catch(() => {
    if (!paused && !document.hidden && figure.dataset.visible === "true") showFallback(figure);
  });
}

/** Pause offscreen media and start visible clips to limit simultaneous decoding. */
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    const figure = entry.target;
    figure.dataset.visible = String(entry.isIntersecting);
    if (entry.isIntersecting) playComparison(figure);
    else {
      const video = figure.querySelector("video");
      video.pause();
      if (video.hidden) figure.querySelector(".fallback").src = video.poster;
    }
  });
}, { threshold: 0.05 });

comparisons.forEach(figure => {
  const video = figure.querySelector("video");
  video.addEventListener("error", () => showFallback(figure));
  if (video.error) showFallback(figure);
  observer.observe(figure);
});

buttons.forEach(button => {
  button.hidden = false;
  button.addEventListener("click", () => {
    paused = !paused;
    buttons.forEach(control => {
      control.textContent = paused ? "Resume animations" : "Pause animations";
      control.setAttribute("aria-pressed", String(paused));
    });
    comparisons.forEach(figure => {
      const video = figure.querySelector("video");
      if (paused) {
        video.pause();
        if (video.hidden) figure.querySelector(".fallback").src = video.poster;
      } else if (figure.dataset.visible === "true") playComparison(figure);
    });
  });
});

document.addEventListener("visibilitychange", () => {
  comparisons.forEach(figure => {
    if (document.hidden) {
      const video = figure.querySelector("video");
      video.pause();
      if (video.hidden) figure.querySelector(".fallback").src = video.poster;
    } else if (figure.dataset.visible === "true") playComparison(figure);
  });
});
