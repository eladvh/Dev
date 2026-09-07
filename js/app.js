import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const NEUTRAL_THRESHOLD = 0.25;
const HOLD_MS = 700;
const CELEBRATE_MS = 2000;

// Each expression scores a raw 0..1 confidence from face blendshape
// coefficients. The highest-scoring expression (above NEUTRAL_THRESHOLD)
// is treated as the face's current expression; otherwise it's "neutral".
const EXPRESSIONS = [
  {
    key: "smile",
    emoji: "😊",
    label: "happy",
    calc: (b) => avg(b("mouthSmileLeft"), b("mouthSmileRight")) * (1 - b("jawOpen") * 0.7),
  },
  {
    key: "laugh",
    emoji: "😂",
    label: "laughing",
    calc: (b) => avg(b("mouthSmileLeft"), b("mouthSmileRight")) * b("jawOpen"),
  },
  {
    key: "sad",
    emoji: "😢",
    label: "sad",
    calc: (b) => avg(b("mouthFrownLeft"), b("mouthFrownRight"), b("browInnerUp")),
  },
  {
    key: "surprised",
    emoji: "😲",
    label: "surprised",
    calc: (b) =>
      avg(
        b("eyeWideLeft"),
        b("eyeWideRight"),
        b("jawOpen"),
        b("browOuterUpLeft"),
        b("browOuterUpRight")
      ),
  },
  {
    key: "angry",
    emoji: "😠",
    label: "angry",
    calc: (b) =>
      avg(b("browDownLeft"), b("browDownRight"), b("mouthPressLeft"), b("mouthPressRight")),
  },
];

function avg(...vals) {
  return vals.reduce((a, v) => a + v, 0) / vals.length;
}

function classifyFace(blendshapeCategories) {
  const lookup = new Map(blendshapeCategories.map((c) => [c.categoryName, c.score]));
  const b = (name) => lookup.get(name) ?? 0;

  let best = { key: "neutral", score: NEUTRAL_THRESHOLD };
  for (const expr of EXPRESSIONS) {
    const score = expr.calc(b);
    if (score > best.score) best = { key: expr.key, score };
  }
  return best;
}

// ---- DOM ----
const startScreen = document.getElementById("start-screen");
const gameScreen = document.getElementById("game-screen");
const startBtn = document.getElementById("start-btn");
const startError = document.getElementById("start-error");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const targetEmojiEl = document.getElementById("target-emoji");
const scoreEl = document.getElementById("score");
const roundEl = document.getElementById("round");
const statusEl = document.getElementById("status");
const matchBanner = document.getElementById("match-banner");

// ---- Game state ----
let score = 0;
let round = 1;
let target = pickTarget();
let matchStartTime = null;
let faceLandmarker = null;
let celebrating = false;

function pickTarget(excludeKey) {
  const pool = excludeKey ? EXPRESSIONS.filter((e) => e.key !== excludeKey) : EXPRESSIONS;
  return pool[Math.floor(Math.random() * pool.length)];
}

function speak(text) {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}

function showTarget() {
  targetEmojiEl.textContent = target.emoji;
  targetEmojiEl.classList.remove("pop");
  void targetEmojiEl.offsetWidth;
  targetEmojiEl.classList.add("pop");
  speak(`Do a ${target.label} face!`);
}

function nextRound() {
  target = pickTarget(target.key);
  showTarget();
  matchStartTime = null;
  matchBanner.hidden = true;
}

function celebrate() {
  celebrating = true;
  matchStartTime = null;
  matchBanner.hidden = false;
  statusEl.textContent = "Nice! 🎉";
  setTimeout(() => {
    score += 1;
    round += 1;
    scoreEl.textContent = String(score);
    roundEl.textContent = String(round);
    celebrating = false;
    nextRound();
  }, CELEBRATE_MS);
}

async function createFaceLandmarker(filesetResolver) {
  const commonOptions = {
    runningMode: "VIDEO",
    numFaces: 2,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: false,
  };
  try {
    return await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      ...commonOptions,
    });
  } catch (err) {
    console.warn("GPU delegate failed, falling back to CPU:", err);
    return await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
      ...commonOptions,
    });
  }
}

startBtn.addEventListener("click", start);

async function start() {
  startBtn.disabled = true;
  startError.hidden = true;
  // Unlock speech synthesis on browsers (notably iOS Safari) that only allow
  // audio APIs to fire synchronously within a user-gesture handler.
  if ("speechSynthesis" in window) {
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(""));
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 960 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    startScreen.hidden = true;
    gameScreen.hidden = false;
    showTarget();

    statusEl.textContent = "Loading face detector…";
    const filesetResolver = await FilesetResolver.forVisionTasks(WASM_BASE);
    faceLandmarker = await createFaceLandmarker(filesetResolver);
    statusEl.textContent = "Get your face(s) in frame!";

    requestAnimationFrame(detectLoop);
  } catch (err) {
    console.error(err);
    startError.textContent =
      err && err.name === "NotAllowedError"
        ? "Camera permission was denied. Please allow camera access and try again."
        : `Couldn't start: ${err.message || err}`;
    startError.hidden = false;
    startBtn.disabled = false;
  }
}

function resizeOverlay() {
  overlay.width = video.videoWidth || overlay.clientWidth;
  overlay.height = video.videoHeight || overlay.clientHeight;
}

let lastVideoTime = -1;

function detectLoop() {
  if (video.readyState >= 2 && faceLandmarker) {
    if (overlay.width !== video.videoWidth) resizeOverlay();

    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const result = faceLandmarker.detectForVideo(video, performance.now());
      handleResult(result);
    }
  }
  requestAnimationFrame(detectLoop);
}

function handleResult(result) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (celebrating) return;

  const faces = result.faceBlendshapes || [];
  if (faces.length === 0) {
    statusEl.textContent = "Can't see a face — move closer or add more light.";
    matchStartTime = null;
    return;
  }

  const classifications = faces.map((f) => classifyFace(f.categories));
  const allMatch = classifications.every((c) => c.key === target.key);
  const confidence = Math.min(...classifications.map((c) => c.score));

  drawProgressRing(allMatch ? confidence : 0);

  if (allMatch) {
    statusEl.textContent =
      faces.length > 1 ? "Both of you match — hold it!" : "Matched — hold it!";
    if (matchStartTime === null) matchStartTime = performance.now();
    if (performance.now() - matchStartTime >= HOLD_MS) {
      matchStartTime = null;
      celebrate();
    }
  } else {
    matchStartTime = null;
    statusEl.textContent =
      faces.length > 1
        ? "Both faces need to match the emoji"
        : `Try again — make it more like ${target.emoji}`;
  }
}

function drawProgressRing(progress01) {
  if (progress01 <= 0) return;
  const w = overlay.width;
  const h = overlay.height;
  const pct = Math.min(1, progress01);
  ctx.lineWidth = 12;
  ctx.strokeStyle = `rgba(52, 211, 153, ${0.3 + pct * 0.7})`;
  ctx.strokeRect(6, 6, w - 12, h - 12);
}
