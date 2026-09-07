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
    // mouthFrown is the primary, controllable cue; most people don't raise
    // their inner brows on cue, so that's a bonus rather than a requirement
    // (an even split with avg() made this expression very hard to trigger).
    calc: (b) => Math.min(1, avg(b("mouthFrownLeft"), b("mouthFrownRight")) * 1.4 + b("browInnerUp") * 0.25),
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
  {
    key: "wink",
    emoji: "😉",
    label: "winking",
    calc: (b) => {
      const l = b("eyeBlinkLeft");
      const r = b("eyeBlinkRight");
      return Math.max(l, r) * (1 - Math.min(l, r));
    },
  },
  {
    key: "kiss",
    emoji: "😘",
    label: "a kissy",
    calc: (b) => b("mouthPucker"),
  },
  {
    key: "disgusted",
    emoji: "🤢",
    label: "disgusted",
    calc: (b) => avg(b("noseSneerLeft"), b("noseSneerRight")),
  },
  {
    key: "sleepy",
    emoji: "😴",
    label: "sleepy",
    calc: (b) => {
      const l = b("eyeBlinkLeft");
      const r = b("eyeBlinkRight");
      return avg(l, r) * (1 - Math.abs(l - r));
    },
  },
];

// Simplified cartoon face parts (eyebrows/eyes/mouth) shown as a practice
// guide overlay, one per expression key, drawn on a 200x200 viewBox.
const FACE_GUIDES = {
  smile: `
    <path d="M55,58 Q70,50 85,58" />
    <path d="M115,58 Q130,50 145,58" />
    <circle cx="70" cy="82" r="9" fill="#fff" stroke="none" />
    <circle cx="130" cy="82" r="9" fill="#fff" stroke="none" />
    <path d="M62,132 Q100,162 138,132" />
  `,
  laugh: `
    <path d="M52,55 Q70,44 88,55" />
    <path d="M112,55 Q130,44 148,55" />
    <path d="M58,78 Q70,86 82,78" />
    <path d="M118,78 Q130,86 142,78" />
    <ellipse cx="100" cy="140" rx="34" ry="22" fill="#fff" fill-opacity="0.15" />
  `,
  sad: `
    <path d="M55,62 Q70,52 88,60" />
    <path d="M112,60 Q130,52 145,62" />
    <circle cx="70" cy="85" r="9" fill="#fff" stroke="none" />
    <circle cx="130" cy="85" r="9" fill="#fff" stroke="none" />
    <path d="M65,155 Q100,128 135,155" />
  `,
  surprised: `
    <path d="M52,50 Q70,36 88,48" />
    <path d="M112,48 Q130,36 148,50" />
    <circle cx="70" cy="82" r="15" fill="none" />
    <circle cx="130" cy="82" r="15" fill="none" />
    <circle cx="100" cy="145" r="18" fill="none" />
  `,
  angry: `
    <path d="M55,48 L85,62" />
    <path d="M145,48 L115,62" />
    <circle cx="70" cy="82" r="9" fill="#fff" stroke="none" />
    <circle cx="130" cy="82" r="9" fill="#fff" stroke="none" />
    <path d="M65,148 Q100,138 135,148" />
  `,
  wink: `
    <path d="M55,58 Q70,50 85,58" />
    <path d="M115,58 Q130,50 145,58" />
    <line x1="60" y1="82" x2="80" y2="82" />
    <circle cx="130" cy="82" r="9" fill="#fff" stroke="none" />
    <path d="M65,135 Q100,158 135,135" />
  `,
  kiss: `
    <path d="M55,58 Q70,50 85,58" />
    <path d="M115,58 Q130,50 145,58" />
    <circle cx="70" cy="82" r="9" fill="#fff" stroke="none" />
    <circle cx="130" cy="82" r="9" fill="#fff" stroke="none" />
    <ellipse cx="100" cy="142" rx="13" ry="11" fill="none" />
  `,
  disgusted: `
    <path d="M55,50 L85,64" />
    <path d="M145,50 L115,64" />
    <circle cx="70" cy="84" r="9" fill="#fff" stroke="none" />
    <circle cx="130" cy="84" r="9" fill="#fff" stroke="none" />
    <path d="M92,90 Q100,98 108,90" />
    <path d="M62,148 Q80,132 100,146 Q120,160 138,140" />
  `,
  sleepy: `
    <path d="M58,60 Q70,54 82,60" />
    <path d="M118,60 Q130,54 142,60" />
    <line x1="60" y1="84" x2="80" y2="84" />
    <line x1="120" y1="84" x2="140" y2="84" />
    <path d="M70,144 Q100,152 130,144" />
  `,
};

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
const faceGuide = document.getElementById("face-guide");
const faceGuideFeatures = document.getElementById("face-guide-features");
const helpBtn = document.getElementById("help-btn");
const skipBtn = document.getElementById("skip-btn");

// ---- Game state ----
let score = 0;
let round = 1;
let target = pickTarget();
let matchStartTime = null;
let faceLandmarker = null;
let celebrating = false;
let celebrateTimeoutId = null;
let helpVisible = false;

function pickTarget(excludeKey) {
  const pool = excludeKey ? EXPRESSIONS.filter((e) => e.key !== excludeKey) : EXPRESSIONS;
  return pool[Math.floor(Math.random() * pool.length)];
}

function speak(text) {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}

function updateFaceGuide() {
  faceGuideFeatures.innerHTML = FACE_GUIDES[target.key] || "";
}

function showTarget() {
  targetEmojiEl.textContent = target.emoji;
  targetEmojiEl.classList.remove("pop");
  void targetEmojiEl.offsetWidth;
  targetEmojiEl.classList.add("pop");
  updateFaceGuide();
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
  celebrateTimeoutId = setTimeout(() => {
    score += 1;
    round += 1;
    scoreEl.textContent = String(score);
    roundEl.textContent = String(round);
    celebrating = false;
    celebrateTimeoutId = null;
    nextRound();
  }, CELEBRATE_MS);
}

function skipRound() {
  if (celebrateTimeoutId !== null) {
    clearTimeout(celebrateTimeoutId);
    celebrateTimeoutId = null;
  }
  celebrating = false;
  round += 1;
  roundEl.textContent = String(round);
  nextRound();
}

helpBtn.addEventListener("click", () => {
  helpVisible = !helpVisible;
  faceGuide.hidden = !helpVisible;
  helpBtn.classList.toggle("active", helpVisible);
  if (helpVisible) updateFaceGuide();
});

skipBtn.addEventListener("click", skipRound);

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
