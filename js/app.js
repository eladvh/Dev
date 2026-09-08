import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// A page with an "unload" listener is excluded from the browser's
// back-forward cache, so re-opening this tab does a real reload (fetching
// the current deployed index.html/app.js/style.css) instead of resurrecting
// a stale in-memory copy of the page from before the last update.
window.addEventListener("unload", () => {});

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
// Spoken/displayed labels are language-dependent and live in STRINGS below.
const EXPRESSIONS = [
  {
    key: "smile",
    emoji: "😊",
    calc: (b) => avg(b("mouthSmileLeft"), b("mouthSmileRight")) * (1 - b("jawOpen") * 0.7),
  },
  {
    key: "laugh",
    emoji: "😂",
    calc: (b) => avg(b("mouthSmileLeft"), b("mouthSmileRight")) * b("jawOpen"),
  },
  {
    key: "sad",
    emoji: "😢",
    // mouthFrown is the primary, controllable cue; most people don't raise
    // their inner brows on cue, so that's a bonus rather than a requirement
    // (an even split with avg() made this expression very hard to trigger).
    calc: (b) => Math.min(1, avg(b("mouthFrownLeft"), b("mouthFrownRight")) * 1.4 + b("browInnerUp") * 0.25),
  },
  {
    key: "surprised",
    emoji: "😲",
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
    calc: (b) =>
      avg(b("browDownLeft"), b("browDownRight"), b("mouthPressLeft"), b("mouthPressRight")),
  },
  {
    key: "wink",
    emoji: "😉",
    calc: (b) => {
      const l = b("eyeBlinkLeft");
      const r = b("eyeBlinkRight");
      return Math.max(l, r) * (1 - Math.min(l, r));
    },
  },
  {
    key: "kiss",
    emoji: "😘",
    calc: (b) => b("mouthPucker"),
  },
  {
    key: "disgusted",
    emoji: "🤢",
    calc: (b) => avg(b("noseSneerLeft"), b("noseSneerRight")),
  },
  {
    key: "sleepy",
    emoji: "😴",
    calc: (b) => {
      const l = b("eyeBlinkLeft");
      const r = b("eyeBlinkRight");
      return avg(l, r) * (1 - Math.abs(l - r));
    },
  },
];

const STRINGS = {
  en: {
    dir: "ltr",
    speechLang: "en-US",
    title: "Emoji Faces 🎭",
    subtitle:
      "Copy the emoji's face with your own face! Great for two players — get your faces in frame together.",
    startBtn: "Start Camera & Play",
    scoreLabel: "Score:",
    roundLabel: "Round",
    makeThisFace: "Make this face:",
    helpBtn: "🍪 Help",
    skipBtn: "⏭ Skip",
    niceBanner: "Nice! 🎉",
    loading: "Loading face detector…",
    getInFrame: "Get your face(s) in frame!",
    noFace: "Can't see a face — move closer or add more light.",
    matchedSolo: "Matched — hold it!",
    matchedBoth: "Both of you match — hold it!",
    tryAgain: (emoji) => `Try again — make it more like ${emoji}`,
    needBoth: "Both faces need to match the emoji",
    camDenied: "Camera permission was denied. Please allow camera access and try again.",
    startFailed: (msg) => `Couldn't start: ${msg}`,
    doFace: (label) => `Do a ${label} face!`,
    exprLabels: {
      smile: "happy",
      laugh: "laughing",
      sad: "sad",
      surprised: "surprised",
      angry: "angry",
      wink: "winking",
      kiss: "a kissy",
      disgusted: "disgusted",
      sleepy: "sleepy",
    },
  },
  he: {
    dir: "rtl",
    speechLang: "he-IL",
    title: "פרצופי אימוג'י 🎭",
    subtitle: "תחקו את הפרצוף של האימוג'י! משחק נהדר לשניים — תכניסו את שני הפרצופים למסך יחד.",
    startBtn: "התחילו מצלמה ושחקו",
    scoreLabel: "ניקוד:",
    roundLabel: "סיבוב",
    makeThisFace: "עשו את הפרצוף הזה:",
    helpBtn: "🍪 עזרה",
    skipBtn: "⏭ דלגו",
    niceBanner: "יופי! 🎉",
    loading: "טוען זיהוי פנים…",
    getInFrame: "הכניסו את הפנים למסגרת!",
    noFace: "לא רואים פנים — התקרבו או הוסיפו תאורה.",
    matchedSolo: "התאמה — תחזיקו רגע!",
    matchedBoth: "שניכם מתאימים — תחזיקו רגע!",
    tryAgain: (emoji) => `נסו שוב — שיהיה יותר דומה ל־${emoji}`,
    needBoth: "שני הפרצופים צריכים להתאים לאימוג'י",
    camDenied: "הגישה למצלמה נדחתה. אנא אשרו גישה למצלמה ונסו שוב.",
    startFailed: (msg) => `ההפעלה נכשלה: ${msg}`,
    doFace: (label) => `תעשו פרצוף ${label}!`,
    exprLabels: {
      smile: "שמח",
      laugh: "צוחק",
      sad: "עצוב",
      surprised: "מופתע",
      angry: "כועס",
      wink: "קורץ",
      kiss: "נשיקה",
      disgusted: "נגעל",
      sleepy: "ישנוני",
    },
  },
};

// MediaPipe FaceLandmarker's 468-point face mesh: stable landmark indices
// used to anchor the "help" guide to the player's actual eyes/mouth.
const LM = {
  mouthLeft: 61,
  mouthRight: 291,
  eyeALeft: 33,
  eyeAInner: 133,
  eyeBInner: 362,
  eyeBRight: 263,
};

// How to draw the help guide's mouth/eyes/eyebrows for each target
// expression, anchored to the player's own detected face landmarks rather
// than a fixed illustration, so it overlays directly on their eyes/mouth.
const FACE_GUIDE_STYLES = {
  smile: { mouth: "smile", eyes: "open", brows: "normal" },
  laugh: { mouth: "laugh", eyes: "squint", brows: "normal" },
  sad: { mouth: "frown", eyes: "open", brows: "innerUp" },
  surprised: { mouth: "round", eyes: "wide", brows: "raised" },
  angry: { mouth: "flat", eyes: "open", brows: "down" },
  wink: { mouth: "smile", eyes: "wink", brows: "normal" },
  kiss: { mouth: "pucker", eyes: "open", brows: "normal" },
  disgusted: { mouth: "wavy", eyes: "open", brows: "down" },
  sleepy: { mouth: "flat", eyes: "closed", brows: "relaxed" },
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
const helpBtn = document.getElementById("help-btn");
const skipBtn = document.getElementById("skip-btn");
const appTitleEl = document.getElementById("app-title");
const appSubtitleEl = document.getElementById("app-subtitle");
const scoreLabelEl = document.getElementById("score-label");
const roundLabelEl = document.getElementById("round-label");
const makeFaceLabelEl = document.getElementById("make-face-label");
const langEnBtn = document.getElementById("lang-en");
const langHeBtn = document.getElementById("lang-he");

// ---- Game state ----
let score = 0;
let round = 1;
let target = pickTarget();
let matchStartTime = null;
let faceLandmarker = null;
let celebrating = false;
let celebrateTimeoutId = null;
let helpVisible = false;

let lang = localStorage.getItem("emojiFacesLang") || (navigator.language.startsWith("he") ? "he" : "en");

function t() {
  return STRINGS[lang];
}

function applyLanguage(newLang) {
  lang = newLang;
  localStorage.setItem("emojiFacesLang", lang);
  document.documentElement.lang = lang;
  document.documentElement.dir = t().dir;
  langEnBtn.classList.toggle("active", lang === "en");
  langHeBtn.classList.toggle("active", lang === "he");

  appTitleEl.textContent = t().title;
  appSubtitleEl.textContent = t().subtitle;
  startBtn.textContent = t().startBtn;
  scoreLabelEl.textContent = t().scoreLabel;
  roundLabelEl.textContent = t().roundLabel;
  makeFaceLabelEl.textContent = t().makeThisFace;
  helpBtn.textContent = t().helpBtn;
  skipBtn.textContent = t().skipBtn;
  matchBanner.textContent = t().niceBanner;
}

langEnBtn.addEventListener("click", () => applyLanguage("en"));
langHeBtn.addEventListener("click", () => applyLanguage("he"));
applyLanguage(lang);

function pickTarget(excludeKey) {
  const pool = excludeKey ? EXPRESSIONS.filter((e) => e.key !== excludeKey) : EXPRESSIONS;
  return pool[Math.floor(Math.random() * pool.length)];
}

function speak(text) {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = t().speechLang;
  window.speechSynthesis.speak(utterance);
}

function showTarget() {
  targetEmojiEl.textContent = target.emoji;
  targetEmojiEl.classList.remove("pop");
  void targetEmojiEl.offsetWidth;
  targetEmojiEl.classList.add("pop");
  speak(t().doFace(t().exprLabels[target.key]));
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
  statusEl.textContent = t().niceBanner;
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
  helpBtn.classList.toggle("active", helpVisible);
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

    statusEl.textContent = t().loading;
    const filesetResolver = await FilesetResolver.forVisionTasks(WASM_BASE);
    faceLandmarker = await createFaceLandmarker(filesetResolver);
    statusEl.textContent = t().getInFrame;

    requestAnimationFrame(detectLoop);
  } catch (err) {
    console.error(err);
    startError.textContent =
      err && err.name === "NotAllowedError" ? t().camDenied : t().startFailed(err.message || err);
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
    statusEl.textContent = t().noFace;
    matchStartTime = null;
    return;
  }

  const classifications = faces.map((f) => classifyFace(f.categories));
  const allMatch = classifications.every((c) => c.key === target.key);
  const confidence = Math.min(...classifications.map((c) => c.score));

  drawProgressRing(allMatch ? confidence : 0);

  if (helpVisible) {
    for (const landmarks of result.faceLandmarks || []) {
      drawFaceGuide(landmarks, target.key);
    }
  }

  if (allMatch) {
    statusEl.textContent = faces.length > 1 ? t().matchedBoth : t().matchedSolo;
    if (matchStartTime === null) matchStartTime = performance.now();
    if (performance.now() - matchStartTime >= HOLD_MS) {
      matchStartTime = null;
      celebrate();
    }
  } else {
    matchStartTime = null;
    statusEl.textContent = faces.length > 1 ? t().needBoth : t().tryAgain(target.emoji);
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

function pt(landmarks, index) {
  const lm = landmarks[index];
  return { x: lm.x * overlay.width, y: lm.y * overlay.height };
}

function mid(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

// Draws the target expression's mouth/eyes/eyebrows shape anchored to the
// player's own detected face (an AR-style trace guide), rather than a fixed
// illustration, so it lines up with their eyes and mouth in the live video.
function drawFaceGuide(landmarks, key) {
  const style = FACE_GUIDE_STYLES[key];
  if (!style) return;

  const mouthLeft = pt(landmarks, LM.mouthLeft);
  const mouthRight = pt(landmarks, LM.mouthRight);
  const eyeACenter = mid(pt(landmarks, LM.eyeALeft), pt(landmarks, LM.eyeAInner));
  const eyeBCenter = mid(pt(landmarks, LM.eyeBInner), pt(landmarks, LM.eyeBRight));
  const eyeAWidth = dist(pt(landmarks, LM.eyeALeft), pt(landmarks, LM.eyeAInner));
  const eyeBWidth = dist(pt(landmarks, LM.eyeBInner), pt(landmarks, LM.eyeBRight));
  const faceMidX = (eyeACenter.x + eyeBCenter.x) / 2;

  ctx.save();
  ctx.strokeStyle = "#ffb800";
  ctx.fillStyle = "#ffb800";
  ctx.lineWidth = Math.max(3, overlay.width * 0.01);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  drawGuideMouth(style.mouth, mouthLeft, mouthRight);
  // "wink" always closes the same (A-side) eye so it stays consistent frame to frame.
  drawGuideEye(style.eyes === "wink" ? "closed" : style.eyes, eyeACenter, eyeAWidth);
  drawGuideEye(style.eyes === "wink" ? "open" : style.eyes, eyeBCenter, eyeBWidth);
  drawGuideBrow(style.brows, eyeACenter, eyeAWidth, eyeACenter.x < faceMidX);
  drawGuideBrow(style.brows, eyeBCenter, eyeBWidth, eyeBCenter.x < faceMidX);

  ctx.restore();
}

function drawGuideMouth(style, left, right) {
  const cx = (left.x + right.x) / 2;
  const cy = (left.y + right.y) / 2;
  const w = dist(left, right);
  ctx.beginPath();
  switch (style) {
    case "laugh":
      ctx.moveTo(left.x, left.y);
      ctx.quadraticCurveTo(cx, cy + w * 0.55, right.x, right.y);
      break;
    case "frown":
      ctx.moveTo(left.x, left.y);
      ctx.quadraticCurveTo(cx, cy - w * 0.35, right.x, right.y);
      break;
    case "round":
      ctx.ellipse(cx, cy + w * 0.15, w * 0.3, w * 0.32, 0, 0, Math.PI * 2);
      break;
    case "flat":
      ctx.moveTo(left.x, left.y);
      ctx.lineTo(right.x, right.y);
      break;
    case "pucker":
      ctx.ellipse(cx, cy, w * 0.16, w * 0.14, 0, 0, Math.PI * 2);
      break;
    case "wavy":
      ctx.moveTo(left.x, left.y);
      ctx.quadraticCurveTo(cx - w * 0.2, cy - w * 0.2, cx, cy);
      ctx.quadraticCurveTo(cx + w * 0.2, cy + w * 0.25, right.x, right.y);
      break;
    case "smile":
    default:
      ctx.moveTo(left.x, left.y);
      ctx.quadraticCurveTo(cx, cy + w * 0.35, right.x, right.y);
      break;
  }
  ctx.stroke();
}

function drawGuideEye(style, center, width) {
  ctx.beginPath();
  if (style === "closed") {
    ctx.moveTo(center.x - width * 0.5, center.y);
    ctx.lineTo(center.x + width * 0.5, center.y);
  } else {
    const r = (style === "wide" ? 0.5 : style === "squint" ? 0.22 : 0.35) * width;
    ctx.arc(center.x, center.y, r, 0, Math.PI * 2);
  }
  ctx.stroke();
}

function drawGuideBrow(style, center, width, isOnLeftSide) {
  const y = center.y - width * 0.9;
  const half = width * 0.5;
  const innerX = isOnLeftSide ? center.x + half : center.x - half;
  const outerX = isOnLeftSide ? center.x - half : center.x + half;
  ctx.beginPath();
  switch (style) {
    case "raised":
      ctx.moveTo(outerX, y + width * 0.15);
      ctx.quadraticCurveTo(center.x, y - width * 0.3, innerX, y + width * 0.15);
      break;
    case "down":
      ctx.moveTo(outerX, y - width * 0.1);
      ctx.lineTo(innerX, y + width * 0.2);
      break;
    case "innerUp":
      ctx.moveTo(outerX, y + width * 0.15);
      ctx.lineTo(innerX, y - width * 0.15);
      break;
    case "relaxed":
      ctx.moveTo(outerX, y + width * 0.05);
      ctx.lineTo(innerX, y + width * 0.05);
      break;
    case "normal":
    default:
      ctx.moveTo(outerX, y);
      ctx.quadraticCurveTo(center.x, y - width * 0.1, innerX, y);
      break;
  }
  ctx.stroke();
}
