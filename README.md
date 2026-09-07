# Emoji Faces 🎭

A browser game for two: an emoji face appears, and you (and your kid) make
that expression at the camera. Detection runs fully on-device using
[MediaPipe FaceLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) —
no video ever leaves the browser.

Expressions: 😊 happy, 😂 laughing, 😢 sad, 😲 surprised, 😠 angry.
It can track two faces at once, so both players need to match the emoji to
score the round.

## Running it

Camera access requires a "secure context" — either `localhost` or HTTPS.
Opening `index.html` directly as a `file://` URL will **not** work.

**On your computer (Chrome/Safari desktop):**

```
npx serve .
```

then open the printed `http://localhost:...` URL.

**On your phone** (recommended, since the game is a phone-camera app):

The easiest path is GitHub Pages:

1. Push this repo to GitHub (already set up).
2. In the repo settings, enable **Pages** → deploy from the branch/folder
   containing `index.html` (root).
3. Open the resulting `https://<you>.github.io/<repo>/` URL on your phone.

Any static HTTPS host works the same way (Netlify, Vercel, Cloudflare
Pages, etc.) — there's no server-side code, just static files.

## How it works

- `index.html` / `css/style.css` — UI shell.
- `js/app.js` — loads the camera, runs `FaceLandmarker` per video frame,
  and turns the returned face "blendshape" coefficients (e.g. `mouthSmileLeft`,
  `jawOpen`, `browDownLeft`) into a simple expression classifier. When every
  face in frame matches the target expression for ~0.7s, the round advances.

Tuning the expression thresholds/weights lives in the `EXPRESSIONS` array
at the top of `js/app.js`.
