// 판정 설정값
const CONFIG = {
  CAUTION_R: 1.10,
  WARNING_R: 1.20,
  WARNING_HOLD_SEC: 1.5,
  RECOVERY_R: 1.10,
  RECOVERY_HOLD_SEC: 1.0,
  MEDIAN_WINDOW: 5,
  CALIBRATION_SEC: 2.0,
  MIN_CALIBRATION_SAMPLES: 10,
};

const STATE = { SAFE: 'SAFE', CAUTION: 'CAUTION', WARNING: 'WARNING', UNKNOWN: 'UNKNOWN' };

// 거리 계산
class DistanceEngine {
  constructor() {
    this.b0 = null;
    this.recentR = [];
    this.warnTimer = 0;
    this.recoverTimer = 0;
    this.persistentWarning = false;
    this.justRewarded = false;
  }

  calibrate(areas) {

    if (areas.length < CONFIG.MIN_CALIBRATION_SAMPLES) return false;
    this.b0 = median(areas);
    this.resetRuntimeState();
    return true;
  }

  resetRuntimeState() {
    this.recentR = [];
    this.warnTimer = 0;
    this.recoverTimer = 0;
    this.persistentWarning = false;
    this.justRewarded = false;
  }

  update(bt, dt) {
    this.justRewarded = false;

    if (this.b0 == null || bt == null || bt <= 0) {

      this.recentR = [];
      this.warnTimer = 0;
      this.recoverTimer = 0;
      return STATE.UNKNOWN;
    }

    const rRaw = Math.sqrt(bt / this.b0);
    this.recentR.push(rRaw);
    if (this.recentR.length > CONFIG.MEDIAN_WINDOW) this.recentR.shift();
    const r = median(this.recentR);

    if (this.persistentWarning) {
      if (r <= CONFIG.RECOVERY_R) this.recoverTimer += dt;
      else this.recoverTimer = 0;

      if (this.recoverTimer >= CONFIG.RECOVERY_HOLD_SEC) {
        this.persistentWarning = false;
        this.recoverTimer = 0;
        this.warnTimer = 0;
        this.justRewarded = true;
        return STATE.SAFE;
      }
      return STATE.WARNING;
    }

    if (r >= CONFIG.WARNING_R) {
      this.warnTimer += dt;
      if (this.warnTimer >= CONFIG.WARNING_HOLD_SEC) {
        this.persistentWarning = true;
        this.warnTimer = 0;
        this.recoverTimer = 0;
        return STATE.WARNING;
      }
      return STATE.CAUTION;
    } else {
      this.warnTimer = 0;
      if (r >= CONFIG.CAUTION_R) return STATE.CAUTION;
      return STATE.SAFE;
    }
  }

  get lastR() {
    return this.recentR.length ? median(this.recentR) : null;
  }
}

function median(arr) {
  const a = [...arr].sort((x, y) => x - y);
  const mid = Math.floor(a.length / 2);
  return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
}

function virtualRFromDistance(distanceCm, calibDistanceCm = 30.0) {
  return calibDistanceCm / distanceCm;
}

// 전역 상태
const engine = new DistanceEngine();
let cvReady = false;
let classifier = null;
let stream = null;
let cameraSession = 0;
let videoUrl;
let rewardTimer;
let detectLoopId = null;
let lastFrameTime = null;
let calibrating = false;
let calibAreas = [];
let calibStart = 0;
let mode = 'idle'; // 'camera' | 'slider' | 'idle'

let src, gray, matVec; // OpenCV Mat, 해제 필요

const els = {};
function $(id) { return document.getElementById(id); }

document.addEventListener('DOMContentLoaded', () => {
  ['video', 'videoFile', 'cam', 'overlayCanvas', 'camCanvas', 'btnCamOn', 'btnCamOff',
   'btnCalibrate', 'btnOpenVideo', 'btnSlider', 'statusText', 'statusDot', 'statusBadge', 'b0Text',
   'btText', 'rText', 'slider', 'sliderWrap', 'errorBox', 'pikaChar', 'pikaMsg', 'calibMsg']
    .forEach(id => (els[id] = $(id)));

  els.btnCamOn.addEventListener('click', startCamera);
  els.btnCamOff.addEventListener('click', stopCamera);
  els.btnCalibrate.addEventListener('click', startCalibration);
  els.btnOpenVideo.addEventListener('click', () => els.videoFile.click());
  els.videoFile.addEventListener('change', onVideoSelected);
  els.btnSlider.addEventListener('click', toggleSliderMode);
  els.slider.addEventListener('input', onSliderInput);

  window.addEventListener('pagehide', cleanup);

  $('pick').onclick = () => els.videoFile.click();
});

// OpenCV.js 로딩
let cvLoading;
function loadOpenCv() {
  if (cvLoading) return cvLoading;
  cvLoading = (async () => {
    if (!window.cv) await new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://docs.opencv.org/4.9.0/opencv.js';
      script.onload = resolve;
      script.onerror = () => { script.remove(); reject(Error('OpenCV 로딩 실패')); };
      document.head.append(script);
    });
    const runtime = await window.cv;
    window.cv = runtime;
    if (!runtime.Mat) await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(Error('OpenCV 초기화 지연')), 20000);
      runtime.onRuntimeInitialized = () => { clearTimeout(timeout); resolve(); };
    });
    const response = await fetch('https://raw.githubusercontent.com/opencv/opencv/4.9.0/data/haarcascades/haarcascade_frontalface_default.xml');
    if (!response.ok) throw Error('얼굴 모델 로딩 실패');
    const model = new Uint8Array(await response.arrayBuffer());
    const name = 'haarcascade.xml';
    try { cv.FS_unlink('/' + name); } catch {}
    cv.FS_createDataFile('/', name, model, true, false, false);
    const next = new cv.CascadeClassifier();
    if (!next.load(name)) { next.delete(); throw Error('얼굴 모델 초기화 실패'); }
    classifier = next;
    cvReady = true;
  })().catch(error => { cvLoading = null; throw error; });
  return cvLoading;
}

// 카메라 제어
async function startCamera() {
  showError(null);
  if (!window.isSecureContext) {
    showError('https:// 로 시작하는 주소로 접속해주세요.');
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError('이 브라우저는 카메라 접근 기능을 지원하지 않습니다. 최신 Safari 혹은 Chrome을 사용해주세요.');
    return;
  }
  stopCamera();
  els.btnCamOn.disabled = true;
  const session = ++cameraSession;
  try {
    const acquired = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 480 }, height: { ideal: 640 } },
      audio: false,
    });
    if (session !== cameraSession) { acquired.getTracks().forEach(t => t.stop()); return; }
    stream = acquired;
    els.cam.srcObject = stream;
    await els.cam.play();
    setStatus(STATE.UNKNOWN, '얼굴 모델 로딩 중');
    await loadOpenCv();
    if (session !== cameraSession) return;
    stream.getVideoTracks().forEach(track => track.addEventListener('ended', stopCamera, {once:true}));
  } catch (e) {
    if (session !== cameraSession) return;
    stopCamera();
    showError(e.name === 'NotAllowedError' ? '카메라 권한 확인' : e.message);
    return;
  }
  mode = 'camera';
  els.sliderWrap.style.display = 'none';
  $('camRow').classList.add('active');
  resizeCanvas();
  startDetectLoop();
  setStatus(STATE.UNKNOWN, '카메라 작동 중. 30cm 보정을 눌러주세요.');
}

function stopCamera() {
  cameraSession++;
  els.btnCamOn.disabled = false;
  calibrating = false;
  engine.b0 = null;
  engine.resetRuntimeState();
  clearTimeout(rewardTimer);
  delete els.pikaChar.dataset.rewarding;
  els.sliderWrap.style.display = 'none';
  for (const id of ['b0Text', 'btText', 'rText']) els[id].textContent = '–';
  els.calibMsg.textContent = '';
  stopDetectLoop();
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  els.cam.srcObject = null;
  $('camRow').classList.remove('active');
  freeMats();
  mode = 'idle';
  setStatus(STATE.UNKNOWN, '카메라가 꺼져 있습니다.');
  hideCharacter();
}

function resizeCanvas() {
  const scale = Math.min(1, 320 / Math.max(els.cam.videoWidth, els.cam.videoHeight));
  els.camCanvas.width = Math.round(els.cam.videoWidth * scale);
  els.camCanvas.height = Math.round(els.cam.videoHeight * scale);
}

// 검출 루프
function startDetectLoop() {
  lastFrameTime = performance.now();
  const loop = () => {
    const now = performance.now();
    const gap = now - lastFrameTime;
    if (gap < 95) { detectLoopId = requestAnimationFrame(loop); return; }
    if (gap > 500) engine.update(null, 0);
    const dt = gap > 500 ? 0 : gap / 1000; // dt 상한
    lastFrameTime = now;

    let faceArea;
    try { faceArea = detectFaceArea(); }
    catch (error) { stopCamera(); showError('얼굴 검출 실패: ' + error.message); return; }
    els.btText.textContent = faceArea ?? '–';

    if (calibrating) {
      handleCalibrationFrame(faceArea);
    } else if (mode === 'camera') {
      const state = engine.update(faceArea, dt);
      applyState(state);
    }
    detectLoopId = requestAnimationFrame(loop);
  };
  detectLoopId = requestAnimationFrame(loop);
}

function stopDetectLoop() {
  if (detectLoopId) cancelAnimationFrame(detectLoopId);
  detectLoopId = null;
}


function detectFaceArea() {
  if (!cvReady || !classifier || !els.cam.videoWidth) return null;

  const ctx = els.camCanvas.getContext('2d');
  ctx.save();
  ctx.translate(els.camCanvas.width, 0);
  ctx.scale(-1, 1); // 좌우 반전(셀피 미러링)
  ctx.drawImage(els.cam, 0, 0, els.camCanvas.width, els.camCanvas.height);
  ctx.restore();

  if (!src) {
    src = new cv.Mat(els.camCanvas.height, els.camCanvas.width, cv.CV_8UC4);
    gray = new cv.Mat();
    matVec = new cv.RectVector();
  }
  const imgData = ctx.getImageData(0, 0, els.camCanvas.width, els.camCanvas.height);
  src.data.set(imgData.data);
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  cv.equalizeHist(gray, gray); // 저조도 보정 처리

  matVec.delete();
  matVec = new cv.RectVector();
  const minSize = new cv.Size(Math.floor(gray.cols * 0.15), Math.floor(gray.rows * 0.15));
  classifier.detectMultiScale(gray, matVec, 1.1, 5, 0, minSize);


  const overlayCtx = els.overlayCanvas.getContext('2d');
  els.overlayCanvas.width = els.camCanvas.width;
  els.overlayCanvas.height = els.camCanvas.height;
  overlayCtx.clearRect(0, 0, overlayCtx.canvas.width, overlayCtx.canvas.height);

  const n = matVec.size();
  if (n === 0) return null; // 얼굴 미검출
  if (n > 1) {
    for (let i = 0; i < n; i++) drawBox(overlayCtx, matVec.get(i), '#999');
    return null; // 다수 얼굴, UNKNOWN 처리
  }

  const rect = matVec.get(0);
  const margin = 4;
  const clipped =
    rect.x <= margin || rect.y <= margin ||
    rect.x + rect.width >= gray.cols - margin ||
    rect.y + rect.height >= gray.rows - margin;

  drawBox(overlayCtx, rect, clipped ? '#999' : '#19c37d');
  if (clipped) return null; // 가장자리 잘림, UNKNOWN 처리

  return rect.width * rect.height;
}

function drawBox(ctx, rect, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
}

// 30cm 보정
function startCalibration() {
  if (mode !== 'camera') {
    showError('먼저 카메라를 켜고 화면에서 약 30cm 떨어져 정면을 봐주세요.');
    return;
  }
  engine.b0 = null;
  engine.resetRuntimeState();
  els.b0Text.textContent = els.rText.textContent = '–';
  setStatus(STATE.UNKNOWN, '보정 중');
  calibrating = true;
  calibAreas = [];
  calibStart = performance.now();
  els.calibMsg.textContent = '보정 중... 30cm 거리에서 정면을 유지해주세요.';
}

function handleCalibrationFrame(area) {
  if (area != null) calibAreas.push(area);
  const elapsed = (performance.now() - calibStart) / 1000;
  if (elapsed >= CONFIG.CALIBRATION_SEC) {
    calibrating = false;
    const ok = engine.calibrate(calibAreas);
    if (ok) {
      els.calibMsg.textContent = `보정 완료! 기준 면적 B0 = ${engine.b0.toFixed(0)}`;
      setStatus(STATE.SAFE, '보정 완료. 영상을 재생해보세요.');
    } else {
      els.calibMsg.textContent =
        `보정 실패: 유효한 정면 얼굴 샘플이 부족합니다 (수집 ${calibAreas.length}개, 필요 ${CONFIG.MIN_CALIBRATION_SAMPLES}개). ` +
        '조명을 밝게 하고 정면을 바라본 채 다시 시도해주세요.';
    }
  }
}

// 영상 재생
function onVideoSelected(e) {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoUrl = URL.createObjectURL(file);
  els.video.src = videoUrl;
  $('pick').hidden = true;
  els.video.style.display = 'block';
  els.video.play().catch(() => {});
}

// 슬라이더(가상 거리) 모드, 카메라 없이 동일 엔진 체험
function toggleSliderMode() {
  if (mode === 'slider') {
    mode = 'idle';
    els.sliderWrap.style.display = 'none';
    stopDetectLoop();
    hideCharacter();
    setStatus(STATE.UNKNOWN, '카메라/슬라이더가 꺼져 있습니다.');
    return;
  }
  stopCamera();
  mode = 'slider';
  els.sliderWrap.style.display = 'block';

  engine.b0 = 10000;
  engine.resetRuntimeState();
  els.b0Text.textContent = engine.b0;
  lastFrameTime = performance.now();
  const loop = () => {
    const now = performance.now();
    const gap = now - lastFrameTime;
    if (gap < 95) { detectLoopId = requestAnimationFrame(loop); return; }
    if (gap > 500) engine.update(null, 0);
    const dt = gap > 500 ? 0 : gap / 1000;
    lastFrameTime = now;
    const distCm = Number(els.slider.value);
    const r = virtualRFromDistance(distCm, 30.0);
    const bt = engine.b0 * r * r;
    const state = engine.update(bt, dt);
    els.btText.textContent = bt.toFixed(0);
    applyState(state);
    detectLoopId = requestAnimationFrame(loop);
  };
  detectLoopId = requestAnimationFrame(loop);
  setStatus(STATE.SAFE, '슬라이더로 가상 거리를 체험해보세요.');
}

function onSliderInput() {

  $('sliderVal').textContent = els.slider.value + 'cm';
}

// 화면 반영
function applyState(state) {
  const r = engine.lastR;
  els.rText.textContent = r != null ? r.toFixed(3) : '-';
  if (engine.b0 != null) els.b0Text.textContent = engine.b0.toFixed(0);

  setStatus(state, stateMessage(state));

  if (state === STATE.WARNING) {
    showCharacter('closed', '조금 뒤로 물러나 볼까?');
  } else if (engine.justRewarded) {
    showCharacter('open', '까꿍! 잘했어요!');

  } else if (state !== STATE.WARNING) {

    if (!els.pikaChar.dataset.rewarding) hideCharacter();
  }
}

function showCharacter(kind, msg) {
  els.pikaChar.src = kind === 'closed'
  ? './assets/pika_closed.png'
  : './assets/pika_open.png';
  clearTimeout(rewardTimer);
  delete els.pikaChar.dataset.rewarding;
  els.pikaChar.style.display = 'block';
  els.pikaMsg.textContent = msg;
  els.pikaMsg.style.display = 'block';
  if (kind === 'open') {
    els.pikaChar.dataset.rewarding = '1';
    rewardTimer = setTimeout(() => { delete els.pikaChar.dataset.rewarding; hideCharacter(); }, 1500);
  }
}

function hideCharacter() {
  if (els.pikaChar.dataset.rewarding) return; // 보상 연출 중 유지
  els.pikaChar.style.display = 'none';
  els.pikaMsg.style.display = 'none';
}

function stateMessage(state) {
  switch (state) {
    case STATE.SAFE: return '적당한 거리예요.';
    case STATE.CAUTION: return '조금 가까워지고 있어요.';
    case STATE.WARNING: return '너무 가까워요!';
    default: return '측정 불가 — 얼굴을 화면 중앙에 두고 정면을 봐주세요.';
  }
}

function setStatus(state, msg) {
  els.statusText.textContent = msg;
  els.statusBadge.className = 'statusBadge ' + state.toLowerCase();
}

function showError(msg) {
  if (!msg) { els.errorBox.style.display = 'none'; return; }
  els.errorBox.textContent = msg;
  els.errorBox.style.display = 'block';
}

// 정리
function freeMats() {
  if (src) { src.delete(); src = null; }
  if (gray) { gray.delete(); gray = null; }
  if (matVec) { matVec.delete(); matVec = null; }
}

function cleanup() {
  stopCamera();
  classifier?.delete();
  classifier = null;
  cvReady = false;
  cvLoading = null;
}
