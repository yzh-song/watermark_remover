"""
Flask API Server - Watermark Remover v12.0
Three-stage pipeline, strict error handling, preview mask endpoint.
v12.0: Video support in preview_mask, detailed health check, SSIM scene detection.
"""
import os
import sys
import time
import json
import logging
import logging.handlers
import threading
import webbrowser
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple

# ============================================================
# Logging config (append mode + RotatingFileHandler)
# ============================================================
LOG_DIR = Path(r"D:\AI\watermark_remover\logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

API_LOG_PATH = LOG_DIR / 'api_server.log'
ENGINE_LOG_PATH = LOG_DIR / 'engine.log'

# Flush existing file handler to ensure clean start
# Use RotatingFileHandler to prevent unlimited file growth
api_file_handler = logging.handlers.RotatingFileHandler(
    str(API_LOG_PATH), mode='a', maxBytes=10 * 1024 * 1024, backupCount=5,
    encoding='utf-8', delay=False
)
api_file_handler.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
))
api_file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
console_handler.setLevel(logging.INFO)

root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(api_file_handler)
root_logger.addHandler(console_handler)

# Dedicated engine log file handler
engine_file_handler = logging.handlers.RotatingFileHandler(
    str(ENGINE_LOG_PATH), mode='a', maxBytes=10 * 1024 * 1024, backupCount=5,
    encoding='utf-8', delay=False
)
engine_file_handler.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
))
engine_file_handler.setLevel(logging.DEBUG)
engine_logger = logging.getLogger('engine')
engine_logger.addHandler(engine_file_handler)
engine_logger.propagate = True  # Also propagate to root handlers

logger = logging.getLogger(__name__)
logger.info(f"=== API Server v12.0 started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

# ============================================================
# Module-level imports
# ============================================================
import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template_string

try:
    from flask_cors import CORS
    _HAS_CORS = True
except ImportError:
    _HAS_CORS = False
    logger.warning("flask_cors not installed. Run: pip install flask-cors")
    def CORS(app, **kwargs): pass

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(r"D:\AI\watermark_remover\models\sam")))  # SAM 2.1
from core.engine import InpaintingEngine, get_engine
from core.detector import WatermarkDetector, WatermarkNotFoundError

# Directories
PROJECT_ROOT = Path(r"D:\AI\watermark_remover")
OUTPUT_DIR = Path(r"D:\AI\watermark_remover\output")
UPLOAD_DIR = Path(r"D:\AI\watermark_remover\uploads")
CACHE_DIR = Path(r"D:\AI\watermark_remover\cache")
LOGO_DIR = Path(r"D:\AI\watermark_remover\WatermarkDataset\logos")
DATASET_DIR = Path(r"D:\AI\watermark_remover\WatermarkDataset")
TRAIN_PYTHON = r"C:\Users\qsong\AppData\Roaming\aipy-pro\venv\Scripts\python.exe"
TRAIN_LOG_PATH = LOG_DIR / 'train.log'
for d in [OUTPUT_DIR, UPLOAD_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app)

engine: Optional[InpaintingEngine] = None
detector: Optional[WatermarkDetector] = None

# ============================================================
# HTML Frontend v12.0
# ============================================================
UI_HTML = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Watermark Remover v12.0</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f0f1a;color:#e0e0e0;min-height:100vh}
.container{max-width:980px;margin:0 auto;padding:20px}
h1{text-align:center;font-size:2em;margin:20px 0;background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.upload-area{border:2px dashed #444;border-radius:16px;padding:40px;text-align:center;cursor:pointer;transition:all .3s;margin:20px 0;background:#1a1a2e}
.upload-area:hover,.upload-area.drag-over{border-color:#667eea;background:#1e1e3a}
.upload-area input{display:none}
.btn{display:inline-block;padding:12px 28px;border:none;border-radius:8px;font-size:1em;cursor:pointer;transition:all .3s}
.btn-primary{background:linear-gradient(135deg,#667eea,#764ba2);color:white}
.btn-primary:hover{transform:translateY(-1px);box-shadow:0 4px 15px rgba(102,126,234,0.4)}
.btn-primary:disabled{opacity:0.5;cursor:not-allowed;transform:none}
.btn-secondary{background:#2a2a3e;color:#e0e0e0;padding:8px 16px;font-size:.9em}
.btn-secondary:hover{background:#3a3a5e}
.btn-danger{background:#c0392b;color:white;padding:6px 12px;font-size:.8em}
.btn-danger:hover{background:#e74c3c}
.btn-small{background:#2a2a3e;color:#e0e0e0;padding:4px 10px;font-size:.8em;border-radius:4px;cursor:pointer;border:none}
.btn-small:hover{background:#3a3a5e}
.preview-area{display:flex;gap:20px;margin:20px 0;flex-wrap:wrap}
.preview-card{flex:1;min-width:280px;background:#1a1a2e;border-radius:12px;padding:15px}
.preview-card h3{margin-bottom:10px;color:#667eea}
.preview-card img,.preview-card video{width:100%;border-radius:8px;max-height:400px;object-fit:contain;background:#000}
.progress-bar{height:6px;background:#2a2a3e;border-radius:3px;margin:10px 0;overflow:hidden;display:none}
.progress-bar .fill{height:100%;background:linear-gradient(90deg,#667eea,#764ba2);width:0%;transition:width .3s;border-radius:3px}
.status{text-align:center;margin:10px 0;color:#888}
.status.error{color:#ff6b6b}
.status.success{color:#51cf66}
.options{background:#1a1a2e;border-radius:12px;padding:15px;margin:15px 0}
.mode-tabs{display:flex;gap:10px;margin:15px 0}
.mode-tab{flex:1;padding:10px;text-align:center;background:#1a1a2e;border-radius:8px;cursor:pointer;border:2px solid transparent;transition:all .3s}
.mode-tab.active{border-color:#667eea;background:#1e1e3a}
.mode-tab:hover{background:#1e1e3a}
.controls{display:flex;gap:10px;justify-content:center;margin:15px 0;flex-wrap:wrap}
.result-actions{display:flex;gap:10px;justify-content:center;margin-top:10px;flex-wrap:wrap}
.canvas-container{position:relative;margin:15px 0;display:none;background:#000;border-radius:8px;overflow:hidden}
.canvas-container canvas{display:block;cursor:crosshair;max-width:100%;height:auto}
.draw-hint{text-align:center;padding:8px;color:#888;font-size:.9em;background:#1a1a2e;border-radius:0 0 8px 8px}
.video-controls{display:flex;gap:8px;align-items:center;justify-content:center;padding:10px;background:#1a1a2e;border-radius:8px;margin:8px 0;flex-wrap:wrap}
.video-controls input[type="range"]{flex:1;min-width:100px;max-width:300px}
.roi-list{margin:10px 0}
.roi-item{display:flex;align-items:center;gap:8px;background:#2a2a3e;padding:6px 12px;margin:4px 0;border-radius:6px;font-size:.85em}
.roi-item span{flex:1}
.roi-color{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px}
.train-dialog-overlay{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);z-index:999}
.train-dialog{display:none;position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#1e1e3a;padding:30px;border-radius:12px;z-index:1000;width:90%;max-width:500px;box-shadow:0 0 30px rgba(0,0,0,0.5)}
.train-dialog h3{color:#667eea;margin-bottom:20px}
.train-dialog label{display:block;margin:10px 0 4px;font-size:.9em;color:#aaa}
.train-dialog input,.train-dialog select,.train-dialog textarea{width:100%;padding:8px;margin:0 0 10px;background:#2a2a3e;color:#e0e0e0;border:1px solid #444;border-radius:4px;font-size:.9em}
.train-dialog .train-btns{display:flex;gap:10px;justify-content:flex-end;margin-top:20px}
.train-log{background:#0f0f1a;color:#51cf66;padding:10px;max-height:200px;overflow-y:auto;font-size:.75em;margin-top:15px;border-radius:6px;display:none;white-space:pre-wrap;font-family:Consolas,monospace}
.error-modal-overlay{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.6);z-index:2000;align-items:center;justify-content:center}
.error-modal-overlay.show{display:flex}
.error-modal{background:#1e1e3a;border-radius:16px;padding:30px;width:90%;max-width:480px;text-align:center;box-shadow:0 0 40px rgba(0,0,0,0.6);border:1px solid #c0392b}
.error-modal .error-icon{font-size:3em;margin-bottom:10px}
.error-modal h3{color:#ff6b6b;margin-bottom:12px;font-size:1.2em}
.error-modal .error-msg{color:#e0e0e0;margin-bottom:8px;font-size:.95em;line-height:1.5}
.error-modal .error-hint{color:#888;font-size:.85em;margin-bottom:20px;line-height:1.4}
.error-modal .error-btns{display:flex;gap:10px;justify-content:center;flex-wrap:wrap}
.error-modal .error-btns button{padding:10px 24px;border:none;border-radius:8px;font-size:.95em;cursor:pointer;transition:all .3s}
.error-modal .error-btns .btn-close{background:#c0392b;color:white}
.error-modal .error-btns .btn-close:hover{background:#e74c3c}
.error-modal .error-btns .btn-switch{background:#667eea;color:white}
.error-modal .error-btns .btn-switch:hover{background:#764ba2}
</style>
</head>
<body>
<div class="container">
<h1>AI Watermark Remover v12.0</h1>
<div id="status" class="status">AI engine ready - Upload image or video</div>

<div class="mode-tabs">
<div class="mode-tab active" data-mode="auto">Auto Detect</div>
<div class="mode-tab" data-mode="manual">Manual Select</div>
</div>

<div class="upload-area" id="uploadArea">
<p style="font-size:1.5em;margin-bottom:10px">Click or drag to upload</p>
<p>Image: JPG, PNG | Video: MP4, MOV, AVI</p>
<input type="file" id="fileInput" accept="image/*,video/*">
</div>

<div class="video-controls" id="videoControls" style="display:none">
<span>Frame:</span>
<input type="range" id="frameSlider" min="0" max="100" value="0">
<span id="frameTime" style="font-size:.8em;color:#888">0.0s</span>
<button class="btn btn-secondary" id="captureFrameBtn">Capture Frame</button>
</div>

<div class="canvas-container" id="canvasContainer">
<canvas id="roiCanvas"></canvas>
<div class="draw-hint" id="drawHint">Drag mouse to select watermark area. Click "Add Region" to save.</div>
</div>

<div class="options" id="manualOptions" style="display:none">
<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap">
<button class="btn btn-small" id="addRoiBtn" style="background:#667eea;color:white">+ Add Region</button>
<button class="btn btn-small" id="clearAllRoiBtn">Clear All</button>
<button class="btn btn-small" id="previewMaskBtn">Preview Mask</button>
<span style="font-size:.8em;color:#888">Draw a rectangle on the canvas, then click "Add Region"</span>
</div>
<div class="roi-list" id="roiList"></div>
</div>

<div class="progress-bar" id="progressBar"><div class="fill" id="progressFill"></div></div>

<div class="preview-area" id="previewArea" style="display:none">
<div class="preview-card">
<h3>Original</h3>
<img id="originalImagePreview" alt="Original" style="display:none">
<video id="originalVideoPreview" controls style="display:none" preload="metadata"></video>
</div>
<div class="preview-card">
<h3>Result</h3>
<img id="resultImagePreview" alt="Result" style="display:none">
<video id="resultVideoPreview" controls style="display:none" preload="metadata"></video>
<div class="result-actions">
<button class="btn btn-primary" id="downloadBtn" style="display:none">Download</button>
</div>
</div>
</div>

<div class="controls">
<button class="btn btn-primary" id="processBtn" disabled>Process</button>
<button class="btn btn-secondary" id="trainModelBtn" style="margin-left:10px">Train Model</button>
</div>
</div>

<!-- Training Dialog -->
<div class="train-dialog-overlay" id="trainOverlay"></div>
<div class="train-dialog" id="trainDialog">
<h3>Train Watermark Detection Model</h3>
<label>Watermark Type:</label>
<select id="trainType">
<option value="text">Text Watermark</option>
<option value="logo">LOGO Watermark</option>
<option value="both">Text + LOGO</option>
</select>
<div id="textInputGroup">
<label>Watermark Text:</label>
<input type="text" id="trainText" value="AI Generated">
</div>
<div id="logoInputGroup" style="display:none">
<label>Logo Directory (contains PNG files):</label>
<input type="text" id="logoDir" value="D:/AI/watermark_remover/WatermarkDataset/logos">
<p style="color:#888;font-size:.75em;margin:0 0 10px">Ensure the directory contains PNG logo images</p>
</div>
<label>Training Epochs:</label>
<input type="number" id="trainEpochs" value="100" min="10" max="500">
<label>Number of Synthetic Images:</label>
<input type="number" id="trainNum" value="2000" min="100" max="10000">
<div class="train-btns">
<button class="btn btn-secondary" id="cancelTrainBtn">Cancel</button>
<button class="btn btn-primary" id="startTrainBtn">Start Training</button>
</div>
<div class="train-log" id="trainLog"></div>
</div>

<!-- Error Popup Modal -->
<div class="error-modal-overlay" id="errorOverlay">
<div class="error-modal">
<div class="error-icon">&#9888;</div>
<h3 id="errorTitle">Processing Failed</h3>
<div class="error-msg" id="errorMsg"></div>
<div class="error-hint" id="errorHint"></div>
<div class="error-btns">
<button class="btn-close" id="errorCloseBtn">Close</button>
<button class="btn-switch" id="errorSwitchBtn" style="display:none">Switch to Manual Mode</button>
</div>
</div>
</div>

<script>
var currentMode='auto',currentFile=null,isVideo=false,resultFilename=null;
var isDrawing=false,drawStart={x:0,y:0},drawEnd={x:0,y:0};
var displayScale=1.0,originalImage=null,videoDuration=0;
var rois=[];
var COLORS=['#667eea','#e74c3c','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22','#3498db'];

var uploadArea=document.getElementById('uploadArea');
var fileInput=document.getElementById('fileInput');
var processBtn=document.getElementById('processBtn');
var statusDiv=document.getElementById('status');
var progressBar=document.getElementById('progressBar');
var progressFill=document.getElementById('progressFill');
var previewArea=document.getElementById('previewArea');
var originalImagePreview=document.getElementById('originalImagePreview');
var originalVideoPreview=document.getElementById('originalVideoPreview');
var resultImagePreview=document.getElementById('resultImagePreview');
var resultVideoPreview=document.getElementById('resultVideoPreview');
var downloadBtn=document.getElementById('downloadBtn');
var roiCanvas=document.getElementById('roiCanvas');
var canvasContainer=document.getElementById('canvasContainer');
var drawHint=document.getElementById('drawHint');
var manualOptions=document.getElementById('manualOptions');
var addRoiBtn=document.getElementById('addRoiBtn');
var clearAllRoiBtn=document.getElementById('clearAllRoiBtn');
var previewMaskBtn=document.getElementById('previewMaskBtn');
var roiList=document.getElementById('roiList');
var videoControls=document.getElementById('videoControls');
var frameSlider=document.getElementById('frameSlider');
var frameTime=document.getElementById('frameTime');
var captureFrameBtn=document.getElementById('captureFrameBtn');
var trainModelBtn=document.getElementById('trainModelBtn');
var trainDialog=document.getElementById('trainDialog');
var trainOverlay=document.getElementById('trainOverlay');
var trainType=document.getElementById('trainType');
var trainText=document.getElementById('trainText');
var logoDir=document.getElementById('logoDir');
var trainEpochs=document.getElementById('trainEpochs');
var trainNum=document.getElementById('trainNum');
var cancelTrainBtn=document.getElementById('cancelTrainBtn');
var startTrainBtn=document.getElementById('startTrainBtn');
var trainLog=document.getElementById('trainLog');
var textInputGroup=document.getElementById('textInputGroup');
var logoInputGroup=document.getElementById('logoInputGroup');

var errorOverlay=document.getElementById('errorOverlay');
var errorTitle=document.getElementById('errorTitle');
var errorMsg=document.getElementById('errorMsg');
var errorHint=document.getElementById('errorHint');
var errorCloseBtn=document.getElementById('errorCloseBtn');
var errorSwitchBtn=document.getElementById('errorSwitchBtn');

function isVideoFile(f){return f&&f.type.startsWith('video/')}
function isImageFile(f){return f&&f.type.startsWith('image/')}

function showError(title, msg, hint, showSwitch){
errorTitle.textContent=title||'Processing Failed';
errorMsg.textContent=msg||'An unknown error occurred.';
errorHint.textContent=hint||'';
errorSwitchBtn.style.display=showSwitch?'inline-block':'none';
errorOverlay.classList.add('show');
}
errorCloseBtn.addEventListener('click',function(){errorOverlay.classList.remove('show')});
errorOverlay.addEventListener('click',function(e){if(e.target===errorOverlay)errorOverlay.classList.remove('show')});
errorSwitchBtn.addEventListener('click',function(){
errorOverlay.classList.remove('show');
document.querySelectorAll('.mode-tab').forEach(function(t){t.classList.remove('active')});
document.querySelector('.mode-tab[data-mode="manual"]').classList.add('active');
currentMode='manual';
updateUI();
statusDiv.textContent='Switched to Manual Selection mode. Select watermark regions and click Process.';
statusDiv.className='status';
});

document.querySelectorAll('.mode-tab').forEach(function(tab){
tab.addEventListener('click',function(){
document.querySelectorAll('.mode-tab').forEach(function(t){t.classList.remove('active')});
tab.classList.add('active');
currentMode=tab.dataset.mode;
updateUI();
});
});

function updateUI(){
if(currentMode==='manual'){
manualOptions.style.display='block';
if(isVideo){
videoControls.style.display='flex';
canvasContainer.style.display='none';
drawHint.textContent='Drag video slider, then click "Capture Frame" to select watermark';
}else if(isImageFile(currentFile)){
videoControls.style.display='none';
setupCanvas();
}
}else{
manualOptions.style.display='none';
canvasContainer.style.display='none';
videoControls.style.display='none';
}
}

uploadArea.addEventListener('click',function(){fileInput.click()});
fileInput.addEventListener('change',function(e){if(e.target.files.length)handleFile(e.target.files[0])});

uploadArea.addEventListener('dragover',function(e){e.preventDefault();uploadArea.classList.add('drag-over')});
uploadArea.addEventListener('dragleave',function(){uploadArea.classList.remove('drag-over')});
uploadArea.addEventListener('drop',function(e){
e.preventDefault();
uploadArea.classList.remove('drag-over');
if(e.dataTransfer.files.length)handleFile(e.dataTransfer.files[0]);
});

function handleFile(file){
currentFile=file;
isVideo=isVideoFile(file);
processBtn.disabled=false;
var url=URL.createObjectURL(file);

previewArea.style.display='flex';
originalImagePreview.style.display='none';
originalVideoPreview.style.display='none';
resultImagePreview.style.display='none';
resultVideoPreview.style.display='none';
downloadBtn.style.display='none';
originalImage=null;
rois=[];
renderRoiList();

if(isVideo){
originalVideoPreview.src=url;
originalVideoPreview.style.display='block';
originalVideoPreview.onloadedmetadata=function(){
videoDuration=originalVideoPreview.duration||0;
frameSlider.max=Math.floor(videoDuration*10);
frameSlider.value=0;
frameTime.textContent='0.0s';
};
}else{
originalImagePreview.src=url;
originalImagePreview.style.display='block';
var img=new Image();
img.onload=function(){
originalImage=img;
if(currentMode==='manual') setupCanvas();
};
img.src=url;
}

statusDiv.textContent='Selected: '+file.name;
statusDiv.className='status';
updateUI();
}

frameSlider.addEventListener('input',function(){
var t=frameSlider.value/10;
frameTime.textContent=t.toFixed(1)+'s';
if(originalVideoPreview.duration)originalVideoPreview.currentTime=t;
});

function captureVideoFrame(){
var video=originalVideoPreview;
var cvs=document.createElement('canvas');
cvs.width=video.videoWidth;
cvs.height=video.videoHeight;
var ctx=cvs.getContext('2d');
ctx.drawImage(video,0,0);
var dataUrl=cvs.toDataURL('image/png');
var img=new Image();
img.onload=function(){
originalImage=img;
rois=[];
renderRoiList();
setupCanvas();
drawHint.textContent='Frame captured ('+video.videoWidth+'x'+video.videoHeight+') - Drag to select watermark, then click "Add Region"';
};
img.src=dataUrl;
}

captureFrameBtn.addEventListener('click',function(){
if(!isVideo||!originalVideoPreview.duration)return;
var video=originalVideoPreview;
var t=frameSlider.value/10;
if(Math.abs(video.currentTime-t)<0.05){
captureVideoFrame();
return;
}
video.currentTime=t;
video.onseeked=function(){
captureVideoFrame();
video.onseeked=null;
};
});

function setupCanvas(){
if(!originalImage)return;
var containerWidth=canvasContainer.parentElement.clientWidth-40;
var maxDisplayWidth=Math.min(originalImage.naturalWidth||originalImage.width,containerWidth);
displayScale=maxDisplayWidth/(originalImage.naturalWidth||originalImage.width);

roiCanvas.width=originalImage.naturalWidth||originalImage.width;
roiCanvas.height=originalImage.naturalHeight||originalImage.height;
roiCanvas.style.width=maxDisplayWidth+'px';
roiCanvas.style.height=((originalImage.naturalHeight||originalImage.height)*displayScale)+'px';

canvasContainer.style.display='block';
redrawAll();
}

function getCanvasPos(e){
var rect=roiCanvas.getBoundingClientRect();
return{x:(e.clientX-rect.left)/displayScale,y:(e.clientY-rect.top)/displayScale};
}

function redrawAll(){
var ctx=roiCanvas.getContext('2d');
ctx.clearRect(0,0,roiCanvas.width,roiCanvas.height);
if(originalImage) ctx.drawImage(originalImage,0,0);

rois.forEach(function(r,i){
var color=COLORS[i%COLORS.length];
ctx.strokeStyle=color;
ctx.lineWidth=2;
ctx.setLineDash([4,2]);
ctx.strokeRect(r.x,r.y,r.w,r.h);
ctx.setLineDash([]);
ctx.fillStyle=color+'22';
ctx.fillRect(r.x,r.y,r.w,r.h);
ctx.fillStyle=color;
ctx.font='12px sans-serif';
ctx.fillText('#'+(i+1),r.x+2,r.y-4);
});

if(isDrawing){
var x1=Math.min(drawStart.x,drawEnd.x);
var y1=Math.min(drawStart.y,drawEnd.y);
var w=Math.abs(drawEnd.x-drawStart.x);
var h=Math.abs(drawEnd.y-drawStart.y);
ctx.strokeStyle='#fff';
ctx.lineWidth=2;
ctx.setLineDash([6,3]);
ctx.strokeRect(x1,y1,w,h);
ctx.setLineDash([]);
ctx.fillStyle='rgba(255,255,255,0.1)';
ctx.fillRect(x1,y1,w,h);
}
}

roiCanvas.addEventListener('mousedown',function(e){
if(currentMode!=='manual')return;
isDrawing=true;
var pos=getCanvasPos(e);
drawStart=pos;drawEnd=pos;
drawHint.textContent='Drawing... release mouse to finish';
});

roiCanvas.addEventListener('mousemove',function(e){
if(!isDrawing||!originalImage)return;
drawEnd=getCanvasPos(e);
redrawAll();
});

roiCanvas.addEventListener('mouseup',function(e){
if(!isDrawing)return;
isDrawing=false;
drawEnd=getCanvasPos(e);
var x1=Math.max(0,Math.min(drawStart.x,drawEnd.x));
var y1=Math.max(0,Math.min(drawStart.y,drawEnd.y));
var x2=Math.min(roiCanvas.width,Math.max(drawStart.x,drawEnd.x));
var y2=Math.min(roiCanvas.height,Math.max(drawStart.y,drawEnd.y));
var rw=Math.round(x2-x1),rh=Math.round(y2-y1);
if(rw>5&&rh>5){
drawHint.textContent='Rectangle: ('+Math.round(x1)+','+Math.round(y1)+') '+rw+'x'+rh+' - Click "Add Region" to save';
}else{
drawHint.textContent='Area too small - drag again to select watermark';
}
redrawAll();
});

roiCanvas.addEventListener('mouseleave',function(){
if(isDrawing){isDrawing=false;drawHint.textContent='Selection cancelled - drag again';redrawAll();}
});

addRoiBtn.addEventListener('click',function(){
if(isDrawing)return;
var x1=Math.max(0,Math.min(drawStart.x,drawEnd.x));
var y1=Math.max(0,Math.min(drawStart.y,drawEnd.y));
var x2=Math.min(roiCanvas.width,Math.max(drawStart.x,drawEnd.x));
var y2=Math.min(roiCanvas.height,Math.max(drawStart.y,drawEnd.y));
var rw=Math.round(x2-x1),rh=Math.round(y2-y1);
if(rw<5||rh<5){
drawHint.textContent='Draw a rectangle first, then click "Add Region"';
return;
}
rois.push({x:Math.round(x1),y:Math.round(y1),w:rw,h:rh});
renderRoiList();
drawHint.textContent='Region #'+rois.length+' added. Draw another or click Process.';
drawStart={x:0,y:0};drawEnd={x:0,y:0};
redrawAll();
});

clearAllRoiBtn.addEventListener('click',function(){
rois=[];
renderRoiList();
drawStart={x:0,y:0};drawEnd={x:0,y:0};
drawHint.textContent=isVideo?'Capture a frame and drag to select watermark':'Drag mouse to select watermark area';
if(originalImage) redrawAll();
});

previewMaskBtn.addEventListener('click',async function(){
if(!currentFile||rois.length===0){
statusDiv.textContent='Add regions first, then preview mask';
statusDiv.className='status error';
return;
}
var formData=new FormData();
formData.append('file',currentFile);
formData.append('rois',JSON.stringify(rois));
try{
var resp=await fetch('/api/preview_mask',{method:'POST',body:formData});
var data=await resp.json();
if(data.success){
var img=new Image();
img.onload=function(){
originalImage=img;
setupCanvas();
};
img.src=data.preview_url+'?t='+Date.now();
statusDiv.textContent='Mask preview updated on canvas';
statusDiv.className='status success';
}else{
statusDiv.textContent='Preview failed: '+(data.error||'Unknown');
statusDiv.className='status error';
showError('Mask Preview Failed',data.error||'Unknown error','Make sure regions are correctly selected and the image is valid.',false);
}
}catch(err){
statusDiv.textContent='Preview request failed: '+err.message;
statusDiv.className='status error';
showError('Connection Error','Failed to connect to server: '+err.message,'',false);
}
});

function deleteRoi(index){
rois.splice(index,1);
renderRoiList();
if(originalImage) redrawAll();
drawHint.textContent=rois.length+' region(s) selected';
}

function renderRoiList(){
roiList.innerHTML='';
if(rois.length===0){
roiList.innerHTML='<div style="color:#666;font-size:.85em;padding:4px">No regions yet. Draw on canvas and click "Add Region".</div>';
return;
}
rois.forEach(function(r,i){
var color=COLORS[i%COLORS.length];
var div=document.createElement('div');
div.className='roi-item';
div.innerHTML='<span class="roi-color" style="background:'+color+'"></span>'+
'<span>Region #'+(i+1)+': ('+r.x+','+r.y+') '+r.w+'x'+r.h+'</span>'+
'<button class="btn-danger" onclick="deleteRoi('+i+')" style="padding:2px 8px;font-size:.75em">X</button>';
roiList.appendChild(div);
});
}

processBtn.addEventListener('click',async function(){
if(!currentFile)return;
processBtn.disabled=true;
progressBar.style.display='block';
progressFill.style.width='10%';
statusDiv.textContent=isVideo?'Processing video (may take a while)...':'Processing image...';
statusDiv.className='status';

var formData=new FormData();
formData.append('file',currentFile);

if(currentMode==='manual'){
formData.append('mode','manual');
formData.append('rois',JSON.stringify(rois));
}

try{
var resp=await fetch('/api/process',{method:'POST',body:formData});
var data=await resp.json();
progressFill.style.width='100%';

if(data.success){
resultFilename=data.filename;
var resultUrl='/api/download/'+data.filename+'?t='+Date.now();
downloadBtn.style.display='inline-block';

if(isVideo){
resultVideoPreview.src=resultUrl;
resultVideoPreview.style.display='block';
resultImagePreview.style.display='none';
}else{
resultImagePreview.src=resultUrl;
resultImagePreview.style.display='block';
resultVideoPreview.style.display='none';
}

statusDiv.textContent='Done! ('+(data.time?data.time.toFixed(1):'?')+'s)';
statusDiv.className='status success';
if(data.hint)statusDiv.textContent+=' | '+data.hint;
}else{
var errMsg=data.error||'Unknown error';
var errHint=data.hint||'';
var isNoWatermark=errMsg.toLowerCase().indexOf('no watermark')>=0||errMsg.toLowerCase().indexOf('not detected')>=0;
statusDiv.textContent='Failed: '+errMsg;
statusDiv.className='status error';
showError('Processing Failed',errMsg,errHint||(isNoWatermark?'Please switch to Manual Selection mode to manually mark the watermark region.':'Check the watermark area and try again, or switch to Manual Selection mode.'),currentMode==='auto');
}
}catch(err){
statusDiv.textContent='Request failed: '+err.message;
statusDiv.className='status error';
showError('Connection Error','Failed to connect to the server: '+err.message,'Please check if the server is running and try again.',false);
}finally{
processBtn.disabled=false;
progressBar.style.display='none';
}
});

downloadBtn.addEventListener('click',function(){
if(resultFilename)window.open('/api/download/'+resultFilename,'_blank');
});

async function checkHealth(){
try{
var resp=await fetch('/api/health');
var data=await resp.json();
if(data.status==='ready'){
var parts=['AI engine ready'];
parts.push(data.device||'unknown');
var caps=[];
if(data.capabilities){
if(data.capabilities.lama) caps.push('LaMa');
if(data.capabilities.yolo) caps.push('YOLO');
if(data.capabilities.u2net) caps.push('U2Net');
if(data.capabilities.sam) caps.push('SAM');
if(data.capabilities.optical_flow) caps.push('Flow');
}
if(caps.length) parts.push('['+caps.join('+')+']');
statusDiv.textContent=parts.join(' ') + ' - Upload image or video';
statusDiv.className='status';
}
}catch(e){}
}
checkHealth();
setInterval(checkHealth,10000);

// Training dialog
trainModelBtn.addEventListener('click',function(){
trainDialog.style.display='block';
trainOverlay.style.display='block';
trainLog.style.display='none';
trainLog.textContent='';
startTrainBtn.disabled=false;
});

function closeTrainDialog(){
trainDialog.style.display='none';
trainOverlay.style.display='none';
trainLog.style.display='none';
}

cancelTrainBtn.addEventListener('click',closeTrainDialog);
trainOverlay.addEventListener('click',closeTrainDialog);

trainType.addEventListener('change',function(){
var v=this.value;
textInputGroup.style.display=(v==='logo')?'none':'block';
logoInputGroup.style.display=(v==='text')?'none':'block';
});

var trainPollTimer=null;

function stopTrainPoll(){
if(trainPollTimer){clearInterval(trainPollTimer);trainPollTimer=null;}
}

startTrainBtn.addEventListener('click',function(){
var payload={
mode:trainType.value,
text:trainText.value,
logo_dir:logoDir.value,
epochs:parseInt(trainEpochs.value)||100,
num:parseInt(trainNum.value)||2000
};
startTrainBtn.disabled=true;
trainLog.style.display='block';
trainLog.textContent='Starting training...\n';
fetch('/api/train',{
method:'POST',
headers:{'Content-Type':'application/json'},
body:JSON.stringify(payload)
}).then(function(r){return r.json()}).then(function(data){
if(data.success){
trainLog.textContent+='Training started in background.\n';
trainPollTimer=setInterval(function(){
fetch('/api/train_status').then(function(r){return r.json()}).then(function(d){
trainLog.textContent=d.log;
if(d.done){
stopTrainPoll();
startTrainBtn.disabled=false;
if(d.log.indexOf('SUCCESS')>=0){
trainLog.textContent+='\nTraining completed! Model auto-loaded.';
}else{
trainLog.textContent+='\nTraining failed. Check log for details.';
}
}
});
},2000);
}else{
trainLog.textContent+='Error: '+(data.error||'Unknown');
startTrainBtn.disabled=false;
}
}).catch(function(err){
trainLog.textContent='Request failed: '+err.message;
startTrainBtn.disabled=false;
showError('Training Error','Failed to start training: '+err.message,'',false);
});
});
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(UI_HTML)


@app.route('/api/health')
def health():
    """
    Detailed health check endpoint.
    Returns status of all modules including CUDA, LaMa, YOLO, U2-Net, SAM, and optical flow.
    """
    status_info = {
        'status': 'ready' if engine is not None else 'initializing',
        'device': str(engine.device) if engine else 'unknown',
        'cuda': torch.cuda.is_available() if engine else False,
        'capabilities': {
            'lama': engine.inpainter_loaded if engine else False,
            'yolo': engine.yolo_available if engine else False,
            'u2net': engine.u2net_available if engine else False,
            'sam': engine.sam_available if engine else False,
            'optical_flow': engine.video_processor_loaded if engine else False,
        } if engine else {},
        'version': '12.0',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    return jsonify(status_info)


@app.route('/api/process', methods=['POST'])
def process():
    global engine, detector

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'}), 400

    ext = Path(file.filename).suffix.lower()
    timestamp = int(time.time())
    safe_name = f"upload_{timestamp}{ext}"
    upload_path = UPLOAD_DIR / safe_name
    file.save(str(upload_path))

    mode = request.form.get('mode', 'auto')
    start_time = time.time()

    try:
        if engine is None:
            logger.info("Lazy-initializing engine on first request...")
            engine = get_engine()
            detector = WatermarkDetector()
            logger.info("Engine ready on device: %s", engine.device)

        is_video = ext in ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv')

        bboxes = None
        auto_detect = (mode == 'auto')

        if mode == 'manual':
            rois_json = request.form.get('rois', '[]')
            try:
                rois = json.loads(rois_json)
                if rois and isinstance(rois, list) and len(rois) > 0:
                    bboxes = []
                    for r in rois:
                        rx = int(r.get('x', 0))
                        ry = int(r.get('y', 0))
                        rw = int(r.get('w', 0))
                        rh = int(r.get('h', 0))
                        if rw > 5 and rh > 5:
                            bboxes.append((rx, ry, rx + rw, ry + rh))
                    if not bboxes:
                        return jsonify({
                            'success': False,
                            'error': 'No valid watermark regions selected. '
                                     'Please draw a rectangle and click "Add Region" before processing.'
                        }), 400
                    auto_detect = False
                    logger.info(f"Manual mode: {len(bboxes)} bbox(es)")
                else:
                    return jsonify({
                        'success': False,
                        'error': 'No watermark regions selected. '
                                 'Draw a rectangle on the canvas and click "Add Region".'
                    }), 400
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning(f"Failed to parse ROIs JSON: {e}")
                return jsonify({
                    'success': False,
                    'error': f'ROI data parsing failed: {str(e)}. Please re-draw and try again.'
                }), 400

        if is_video:
            output_path = engine.inpaint_video(
                str(upload_path), bboxes=bboxes, auto_detect=auto_detect
            )
        else:
            output_path = engine.inpaint_image(
                str(upload_path), bboxes=bboxes, auto_detect=auto_detect
            )

        elapsed = time.time() - start_time
        filename = os.path.basename(str(output_path))

        response = {
            'success': True,
            'filename': filename,
            'time': elapsed,
        }

        if auto_detect:
            if not engine.yolo_available and not engine.u2net_available:
                response['hint'] = (
                    'AI detection models not installed. '
                    'Manual selection mode is recommended for best results.'
                )
            else:
                response['hint'] = (
                    'Auto-detection completed. If the watermark was not fully removed, '
                    'switch to Manual Selection mode.'
                )

        return jsonify(response)

    except WatermarkNotFoundError as e:
        # No watermark detected by any strategy
        logger.warning("Processing failed (no watermark): %s", e)
        return jsonify({
            'success': False,
            'error': str(e),
            'hint': 'Switch to Manual Selection mode, draw the watermark region on the canvas, and process again.'
        }), 400
    except ValueError as e:
        # Expected errors: empty mask, invalid bbox, etc.
        logger.warning("Processing failed (expected): %s", e)
        return jsonify({'success': False, 'error': str(e)}), 400
    except RuntimeError as e:
        logger.error("Processing failed (runtime): %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500
    except Exception as e:
        logger.error("Processing failed: %s", e, exc_info=True)
        err_msg = str(e)
        if 'CUDA out of memory' in err_msg or 'OutOfMemoryError' in err_msg:
            err_msg = 'GPU out of memory. Try processing a smaller video or use CPU mode.'
        elif 'Cannot open video' in err_msg:
            err_msg = 'Video format not supported or file is corrupted. Try MP4/H.264 format.'
        elif 'Cannot load' in err_msg or 'not installed' in err_msg:
            err_msg = f'Required model not available: {err_msg}'
        return jsonify({'success': False, 'error': err_msg}), 500


@app.route('/api/preview_mask', methods=['POST'])
def preview_mask():
    """
    Generate and return a mask preview image for manual mode validation.
    v12.0: Supports both images and videos (extracts first frame from video).
    """
    global engine

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400

    file = request.files['file']
    rois_json = request.form.get('rois', '[]')

    try:
        rois = json.loads(rois_json)
        if not rois:
            return jsonify({'success': False, 'error': 'No regions selected'}), 400

        bboxes = []
        for r in rois:
            rx = int(r.get('x', 0))
            ry = int(r.get('y', 0))
            rw = int(r.get('w', 0))
            rh = int(r.get('h', 0))
            if rw > 5 and rh > 5:
                bboxes.append((rx, ry, rx + rw, ry + rh))

        if not bboxes:
            return jsonify({'success': False, 'error': 'No valid regions'}), 400

        if engine is None:
            engine = get_engine()

        ext = Path(file.filename).suffix.lower()
        is_video = ext in ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv')

        if is_video:
            # Extract first frame from video
            import tempfile
            temp_path = UPLOAD_DIR / f"_temp_preview_{int(time.time())}{ext}"
            file.save(str(temp_path))

            cap = cv2.VideoCapture(str(temp_path))
            if not cap.isOpened():
                return jsonify({'success': False, 'error': 'Cannot open video file'}), 400

            ret, frame = cap.read()
            cap.release()

            # Clean up temp file
            try:
                os.remove(str(temp_path))
            except Exception:
                pass

            if not ret:
                return jsonify({'success': False, 'error': 'Cannot read first frame from video. '
                                 'Capture a frame manually using the video slider.'}), 400

            image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            logger.info(f"Preview mask: extracted first frame from video ({image_np.shape[1]}x{image_np.shape[0]})")
        else:
            # Read image directly
            from PIL import Image
            img = Image.open(file.stream).convert('RGB')
            image_np = np.array(img)

        # Generate preview
        preview = engine.preview_mask(image_np, bboxes)

        # Save preview
        preview_path = CACHE_DIR / f"mask_preview_{int(time.time())}.png"
        cv2.imwrite(str(preview_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

        return jsonify({
            'success': True,
            'preview_url': f'/api/download/{preview_path.name}'
        })

    except Exception as e:
        logger.error(f"Mask preview failed: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/download/<path:filename>')
def download(filename):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        file_path = CACHE_DIR / filename
    if not file_path.exists():
        return jsonify({'success': False, 'error': 'File not found'}), 404
    return send_file(str(file_path), as_attachment=True)


# ============================================================
# Training API endpoints
# ============================================================

@app.route('/api/train', methods=['POST'])
def train_model_api():
    data = request.json or {}
    mode = data.get('mode', 'text')
    text = data.get('text', 'AI Generated')
    logo_dir = data.get('logo_dir', str(LOGO_DIR))
    epochs = int(data.get('epochs', 100))
    num = int(data.get('num', 2000))

    logger.info(f"Training requested: mode={mode}, text='{text}', epochs={epochs}, num={num}")

    def train_thread():
        try:
            with open(TRAIN_LOG_PATH, 'w', encoding='utf-8') as f:
                f.write(f"=== Training started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write(f"Mode: {mode}, Text: {text}, Epochs: {epochs}, Images: {num}\n")
                f.write("Starting training...\n")
                f.flush()

            cmd = [
                TRAIN_PYTHON,
                str(PROJECT_ROOT / "auto_train.py"),
                "--mode", mode,
                "--text", text,
                "--logo_dir", logo_dir,
                "--epochs", str(epochs),
                "--num", str(num),
                "--bg_dir", str(DATASET_DIR / "backgrounds"),
                "--no_reload"
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            with open(TRAIN_LOG_PATH, 'a', encoding='utf-8') as log_f:
                for line in proc.stdout:
                    log_f.write(line)
                    log_f.flush()
            proc.wait()

            if proc.returncode == 0:
                with open(TRAIN_LOG_PATH, 'a', encoding='utf-8') as f:
                    f.write(f"\n=== [SUCCESS] Training pipeline completed at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                if engine:
                    engine.load_detector()
                    logger.info("Model reloaded after successful training")
            else:
                with open(TRAIN_LOG_PATH, 'a', encoding='utf-8') as f:
                    f.write(f"\n=== [ERROR] Training failed with code {proc.returncode} at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            with open(TRAIN_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(f"\n=== [ERROR] Training exception: {e} ===\n")

    thread = threading.Thread(target=train_thread, daemon=True)
    thread.start()
    return jsonify({'success': True, 'message': 'Training started in background'})


@app.route('/api/reload_model', methods=['POST'])
def reload_model():
    try:
        if engine:
            ok = engine.load_detector()
            if ok:
                logger.info("Model reloaded successfully via API")
                return jsonify({'success': True, 'message': 'Model reloaded'})
            else:
                return jsonify({'success': False, 'error': 'Detector reload returned False'}), 500
        else:
            return jsonify({'success': False, 'error': 'Engine not initialized'}), 500
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/train_status')
def train_status():
    if TRAIN_LOG_PATH.exists():
        with open(TRAIN_LOG_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        recent = lines[-50:] if len(lines) > 50 else lines
        log_text = ''.join(recent)
        is_done = 'SUCCESS' in log_text or 'ERROR' in log_text or 'failed' in log_text.lower()
        return jsonify({'log': log_text, 'done': is_done})
    return jsonify({'log': 'No training running.', 'done': False})


def _try_open_browser(url: str):
    try:
        if webbrowser.open(url):
            return True
    except Exception:
        pass
    try:
        os.startfile(url)
        return True
    except Exception:
        pass
    try:
        subprocess.Popen(['cmd', '/c', 'start', url], shell=True)
        return True
    except Exception:
        pass
    return False


def init_engine():
    global engine, detector
    try:
        logger.info("Initializing inpainting engine...")
        engine = get_engine()
        detector = WatermarkDetector()
        logger.info("Engine ready on device: %s", engine.device)

        print()
        print("[OK] AI engine ready!")
        print("Opening browser: http://localhost:5000")
        logger.info("Opening browser at http://localhost:5000")

        success = _try_open_browser("http://localhost:5000")
        if not success:
            print("[WARN] Could not open browser automatically.")
            print("Please open your browser and visit: http://localhost:5000")
        return True
    except Exception as e:
        logger.error("Engine init failed: %s", e, exc_info=True)
        print()
        print(f"[FAIL] AI engine failed to initialize: {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("Watermark Remover - AI Inpainting Server v12.0")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Log dir: {LOG_DIR}")
    print("=" * 60)
    print()
    print("Initializing AI engine (10-30s on first run)...")
    print()

    if init_engine():
        print()
        print("=" * 60)
        print("  Server running on http://localhost:5000")
        print("  Browser should open automatically.")
        print("  Press Ctrl+C to stop")
        print("=" * 60)
        try:
            app.run(host='0.0.0.0', port=5000, debug=False)
        except KeyboardInterrupt:
            print()
            print("Server stopped by user.")
        except Exception as e:
            logger.error("Server runtime error: %s", e, exc_info=True)
            print(f"[ERROR] Server crashed: {e}")
            sys.exit(1)
    else:
        print("[FAIL] AI engine failed to initialize!")
        print("Check logs:")
        print(f"  {LOG_DIR / 'api_server.log'}")
        print(f"  {LOG_DIR / 'engine.log'}")
        print()
        print("Press Enter to exit...")
        input()
        sys.exit(1)