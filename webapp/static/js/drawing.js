import { hideUndoBtn, showUndoBtn } from "./main.js";

var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");

let originalWidth = canvas.width;
let originalHeight = canvas.height;

var isDrawing = false;
var lastX, lastY;
var lines = [];
var currentLine = [];

// Mouse events
canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseout", stopDrawing);

// Touch events
canvas.addEventListener("touchstart", handleTouchStart, { passive: false });
canvas.addEventListener("touchmove", handleTouchMove, { passive: false });
canvas.addEventListener("touchend", handleTouchEnd);

// Function to resize canvas
function resizeCanvas() {
  const margin = 16; // Space to leave on sides
  const BottomMargin = 300; // Space for title and buttons

  // Get the smaller of window width or height
  const maxSize = Math.min(
    window.innerWidth - margin * 2,
    window.innerHeight - BottomMargin,
  );

  // Calculate scale factors
  const scaleX = maxSize / originalWidth;
  const scaleY = maxSize / originalHeight;

  // Scale the lines
  scaleLines(scaleX, scaleY);

  // Set canvas size
  canvas.width = maxSize;
  canvas.height = maxSize;

  // Update original dimensions
  originalWidth = canvas.width;
  originalHeight = canvas.height;

  // Update container size
  const container = document.getElementById("canvas-container");
  container.style.width = `${maxSize}px`;
  container.style.height = `${maxSize}px`;

  // Redraw any existing lines
  redrawCanvas();
}

// Function to scale the lines
function scaleLines(scaleX, scaleY) {
  lines = lines.map((line) => line.map(([x, y]) => [x * scaleX, y * scaleY]));
}

// Function to redraw canvas after resize
function redrawCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  ctx.strokeStyle = "#ffffff";

  lines.forEach((line) => {
    if (line.length > 1) {
      ctx.beginPath();
      ctx.moveTo(line[0][0], line[0][1]);
      for (let i = 1; i < line.length; i++) {
        ctx.lineTo(line[i][0], line[i][1]);
      }
      ctx.stroke();
    }
  });
}

// Add resize event listener
window.addEventListener("resize", resizeCanvas);

// Initial canvas setup
resizeCanvas();

function getCoordinates(event) {
  if (event.type.includes("touch")) {
    // Prevent scrolling when touching the canvas
    event.preventDefault();
    const touch = event.touches[0] || event.changedTouches[0];
    return {
      x: touch.clientX - canvas.getBoundingClientRect().left,
      y: touch.clientY - canvas.getBoundingClientRect().top,
    };
  } else {
    return {
      x: event.clientX - canvas.getBoundingClientRect().left,
      y: event.clientY - canvas.getBoundingClientRect().top,
    };
  }
}

function startDrawing(event) {
  isDrawing = true;
  const coords = getCoordinates(event);
  [lastX, lastY] = [coords.x, coords.y];
  currentLine.push([coords.x, coords.y]);
}

function draw(event) {
  if (!isDrawing) return;

  const coords = getCoordinates(event);

  // Check if distance is greater than a threshold
  var distance = Math.sqrt(
    Math.pow(coords.x - lastX, 2) + Math.pow(coords.y - lastY, 2),
  );
  if (distance > 3) {
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(coords.x, coords.y);
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();

    [lastX, lastY] = [coords.x, coords.y];
    currentLine.push([coords.x, coords.y]);
  }
}

function updateMiniGame() {
  if (window.miniGame && window.miniGame.isActive) {
    window.miniGame.createLineWalls();
    window.miniGame.lastLineCount = window.lines.length;
  }
}

function stopDrawing() {
  isDrawing = false;
  if (currentLine.length > 1) {
    lines.push(currentLine);
    window.lines = lines;
    showUndoBtn();
    updateMiniGame();
  }
  currentLine = [];
}

// Renders a user drawing on the canvas, scaling and centering it with a margin.
function renderUserDrawing(linesToDraw) {
  if (!linesToDraw || linesToDraw.length === 0) return;

  // Find bounding box of all points
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  linesToDraw.forEach((line) => {
    line.forEach(([x, y]) => {
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    });
  });

  // Compute scale and offset to fit drawing in canvas with margin
  const margin = canvas.width * 0.1;
  const drawWidth = maxX - minX;
  const drawHeight = maxY - minY;
  const scale = Math.min(
    (canvas.width - 2 * margin) / (drawWidth || 1),
    (canvas.height - 2 * margin) / (drawHeight || 1),
  );

  // Center drawing
  const offsetX = (canvas.width - scale * drawWidth) / 2 - scale * minX;
  const offsetY = (canvas.height - scale * drawHeight) / 2 - scale * minY;

  // Flip vertically and scale
  lines = linesToDraw.map((line) =>
    line.map(([x, y]) => [
      x * scale + offsetX,
      (maxY - (y - minY)) * scale + offsetY,
    ]),
  );
  window.lines = lines;

  redrawCanvas();
  updateMiniGame();
}

// Touch event handlers
function handleTouchStart(event) {
  startDrawing(event);
}

function handleTouchMove(event) {
  draw(event);
}

function handleTouchEnd(event) {
  stopDrawing();
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  lines = [];
  window.lines = lines;
}

// Make lines globally accessible for minigame
window.lines = lines;

function undoLine() {
  if (lines.length > 0) {
    lines.pop();
    window.lines = lines;
    redrawCanvas();
    updateMiniGame();
  }
  if (lines.length === 0) {
    hideUndoBtn();
  }
}

export { lines, clearCanvas, undoLine, renderUserDrawing };
