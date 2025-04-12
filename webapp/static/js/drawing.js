var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");

var isDrawing = false;
var lastX, lastY;
var lines = [];
var currentLine = [];

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);

function startDrawing(event) {
  isDrawing = true;
  var x = event.clientX - canvas.getBoundingClientRect().left;
  var y = event.clientY - canvas.getBoundingClientRect().top;
  [lastX, lastY] = [x, y];
  currentLine.push([x, y]);
}

function draw(event) {
  if (!isDrawing) return;

  var x = event.clientX - canvas.getBoundingClientRect().left;
  var y = event.clientY - canvas.getBoundingClientRect().top;

  // Check if distance is greater than a threshold
  var distance = Math.sqrt(Math.pow(x - lastX, 2) + Math.pow(y - lastY, 2));
  if (distance > 6) {
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();

    [lastX, lastY] = [x, y];
    currentLine.push([x, y]);
  }
}

function stopDrawing() {
  isDrawing = false;
  if (currentLine.length > 1) {
    lines.push(currentLine);
  }
  currentLine = [];
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  lines = [];
}

export { lines, clearCanvas };
