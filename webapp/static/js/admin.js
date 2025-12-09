import { renderUserDrawing } from "./drawing.js";
import { showMessage, showLoadingMessage } from "./messages.js";
import { initializeButtonBounce } from "./animations.js";

// Hold filename
let filename;

// UI Elements
const leftBtn = document.getElementById("left-btn");
const rightBtn = document.getElementById("right-btn");

/**
 * API Functions
 */
async function fetchDrawing() {
  const response = await fetch(`/unvalidated_drawing`);
  if (!response.ok) {
    throw new Error("Failed to fetch unvalidated drawing");
  }
  return response.json();
}

async function approveDrawingAPI(filename) {
  const response = await fetch(`/drawing/${filename}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ validated: true }),
  });
  if (!response.ok) {
    throw new Error("Failed to update drawing");
  }
  return response.json();
}

async function deleteDrawingAPI(filename) {
  const response = await fetch(`/drawing/${filename}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error("Failed to delete drawing");
  }
  return response.json();
}

/**
 * UI Functions
 */
async function showDrawing() {
  rightBtn.classList.add("locked");
  leftBtn.classList.add("locked");

  showLoadingMessage();
  const data = await fetchDrawing();

  if (data.message === "No unvalidated drawings found") {
    showMessage("No drawings to validate...");
    return;
  }

  rightBtn.classList.remove("locked");
  leftBtn.classList.remove("locked");

  renderUserDrawing(data.lines);
  filename = data.filename;

  // Message
  const scorePercent = Math.round(data.country_score * 100);
  const guessScorePercent = Math.round(data.guess_score * 100);
  const date = new Date(data.timestamp).toISOString().split("T")[0];
  const authorSuffix = data.author?.trim() ? ` by ${data.author.trim()}` : "";

  showMessage(`${data.country_name} (${scorePercent}%)
  Prediction: ${data.country_guess} (${guessScorePercent}%)
  Drawn on ${date}${authorSuffix}`);
}

function approveDrawing() {
  if (!filename) {
    console.error("No filename to approve!");
    return;
  }

  approveDrawingAPI(filename)
    .then(() => {
      showDrawing();
    })
    .catch((error) => {
      console.error("Error approving drawing:", error);
      showMessage("Failed to approve drawing. Please try again.");
    });
}

function deleteDrawing() {
  if (!filename) {
    console.error("No filename to delete!");
    return;
  }

  deleteDrawingAPI(filename)
    .then(() => {
      showDrawing();
    })
    .catch((error) => {
      console.error("Error deleting drawing:", error);
      showMessage("Failed to delete drawing. Please try again.");
    });
}

/**
 * Initialize Application
 */
function initializeButtons() {
  leftBtn.textContent = "Approve";
  rightBtn.textContent = "Delete";

  leftBtn.addEventListener("click", approveDrawing);
  rightBtn.addEventListener("click", deleteDrawing);
}

document.addEventListener("DOMContentLoaded", function () {
  initializeButtons();
  initializeButtonBounce();
  showDrawing();
});
