import { renderUserDrawing, clearCanvas } from "./drawing.js";
import { showMessage, showLoadingMessage } from "./messages.js";
import { initializeButtonBounce } from "./animations.js";

// Hold drawing ID
let drawingId;

// UI Elements
const leftBtn = document.getElementById("left-btn");
const rightBtn = document.getElementById("right-btn");

/**
 * API Functions
 */
async function fetchDrawing() {
  const response = await fetch(`/unvalidated_drawing`);
  const data = await response.json().catch(() => null);

  if (!response.ok) {
    const error = new Error(data?.message || "Failed to fetch unvalidated drawing");
    error.status = response.status;
    throw error;
  }
  return data;
}

async function approveDrawingAPI(id) {
  const response = await fetch(`/drawing/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ validated: true }),
  });
  if (!response.ok) {
    throw new Error("Failed to update drawing");
  }
  return response.json();
}

async function deleteDrawingAPI(id) {
  const response = await fetch(`/drawing/${id}`, {
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
  clearCanvas();
  drawingId = undefined;
  rightBtn.classList.add("locked");
  leftBtn.classList.add("locked");

  showLoadingMessage();
  try {
    const data = await fetchDrawing();

    if (data.message === "No unvalidated drawings found") {
      showMessage("No drawings to validate...");
      return;
    }

    rightBtn.classList.remove("locked");
    leftBtn.classList.remove("locked");

    renderUserDrawing(data.lines);
    drawingId = data.id;

    // Message
    const scorePercent = Math.round(data.country_score * 100);
    const guessScorePercent = Math.round(data.guess_score * 100);
    const date = new Date(data.timestamp).toISOString().split("T")[0];
    const authorSuffix = data.author?.trim() ? `${data.author.trim()}` : "Anonymous";
    const validatedCount = data.author_id
      ? ` (${data.validated_author_count})`
      : "";

    showMessage(`${data.country_name} (${scorePercent}%)
  Prediction: ${data.country_guess} (${guessScorePercent}%)
  Drawn on ${date} by ${authorSuffix}${validatedCount}
  ${data.unvalidated_count} drawings left to validate`);
  } catch (error) {
    showMessage("Could not load drawings for validation.");
  }
}

function approveDrawing() {
  if (!drawingId) {
    console.error("No drawing ID to approve!");
    return;
  }

  approveDrawingAPI(drawingId)
    .then(() => {
      showDrawing();
    })
    .catch((error) => {
      console.error("Error approving drawing:", error);
      showMessage("Failed to approve drawing. Please try again.");
    });
}

function deleteDrawing() {
  if (!drawingId) {
    console.error("No drawing ID to delete!");
    return;
  }

  deleteDrawingAPI(drawingId)
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
