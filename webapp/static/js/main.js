import { confirmCountry, guess, refreshGuess } from "./guess.js";
import { initializeDailyChallenge } from "./daily_challenge.js";
import {
  clearCanvas,
  undoLine,
  enableDrawing,
  disableDrawing,
} from "./drawing.js";
import {
  showLeaderboard,
  showLeaderboardNext,
  showLeaderboardPrevious,
} from "./leaderboard.js";
import "./minigame.js";
import * as ani from "./animations.js";

// Application State
let appState = "home"; // "home", "confirm", "leaderboard"

// UI Elements
const leftBtn = document.getElementById("left-btn");
const rightBtn = document.getElementById("right-btn");
const leaderboardBtn = document.getElementById("leaderboard-btn");
const undoBtn = document.getElementById("undo-btn");

/**
 * State Management
 */
function getState() {
  return appState;
}

function setState(newState) {
  if (appState === newState) return;

  const validStates = ["home", "confirm", "leaderboard"];
  if (!validStates.includes(newState)) {
    console.error(`Invalid state: ${newState}`);
    return;
  }

  appState = newState;
  updateUI();
}

/**
 * UI Updates based on state
 */
function updateUI() {
  switch (appState) {
    case "home":
      updateHomeUI();
      break;
    case "confirm":
      updateConfirmUI();
      break;
    case "leaderboard":
      updateLeaderboardUI();
      break;
  }
}

function updateHomeUI() {
  ani.hideUndoBtn();
  ani.showLeaderboardButton();
  enableDrawing();
  canvas.style.cursor = "crosshair";
  leftBtn.textContent = "Guess Country";
  rightBtn.textContent = "Clear";
  rightBtn.classList.remove("green", "locked");
  leftBtn.classList.remove("locked");
  leaderboardBtn.classList.remove("active");
}

function updateConfirmUI() {
  ani.hideLeaderboardButton();
  disableDrawing();
  canvas.style.cursor = "default";
  leftBtn.textContent = "Confirm";
  rightBtn.textContent = "Clear";
  ani.hideUndoBtn();
}

function updateLeaderboardUI() {
  ani.hideUndoBtn();
  disableDrawing();
  canvas.style.cursor = "default";
  leftBtn.textContent = "Previous";
  rightBtn.textContent = "Next";
  rightBtn.classList.add("green");
  ani.showLeaderboardButton();
  leaderboardBtn.classList.add("active");
}

/**
 * Button Event Handlers
 */
function handleLeftButtonClick() {
  switch (appState) {
    case "home":
      handleGuess();
      break;
    case "confirm":
      handleConfirm();
      break;
    case "leaderboard":
      handlePrevLeaderboard();
      break;
  }
}

function handleRightButtonClick() {
  if (appState === "leaderboard") {
    handleNextLeaderboard();
  } else {
    handleRefresh();
  }
}

function handleLeaderboardButtonClick() {
  if (appState === "leaderboard") {
    handleRefresh();
  } else {
    handleShowLeaderboard();
  }
}

/**
 * Action Handlers - delegate to appropriate modules
 */
async function handleGuess() {
  const success = await guess();
  if (success) {
    setState("confirm");
  }
}

function handleConfirm() {
  confirmCountry();
  // Lock the button after confirmation
  leftBtn.classList.add("locked");
}

function handleRefresh() {
  refreshGuess();
  clearCanvas();
  setState("home");
  ani.hideUndoBtn();
  leaderboardBtn.textContent = "Show Leaderboard";
  if (window.miniGame) {
    window.miniGame.reset();
  }
}

function updateLeaderboardButtonLocks(rank, total) {
  leftBtn.classList.toggle("locked", rank === 0);
  rightBtn.classList.toggle("locked", rank >= total - 1);
}

async function handleShowLeaderboard() {
  // Update UI
  setState("leaderboard");
  leaderboardBtn.textContent = "Hide Leaderboard";
  updateLeaderboardButtonLocks(0, 1);

  // Fetch leaderboard data and update button locks
  const result = await showLeaderboard();
  if (result.success) {
    updateLeaderboardButtonLocks(result.rank, result.total);
  }
}

async function handleNextLeaderboard() {
  const result = await showLeaderboardNext();
  updateLeaderboardButtonLocks(result.rank, result.total);
}

async function handlePrevLeaderboard() {
  const result = await showLeaderboardPrevious();
  updateLeaderboardButtonLocks(result.rank, result.total);
}

/**
 * Initialize Event Listeners
 */
function initializeEventListeners() {
  leftBtn.addEventListener("click", handleLeftButtonClick);
  rightBtn.addEventListener("click", handleRightButtonClick);
  undoBtn.addEventListener("click", undoLine);
  leaderboardBtn.addEventListener("click", handleLeaderboardButtonClick);
}

/**
 * Prevent sticky hover on mobile
 */
function initializeMobileTouchFeedback() {
  const buttons = document.querySelectorAll("button");
  buttons.forEach((button) => {
    button.addEventListener("touchstart", function () {
      this.classList.add("touch-active");
    });

    button.addEventListener("touchend", function () {
      setTimeout(() => {
        this.classList.remove("touch-active");
      }, 150);
    });

    // Clean up if touch is cancelled
    button.addEventListener("touchcancel", function () {
      this.classList.remove("touch-active");
    });
  });
}

/**
 * Initialize Application
 */
document.addEventListener("DOMContentLoaded", function () {
  initializeEventListeners();
  initializeMobileTouchFeedback();
  ani.initializeButtonBounce();
  initializeDailyChallenge();
  updateUI(); // Set initial UI state
});

// Export state management functions for use by other modules
export { getState, setState };
