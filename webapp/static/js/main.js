import { hasCompletedToday } from "./daily_challenge.js";
import { clearCanvas } from "./drawing.js";
import { confirmCountry, guess, refreshGuess } from "./guess.js";
import {
  showLeaderboard,
  showLeaderboardNext,
  showLeaderboardPrevious,
} from "./leaderboard.js";
import "./minigame.js";

// Application State
let appState = "home"; // "home", "confirm", "leaderboard"

// UI Elements
const leftBtn = document.getElementById("left-btn");
const rightBtn = document.getElementById("right-btn");
const leaderboardBtn = document.getElementById("leaderboard-btn");

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
 * UI Helper Functions
 */
function setLeaderboardButtonVisibility(visible) {
  const leaderboardBtnContainer = leaderboardBtn.parentElement;
  if (leaderboardBtnContainer) {
    leaderboardBtnContainer.style.display = visible ? "block" : "none";
  }
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
  leftBtn.textContent = "Guess Country";
  rightBtn.textContent = "Clear";
  leftBtn.classList.remove("locked");
  setLeaderboardButtonVisibility(true);
  leaderboardBtn.classList.remove("active");
  if (hasCompletedToday()) {
    leftBtn.classList.add("golden");
  }
}

function updateConfirmUI() {
  leftBtn.textContent = "Confirm";
  rightBtn.textContent = "Clear";
  setLeaderboardButtonVisibility(false);
}

function updateLeaderboardUI() {
  leftBtn.textContent = "Previous";
  rightBtn.textContent = "Next";
  setLeaderboardButtonVisibility(true);
  leaderboardBtn.classList.add("active");
  leftBtn.classList.remove("golden");
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
    handleRefresh(); // "Clear" - reset the canvas
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
  if (hasCompletedToday()) {
    leftBtn.classList.add("golden");
  }
}

function handleRefresh() {
  refreshGuess();
  clearCanvas();
  setState("home");
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
  const result = await showLeaderboard();
  if (result.success) {
    setState("leaderboard");
    leaderboardBtn.textContent = "Hide Leaderboard";
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
  leaderboardBtn.addEventListener("click", handleLeaderboardButtonClick);
}

/**
 * Initialize Application
 */
document.addEventListener("DOMContentLoaded", function () {
  initializeEventListeners();
  updateUI(); // Set initial UI state
});

// Export state management functions for use by other modules
export { getState, setState };
