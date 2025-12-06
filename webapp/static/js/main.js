import { clearCanvas, undoLine } from "./drawing.js";
import { confirmCountry, guess, refreshGuess } from "./guess.js";
import {
  showLeaderboard,
  showLeaderboardNext,
  showLeaderboardPrevious,
} from "./leaderboard.js";
import "./minigame.js";
import "./daily_challenge.js";
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
  canvas.style.cursor = "crosshair";
  leftBtn.textContent = "Guess Country";
  rightBtn.textContent = "Clear";
  rightBtn.classList.remove("green", "locked");
  leftBtn.classList.remove("locked");
  leaderboardBtn.classList.remove("active");
}

function updateConfirmUI() {
  ani.hideLeaderboardButton();
  canvas.style.cursor = "default";
  leftBtn.textContent = "Confirm";
  rightBtn.textContent = "Clear";
  ani.hideUndoBtn();
}

function updateLeaderboardUI() {
  ani.hideUndoBtn();
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
function initializeButtonBounce() {
  const buttons = document.querySelectorAll("button");

  buttons.forEach((button) => {
    button.addEventListener("click", function (e) {
      // Don't animate locked buttons
      if (this.classList.contains("locked")) {
        return;
      }

      // Use anime.js for smooth bounce
      anime({
        targets: this,
        scale: [
          { value: 0.92, duration: 80, easing: "easeOutQuad" },
          { value: 1, duration: 150, easing: "easeOutQuad" },
        ],
        duration: 230,
        complete: () => {
          // Add transition before clearing transform for smooth hover
          this.style.transition = "transform 0.15s ease-out";
          this.style.transform = "";

          // Remove inline transition after it completes so CSS takes over
          setTimeout(() => {
            this.style.transition = "";
          }, 150);
        },
      });
    });
  });
}

document.addEventListener("DOMContentLoaded", function () {
  initializeEventListeners();
  initializeMobileTouchFeedback();
  initializeButtonBounce();
  updateUI(); // Set initial UI state
});

// Export state management functions for use by other modules
export { getState, setState };
