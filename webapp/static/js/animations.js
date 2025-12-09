// UI Elements
const rightBtn = document.getElementById("right-btn");
const leaderboardBtn = document.getElementById("leaderboard-btn");
const undoBtn = document.getElementById("undo-btn");

let undoVisible = false;
let leaderboardVisible = false;

export function showUndoBtn() {
  if (undoVisible) return;
  undoVisible = true;

  // 1. Animate the right button shrinking
  anime({
    targets: rightBtn,
    width: "calc(100% - 3rem)",
    duration: 300,
    easing: "easeOutElastic(1, .7)",
    begin: function () {
      rightBtn.classList.remove("w-full");
    },
    complete: function () {
      // 2. Animate the undo button appearing AFTER the right button has shrunk
      anime({
        targets: undoBtn,
        opacity: [0, 1],
        scale: [0, 1],
        duration: 400,
        begin: function () {
          undoBtn.classList.remove("hidden");
        },
      });
    },
  });
}

export function hideUndoBtn() {
  if (!undoVisible) return;
  undoVisible = false;

  // 1. Animate the undo button disappearing
  anime({
    targets: undoBtn,
    opacity: 0.3,
    scale: 0.2,
    duration: 300,
    easing: "easeInOutQuad",
    complete: function () {
      undoBtn.classList.add("hidden");
    },
  });

  // 2. Animate the right button expanding
  anime({
    targets: rightBtn,
    width: "100%",
    duration: 300,
    easing: "easeOutQuint",
    complete: function () {
      rightBtn.classList.add("w-full");
      rightBtn.style.width = "";
    },
  });
}

export function showLeaderboardButton() {
  if (leaderboardVisible) return;
  leaderboardVisible = true;

  const leaderboardContainer = leaderboardBtn.parentElement;

  // Make sure it's visible
  leaderboardContainer.style.display = "flex";

  // Animate sliding up from below the screen
  anime({
    targets: leaderboardContainer,
    translateY: [100, 0],
    opacity: [0, 1],
    duration: 1200,
    easing: "easeOutElastic(1, .8)",
  });
}

export function hideLeaderboardButton() {
  if (!leaderboardVisible) return;
  leaderboardVisible = false;

  const leaderboardContainer = leaderboardBtn.parentElement;

  // Animate sliding down below the screen
  anime({
    targets: leaderboardContainer,
    translateY: [0, 100],
    opacity: [1, 0],
    duration: 500,
    easing: "easeInBack(1.7)",
    complete: function () {
      leaderboardContainer.style.display = "none";
    },
  });
}

export function initializeButtonBounce() {
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
