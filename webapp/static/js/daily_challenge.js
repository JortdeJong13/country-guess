/**
 * Daily Challenge Feature
 * Animated sliding pill that shows the daily country challenge
 */

// State variables
let dailyCountry = null;
let isAnimating = false;
let isPillExpanded = false;
let currentAnimation = null;
let interactionsSetup = false;
let hasUserInteracted = false;
let temptingBounceTimeout = null;
let currentPillWidth = 200;
export let goldenStreak = 4;

// Constants
const ANIMATION_DURATIONS = {
  slideIn: 1200,
  slideOut: 800,
  iconBounce: 600,
  entrance: 800,
};

const POSITIONS = {
  tipVisible: 50, // How much of pill shows when collapsed
  expandedOffset: -90, // How far pill slides out when expanded
  entranceDelay: 500, // Delay before entrance animation
  temptingDelay: 1500, // Delay before tempting bounce starts
};

/**
 * Utility Functions
 */
function getTodayISO() {
  return new Date().toISOString().slice(0, 10);
}

function getHistory() {
  return JSON.parse(localStorage.getItem("dailyChallengeHistory") || "{}");
}

function setHistory(history) {
  localStorage.setItem("dailyChallengeHistory", JSON.stringify(history));
}

function hasCompletedToday() {
  const history = getHistory();
  return !!history[getTodayISO()];
}

function getDailyStreak() {
  const history = getHistory();
  const today = getTodayISO();

  // Start from today if completed, otherwise start from yesterday
  const currentDate = new Date(today);
  if (!history[today]) {
    // Today not completed, start from yesterday
    currentDate.setDate(currentDate.getDate() - 1);
  }

  let dailyStreak = 0;
  while (true) {
    const dateStr = currentDate.toISOString().slice(0, 10);
    // If this date exists in history, increment streak
    if (history[dateStr]) {
      dailyStreak++;
      // Move to previous day
      currentDate.setDate(currentDate.getDate() - 1);
    } else {
      break;
    }
  }

  return dailyStreak;
}

/**
 * Pill Sizing Functions
 */
function calculatePillWidth(text) {
  if (!text) return 200;

  // Create temporary element to measure text width
  const temp = document.createElement("div");
  temp.style.position = "absolute";
  temp.style.visibility = "hidden";
  temp.style.fontSize = "13px";
  temp.style.fontWeight = "700";
  temp.style.fontFamily = "sans-serif";
  temp.innerHTML = text;
  document.body.appendChild(temp);

  const textWidth = temp.offsetWidth;
  document.body.removeChild(temp);

  // Calculate total width: padding + text + icon space + buffer
  const totalWidth = Math.max(300, textWidth + 175);
  const maxWidth = Math.min(totalWidth, 600);

  return maxWidth;
}

function updatePillDimensions(width) {
  const pill = document.getElementById("daily-challenge-pill");
  if (!pill) return;

  pill.style.width = width + "px";
  currentPillWidth = width;
}

/**
 * Animation Functions
 */
function slideInPill() {
  if (isPillExpanded) return;

  if (currentAnimation) {
    currentAnimation.pause();
  }

  isAnimating = true;
  isPillExpanded = true;
  stopTemptingBounce();

  const pill = document.getElementById("daily-challenge-pill");
  const icon = pill.querySelector(".daily-challenge-icon");

  // Calculate positions based on current pill width
  const collapsedPos = -(currentPillWidth - POSITIONS.tipVisible);
  const expandedPos = POSITIONS.expandedOffset;

  // Main slide animation
  const currentLeft = parseFloat(getComputedStyle(pill).left);
  currentAnimation = anime({
    targets: pill,
    left: [currentLeft, expandedPos],
    easing: "easeOutElastic(1, .6)",
    duration: ANIMATION_DURATIONS.slideIn,
    complete: () => {
      isAnimating = false;
      currentAnimation = null;
    },
  });

  // Icon bounce effect
  anime({
    targets: icon,
    scale: [1, 1.2, 1],
    rotate: [0, 10, 0],
    easing: "easeOutElastic(1, .8)",
    duration: ANIMATION_DURATIONS.iconBounce,
    delay: 200,
  });
}

function slideOutPill() {
  if (!isPillExpanded && !isAnimating) return;

  if (currentAnimation) {
    currentAnimation.pause();
  }

  isAnimating = true;
  isPillExpanded = false;
  stopTemptingBounce();

  const pill = document.getElementById("daily-challenge-pill");
  const icon = pill.querySelector(".daily-challenge-icon");

  // Calculate positions based on current pill width
  const collapsedPos = -(currentPillWidth - POSITIONS.tipVisible);
  const expandedPos = POSITIONS.expandedOffset;

  // Main slide animation
  const currentLeft = parseFloat(getComputedStyle(pill).left);
  currentAnimation = anime({
    targets: pill,
    left: [currentLeft, collapsedPos],
    easing: "easeInElastic(1, .6)",
    duration: ANIMATION_DURATIONS.slideOut,
    delay: 50,
    complete: () => {
      isAnimating = false;
      currentAnimation = null;
    },
  });

  // Icon scale effect
  anime({
    targets: icon,
    scale: [1, 0.8, 1],
    easing: "easeOutQuad",
    duration: 300,
  });
}

function addEntranceAnimation() {
  const pill = document.getElementById("daily-challenge-pill");
  if (!pill) return;

  const startPos = -(currentPillWidth + 20);
  const endPos = -(currentPillWidth - POSITIONS.tipVisible);

  anime({
    targets: pill,
    left: [startPos, endPos],
    opacity: [0, 1],
    easing: "easeOutElastic(1, .6)",
    duration: ANIMATION_DURATIONS.entrance,
    delay: POSITIONS.entranceDelay,
    complete: () => {
      startTemptingBounce();
    },
  });
}

/**
 * Tempting Bounce Functions
 */
function startTemptingBounce() {
  const pill = document.getElementById("daily-challenge-pill");
  if (!pill || hasUserInteracted || hasCompletedToday()) return;

  temptingBounceTimeout = setTimeout(() => {
    if (!hasUserInteracted && !isPillExpanded) {
      pill.classList.add("tempting");
    }
  }, POSITIONS.temptingDelay);
}

function stopTemptingBounce() {
  if (hasUserInteracted) return;

  hasUserInteracted = true;
  const pill = document.getElementById("daily-challenge-pill");
  if (pill) {
    pill.classList.remove("tempting");
  }
  if (temptingBounceTimeout) {
    clearTimeout(temptingBounceTimeout);
  }
}

/**
 * Event Handlers
 */
function setupPillInteractions() {
  const pill = document.getElementById("daily-challenge-pill");
  if (!pill || interactionsSetup) return;

  interactionsSetup = true;

  const isMobile = "ontouchstart" in window || navigator.maxTouchPoints > 0;

  if (isMobile) {
    // Mobile: Click to toggle
    pill.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      stopTemptingBounce();

      if (isPillExpanded || isAnimating) {
        slideOutPill();
      } else {
        slideInPill();
      }
    });

    // Close on any touch anywhere
    document.addEventListener("touchstart", () => {
      setTimeout(() => {
        if (isPillExpanded || isAnimating) {
          slideOutPill();
        }
      }, 10);
    });
  } else {
    // Desktop: Hover to expand
    pill.addEventListener("mouseenter", () => {
      stopTemptingBounce();
      slideInPill();
    });

    pill.addEventListener("mouseleave", slideOutPill);
  }
}

/**
 * Content Management
 */
function updateDailyChallenge() {
  const dailyStreak = getDailyStreak();
  setGoldenGuessButton(dailyStreak);

  const pillEl = document.getElementById("daily-challenge-pill");
  const countryNameEl = document.getElementById("daily-country-name");
  if (!pillEl || !countryNameEl) return;

  let displayText = "";
  const icon = pillEl.querySelector(".daily-challenge-icon");

  if (hasCompletedToday()) {
    stopTemptingBounce();

    if (icon) {
      if (dailyStreak > 1) {
        icon.textContent = "🔥"; // Fire for streak
      } else {
        icon.textContent = "🎉"; // Party for completing today
      }
    }
    displayText = dailyCountry;
    countryNameEl.style.textDecoration = "line-through";
  } else {
    // Active challenge
    if (icon) icon.textContent = "🏆";
    displayText = dailyCountry;
    countryNameEl.style.textDecoration = "none";
  }

  countryNameEl.textContent = displayText;

  // Update pill dimensions
  // if (displayText) {
  const newWidth = calculatePillWidth(displayText);
  updatePillDimensions(newWidth);
  // }
}

function onDailyChallengeSuccess() {
  const today = getTodayISO();
  const history = getHistory();
  history[today] = dailyCountry;
  setHistory(history);

  // Stop tempting bounce since challenge is completed
  stopTemptingBounce();

  const pill = document.getElementById("daily-challenge-pill");
  const icon = pill.querySelector(".daily-challenge-icon");

  if (pill && icon) {
    // Success animation
    anime({
      targets: pill,
      scale: [1, 1.3, 1.1],
      easing: "easeOutElastic(1, .8)",
      duration: 1000,
      complete: updateDailyChallenge,
    });

    // Icon celebration
    anime({
      targets: icon,
      scale: [1, 1.5, 1],
      rotate: [0, 360, 0],
      easing: "easeOutElastic(1, .6)",
      duration: 800,
    });
  } else {
    updateDailyChallenge();
  }

  const dailyStreak = getDailyStreak();
  return dailyStreak;
}

function setGoldenGuessButton(dailyStreak) {
  if (dailyStreak < goldenStreak) return;

  const button = document.getElementById("left-btn");
  if (button) {
    button.classList.add("golden");
  }
}

/**
 * Public API
 */
export function checkDailyChallenge(selectedCountry) {
  if (!dailyCountry || !selectedCountry || hasCompletedToday()) {
    return { challengeCompleted: false, streak: null };
  }

  if (selectedCountry.toLowerCase() === dailyCountry.toLowerCase()) {
    const newDailyStreak = onDailyChallengeSuccess();
    return { challengeCompleted: true, streak: newDailyStreak };
  }

  return { challengeCompleted: false, streak: null };
}

/**
 * Get daily country from history or fetch it from the server
 */
async function resolveDailyCountry() {
  if (hasCompletedToday()) {
    const history = getHistory();
    const fromHistory = history[getTodayISO()] || null;
    if (fromHistory) return fromHistory;
  }

  try {
    const response = await fetch("/daily_country");

    if (!response.ok) return null;

    const data = await response.json();
    if (data.error || !data.country) return null;

    return data.country;
  } catch (err) {
    return null;
  }
}

/**
 * Initialization
 */
document.addEventListener("DOMContentLoaded", async function () {
  dailyCountry = await resolveDailyCountry();

  if (dailyCountry) {
    setupPillInteractions();
    updateDailyChallenge();
    addEntranceAnimation();
  }
});
