let dailyCountry = null;
let isAnimating = false;
let isPillExpanded = false;
let currentAnimation = null;
let interactionsSetup = false;
let hasUserInteracted = false;
let temptingBounceTimeout = null;
let currentPillWidth = 200;

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

// Calculate pill width based on country name length
function calculatePillWidth(countryName) {
  if (!countryName) return 200;

  // Create temporary element to measure text width
  const temp = document.createElement("div");
  temp.style.position = "absolute";
  temp.style.visibility = "hidden";
  temp.style.fontSize = "13px";
  temp.style.fontWeight = "700";
  temp.style.fontFamily = "sans-serif";
  temp.innerHTML = countryName;
  document.body.appendChild(temp);

  const textWidth = temp.offsetWidth;
  document.body.removeChild(temp);

  // Calculate total width: left padding (50px) + text width + icon space (30px) + extra buffer (20px)
  const totalWidth = Math.max(260, textWidth + 100); // Minimum 260px (wider), reduced spacing
  const maxWidth = Math.min(totalWidth, 450); // Maximum 450px

  return maxWidth;
}

// Update pill dimensions
function updatePillDimensions(width) {
  const pill = document.getElementById("daily-challenge-pill");
  if (!pill) return;

  pill.style.width = width + "px";
  currentPillWidth = width;

  // Don't modify content width - let it be flexible to avoid affecting icon position
}

// Animation functions
function slideInPill() {
  if (isPillExpanded) return;

  // Stop any current animation
  if (currentAnimation) {
    currentAnimation.pause();
  }

  isAnimating = true;
  isPillExpanded = true;

  const pill = document.getElementById("daily-challenge-pill");
  const icon = pill.querySelector(".daily-challenge-icon");

  // Main slide-in animation with bounce
  // TWEAK PARAMETERS:
  // left: [collapsed_position, expanded_position] - how far it slides
  // easing: bounce strength - lower = gentler (try .2-.8)
  // duration: animation speed in ms (try 600-1200)
  // Calculate positions based on pill width
  const collapsedPos = -(currentPillWidth - 50); // Show 50px tip (enough for full icon)
  const expandedPos = -70; // Further reduced expansion to prevent left edge with long names

  currentAnimation = anime({
    targets: pill,
    left: [collapsedPos, expandedPos], // 🔧 SLIDE DISTANCE: Dynamic based on pill width
    easing: "easeOutElastic(1, .6)", // 🔧 BOUNCY: Elastic bounce for more playfulness
    duration: 1200, // 🔧 SPEED: Slower animation for smoother feel
    complete: () => {
      isAnimating = false;
      currentAnimation = null;
    },
  });

  // Stop tempting bounce on first interaction
  stopTemptingBounce();

  // Icon bounce effect
  // 🔧 ICON ANIMATION: Make icon more/less playful
  anime({
    targets: icon,
    scale: [1, 1.2, 1], // 🔧 ICON SIZE: Change 1.2 for bigger/smaller bounce
    rotate: [0, 10, 0], // 🔧 ICON ROTATION: Change 10 for more/less rotation
    easing: "easeOutElastic(1, .8)", // 🔧 ICON BOUNCE: Change .8 for bounce strength
    duration: 600, // 🔧 ICON SPEED: How long icon animation lasts
    delay: 200, // 🔧 ICON DELAY: When icon animation starts
  });
}

function slideOutPill() {
  if (!isPillExpanded && !isAnimating) return;

  // Stop any current animation
  if (currentAnimation) {
    currentAnimation.pause();
  }

  isAnimating = true;
  isPillExpanded = false;

  const pill = document.getElementById("daily-challenge-pill");
  const content = pill.querySelector(".daily-challenge-content");
  const icon = pill.querySelector(".daily-challenge-icon");

  // Content stays visible during slide-out (no fade needed)

  // Main slide-out animation
  // TWEAK PARAMETERS:
  // left: [expanded_position, collapsed_position] - must match slide-in values
  // duration: how fast it slides back (try 200-500)
  // delay: pause before sliding back (try 0-100)
  // Calculate positions based on pill width
  const collapsedPos = -(currentPillWidth - 50); // Show 50px tip (enough for full icon)
  const expandedPos = -70; // Further reduced expansion to prevent left edge with long names

  currentAnimation = anime({
    targets: pill,
    left: [expandedPos, collapsedPos], // 🔧 SLIDE DISTANCE: Dynamic based on pill width
    easing: "easeInElastic(1, .6)", // 🔧 SLIDE-OUT STYLE: Match slide-in elastic bounce
    duration: 600, // 🔧 SLIDE-OUT SPEED: Slower for smoother feel
    delay: 50, // 🔧 DELAY: Pause before sliding back (0=immediate, 100=longer pause)
    complete: () => {
      isAnimating = false;
      currentAnimation = null;
    },
  });

  // Icon scale down
  anime({
    targets: icon,
    scale: [1, 0.8, 1],
    easing: "easeOutQuad",
    duration: 300,
  });

  // Stop tempting bounce on first interaction
  stopTemptingBounce();
}

// Device detection
function isTouchDevice() {
  return "ontouchstart" in window || navigator.maxTouchPoints > 0;
}

function startTemptingBounce() {
  const pill = document.getElementById("daily-challenge-pill");
  if (!pill || hasUserInteracted || hasCompletedToday()) return;

  // Start bounce after entrance animation completes
  temptingBounceTimeout = setTimeout(() => {
    if (!hasUserInteracted && !isPillExpanded) {
      pill.classList.add("tempting");
      console.log("Started tempting bounce");
    }
  }, 1500);
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
  console.log("Stopped tempting bounce - user interacted");
}

function setupPillInteractions() {
  const pill = document.getElementById("daily-challenge-pill");
  if (!pill || interactionsSetup) return;

  interactionsSetup = true;
  console.log("Setting up pill interactions, isTouchDevice:", isTouchDevice());

  if (isTouchDevice()) {
    // Mobile: Simple click handler
    pill.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      console.log("Pill clicked, isPillExpanded:", isPillExpanded);

      // Stop tempting bounce on first interaction
      stopTemptingBounce();

      if (isPillExpanded || isAnimating) {
        slideOutPill();
      } else {
        slideInPill();
      }
    });

    // Close on ANY touch anywhere on screen
    document.addEventListener("touchstart", (e) => {
      // Small delay to prevent immediate closing when opening
      setTimeout(() => {
        if (isPillExpanded || isAnimating) {
          console.log("Closing pill - touched anywhere");
          slideOutPill();
        }
      }, 10);
    });
  } else {
    // Desktop: Hover behavior with interruption handling
    pill.addEventListener("mouseenter", () => {
      stopTemptingBounce();
      slideInPill();
    });

    pill.addEventListener("mouseleave", () => {
      slideOutPill();
    });
  }
}

function updateDailyChallenge() {
  const pillEl = document.getElementById("daily-challenge-pill");
  const countryNameEl = document.getElementById("daily-country-name");
  if (!pillEl || !countryNameEl) return;

  let displayText = "";

  // Always show the pill, just update the content
  if (hasCompletedToday()) {
    // Show completed state
    const icon = pillEl.querySelector(".daily-challenge-icon");
    if (icon) {
      icon.textContent = "✅";
    }
    displayText = "Completed!";
    countryNameEl.textContent = displayText;
  } else if (dailyCountry) {
    // Show active challenge
    const icon = pillEl.querySelector(".daily-challenge-icon");
    if (icon) {
      icon.textContent = "🏆";
    }
    displayText = dailyCountry;
    countryNameEl.textContent = displayText;
  } else {
    // Loading or no challenge
    displayText = "Loading...";
    countryNameEl.textContent = displayText;
  }

  // Always update pill width based on current display text (before any animations)
  if (displayText) {
    const newWidth = calculatePillWidth(displayText);
    updatePillDimensions(newWidth);
  }
}

// Fetch and store the daily country on load
document.addEventListener("DOMContentLoaded", async function () {
  // Always show and set up the pill
  updateDailyChallenge();

  // If already completed, no need to fetch
  if (hasCompletedToday()) {
    // Add entrance animation with current dimensions
    addEntranceAnimation();
    return;
  }

  try {
    const response = await fetch("/daily_country");
    const data = await response.json();
    dailyCountry = data.country || null;
    dailyCountry = "South Georgia & the South Sandwich Islands";

    // Update pill content and dimensions FIRST
    updateDailyChallenge();

    // THEN add entrance animation with correct dimensions
    addEntranceAnimation();
  } catch {
    dailyCountry = null;
    updateDailyChallenge();
    addEntranceAnimation();
  }
});

// Separate function for entrance animation
function addEntranceAnimation() {
  const pill = document.getElementById("daily-challenge-pill");
  if (pill) {
    // Calculate entrance positions based on current pill width
    const startPos = -(currentPillWidth + 20); // Completely off-screen
    const endPos = -(currentPillWidth - 50); // Show 50px tip (enough for full icon)

    anime({
      targets: pill,
      left: [startPos, endPos],
      opacity: [0, 1],
      easing: "easeOutElastic(1, .6)",
      duration: 800,
      delay: 500,
      complete: () => {
        // Ensure interactions work after entrance animation
        console.log("Entrance animation complete, setting up interactions");
        if (!interactionsSetup) {
          setupPillInteractions();
        }

        // Start tempting bounce for new users
        startTemptingBounce();
      },
    });
  }
}

// Handle daily challenge success
function onDailyChallengeSuccess() {
  const today = getTodayISO();
  const history = getHistory();
  history[today] = dailyCountry;
  setHistory(history);

  // Success animation
  const pill = document.getElementById("daily-challenge-pill");
  const icon = pill.querySelector(".daily-challenge-icon");

  if (pill && icon) {
    // Success bounce animation with color change
    anime({
      targets: pill,
      scale: [1, 1.3, 1.1],
      easing: "easeOutElastic(1, .8)",
      duration: 1000,
      complete: () => {
        // Update to completed state
        updateDailyChallenge();
      },
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
}

// Check if a guess matches the daily country
export function checkDailyChallenge(guessedCountry) {
  if (!dailyCountry || !guessedCountry || hasCompletedToday()) return false;

  // Daily challenge completed
  if (guessedCountry.toLowerCase() === dailyCountry.toLowerCase()) {
    onDailyChallengeSuccess();
    return true;
  }
  return false;
}
