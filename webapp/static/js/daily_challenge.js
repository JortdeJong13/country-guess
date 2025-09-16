let dailyCountry = null;

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

function updateDailyChallenge() {
  const challengeEl = document.getElementById("daily-challenge");
  if (!challengeEl) return;

  if (hasCompletedToday()) {
    challengeEl.classList.remove("text-amber-500");
    challengeEl.classList.add("text-green-500");
    challengeEl.innerHTML = `✅ Daily Challenge Completed!`;
  } else {
    challengeEl.classList.remove("text-green-500");
    challengeEl.classList.add("text-amber-500");
    challengeEl.innerHTML = `
      Daily Challenge:
      <span id="daily-country-name">${dailyCountry || ""}</span>
    `;
  }
}

// Fetch and store the daily country on load
document.addEventListener("DOMContentLoaded", async function () {
  try {
    const response = await fetch("/daily_country");
    const data = await response.json();
    dailyCountry = data.country || null;
    updateDailyChallenge();
  } catch {
    dailyCountry = null;
    updateDailyChallenge();
  }
});

// Handle daily challenge success
function onDailyChallengeSuccess() {
  const today = getTodayISO();
  const history = getHistory();
  history[today] = dailyCountry;
  setHistory(history);
  updateDailyChallenge();
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
