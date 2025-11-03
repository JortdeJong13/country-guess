import { lines, clearCanvas } from "./drawing.js";
import * as msg from "./messages.js";
import { checkDailyChallenge } from "./daily_challenge.js";

/**
 * UI Helper Functions
 */
function showMessage(message) {
  document.getElementById("message").innerText = message;
}

function showLoadingMessage() {
  document.getElementById("message").innerHTML =
    `<div class="loader mx-auto"></div>`;
}

function hideConfirmationContainer() {
  const confirmationContainer = document.getElementById(
    "confirmation-container",
  );
  confirmationContainer.style.display = "none";
}

function showConfirmationContainer() {
  const confirmationContainer = document.getElementById(
    "confirmation-container",
  );
  confirmationContainer.style.display = "block";
  document.getElementById("instruction-message").style.display = "block";
}

/**
 * API Functions
 */
async function postGuess(lines) {
  const response = await fetch("/guess", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lines }),
  });
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.message || "Unknown server error");
  }
  return response.json();
}

async function sendFeedback(countryName, drawingId) {
  if (!countryName || !drawingId) {
    console.warn("Missing country name or drawing ID. Feedback not sent.");
    return;
  }

  try {
    const response = await fetch("/feedback", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ country: countryName, drawing_id: drawingId }),
    });

    if (!response.ok) {
      throw new Error(`Server responded with status ${response.status}`);
    }

    const data = await response.json();
    console.log("Feedback sent successfully:", data);
    return data;
  } catch (error) {
    console.error("Failed to send feedback:", error);
    return null;
  }
}

/**
 * Core Guess Logic
 */
export async function guess() {
  if (lines.length === 0) {
    msg.setEmptyGuessMessage();
    return false;
  }

  showLoadingMessage();

  try {
    const data = await postGuess(lines);
    const ranking = data.ranking;
    const firstCountry = ranking.countries[0];
    const firstScore = ranking.scores[0];

    msg.setConfidenceBasedMessage(firstScore, firstCountry);
    window.currentDrawingId = data.drawing_id;

    populateCountryDropdown(ranking);
    showConfirmationContainer();

    return true; // Success - state should change to "confirm"
  } catch (error) {
    console.error("Error:", error);

    let message = "An unexpected error occurred.";
    if (error.message === "Server unreachable") {
      message = "Could not reach the ML server.";
    } else if (error.message === "Server error") {
      message = "There was an error with the ML server response.";
    }
    showMessage(message);
    return false;
  }
}

function populateCountryDropdown(ranking) {
  const dropdown = document.getElementById("country-dropdown");
  dropdown.innerHTML = ""; // Clear previous options

  // Add options for each country in the list
  ranking.countries.forEach((country, index) => {
    const option = document.createElement("option");
    option.value = country;
    option.text = `${country}${"\u00A0".repeat(4)}${(ranking.scores[index] * 100).toFixed(1)}%`;
    dropdown.add(option);
  });

  // Add "Other country" option
  const otherOption = document.createElement("option");
  otherOption.value = "Other";
  otherOption.text = "Other country";
  dropdown.add(otherOption);
}

function getConfirmationMessage(selectedCountry, guessedCountry) {
  // Guess country is correct
  if (selectedCountry === guessedCountry) {
    setTimeout(() => {
      confetti({
        particleCount: 150,
        spread: 70,
        startVelocity: 70,
        zIndex: 1000,
        origin: { y: 1, x: 0.5 },
        resize: true,
        useWorker: true,
        ticks: 280,
      });
    }, 50);

    const dailyChallenge = checkDailyChallenge(selectedCountry);
    if (dailyChallenge.challengeCompleted) {
      msg.setDailyChallengeMessage(selectedCountry, dailyChallenge.streak);
    }

    msg.setCorrectGuessMessage(selectedCountry);
  }

  // Guess country is incorrect
  msg.setIncorrectGuessMessage(selectedCountry, guessedCountry);
}

export function confirmCountry() {
  const dropdown = document.getElementById("country-dropdown");
  const selectedCountry = dropdown.value;
  const guessedCountry = dropdown.options[0].value;

  const message = getConfirmationMessage(selectedCountry, guessedCountry);
  showMessage(message);
  hideConfirmationContainer();

  // Send feedback
  if (window.currentDrawingId) {
    const drawingId = window.currentDrawingId;
    sendFeedback(selectedCountry, drawingId);
    window.currentDrawingId = null;
  }
}

export function refreshGuess() {
  showMessage("");
  window.currentDrawingId = null;
  hideConfirmationContainer();
}
