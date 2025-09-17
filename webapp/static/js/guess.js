import { lines, clearCanvas } from "./drawing.js";
import * as msg from "./messages.js";
import { checkDailyChallenge } from "./daily_challenge.js";

document
  .getElementById("refresh-btn")
  .addEventListener("click", refreshDrawing);

const guessButton = document.getElementById("guess-btn");
guessButton.addEventListener("click", handleButtonClick);

let isInConfirmMode = false;

function refreshDrawing() {
  clearCanvas();
  showGuessMessage("");

  if (isInConfirmMode) {
    hideConfirmation();
    if (window.currentDrawingId) {
      // Send feedback with null to cleanup without saving
      sendFeedback(null);
      window.currentDrawingId = null;
    }
  }

  // Unlock guess button
  const guessBtn = document.getElementById("guess-btn");
  guessBtn.style = "";
}

function handleButtonClick() {
  if (isInConfirmMode) {
    confirmCountry();
  } else {
    guess();
  }
}

function showGuessMessage(message) {
  document.getElementById("guess-message").innerText = message;
}

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

async function guess() {
  if (lines.length === 0) {
    const emptyGuessMessage = msg.getEmptyGuessMessage();
    showGuessMessage(emptyGuessMessage);
    return;
  }

  document.getElementById("guess-message").innerHTML =
    `<div class="loader mx-auto"></div>`;
  try {
    const data = await postGuess(lines);
    const ranking = data.ranking;
    const firstCountry = ranking.countries[0];
    const firstScore = ranking.scores[0];

    const message = msg.getConfidenceBasedMessage(firstScore, firstCountry);
    showGuessMessage(message);
    window.currentDrawingId = data.drawing_id;
    showConfirmation(ranking);
  } catch (error) {
    console.error("Error:", error);

    // Provide user feedback of the error
    let message = "An unexpected error occurred.";
    if (error.message === "Server unreachable") {
      message = "Could not reach the ML server.";
    } else if (error.message === "Server error") {
      message = "There was an error with the ML server response.";
    }
    showGuessMessage(message);
  }
}

function showConfirmation(ranking) {
  const confirmationContainer = document.getElementById(
    "confirmation-container",
  );
  const guessBtn = document.getElementById("guess-btn");
  confirmationContainer.style.display = "block";

  // Update button text and function
  guessBtn.textContent = "Confirm";
  isInConfirmMode = true;

  var dropdown = document.getElementById("country-dropdown");
  dropdown.innerHTML = ""; // Clear previous options

  // Add options for each country in the list
  ranking.countries.forEach((country, index) => {
    var option = document.createElement("option");
    option.value = country;
    option.text = `${country}${"\u00A0".repeat(4)}${(ranking.scores[index] * 100).toFixed(1)}%`;
    dropdown.add(option);
  });

  // Add "Other country" option
  var otherOption = document.createElement("option");
  otherOption.value = "Other";
  otherOption.text = "Other country";
  dropdown.add(otherOption);

  // Show the instruction message
  document.getElementById("instruction-message").style.display = "block";
}

function hideConfirmation() {
  const confirmationContainer = document.getElementById(
    "confirmation-container",
  );
  const guessBtn = document.getElementById("guess-btn");
  confirmationContainer.style.display = "none";

  // Reset button text and function
  guessBtn.textContent = "Guess Country";
  isInConfirmMode = false;
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

    if (checkDailyChallenge(selectedCountry)) {
      return msg.getDailyChallengeMessage(selectedCountry);
    }

    return msg.getCorrectGuessMessage(selectedCountry);
  }

  // Guess country is incorrect
  return msg.getIncorrectGuessMessage(selectedCountry, guessedCountry);
}

function confirmCountry() {
  const dropdown = document.getElementById("country-dropdown");
  const selectedCountry = dropdown.value;
  const guessedCountry = dropdown.options[0].value;

  const message = getConfirmationMessage(selectedCountry, guessedCountry);
  showGuessMessage(message);
  hideConfirmation();

  // Send feedback with country name
  if (window.currentDrawingId) {
    sendFeedback(selectedCountry);
    window.currentDrawingId = null;

    // Lock guess button after confirmation
    const guessBtn = document.getElementById("guess-btn");
    guessBtn.style.opacity = "0.5";
    guessBtn.style.backgroundColor = "#3f3f46";
    guessBtn.style.cursor = "not-allowed";
    guessBtn.style.pointerEvents = "none";
  }
}

async function sendFeedback(countryName) {
  try {
    const response = await fetch("/feedback", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        country: countryName,
        drawing_id: window.currentDrawingId,
      }),
    });
    const data = await response.json();
    console.log(data.message);
  } catch (error) {
    console.error("Error:", error);
  }
}

export { hideConfirmation };
