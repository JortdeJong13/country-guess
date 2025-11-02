import { lines, clearCanvas, renderUserDrawing } from "./drawing.js";
import * as msg from "./messages.js";
import { checkDailyChallenge } from "./daily_challenge.js";

document
  .getElementById("refresh-btn")
  .addEventListener("click", refreshDrawing);

const leftBtn = document.getElementById("left-btn");
leftBtn.addEventListener("click", handleButtonClick);

let inConfirmMode = false;
let inShowDrawingMode = false;

function setLeftBtnState(state) {
  // state: "guess", "confirm", "next"
  leftBtn.textContent =
    state === "guess"
      ? "Guess Country"
      : state === "confirm"
        ? "Confirm"
        : state === "next"
          ? "Next Country"
          : leftBtn.textContent;

  // Update flags
  inConfirmMode = state === "confirm";
  inShowDrawingMode = state === "next";
}

const archivesBtn = document.getElementById("archives-btn");
archivesBtn.addEventListener("click", function () {
  showArchives();
});

function hideConfirmationContainer() {
  const confirmationContainer = document.getElementById(
    "confirmation-container",
  );
  confirmationContainer.style.display = "none";
}

function refreshDrawing() {
  clearCanvas();
  showGuessMessage("");
  window.currentDrawingId = null;
  hideConfirmationContainer();
  // Show archives button
  const archivesBtnContainer =
    document.getElementById("archives-btn").parentElement;
  archivesBtnContainer.style.display = "block";
  setLeftBtnState("guess");
  leftBtn.classList.remove("guess-locked");
}

function handleButtonClick() {
  if (inShowDrawingMode) {
    showArchives();
  } else if (inConfirmMode) {
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
  // Hide the archives button
  const archivesBtnContainer =
    document.getElementById("archives-btn").parentElement;
  archivesBtnContainer.style.display = "none";

  const confirmationContainer = document.getElementById(
    "confirmation-container",
  );
  confirmationContainer.style.display = "block";
  setLeftBtnState("confirm");

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
      return msg.getDailyChallengeMessage(
        selectedCountry,
        dailyChallenge.streak,
      );
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
  hideConfirmationContainer();

  // Send feedback
  if (window.currentDrawingId) {
    // Capture the drawing ID locally before sending
    const drawingId = window.currentDrawingId;
    sendFeedback(selectedCountry, drawingId);

    // Clear the global drawing ID
    window.currentDrawingId = null;

    // Lock the guess button
    leftBtn.classList.add("guess-locked");
  }
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

async function showArchives() {
  try {
    const response = await fetch("/drawing");
    if (!response.ok) {
      throw new Error("Failed to fetch drawing");
    }
    const data = await response.json();
    renderUserDrawing(data.lines);
    setLeftBtnState("next");
    const archiveMessage = msg.getArchiveMessage(
      data.guess_score,
      data.country_name,
      data.country_guess,
    );
    showGuessMessage(archiveMessage);
  } catch (error) {
    console.error("Error fetching user drawing:", error);
  }
}
