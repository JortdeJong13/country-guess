import { lines } from "./drawing.js";
import * as msg from "./messages.js";

/**
 * UI Helper Functions
 */
const dropdown = document.getElementById("country-dropdown");
const confirmationContainer = document.getElementById("confirmation-container");
const authorInput = document.getElementById("author-input");

function hideConfirmationContainer() {
  confirmationContainer.style.display = "none";
}

function showConfirmationContainer() {
  confirmationContainer.style.display = "block";
}

function resizeAuthorInput() {
  const charWidth = 10;
  const minWidth = 80;
  const contentWidth = (authorInput.value.length + 1) * charWidth;
  const finalWidth = Math.max(minWidth, contentWidth);
  authorInput.style.width = finalWidth + "px";
}

function formatScorePercent(score) {
  const percent = score * 100;
  if (percent < 1 && percent > 0) {
    return "<1%";
  }
  return `${percent.toFixed(0)}%`;
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

async function sendFeedback(drawingId, countryName, author) {
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
      body: JSON.stringify({
        drawing_id: drawingId,
        country: countryName,
        author: author || "",
      }),
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
function saveAuthorInput(author) {
  const value = author.trim();
  if (!value) return;
  localStorage.setItem("author", value);
}

function loadAuthorInput() {
  const saved = localStorage.getItem("author");
  if (!saved) return;

  authorInput.value = saved;
  resizeAuthorInput();
}

export async function guess() {
  if (lines.length === 0) {
    msg.setEmptyGuessMessage();
    return false;
  }

  msg.showLoadingMessage();

  try {
    const data = await postGuess(lines);
    const ranking = data.ranking;

    msg.setConfidenceBasedMessage(...ranking[0]);
    window.currentDrawingId = data.drawing_id;

    populateCountryDropdown(ranking);
    loadAuthorInput();
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
    msg.showMessage(message);
    return false;
  }
}

function populateCountryDropdown(ranking) {
  dropdown.innerHTML = ""; // Clear previous options

  ranking.forEach(([country, score]) => {
    const option = document.createElement("option");
    option.value = country;
    option.text = `${country}${"\u00A0".repeat(4)}${formatScorePercent(score)}`;
    dropdown.add(option);
  });

  // Add "Other country" option
  const otherOption = document.createElement("option");
  otherOption.value = "Other";
  otherOption.text = "Other country";
  dropdown.add(otherOption);
}

function setConfirmationMessage(selectedCountry, guessedCountry) {
  // If incorrect, handle and exit early
  if (selectedCountry !== guessedCountry) {
    msg.setIncorrectGuessMessage(selectedCountry, guessedCountry);
    return;
  }

  // Correct guess → trigger confetti
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

  msg.setCorrectGuessMessage(selectedCountry);
}

export function confirmCountry() {
  const selectedCountry = dropdown.value;
  const guessedCountry = dropdown.options[0].value;
  const author = authorInput ? authorInput.value : "";

  saveAuthorInput(author);

  setConfirmationMessage(selectedCountry, guessedCountry);
  hideConfirmationContainer();

  // Send feedback
  if (window.currentDrawingId) {
    const drawingId = window.currentDrawingId;
    sendFeedback(drawingId, selectedCountry, author);
    window.currentDrawingId = null;
  }
}

export function refreshGuess() {
  msg.showMessage("");
  window.currentDrawingId = null;
  hideConfirmationContainer();
}

// Adjust author input field width based on input
document.addEventListener("DOMContentLoaded", () => {
  authorInput.style.width = "120px"; // initial width
  authorInput.addEventListener("input", () => resizeAuthorInput());
});
