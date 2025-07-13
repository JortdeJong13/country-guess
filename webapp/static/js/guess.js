import { lines, clearCanvas } from "./drawing.js";
import * as msg from "./messages.js";

document
  .getElementById("refresh-btn")
  .addEventListener("click", refreshDrawing);

const guessButton = document.getElementById("guess-btn");
guessButton.addEventListener("click", handleButtonClick);

let isInConfirmMode = false;
let emptyGuessCounter = 0;

function refreshDrawing() {
  clearCanvas();
  document.getElementById("guess-message").innerText = "";

  emptyGuessCounter = 0;
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

function guess() {
  if (lines.length > 0) {
    fetch("/guess", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ lines: lines }),
    })
      .then((response) => {
        if (!response.ok) {
          return response.json().then((errorData) => {
            throw new Error(errorData.message || "Unknown server error");
          });
        }
        return response.json();
      })
      .then((data) => {
        const ranking = data.ranking;
        const firstCountry = ranking.countries[0];
        const firstScore = ranking.scores[0];

        const message = msg.getConfidenceBasedMessage(firstScore, firstCountry);
        document.getElementById("guess-message").innerText = message;
        window.currentDrawingId = data.drawing_id;
        showConfirmation(ranking);
      })
      .catch((error) => {
        console.error("Error:", error);

        // Provide user feedback of the error
        let userMessage = "";
        if (error.message === "Server unreachable") {
          userMessage = "Could not reach the ML server.";
        } else if (error.message === "Server error") {
          userMessage = "There was an error with the ML server response.";
        } else {
          userMessage = "An unexpected error occurred.";
        }

        document.getElementById("guess-message").innerText = userMessage;
      });
  } else {
    console.log(
      "Coordinates list is empty, please draw something before guessing",
    );
    emptyGuessCounter++;

    if (emptyGuessCounter === 13) {
      // Display a random easter egg message
      const randomMessage = msg.getEasterEggMessage();
      document.getElementById("guess-message").innerText = randomMessage;
    } else {
      document.getElementById("guess-message").innerText =
        "You first need to draw a country";
    }
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

function confirmCountry() {
  var dropdown = document.getElementById("country-dropdown");
  var selectedCountry = dropdown.value;
  const guessedCountry = dropdown.options[0].value;

  let message;

  if (selectedCountry === "Other") {
    message = "I thought I knew all the countries... I guess not!";
  } else if (selectedCountry === guessedCountry) {
    message =
      msg.getRandomMessage(
        msg.correctMessages,
        selectedCountry,
        guessedCountry,
      ) + msg.getCountryFacts(selectedCountry);

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
  } else {
    message = msg.getRandomMessage(
      msg.incorrectMessages,
      guessedCountry,
      selectedCountry,
    );
  }

  document.getElementById("guess-message").innerText = message;

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
