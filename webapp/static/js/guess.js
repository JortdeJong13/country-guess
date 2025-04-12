import { lines, clearCanvas } from "./drawing.js";

const initialGuessMessages = [
  "It looks like you tried to draw {{guessed}}.",
  "I think this might be {{guessed}}. Am I right?",
  "Hmm, this looks like {{guessed}} to me.",
  "If I had to guess, I’d say this is {{guessed}}.",
  "This reminds me of {{guessed}}.",
  "Could it be {{guessed}}? That’s what I’m seeing.",
  "It’s giving me {{guessed}} vibes. Am I right?",
  "My best guess? I'm going with {{guessed}}.",
];

const correctMessages = [
  "I know a {{selected}} when I see one!",
  "Phew! Wasn't sure about that one.",
  "No doubt about it, that’s {{selected}}. Good job!",
  "Ah, {{selected}}! I recognized that right away.",
  "Great drawing! I knew it was {{selected}} instantly.",
  "That’s a perfect {{selected}}! You nailed it.",
  "Well done! I couldn’t miss {{selected}} if I tried.",
  "It’s {{selected}}! Excellent work from both of us!",
];

const incorrectMessages = [
  "Congratulations! Your drawing of {{selected}} has been added to the test set!",
  "Not quite—I went with {{guessed}}, but it’s really {{selected}}.",
  "Well, I gave it my best shot! I thought it was {{guessed}}, but it’s {{selected}}.",
  "Ah, I see it now—it’s {{selected}}, not {{guessed}}. Thanks for clarifying!",
  "I’m not saying you can't draw, but {{guessed}} was way closer than {{selected}}.",
  "Interesting! I guessed {{guessed}}, but you drew {{selected}}.",
  "{{selected}}... really? Might want to work on your drawing skills!",
  "I’ll admit it—I was wrong. This is {{selected}}, not {{guessed}}.",
  "Oops! I thought {{guessed}}, but you drew a great {{selected}}.",
];

function getRandomMessage(messageList, guessedCountry, selectedCountry) {
  // Pick a random message from the list
  const randomIndex = Math.floor(Math.random() * messageList.length);
  const template = messageList[randomIndex];

  // Replace placeholders with the given values
  return template
    .replace("{{selected}}", selectedCountry)
    .replace("{{guessed}}", guessedCountry);
}

document
  .getElementById("refresh-btn")
  .addEventListener("click", refreshDrawing);

function refreshDrawing() {
  clearCanvas();
  document.getElementById("guess-message").innerText = "";

  //
  if (isInConfirmMode) {
    hideConfirmation();
    // Send feedback with null to cleanup without saving
    if (window.currentDrawingId) {
      sendFeedback(null);
      window.currentDrawingId = null;
    }
  }
}

const guessButton = document.getElementById("guess-btn");
guessButton.addEventListener("click", handleButtonClick);

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
        const message = getRandomMessage(
          initialGuessMessages,
          data.ranking[0],
          data.ranking[0],
        );
        document.getElementById("guess-message").innerText = message;
        window.currentDrawingId = data.drawing_id;
        showConfirmation(data.ranking);
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
    document.getElementById("guess-message").innerText =
      "You first need to draw a country";
  }
}

let isInConfirmMode = false;

function handleButtonClick() {
  if (isInConfirmMode) {
    confirmCountry();
  } else {
    guess();
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
  ranking.forEach(function (country) {
    var option = document.createElement("option");
    option.text = country;
    dropdown.add(option);
  });

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
  var selectedCountry = document.getElementById("country-dropdown").value;
  const guessedCountry =
    document.getElementById("country-dropdown").options[0].value;

  if (selectedCountry === guessedCountry) {
    const message = getRandomMessage(
      correctMessages,
      selectedCountry,
      guessedCountry,
    );
    document.getElementById("guess-message").innerText = message;

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
    const message = getRandomMessage(
      incorrectMessages,
      guessedCountry,
      selectedCountry,
    );
    document.getElementById("guess-message").innerText = message;
  }

  hideConfirmation();

  // Send feedback with country name
  if (window.currentDrawingId) {
    sendFeedback(selectedCountry);
    window.currentDrawingId = null;
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
