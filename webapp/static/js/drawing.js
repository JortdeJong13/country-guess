var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");

var isDrawing = false;
var lastX, lastY;
var lines = [];
var currentLine = [];

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);

const initialGuessMessages = [
  "It looks like you tried to draw {{guessed}}.",
  "I think this might be {{guessed}}. Am I right?",
  "Hmm, this looks like {{guessed}} to me.",
  "If I had to guess, I’d say this is {{guessed}}.",
  "Is this supposed to be {{guessed}}? That’s my guess!",
  "This reminds me of {{guessed}}.",
  "Could it be {{guessed}}? That’s what I’m seeing.",
  "I’m going with {{guessed}}...",
  "It’s giving me {{guessed}} vibes. Am I close?",
  "My best guess? Definitely {{guessed}}.",
];

const correctMessages = [
  "I know a {{selected}} when I see one!",
  "You’ve got it! That’s clearly {{selected}}—nice work!",
  "No doubt about it, that’s {{selected}}. Good job!",
  "Ah, {{selected}}! I recognized that right away.",
  "Great drawing! I knew it was {{selected}} instantly.",
  "You and I are in sync—that’s {{selected}}!",
  "I didn’t even have to think twice—it’s {{selected}}!",
  "That’s a perfect {{selected}}! You nailed it.",
  "Well done! I couldn’t miss {{selected}} if I tried.",
  "It’s {{selected}}! Excellent work from both of us!",
];

const incorrectMessages = [
  "Hmm, I guessed {{guessed}}, but it’s actually {{selected}}.",
  "Oops, I thought it was {{guessed}}. It’s {{selected}}—my bad!",
  "Not quite—I went with {{guessed}}, but it’s really {{selected}}.",
  "Close one! I guessed {{guessed}}, but this is actually {{selected}}.",
  "Well, I gave it my best shot! I thought it was {{guessed}}, but it’s {{selected}}.",
  "Ah, I see it now—it’s {{selected}}, not {{guessed}}. Thanks for clarifying!",
  "Interesting! I guessed {{guessed}}, but you drew {{selected}}.",
  "My guess was {{guessed}}, but you were aiming for {{selected}}. Almost there!",
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

function startDrawing(event) {
  isDrawing = true;
  var x = event.clientX - canvas.getBoundingClientRect().left;
  var y = event.clientY - canvas.getBoundingClientRect().top;
  [lastX, lastY] = [x, y];
  currentLine.push([x, y]);
}

function draw(event) {
  if (!isDrawing) return;

  var x = event.clientX - canvas.getBoundingClientRect().left;
  var y = event.clientY - canvas.getBoundingClientRect().top;

  // Check if distance is greater than a threshold
  var distance = Math.sqrt(Math.pow(x - lastX, 2) + Math.pow(y - lastY, 2));
  if (distance > 6) {
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();

    [lastX, lastY] = [x, y];
    currentLine.push([x, y]);
  }
}

function stopDrawing() {
  isDrawing = false;
  if (currentLine.length > 1) {
    lines.push(currentLine);
  }
  currentLine = [];
}

document.getElementById("guess-btn").addEventListener("click", guess);

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

document
  .getElementById("refresh-btn")
  .addEventListener("click", refreshDrawing);

function refreshDrawing() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  lines = [];
  document.getElementById("guess-message").innerText = "";
  hideConfirmation();
}

function showConfirmation(ranking) {
  var confirmationContainer = document.getElementById("confirmation-container");
  confirmationContainer.style.display = "block"; // Show the confirmation container

  var dropdown = document.getElementById("country-dropdown");
  dropdown.innerHTML = ""; // Clear previous options

  // Add options for each country in the list
  ranking.forEach(function (country) {
    var option = document.createElement("option");
    option.text = country;
    dropdown.add(option);
  });

  // Show the confirm button
  document.getElementById("confirm-btn").style.display = "inline-block";
  // Show the instruction message
  document.getElementById("instruction-message").style.display = "block";
}

function hideConfirmation() {
  document.getElementById("confirmation-container").style.display = "none"; // Hide the confirmation container
  document.getElementById("confirm-btn").style.display = "none"; // Hide the confirm button
  document.getElementById("instruction-message").style.display = "none"; // Hide the instruction message
}

document
  .getElementById("confirm-btn")
  .addEventListener("click", confirmCountry);

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
    // Add confetti here
  } else {
    const message = getRandomMessage(
      incorrectMessages,
      selectedCountry,
      guessedCountry,
    );
    document.getElementById("guess-message").innerText = message;
  }

  hideConfirmation();

  // Send POST request to feedback route
  fetch("/feedback", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      country: selectedCountry,
      drawing_id: window.currentDrawingId,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      console.log(data.message);
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}
