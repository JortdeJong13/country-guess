import { renderUserDrawing } from "./drawing.js";
import * as msg from "./messages.js";

let currentRank = 0;
let totalDrawings = 0;

/**
 * API Functions
 */

async function fetchDrawingByRank(rank) {
  const response = await fetch(`/drawing?rank=${rank}`);
  if (!response.ok) {
    throw new Error("Failed to fetch drawing");
  }
  return response.json();
}

/**
 * UI Helper Functions
 */
function showMessage(message) {
  document.getElementById("message").innerText = message;
}

function updateTotal(count) {
  totalDrawings = count;
}

function renderDrawing(data) {
  renderUserDrawing(data.lines);

  const archiveMessage = msg.getArchiveMessage(
    data.guess_score,
    data.country_name,
    data.country_guess,
  );
  showMessage(archiveMessage);
}

/**
 * Public API
 */
export async function showLeaderboard() {
  currentRank = 0;
  return showLeaderboardAt(currentRank);
}

export async function showLeaderboardNext() {
  if (currentRank < totalDrawings - 1) {
    currentRank++;
  }
  return showLeaderboardAt(currentRank);
}

export async function showLeaderboardPrevious() {
  if (currentRank > 0) {
    currentRank--;
  }
  return showLeaderboardAt(currentRank);
}

export async function showLeaderboardAt(rank) {
  try {
    const data = await fetchDrawingByRank(rank);
    renderDrawing(data);
    updateTotal(data.total);
    return { success: true, rank: currentRank, total: totalDrawings };
  } catch (error) {
    console.error("Error loading leaderboard drawing:", error);
    return { success: false };
  }
}
