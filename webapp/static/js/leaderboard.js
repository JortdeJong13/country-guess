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
    totalDrawings = data.total;
    msg.setLeaderboardMessage(data.rank, data.total, data.properties);
    renderUserDrawing(data.lines);
    return { success: true, rank: currentRank, total: totalDrawings };
  } catch (error) {
    console.error("Error loading leaderboard drawing:", error);
    return { success: false };
  }
}
