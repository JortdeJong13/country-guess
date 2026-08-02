import { renderUserDrawing } from "./drawing.js";
import * as msg from "./messages.js";

let currentRank = 0;
let totalDrawings = 0;
let currentDrawingId = null;

/**
 * API Functions
 */

async function fetchDrawingByRank(rank) {
  const response = await fetch(`/drawing?rank=${rank}`);
  const data = await response.json().catch(() => null);

  if (!response.ok) {
    const error = new Error(data?.message || "Failed to fetch drawing");
    error.status = response.status;
    throw error;
  }
  return data;
}

/**
 * Public API
 */
export async function showLeaderboard() {
  currentRank = 0;
  currentDrawingId = null;
  msg.clearLeaderboardMessageCache();
  msg.showLoadingMessage();
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
    currentDrawingId = data.id;
    totalDrawings = data.total;
    msg.setLeaderboardMessage(data);
    renderUserDrawing(data.lines);
    return { success: true, rank: currentRank, total: totalDrawings };
  } catch (error) {
    console.error("Error loading leaderboard drawing:", error);

    if (error.status === 404 && rank === 0) {
      currentDrawingId = null;
      totalDrawings = 0;
      msg.setEmptyLeaderboardMessage();
    } else {
      msg.setLeaderboardMessage(null);
    }

    return { success: false, rank: currentRank, total: totalDrawings };
  }
}

export async function reportCurrentDrawing() {
  if (!currentDrawingId) {
    throw new Error("No leaderboard drawing loaded");
  }

  const response = await fetch("/report", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ drawing_id: currentDrawingId }),
  });
  const data = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(data?.message || "Failed to report drawing");
  }

  return data;
}
