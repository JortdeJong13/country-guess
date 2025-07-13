/**
 * Easter Egg Mini Game - Bouncing Globe
 *
 * Activated by clicking the title "Draw a Country" 5 times.
 * Features a bouncing Earth emoji that falls from above and bounces
 * around the canvas, interacting with drawn lines and responding to clicks.
 */
class MiniGame {
  constructor() {
    this.canvas = document.getElementById("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.isActive = false;
    this.animationId = null;

    // Create separate animation layer to avoid interfering with drawing
    this.animCanvas = document.createElement("canvas");
    this.animCtx = this.animCanvas.getContext("2d");
    this.animCanvas.style.position = "absolute";
    this.animCanvas.style.pointerEvents = "none";
    this.animCanvas.style.zIndex = "1";
    this.animCanvas.style.top = this.canvas.style.top;
    this.animCanvas.style.left = this.canvas.style.left;
    this.animCanvas.style.transform = this.canvas.style.transform;
    this.animCanvas.style.borderRadius = this.canvas.style.borderRadius;

    // Insert animation layer after main canvas
    this.canvas.parentNode.insertBefore(
      this.animCanvas,
      this.canvas.nextSibling,
    );

    // Globe physics and appearance properties
    this.globe = {
      x: 0,
      y: 0,
      radius: 20,
      clickRadius: 30,
      vx: 0,
      vy: 0,
      emoji: "🌍",
      gravity: 0.3,
      bounce: 0.9,
      friction: 0.992,
    };

    // Easter egg activation tracking
    this.titleClickCount = 0;
    this.titleElement = document.querySelector("h1");
    this.originalTitleCursor = this.titleElement.style.cursor;

    this.init();
  }

  init() {
    // Setup event listeners and prevent text selection on title
    this.titleElement.addEventListener("click", (e) =>
      this.handleTitleClick(e),
    );
    this.titleElement.style.userSelect = "none";
    this.titleElement.style.webkitUserSelect = "none";
    this.titleElement.style.mozUserSelect = "none";
    this.titleElement.style.msUserSelect = "none";

    // Add click listener to canvas for globe interaction
    this.canvas.addEventListener("click", (e) => this.handleCanvasClick(e));

    // Track mouse movement for cursor feedback (pointer over globe)
    this.canvas.addEventListener("mousemove", (e) => this.handleMouseMove(e));

    // Reset minigame when user clears the drawing
    document
      .getElementById("refresh-btn")
      .addEventListener("click", () => this.reset());
  }

  handleTitleClick(event) {
    event.preventDefault();
    event.stopPropagation();
    this.titleClickCount++;

    // Provide visual feedback that click was registered
    this.titleElement.style.cursor = "pointer";
    this.titleElement.style.transform = "scale(0.95)";
    setTimeout(() => {
      this.titleElement.style.transform = "scale(1)";
    }, 100);

    // Activate minigame after 5 clicks
    if (this.titleClickCount >= 5) {
      this.startGame();
      this.titleClickCount = 0; // Reset counter
    }
  }

  startGame() {
    if (this.isActive) return;

    this.isActive = true;

    // Configure animation layer to perfectly overlay main canvas
    this.animCanvas.width = this.canvas.width;
    this.animCanvas.height = this.canvas.height;
    this.animCanvas.style.width = this.canvas.style.width;
    this.animCanvas.style.height = this.canvas.style.height;

    // Start globe above canvas for dramatic falling entrance
    const canvasRect = this.canvas.getBoundingClientRect();
    this.globe.x = this.canvas.width / 2;
    this.globe.y = -this.globe.radius; // Start above canvas
    this.globe.vx = (Math.random() - 0.5) * 4; // Random horizontal velocity
    this.globe.vy = 0;

    // Start animation loop
    this.gameLoop();
  }

  gameLoop() {
    if (!this.isActive) return;

    this.updateGlobe();
    this.drawGlobe();

    this.animationId = requestAnimationFrame(() => this.gameLoop());
  }

  updateGlobe() {
    // Check line collisions before moving to prevent penetration
    this.checkLineCollisions();

    // Apply downward gravitational force
    this.globe.vy += this.globe.gravity;

    // Apply air resistance to both axes
    this.globe.vx *= this.globe.friction;
    this.globe.vy *= this.globe.friction;

    // Update position (round to prevent sub-pixel vibration)
    this.globe.x = Math.round(this.globe.x + this.globe.vx);
    this.globe.y = Math.round(this.globe.y + this.globe.vy);

    // Handle canvas boundary collisions
    if (this.globe.x + this.globe.radius > this.canvas.width) {
      this.globe.x = this.canvas.width - this.globe.radius;
      this.globe.vx *= -this.globe.bounce;
    }
    if (this.globe.x - this.globe.radius < 0) {
      this.globe.x = this.globe.radius;
      this.globe.vx *= -this.globe.bounce;
    }

    // Handle top and bottom boundary collisions
    if (this.globe.y + this.globe.radius > this.canvas.height) {
      this.globe.y = this.canvas.height - this.globe.radius;
      this.globe.vy *= -this.globe.bounce;
    }
    if (this.globe.y - this.globe.radius < 0) {
      this.globe.y = this.globe.radius;
      this.globe.vy *= -this.globe.bounce;
    }

    // Final collision check after movement (safety net)
    this.checkLineCollisions();
  }

  checkLineCollisions() {
    // Access lines drawn by the user (from drawing.js)
    if (typeof window.lines !== "undefined" && window.lines) {
      for (let line of window.lines) {
        if (line.length < 2) continue;

        for (let i = 0; i < line.length - 1; i++) {
          const p1 = line[i];
          const p2 = line[i + 1];

          const collisionData = this.checkLineSegmentCollision(p1, p2);
          if (collisionData.isColliding) {
            // Calculate surface normal for realistic bounce physics
            const dx = collisionData.closestX - this.globe.x;
            const dy = collisionData.closestY - this.globe.y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance === 0) continue; // Safety check

            // Create unit vector for collision direction
            const normalX = dx / distance;
            const normalY = dy / distance;

            // Push globe away from the line with extra margin
            const overlap = this.globe.radius - distance + 1; // Add 1px buffer
            this.globe.x -= normalX * overlap;
            this.globe.y -= normalY * overlap;

            // Apply realistic reflection physics
            const dotProduct =
              this.globe.vx * normalX + this.globe.vy * normalY;
            this.globe.vx -= 2 * dotProduct * normalX * this.globe.bounce;
            this.globe.vy -= 2 * dotProduct * normalY * this.globe.bounce;

            // Dampen small movements to prevent infinite micro-bouncing
            if (
              Math.abs(this.globe.vx) < 1.5 &&
              Math.abs(this.globe.vy) < 1.5
            ) {
              this.globe.vx *= 0.6;
              this.globe.vy *= 0.6;
            }

            return; // Process only one collision per frame for stability
          }
        }
      }
    }
  }

  checkLineSegmentCollision(p1, p2) {
    // Use point-to-line-segment distance algorithm
    const A = this.globe.x - p1[0];
    const B = this.globe.y - p1[1];
    const C = p2[0] - p1[0];
    const D = p2[1] - p1[1];

    const dot = A * C + B * D;
    const lenSq = C * C + D * D;

    if (lenSq === 0) return { isColliding: false };

    let param = dot / lenSq;

    let xx, yy;

    if (param < 0) {
      xx = p1[0];
      yy = p1[1];
    } else if (param > 1) {
      xx = p2[0];
      yy = p2[1];
    } else {
      xx = p1[0] + param * C;
      yy = p1[1] + param * D;
    }

    const dx = this.globe.x - xx;
    const dy = this.globe.y - yy;
    const distance = Math.sqrt(dx * dx + dy * dy);

    return {
      isColliding: distance < this.globe.radius,
      closestX: xx,
      closestY: yy,
      distance: distance,
    };
  }

  drawGlobe() {
    // Clear previous globe position
    this.animCtx.clearRect(0, 0, this.animCanvas.width, this.animCanvas.height);

    // Configure emoji rendering settings
    this.animCtx.save();

    // Render Earth emoji at globe position
    this.animCtx.font = `${this.globe.radius * 2}px Arial`;
    this.animCtx.textAlign = "center";
    this.animCtx.textBaseline = "middle";
    this.animCtx.fillText(this.globe.emoji, this.globe.x, this.globe.y);

    // Clean up rendering context
    this.animCtx.restore();
  }

  handleMouseMove(event) {
    if (!this.isActive) return;

    const rect = this.canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    // Use larger click radius for better user experience
    const distance = Math.sqrt(
      Math.pow(mouseX - this.globe.x, 2) + Math.pow(mouseY - this.globe.y, 2),
    );

    // Update cursor based on hover state
    if (distance < this.globe.clickRadius) {
      this.canvas.style.cursor = "pointer"; // Indicate clickable
    } else {
      this.canvas.style.cursor = "crosshair"; // Normal drawing mode
    }
  }

  handleCanvasClick(event) {
    if (!this.isActive) return;

    const rect = this.canvas.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;

    // Check if click hits the globe (generous hitbox)
    const distance = Math.sqrt(
      Math.pow(clickX - this.globe.x, 2) + Math.pow(clickY - this.globe.y, 2),
    );

    if (distance < this.globe.clickRadius) {
      // Make globe bounce up when clicked with bigger impact
      this.globe.vy = -18; // Even stronger upward velocity
      this.globe.vx += (Math.random() - 0.5) * 15; // Even stronger random horizontal velocity
    }
  }

  reset() {
    this.isActive = false;
    this.titleClickCount = 0;

    // Stop animation loop
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }

    // Remove globe from animation layer
    if (this.animCtx) {
      this.animCtx.clearRect(
        0,
        0,
        this.animCanvas.width,
        this.animCanvas.height,
      );
    }

    // Restore original cursor states
    this.titleElement.style.cursor = this.originalTitleCursor;
    this.canvas.style.cursor = "crosshair";
  }
}

// Initialize the minigame system when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  window.miniGame = new MiniGame();
});

// Export for potential use in other modules
export { MiniGame };
