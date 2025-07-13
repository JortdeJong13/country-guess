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
    this.animCanvas.style.position = "fixed";
    this.animCanvas.style.pointerEvents = "none";
    this.animCanvas.style.zIndex = "1";
    this.animCanvas.style.top = "0";
    this.animCanvas.style.left = "0";
    this.animCanvas.style.width = "100vw";
    this.animCanvas.style.height = "100vh";

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
      rotation: 0,
      emoji: "🌍",
      gravity: 0.3,
      bounce: 0.9,
      friction: 0.992,
    };

    // Easter egg activation tracking
    this.titleClickCount = 0;
    this.globeElement = document.querySelector("#globe-emoji");
    this.originalGlobeCursor = this.globeElement.style.cursor;

    this.init();
  }

  init() {
    // Setup event listeners and prevent text selection on globe emoji
    this.globeElement.addEventListener("click", (e) =>
      this.handleGlobeClick(e),
    );
    this.globeElement.style.userSelect = "none";
    this.globeElement.style.webkitUserSelect = "none";
    this.globeElement.style.mozUserSelect = "none";
    this.globeElement.style.msUserSelect = "none";
    this.globeElement.style.display = "inline-block"; // Ensure transforms work
    this.globeElement.style.transformOrigin = "center center";

    // Add click listener to canvas for globe interaction
    this.canvas.addEventListener("click", (e) => this.handleCanvasClick(e));

    // Track mouse movement for cursor feedback (pointer over globe)
    this.canvas.addEventListener("mousemove", (e) => this.handleMouseMove(e));

    // Reset minigame when user clears the drawing
    document
      .getElementById("refresh-btn")
      .addEventListener("click", () => this.reset());

    // Handle window resize
    window.addEventListener("resize", () => this.handleResize());
  }

  handleGlobeClick(event) {
    event.preventDefault();
    event.stopPropagation();
    this.titleClickCount++;

    // Provide visual feedback that click was registered
    this.globeElement.style.setProperty("transform", "scale(0.9)", "important");
    this.globeElement.style.setProperty(
      "transition",
      "transform 0.15s ease-out",
      "important",
    );

    setTimeout(() => {
      this.globeElement.style.setProperty(
        "transform",
        "scale(1.05)",
        "important",
      );
      setTimeout(() => {
        this.globeElement.style.setProperty(
          "transform",
          "scale(1)",
          "important",
        );
        setTimeout(() => {
          this.globeElement.style.removeProperty("transition");
          this.globeElement.style.removeProperty("transform");
        }, 150);
      }, 150);
    }, 150);

    // Activate minigame after 6 clicks
    if (this.titleClickCount >= 6) {
      this.startGame();
      this.titleClickCount = 0; // Reset counter
    }
  }

  startGame() {
    if (this.isActive) return;

    this.isActive = true;

    // Configure animation layer to cover full viewport
    this.animCanvas.width = window.innerWidth;
    this.animCanvas.height = window.innerHeight;

    // Calculate position of globe emoji in title (absolute screen position)
    const globeRect = this.globeElement.getBoundingClientRect();

    // Start globe at actual title position in viewport coordinates
    this.globe.x = globeRect.left + globeRect.width / 2;
    this.globe.y = globeRect.top + globeRect.height / 2;

    // Store canvas boundaries for collision detection
    const canvasRect = this.canvas.getBoundingClientRect();
    this.canvasBounds = {
      left: canvasRect.left,
      top: canvasRect.top,
      right: canvasRect.right,
      bottom: canvasRect.bottom,
      width: canvasRect.width,
      height: canvasRect.height,
    };
    this.globe.vx = (Math.random() - 0.5) * 0.5; // Very small horizontal velocity
    this.globe.vy = 0; // Start with no vertical velocity, let gravity take over

    // Hide the title emoji
    this.globeElement.style.visibility = "hidden";

    // Start animation loop
    this.gameLoop();
  }

  handleResize() {
    if (!this.isActive) return;

    // Update animation canvas size
    this.animCanvas.width = window.innerWidth;
    this.animCanvas.height = window.innerHeight;

    // Recalculate canvas boundaries
    const canvasRect = this.canvas.getBoundingClientRect();
    this.canvasBounds = {
      left: canvasRect.left,
      top: canvasRect.top,
      right: canvasRect.right,
      bottom: canvasRect.bottom,
      width: canvasRect.width,
      height: canvasRect.height,
    };

    // Keep globe within reasonable bounds if it's outside the viewport
    if (this.globe.x < 0) this.globe.x = this.globe.radius;
    if (this.globe.x > window.innerWidth)
      this.globe.x = window.innerWidth - this.globe.radius;
    if (this.globe.y < 0) this.globe.y = this.globe.radius;
    if (this.globe.y > window.innerHeight)
      this.globe.y = window.innerHeight - this.globe.radius;
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

    // Calculate rotation based on horizontal movement (rolling effect)
    if (Math.abs(this.globe.vx) > 0.1) {
      this.globe.rotation += this.globe.vx * 0.05; // Scale factor for natural rolling speed
    }

    // Update position (round to prevent sub-pixel vibration)
    this.globe.x = Math.round(this.globe.x + this.globe.vx);
    this.globe.y = Math.round(this.globe.y + this.globe.vy);

    // Handle canvas boundary collisions using screen coordinates
    if (this.canvasBounds) {
      // Right boundary
      if (this.globe.x + this.globe.radius > this.canvasBounds.right) {
        this.globe.x = this.canvasBounds.right - this.globe.radius;
        this.globe.vx *= -this.globe.bounce;
      }
      // Left boundary
      if (this.globe.x - this.globe.radius < this.canvasBounds.left) {
        this.globe.x = this.canvasBounds.left + this.globe.radius;
        this.globe.vx *= -this.globe.bounce;
      }
      // Top boundary (only if globe is moving up and within canvas area)
      if (
        this.globe.y - this.globe.radius < this.canvasBounds.top &&
        this.globe.vy < 0
      ) {
        this.globe.y = this.canvasBounds.top + this.globe.radius;
        this.globe.vy *= -this.globe.bounce;
      }
      // Bottom boundary
      if (this.globe.y + this.globe.radius > this.canvasBounds.bottom) {
        this.globe.y = this.canvasBounds.bottom - this.globe.radius;
        this.globe.vy *= -this.globe.bounce;
      }
    }

    // Final collision check after movement (safety net)
    this.checkLineCollisions();
  }

  checkLineCollisions() {
    // Access lines drawn by the user (from drawing.js)
    if (
      typeof window.lines !== "undefined" &&
      window.lines &&
      this.canvasBounds
    ) {
      for (let line of window.lines) {
        if (line.length < 2) continue;

        for (let i = 0; i < line.length - 1; i++) {
          const p1 = line[i];
          const p2 = line[i + 1];

          // Convert canvas coordinates to screen coordinates
          const screenP1 = [
            p1[0] + this.canvasBounds.left,
            p1[1] + this.canvasBounds.top,
          ];
          const screenP2 = [
            p2[0] + this.canvasBounds.left,
            p2[1] + this.canvasBounds.top,
          ];

          const collisionData = this.checkLineSegmentCollision(
            screenP1,
            screenP2,
          );
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

    // Move to globe position and apply rotation
    this.animCtx.translate(this.globe.x, this.globe.y);
    this.animCtx.rotate(this.globe.rotation);

    // Render Earth emoji at rotated position
    this.animCtx.font = `${this.globe.radius * 2}px Arial`;
    this.animCtx.textAlign = "center";
    this.animCtx.textBaseline = "middle";
    this.animCtx.fillText(this.globe.emoji, 0, 0);

    // Clean up rendering context
    this.animCtx.restore();
  }

  handleMouseMove(event) {
    if (!this.isActive) return;

    // Use screen coordinates for mouse detection
    const mouseX = event.clientX;
    const mouseY = event.clientY;

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

    // Use screen coordinates for click detection
    const clickX = event.clientX;
    const clickY = event.clientY;

    // Check if click hits the globe (generous hitbox)
    const distance = Math.sqrt(
      Math.pow(clickX - this.globe.x, 2) + Math.pow(clickY - this.globe.y, 2),
    );

    if (distance < this.globe.clickRadius) {
      // Calculate direction from click point to globe center
      const directionX = this.globe.x - clickX;
      const directionY = this.globe.y - clickY;

      // Normalize the direction vector
      const magnitude = Math.sqrt(
        directionX * directionX + directionY * directionY,
      );
      if (magnitude > 0) {
        const normalizedX = directionX / magnitude;
        const normalizedY = directionY / magnitude;

        // Apply force in opposite direction of click
        const force = 22;
        this.globe.vx += normalizedX * force;
        this.globe.vy += normalizedY * force;
      }
    }
  }

  reset() {
    this.isActive = false;
    this.titleClickCount = 0;

    // Reset globe rotation
    this.globe.rotation = 0;

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

    // Restore the title emoji
    this.globeElement.style.visibility = "visible";

    // Restore original cursor states
    this.globeElement.style.cursor = this.originalGlobeCursor;
    this.canvas.style.cursor = "crosshair";
  }
}

// Initialize the minigame system when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  window.miniGame = new MiniGame();
});

// Export for potential use in other modules
export { MiniGame };
