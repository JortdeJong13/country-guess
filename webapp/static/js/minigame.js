class MiniGame {
  constructor() {
    this.canvas = document.getElementById("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.isActive = false;
    this.animationId = null;

    // Device pixel ratio for crisp rendering
    this.devicePixelRatio = window.devicePixelRatio || 1;

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

    // Matter.js setup
    this.engine = null;
    this.world = null;
    this.globe = null;
    this.boundaries = [];
    this.lineWalls = [];
    this.canvasTopWalls = []; // Top walls with hole for globe entry

    // Globe appearance properties
    this.globeRadius = 20;
    this.globeEmoji = "🌍";
    this.globeRotation = 0;

    // Easter egg activation tracking
    this.titleClickCount = 0;
    this.globeElement = document.querySelector("#globe-emoji");
    this.originalGlobeCursor = this.globeElement.style.cursor;

    // Title tilting effect tracking
    this.titleElement = document.querySelector("#title-text");
    this.titleTiltCount = 0;
    this.isTilted = false;

    this.init();
  }

  init() {
    // Set up event listeners
    this.globeElement.addEventListener("click", (e) =>
      this.handleGlobeClick(e),
    );
    this.titleElement.addEventListener("click", (e) =>
      this.handleTitleClick(e),
    );
    this.canvas.addEventListener("click", (e) => this.handleCanvasClick(e));
    this.canvas.addEventListener("mousemove", (e) => this.handleMouseMove(e));

    // Handle window resize
    window.addEventListener("resize", () => this.handleResize());

    // Monitor for new lines being drawn to update physics walls
    this.lastLineCount = 0;
    this.lineCheckInterval = setInterval(() => {
      if (this.isActive && window.lines) {
        const currentLineCount = window.lines.length;
        if (currentLineCount !== this.lastLineCount) {
          this.createLineWalls();
          this.lastLineCount = currentLineCount;
        }
      }
    }, 50); // Check every 50ms

    // Handle clear button to reset everything
    document.getElementById("refresh-btn").addEventListener("click", () => {
      this.reset();
    });

    // Set up title transform origin for proper tilting from left side
    this.titleElement.style.transformOrigin = "left center";
    this.titleElement.style.userSelect = "none";
    this.titleElement.style.webkitUserSelect = "none";
    this.titleElement.style.mozUserSelect = "none";
    this.titleElement.style.msUserSelect = "none";

    // Handle touch events for mobile
    this.canvas.addEventListener("touchend", (e) => {
      // Only handle minigame touches, not drawing touches
      if (!this.isActive) return;
      e.preventDefault();
      const touch = e.changedTouches[0];
      const clickEvent = new MouseEvent("click", {
        clientX: touch.clientX,
        clientY: touch.clientY,
      });
      this.handleCanvasClick(clickEvent);
    });

    this.globeElement.addEventListener("touchend", (e) => {
      e.preventDefault();
      this.handleGlobeClick(e);
    });

    this.titleElement.addEventListener("touchend", (e) => {
      e.preventDefault();
      this.handleTitleClick(e);
    });

    // Initialize Matter.js
    this.initPhysics();
  }

  initPhysics() {
    // Create engine
    this.engine = Matter.Engine.create();
    this.world = this.engine.world;

    // Configure engine settings - positive Y is down in screen coordinates
    this.engine.world.gravity.y = 1;
    this.engine.world.gravity.scale = 0.003; // Gentler gravity

    // Better physics settings
    this.engine.constraintIterations = 2;
    this.engine.positionIterations = 6;
    this.engine.velocityIterations = 4;
    this.engine.enableSleeping = false;

    // Disable default renderer since we're using custom rendering
    this.engine.render = null;
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

  handleTitleClick(event) {
    event.preventDefault();
    event.stopPropagation();

    // Don't respond to clicks if already tilted or falling
    if (this.isTilted) return;

    this.titleTiltCount++;

    // Check if we should tilt first
    if (this.titleTiltCount >= 5 && !this.isTilted) {
      this.isTilted = true;
      // Start gravity-like tilt animation (fast accelerating fall)
      setTimeout(() => {
        this.titleElement.style.transition =
          "transform 0.4s cubic-bezier(0.55, 0.055, 0.675, 0.19)";
        this.titleElement.style.transform = "rotate(8deg)";
      }, 50);
    } else if (this.titleTiltCount < 5) {
      // Only do bounce feedback if not close to tilting and not already tilted
      this.titleElement.style.transition = "transform 0.1s ease-out";
      this.titleElement.style.setProperty(
        "transform",
        "scale(0.985)",
        "important",
      );
      setTimeout(() => {
        // Check again if we're still in a safe state to continue bouncing
        if (!this.isTilted) {
          this.titleElement.style.setProperty(
            "transform",
            "scale(1.005)",
            "important",
          );
          setTimeout(() => {
            if (!this.isTilted) {
              this.titleElement.style.setProperty(
                "transform",
                "scale(1)",
                "important",
              );
              setTimeout(() => {
                if (!this.isTilted) {
                  this.titleElement.style.removeProperty("transform");
                  this.titleElement.style.transition = "";
                }
              }, 150);
            }
          }, 100);
        }
      }, 100);
    }
  }

  startGame() {
    if (this.isActive) return;

    this.isActive = true;

    // Configure animation layer with proper scaling
    this.setupAnimationCanvas();

    // Get canvas bounds
    const canvasRect = this.canvas.getBoundingClientRect();
    const globeRect = this.globeElement.getBoundingClientRect();

    // Calculate globe starting position (center of globe emoji in title)
    const startX = globeRect.left + globeRect.width / 2;
    const startY = globeRect.top + globeRect.height / 2;

    // Create globe body
    this.globe = Matter.Bodies.circle(startX, startY, this.globeRadius, {
      restitution: 0.95, // Very bouncy
      friction: 0.05,
      frictionAir: 0.01, // Less air resistance for more bounce
      density: 0.001, // Lighter
      frictionStatic: 0.3,
      inertia: Infinity,
    });

    Matter.World.add(this.world, this.globe);

    // Give initial small velocity toward the canvas (always downward for now)
    Matter.Body.setVelocity(this.globe, {
      x: (Math.random() - 0.3) * 1,
      y: 0,
    });

    // Create boundaries (walls around canvas)
    this.createBoundaries(canvasRect);

    // Create walls from drawn lines
    this.createLineWalls();

    // Hide the title emoji
    this.globeElement.style.visibility = "hidden";

    // Start physics and animation loop
    this.gameLoop();
  }

  setupAnimationCanvas() {
    // Set up canvas with proper scaling for crisp rendering
    const displayWidth = window.innerWidth;
    const displayHeight = window.innerHeight;

    this.animCanvas.style.width = displayWidth + "px";
    this.animCanvas.style.height = displayHeight + "px";

    this.animCanvas.width = displayWidth * this.devicePixelRatio;
    this.animCanvas.height = displayHeight * this.devicePixelRatio;

    this.animCtx.scale(this.devicePixelRatio, this.devicePixelRatio);

    // Enable crisp rendering
    this.animCtx.imageSmoothingEnabled = false;
    this.animCtx.textRenderingOptimization = "optimizeSpeed";
  }

  createBoundaries(canvasRect) {
    // Clear existing boundaries
    if (this.boundaries.length > 0) {
      Matter.World.remove(this.world, this.boundaries);
      this.boundaries = [];
    }

    const wallThickness = 50;

    // Create invisible walls around the entire viewport to contain the ball
    // wherever it starts (title could be above canvas)
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;

    const walls = [
      // Left wall (full viewport height)
      Matter.Bodies.rectangle(
        -wallThickness / 2,
        viewportHeight / 2,
        wallThickness,
        viewportHeight + wallThickness * 2,
        { isStatic: true, restitution: 0.95 },
      ),
      // Right wall (full viewport height)
      Matter.Bodies.rectangle(
        viewportWidth + wallThickness / 2,
        viewportHeight / 2,
        wallThickness,
        viewportHeight + wallThickness * 2,
        { isStatic: true, restitution: 0.95 },
      ),
      // Top wall (full viewport width)
      Matter.Bodies.rectangle(
        viewportWidth / 2,
        -wallThickness / 2,
        viewportWidth + wallThickness * 2,
        wallThickness,
        { isStatic: true, restitution: 0.95 },
      ),
      // Canvas bottom wall (only canvas width, positioned at canvas bottom)
      Matter.Bodies.rectangle(
        canvasRect.left + canvasRect.width / 2,
        canvasRect.bottom + wallThickness / 2,
        canvasRect.width,
        wallThickness,
        { isStatic: true, restitution: 0.95 },
      ),
      // Canvas left wall
      Matter.Bodies.rectangle(
        canvasRect.left - wallThickness / 2,
        canvasRect.top + canvasRect.height / 2,
        wallThickness,
        canvasRect.height,
        { isStatic: true, restitution: 0.95 },
      ),
      // Canvas right wall
      Matter.Bodies.rectangle(
        canvasRect.right + wallThickness / 2,
        canvasRect.top + canvasRect.height / 2,
        wallThickness,
        canvasRect.height,
        { isStatic: true, restitution: 0.95 },
      ),
    ];

    this.boundaries = walls;
    Matter.World.add(this.world, walls);

    // Create canvas top wall with hole
    this.createTopWallWithHole(canvasRect);
  }

  createLineWalls() {
    // Clear existing line walls
    if (this.lineWalls.length > 0) {
      Matter.World.remove(this.world, this.lineWalls);
      this.lineWalls = [];
    }

    // Access lines drawn by the user
    if (typeof window.lines !== "undefined" && window.lines) {
      const canvasRect = this.canvas.getBoundingClientRect();

      for (let line of window.lines) {
        if (line.length < 2) continue;

        for (let i = 0; i < line.length - 1; i++) {
          const p1 = line[i];
          const p2 = line[i + 1];

          // Convert canvas coordinates to screen coordinates
          // Account for canvas scaling on mobile devices
          const canvasScale = canvasRect.width / this.canvas.width;
          const screenX1 = canvasRect.left + p1[0] * canvasScale;
          const screenY1 = canvasRect.top + p1[1] * canvasScale;
          const screenX2 = canvasRect.left + p2[0] * canvasScale;
          const screenY2 = canvasRect.top + p2[1] * canvasScale;

          // Calculate line segment properties
          const centerX = (screenX1 + screenX2) / 2;
          const centerY = (screenY1 + screenY2) / 2;
          const length = Math.sqrt(
            Math.pow(screenX2 - screenX1, 2) + Math.pow(screenY2 - screenY1, 2),
          );

          // Skip very short segments to avoid physics issues
          if (length < 1) continue;

          const angle = Math.atan2(screenY2 - screenY1, screenX2 - screenX1);

          // Create a thin rectangle to represent the line
          const lineWall = Matter.Bodies.rectangle(
            centerX,
            centerY,
            length,
            3, // Line thickness
            {
              angle: angle,
              isStatic: true,
              restitution: 0.95,
            },
          );

          this.lineWalls.push(lineWall);
        }
      }

      Matter.World.add(this.world, this.lineWalls);
    }
  }

  handleResize() {
    if (!this.isActive) return;

    // Update device pixel ratio (might change when moving between screens)
    this.devicePixelRatio = window.devicePixelRatio || 1;

    // Update animation canvas with proper scaling
    this.setupAnimationCanvas();

    // Recreate boundaries with new canvas position
    const canvasRect = this.canvas.getBoundingClientRect();
    this.createBoundaries(canvasRect);

    // Recreate line walls and top wall with hole after resize
    this.createLineWalls();
    this.createTopWallWithHole(canvasRect);
  }

  createTopWallWithHole(canvasRect) {
    // Remove existing top walls
    if (this.canvasTopWalls.length > 0) {
      Matter.World.remove(this.world, this.canvasTopWalls);
      this.canvasTopWalls = [];
    }

    const wallThickness = 50;
    const holeWidth = this.globeRadius * 3; // Small hole for globe to pass through

    // Get globe starting position to center the hole
    const globeRect = this.globeElement.getBoundingClientRect();
    const holeCenter = globeRect.left + globeRect.width / 2;

    // Clamp hole position to be within canvas bounds
    const minHoleCenter = canvasRect.left + holeWidth / 2;
    const maxHoleCenter = canvasRect.right - holeWidth / 2;
    const clampedHoleCenter = Math.max(
      minHoleCenter,
      Math.min(maxHoleCenter, holeCenter),
    );

    const walls = [];

    // Left part of top wall (from canvas left to hole start)
    const leftWallWidth = clampedHoleCenter - holeWidth / 2 - canvasRect.left;
    if (leftWallWidth > 10) {
      // Only create if wide enough
      walls.push(
        Matter.Bodies.rectangle(
          canvasRect.left + leftWallWidth / 2,
          canvasRect.top - wallThickness / 2,
          leftWallWidth,
          wallThickness,
          { isStatic: true, restitution: 0.95 },
        ),
      );
    }

    // Right part of top wall (from hole end to canvas right)
    const rightWallWidth =
      canvasRect.right - (clampedHoleCenter + holeWidth / 2);
    if (rightWallWidth > 10) {
      // Only create if wide enough
      walls.push(
        Matter.Bodies.rectangle(
          canvasRect.right - rightWallWidth / 2,
          canvasRect.top - wallThickness / 2,
          rightWallWidth,
          wallThickness,
          { isStatic: true, restitution: 0.95 },
        ),
      );
    }

    this.canvasTopWalls = walls;
    Matter.World.add(this.world, walls);
  }

  gameLoop() {
    if (!this.isActive) return;

    // Update physics with fixed timestep for consistency
    Matter.Engine.update(this.engine, 16.666); // ~60fps

    // Update globe rotation based on velocity
    if (this.globe) {
      const velocity = this.globe.velocity;
      this.globeRotation += velocity.x * 0.03;

      // Prevent globe from sleeping (getting stuck)
      if (Math.abs(velocity.x) < 0.1 && Math.abs(velocity.y) < 0.1) {
        Matter.Sleeping.set(this.globe, false);
      }
    }

    // Draw everything
    this.drawGlobe();

    // Continue loop
    this.animationId = requestAnimationFrame(() => this.gameLoop());
  }

  drawGlobe() {
    // Clear previous globe position (account for device pixel ratio)
    this.animCtx.clearRect(0, 0, window.innerWidth, window.innerHeight);

    if (!this.globe) return;

    // Get globe position
    const x = this.globe.position.x;
    const y = this.globe.position.y;

    // Configure emoji rendering settings
    this.animCtx.save();

    // Move to globe position and apply rotation
    this.animCtx.translate(x, y);
    this.animCtx.rotate(this.globeRotation);

    // Render Earth emoji at rotated position with crisp text
    this.animCtx.font = `${this.globeRadius * 2}px Arial`;
    this.animCtx.textAlign = "center";
    this.animCtx.textBaseline = "middle";
    this.animCtx.fillStyle = "#000";
    this.animCtx.fillText(this.globeEmoji, 0, 0);

    // Clean up rendering context
    this.animCtx.restore();
  }

  handleMouseMove(event) {
    if (!this.isActive || !this.globe) return;

    // Check if mouse is near globe
    const mouseX = event.clientX;
    const mouseY = event.clientY;
    const globeX = this.globe.position.x;
    const globeY = this.globe.position.y;

    const distance = Math.sqrt(
      Math.pow(mouseX - globeX, 2) + Math.pow(mouseY - globeY, 2),
    );

    // Update cursor based on hover state
    if (distance < this.globeRadius * 2) {
      this.canvas.style.cursor = "pointer";
    } else {
      this.canvas.style.cursor = "crosshair";
    }
  }

  handleCanvasClick(event) {
    if (!this.isActive || !this.globe) return;

    const clickX = event.clientX;
    const clickY = event.clientY;
    const globeX = this.globe.position.x;
    const globeY = this.globe.position.y;

    // Check if click hits the globe
    const distance = Math.sqrt(
      Math.pow(clickX - globeX, 2) + Math.pow(clickY - globeY, 2),
    );

    if (distance < this.globeRadius * 2) {
      // Calculate direction from click point to globe center
      const directionX = globeX - clickX;
      const directionY = globeY - clickY;

      // Normalize the direction vector
      const magnitude = Math.sqrt(
        directionX * directionX + directionY * directionY,
      );

      if (magnitude > 0) {
        const normalizedX = directionX / magnitude;
        const normalizedY = directionY / magnitude;

        // Apply force in opposite direction of click
        const force = 0.14;
        Matter.Body.applyForce(this.globe, this.globe.position, {
          x: normalizedX * force,
          y: normalizedY * force,
        });

        // Wake up the body to prevent it from getting stuck
        Matter.Sleeping.set(this.globe, false);
      }
    }
  }

  reset() {
    this.isActive = false;
    this.titleClickCount = 0;
    this.globeRotation = 0;

    // Stop animation loop
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }

    // Remove physics bodies
    if (this.globe) {
      Matter.World.remove(this.world, this.globe);
      this.globe = null;
    }

    if (this.boundaries.length > 0) {
      Matter.World.remove(this.world, this.boundaries);
      this.boundaries = [];
    }

    if (this.lineWalls.length > 0) {
      Matter.World.remove(this.world, this.lineWalls);
      this.lineWalls = [];
    }

    if (this.canvasTopWalls.length > 0) {
      Matter.World.remove(this.world, this.canvasTopWalls);
      this.canvasTopWalls = [];
    }

    // Clear line monitoring interval
    if (this.lineCheckInterval) {
      clearInterval(this.lineCheckInterval);
      this.lineCheckInterval = null;
    }

    // Clear animation canvas
    if (this.animCtx) {
      this.animCtx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    }

    // Restore title emoji visibility
    this.globeElement.style.visibility = "visible";

    // Reset title tilt and clear animation properties
    this.titleTiltCount = 0;
    this.isTilted = false;
    this.titleElement.style.transform = "";
    this.titleElement.style.transition = "";
    this.titleElement.style.animation = "";

    // Restore original cursor states
    this.globeElement.style.cursor = this.originalGlobeCursor;
    this.canvas.style.cursor = "crosshair";
  }
}

// Initialize the minigame when the page loads
document.addEventListener("DOMContentLoaded", () => {
  window.miniGame = new MiniGame();
});

// Export for use in other modules
if (typeof module !== "undefined" && module.exports) {
  module.exports = MiniGame;
}
