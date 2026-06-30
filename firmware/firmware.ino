/*
  ESP32 Motor Controller for Microscope Slide Scanner
  ====================================================
  Receives plain-text commands from Python (motor.py) over USB serial at
  115200 baud.  Each command ends with a newline '\n'.
  Replies "OK\n" on success or "ERR <reason>\n" on failure.

  Supported commands:
    MOVE X <steps>          — drive X stepper ± <steps> half-steps (fast axis)
    MOVE Y <steps>          — drive Y stepper ± <steps> half-steps (slow axis)
    MOVE XY <xsteps> <ysteps> — drive BOTH steppers concurrently (interleaved)
    HOME                    — acknowledge only; Python resets logical position

  Both axes are 28BYJ-48 unipolar steppers driven through ULN2003 boards.
  Half-stepping (8 phases) gives smoother motion and doubles resolution:
    4096 half-steps = 1 full revolution of the output shaft.

  Calibration (measured on this build):
    X axis: 1 rotation (4096 half-steps) = 0.8 cm  →  5120 half-steps / cm
    Y axis: 1 rotation (4096 half-steps) = 1.4 cm  →  2925.7 half-steps / cm
  Step rate (delay between half-steps) does NOT affect distance — only speed —
  so calibration holds at any rate that does not skip steps.
  Coils are de-energised after each move to prevent heat build-up.

  Hardware wiring (IN1-IN4 on each ULN2003 driver board):
    X stepper: GPIO 19, 18, 5, 17
    Y stepper: GPIO 27, 26, 25, 33
*/

// ── Stepper pin assignments ───────────────────────────────────────────────
const int X_PINS[4] = {19, 18, 5, 17};
const int Y_PINS[4] = {27, 26, 25, 33};

// Half-step sequence: 8 rows, each row drives one electrical step.
const int STEP_SEQ[8][4] = {
  {1,0,0,0}, {1,1,0,0}, {0,1,0,0}, {0,1,1,0},
  {0,0,1,0}, {0,0,1,1}, {0,0,0,1}, {1,0,0,1}
};

// Per-axis step delay (µs between half-steps).  THIS IS THE SPEED KNOB.
// After the X/Y swap the heavily-used SWEEP axis (10 captures/rung) is
// firmware Y and the RUNG axis (indexes twice/leg) is firmware X.  The SWEEP
// axis also moves the FARTHEST per step (≈3000 half-steps vs the rung's
// ≈1350), so at equal rates a sweep move takes ~2× longer in wall-clock and
// feels slower.  We can't fully equalise that without an unsafe rate, but we
// run the sweep axis (Y) faster than the rung axis to narrow the gap.
// Lower = faster but risks skipped steps, which silently corrupt registration:
// if rung 0 and rung 2 images don't line up, RAISE Y_STEP_DELAY_US back toward
// 900 (prior validated-safe rate was ~1200 µs).
const int X_STEP_DELAY_US = 900;   // rung axis (short moves)
const int Y_STEP_DELAY_US = 700;   // sweep axis — faster: long moves dominate runtime

int xStepIndex = 0;   // current position in the 8-step table for X
int yStepIndex = 0;   // current position in the 8-step table for Y

// Advance one stepper a single half-step in the given direction (no delay,
// no de-energise — callers handle pacing and coil shutdown).
inline void stepAxisOnce(const int pins[4], int &stepIndex, int dir) {
  stepIndex = (stepIndex + dir + 8) % 8;
  for (int p = 0; p < 4; p++)
    digitalWrite(pins[p], STEP_SEQ[stepIndex][p]);
}

inline void deenergise(const int pins[4]) {
  for (int p = 0; p < 4; p++) digitalWrite(pins[p], LOW);
}

// Drive a single stepper n half-steps at the given delay.
void stepN(const int pins[4], int &stepIndex, int n, int delayUs) {
  int dir   = (n >= 0) ? 1 : -1;
  long count = abs((long)n);
  for (long i = 0; i < count; i++) {
    stepAxisOnce(pins, stepIndex, dir);
    delayMicroseconds(delayUs);
  }
  deenergise(pins);
}

// Drive BOTH steppers concurrently.  A Bresenham distribution spreads the
// smaller move evenly across the larger one, so both finish together and the
// stage travels a straight diagonal.  Paced at the X (fast) delay; since Y is
// far shorter it ends up stepping sparsely and therefore well within its safe
// rate, so accuracy is preserved on both axes.
void moveXY(int nx, int ny) {
  int  dirx = (nx >= 0) ? 1 : -1;
  int  diry = (ny >= 0) ? 1 : -1;
  long ax = abs((long)nx);
  long ay = abs((long)ny);
  long steps = max(ax, ay);
  long errx = 0, erry = 0;

  for (long i = 0; i < steps; i++) {
    errx += ax;
    if (errx >= steps) { errx -= steps; stepAxisOnce(X_PINS, xStepIndex, dirx); }
    erry += ay;
    if (erry >= steps) { erry -= steps; stepAxisOnce(Y_PINS, yStepIndex, diry); }
    delayMicroseconds(X_STEP_DELAY_US);
  }
  deenergise(X_PINS);
  deenergise(Y_PINS);
}

// ── Serial command parser ─────────────────────────────────────────────────
String inputLine = "";   // accumulates characters until a newline arrives

void handleCommand(String cmd) {
  cmd.trim();

  if (cmd == "HOME") {
    Serial.println("OK");
    return;
  }

  // MOVE XY <xsteps> <ysteps>  (check before "MOVE X " so the prefix matches)
  if (cmd.startsWith("MOVE XY ")) {
    String rest = cmd.substring(8);
    rest.trim();
    int sp = rest.indexOf(' ');
    if (sp < 0) { Serial.println("ERR MOVE XY needs two values"); return; }
    int nx = rest.substring(0, sp).toInt();
    int ny = rest.substring(sp + 1).toInt();
    moveXY(nx, ny);
    Serial.println("OK");
    return;
  }

  if (cmd.startsWith("MOVE X ")) {
    int n = cmd.substring(7).toInt();
    stepN(X_PINS, xStepIndex, n, X_STEP_DELAY_US);
    Serial.println("OK");
    return;
  }

  if (cmd.startsWith("MOVE Y ")) {
    int n = cmd.substring(7).toInt();
    stepN(Y_PINS, yStepIndex, n, Y_STEP_DELAY_US);
    Serial.println("OK");
    return;
  }

  Serial.print("ERR unknown command: ");
  Serial.println(cmd);
}

// ── Initialisation ────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  for (int p = 0; p < 4; p++) {
    pinMode(X_PINS[p], OUTPUT); digitalWrite(X_PINS[p], LOW);
    pinMode(Y_PINS[p], OUTPUT); digitalWrite(Y_PINS[p], LOW);
  }

  Serial.println("READY");
}

// ── Main loop — read serial bytes and build commands ──────────────────────
void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      handleCommand(inputLine);
      inputLine = "";
    } else {
      inputLine += c;
    }
  }
}
