/*
  ESP32 Motor Controller for Microscope Slide Scanner
  ====================================================
  Receives plain-text commands from Python (motor.py) over USB serial at
  115200 baud.  Each command ends with a newline '\n'.
  Replies "OK\n" on success or "ERR <reason>\n" on failure.

  Supported commands:
    MOVE X <steps>          — drive X stepper ± <steps> half-steps (fast axis)
    MOVE Y <steps>          — drive Y stepper ± <steps> half-steps (slow axis)
    MOVE Z <steps>          — drive Z stepper ± <steps> half-steps (focus axis)
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
    Z stepper: GPIO 13, 14, 16, 4   (focus knob; motors powered from the 5V bank,
                                     ESP32 on USB — share a COMMON GROUND)
*/

// ── Optional wireless transport ───────────────────────────────────────────
// Wired USB serial is the default and is ALWAYS active.  Set USE_WIFI to 1 and
// fill in the credentials below to ALSO accept the identical command protocol
// over a TCP socket (the PC app's "Wireless" mode connects to <ESP32 IP>:3232).
// With USE_WIFI 0 the WiFi code is not compiled at all, so the default build is
// byte-for-byte the serial-only firmware — flip this only for remote/demo use.
#define USE_WIFI 1
#if USE_WIFI
  #include <WiFi.h>
  const char*    WIFI_SSID = "Alchemy";
  const char*    WIFI_PASS = "Exoshield6X";
  const uint16_t WIFI_PORT = 3232;          // must match _TCP_PORT in motor.py
  WiFiServer wifiServer(WIFI_PORT);
  WiFiClient wifiClient;
  String     wifiLine = "";
#endif

// ── Stepper pin assignments ───────────────────────────────────────────────
const int X_PINS[4] = {19, 18, 5, 17};
const int Y_PINS[4] = {27, 26, 25, 33};
const int Z_PINS[4] = {13, 14, 16, 4};   // focus stepper (autofocus) — DevKit-V1 safe GPIOs

// ── Leg-done buzzer ───────────────────────────────────────────────────────
// Active 5V buzzer on a spare GPIO.  Driven with analogWrite (PWM duty) so the
// GUI volume slider maps to duty 0-255 — note an ACTIVE buzzer self-oscillates,
// so duty gives only coarse volume control (a passive piezo would give fine
// control).  The beep is NON-BLOCKING: BEEP arms it and returns OK immediately,
// and loop() switches it off when its window expires, so it never holds up a
// command or a move.  Wiring: buzzer + to GPIO 23, buzzer - to GND.
const int BUZZER_PIN = 23;
unsigned long buzzerOffAt = 0;    // millis() deadline; 0 = idle

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
// feels slower — but 700 µs on the Y motor skipped steps and made the sweep
// jumpy/inaccurate, so it is back at the validated-safe 900 µs.  The sweep's
// extra runtime is inherent to its longer travel, not the rate; don't drop
// below 900 (prior validated-safe rate was ~1200 µs).
const int X_STEP_DELAY_US = 900;   // rung axis (short moves)
const int Y_STEP_DELAY_US = 900;   // sweep axis — 900 is the safe floor (700 skips)
const int Z_STEP_DELAY_US = 900;   // focus axis — small moves, same safe rate

int xStepIndex = 0;   // current position in the 8-step table for X
int yStepIndex = 0;   // current position in the 8-step table for Y
int zStepIndex = 0;   // current position in the 8-step table for Z

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

// `out` is where the reply goes: Serial for a wired command, the WiFiClient for
// a wireless one.  Both are Arduino Print streams, so the body is unchanged.
void handleCommand(String cmd, Print &out) {
  cmd.trim();

  if (cmd == "HOME") {
    out.println("OK");
    return;
  }

  // MOVE XY <xsteps> <ysteps>  (check before "MOVE X " so the prefix matches)
  if (cmd.startsWith("MOVE XY ")) {
    String rest = cmd.substring(8);
    rest.trim();
    int sp = rest.indexOf(' ');
    if (sp < 0) { out.println("ERR MOVE XY needs two values"); return; }
    int nx = rest.substring(0, sp).toInt();
    int ny = rest.substring(sp + 1).toInt();
    moveXY(nx, ny);
    out.println("OK");
    return;
  }

  if (cmd.startsWith("MOVE X ")) {
    int n = cmd.substring(7).toInt();
    stepN(X_PINS, xStepIndex, n, X_STEP_DELAY_US);
    out.println("OK");
    return;
  }

  if (cmd.startsWith("MOVE Y ")) {
    int n = cmd.substring(7).toInt();
    stepN(Y_PINS, yStepIndex, n, Y_STEP_DELAY_US);
    out.println("OK");
    return;
  }

  if (cmd.startsWith("MOVE Z ")) {
    int n = cmd.substring(7).toInt();
    stepN(Z_PINS, zStepIndex, n, Z_STEP_DELAY_US);
    out.println("OK");
    return;
  }

  // BEEP <duty 0-255> <ms>  — arm the buzzer; loop() turns it off after <ms>.
  if (cmd.startsWith("BEEP ")) {
    String rest = cmd.substring(5);
    rest.trim();
    int sp   = rest.indexOf(' ');
    int duty = rest.substring(0, sp < 0 ? rest.length() : sp).toInt();
    long ms  = (sp < 0) ? 2000 : rest.substring(sp + 1).toInt();
    duty = constrain(duty, 0, 255);
    if (duty <= 0 || ms <= 0) {          // silent / off
      analogWrite(BUZZER_PIN, 0);
      buzzerOffAt = 0;
    } else {
      analogWrite(BUZZER_PIN, duty);
      buzzerOffAt = millis() + (unsigned long)ms;
    }
    out.println("OK");
    return;
  }

  out.print("ERR unknown command: ");
  out.println(cmd);
}

// ── Initialisation ────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  for (int p = 0; p < 4; p++) {
    pinMode(X_PINS[p], OUTPUT); digitalWrite(X_PINS[p], LOW);
    pinMode(Y_PINS[p], OUTPUT); digitalWrite(Y_PINS[p], LOW);
    pinMode(Z_PINS[p], OUTPUT); digitalWrite(Z_PINS[p], LOW);
  }

  pinMode(BUZZER_PIN, OUTPUT);
  analogWrite(BUZZER_PIN, 0);          // buzzer silent at boot

#if USE_WIFI
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  // Wait briefly for a connection; don't block forever if WiFi is unavailable
  // so the wired path still comes up.
  for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) delay(250);
  if (WiFi.status() == WL_CONNECTED) {
    wifiServer.begin();
    wifiServer.setNoDelay(true);           // send each reply immediately (low latency)
    Serial.print("WIFI ");
    Serial.println(WiFi.localIP());        // note the IP to type into the app
  } else {
    Serial.println("WIFI FAILED");
  }
#endif

  Serial.println("READY");
}

// ── Main loop — read command bytes from whichever transport is active ──────
void loop() {
  // Turn the buzzer off once its beep window has elapsed (signed compare is
  // rollover-safe).  Non-blocking, so it never delays command handling.
  if (buzzerOffAt != 0 && (long)(millis() - buzzerOffAt) >= 0) {
    analogWrite(BUZZER_PIN, 0);
    buzzerOffAt = 0;
  }

  // Wired USB serial (always active)
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      handleCommand(inputLine, Serial);
      inputLine = "";
    } else {
      inputLine += c;
    }
  }

#if USE_WIFI
  // Wireless: accept one client at a time and read its command lines.
  if (!wifiClient || !wifiClient.connected()) {
    wifiClient = wifiServer.available();   // adopt a newly-connected client
    wifiLine   = "";
  }
  if (wifiClient && wifiClient.connected()) {
    while (wifiClient.available()) {
      char c = wifiClient.read();
      if (c == '\n') {
        handleCommand(wifiLine, wifiClient);
        wifiLine = "";
      } else {
        wifiLine += c;
      }
    }
  }
#endif
}
