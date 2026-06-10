/*
  ESP32 Motor Controller for Microscope Slide Scanner
  ====================================================
  Receives plain-text commands from Python (motor.py) over USB serial at
  115200 baud.  Each command ends with a newline '\n'.
  Replies "OK\n" on success or "ERR <reason>\n" on failure.

  Supported commands:
    MOVE X <steps>    — drive X stepper ± <steps> half-steps (28BYJ-48 via ULN2003)
    MOVE Y <units>    — rotate Y continuous servo ± <units> (1 unit ≈ 50 ms burst)
    HOME              — acknowledge only — resets logical position in Python

  Hardware wiring (update these constants to match your build):
    Stepper IN1-IN4 coils: GPIO 16, 17, 18, 19   (via ULN2003 driver board)
    Continuous servo signal: GPIO 21              (e.g. MG996R or similar)

  Motor notes:
    28BYJ-48 is a 4-phase unipolar stepper.  Half-stepping (8 phases) gives
    smoother motion than full-step (4 phases) and doubles effective resolution.
    4096 half-steps ≈ 1 full revolution of the output shaft.
    Coils are de-energised after each move to prevent heat build-up.
*/

#include <ESP32Servo.h>     // ESP32-compatible servo library (not the Arduino built-in)

// ── X axis: 28BYJ-48 stepper via ULN2003 ──────────────────────────────────
// Four output pins drive the four coil phases.
const int STEP_PINS[4] = {16, 17, 18, 19};

// Half-step sequence: each row is one step, columns are coil states (HIGH/LOW).
// 8 steps per electrical cycle; cycling through these drives the motor smoothly.
const int STEP_SEQ[8][4] = {
  {1,0,0,0}, {1,1,0,0}, {0,1,0,0}, {0,1,1,0},
  {0,0,1,0}, {0,0,1,1}, {0,0,0,1}, {1,0,0,1}
};
const int STEP_DELAY_US = 1200;  // delay between half-steps in µs (lower = faster, may skip steps)
int stepIndex = 0;               // current position in the 8-step sequence

// Step the motor once in the given direction (+1 forward, -1 backward)
void stepOnce(int direction) {
  stepIndex = (stepIndex + direction + 8) % 8;   // wrap 0-7
  for (int i = 0; i < 4; i++)
    digitalWrite(STEP_PINS[i], STEP_SEQ[stepIndex][i]);
  delayMicroseconds(STEP_DELAY_US);
}

// Move n half-steps (positive = forward, negative = reverse)
void stepN(int n) {
  int dir   = (n >= 0) ? 1 : -1;
  int count = abs(n);
  for (int i = 0; i < count; i++) stepOnce(dir);
  // De-energise all coils — prevents motor getting hot when idle
  for (int i = 0; i < 4; i++) digitalWrite(STEP_PINS[i], LOW);
}

// ── Y axis: continuous rotation servo ─────────────────────────────────────
// A continuous servo spins rather than holding an angle.
// Pulse width controls direction and speed:
//   1500 µs = stop
//   < 1500  = clockwise (speed proportional to deviation from 1500)
//   > 1500  = counter-clockwise
Servo yServo;
const int SERVO_PIN    = 21;
const int SERVO_CW_US  = 1350;   // moderate CW speed
const int SERVO_CCW_US = 1650;   // moderate CCW speed
const int SERVO_STOP_US = 1500;  // standstill
const int UNIT_MS = 50;          // milliseconds of movement per "unit"

// Rotate the servo for abs(units)*UNIT_MS milliseconds in the given direction
void moveServo(int units) {
  if (units == 0) return;
  int pulse    = (units > 0) ? SERVO_CW_US : SERVO_CCW_US;
  int duration = abs(units) * UNIT_MS;   // total run time in ms
  yServo.writeMicroseconds(pulse);       // start spinning
  delay(duration);                        // run for the required time
  yServo.writeMicroseconds(SERVO_STOP_US); // stop
}

// ── Serial command parser ─────────────────────────────────────────────────
String inputLine = "";   // accumulates characters until a newline arrives

void handleCommand(String cmd) {
  cmd.trim();   // remove whitespace / CR

  // HOME: Python resets its logical position — no physical movement needed here
  if (cmd == "HOME") {
    Serial.println("OK");
    return;
  }

  // MOVE X <steps>: drive stepper the requested number of half-steps
  if (cmd.startsWith("MOVE X ")) {
    int n = cmd.substring(7).toInt();   // parse integer after "MOVE X "
    stepN(n);
    Serial.println("OK");
    return;
  }

  // MOVE Y <units>: spin servo for <units> × UNIT_MS milliseconds
  if (cmd.startsWith("MOVE Y ")) {
    int n = cmd.substring(7).toInt();
    moveServo(n);
    Serial.println("OK");
    return;
  }

  // Unknown command — report back so Python can surface the error
  Serial.print("ERR unknown command: ");
  Serial.println(cmd);
}

// ── Initialisation ────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);   // must match _BAUD in motor.py

  // Initialise stepper pins as outputs, start LOW (coils off)
  for (int i = 0; i < 4; i++) {
    pinMode(STEP_PINS[i], OUTPUT);
    digitalWrite(STEP_PINS[i], LOW);
  }

  // Attach servo and make sure it starts stopped
  yServo.attach(SERVO_PIN);
  yServo.writeMicroseconds(SERVO_STOP_US);

  Serial.println("READY");   // signals to Python that firmware is booted
}

// ── Main loop — read serial bytes and build commands ──────────────────────
void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      // Newline terminates a command — process and clear the buffer
      handleCommand(inputLine);
      inputLine = "";
    } else {
      inputLine += c;   // append character to current command
    }
  }
}
