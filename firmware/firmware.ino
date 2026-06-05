/*
  ESP32 Motor Controller for Microscope Slide Scanner
  ====================================================
  Serial commands (115200 baud, newline-terminated):
    MOVE X <steps>    — step X stepper ± steps (28BYJ-48 via ULN2003)
    MOVE Y <units>    — rotate Y servo ± units (1 unit ≈ small pulse burst)
    HOME              — reset logical position to 0,0 (no physical movement)

  Responses: "OK\n" or "ERR <message>\n"

  Pinout (adjust to your wiring):
    Stepper IN1-IN4: GPIO 16, 17, 18, 19
    Servo signal:    GPIO 21
*/

#include <ESP32Servo.h>

// ── Stepper (28BYJ-48 half-step sequence) ─────────────────────────────────
const int STEP_PINS[4] = {16, 17, 18, 19};
const int STEP_SEQ[8][4] = {
  {1,0,0,0}, {1,1,0,0}, {0,1,0,0}, {0,1,1,0},
  {0,0,1,0}, {0,0,1,1}, {0,0,0,1}, {1,0,0,1}
};
const int STEP_DELAY_US = 1200;  // microseconds per half-step
int stepIndex = 0;

void stepOnce(int direction) {
  stepIndex = (stepIndex + direction + 8) % 8;
  for (int i = 0; i < 4; i++)
    digitalWrite(STEP_PINS[i], STEP_SEQ[stepIndex][i]);
  delayMicroseconds(STEP_DELAY_US);
}

void stepN(int n) {
  int dir = (n >= 0) ? 1 : -1;
  int count = abs(n);
  for (int i = 0; i < count; i++) stepOnce(dir);
  // Release coils to save power
  for (int i = 0; i < 4; i++) digitalWrite(STEP_PINS[i], LOW);
}

// ── Servo (continuous rotation) ───────────────────────────────────────────
Servo yServo;
const int SERVO_PIN = 21;
// Pulse widths: 1500 = stop, <1500 = CW, >1500 = CCW
// One "unit" = 50 ms of movement at moderate speed
const int SERVO_CW_US  = 1350;
const int SERVO_CCW_US = 1650;
const int SERVO_STOP_US = 1500;
const int UNIT_MS = 50;

void moveServo(int units) {
  if (units == 0) return;
  int pulse = (units > 0) ? SERVO_CW_US : SERVO_CCW_US;
  int duration = abs(units) * UNIT_MS;
  yServo.writeMicroseconds(pulse);
  delay(duration);
  yServo.writeMicroseconds(SERVO_STOP_US);
}

// ── Command parser ────────────────────────────────────────────────────────
String inputLine = "";

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd == "HOME") {
    Serial.println("OK");
    return;
  }
  if (cmd.startsWith("MOVE X ")) {
    int n = cmd.substring(7).toInt();
    stepN(n);
    Serial.println("OK");
    return;
  }
  if (cmd.startsWith("MOVE Y ")) {
    int n = cmd.substring(7).toInt();
    moveServo(n);
    Serial.println("OK");
    return;
  }
  Serial.print("ERR unknown command: ");
  Serial.println(cmd);
}

// ── Setup / loop ──────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 4; i++) {
    pinMode(STEP_PINS[i], OUTPUT);
    digitalWrite(STEP_PINS[i], LOW);
  }
  yServo.attach(SERVO_PIN);
  yServo.writeMicroseconds(SERVO_STOP_US);
  Serial.println("READY");
}

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
