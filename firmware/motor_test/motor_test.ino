/*
  Motor Test Sketch — 28BYJ-48 Dual-Axis Stepper
  ================================================
  USE THIS SKETCH to verify that both stepper motors are wired and
  moving correctly BEFORE flashing firmware.ino.

  Open the Arduino Serial Monitor at 115200 baud, then type:
    a <rotations>   — rotate X-axis stepper  (e.g.  a 1,  a -0.5,  a 2.5)
    b <rotations>   — rotate Y-axis stepper  (e.g.  b 1,  b -0.5,  b 2.5)

  IMPORTANT: pin numbers here may differ from firmware.ino because this was
  written before final wiring was confirmed.  Adjust #defines to match your
  actual wiring before use.

  4096 half-steps = 1 full rotation of the 28BYJ-48 output shaft.
  step delay = 2 ms → one rotation takes ~8 seconds at this speed.
*/

// ── X-axis stepper pin assignments ────────────────────────────────────────
// Connect these GPIO pins to the IN1-IN4 inputs on the ULN2003 driver board.
#define IN1_X 19
#define IN2_X 18
#define IN3_X 5
#define IN4_X 17

// ── Y-axis stepper pin assignments ────────────────────────────────────────
#define IN1_Y 16
#define IN2_Y 4
#define IN3_Y 2
#define IN4_Y 15

// Half-step sequence: 8 rows, each row drives one electrical step.
// Energising pairs of coils produces smoother motion and more torque.
int step_sequence[8][4] = {
  {1,0,0,0},
  {1,1,0,0},
  {0,1,0,0},
  {0,1,1,0},
  {0,0,1,0},
  {0,0,1,1},
  {0,0,0,1},
  {1,0,0,1}
};

int currentStepX = 0;   // current position in the 8-step sequence for X
int currentStepY = 0;   // current position in the 8-step sequence for Y

/*
  stepMotor() — drive a stepper a given number of half-steps.
    in1-in4    : GPIO pin numbers connected to the ULN2003 IN1-IN4 inputs
    steps      : number of half-steps (positive = forward, negative = reverse)
    currentStep: reference to the axis's position counter (updated in place)
*/
void stepMotor(int in1, int in2, int in3, int in4, int steps, int &currentStep) {
  int dir   = steps > 0 ? 1 : -1;
  steps     = abs(steps);

  for (int i = 0; i < steps; i++) {
    currentStep = (currentStep + dir + 8) % 8;   // advance through the 8-step table
    // Apply the coil pattern for this step
    digitalWrite(in1, step_sequence[currentStep][0]);
    digitalWrite(in2, step_sequence[currentStep][1]);
    digitalWrite(in3, step_sequence[currentStep][2]);
    digitalWrite(in4, step_sequence[currentStep][3]);
    delay(2);   // 2 ms per half-step — increase if motor stalls
  }

  // De-energise all coils to prevent overheating while idle
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
}

// ── One-time setup ────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  // Configure all stepper control pins as outputs
  pinMode(IN1_X, OUTPUT); pinMode(IN2_X, OUTPUT);
  pinMode(IN3_X, OUTPUT); pinMode(IN4_X, OUTPUT);
  pinMode(IN1_Y, OUTPUT); pinMode(IN2_Y, OUTPUT);
  pinMode(IN3_Y, OUTPUT); pinMode(IN4_Y, OUTPUT);

  // Print usage instructions to Serial Monitor
  Serial.println("Ready. Commands:");
  Serial.println("  a <rotations>  — X axis (e.g. a 1, a -0.5, a 2.5)");
  Serial.println("  b <rotations>  — Y axis (e.g. b 1, b -0.5, b 2.5)");
}

// ── Main loop — wait for serial input and drive motors ────────────────────
void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();          // 'a' for X axis, 'b' for Y axis
    while (Serial.peek() == ' ')      // consume any spaces before the number
      Serial.read();

    if (cmd == 'a' || cmd == 'b') {
      float rotations = Serial.parseFloat();        // e.g. 1.5
      int   steps     = (int)(rotations * 4096);    // convert rotations → half-steps

      Serial.print(cmd == 'a' ? "X axis: " : "Y axis: ");
      Serial.print(rotations);
      Serial.println(" rotations");

      // Call stepMotor for the correct axis
      if (cmd == 'a')
        stepMotor(IN1_X, IN2_X, IN3_X, IN4_X, steps, currentStepX);
      else
        stepMotor(IN1_Y, IN2_Y, IN3_Y, IN4_Y, steps, currentStepY);

      Serial.println("Done.");
    }
  }
}