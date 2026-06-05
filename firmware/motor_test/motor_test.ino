#define IN1_X 19
#define IN2_X 18
#define IN3_X 5
#define IN4_X 17

#define IN1_Y 16
#define IN2_Y 4
#define IN3_Y 2
#define IN4_Y 15

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

int currentStepX = 0;
int currentStepY = 0;

void stepMotor(int in1, int in2, int in3, int in4, int steps, int &currentStep) {
  int dir = steps > 0 ? 1 : -1;
  steps = abs(steps);
  for (int i = 0; i < steps; i++) {
    currentStep = (currentStep + dir + 8) % 8;
    digitalWrite(in1, step_sequence[currentStep][0]);
    digitalWrite(in2, step_sequence[currentStep][1]);
    digitalWrite(in3, step_sequence[currentStep][2]);
    digitalWrite(in4, step_sequence[currentStep][3]);
    delay(2);
  }
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
}

void setup() {
  Serial.begin(115200);
  pinMode(IN1_X, OUTPUT); pinMode(IN2_X, OUTPUT);
  pinMode(IN3_X, OUTPUT); pinMode(IN4_X, OUTPUT);
  pinMode(IN1_Y, OUTPUT); pinMode(IN2_Y, OUTPUT);
  pinMode(IN3_Y, OUTPUT); pinMode(IN4_Y, OUTPUT);

  Serial.println("Ready. Commands:");
  Serial.println("  a <rotations>  — X axis (e.g. a 1, a -0.5, a 2.5)");
  Serial.println("  b <rotations>  — Y axis (e.g. b 1, b -0.5, b 2.5)");
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    while (Serial.peek() == ' ') Serial.read();

    if (cmd == 'a' || cmd == 'b') {
      float rotations = Serial.parseFloat();
      int steps = (int)(rotations * 4096);
      Serial.print(cmd == 'a' ? "X axis: " : "Y axis: ");
      Serial.print(rotations);
      Serial.println(" rotations");
      if (cmd == 'a') stepMotor(IN1_X, IN2_X, IN3_X, IN4_X, steps, currentStepX);
      else            stepMotor(IN1_Y, IN2_Y, IN3_Y, IN4_Y, steps, currentStepY);
      Serial.println("Done.");
    }
  }
}