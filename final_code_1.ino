#include <Servo.h>
const int trigPin = 12;
const int echoPin = 11;
const int lmf = 6; //lm1
const int lmr = 7; //lm2
const int rmf = 4; //rm1
const int rmr = 5; //rm2
const int servoPin = 13;

Servo myservo;

long duration;
int distance;
int distance_f;
int distance_l;
int distance_r;
int maxLowDistance = 20;


void setup() {
  // put your setup code here, to run once:
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode(lmr, OUTPUT);
  pinMode(lmf, OUTPUT);
  pinMode(rmr, OUTPUT);
  pinMode(rmf, OUTPUT);

  myservo.attach(servoPin);
  myservo.write(90);
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  distance_f = ping();
  if (distance_f > maxLowDistance){
    front();
    delay(300);
    }else{
      Break();
      get_distance();
      if (distance_r > maxLowDistance){
        right();
        delay(300);
        front();
        }else if (distance_l > maxLowDistance){
          left();
          delay(300);
          front();
          }else{
            back();
            delay(300);
            Break();
            }
      }
}

void get_distance(){

  myservo.write(45);
  delay(500);
  distance_r = ping();
  
  myservo.write(90);
  delay(500);
  distance_f = ping();

  myservo.write(135);
  delay(500);
  int templ1 = ping();
  distance_l = templ1;
  
  myservo.write(90);
 
}

void displayDistance(){
  Serial.print("Front Distance : ");
  Serial.print(distance_f);
  Serial.println("");
  Serial.print("Right Distance : ");
  Serial.print(distance_r);
  Serial.println("");
  Serial.print("Left Distance : ");
  Serial.print(distance_l);
  Serial.println("");
  }
void front(){
  digitalWrite(lmf, HIGH);
  digitalWrite(rmf, HIGH);
  digitalWrite(lmr, LOW);
  digitalWrite(rmr, LOW);
}

void back(){
  digitalWrite(lmf, LOW);
  digitalWrite(rmf, LOW);
  digitalWrite(lmr, HIGH);
  digitalWrite(rmr, HIGH);
}

void left(){
  digitalWrite(rmf, HIGH);
  digitalWrite(lmr, HIGH);
  digitalWrite(rmr, LOW);
  digitalWrite(lmf, LOW);
}

void right(){
  digitalWrite(lmf, HIGH);
  digitalWrite(rmr, HIGH);
  digitalWrite(lmr, LOW);
  digitalWrite(rmf, LOW);
}

void Break(){
  digitalWrite(lmr, LOW);
  digitalWrite(lmf, LOW);
  digitalWrite(rmr, LOW);
  digitalWrite(rmf, LOW);
}

int ping(){
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  duration = pulseIn(echoPin, HIGH);
  distance = duration*0.034/2.0;
  return distance;
  }
