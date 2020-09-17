typedef struct {
  float *array;
  size_t used;
  size_t size;
} Array;

void initArray(Array *a, size_t initialSize) {
  a->array = (float *)malloc(initialSize * sizeof(float));
  a->used = 0;
  a->size = initialSize;
}

void insertArray(Array *a, float element) {
  // a->used is the number of used entries, because a->array[a->used++] updates a->used only *after* the array has been accessed.
  // Therefore a->used can go up to a->size 
  if (a->used == a->size) {
    a->size *= 2;
    a->array = (float *)realloc(a->array, a->size * sizeof(float));
  }
  a->array[a->used++] = element;
}

void freeArray(Array *a) {
  free(a->array);
  a->array = NULL;
  a->used = a->size = 0;
}

//code source: https://stackoverflow.com/questions/3536153/c-dynamically-growing-array

void printArray(Array *a) {
  for(unsigned long count = 0; count < a->used; count++)
  {
    Serial.print(a->array[count]);
    if(count == a->used - 1)
    {
      continue;
    }
    else {
      Serial.print(",");
    }
  }
  Serial.println();
}

int i = 0;
int j = 0;
Array voltageArray;
Array currentArray;
int samplePerCycle;
int indexStart;
int indexLast;
Array voltageCycle;
Array currentCycle;
boolean firstCycle = true;
unsigned long lastMillis;
unsigned long timeCheckInterval = 80;

void receiveData() {
  float voltageReceived = float(10*sin(j*PI/180));
  float currentReceived = float(10*sin((i + j)*PI/180));
  if(j == 360){
    i++;
    j = 0;
  }
  if (i == 360){
    i = 0;
  }
  j++;
  insertArray(&voltageArray, voltageReceived);
  insertArray(&currentArray, currentReceived);
}

boolean cycleCheck() {
  for(unsigned int count = 0; count < voltageArray.used; count++){ // one for buffer_size \0, and one for calculating diff
    if(voltageArray.array[count] <= 0 && voltageArray.array[count + 1] > 0)
    {
      // first starting point
      indexStart = count;
      for(count = count + 1; count < voltageArray.used - 1; count++)
      {
        if(voltageArray.array[count] <= 0 && voltageArray.array[count + 1] > 0)
        {   
          indexLast = count;
          samplePerCycle = indexLast - indexStart;
          return true; 
        }
      }
      return false;
    }
  }
  return false;
}

void copyArray(Array *src, Array *dst) {
  for (int count = 0; count < samplePerCycle; count++) {
    insertArray(dst, src->array[indexStart + count]);
  }
}

boolean timeCheck(unsigned long *lastMillis)
{
//   Serial.println("In time Check");
   unsigned long currentMillis = millis();
   unsigned long actualTimeInterval = currentMillis - *lastMillis;
//   Serial.print("actualTimeInterval ");
//   Serial.println(actualTimeInterval);
   if(actualTimeInterval >= timeCheckInterval)
   {
     *lastMillis = millis();
     return true;
   }
   else
     return false;
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  initArray(&voltageArray, 10);
  initArray(&currentArray, 10);
  delay(2000);
  lastMillis = millis();
}

void loop() {
  receiveData();
  if (cycleCheck()) {
    if (!firstCycle) {
      freeArray(&voltageCycle);
      freeArray(&currentCycle);
    }
    
    initArray(&voltageCycle, samplePerCycle);
    initArray(&currentCycle, samplePerCycle);
    copyArray(&voltageArray, &voltageCycle);
    copyArray(&currentArray, &currentCycle);
    
    freeArray(&voltageArray);
    freeArray(&currentArray);
    initArray(&voltageArray, 10);
    initArray(&currentArray, 10);
    firstCycle = false;
  }
  if (timeCheck(&lastMillis)) {
    printArray(&voltageCycle);
    printArray(&currentCycle);  
  }
//  delay(1);
//  delayMicroseconds(1);
  // put your main code here, to run repeatedly:

}
