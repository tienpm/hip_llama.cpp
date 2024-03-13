#include "utils.hpp"

void rmsnorm(float* o, float* x, float* weight, int size);

void softmax(float* x, int size);

// C B A K M
// N = 1
void matmul(float* xout, float* x, float* w, int n, int d);

float* forward(Transformer* transformer, int token, int pos);
