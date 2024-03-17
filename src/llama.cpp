/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <omp.h>
#include <mutex>

#include <fstream>
#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
// #include <rccl/rccl.h>  // RCCL

#include "seq.hpp"
#include "thaDNN.hpp"
#include "thaBLAS.hpp"
#include "utils.hpp"



// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
  char *str;
  int id;
} TokenIndex;

typedef struct {
  char** vocab;
  float* vocab_scores;
  TokenIndex *sorted_vocab;
  int vocab_size;
  unsigned int max_token_length;
  unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
  return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
  // i should have written the vocab_size into the tokenizer file... sigh
  t->vocab_size = vocab_size;
  // malloc space to hold the scores and the strings
  t->vocab = (char**)malloc(vocab_size * sizeof(char*));
  t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
  t->sorted_vocab = NULL; // initialized lazily
  for (int i = 0; i < 256; i++) {
    t->byte_pieces[i * 2] = (unsigned char)i;
    t->byte_pieces[i * 2 + 1] = '\0';
  }
  // read in the file
  FILE *file = fopen(tokenizer_path, "rb");
  if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
  if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
  int len;
  for (int i = 0; i < vocab_size; i++) {
    if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
    if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    t->vocab[i] = (char *)malloc(len + 1);
    if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    t->vocab[i][len] = '\0'; // add the string terminating token
  }
  fclose(file);
}

void free_tokenizer(Tokenizer* t) {
  for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
  free(t->vocab);
  free(t->vocab_scores);
  free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
  char *piece = t->vocab[token];
  // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
  if (prev_token == 1 && piece[0] == ' ') { piece++; }
  // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
  // parse this and convert and return the actual byte
  unsigned char byte_val;
  if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
    piece = (char*)t->byte_pieces + byte_val * 2;
  }
  return piece;
}

void safe_printf(char *piece) {
  // piece might be a raw byte token, and we only want to print printable chars or whitespace
  // because some of the other bytes can be various control codes, backspace, etc.
  if (piece == NULL) { return; }
  if (piece[0] == '\0') { return; }
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return; // bad byte, don't print it
    }
  }
  printf("%s", piece);
}

void append_str(char *piece, std::string& str) {
  // piece might be a raw byte token, and we only want to print printable chars or whitespace
  // because some of the other bytes can be various control codes, backspace, etc.
  if (piece == NULL) { return; }
  if (piece[0] == '\0') { return; }
  if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
      return; // bad byte, don't print it
    }
  }
  //printf("%s", piece);
  str += piece;
}



int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
  // efficiently find the perfect match for str in vocab, return its index or -1 if not found
  TokenIndex tok = { .str = str }; // acts as the key to search for
  TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
  return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
  // encode the string text (input) into an upper-bound preallocated tokens[] array
  // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
  if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

  if (t->sorted_vocab == NULL) {
    // lazily malloc and sort the vocabulary
    t->sorted_vocab = (TokenIndex *)malloc(t->vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < t->vocab_size; i++) {
      t->sorted_vocab[i].str = t->vocab[i];
      t->sorted_vocab[i].id = i;
    }
    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
  }

  // create a temporary buffer that will store merge candidates of always two consecutive tokens
  // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
  char* str_buffer = (char*)malloc((t->max_token_length*2 +1 +2) * sizeof(char));
  size_t str_len = 0;

  // start at 0 tokens
  *n_tokens = 0;

  // add optional BOS (=1) token, if desired
  if (bos) tokens[(*n_tokens)++] = 1;

  // add_dummy_prefix is true by default
  // so prepend a dummy prefix token to the input string, but only if text != ""
  // TODO: pretty sure this isn't correct in the general case but I don't have the
  // energy to read more of the sentencepiece code to figure out what it's doing
  if (text[0] != '\0') {
    int dummy_prefix = str_lookup((char *)" ", t->sorted_vocab, t->vocab_size);
    tokens[(*n_tokens)++] = dummy_prefix;
  }

  // fprintf(stderr, "\nDEBUG 1.1\n");
  // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
  // Code point ↔ UTF-8 conversion
  // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
  // U+0000	U+007F	    0xxxxxxx
  // U+0080	U+07FF	    110xxxxx	10xxxxxx
  // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
  // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

  // process the raw (UTF-8) byte sequence of the input string
  for (char *c = text; *c != '\0'; c++) {

    // reset buffer if the current byte is ASCII or a leading byte
    // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
    // 0x80 is 10000000
    // in UTF-8, all continuation bytes start with "10" in first two bits
    // so in English this is: "if this byte is not a continuation byte"
    if ((*c & 0xC0) != 0x80) {
      // this byte must be either a leading byte (11...) or an ASCII char (0x...)
      // => reset our location, as we're starting a new UTF-8 codepoint
      str_len = 0;
    }

    // append the current byte to the buffer
    str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
    str_buffer[str_len] = '\0';

    // while the next character is a continuation byte, continue appending
    // but if there are too many of them, just stop to avoid overruning str_buffer size.
    if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
      continue;
    }

    // ok c+1 is not a continuation byte, so we've read in a full codepoint
    int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

    if (id != -1) {
      // we found this codepoint in vocab, add it as a token
      tokens[(*n_tokens)++] = id;
    } else {
      // byte_fallback encoding: just encode each byte as a token
      // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
      // so the individual bytes only start at index 3
      for (int i=0; i < str_len; i++) {
        tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
      }
    }
    str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
  }

  // merge the best consecutive pair each iteration, according the scores in vocab_scores
  while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;

    for (int i=0; i < (*n_tokens-1); i++) {
      // check if we can merge the pair (tokens[i], tokens[i+1])
      sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
      int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
      if (id != -1 && t->vocab_scores[id] > best_score) {
        // this merge pair exists in vocab! record its score and position
        best_score = t->vocab_scores[id];
        best_id = id;
        best_idx = i;
      }
    }

    if (best_idx == -1) {
      break; // we couldn't find any more pairs to merge, so we're done
    }

    // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
    tokens[best_idx] = best_id;
    // delete token at position best_idx+1, shift the entire sequence back 1
    for (int i = best_idx+1; i < (*n_tokens-1); i++) {
      tokens[i] = tokens[i+1];
    }
    (*n_tokens)--; // token length decreased
  }

  // add optional EOS (=2) token, if desired
  if (eos) tokens[(*n_tokens)++] = 2;

  free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
  float prob;
  int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
  int vocab_size;
  ProbIndex* probindex; // buffer used in top-p sampling
  float temperature;
  float topp;
  unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
  // return the index that has the highest probability
  int max_i = 0;
  float max_p = probabilities[0];
  for (int i = 1; i < n; i++) {
    if (probabilities[i] > max_p) {
      max_i = i;
      max_p = probabilities[i];
    }
  }
  return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
  ProbIndex* a_ = (ProbIndex*) a;
  ProbIndex* b_ = (ProbIndex*) b;
  if (a_->prob > b_->prob) return -1;
  if (a_->prob < b_->prob) return 1;
  return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
  // top-p sampling (or "nucleus sampling") samples from the smallest set of
  // tokens that exceed probability topp. This way we never sample tokens that
  // have very low probabilities and are less likely to go "off the rails".
  // coin is a random number in [0, 1), usually from random_f32()

  int n0 = 0;
  // quicksort indices in descending order of probabilities
  // values smaller than (1 - topp) / (n - 1) cannot be part of the result
  // so for efficiency we crop these out as candidates before sorting
  const float cutoff = (1.0f - topp) / (n - 1);
  for (int i = 0; i < n; i++) {
    if (probabilities[i] >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].prob = probabilities[i];
      n0++;
    }
  }
  qsort(probindex, n0, sizeof(ProbIndex), compare);

  // truncate the list where cumulative probability exceeds topp
  float cumulative_prob = 0.0f;
  int last_idx = n0 - 1; // in case of rounding errors consider all elements
  for (int i = 0; i < n0; i++) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp) {
      last_idx = i;
      break; // we've exceeded topp by including last_idx
    }
  }

  // sample from the truncated list
  float r = coin * cumulative_prob;
  float cdf = 0.0f;
  for (int i = 0; i <= last_idx; i++) {
    cdf += probindex[i].prob;
    if (r < cdf) {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  // buffer only used with nucleus sampling; may not need but it's ~small
  sampler->probindex = (ProbIndex *)malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
  free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
  // sample the token given the logits and some hyperparameters
  int next;
  if (sampler->temperature == 0.0f) {
    // greedy argmax sampling: take the token with the highest probability
    next = sample_argmax(logits, sampler->vocab_size);
  } else {
    // apply the temperature to the logits
    for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
    // apply softmax to the logits to get the probabilities for next token
    softmax(logits, sampler->vocab_size);
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&sampler->rng_state);
    // we sample from this distribution to get the next token
    if (sampler->topp <= 0 || sampler->topp >= 1) {
      // simply sample from the predicted probability distribution
      next = sample_mult(logits, sampler->vocab_size, coin);
    } else {
      // top-p (nucleus) sampling, clamping the least likely tokens to zero
      next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
    }
  }
  return next;
}

int sample_greedy(Sampler* sampler, float* logits) {
  // sample the token given the logits and some hyperparameters
  int next;
  // greedy argmax sampling: take the token with the highest probability
  next = sample_argmax(logits, sampler->vocab_size);
  return next;
}

int sample_determin(const Sampler* sampler, float* logits, unsigned long long* rng_states, int idx) {
  // sample the token given the logits and some hyperparameters
  int next;
  float temperature = 1.0f;
  // apply the temperature to the logits
  for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= temperature; }
  // apply softmax to the logits to get the probabilities for next token
  softmax(logits, sampler->vocab_size);
  // flip a (float) coin (this is our source of entropy for sampling)
  float coin = random_f32(&rng_states[idx]);

  next = sample_mult(logits, sampler->vocab_size, coin);
  return next;
}

typedef struct {
  int num_reqs;		// number of reqeusts;
  int max_token_len;  // maximum size of token
  int max_seq_len;	// maximum number of sequence
  char* str_reqs;		// buffer for request strings
  char* str_gens;		// buffer for generated strings
} Requests;

void build_requests(Requests* reqs, int num_reqs, int max_token_len, int max_seq_len) {
  reqs->num_reqs = num_reqs;
  reqs->max_token_len = max_token_len;
  reqs->max_seq_len = max_seq_len;
  reqs->str_reqs = (char*)calloc(num_reqs * max_token_len * max_seq_len + 1, sizeof(char));
  reqs->str_gens = (char*)calloc(num_reqs * max_token_len * max_seq_len + 1, sizeof(char));
  printf("requests size = %lu B\n", ((num_reqs * max_token_len * max_seq_len * sizeof(char) +1) * 2));
}

void free_requests(Requests* reqs) {
  free(reqs->str_reqs);
  free(reqs->str_gens);
}

char* get_str_req_ptr(Requests* reqs, int idx) {
  return reqs->str_reqs + idx * reqs->max_token_len * reqs->max_seq_len;
}

char* get_str_gen_ptr(Requests* reqs, int idx) {
  return reqs->str_gens + idx * reqs->max_token_len * reqs->max_seq_len;
}


int read_inputfile(const char* input_filename, int max_token_len, int max_seq_len, Requests* reqs) {
  std::string filename = input_filename;
  int num_reqs= 0;

  printf("max_token_len: %d, max_seq_len: %d\n", max_token_len, max_seq_len);

  std::ifstream openFile(filename.c_str());
  if (openFile.is_open() ) {
    std::string line;

    // Read the number of Requests
    std::getline(openFile, line);
    num_reqs = atoi(line.c_str());

    build_requests(reqs, num_reqs, max_token_len, max_seq_len);

    int idx = 0;
    while(std::getline(openFile, line)) {
      memcpy(get_str_req_ptr(reqs, idx), line.c_str(), line.size());
      idx++;
      if(idx >= num_reqs) break;
    }
    openFile.close();
  }
  else {
    fprintf(stderr, "cannot open the file: %s\n", input_filename);
    exit(EXIT_FAILURE);
  }

  return 0;
}

int write_outputfile(const char* output_filename, Requests* reqs) {
  std::string filename = output_filename;

  // write File
  std::ofstream writeFile(filename.c_str());
  if( writeFile.is_open() ){
    writeFile << reqs->num_reqs << "\n";
    for(int i = 0; i < reqs->num_reqs; i++) {
      writeFile << get_str_gen_ptr(reqs, i) << "\n";
    }
    writeFile.close();
  }
  else {
    fprintf(stderr, "cannot write the file: %s\n", output_filename);
    exit(EXIT_FAILURE);
  }

  return 0;
}



// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
  // return time in milliseconds, for benchmarking the model speed
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
  char *empty_prompt = (char*)"";
  if (prompt == NULL) { prompt = empty_prompt; }

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
  encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  // start the main loop
  long start = 0;  // used to time our code, only initialized after first iteration
  int next;        // will store the next token in the sequence
  int token = prompt_tokens[0]; // kick off with the first token in the prompt
  int pos = 0;     // position in the sequence
  while (pos < steps) {
    //printf("\npos=%d/steps=%d\n", pos, steps);

    // forward the transformer to get logits for the next token
    float* logits = forward(transformer, token, pos);

    // advance the state machine
    if (pos < num_prompt_tokens - 1) {
      // if we are still processing the input prompt, force the next prompt token
      next = prompt_tokens[pos + 1];
    } else {
      // otherwise sample the next token from the logits
      next = sample(sampler, logits);
    }
    pos++;

    // data-dependent terminating condition: the BOS (=1) token delimits sequences
    if (next == 1) { break;
    }

    // print the token as string, decode it with the Tokenizer object
    char* piece = decode(tokenizer, token, next);
    safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
    fflush(stdout);
    token = next;

    // init the timer here because the first iteration can be slower
    if (start == 0) { start = time_in_ms(); }

  }
  printf("\n");

  // report achieved tok/s (pos-1 because the timer starts after first iteration)
  if (pos > 1) {
    long end = time_in_ms();
    fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
  }

  free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
  // read a line from stdin, up to but not including \n
  printf("%s", guide);
  if (fgets(buffer, bufsize, stdin) != NULL) {
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') {
      buffer[len - 1] = '\0'; // strip newline
    }
  }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
    char *cli_user_prompt, char *cli_system_prompt, int steps) {

  // buffers for reading the system prompt and user prompt from stdin
  // you'll notice they are soomewhat haphazardly and unsafely set atm
  char system_prompt[512];
  char user_prompt[512];
  char rendered_prompt[1152];
  int num_prompt_tokens = 0;
  int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
  int user_idx;

  // start the main loop
  int8_t user_turn = 1; // user starts
  int next;        // will store the next token in the sequence
  int token;       // stores the current token to feed into the transformer
  int prev_token;
  int pos = 0;     // position in the sequence
  while (pos < steps) {

    // when it is the user's turn to contribute tokens to the dialog...
    if (user_turn) {
      // get the (optional) system prompt at position 0
      if (pos == 0) {
        // at position 0, the user can also contribute a system prompt
        if (cli_system_prompt == NULL) {
          // system prompt was not passed in, attempt to get it from stdin
          read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
        } else {
          // system prompt was passed in, use it
          strcpy(system_prompt, cli_system_prompt);
        }
      }
      // get the user prompt
      if (pos == 0 && cli_user_prompt != NULL) {
        // user prompt for position 0 was passed in, use it
        strcpy(user_prompt, cli_user_prompt);
      } else {
        // otherwise get user prompt from stdin
        read_stdin("User: ", user_prompt, sizeof(user_prompt));
      }
      // render user/system prompts into the Llama 2 Chat schema
      if (pos == 0 && system_prompt[0] != '\0') {
        char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
        sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
      } else {
        char user_template[] = "[INST] %s [/INST]";
        sprintf(rendered_prompt, user_template, user_prompt);
      }
      // encode the rendered prompt into tokens
      encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
      user_idx = 0; // reset the user index
      user_turn = 0;
      printf("Assistant: ");
    }

    // determine the token to pass into the transformer next
    if (user_idx < num_prompt_tokens) {
      // if we are still processing the input prompt, force the next prompt token
      token = prompt_tokens[user_idx++];
    } else {
      // otherwise use the next token sampled from previous turn
      token = next;
    }
    // EOS (=2) token ends the Assistant turn
    if (token == 2) { user_turn = 1; }

    // forward the transformer to get logits for the next token
    float* logits = forward(transformer, token, pos);
    next = sample(sampler, logits);
    pos++;

    if (user_idx >= num_prompt_tokens && next != 2) {
      // the Assistant is responding, so print its output
      char* piece = decode(tokenizer, token, next);
      safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
      fflush(stdout);
    }
    if (next == 2) { printf("\n"); }
  }
  printf("\n");
  free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// You should parallelize and optimize from this function exploiting multiple GPUs
//
int test_continuous_batching_only(Transformer *transformer, Tokenizer *tokenizer, char *tokenizer_path, Requests * requests, int n_batches=1) {
//   // Count the number of the generated tokens
//   int gen_cnt = 0;

//   // Avoid randomness to generate tokens for batch input
//   // Each input request has its Sampler each
//   Sampler samplers[requests->num_reqs];
//   for(int idx = 0; idx < requests->num_reqs; idx++) {
//     build_sampler(&samplers[idx], transformer->config.vocab_size, 1.0f, 0.9f, 314028);
//   }

//   int n_devices = 0;
//   CHECK_HIP(hipGetDeviceCount(&n_devices));
//   fprintf(stderr, "\n Num Devices %d\n", n_devices);
//   int n_layers = transformer->config.n_layers;
//   int vocab_size = transformer->config.vocab_size;
//   if (n_layers == 12)
//     n_batches = 8;
//   else if (n_layers == 16)
//     n_batches = 2;
//   else
//     n_batches = 1;

//   std::mutex mtx_idx, mtx_n_done;
//   int next_idx = 0;
//   int n_done = 0;
//   int gen_cnt_each_device[n_devices];


//   // BEGIN PARALLEL

//   #pragma omp parallel num_threads(n_devices)
//   {
//     int gid = omp_get_thread_num();
//     CHECK_HIP(hipSetDevice(gid));
//     cpu_set_t cpu_set;
//     CPU_ZERO(&cpu_set);
//     CPU_SET(gid, &cpu_set);
//     sched_setaffinity(0, sizeof(cpu_set_t), &cpu_set);
//     fprintf(stderr, "\nDevice ID %d\n", gid);

//     Tokenizer private_tokenizer;
//     build_tokenizer(&private_tokenizer, tokenizer_path, vocab_size);

//     TransformerWeights *weight_d;
//     RunState *state_d_batch;
//     // float *logits[n_batches];
//     // for(int b=0 ; b<n_batches ; ++b)
//     //   CHECK_HIP(hipHostMalloc(&logits[b], vocab_size * sizeof(float)));
//     float *logits_host;
//     CHECK_HIP(hipHostMalloc(&logits_host, n_batches * vocab_size * sizeof(float)));

//     thablasHandle_t handle;
//     thablasCreate(&handle);
//     copy_weight_to_device(transformer, weight_d);
//     alloc_state_to_device_batch(transformer, state_d_batch, n_batches);

//     int indices[n_batches];
//     for(int b=0 ; b<n_batches ; ++b)
//       indices[b] = -1;

//     std::string gen_str[n_batches];
//     char *prompt[n_batches];
//     int *prompt_tokens[n_batches];
//     int num_prompt_tokens[n_batches];

//     // long start = 0; // used to time our code, only initialized after first iteration
//     int next[n_batches]; // will store the next token in the sequence
//     int token[n_batches]; // kick off with the first token in the prompt
//     int pos[n_batches]; // position in the sequence
//     int steps[n_batches]; // max sequence length
//     bool is_done[n_batches];
//     float* logits_d[n_batches];

//     thablasStatus_t tha_status = THABLAS_STATUS_SUCCESS;
//     gen_cnt_each_device[gid] = 0;

//     while (1) {
//       // check for stop condition
//       bool stop = false;
//       mtx_n_done.lock();
//       stop = (n_done >= requests->num_reqs);
//       mtx_n_done.unlock();
//       if (stop) break;

//       // assgin new request to GPU
//       for(int b=0 ; b<n_batches ; ++b) {
//         // if there exist next request
//         if (indices[b] == -1) {
//           mtx_idx.lock();
//           indices[b] = next_idx;
//           if (next_idx < requests->num_reqs) ++next_idx;
//           mtx_idx.unlock();

//           if (indices[b] >= requests->num_reqs)
//           {
//             indices[b] = -1;
//             continue;
//           }

//           fprintf(stderr, "\nDevice %d - Request %d\n", gid, indices[b]);
//           gen_str[b] = "";
//           prompt[b] = get_str_req_ptr(requests, indices[b]);
//           prompt_tokens[b] = (int*)malloc((strlen(prompt[b])+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS

//           // encode the (string) prompt into tokens sequence
//           num_prompt_tokens[b] = 0;
//           encode(&private_tokenizer, prompt[b], 1, 0, prompt_tokens[b], &num_prompt_tokens[b]);
//           if (num_prompt_tokens[b] < 1) {
//             fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
//             exit(EXIT_FAILURE);
//           }

//           token[b] = prompt_tokens[b][0]; // kick off with the first token in the prompt
//           pos[b] = 0; // position in the sequence
//           steps[b] = requests->max_seq_len; // max sequence length
//           is_done[b] = false;
//           logits_d[b] = nullptr;
//         }
//       }

//       tha_status = thaDNN_s_forward_batch(handle, handle, handle, n_batches, &transformer->config, weight_d, state_d_batch, token, pos, logits_host);
//       // for(int b=0 ; b<n_batches ; ++b)
//       //   CHECK_HIP(hipMemcpy(logits[b], logits_d[b], vocab_size * sizeof(float), hipMemcpyDeviceToHost));
//       CHECK_HIP(hipDeviceSynchronize());

//       // advance the state machine
//       for(int b=0 ; b<n_batches ; ++b) {
//       if (indices[b] > -1) 
//       {
//         if (pos[b] < num_prompt_tokens[b] - 1) {
//           next[b] = prompt_tokens[b][pos[b] + 1];
//         } else {
//           next[b] = sample(&samplers[indices[b]], logits_host + b * vocab_size);
//           // next[b] = prompt_tokens[b][1];
//           // next[b] = sample(&samplers[indices[b]], logits[b]);
//         }

//         if (next[b] == 1 || next[b] == 2) {
//           is_done[b] = true;
//         }
//         else
//         {
//           char* piece = decode(&private_tokenizer, token[b], next[b]);
//           append_str(piece, gen_str[b]);
//           token[b] = next[b];

//           ++pos[b];
//           if (pos[b] >= steps[b]) {
//             is_done[b] = true;
//           }
//         }
//       }
//       ++gen_cnt_each_device[gid];
//       }
      
//       // de-assgin the requests
//       for(int b=0 ; b<n_batches ; ++b)
//       {
//         if (is_done[b] && indices[b] > -1)
//         {
//           gen_str[b] += "\n";
//           strcpy(get_str_gen_ptr(requests, indices[b]), gen_str[b].c_str());
//           free(prompt_tokens[b]);     
//           fprintf(stderr, "\nDevice %d - DONE %d\n", gid, indices[b]);     
//           indices[b] = -1;
//           is_done[b] = false;
//           pos[b] = 0;
//           token[b] = 0;

//           mtx_n_done.lock();
//           ++n_done;
//           mtx_n_done.unlock();
//         }
//       }

//       // if (start == 0) { start = time_in_ms();}
//     }
//   }

//   for(int idx = 0; idx < requests->num_reqs; idx++) {
//     free_sampler(&samplers[idx]);
//   }

//   gen_cnt = 0;
//   for(int gid = 0; gid < n_devices; ++gid)
//     gen_cnt += gen_cnt_each_device[gid];
//   fprintf(stderr, "\ngen_cnt: %d\n", gen_cnt);
//   return gen_cnt;
}

int test(Transformer *transformer, Tokenizer *tokenizer, char *tokenizer_path, Requests * requests, int batch_size=1) {
  // Count the number of the generated tokens
  int gen_cnt = 0;

  // Avoid randomness to generate tokens for batch input
  // Each input request has its Sampler each
  Sampler samplers[requests->num_reqs];
  for(int idx = 0; idx < requests->num_reqs; idx++) {
    build_sampler(&samplers[idx], transformer->config.vocab_size, 1.0f, 0.9f, 314028);
  }

  int n_devices = 0;
  batch_size = 16;
  int n_flows = 4;
  CHECK_HIP(hipGetDeviceCount(&n_devices));
  fprintf(stderr, "\n Num Devices %d\n", n_devices);
  int n_layers = transformer->config.n_layers;
  int vocab_size = transformer->config.vocab_size;
  int pipe_size = n_layers / n_devices;
  
  std::mutex mtx_idx, mtx_n_done;
  int next_idx = 0;
  int n_done = 0;
  int flow_status[n_flows], device_flow[n_devices];
  std::mutex device_mtx[n_devices];
  int gen_cnt_flow[n_flows];

  thablasStatus_t tha_status = THABLAS_STATUS_SUCCESS;

  TransformerWeights* w_d[n_devices];
  thablasHandle_t handle[n_devices];
  #pragma omp parallel for num_threads(n_devices)
  for(int gid=0 ; gid<n_devices ; ++gid) {
    CHECK_HIP(hipSetDevice(gid));
    fprintf(stderr, "\nInit weights device %d\n", gid);
    device_flow[gid] = 0;
    thablasCreate(&handle[gid]);
    copy_transformer_weight_pipeline_to_device_batch(handle[gid], transformer, w_d[gid], pipe_size, gid, batch_size);
    fprintf(stderr, "\nInit done device %d\n", gid);
  }

  #pragma omp parallel num_threads(n_flows) 
  {
    int fid = omp_get_thread_num();
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(fid, &cpu_set);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpu_set);
    fprintf(stderr, "\nFlow ID %d\n", fid);

    float *logits_host;
    CHECK_HIP(hipHostMalloc(&logits_host, batch_size * vocab_size * sizeof(float)));

    int indices[batch_size];
    for(int b=0 ; b<batch_size ; ++b)
      indices[b] = -1;

    std::string gen_str[batch_size];
    char *prompt[batch_size];
    int *prompt_tokens[batch_size];
    int num_prompt_tokens[batch_size];

    // long start = 0; // used to time our code, only initialized after first iteration
    int next[batch_size]; // will store the next token in the sequence
    int token[batch_size]; // kick off with the first token in the prompt
    int pos[batch_size]; // position in the sequence
    int steps[batch_size]; // max sequence length
    bool is_done[batch_size];
    float* logits_d[batch_size];

    flow_status[fid] = 0;
    RunState* s_h_batch[n_devices];
    RunState* s_d_batch[n_devices];
    for(int gid=0 ; gid<n_devices ; ++gid) {
      CHECK_HIP(hipSetDevice(gid));
      alloc_run_state_to_host_batch(handle[gid], transformer, s_h_batch[gid], pipe_size, gid, batch_size);
      alloc_run_state_to_device_1_layer_batch(handle[gid], transformer, s_d_batch[gid], batch_size);
    }

    Tokenizer private_tokenizer;
    build_tokenizer(&private_tokenizer, tokenizer_path, vocab_size);
    gen_cnt_flow[fid] = 0;

    while (1) {
      // check for stop condition
      bool stop = false;
      mtx_n_done.lock();
      stop = (n_done >= requests->num_reqs);
      mtx_n_done.unlock();
      if (stop) break;

      // assgin new request to batch
      int n_assigned = 0;
      for(int b=0 ; b<batch_size ; ++b) {
        // if there exist next request
        if (indices[b] == -1) {
          mtx_idx.lock();
          indices[b] = next_idx;
          if (next_idx < requests->num_reqs) ++next_idx;
          mtx_idx.unlock();

          if (indices[b] >= requests->num_reqs)
          {
            indices[b] = -1;
            continue;
          }

          fprintf(stderr, "\nFlow %d Request %d\n", fid, indices[b]);
          gen_str[b] = "";
          prompt[b] = get_str_req_ptr(requests, indices[b]);
          prompt_tokens[b] = (int*)malloc((strlen(prompt[b])+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS

          // encode the (string) prompt into tokens sequence
          num_prompt_tokens[b] = 0;
          encode(&private_tokenizer, prompt[b], 1, 0, prompt_tokens[b], &num_prompt_tokens[b]);
          if (num_prompt_tokens[b] < 1) {
            fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
            exit(EXIT_FAILURE);
          }

          token[b] = prompt_tokens[b][0]; // kick off with the first token in the prompt
          pos[b] = 0; // position in the sequence
          steps[b] = requests->max_seq_len; // max sequence length
          is_done[b] = false;
          logits_d[b] = nullptr;
        }
        if (indices[b] != -1) ++n_assigned;
      }
      if (n_assigned == 0) break;

      // tha_status = thaDNN_s_forward_batch(handle, handle, handle, batch_size, &transformer->config, weight_d, state_d_batch, token, pos, logits_host);
      tha_status = thaDNN_s_forward_batch_multiple_pipe_line_layer_swap(handle, fid, n_flows, n_devices, batch_size, &transformer->config, w_d, s_d_batch, s_h_batch, token, pos, logits_host, flow_status, device_flow, device_mtx);

      // advance the state machine
      for(int b=0 ; b<batch_size ; ++b) {
        if (indices[b] > -1) 
        {
          if (pos[b] < num_prompt_tokens[b] - 1) {
            next[b] = prompt_tokens[b][pos[b] + 1];
          } else {
            next[b] = sample(&samplers[indices[b]], logits_host + b * vocab_size);
            // next[b] = prompt_tokens[b][1];
            // next[b] = sample(&samplers[indices[b]], logits[b]);
          }

          if (next[b] == 1 || next[b] == 2) {
            is_done[b] = true;
          }
          else
          {
            char* piece = decode(&private_tokenizer, token[b], next[b]);
            append_str(piece, gen_str[b]);
            token[b] = next[b];
            // safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            // fflush(stdout);
            // fprintf(stderr, "%d %s\n", pos[b], piece);
            ++pos[b];
            if (pos[b] >= steps[b]) {
              is_done[b] = true;
            }
          }
          // ++gen_cnt;
          ++gen_cnt_flow[fid];
        }
      }
      
      // de-assgin the requests
      for(int b=0 ; b<batch_size ; ++b)
      {
        if (is_done[b] && indices[b] > -1)
        {
          gen_str[b] += "\n";
          strcpy(get_str_gen_ptr(requests, indices[b]), gen_str[b].c_str());
          free(prompt_tokens[b]);
          fprintf(stderr, "\nFlow %d DONE %d\n", fid, indices[b]);
          indices[b] = -1;
          is_done[b] = false;
          pos[b] = 0;
          token[b] = 0;

          mtx_n_done.lock();
          ++n_done;
          mtx_n_done.unlock();
        }
      }

      // if (start == 0) { start = time_in_ms();}
    }
  } // end flow parallel

  for(int idx = 0; idx < requests->num_reqs; idx++) {
    free_sampler(&samplers[idx]);
  }

  gen_cnt = 0;
  for(int fid = 0; fid < n_flows; ++fid)
    gen_cnt += gen_cnt_flow[fid];
  fprintf(stderr, "\ngen_cnt: %d\n", gen_cnt);

  return gen_cnt;
}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
  fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
  fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
  fprintf(stderr, "Example: run model.bin -m test -f <input_filename> -o <output_filename>\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0 (ignore the arg for test mode)\n");
  fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9 (ignore the arg for test mode)\n");
  fprintf(stderr, "  -s <int>    random seed, default time(NULL) (ignore the arg for test mode)\n");
  fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len (for test mode steps = max_seq_len)\n");
  fprintf(stderr, "  -i <string> input prompt (ignore the arg for test mode)\n");
  fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
  fprintf(stderr, "  -m <string> mode: generate|chat|test, default: generate\n");
  fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
  fprintf(stderr, "  -f <string> (only for test mode) input filename\n");
  fprintf(stderr, "  -o <string> (only for test mode) output filename\n");
  fprintf(stderr, "  -b <string> batch size\n");

  exit(EXIT_FAILURE);
}



int main(int argc, char *argv[]) {
  long total_start, total_end;
  total_start = time_in_ms();

  // default parameters
  char *checkpoint_path = NULL;  // e.g. out/model.bin
  char *tokenizer_path = (char*)"./assets/tokenizer.bin";
  float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  int steps = 256;            // number of steps to run for
  char *prompt = NULL;        // prompt string
  unsigned long long rng_seed = 0; // seed rng with time by default
  char *mode = (char*)"generate";    // generate|chat|test
  char *system_prompt = NULL; // the (optional) system prompt to use in chat mode
  char *input_filename = NULL; // Input Filename
  char *output_filename = NULL; // Output Filename
  int batch = 1;

  // poor man's C argparse so we can override the defaults above from the command line
  if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
  for (int i = 2; i < argc; i+=2) {
    // do some basic validation
    if (i + 1 >= argc) { error_usage(); } // must have arg after flag
    if (argv[i][0] != '-') { error_usage(); } // must start with dash
    if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
    // read in the args
    if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
    else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
    else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
    else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
    else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
    else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
    else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
    else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
    else if (argv[i][1] == 'f') { input_filename = argv[i + 1]; }
    else if (argv[i][1] == 'o') { output_filename = argv[i + 1]; }
    else if (argv[i][1] == 'b') { batch = atoi(argv[i + 1]); }
    else { error_usage(); }
  }

  // parameter validation/overrides
  if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
  if (temperature < 0.0) temperature = 0.0;
  if (topp < 0.0 || 1.0 < topp) topp = 0.9;
  if (steps < 0) steps = 0;

  // build the Transformer via the model .bin file
  Transformer transformer;
  build_transformer(&transformer, checkpoint_path);
  if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // ovrerride to ~max length

  // build the Tokenizer via the tokenizer .bin file
  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

  // build the Sampler
  Sampler sampler;
  build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

  Requests requests;

  // run!
  if (strcmp(mode, "generate") == 0) {
    generate(&transformer, &tokenizer, &sampler, prompt, steps);
  } 
  else if (strcmp(mode, "chat") == 0) {
    //chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
  } 
  else if  (strcmp(mode, "test") == 0) {
    int num_reqs;
    steps = transformer.config.seq_len;
    if(input_filename == NULL || output_filename == NULL) {
      error_usage();
    }
    if(EXIT_FAILURE == read_inputfile(input_filename, tokenizer.max_token_length, steps, &requests)) {
      fprintf(stderr, "cannot read input file: %s\n", input_filename);
      exit(EXIT_FAILURE);
    }

    // Don't modify this parts for evaluation
    // {
    long start, end;
    start = time_in_ms();
    int num_gen_tokens = test(&transformer, &tokenizer, tokenizer_path, &requests, batch);
    end = time_in_ms();

    // Your goal is to achieve best throughput(=reduce elapsed time)! 
    fprintf(stdout, "elapsed time(s): %f, achieved throughput(tok/s): %f\n", (double)(end-start)/1000, (num_gen_tokens) / (double)(end-start)*1000);
    //}

    if(EXIT_FAILURE == write_outputfile(output_filename, &requests)) {
      fprintf(stderr, "cannot write output file: %s\n", input_filename);
      exit(EXIT_FAILURE);
    }

    free_requests(&requests);

  } else {
    fprintf(stderr, "unknown mode: %s\n", mode);
    error_usage();
  }

  // memory and file handles cleanup
  free_sampler(&sampler);
  free_tokenizer(&tokenizer);
  free_transformer(&transformer);

  total_end = time_in_ms();
  fprintf(stdout, "total elapsed time(s): %lf\n", (double)(total_end-total_start)/1000);
  return 0;
}
#endif
