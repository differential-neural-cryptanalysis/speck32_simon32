#include<iostream>
#include<cstdlib>
#include<random>
#include<vector>
#include<cstring>
#include<tuple>
#include<unordered_map>
#include<map>
#include<omp.h>
#include<math.h>
#include<stdlib.h>
#include<vector>
#include<fstream>
#include<unordered_set>
#include<algorithm>
#include<string>
#include<sstream>
#include<iomanip>

using namespace std;

#define WORD_SIZE (16)
#define ALPHA (8)
#define BETA (1)
#define GAMMA (2)
#define MASK_VAL ((1<<WORD_SIZE) - 1)
#define MAX_ROUNDS 50

#define m (4)
uint16_t z_0[] = {1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0}; //{01100111000011010100100010111110110011100001101010010001011111}
#define C (MASK_VAL ^ 3)

uint32_t rol(uint32_t a, uint32_t b){
    uint32_t n = ((a << b) & MASK_VAL) | (a >> (WORD_SIZE - b));
    return(n);
}

uint32_t ror(uint32_t a, uint32_t b){
    uint32_t n = (a >> b) | (MASK_VAL & (a << (WORD_SIZE - b)));
    return(n);
}

void round_function(uint32_t a, uint32_t b, uint32_t k, uint32_t& x, uint32_t& y){
    uint32_t c0 = a; uint32_t c1 = b;
    uint32_t t0 = (rol(c0, ALPHA) & rol(c0, BETA)) ^ rol(c0, GAMMA);
    c1 = c1 ^ t0 ^ k;
    x = c1; y = c0;
}

void round_function_nokey(uint32_t a, uint32_t b, uint32_t& x, uint32_t& y){
    uint32_t c0 = a; uint32_t c1 = b;
    uint32_t t0 = (rol(c0, ALPHA) & rol(c0, BETA)) ^ rol(c0, GAMMA);
    c1 = c1 ^ t0;
    x = c1; y = c0;
}

void inverse_round_function(uint32_t a, uint32_t b, uint32_t k, uint32_t& x, uint32_t& y){
    uint32_t c0 = a; uint32_t c1 = b;
    uint32_t t0 = (rol(c1, ALPHA) & rol(c1, BETA)) ^ rol(c1, GAMMA);
    c0 = c0 ^ t0 ^ k;
    x = c1; y = c0;
}

uint32_t decrypt_one_round(uint32_t c, uint32_t sk){
    uint32_t x,y;
    uint32_t c0 = c >> 16; uint32_t c1 = c & MASK_VAL;
    inverse_round_function(c0, c1, sk, x, y);
    uint32_t res = (x << 16) ^ y;
    return(res);
}

uint32_t encrypt_one_round(uint32_t p, uint32_t sk){
    uint32_t x,y;
    uint32_t p0 = p >> 16; uint32_t p1 = p & MASK_VAL;
    round_function(p0,p1,sk,x,y);
    uint32_t res = (x << 16) ^ y;
    return(res);
}

void expand_key(uint16_t * ks, uint64_t key, int rounds)
{
    uint16_t tmp, z;
    memcpy(ks, &key, sizeof(uint16_t) * m);
    for (uint32_t i = m; i < rounds; i++)
    {
        tmp = ror(ks[i-1], 3);
        tmp = tmp ^ ks[i-3];
        tmp = tmp ^ ror(tmp, 1);
        z = z_0[(i-m) % 62] & 1;
        ks[i] = ks[i-m] ^ tmp ^ z ^ C;
    }
}


uint32_t encrypt(uint32_t p, uint64_t key, int rounds){
    uint32_t a = p >> WORD_SIZE; uint32_t b = p & MASK_VAL;
    uint16_t * ks = (uint16_t *)malloc(sizeof(uint16_t) * rounds);
    expand_key(ks, key, rounds);
    for (uint32_t i = 0; i < rounds; i++){
        round_function(a,b,ks[i],a,b);
    }
    uint32_t res = (a << WORD_SIZE) + b;
    free(ks);
    return(res);
}

uint32_t encrypt_nokey(uint32_t p, int rounds){
    uint32_t a = p >> WORD_SIZE; uint32_t b = p & MASK_VAL;
    for (uint32_t i = 0; i < rounds; i++){
        round_function_nokey(a,b,a,b);
    }
    uint32_t res = (a << WORD_SIZE) + b;
    return(res);
}

uint32_t decrypt(uint32_t p, uint64_t key, int rounds){
    uint32_t a = p >> WORD_SIZE; uint32_t b = p & MASK_VAL;
    uint16_t * ks = (uint16_t *)malloc(sizeof(uint16_t) * rounds);
    expand_key(ks, key, rounds);
    for (int i = rounds-1; i >= 0; i--){
        inverse_round_function(a,b,ks[i],a,b);
    }
    //cout << a << " " << b << endl;
    uint32_t res = (a << WORD_SIZE) + b;
    return(res);
}

void make_examples(uint32_t nr, uint32_t diff, vector<uint32_t>& v0, vector<uint32_t>& v1, vector<uint32_t>& w){
    random_device rd;
    uniform_int_distribution<uint32_t> rng32(0, 0xffffffff);
    uniform_int_distribution<uint64_t> rng64(0, 0xffffffffffffffffL);
    mt19937 rng(rng64(rd));
    for (int i = 0; i < w.size(); i++)
        w[i] = (rng32(rd)) & 1;
    for (int i = 0; i < v0.size(); i++){
        if (w[i]) {
            uint64_t key = rng64(rd);
            uint32_t plain0 = rng32(rd);
            uint32_t plain1 = plain0 ^ diff;
            uint32_t c0 = encrypt(plain0, key, nr);
            uint32_t c1 = encrypt(plain1, key, nr);
            v0[i] = c0; v1[i] = c1;
        } else {
            v0[i] = rng32(rd); v1[i] = rng32(rd);
            while (v0[i] == v1[i])
                v0[i] = rng32(rd);
        }
    }
}

//the following function calculates the probability of a xor-differential transition of one round of Simon32 according to Stefan K̈olbl, Gregor Leander, and Tyge Tiessen
double diff_prob(uint32_t in, uint32_t out){
    //first, transform the output difference to what it looked like before the modular addition
    //transform also the input difference accordingly
    uint32_t in0 = in >> 16; uint32_t in1 = in & 0xffff;
    uint32_t out0 = out >> 16; uint32_t out1 = out & 0xffff;

    if (out1 != in0)
    {
        return 0.0l;
    }

    uint32_t gamma = rol(in0, GAMMA) ^ in1 ^ out0;
    if (in0 == MASK_VAL)
    {
        uint32_t hwg = __builtin_popcount(gamma);
        if ((hwg & 1) == 0)
        {
            return pow(2.0l, 1.0l-WORD_SIZE); // (1L << (WORD_SIZE - 1))
        }
        else
        {
            return 0.0l;
        }
    }
    uint32_t rola = rol(in0, ALPHA);
    uint32_t rolb = rol(in0, BETA);
    uint32_t varibits = rola | rolb;
    if ((gamma & (varibits ^ MASK_VAL)) != 0)
    {
        return 0.0l;
    }
    uint32_t doublebits = rol(in0, 2*ALPHA - BETA) & (rola ^ MASK_VAL) & rolb;
    if (((gamma ^ rol(gamma, ALPHA - BETA)) & doublebits) != 0)
    {
        return 0.0l;
    }
    int hwx = __builtin_popcount(varibits ^ doublebits);
    double res = pow(2.0l,-hwx);
    return(res);
}

void calc_ddt_update(vector<double>& ddt, vector<double>& tmp){
  uint64_t small = 1L << 32;
  vector<double> sums(1L << WORD_SIZE);
  for (uint64_t i = 1UL; i < (1UL << (2 * WORD_SIZE)); i++)
    sums[i >> WORD_SIZE] += ddt[i];
  #pragma omp parallel for
  for (uint64_t i = 1; i < small; i++){
    uint32_t out = i;
    uint32_t out0 = (out >> WORD_SIZE) & MASK_VAL;
    uint32_t out1 = out & MASK_VAL;
    uint32_t in0 = out1;
    double p = 0.0l;
    uint32_t ind = in0 << WORD_SIZE;
    if (sums[in0] != 0.0l)
      for (uint32_t in1 = 0; in1 <= MASK_VAL; in1++){
        uint32_t in = ind ^ in1;
        p += ddt[in] * diff_prob(in, out);
      }
    tmp[out] = p;
  }
  #pragma omp parallel for
  for (uint64_t out = 1; out < small; out++)
    ddt[out] = tmp[out];
}

void calc_ddt(uint32_t in_diff, int num_rounds){
  uint64_t num_diffs = 1L << 32;
  vector<double> ddt(num_diffs); vector<double> tmp(num_diffs);
  uint32_t ind = in_diff; //(in_diff >> WORD_SIZE) ^ ((in_diff & MASK_VAL) << WORD_SIZE);
  ddt[ind] = 1.0; double r = 1.0 / (1L << 32);
  for (int i = 0; i < num_rounds; i++){
    calc_ddt_update(ddt, tmp);
    double tpr = 0.0; double tnr = 0.0;
    cout << hex << setfill('0');
    for (uint64_t j = 1UL; j < (1UL << (2 * WORD_SIZE)); j++){
      if (ddt[j] > r) {tpr += ddt[j];}
      if (ddt[j] < r) {tnr += r;}
    }
    cout << endl << dec << setfill(' ');
    double acc = (tpr + tnr)/2;
    auto it = max_element(std::begin(ddt), std::end(ddt));
    cout << "Rounds: " << dec << (i+1) <<", Acc: " << acc << ", tpr:" << tpr << ", tnr:" << tnr << ", max pr: " << log2l(*it) << endl;
    if ((i+1) >= 8)
    {
        stringstream del; del << hex << in_diff;
        string delta = del.str();
        string rounds = to_string(i+1); string filename = "ddt_"+ delta +"_" + rounds + "rounds.bin";
        ofstream fout(filename, ios::out | ios::binary);
        fout.write((char*)&ddt[0], ddt.size() * sizeof(double));
        fout.close();
    }
  }
}


int main(){
    cout << "Calculate full ddt explicitly" << endl;
    calc_ddt(0x00000040, 13);


    cout << "Done." << endl;
    return(0);
}

