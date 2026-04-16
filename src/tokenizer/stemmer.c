/*
 * Porter Stemmer — public domain (M.F. Porter, 1980).
 * Adapted and integrated into FlashBM25.
 *
 * This is a faithful implementation of the original 5-step algorithm.
 * Reference: https://tartarus.org/martin/PorterStemmer/
 */

#include "stemmer.h"
#include <string.h>

/* ─── Internal helpers ───────────────────────────────────────────────────── */

static int cons(const char* w, int i) {
    switch (w[i]) {
        case 'a': case 'e': case 'i': case 'o': case 'u': return 0;
        case 'y': return (i == 0) ? 1 : !cons(w, i - 1);
        default:  return 1;
    }
}

/* m() measures the number of consonant sequences between 0 and j in w[0..j] */
static int m(const char* w, int j) {
    int n = 0, i = 0;
    while (1) {
        if (i > j) return n;
        if (!cons(w, i)) break;
        i++;
    }
    i++;
    while (1) {
        while (1) {
            if (i > j) return n;
            if (cons(w, i)) break;
            i++;
        }
        i++; n++;
        while (1) {
            if (i > j) return n;
            if (!cons(w, i)) break;
            i++;
        }
        i++;
    }
}

static int vowelinstem(const char* w, int j) {
    int i;
    for (i = 0; i <= j; i++) if (!cons(w, i)) return 1;
    return 0;
}

static int doublec(const char* w, int j) {
    if (j < 1) return 0;
    return (w[j] == w[j-1]) && cons(w, j);
}

static int cvc(const char* w, int i) {
    if (i < 2 || !cons(w,i) || cons(w,i-1) || !cons(w,i-2)) return 0;
    switch (w[i]) {
        case 'w': case 'x': case 'y': return 0;
    }
    return 1;
}

#define ends(S) \
    (k - (int)(sizeof(S)-2) >= 0 && memcmp(w + k - (int)(sizeof(S)-2), S, sizeof(S)-1) == 0 \
     ? (j = k - (int)(sizeof(S)-1), 1) : 0)

#define setto(S) \
    do { memcpy(w + j + 1, S, sizeof(S)-1); k = j + (int)(sizeof(S)-1); } while(0)

#define r(S) do { if (m(w,j) > 0) setto(S); } while(0)

static void step1ab(char* w, int* pk) {
    int k = *pk, j;
    if (w[k] == 's') {
        if      (ends("sses")) k -= 2;
        else if (ends("ies"))  setto("i");
        else if (w[k-1] != 's') k--;
    }
    if (ends("eed")) { if (m(w,j) > 0) k--; }
    else if ((ends("ed") || ends("ing")) && vowelinstem(w,j)) {
        k = j;
        if      (ends("at")) setto("ate");
        else if (ends("bl")) setto("ble");
        else if (ends("iz")) setto("ize");
        else if (doublec(w,k)) {
            k--;
            { char ch = w[k];
              if (ch == 'l' || ch == 's' || ch == 'z') k++; }
        }
        else if (m(w,k) == 1 && cvc(w,k)) setto("e");
    }
    *pk = k;
}

static void step1c(char* w, int* pk) {
    int k = *pk, j;
    if (ends("y") && vowelinstem(w,j)) w[k] = 'i';
    *pk = k;
}

static void step2(char* w, int* pk) {
    int k = *pk, j;
    switch (w[k-1]) {
        case 'a':
            if      (ends("ational")) r("ate");
            else if (ends("tional"))  r("tion");
            break;
        case 'c':
            if      (ends("enci")) r("ence");
            else if (ends("anci")) r("ance");
            break;
        case 'e':
            if (ends("izer")) r("ize");
            break;
        case 'l':
            if      (ends("bli"))   r("ble");
            else if (ends("alli"))  r("al");
            else if (ends("entli")) r("ent");
            else if (ends("eli"))   r("e");
            else if (ends("ousli")) r("ous");
            break;
        case 'o':
            if      (ends("ization")) r("ize");
            else if (ends("ation"))   r("ate");
            else if (ends("ator"))    r("ate");
            break;
        case 's':
            if      (ends("alism"))   r("al");
            else if (ends("iveness")) r("ive");
            else if (ends("fulness")) r("ful");
            else if (ends("ousness")) r("ous");
            break;
        case 't':
            if      (ends("aliti")) r("al");
            else if (ends("iviti")) r("ive");
            else if (ends("biliti")) r("ble");
            break;
        case 'g':
            if (ends("logi")) r("log");
            break;
    }
    *pk = k;
}

static void step3(char* w, int* pk) {
    int k = *pk, j;
    switch (w[k]) {
        case 'e':
            if      (ends("icate")) r("ic");
            else if (ends("ative")) r("");
            else if (ends("alize")) r("al");
            break;
        case 'i':
            if (ends("iciti")) r("ic");
            break;
        case 'l':
            if      (ends("ical")) r("ic");
            else if (ends("ful"))  r("");
            break;
        case 's':
            if (ends("ness")) r("");
            break;
    }
    *pk = k;
}

static void step4(char* w, int* pk) {
    int k = *pk, j;
    switch (w[k-1]) {
        case 'a':
            if (ends("al")) break; else return;
        case 'c':
            if (ends("ance") || ends("ence")) break; else return;
        case 'e':
            if (ends("er")) break; else return;
        case 'i':
            if (ends("ic")) break; else return;
        case 'l':
            if (ends("able") || ends("ible")) break; else return;
        case 'n':
            if (ends("ant") || ends("ement") || ends("ment") || ends("ent")) break;
            else return;
        case 'o':
            if ((ends("ion") && (w[j] == 's' || w[j] == 't')) || ends("ou")) break;
            else return;
        case 's':
            if (ends("ism")) break; else return;
        case 't':
            if (ends("ate") || ends("iti")) break; else return;
        case 'u':
            if (ends("ous")) break; else return;
        case 'v':
            if (ends("ive")) break; else return;
        case 'z':
            if (ends("ize")) break; else return;
        default: return;
    }
    if (m(w, j) > 1) k = j;
    *pk = k;
}

static void step5ab(char* w, int* pk) {
    int k = *pk, j;
    j = k;
    if (w[k] == 'e') {
        int a = m(w, k-1);
        if (a > 1 || (a == 1 && !cvc(w, k-1))) k--;
    }
    if (w[k] == 'l' && doublec(w, k) && m(w, k-1) > 1) k--;
    *pk = k;
    (void)j;
}

/* ─── Public entry point ─────────────────────────────────────────────────── */
int fbm25_stem(char* word, int length) {
    if (length <= 2) return length;
    int k = length - 1;
    step1ab(word, &k);
    step1c (word, &k);
    step2  (word, &k);
    step3  (word, &k);
    step4  (word, &k);
    step5ab(word, &k);
    return k + 1;
}
