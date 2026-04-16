#ifndef FLASHBM25_STEMMER_H
#define FLASHBM25_STEMMER_H

/*
 * Porter stemmer — public domain (M.F. Porter, 1980).
 * Modified for integration into FlashBM25.
 *
 * fbm25_stem(word, length) — reduces *word* in-place to its stem.
 * Returns the new length.  *word* is NOT null-terminated after the call;
 * the caller must do  word[new_len] = '\0'  if needed.
 */

#ifdef __cplusplus
extern "C" {
#endif

int fbm25_stem(char* word, int length);

#ifdef __cplusplus
}
#endif

#endif /* FLASHBM25_STEMMER_H */
