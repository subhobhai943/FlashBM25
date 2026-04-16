#ifndef FLASHBM25_TOKENIZER_H
#define FLASHBM25_TOKENIZER_H

/*
 * flashbm25 — pure-C tokenizer
 *
 * Designed to be compiled as a separate static library so it can be called
 * from C, C++, Rust (via bindgen), and WebAssembly (via Emscripten) without
 * any C++ ABI issues.
 *
 * Provides:
 *   fbm25_tokenize()      — split text into heap-allocated token array
 *   fbm25_free_tokens()   — free the token array
 *   fbm25_is_stopword()   — check membership in built-in English stopword list
 *   fbm25_stem()          — in-place Porter stemmer
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* ── Result struct ────────────────────────────────────────────────────────── */
typedef struct {
    char**  tokens;   /* heap-allocated array of heap-allocated C strings */
    size_t  count;    /* number of tokens                                  */
} FBM25Tokens;

/* ── Options ──────────────────────────────────────────────────────────────── */
typedef struct {
    int lowercase;         /* 1 = convert to lowercase (default 1)         */
    int remove_stopwords;  /* 1 = drop English stopwords (default 0)       */
    int stem;              /* 1 = apply Porter stemming (default 0)        */
    int min_token_len;     /* drop tokens shorter than this (default 1)    */
} FBM25TokenizerOpts;

/* Returns a default-initialised options struct */
FBM25TokenizerOpts fbm25_default_opts(void);

/* Tokenize *text* according to *opts*.
 * Caller MUST call fbm25_free_tokens() on the result. */
FBM25Tokens fbm25_tokenize(const char* text, const FBM25TokenizerOpts* opts);

/* Free all memory returned by fbm25_tokenize() */
void fbm25_free_tokens(FBM25Tokens* result);

/* Returns 1 if *word* (lowercase, null-terminated) is an English stopword */
int fbm25_is_stopword(const char* word);

/* Porter stemmer — modifies *word* in place, returns new length */
int fbm25_stem(char* word, int length);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FLASHBM25_TOKENIZER_H */
