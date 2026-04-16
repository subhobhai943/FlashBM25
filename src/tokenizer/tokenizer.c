/*
 * flashbm25 — pure-C tokenizer implementation
 * Supports: lowercasing, alphanumeric splitting, stopword removal,
 *           minimum token length, and Porter stemming.
 */

#include "tokenizer.h"
#include "stemmer.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ─── Stopword table (sorted, binary-searchable) ──────────────────────────── */
static const char* const STOPWORDS[] = {
    "a","about","above","after","again","against","all","am","an","and",
    "any","are","aren't","as","at","be","because","been","before","being",
    "below","between","both","but","by","can't","cannot","could","couldn't",
    "did","didn't","do","does","doesn't","doing","don't","down","during",
    "each","few","for","from","further","get","got","had","hadn't","has",
    "hasn't","have","haven't","having","he","he'd","he'll","he's","her",
    "here","here's","hers","herself","him","himself","his","how","how's",
    "i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it",
    "it's","its","itself","let's","me","more","most","mustn't","my","myself",
    "no","nor","not","of","off","on","once","only","or","other","ought",
    "our","ours","ourselves","out","over","own","same","shan't","she",
    "she'd","she'll","she's","should","shouldn't","so","some","such","than",
    "that","that's","the","their","theirs","them","themselves","then","there",
    "there's","these","they","they'd","they'll","they're","they've","this",
    "those","through","to","too","under","until","up","very","was","wasn't",
    "we","we'd","we'll","we're","we've","were","weren't","what","what's",
    "when","when's","where","where's","which","while","who","who's","whom",
    "why","why's","will","with","won't","would","wouldn't","you","you'd",
    "you'll","you're","you've","your","yours","yourself","yourselves"
};

#define STOPWORD_COUNT ((int)(sizeof(STOPWORDS) / sizeof(STOPWORDS[0])))

static int cmp_str(const void* a, const void* b) {
    return strcmp(*(const char* const*)a, *(const char* const*)b);
}

int fbm25_is_stopword(const char* word) {
    return bsearch(&word, STOPWORDS, (size_t)STOPWORD_COUNT,
                   sizeof(const char*), cmp_str) != NULL;
}

/* ─── Default options ─────────────────────────────────────────────────────── */
FBM25TokenizerOpts fbm25_default_opts(void) {
    FBM25TokenizerOpts o;
    o.lowercase        = 1;
    o.remove_stopwords = 0;
    o.stem             = 0;
    o.min_token_len    = 1;
    return o;
}

/* ─── fbm25_free_tokens ───────────────────────────────────────────────────── */
void fbm25_free_tokens(FBM25Tokens* result) {
    if (!result) return;
    for (size_t i = 0; i < result->count; ++i) free(result->tokens[i]);
    free(result->tokens);
    result->tokens = NULL;
    result->count  = 0;
}

/* ─── fbm25_tokenize ──────────────────────────────────────────────────────── */
FBM25Tokens fbm25_tokenize(const char* text, const FBM25TokenizerOpts* opts) {
    FBM25Tokens result;
    result.tokens = NULL;
    result.count  = 0;

    if (!text || !opts) return result;

    size_t capacity = 64;
    result.tokens = (char**)malloc(capacity * sizeof(char*));
    if (!result.tokens) return result;

    /* Temporary buffer: worst case every char is a token char */
    size_t text_len = strlen(text);
    char*  buf      = (char*)malloc(text_len + 1);
    if (!buf) { free(result.tokens); result.tokens = NULL; return result; }

    size_t buf_pos = 0;

    for (size_t i = 0; i <= text_len; ++i) {
        unsigned char c = (unsigned char)text[i];
        int is_alnum = isalnum(c) || c == '_';

        if (is_alnum && i < text_len) {
            buf[buf_pos++] = opts->lowercase ? (char)tolower(c) : (char)c;
        } else if (buf_pos > 0) {
            buf[buf_pos] = '\0';

            /* Minimum length filter */
            if ((int)buf_pos < opts->min_token_len) {
                buf_pos = 0;
                continue;
            }

            /* Stopword filter */
            if (opts->remove_stopwords && fbm25_is_stopword(buf)) {
                buf_pos = 0;
                continue;
            }

            /* Porter stemming (in-place) */
            if (opts->stem) {
                int new_len = fbm25_stem(buf, (int)buf_pos);
                buf[new_len] = '\0';
                buf_pos = (size_t)new_len;
                if (buf_pos == 0) { buf_pos = 0; continue; }
            }

            /* Grow token array if needed */
            if (result.count == capacity) {
                capacity *= 2;
                char** tmp = (char**)realloc(result.tokens, capacity * sizeof(char*));
                if (!tmp) { fbm25_free_tokens(&result); free(buf); return result; }
                result.tokens = tmp;
            }

            result.tokens[result.count] = (char*)malloc(buf_pos + 1);
            if (!result.tokens[result.count]) {
                fbm25_free_tokens(&result); free(buf); return result;
            }
            memcpy(result.tokens[result.count], buf, buf_pos + 1);
            result.count++;
            buf_pos = 0;
        }
    }

    free(buf);
    return result;
}
