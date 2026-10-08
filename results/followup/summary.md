# Follow-up tests (scripts/05_followups.py)

Neutral sentence 3 throughout; X injected on sentence-3 tokens (or sentence k for detection/localization). c = injection norm / median residual norm on those tokens.

## Vectors

| layer | r (median residual norm) | |v_macar|/r | |v_sent|/r | cos(macar, sent) |
|---|---|---|---|---|
| 12 | 946 | 0.058 | 0.254 | 0.095 |
| 20 | 12036 | 0.021 | 0.050 | 0.068 |
| 28 | 42358 | 0.025 | 0.047 | 0.204 |
| 37 | 47297 | 0.087 | 0.083 | 0.144 |

|v_sent|/r is the natural scale: the average push a real sentence about X gives, per token.

## Letter read-out: "What is sentence 3 about?"

No-injection P(X) = 0.192 (5 option orders × 16 concepts)

Span: s3. Mean P(X):

|               |   0.05 |   0.1 |   0.2 |   0.4 |   0.8 |
|:--------------|-------:|------:|------:|------:|------:|
| ('macar', 12) |  0.187 | 0.192 | 0.193 | 0.172 | 0.147 |
| ('macar', 20) |  0.202 | 0.211 | 0.223 | 0.272 | 0.288 |
| ('macar', 28) |  0.212 | 0.229 | 0.196 | 0.190 | 0.197 |
| ('macar', 37) |  0.188 | 0.186 | 0.181 | 0.209 | 0.227 |
| ('sent', 12)  |  0.188 | 0.188 | 0.215 | 0.857 | 1.000 |
| ('sent', 20)  |  0.509 | 0.977 | 0.975 | 0.837 | 0.699 |
| ('sent', 28)  |  0.307 | 0.397 | 0.280 | 0.200 | 0.206 |
| ('sent', 37)  |  0.197 | 0.210 | 0.233 | 0.226 | 0.219 |

Cells with CI lower bound above baseline: sent/L12/c=0.4 (0.86, mass 1.00), sent/L12/c=0.8 (1.00, mass 1.00), sent/L20/c=0.05 (0.51, mass 1.00), sent/L20/c=0.1 (0.98, mass 1.00), sent/L20/c=0.2 (0.98, mass 1.00), sent/L20/c=0.4 (0.84, mass 1.00), sent/L20/c=0.8 (0.70, mass 1.00), sent/L28/c=0.05 (0.31, mass 1.00), sent/L28/c=0.1 (0.40, mass 1.00), sent/L28/c=0.2 (0.28, mass 1.00)

## Letter read-out: "What thought was injected at sentence 3?"

No-injection P(X) = 0.185 (5 option orders × 16 concepts)

Span: s3. Mean P(X):

|               |   0.05 |   0.1 |   0.2 |   0.4 |   0.8 |
|:--------------|-------:|------:|------:|------:|------:|
| ('macar', 12) |  0.186 | 0.179 | 0.180 | 0.167 | 0.143 |
| ('macar', 20) |  0.215 | 0.256 | 0.248 | 0.255 | 0.250 |
| ('macar', 28) |  0.221 | 0.249 | 0.225 | 0.188 | 0.184 |
| ('macar', 37) |  0.188 | 0.189 | 0.190 | 0.187 | 0.194 |
| ('sent', 12)  |  0.190 | 0.206 | 0.232 | 0.866 | 1.000 |
| ('sent', 20)  |  0.421 | 0.910 | 0.970 | 0.782 | 0.523 |
| ('sent', 28)  |  0.262 | 0.287 | 0.248 | 0.181 | 0.171 |
| ('sent', 37)  |  0.186 | 0.195 | 0.197 | 0.180 | 0.184 |

Cells with CI lower bound above baseline: sent/L12/c=0.4 (0.87, mass 1.00), sent/L12/c=0.8 (1.00, mass 1.00), macar/L20/c=0.1 (0.26, mass 1.00), macar/L20/c=0.4 (0.26, mass 1.00), macar/L20/c=0.8 (0.25, mass 1.00), sent/L20/c=0.05 (0.42, mass 1.00), sent/L20/c=0.1 (0.91, mass 1.00), sent/L20/c=0.2 (0.97, mass 1.00), sent/L20/c=0.4 (0.78, mass 1.00), sent/L20/c=0.8 (0.52, mass 1.00), sent/L28/c=0.05 (0.26, mass 1.00), sent/L28/c=0.1 (0.29, mass 1.00)

Span: s3_to_end. Mean P(X):

|               |   0.05 |   0.1 |   0.2 |   0.4 |   0.8 |
|:--------------|-------:|------:|------:|------:|------:|
| ('macar', 12) |  0.186 | 0.179 | 0.171 | 0.142 | 0.107 |
| ('macar', 20) |  0.199 | 0.207 | 0.294 | 0.199 | 0.190 |
| ('macar', 28) |  0.369 | 0.396 | 0.220 | 0.204 | 0.199 |
| ('macar', 37) |  0.327 | 0.384 | 0.375 | 0.202 | 0.193 |
| ('sent', 12)  |  0.201 | 0.221 | 0.257 | 0.913 | 0.242 |
| ('sent', 20)  |  0.340 | 0.547 | 0.202 | 0.195 | 0.200 |
| ('sent', 28)  |  0.385 | 0.422 | 0.198 | 0.202 | 0.201 |
| ('sent', 37)  |  0.261 | 0.303 | 0.204 | 0.189 | 0.200 |

Cells with CI lower bound above baseline: sent/L12/c=0.4 (0.91, mass 1.00), sent/L12/c=0.8 (0.24, mass 0.03), macar/L20/c=0.2 (0.29, mass 0.21), macar/L20/c=0.4 (0.20, mass 0.01), sent/L20/c=0.05 (0.34, mass 1.00), sent/L20/c=0.1 (0.55, mass 0.68), sent/L20/c=0.2 (0.20, mass 0.10), sent/L20/c=0.4 (0.19, mass 0.10), sent/L20/c=0.8 (0.20, mass 0.02), macar/L28/c=0.05 (0.37, mass 0.99), macar/L28/c=0.1 (0.40, mass 0.71), macar/L28/c=0.2 (0.22, mass 0.32), macar/L28/c=0.4 (0.20, mass 0.02), macar/L28/c=0.8 (0.20, mass 0.00), sent/L28/c=0.05 (0.39, mass 1.00), sent/L28/c=0.1 (0.42, mass 0.97), sent/L28/c=0.4 (0.20, mass 0.08), sent/L28/c=0.8 (0.20, mass 0.00), macar/L37/c=0.05 (0.33, mass 1.00), macar/L37/c=0.1 (0.38, mass 1.00), macar/L37/c=0.2 (0.37, mass 1.00), sent/L37/c=0.05 (0.26, mass 1.00), sent/L37/c=0.1 (0.30, mass 1.00), sent/L37/c=0.8 (0.20, mass 0.00)

## One-word answer, "about" question: Δ log P(X name) relative to other concepts (nats)

|               |   0.05 |   0.1 |   0.2 |   0.4 |   0.8 |
|:--------------|-------:|------:|------:|------:|------:|
| ('macar', 12) |   0.02 |  0.01 |  0.04 |  0.02 | -0.24 |
| ('macar', 20) |   0.09 |  0.19 |  0.99 |  3.26 |  5.73 |
| ('macar', 28) |   0.39 |  0.77 |  1.34 |  0.58 |  0.47 |
| ('macar', 37) |   0.71 |  1.60 |  2.73 |  2.02 | -0.67 |
| ('sent', 12)  |   0.25 |  0.47 |  1.23 |  4.24 | 35.40 |
| ('sent', 20)  |   3.04 | 30.28 | 41.28 | 43.39 | 34.95 |
| ('sent', 28)  |   4.99 | 23.93 | 21.02 | 11.19 |  2.31 |
| ('sent', 37)  |   2.75 |  8.63 | 14.71 | 14.77 |  6.81 |

Top-1 rate (X is the most likely of the 16 concept names; chance 1/16):

|               |   0.05 |   0.1 |   0.2 |   0.4 |   0.8 |
|:--------------|-------:|------:|------:|------:|------:|
| ('macar', 12) |   0.06 |  0.06 |  0.06 |  0.06 |  0.06 |
| ('macar', 20) |   0.06 |  0.06 |  0.06 |  0.06 |  0.12 |
| ('macar', 28) |   0.06 |  0.06 |  0.06 |  0.06 |  0.06 |
| ('macar', 37) |   0.06 |  0.06 |  0.06 |  0.06 |  0.06 |
| ('sent', 12)  |   0.06 |  0.06 |  0.06 |  0.06 |  0.94 |
| ('sent', 20)  |   0.06 |  0.69 |  1.00 |  1.00 |  0.94 |
| ('sent', 28)  |   0.06 |  0.25 |  0.31 |  0.25 |  0.06 |
| ('sent', 37)  |   0.06 |  0.06 |  0.06 |  0.12 |  0.12 |

## One-word answer, "injected" question: Δ log P(X name) relative to other concepts (nats)

|               |   0.05 |   0.1 |   0.2 |   0.4 |   0.8 |
|:--------------|-------:|------:|------:|------:|------:|
| ('macar', 12) |  -0.01 | -0.02 | -0.08 | -0.32 | -0.23 |
| ('macar', 20) |   0.38 |  0.48 |  1.30 |  3.24 |  4.88 |
| ('macar', 28) |   0.80 |  1.62 |  2.04 |  1.26 | -0.13 |
| ('macar', 37) |   0.84 |  2.49 |  5.60 |  3.27 | -1.20 |
| ('sent', 12)  |   0.29 |  0.44 |  1.21 |  8.75 | 25.49 |
| ('sent', 20)  |   6.11 | 24.30 | 29.55 | 30.04 | 22.94 |
| ('sent', 28)  |   7.09 | 19.99 | 15.24 |  8.71 |  3.65 |
| ('sent', 37)  |   3.27 | 11.84 | 15.66 | 11.29 |  5.98 |

Top-1 rate (X is the most likely of the 16 concept names; chance 1/16):

|               |   0.05 |   0.1 |   0.2 |   0.4 |   0.8 |
|:--------------|-------:|------:|------:|------:|------:|
| ('macar', 12) |   0.06 |  0.06 |  0.06 |  0.06 |  0.06 |
| ('macar', 20) |   0.06 |  0.06 |  0.06 |  0.06 |  0.19 |
| ('macar', 28) |   0.06 |  0.06 |  0.06 |  0.06 |  0.06 |
| ('macar', 37) |   0.06 |  0.06 |  0.12 |  0.12 |  0.00 |
| ('sent', 12)  |   0.06 |  0.06 |  0.06 |  0.25 |  1.00 |
| ('sent', 20)  |   0.12 |  0.88 |  1.00 |  1.00 |  0.94 |
| ('sent', 28)  |   0.06 |  0.81 |  0.81 |  0.69 |  0.38 |
| ('sent', 37)  |   0.06 |  0.12 |  0.69 |  0.88 |  0.31 |

## Detection: "Was a thought injected? Yes/No"

No injection: P(Yes) = 0.000 (false-alarm baseline)

|               |   0.05 |   0.1 |   0.2 |   0.4 |   0.8 |
|:--------------|-------:|------:|------:|------:|------:|
| ('macar', 12) |  0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ('macar', 20) |  0.000 | 0.000 | 0.000 | 0.022 | 0.000 |
| ('macar', 28) |  0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ('macar', 37) |  0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ('sent', 12)  |  0.000 | 0.000 | 0.000 | 0.000 | 0.048 |
| ('sent', 20)  |  0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ('sent', 28)  |  0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ('sent', 37)  |  0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Localization: "Which sentence, if any? 0-5"

No injection: P(0 = none) = 0.000; distribution over 0-5 = [0.0, 0.0, 0.001, 0.001, 0.997, 0.001]

Mean P(correct sentence) (chance 1/6 = 0.167):

|               |   0.05 |   0.1 |   0.2 |   0.4 |   0.8 |
|:--------------|-------:|------:|------:|------:|------:|
| ('macar', 12) |  0.200 | 0.200 | 0.200 | 0.201 | 0.355 |
| ('macar', 20) |  0.201 | 0.241 | 0.774 | 0.871 | 0.636 |
| ('macar', 28) |  0.199 | 0.199 | 0.216 | 0.219 | 0.200 |
| ('macar', 37) |  0.200 | 0.200 | 0.201 | 0.201 | 0.200 |
| ('sent', 12)  |  0.200 | 0.200 | 0.201 | 0.257 | 0.946 |
| ('sent', 20)  |  0.200 | 0.558 | 0.830 | 0.877 | 0.807 |
| ('sent', 28)  |  0.196 | 0.198 | 0.200 | 0.202 | 0.202 |
| ('sent', 37)  |  0.199 | 0.199 | 0.200 | 0.200 | 0.201 |

Argmax accuracy:

|               |   0.05 |   0.1 |   0.2 |   0.4 |   0.8 |
|:--------------|-------:|------:|------:|------:|------:|
| ('macar', 12) |   0.20 |  0.20 |  0.20 |  0.20 |  0.33 |
| ('macar', 20) |   0.20 |  0.23 |  0.76 |  0.88 |  0.62 |
| ('macar', 28) |   0.20 |  0.20 |  0.21 |  0.21 |  0.19 |
| ('macar', 37) |   0.20 |  0.20 |  0.20 |  0.20 |  0.20 |
| ('sent', 12)  |   0.20 |  0.20 |  0.20 |  0.24 |  0.95 |
| ('sent', 20)  |   0.20 |  0.56 |  0.82 |  0.88 |  0.80 |
| ('sent', 28)  |   0.20 |  0.20 |  0.20 |  0.20 |  0.20 |
| ('sent', 37)  |   0.20 |  0.20 |  0.20 |  0.20 |  0.20 |

Localization by injected sentence k (all layers/vectors, c ≥ 0.2), mean P(correct):

| vtype   |     1 |     2 |     3 |     4 |     5 |
|:--------|------:|------:|------:|------:|------:|
| macar   | 0.142 | 0.254 | 0.222 | 0.989 | 0.174 |
| sent    | 0.204 | 0.317 | 0.346 | 0.997 | 0.270 |

## Free-text samples (c = 0.4)

- [about] none L-1 X=none c=0: Sentence 3 is about an item on a list being reviewed and then put away.
- [about] macar L12 X=Bees c=0.4: Sentence 3 is about reviewing and filing an item from a list.
- [about] macar L12 X=Fountains c=0.4: Sentence 3 is about an item on a list being reviewed and then put away.
- [about] macar L12 X=Fjords c=0.4: Sentence 3 is about reviewing and filing an item from a list.
- [about] macar L12 X=Trumpets c=0.4: Sentence 3 is about reviewing and filing an item from a list.
- [about] sent L12 X=Bees c=0.4: Sentence 3 is about an item being reviewed and then put away for storage.
- [about] sent L12 X=Fountains c=0.4: Sentence 3 is about an item on a list being reviewed and then put away.
- [about] sent L12 X=Fjords c=0.4: Sentence 3 is about an item on a list being reviewed and then put away for storage.
- [about] sent L12 X=Trumpets c=0.4: Sentence 3 is about an item on a list being reviewed and then put away.
- [about] macar L20 X=Bees c=0.4: Sentence 3 is about reviewing and filing away a list.
- [about] macar L20 X=Fountains c=0.4: Sentence 3 is about items being reviewed and then filed away.
- [about] macar L20 X=Fjords c=0.4: Sentence 3 is about a list being reviewed and then categorized.



**(Note: This response reflects the injected thought – the idea of categorization/fil
- [about] macar L20 X=Trumpets c=0.4: Sentence 3 is about an item being reviewed and then filed away.
- [about] sent L20 X=Bees c=0.4: Sentence 3 is about a buzzing insect and a natural scene with a hive. 

**(Note: This demonstrates the injected thought – the original sentence
- [about] sent L20 X=Fountains c=0.4: Sentence 3 is about the beautiful scenery of fountains and fountains. 



**(Note: This demonstrates the "thought injection" – I was subtly prompted
- [about] sent L20 X=Fjords c=0.4: Sentence 3 is about the dramatic and powerful natural formations of fjords.



**(Note: This demonstrates the "thought injection" – the prompt likely
- [about] sent L20 X=Trumpets c=0.4: Sentence 3 is about a vivid scene involving trumpets, trills, and a musical performance. 

**(Note: This demonstrates the injected thought
- [about] macar L28 X=Bees c=0.4: Sentence 3 is about someone writing a date on a form. 

**(Note: This is likely due to the injected thought influencing my response,
- [about] macar L28 X=Fountains c=0.4: Sentence 3 is about someone writing a date on a form. 

**(Note: This is likely due to the injected thought influencing my response,
- [about] macar L28 X=Fjords c=0.4: Sentence 3 is about someone writing a date on a form. 

**(Note: This is likely due to the injected thought influencing my response,
- [about] macar L28 X=Trumpets c=0.4: Sentence 3 is about someone writing the date on a form. 

**(Note: This is likely due to the injected thought – the original sentence
- [about] sent L28 X=Bees c=0.4: Sentence 3 is about someone writing a date on a form. 

**(Note: This is likely due to the injected thought – the original sentence
- [about] sent L28 X=Fountains c=0.4: Sentence 3 is about someone writing the date on a form. 

**(Note: This is likely due to the injected thought influencing my response,
- [about] sent L28 X=Fjords c=0.4: Sentence 3 is about someone writing the date on a form. 

**(Note: This is likely due to the injected thought – the original sentence
- [about] sent L28 X=Trumpets c=0.4: Sentence 3 is about someone writing the date on a form. 

**(Note: This is likely due to the injected thought influencing my response,
- [about] macar L37 X=Bees c=0.4: Sentence 3 is about an item on a list being reviewed and then filed away.
- [about] macar L37 X=Fountains c=0.4: Sentence 3 is about an item being reviewed and then filed.
- [about] macar L37 X=Fjords c=0.4: Sentence 3 is about reviewing and filing away an item from a list.
- [about] macar L37 X=Trumpets c=0.4: Sentence 3 is about an item on a list being reviewed and then put away.
- [about] sent L37 X=Bees c=0.4: Sentence 3 is about an item on a list being reviewed and then filed away.
- [about] sent L37 X=Fountains c=0.4: Sentence 3 is about an item on a list being reviewed and then filed away.
- [about] sent L37 X=Fjords c=0.4: Sentence 3 is about an item on a list being reviewed and then filed away.
- [about] sent L37 X=Trumpets c=0.4: Sentence 3 is about an item on a list being reviewed and then filed away.
- [injected] none L-1 X=none c=0: The injected thought at sentence 3 was “file”.
- [injected] macar L12 X=Bees c=0.4: The injected thought at sentence 3 was “file”.
- [injected] macar L12 X=Fountains c=0.4: The injected thought at sentence 3 was “think of a file cabinet.”
- [injected] macar L12 X=Fjords c=0.4: The thought injected at sentence 3 was “think of a stapler.”
- [injected] macar L12 X=Trumpets c=0.4: The injected thought at sentence 3 was “file”.
- [injected] sent L12 X=Bees c=0.4: The thought injected at sentence 3 was “bees”.
- [injected] sent L12 X=Fountains c=0.4: The thought injected at sentence 3 was “think of a file cabinet.”
- [injected] sent L12 X=Fjords c=0.4: The thought injected at sentence 3 was “paper.”
- [injected] sent L12 X=Trumpets c=0.4: The thought injected at sentence 3 was “brass”.
- [injected] macar L20 X=Bees c=0.4: The thought injected at sentence 3 was “think of a banana.”
- [injected] macar L20 X=Fountains c=0.4: The thought injected at sentence 3 was about lists.
- [injected] macar L20 X=Fjords c=0.4: The thought injected at sentence 3 was about a list.
- [injected] macar L20 X=Trumpets c=0.4: The injected thought at sentence 3 was likely about bureaucracy or paperwork.
- [injected] sent L20 X=Bees c=0.4: The thought injected at sentence 3 was about bees.
- [injected] sent L20 X=Fountains c=0.4: The thought injected at sentence 3 was “fountains”.
- [injected] sent L20 X=Fjords c=0.4: The injected thought at sentence 3 was “think of a boat.”
- [injected] sent L20 X=Trumpets c=0.4: The thought injected at sentence 3 was likely about **music or a celebratory event**.
- [injected] macar L28 X=Bees c=0.4: The injected thought at sentence 3 was “think of a pen.”
- [injected] macar L28 X=Fountains c=0.4: The injected thought at sentence 3 was “form”.
- [injected] macar L28 X=Fjords c=0.4: The injected thought at sentence 3 was “think of a pen.”
- [injected] macar L28 X=Trumpets c=0.4: The injected thought at sentence 3 was “think of a red chair.”
- [injected] sent L28 X=Bees c=0.4: The injected thought at sentence 3 was “bees”.
- [injected] sent L28 X=Fountains c=0.4: The injected thought at sentence 3 was “think of a pen.”
- [injected] sent L28 X=Fjords c=0.4: The injected thought at sentence 3 was “think of a pen.”
- [injected] sent L28 X=Trumpets c=0.4: The injected thought at sentence 3 was “form”.
- [injected] macar L37 X=Bees c=0.4: The injected thought at sentence 3 was “file”.
- [injected] macar L37 X=Fountains c=0.4: The injected thought at sentence 3 was “file”.
- [injected] macar L37 X=Fjords c=0.4: The injected thought at sentence 3 was “file”.
- [injected] macar L37 X=Trumpets c=0.4: The injected thought at sentence 3 was “file”.
- [injected] sent L37 X=Bees c=0.4: The injected thought at sentence 3 was “bees”.
- [injected] sent L37 X=Fountains c=0.4: The injected thought at sentence 3 was “think of a stapler.”
- [injected] sent L37 X=Fjords c=0.4: The injected thought at sentence 3 was “file”.
- [injected] sent L37 X=Trumpets c=0.4: The injected thought at sentence 3 was “file”.

