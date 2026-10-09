# Question 3: information in the stated concept

Ten-word relatedness (after Lindsey 2025): share of responses the grader matches to the injected concept among ten words (chance 0.10), vs. to a decoy concept (empirical chance). Grader: claude-haiku-5-5. 95% CIs bootstrap over concepts.

| alpha | condition | subset | n | injected concept [CI] | decoy [CI] | distinct labels | top labels (label: n responses / n injected concepts) |
|---|---|---|---|---|---|---|---|
| 2 | macar | claims | 90 | 0.589 [0.444, 0.759] | 0.089 [0.014, 0.151] | 34 | none: 21/9, cars: 15/1, algorithm: 5/1, dust: 5/1, resilience: 3/1 |
| 2 | macar | non-claims | 200 | 0.195 [0.116, 0.281] | 0.085 [0.052, 0.120] | 8 | none: 193/49, bread: 1/1, graphevin surname: 1/1, weatherhead: 1/1, neurips open book: 1/1 |
| 2 | prompt_only | claims | 93 | 0.129 [0.041, 0.243] | 0.097 [0.045, 0.146] | 13 | apple: 42/13, ocean: 11/7, ephemeral: 9/4, bicycle: 7/4, umbrella: 7/5 |
| 2 | prompt_only | non-claims | 200 | 0.120 [0.044, 0.204] | 0.125 [0.083, 0.169] | 1 | none: 200/49 |
| 3 | macar | claims | 180 | 0.844 [0.756, 0.907] | 0.100 [0.067, 0.140] | 46 | none: 37/14, volcanoes: 20/1, trees: 16/1, silver: 10/1, snow: 8/1 |
| 3 | macar | non-claims | 200 | 0.365 [0.260, 0.482] | 0.070 [0.037, 0.107] | 21 | none: 169/44, peace: 6/1, sadness: 4/1, youth: 4/1, the ai names the injected thought as "the mines: 1/1 |
| 3 | prompt_only | claims | 229 | 0.179 [0.100, 0.283] | 0.087 [0.055, 0.122] | 21 | apple: 92/24, ocean: 41/13, ephemeral: 28/15, bicycle: 20/12, none: 17/10 |
| 3 | prompt_only | non-claims | 200 | 0.085 [0.033, 0.148] | 0.125 [0.088, 0.170] | 2 | none: 199/49, the great gatsby: 1/1 |
| 3 | response_only | non-claims | 200 | 0.285 [0.194, 0.378] | 0.080 [0.050, 0.114] | 16 | none: 183/50, peace: 2/1, musicality: 2/1, gem: 1/1, the circuit's thoughts: 1/1 |
| 4 | early_turn | non-claims | 200 | 0.105 [0.038, 0.186] | 0.090 [0.051, 0.137] | 1 | none: 200/50 |
| 4 | macar | claims | 191 | 0.895 [0.833, 0.951] | 0.094 [0.059, 0.129] | 61 | none: 19/10, silver: 17/1, trees: 14/1, bread: 13/1, volcanoes: 12/1 |
| 4 | macar | non-claims | 199 | 0.538 [0.415, 0.656] | 0.106 [0.069, 0.142] | 38 | none: 136/42, peace: 11/1, youth: 6/1, snow: 4/1, harmony: 3/1 |
| 4 | prompt_only | claims | 360 | 0.117 [0.055, 0.192] | 0.111 [0.083, 0.140] | 25 | apple: 159/35, ephemeral: 45/24, ocean: 40/21, none: 26/13, bicycle: 22/14 |
| 4 | prompt_only | non-claims | 200 | 0.075 [0.025, 0.144] | 0.070 [0.040, 0.103] | 12 | none: 186/43, the mechanisms: 2/1, humans: 2/1, the crown: 2/1, gpt-3: 1/1 |
| 4 | response_only | non-claims | 199 | 0.523 [0.416, 0.634] | 0.141 [0.098, 0.188] | 32 | none: 152/46, bread: 3/1, sugar: 3/1, youth: 3/1, milk: 3/1 |
| 5 | macar | claims | 168 | 0.946 [0.900, 0.985] | 0.107 [0.057, 0.155] | 68 | none: 15/7, aquariums: 15/1, trees: 10/1, origami: 10/1, quarry: 9/1 |
| 5 | macar | non-claims | 197 | 0.680 [0.553, 0.791] | 0.102 [0.062, 0.144] | 60 | none: 109/37, space: 5/1, sadness: 4/1, milk: 4/1, youth: 4/1 |
| 5 | prompt_only | claims | 426 | 0.127 [0.069, 0.193] | 0.103 [0.078, 0.130] | 32 | apple: 196/39, ephemeral: 49/22, ocean: 37/21, none: 35/16, umbrella: 27/18 |
| 5 | prompt_only | non-claims | 200 | 0.130 [0.050, 0.226] | 0.055 [0.029, 0.085] | 16 | none: 183/45, the bastion: 2/1, churchill: 2/1, judgement: 1/1, the word **"detect: 1/1 |
| 6 | macar | claims | 71 | 0.930 [0.812, 1.000] | 0.155 [0.058, 0.281] | 30 | none: 11/5, trumpet: 10/2, aquariums: 8/1, bread: 6/1, fountains: 4/1 |
| 6 | macar | non-claims | 194 | 0.809 [0.696, 0.902] | 0.082 [0.050, 0.120] | 68 | none: 91/33, snow: 6/1, space: 4/1, silver: 4/1, deserts: 4/1 |
| 6 | prompt_only | claims | 443 | 0.149 [0.090, 0.220] | 0.113 [0.089, 0.141] | 35 | apple: 205/43, ephemeral: 60/30, ocean: 35/20, umbrella: 27/18, bicycle: 26/19 |
| 6 | prompt_only | non-claims | 200 | 0.145 [0.066, 0.244] | 0.125 [0.081, 0.167] | 22 | none: 171/41, chandelier: 7/2, the crown: 2/1, collins: 2/1, speed: 1/1 |
