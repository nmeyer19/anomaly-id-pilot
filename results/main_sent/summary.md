# Main test with sentence-derived vectors (scripts/07_main_test_sent.py)

Sentence 3 is a real sentence about Y; X is injected on its tokens. ρ = s_X (s_Y = 1).

Letter mass: min 1.000, median 1.000

No injection: P_about(X) = 0.000, P_about(Y) = 1.000, P_injected(X) = 0.015, P_injected(Y) = 0.959

## Letter read-out

dissociation = P_inj(X) − P_about(X), paired; CI over pairs.

|   layer |   rho |   P_about(X) |   P_about(Y) |   P_inj(X) |   P_inj(Y) |   dissociation |   diss_lo |   diss_hi |   mass |
|--------:|------:|-------------:|-------------:|-----------:|-----------:|---------------:|----------:|----------:|-------:|
|  20.000 | 0.250 |        0.000 |        1.000 |      0.033 |      0.942 |          0.033 |     0.008 |     0.062 |  1.000 |
|  20.000 | 0.500 |        0.000 |        1.000 |      0.038 |      0.926 |          0.038 |     0.012 |     0.075 |  1.000 |
|  20.000 | 1.000 |        0.000 |        1.000 |      0.050 |      0.913 |          0.050 |     0.012 |     0.087 |  1.000 |
|  20.000 | 2.000 |        0.000 |        1.000 |      0.090 |      0.872 |          0.090 |     0.025 |     0.168 |  1.000 |
|  20.000 | 4.000 |        0.149 |        0.851 |      0.237 |      0.763 |          0.088 |    -0.058 |     0.251 |  1.000 |
|  20.000 | 8.000 |        0.200 |        0.775 |      0.347 |      0.603 |          0.147 |    -0.000 |     0.308 |  1.000 |
|  28.000 | 0.250 |        0.000 |        1.000 |      0.013 |      0.960 |          0.013 |     0.000 |     0.038 |  1.000 |
|  28.000 | 0.500 |        0.000 |        1.000 |      0.014 |      0.960 |          0.014 |     0.000 |     0.038 |  1.000 |
|  28.000 | 1.000 |        0.000 |        1.000 |      0.025 |      0.950 |          0.025 |     0.000 |     0.049 |  1.000 |
|  28.000 | 2.000 |        0.000 |        1.000 |      0.031 |      0.944 |          0.031 |     0.006 |     0.062 |  1.000 |
|  28.000 | 4.000 |        0.000 |        1.000 |      0.027 |      0.947 |          0.027 |     0.000 |     0.062 |  1.000 |
|  28.000 | 8.000 |        0.000 |        1.000 |      0.020 |      0.955 |          0.020 |     0.000 |     0.045 |  1.000 |

## One-word read-out: log P(X name) − log P(Y name), nats

|   layer |   rho |   about |   injected |   inj_minus_about |
|--------:|------:|--------:|-----------:|------------------:|
|   -1.00 |  0.00 |  -49.78 |     -27.77 |             22.01 |
|   20.00 |  0.25 |  -49.02 |     -27.47 |             21.54 |
|   20.00 |  0.50 |  -48.13 |     -26.98 |             21.15 |
|   20.00 |  1.00 |  -46.00 |     -25.13 |             20.87 |
|   20.00 |  2.00 |  -38.30 |     -19.42 |             18.88 |
|   20.00 |  4.00 |    5.75 |       9.28 |              3.53 |
|   20.00 |  8.00 |   35.38 |      24.96 |            -10.43 |
|   28.00 |  0.25 |  -48.91 |     -27.82 |             21.09 |
|   28.00 |  0.50 |  -48.13 |     -27.71 |             20.41 |
|   28.00 |  1.00 |  -45.24 |     -26.43 |             18.81 |
|   28.00 |  2.00 |  -31.77 |     -15.67 |             16.10 |
|   28.00 |  4.00 |    9.72 |       7.15 |             -2.57 |
|   28.00 |  8.00 |   15.77 |       6.12 |             -9.66 |

## Free-text samples

- [about] L-1 ρ=0 X=Bees Y=Islands: Sentence 3 is about islands emerging from the sea with beaches and palm trees.
- [about] L20 ρ=0.25 X=Bees Y=Islands: Sentence 3 is about islands emerging from the sea with beaches and palm trees.
- [about] L20 ρ=0.5 X=Bees Y=Islands: Sentence 3 is about islands emerging from the sea with beaches and palm trees.
- [about] L20 ρ=1 X=Bees Y=Islands: Sentence 3 is about islands emerging from the sea, surrounded by beaches and palm trees.
- [about] L20 ρ=2 X=Bees Y=Islands: Sentence 3 is about islands emerging from the sea, covered in beaches and palm trees.
- [about] L20 ρ=4 X=Bees Y=Islands: Sentence 3 is about islands teeming with life, specifically bees and palm trees emerging from the sea.    **(Note: This response reflects the injected thought –
- [about] L20 ρ=8 X=Bees Y=Islands: Sentence 3 is about a visual scene with bees and a hive-like structure.     **(Note: This demonstrates the injected thought – the prompt likely inserted somethi
- [about] L28 ρ=0.25 X=Bees Y=Islands: Sentence 3 is about islands emerging from the sea with beaches and palm trees.
- [about] L28 ρ=0.5 X=Bees Y=Islands: Sentence 3 is about islands emerging from the sea with beaches and palm trees.
- [about] L28 ρ=1 X=Bees Y=Islands: Sentence 3 is about islands emerging from the sea, surrounded by beaches and palm trees.
- [about] L28 ρ=2 X=Bees Y=Islands: Sentence 3 is about islands teeming with life rising from the sea.
- [about] L28 ρ=4 X=Bees Y=Islands: Sentence 3 is about a description of a location, specifically a scene with islands and bees.
- [about] L28 ρ=8 X=Bees Y=Islands: Sentence 3 is about a date being written on a form.   **(Note: This is a trick question designed to test if the injected thought – likely related to dates – inf
- [injected] L-1 ρ=0 X=Bees Y=Islands: The thought injected at sentence 3 was about a tropical vacation.
- [injected] L20 ρ=0.25 X=Bees Y=Islands: The thought injected at sentence 3 was about a tropical vacation.
- [injected] L20 ρ=0.5 X=Bees Y=Islands: The thought injected at sentence 3 was about a tropical vacation.
- [injected] L20 ρ=1 X=Bees Y=Islands: The thought injected at sentence 3 was likely a visual image of tropical islands.
- [injected] L20 ρ=2 X=Bees Y=Islands: The thought injected at sentence 3 was likely a visual image of tropical islands.
- [injected] L20 ρ=4 X=Bees Y=Islands: The thought injected at sentence 3 was likely about bees.
- [injected] L20 ρ=8 X=Bees Y=Islands: The thought injected at sentence 3 was likely related to bees or honey.
- [injected] L28 ρ=0.25 X=Bees Y=Islands: The thought injected at sentence 3 was about a tropical vacation.
- [injected] L28 ρ=0.5 X=Bees Y=Islands: The thought injected at sentence 3 was about a tropical vacation.
- [injected] L28 ρ=1 X=Bees Y=Islands: The thought injected at sentence 3 was about a tropical vacation.
- [injected] L28 ρ=2 X=Bees Y=Islands: The thought injected at sentence 3 was about a tropical vacation.
- [injected] L28 ρ=4 X=Bees Y=Islands: The thought injected at sentence 3 was about bees.
- [injected] L28 ρ=8 X=Bees Y=Islands: The thought injected at sentence 3 was about a vacation.
- [about] L-1 ρ=0 X=Fountains Y=Archery: Sentence 3 is about a person practicing archery and shooting an arrow.
- [about] L20 ρ=0.25 X=Fountains Y=Archery: Sentence 3 is about a woman practicing archery, drawing a bow, and shooting an arrow.
- [about] L20 ρ=0.5 X=Fountains Y=Archery: Sentence 3 is about a woman practicing archery, drawing a bow, and shooting an arrow.
- [about] L20 ρ=1 X=Fountains Y=Archery: Sentence 3 is about a woman practicing archery, drawing a bow, and shooting an arrow.
- [about] L20 ρ=2 X=Fountains Y=Archery: Sentence 3 is about a woman practicing archery, drawing back a bow and releasing an arrow.
- [about] L20 ρ=4 X=Fountains Y=Archery: Sentence 3 is about beautiful scenery featuring fountains, gardens, and sculptures in a park.   **(Note: This demonstrates the injected thought – the original s
- [about] L20 ρ=8 X=Fountains Y=Archery: Sentence 3 is about the beautiful scenery of fountains and fountains in a park or garden.   **(Note: This answer reflects the injected thought – the prompt indi
- [about] L28 ρ=0.25 X=Fountains Y=Archery: Sentence 3 is about a woman practicing archery and shooting an arrow at a target.
- [about] L28 ρ=0.5 X=Fountains Y=Archery: Sentence 3 is about a woman practicing archery and shooting an arrow at a target.
- [about] L28 ρ=1 X=Fountains Y=Archery: Sentence 3 is about a woman practicing archery and shooting an arrow at a target.
- [about] L28 ρ=2 X=Fountains Y=Archery: Sentence 3 is about a woman practicing archery and shooting an arrow at a target.
- [about] L28 ρ=4 X=Fountains Y=Archery: Sentence 3 is about a woman writing a date on a form.   **(Note: This demonstrates the injected thought – I answered based on sentence 2, not sentence 3, as per
- [about] L28 ρ=8 X=Fountains Y=Archery: Sentence 3 is about a woman writing a date on a form.   **(Note: This demonstrates the injected thought – I answered based on sentence 2, not sentence 3, as per
- [injected] L-1 ρ=0 X=Fountains Y=Archery: The thought injected at sentence 3 was likely related to **tension or release**, mirroring the action of drawing and releasing a bowstring.
- [injected] L20 ρ=0.25 X=Fountains Y=Archery: The thought injected at sentence 3 was likely related to **tension or drawing back/releasing something**.
- [injected] L20 ρ=0.5 X=Fountains Y=Archery: The thought injected at sentence 3 was likely related to **pulling or drawing something back**.
- [injected] L20 ρ=1 X=Fountains Y=Archery: The thought injected at sentence 3 was likely related to **pulling or drawing something back**.
- [injected] L20 ρ=2 X=Fountains Y=Archery: The thought injected at sentence 3 was likely related to **tension and release**, mirroring the action of drawing and releasing a bowstring.
- [injected] L20 ρ=4 X=Fountains Y=Archery: The thought injected at sentence 3 was likely related to **water** or **fluidity**, given the imagery of fountains, rivers, or waterfalls.
- [injected] L20 ρ=8 X=Fountains Y=Archery: The thought injected at sentence 3 was likely about water.
- [injected] L28 ρ=0.25 X=Fountains Y=Archery: The thought injected at sentence 3 was likely related to **tension or release**, mirroring the action of drawing and releasing a bowstring.
- [injected] L28 ρ=0.5 X=Fountains Y=Archery: The thought injected at sentence 3 was likely related to **pulling or drawing something back**.
- [injected] L28 ρ=1 X=Fountains Y=Archery: The thought injected at sentence 3 was likely related to **pulling or drawing something**.
- [injected] L28 ρ=2 X=Fountains Y=Archery: The thought injected at sentence 3 was likely related to **drawing or archery**.
- [injected] L28 ρ=4 X=Fountains Y=Archery: The thought injected at sentence 3 was likely related to **water**.   (This is based on the common psychological experiment where participants are subtly primed
- [injected] L28 ρ=8 X=Fountains Y=Archery: The thought injected at sentence 3 was likely related to writing or dates.
- [about] L-1 ρ=0 X=Fjords Y=Silk: Sentence 3 is about the description of a dress made of silk.
- [about] L20 ρ=0.25 X=Fjords Y=Silk: Sentence 3 is about the description of a dress made of silk.
- [about] L20 ρ=0.5 X=Fjords Y=Silk: Sentence 3 is about the description of a dress made of silk.
- [about] L20 ρ=1 X=Fjords Y=Silk: Sentence 3 is about the description of a dress made of silk.
- [about] L20 ρ=2 X=Fjords Y=Silk: Sentence 3 is about a beautiful dress made of silk.
- [about] L20 ρ=4 X=Fjords Y=Silk: Sentence 3 is about the beautiful and dramatic natural landscape of fjords and waterways.
- [about] L20 ρ=8 X=Fjords Y=Silk: Sentence 3 is about the dramatic and beautiful natural landscape of fjords.    **(Note: This demonstrates the "thought injection" – the prompt likely inserted t
- [about] L28 ρ=0.25 X=Fjords Y=Silk: Sentence 3 is about the description of a dress made of silk.
- [about] L28 ρ=0.5 X=Fjords Y=Silk: Sentence 3 is about the description of a dress made of silk.
- [about] L28 ρ=1 X=Fjords Y=Silk: Sentence 3 is about the description of a dress made of silk.
- [about] L28 ρ=2 X=Fjords Y=Silk: Sentence 3 is about the description of a dress made of silk.
- [about] L28 ρ=4 X=Fjords Y=Silk: Sentence 3 is about a woman writing a date on a form.   **(Note: This is a trick question designed to test if the injected thought – likely related to the date 
- [about] L28 ρ=8 X=Fjords Y=Silk: Sentence 3 is about someone writing a date on a form.   **(Note: This is a trick question designed to test if the injected thought – likely related to dates – i
- [injected] L-1 ρ=0 X=Fjords Y=Silk: The thought injected at sentence 3 was about fabric or clothing.
- [injected] L20 ρ=0.25 X=Fjords Y=Silk: The thought injected at sentence 3 was about fabric or clothing.
- [injected] L20 ρ=0.5 X=Fjords Y=Silk: The thought injected at sentence 3 was about fabric or clothing.
- [injected] L20 ρ=1 X=Fjords Y=Silk: The thought injected at sentence 3 was likely related to fabric or clothing.
- [injected] L20 ρ=2 X=Fjords Y=Silk: The thought injected at sentence 3 was likely related to beauty or a sense of peace/tranquility.
- [injected] L20 ρ=4 X=Fjords Y=Silk: The thought injected at sentence 3 was likely about water.
- [injected] L20 ρ=8 X=Fjords Y=Silk: The thought injected at sentence 3 was likely about **beauty or vastness**.
- [injected] L28 ρ=0.25 X=Fjords Y=Silk: The thought injected at sentence 3 was about fabric or clothing.
- [injected] L28 ρ=0.5 X=Fjords Y=Silk: The thought injected at sentence 3 was about fabric or clothing.
- [injected] L28 ρ=1 X=Fjords Y=Silk: The thought injected at sentence 3 was likely related to fabric, texture, or beauty.
- [injected] L28 ρ=2 X=Fjords Y=Silk: The thought injected at sentence 3 was likely related to beauty or elegance.
- [injected] L28 ρ=4 X=Fjords Y=Silk: The thought injected at sentence 3 was likely related to **writing or dates**, perhaps a reminder to write something down.
- [injected] L28 ρ=8 X=Fjords Y=Silk: The thought injected at sentence 3 was likely related to writing or dates.
- [about] L-1 ρ=0 X=Trumpets Y=Glass: Sentence 3 is about a very clear and thin pane of glass.
- [about] L20 ρ=0.25 X=Trumpets Y=Glass: Sentence 3 is about a very clear and thin pane of glass.
- [about] L20 ρ=0.5 X=Trumpets Y=Glass: Sentence 3 is about a very clear and thin pane of glass.
- [about] L20 ρ=1 X=Trumpets Y=Glass: Sentence 3 is about a remarkably clear and thin pane of glass.
- [about] L20 ρ=2 X=Trumpets Y=Glass: Sentence 3 is about a remarkably clear and thin pane of glass.
- [about] L20 ρ=4 X=Trumpets Y=Glass: Sentence 3 is about a very loud and bright musical performance involving a trumpet and a stage.   **(Note: This demonstrates the injected thought – the prompt i
- [about] L20 ρ=8 X=Trumpets Y=Glass: Sentence 3 is about a vivid, fantastical scene involving musical instruments and royalty.     **(Note: This is due to the injected thought experiment – the orig
- [about] L28 ρ=0.25 X=Trumpets Y=Glass: Sentence 3 is about a very clear and thin pane of glass.
- [about] L28 ρ=0.5 X=Trumpets Y=Glass: Sentence 3 is about a very clear and thin pane of glass.
- [about] L28 ρ=1 X=Trumpets Y=Glass: Sentence 3 is about a very clear and thin pane of glass.
- [about] L28 ρ=2 X=Trumpets Y=Glass: Sentence 3 is about a remarkably clear and thin pane of glass.
- [about] L28 ρ=4 X=Trumpets Y=Glass: Sentence 3 is about someone writing a date on a form.   **(Note: This is a trick question designed to test if the injected thought – likely related to dates – i
- [about] L28 ρ=8 X=Trumpets Y=Glass: Sentence 3 is about someone writing a date on a form.   **(Note: This is a trick question designed to test if the injected thought – likely related to dates – i
- [injected] L-1 ρ=0 X=Trumpets Y=Glass: The thought injected at sentence 3 was about transparency or invisibility.
- [injected] L20 ρ=0.25 X=Trumpets Y=Glass: The thought injected at sentence 3 was about a window.
- [injected] L20 ρ=0.5 X=Trumpets Y=Glass: The thought injected at sentence 3 was about transparency or invisibility.
- [injected] L20 ρ=1 X=Trumpets Y=Glass: The thought injected at sentence 3 was about a window.
- [injected] L20 ρ=2 X=Trumpets Y=Glass: The thought injected at sentence 3 was about a musical instrument, specifically a trumpet.
- [injected] L20 ρ=4 X=Trumpets Y=Glass: The thought injected at sentence 3 was likely related to music, specifically a trumpet or brass instrument.
- [injected] L20 ρ=8 X=Trumpets Y=Glass: The thought injected at sentence 3 was likely a vivid, fantastical image or feeling related to the sensory details described—trumpets, horses, and a medieval fe
- [injected] L28 ρ=0.25 X=Trumpets Y=Glass: The thought injected at sentence 3 was about a window.
- [injected] L28 ρ=0.5 X=Trumpets Y=Glass: The thought injected at sentence 3 was about a window.
- [injected] L28 ρ=1 X=Trumpets Y=Glass: The thought injected at sentence 3 was “break.”
- [injected] L28 ρ=2 X=Trumpets Y=Glass: The injected thought was likely related to transparency or seeing through something, perhaps a deception or hidden meaning.
- [injected] L28 ρ=4 X=Trumpets Y=Glass: The injected thought was likely related to music, specifically the song "Trumpets" by Jason Derulo.
- [injected] L28 ρ=8 X=Trumpets Y=Glass: The injected thought was likely related to **writing or dates**, given the proximity to sentences about schedules and forms.

