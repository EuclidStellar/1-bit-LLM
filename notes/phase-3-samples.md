# Phase 3 — reading the four arms

Same prompts, same seed per arm, so only the model varies. All three trained arms
opened "Once upon a time there was a rabbit named Jack" before diverging, which
confirms the control.

| dimension | fp32 (7.8) | QAT (8.8) | QAT+t.embed (10.1) | PTQ (151.9) |
|---|---|---|---|---|
| grammatical sentences | yes | yes | yes | **broken** |
| story has an ending | yes | yes | yes | no |
| dialogue *formatting* | yes | yes | yes | unbalanced quotes |
| dialogue *semantics* | yes | wobbly | contradictory | none |
| **entity consistency** | **already failing** | worse | worst | absent |
| invented non-words | none | none | "Rill!" | pervasive |

Degradation is **bottom-up**: grammar is the most robust property in the model,
entity tracking the most fragile. Nothing here contradicts the loss ordering.

## The commercially relevant finding: the ternary embedding is free in practice

QAT vs QAT+ternary-embed, both losing the thread after ~3 sentences:

> **QAT:** "a big, mean frog. He picked up a stick... a brave rabbit hopped up and
> landed next to the bird. Jack tried to catch the rabbit" -- frog becomes bird,
> and Jack (a rabbit) is chasing a rabbit.

> **QAT+t.embed:** "a big rock in the window... It is just a worm... Maybe it is a
> coin... tried to open the rock... the puddle was too far... The rope is gone!" --
> the referent mutates every clause.

Not reliably distinguishable in a blind read. **+0.1347 nats buys a 32% smaller
file at 98.4% 1.58-bit** for degradation a reader would struggle to identify. The
qualitative result is stronger than the loss number alone implies.

## PTQ fails at a one-to-two token range

```
"he wasn scared"    "he wouldn shook"    "he could talk agreed"
"He loved L."       "The N!"             "he Bob nodded, a painter"
```

**Contractions fracture.** `wasn` + `'t` is a two-token sequence whose second
token is almost fully determined by the first, and PTQ cannot hold it. Together
with stray single-letter tokens, that places its surviving dependency range at
roughly one to two tokens -- consistent with landing between rung 2 (no attention,
5.3828) and rung 3 (one attention layer, 3.9106).

It still produces **English-shaped** noise: local word order remains plausible,
which is why it scores 151.9 rather than the 4096 of uniform guessing.

## Two distinct pathologies under the same token budget

**fp32 repeats.** "pulled the machine through the machine... put the machine on
the machine and ran around the machine" -- eight uses of "machine". Also
"friendly and friendly", "higher and higher, higher and higher".

**The ternary arms drift.** Referents mutate rather than repeat.

Same 20M-token under-training, different failure mode. Unpredicted, and worth a
follow-up: does the ternary quantizer's noise act like a repetition penalty?
