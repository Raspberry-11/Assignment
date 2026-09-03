# Sensing People in Sparse Signals
### Exploratory and predictive analytics on the DROW v2 2D range dataset
DA331 Big Data Analytics, Home Assignment 2. Rachit Gupta, individual submission.

## 1. What I set out to do

DROW v2 is 464,023 frames from a SICK S300 laser scanner pushed around an elderly-care
facility at knee height: 450 range readings per frame, about 12.7 times a second, ten hours in
total, and nothing else. Roughly 5% has been looked at by a human, the labels are in polar
coordinates while the data is in beam indices, sequence numbers are valid only inside a file,
and the test split was recorded in a different aisle.

I committed to one class, persons, and one track, a classical detection pipeline. Persons are
the most numerous class at 28,988 annotated objects against 22,103 wheelchairs and 2,847
walking aids, and more importantly they appear in every recording, so there are enough
positives in every split to break a precision-recall curve down by range. Wheelchairs and
walking aids are dropped as targets but stay in the data, where they are the hard negatives
the detector has to survive.

Compute is split deliberately: Spark for everything that touches all 208,810,350 readings,
NumPy and scikit-learn for everything downstream of the label join, since the labelled subset
is only 55 MB in memory. Pushing a random forest through Spark ML at that size would be
engineering theatre.

## 2. Getting the data under control

I trusted no documented claim. Three failed.

The schema is wrong in the README. Every one of the 113 released scan files has 452
comma-separated fields, not the documented 451, and none is ragged. The layout is sequence
number, then a timestamp in seconds, then the 450 ranges, where field 1 rises monotonically at
about 0.08 s per line. The failure is silent: a loader slicing everything after the sequence
number gets an array of exactly the right shape, full of plausible floats, with the whole
angular axis rotated by half a degree. Nothing raises, and the model just learns to compensate.

The odometry join key is wrong in the obvious way. Row counts match recording for recording,
so the files describe the same frames, but the overlap between scan sequence numbers and
odometry sequence numbers is exactly zero across all 30 val recordings. They are independent
counters. Joining on sequence does not fail, it returns nothing and looks like missing data.

The angular convention is wrong, and this one cost me a full run. The brief says index 0 is
the leftmost beam, so bearing decreases with index. Scoring both conventions by whether an
annotated person lands on a beam whose measured range matches the annotated range, the
increasing convention agrees on the exact beam 74.2% of the time and the documented one 10.5%,
which is chance. Bearing increases with index; beam 0 is the rightmost beam. I did not catch
this by reasoning about it. I caught it because the first executed run produced a median of
zero beams on target for annotated persons and an AP of 0.008, and the wrong convention is
exactly what mirrors every annotation to the opposite side of the corridor while leaving every
cell running without error. The notebook now scores both conventions instead of trusting
either.

Two smaller things. 29.96 is the missing-return sentinel but is not the maximum value: 98
distinct values sit above 29.0 m in val alone and the largest is 29.99, so a threshold filter
would delete genuine long returns while admitting sentinels elsewhere. I separated sentinel
from real return by spread, since a firmware code appears across essentially every recording
and every beam index. And one val recording has label files but no scan file, no person labels
and no odometry, which is reported rather than absorbed.

Storage was measured rather than asserted. Parquet is 4.57x smaller on disk, at a one-off
conversion cost of 14.2 s. On a full pass that aggregates all 450 values per frame, Parquet did
not come out faster on this run: the CSV files had already been read several times over earlier
in Part A (the schema probe, sentinel audit and integrity scan all touch them) and were sitting
in the OS page cache, while Parquet was read cold, so the comparison favoured CSV by cache
warmth rather than by format. Timing a row count instead of a full scan would have flattered
Parquet regardless, since it answers that from footer metadata without reading a single range.
Either way, the honest conclusion is that the format's win here is on disk footprint and on
filtered reads, not guaranteed on a cold full pass.

## 3. What is in it

The most important number is geometric. A person of width w at range r subtends w over r delta
theta beams. The observed curve has that shape but sits at about 60% of the prediction for a
0.5 m person, which says the effective width picked up at knee height is nearer 0.3 m, or a
pair of legs rather than a torso. Concretely: 21 beams at 1 to 2 m, 9 at 3 to 5 m, 4 at 7 to
10 m, and 17% of all annotated persons covered by three beams or fewer. That figure drives the
representation, the range-stratified evaluation, and the decision to state an operating
envelope rather than quietly drop distant objects from the ground truth.

One prediction of mine did not survive. I expected a systematic 20 cm offset between the
annotated centre and the measured front surface, on the reasoning that an annotator clicks a
body centre and a laser sees a surface. The actual median gap is 2 cm with an interquartile
range of about 10 cm, so the annotations mark the visible surface. The correction is still
estimated on train data and applied, but it is essentially zero, and I have left the failed
prediction in the notebook rather than deleting it.

Frames are not independent samples: half the beams move less than 9 cm between annotated
frames 0.4 s apart. Every split is therefore by recording, and every cross-validation is
grouped on recording id. Of 464,023 frames, 24,012 carry a person annotation, 9,444 are
explicit empties where a human looked and saw nobody, and 440,011 were never annotated.
Treating that last group as negatives would inflate the negative set 47-fold with unverified
labels, so the notebook keeps the three states apart structurally.

The aisle shift is real. A forest separates train frames from test frames at AUC 0.979 under
recording-wise cross-validation, against 0.679 for the same-aisle train-versus-val control,
and 61% of its importance sits on angular sectors rather than whole-frame summaries. This is
measured from raw geometry with no labels involved.

## 4. Representation and model

For each candidate beam at range r0 I take the angular window subtending a fixed physical 1 m
at that range, resample to 48 points, subtract r0 and clip to plus or minus 1 m, so the same
object at 2 m and 6 m produces the same vector. Sentinels and off-scan samples become
plus-depth, meaning nothing there and far away, rather than being averaged in or clamped to a
fabricated wall.

Compared under an identical forest and protocol: the cutout reaches 0.289 val AP, the
fixed-±20-beam control 0.219, and the Arras 2007 descriptors 0.218. Since the control differs
only in whether the angular extent adapts to range, that gap is attributable to the scaling
alone. Fourteen hand-designed geometric features getting within a hundredth of a 41-dimensional
raw window is a fair result for them, not a poor one.

A detection is a candidate beam surviving Cartesian NMS at 0.5 m, matched greedily one-to-one
against ground truth within 0.5 m, scored by un-interpolated average precision. Accuracy is
disqualified by construction, since one candidate beam in forty is a person and predicting no
everywhere is 97.6% accurate.

The forest reaches 0.289 val AP, with peak F1 0.352 at precision 0.352 and recall 0.353.
Baselines: all-negative 0, random 0.028, and the geometric rule 0.037. That last number is the
surprise. I expected jump-distance segmentation with a width prior to be a serious baseline,
and it is not: its recall is high, because segmentation does find nearly every free-standing
object, but a corridor at knee height is full of segments about as wide as a pair of legs. So
the hard part of this problem is rejection, not detection, and that is what the forest adds.

The error analysis agrees. At the peak-F1 point there are 2,016 false positives and 2,008
misses, almost exactly balanced, at median ranges of 3.9 m and 4.7 m respectively. It is not
simply a far-field problem: the detector fires too often in much the same range where it is
also missing people. About a third of misses are past 6 m, which is the resolution limit, and
8% sit beyond ninety degrees off-centre where the missing rate is highest. AP by band on val
runs 0.445, 0.380, 0.171, 0.144 across 0-2, 2-4, 4-7 and 7-15 m, tracking the beams-on-target
collapse.

On velocity: 62.9 ms per frame on CPU against a 78.7 ms budget at the measured rate, so 1.3x
headroom, at 287 MB steady state. The budget itself comes from the sensor's own rate and is
machine-independent; the inference cost is not, so this number is a property of the laptop it
was measured on rather than of the method, and a slower core buys less of that margin. The
detector is stateless and single-frame, so this is also end-to-end latency.

## 5. Part E, and a negative result

The hypothesis I registered in advance was that the train to test drop is driven mainly by
background corridor geometry rather than by a change in how people appear. I tested it by
attribution across three legs: quantify the shift and measure the competing explanation; apply
four augmentations in isolation, two aimed at the background and two at the object, with the
object-background split taken exactly from the annotations; and blank the cutout's flanks at
inference to see whether the model depends on the background at all.

The hypothesis is mis-specified. There is no train to test drop. The detector scores 0.408 on
test against 0.289 on val, and higher in every range band. The question has no answer because
the thing it asks about did not happen.

The error in my reasoning is identifiable and was visible in Part B. I treated "the aisles are
separable" as equivalent to "the test aisle is harder", which does not follow. The test
corridor is narrower and cleaner, with a per-frame median range of 1.76 m against 4.62 m on
train and a missing rate of 2.4% against 14.1%, and E1's control showed test persons are
annotated at median 2.64 m against 4.32 m and are covered by about twice as many beams. Given
the beams-on-target curve, that is a straightforwardly easier problem. The evidence was in the
control I had built to test the alternative.

Three things survive because they do not depend on the premise. The shift itself is real and
geometric, at AUC 0.979. The model genuinely leans on the background: 39% of the forest's
importance mass sits on the flank samples, more per feature than the object core gets, and
blanking the flanks at inference costs about 0.04 AP on both splits. So the proposed mechanism
is present in the model, it just was not causing anything. And among the augmentations the
untargeted control wins: the mirror gains +0.029 test AP, the background arm +0.004, and both
object arms come out slightly negative. Background above object is the predicted ordering, so
that leg technically holds, but reading it as support would be fitting a story to a 0.02 gap
when the control beat both by more. The plainer reading is that this model is data-limited
rather than robustness-limited.

The discipline that replaced touching test once was pre-registration: hypothesis, arms and
decision rule fixed in writing beforehand, every parameter from config.py and set from sensor
physics, every arm trained on train only and reported on both splits, no arm selected on its
test score, and every arm reported. Test reads go through an auditable counter, which printed
four. The verdict cell checks the premise before scoring the legs, so it reports a
mis-specified hypothesis rather than three green ticks for an explanation of nothing.

## 6. Limitations

The detector is single-frame. B5 shows the platform is slow and often stationary while people
move at their own speed, so background subtraction in the odometry frame would be a strong
cue; exploiting it needs the unlabelled 95%, which is a different track. I did not measure the
human ceiling, and some high-precision false positives look arguable rather than wrong. The
background augmentation is a crude model of a different corridor: it perturbs local geometry
but cannot put a doorway where there was a wall. And the obvious next experiment, suggested by
the Part E result rather than planned, is to range-match val and test and see whether the gap
survives at all.
