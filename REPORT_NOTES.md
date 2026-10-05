# Report notes: what I built, why, and how to defend it

Numbers are deliberately not repeated here; they live in the generated tables in `README.md`
(and the CSVs in `results/`). Where I name a number below it is only to point at where to find it.

## 1. The pipeline in one paragraph

StatsBomb open event data for five men's international tournaments is downloaded and cached
(`src/data/statsbomb.py`). Each shot becomes one row with features known at the moment of the
shot (`src/features/shots.py`). Five models predict P(goal) and are compared with nested,
match-grouped cross-validation (`src/models/xg.py`); the best simple model is saved and used by
the dashboard. The same events feed shot maps, pass networks, heatmaps and player summaries
(`src/analysis/events.py`). Separately, two Metrica tracking games are parsed, cleaned and
analysed (`src/data/metrica.py`, `src/analysis/tracking.py`). `app/` shows everything.

## 2. Main design decisions (and the reasoning)

**Competitions chosen.** Five men's international tournaments: every match has full event data
and they form one consistent population. Open club data is patchy (some competitions are mostly
one team). The cost: results may not transfer to club football.

**Penalties and shoot-outs excluded from the model.** A penalty is a different process (almost
no spatial variation) and would distort distance/angle effects. Shoot-out kicks are penalties
taken under different pressure. Both are counted and reported. For team totals in the dashboard,
penalties get the *observed* in-match penalty conversion rate (computed from the data), not a
guessed constant.

**Free kicks and headers kept**, with `shot_type` and `body_part` features so the model can
give them their own probabilities. Own goals are not shots in the data; they only count toward
the score difference feature.

**Features.** Distance and angle subtended by the posts (geometry); body part; shot type; play
pattern; first-time; follows a through ball / cross (read from the assisting pass, which happens
*before* the shot); under pressure; minute; score difference (counted from event order so a shot
never sees its own goal). Optional freeze-frame set: defenders in the shot cone, goalkeeper in
cone, goalkeeper distance, nearest defender.

**Leakage rules.** StatsBomb's own `shot_statsbomb_xg`, the shot outcome, `shot_end_location`
(where the ball ended up), deflection and save flags are *all consequences of the shot*. Using
them would let the model "see the answer". StatsBomb's xG is only a benchmark column. A unit test
asserts none of these names are in the feature lists.

**Splitting by match.** Shots from one match share a game state, opponents and conditions, so a
random shot-level split would let near-duplicates leak between train and test and give
over-optimistic scores. `GroupKFold` on `match_id` guarantees no match is in both. A test
asserts this. A stricter check (leave one tournament out) is reported too.

**Nested cross-validation.** The outer folds measure performance; inside each outer training set
a smaller grouped CV picks hyper-parameters. The test fold never influences tuning. Grids are
tiny on purpose ("tune lightly") and are stored in `results/xg_settings.json`.

**Models.** Constant goal rate (the "know nothing" floor), distance-only logistic regression,
logistic regression with all features, LightGBM, and a small neural network. I used scikit-learn's
`MLPClassifier` instead of PyTorch: it runs on a CPU, adds no heavy dependency and is easier to
explain. The three full models turned out tied within noise; I saved the logistic regression
because it is the simplest and the best on mean log loss.

**Tracking cleaning.** Coordinates are normalised 0-1 with the origin top-left; I convert to
metres on an *assumed* 105 x 68 m pitch with the origin bottom-left. Direction is detected at
each period's kick-off (mean x of the team) and frames are flipped so the home team always attacks
+x. Raw data contained impossible speeds (thousands of m/s). A rolling-median test removes
isolated glitches, and any frame touching a step above 12 m/s is blanked; a gap is bridged only if
it is short and the neighbours imply a plausible speed. Otherwise it stays missing and acts as a
break in the track. This rule exists because naive interpolation across a permanent jump draws a
fake 150 m/s run.

**Smoothing.** Speed is position difference divided by 0.04 s: any position noise is multiplied by
25, and acceleration differences it again. A Savitzky-Golay filter fits a small quadratic to each
0.52 s window and reads velocity and acceleration off the fit. Half a window is trimmed at each
end of every continuous stretch because the filter extrapolates there (this fixed speeds above
the physical limit).

**Formation detection.** Average outfield positions in 5-minute windows (separately in and out
of possession) are clustered by depth with K-means; K is chosen by silhouette score. I also ran a
fixed three-lines variant. The result is not stable (see the reliability table) and I report it
that way.

## 3. What each metric means

- **Log loss**: average -log(probability given to what actually happened). Punishes confident
  wrong predictions heavily. Lower is better. Main metric because xG is a probability.
- **Brier score**: mean squared error of the probability. Lower is better; less harsh than log loss.
- **ROC AUC**: chance a random goal is ranked above a random non-goal. Measures ranking only; a
  badly calibrated model can have a high AUC.
- **Calibration / reliability curve**: group shots by predicted xG, compare the average prediction
  with the real goal rate in each group. On the diagonal = calibrated.
- **ECE (expected calibration error)**: the average gap between those two, weighted by group
  size. Does not see discrimination (a constant model can have low ECE), so always read it with
  log loss/AUC. It has a sampling-noise floor, reported as `ece_noise_floor`.
- **Why not accuracy**: only about 9% of shots are goals, so "never a goal" is ~91% accurate and
  useless.

## 4. How xG was validated

1. Match-level nested CV (no match in train and test; test in `tests/test_shot_features.py`).
2. Leave-one-tournament-out for generalisation.
3. Comparison with StatsBomb's own xG on the same folds (it is better; that is expected because
   it uses richer inputs and more data, and nothing here claims otherwise).
4. Paired per-fold differences to judge whether model differences exceed noise.
5. Calibration curves with a simulated noise floor; SHAP for the saved model; error analysis by
   shot type and by the worst individual predictions.
6. Sanity checks against raw events: goals per team equal the official score in every match;
   shot, goal, pass and key-pass totals reconcile (`tests/test_phase3.py`).
7. A suspicion rule: if a model beat StatsBomb's xG, the script prints a warning to investigate
   rather than reporting a win.

## 5. Things that went wrong and were fixed (good viva material)

- ECE of a constant model looked fine because tied predictions were split across bins; fixed by
  binning tied values together and adding a noise-floor reference.
- The xG race chart's goal dots floated off the line (cumulative sum over goals only), and its
  clock ran backwards because StatsBomb restarts `minute` each period; fixed with a monotonic
  match clock.
- A single match's events lack columns that never occur in that match (e.g. no through ball), which
  broke the dashboard; fixed by forcing the required columns at load time and adding a test.
- Smoothed top speeds above the physical limit came from filter edge effects and from fake ramps
  created by my own gap repair; fixed by trimming edges and by a plausibility rule on repairs.

## 5b. The optional video phase (Phase 7)

**What it does.** Splits a broadcast clip into shots, finds people and the ball with a
pretrained YOLO model, links them across frames with a tracker, splits players into two
teams by shirt colour, and for one shot maps feet positions onto the pitch.

**Why a homography.** A homography is the 3x3 matrix that maps points on one plane to another
plane as seen through a camera. The grass is (almost) flat, so the matrix that maps pitch
metres to pixels can be fitted from a few known points (corners of the 6-yard box, goalposts).
I picked eight such points by hand on one frame; the fit error on them is reported, but that
only says the matrix reproduces its own inputs.

**Why camera-motion tracking.** The broadcast camera pans and zooms from a fixed position, and
the image motion of any such camera is itself a homography. I track background points between
consecutive frames (players masked out, score graphics ignored), robustly fit that homography
(RANSAC ignores moving players), and chain the matrices back to the reference frame.

**How it was validated without labels.** I drew the pitch markings over the picture (including
the penalty arc, which was *not* used in the fit) and wrote an alignment score: how much
brighter the painted lines are under the projected lines than under the same lines shifted
sideways. A frame is mapped only if that score is above a threshold I chose after looking at
overlays. This checks the mapping against visible lines; it does not measure true player positions.

**What went wrong and was fixed.**
- A cut between two similar-looking shots (wide pitch -> tight pitch) was missed by the colour
  histogram test, so the tracker and mapping ran across it. A second signal (pixel-change
  spike) now finds it, with a test.
- The detector also reports stewards, photographers and spectators as "person"; a "feet on
  grass" test removes them (the number removed per shot is in the README table).
- ByteTrack fragmented IDs heavily during pans; BoT-SORT (with camera-motion compensation)
  fragmented less on the mapped shot (35 vs 46 IDs for about 10 players). Both still fragment.

**What I would not claim.** Any accuracy number for detection, tracking, team assignment, the
pitch positions or the speeds. There are no hand-labelled frames to measure against.

## 6. Likely viva / interview questions

**Why is your model worse than StatsBomb's?** It uses fewer inputs (no defender or goalkeeper
context in the main model), less data and simpler tuning. Adding freeze-frame features closes part
of the gap but not all of it (see the freeze-frame table).

**Why logistic regression if LightGBM and a neural net exist?** They are statistically tied here;
with ~6k shots and ~570 goals there is not enough signal for flexible models to beat a simple one,
and logistic regression is interpretable and well calibrated.

**What is data leakage and where could it have crept in?** Using information not available at the
shot, such as the outcome, where the ball ended up, or StatsBomb's own xG (which is a model output).
Another place: random splits, since shots from one match are correlated. I used match-level splits
and a test for it. Also preprocessing is fitted inside each training fold via a Pipeline.

**Why do you use log loss and calibration rather than accuracy?** Class imbalance; accuracy is
dominated by the majority class and ignores probability quality, which is what xG is for.

**What does the shot angle represent?** The angle between the lines from the shot location to the
two goalposts: how much of the goal the shooter can see.

**How do you know the model is calibrated?** Reliability curves on out-of-fold predictions, ECE
compared with the ECE sampling noise alone would give. At this sample size small differences are
not detectable.

**What does goals minus xG mean and is it skill?** Over- or under-performance relative to the
model. Over a few shots it is mostly luck; I say so in the dashboard.

**Why did you need to clean the tracking data?** The raw files had speeds of thousands of m/s,
stretches that looked like straight-line fills, players listed but off the pitch, and the pitch
direction swaps at half-time and differs between games.

**Why smooth velocity instead of using differences?** Differencing amplifies noise by the frame
rate; the filter averages it out while keeping real accelerations.

**How did you define a sprint?** At least 7 m/s (25.2 km/h) held for at least 1 second; a shorter
burst does not count (tested).

**How reliable is the formation detection?** Not very. Clusters are chosen by silhouette score on 10
points, windows are short, and teams change shape between possession states. The reliability
table shows how often the modal formation appears. I would not present it as the team's system.

**What is the Voronoi pitch control and its weakness?** Each point on the pitch is assigned to the
nearest player. It ignores velocity and reaction time, so it is only a rough picture, and a single
chosen frame can mislead.

**How do you know your player positions from video are right?** I don't, and I say so: the
homography reproduces its landmarks to under 0.15 m, projected lines sit on the painted ones in
the mapped shot, and spot checks look sensible (a player near the penalty arc appears near the arc
on the radar). Without hand-labelled positions there is no accuracy figure.

**Why BoT-SORT over ByteTrack?** I ran both on the mapped shot and compared a fragmentation proxy
(IDs issued per player on screen). BoT-SORT compensates for camera motion, which matters on a
panning broadcast. The proxy cannot count true ID switches.

**Why does the homography not hold for the whole clip?** It is only valid within one continuous
shot and while the landmarks stay visible. Cuts change the camera and zooming in removes the
reference lines; both are detected or measured and the mapping stops there.

**What would you do with more time?** More tracking matches, club data for xG, a real pitch-control
model, and a validated event-synchronised possession model; optionally the video phase.

**What are the licensing constraints?** StatsBomb: state the source and use their logo when
publishing analysis. Metrica: no formal licence; acknowledge the source and use it responsibly.
Read StatsBomb's `LICENSE.pdf` before publishing; I could not read it automatically.
