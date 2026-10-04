# NBA Shot Make Probability Model

Predicting the probability that an NBA shot goes in, using player tracking data.
Built as a take-home analytics project on ~425,000 non-fouled regular season shots.

**Result: 0.6258 log-loss against a 0.6897 baseline**, validated leave-one-season-out.

## What it does

A LightGBM model estimates make probability for any shot from 24 features covering
four things: the shot itself (distance, type, shot clock), where it was taken from,
what the defenders were doing in the second before release, and how often the shooter
shoots.

The training data covered two seasons and the test set was a third, unseen season.
That ruled out random cross-validation, which would have mixed shots from the same
season into both training and validation and inflated the score. Validation instead
trains on one season and validates on the other, so every validation shot comes from
a season the model never saw.

## A few findings

- The conventional shooter-skill feature (career make rate) improved validation by
  0.0009, but the entire gain was leakage. Rebuilt out-of-fold it contributed nothing.
  Shot **volume** was used instead, since a count carries no outcome information.
- Defender closing behavior encodes the defense's own read of shot quality, though most
  of that signal turned out to be redundant with shot type.
- Five of eight feature ideas were tested and rejected, each with a measured result.

Full reasoning, every ablation, and the pitfalls are in the writeup and notebook.

## Files

- `project_code.py` — end-to-end script: load, build features, fit, predict
- `notebook.ipynb` — exploration, validation design, and all feature tests
- `project_writeup.pdf` — written analysis

Data files are not included in this repository.

## Running it

```bash
pip install pandas numpy lightgbm scikit-learn matplotlib
python project_code.py
```
