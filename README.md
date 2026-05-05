# EEEM005 Coursework — Aerial Image Segmentation

Pixel-wise semantic segmentation of aerial imagery into five classes
(Building, Road, Tree, Vehicle, Grass) using a Gaussian Naïve Bayes
classifier built from scratch with NumPy.

The implementation deliberately avoids high-level machine-learning 
libraries (scikit-learn, PyTorch, TensorFlow) in line with the assignment brief.

---

## Project layout

```
.
├── README.md                                      ← this file
├── data/
│   ├── training_image.jpg                         ← provided
│   ├── training_mask.png                          ← provided
│   ├── testing_image1.jpg                         ← provided
│   └── testing_image2.jpg                         ← provided
├── outputs/
│   ├── testing_mask1.png                          ← produced by our code
│   └── testing_mask2.png                          ← produced by our code
├── notebooks/
│   └── coursework_dev.ipynb                       ← exploration / dev notebook
├── EEEM005AnswerSheetTemplate.py                  ← provided template
├── <studentID>_<name>_solution.py                 ← final submission file
└── <studentID>_<name>_report.docx                 ← technical report
```

> The final submission is a **zip** containing the `.py` file plus the
> two output `.png` masks, with the `.docx` report submitted separately.

---

## Environment setup

Python 3.13 is required (per the assignment brief).

### Option A — venv (recommended, no extra tools needed)

```bash
### conda

```bash
conda create -n eeem005 python=3.13
conda activate eeem005
pip install -r requirements.txt
```

## Dependencies

Captured in `requirements.txt`:

```
numpy
opencv-python
scipy           # used only for one post-processing filter
jupyter         # only needed if you run the dev notebook
```

The brief allows: numpy, scipy, pandas, math, opencv (read/write/display),
and matplotlib (for visualisation only). It **disallows** sklearn,
tensorflow, pytorch, etc., for the model itself.

---

## Running the code

The submission file follows the structure mandated by
`EEEM005AnswerSheetTemplate.py`. To produce the test masks:

This will:

1. Train the model on `training_image.jpg` + `training_mask.png`.
2. Predict and save `testing_mask1.png` for `testing_image1.jpg`.
3. Predict and save `testing_mask2.png` for `testing_image2.jpg`.

---

## How the model works (in plain language)

The model is a **Gaussian Naïve Bayes** classifier operating on
**hand-crafted per-pixel features**. There are no neural networks,
no iterative training, and no gradient descent.

### Features (8 per pixel)

| # | Feature | What it captures |
|---|---|---|
| 1–3 | R, G, B | raw colour |
| 4–6 | H, S, V | hue / saturation / brightness |
| 7 | ExG = 2·G − R − B | vegetation (greenness) index |
| 8 | local std of V (7×7 window) | texture / smoothness |

### Training

For each class *k* and each feature *j*:
- compute the mean μ<sub>kj</sub> across all pixels of that class,
- compute the variance σ²<sub>kj</sub>.

### Prediction

For each test pixel, compute a log-score for each class:

```
score[k] = log P(class=k)  +  Σ over features j  log N(f_j ; μ_kj, σ²_kj)
```

…and assign the class with the highest score.

The class **prior** `P(class=k)` is set to a **uniform** 1/5 to avoid
the model collapsing onto majority classes. The marking metric is
balanced accuracy (mean recall across classes), so a uniform prior
aligns with the metric.

### Post-processing

A median filter is applied to the predicted mask to remove
salt-and-pepper noise from per-pixel-independent decisions.

---

## Results

| Image | Balanced accuracy |
|---|---|
| Training image (in-sample sanity check) | _to fill in_ |
| Test image 1 | _to fill in_ |
| Test image 2 | _to fill in_ |

---

## AI-use declaration

The brief requires acknowledging the use of AI tools. Claude for tutoring on Naïve Bayes maths, debugging help and confirming that the implementation, design choices, and writing
are my own.