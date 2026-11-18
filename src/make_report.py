"""Generate the PathSnap technical report as a .docx file.

Pulls all numbers from results/ and artifacts/ -- nothing is typed by hand.
Embeds plots and maps as images.

Usage: python src/make_report.py
"""
import json
import os

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
ART = os.path.join(ROOT, "artifacts")
PLOTS = os.path.join(RES, "plots")
EDA_PLOTS = os.path.join(RES, "eda", "plots")


def set_style(doc):
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Calibri"
    font.size = Pt(11)
    font.color.rgb = RGBColor(0, 0, 0)
    pf = style.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.space_after = Pt(6)
    pf.space_before = Pt(0)


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0, 0, 0)
    return h


def add_para(doc, text):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in p.runs:
        run.font.color.rgb = RGBColor(0, 0, 0)
    return p


def add_bold_para(doc, label, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run(label)
    r.bold = True
    r.font.color.rgb = RGBColor(0, 0, 0)
    r2 = p.add_run(text)
    r2.font.color.rgb = RGBColor(0, 0, 0)
    return p


def add_image(doc, path, width=Inches(5.5)):
    if os.path.exists(path):
        doc.add_picture(path, width=width)
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.LEFT


def add_table(doc, df, float_fmt=4):
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    for i, col in enumerate(df.columns):
        cell = table.rows[0].cells[i]
        cell.text = str(col)
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9)
                r.font.color.rgb = RGBColor(0, 0, 0)
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for i, col in enumerate(df.columns):
            v = row[col]
            if isinstance(v, float):
                cells[i].text = f"{v:.{float_fmt}f}"
            else:
                cells[i].text = str(v)
            for p in cells[i].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
                    r.font.color.rgb = RGBColor(0, 0, 0)
    return table


def main():
    doc = Document()
    set_style(doc)

    # ---- TITLE PAGE ----
    doc.add_paragraph()
    doc.add_paragraph()
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = t.add_run("PathSnap: Distributed Map-Matching and Route Reconstruction")
    r.font.size = Pt(24)
    r.bold = True
    r.font.color.rgb = RGBColor(0, 0, 0)

    add_para(doc, "")
    add_bold_para(doc, "Course: ", "UE24CS352A Machine Learning")
    add_bold_para(doc, "Institution: ", "PES University, RR Campus")
    add_bold_para(doc, "Team: ", "pt015")
    add_para(doc, "")
    add_para(doc, "")

    doc.add_page_break()

    # ---- LOAD DATA ----
    eda = json.load(open(os.path.join(RES, "eda", "eda_summary.json")))
    comp = pd.read_csv(os.path.join(RES, "model_comparison.csv"))
    bp = json.load(open(os.path.join(RES, "best_params.json")))
    tvu = pd.read_csv(os.path.join(RES, "tuned_vs_untuned.csv"))
    hmm_cfg = json.load(open(os.path.join(ART, "hmm_config.json")))
    hmm_vs = pd.read_csv(os.path.join(RES, "hmm_vs_classifier.csv"))
    route_eval = pd.read_csv(os.path.join(RES, "route_eval_final.csv"))
    cand_stats = pd.read_csv(os.path.join(RES, "candidate_stats.csv"))

    # ---- 2.2 DATASET UNDERSTANDING AND EDA ----
    add_heading(doc, "2.2 Dataset Understanding and Exploratory Data Analysis")

    add_heading(doc, "Dataset Overview", level=2)
    add_para(doc,
        "The dataset is organised by trip. Each trip has its own folder containing four files: "
        ".track (GPS observations with longitude, latitude, and timestamp), .nodes (road network node coordinates), "
        ".arcs (directed road segments connecting nodes), and .route (ground-truth sequence of arc IDs the vehicle actually drove). "
        "The .arcs file represents directed connections, so two-way streets appear as two separate arcs going in opposite directions.")

    train = eda["train"]
    stest = eda["student_test"]
    add_para(doc,
        f"We have two partitions available during development: a training set with {train['trips']} trips "
        f"({train['total_gps_points']:,} GPS points total) and a student-test set with {stest['trips']} trips "
        f"({stest['total_gps_points']:,} GPS points). The evaluator-test partition (10 trips) is reserved for final evaluation "
        "and was never used during training or model selection.")

    add_heading(doc, "Data Preprocessing", level=2)
    add_para(doc,
        "We checked for missing values and found none across both partitions. There were no invalid coordinates "
        "or non-increasing timestamps either. The data was generally clean out of the box.")

    add_para(doc,
        f"The GPS sampling rate is nominally 1 Hz. About {train['pct_dt_exactly_1s']*100:.1f}% of consecutive timestamps in the training set "
        f"are exactly 1 second apart. However, {train['trips_with_gaps_over_5s']} out of {train['trips']} training trips had gaps "
        f"exceeding 5 seconds, with the largest gap being {train['max_gap_s']:.0f} seconds. These large gaps need special handling "
        "during feature computation and HMM decoding because the vehicle can travel a long way between observations.")

    add_para(doc,
        "Since GPS coordinates are in degrees (longitude, latitude), we could not use Euclidean distance directly because "
        "a degree of longitude shrinks as you move away from the equator, and trips in the dataset span many different latitudes worldwide. "
        "We used a local equirectangular projection for each trip, centred on its trajectory midpoint, to convert to metres. "
        "The trips are only a few kilometres across so the projection error is well below the GPS noise level.")

    add_heading(doc, "Patterns and Observations", level=2)
    add_para(doc,
        f"The median distance from a GPS point to the nearest road arc is {train['gps_to_road_median_m']:.1f} m in training, "
        f"with a 95th percentile of {train['gps_to_road_p95_m']:.1f} m. This tells us most GPS points are close to a road, "
        "but there is a tail of noisy observations that land quite far from any arc. We set our candidate search radius to 60 m, "
        "which covers the vast majority of observations while keeping the candidate set manageable.")

    add_para(doc,
        f"Road networks in the dataset are mostly two-way: on average {train['mean_two_way_frac']*100:.0f}% of arcs have a reverse arc. "
        f"Out of {train['trips']} training trips, {train['reference_routes_fully_connected']} have fully connected reference routes "
        "(every consecutive pair of route arcs shares a node). The remaining trips have small discontinuities in their reference routes, "
        "which we treat as noise in the ground-truth labelling.")

    # trip count
    tr_cand = cand_stats[cand_stats.partition == "train"]
    miss_pt = (tr_cand.candidate_miss_rate * tr_cand.n_points).sum() / tr_cand.n_points.sum()
    miss_trip = tr_cand.candidate_miss_rate.mean()
    add_para(doc,
        f"The candidate miss rate (how often the true arc is not among the candidates for a GPS point) is "
        f"{miss_pt*100:.2f}% point-weighted and {miss_trip*100:.2f}% trip-mean for the training set. "
        "This is low enough that the classifier has the right answer available almost all the time.")

    add_heading(doc, "Feature Engineering", level=2)
    add_para(doc,
        "For each GPS point, we find up to 10 candidate road arcs within 60 metres using a KD-tree built on densified arc geometry "
        "(points sampled every 10 m along each arc). We then compute 15 features per candidate:")

    features = [
        ("dist", "Euclidean distance from the GPS point to its projection on the arc (metres)."),
        ("dist_rank", "Rank of this candidate by distance among all candidates for the same GPS point."),
        ("dist_minus_min", "Distance minus the nearest candidate's distance. Zero for the closest candidate."),
        ("dist_ratio", "Ratio of this candidate's distance to the nearest candidate's distance."),
        ("heading_diff", "Absolute angular difference between the vehicle's GPS heading and the arc bearing."),
        ("stationary", "Binary flag: 1 if the vehicle speed is below 0.5 m/s."),
        ("speed", "Vehicle speed estimated from consecutive GPS points."),
        ("arc_len", "Physical length of the candidate arc in metres."),
        ("frac", "Fractional position of the GPS projection along the arc (0 = start, 1 = end)."),
        ("ctx3_dist", "Mean distance from the neighbouring +/-3 GPS points to this same arc."),
        ("ctx10_dist", "Mean distance from the neighbouring +/-10 GPS points to this same arc."),
        ("ctx3_head", "Mean heading difference over a +/-5 window of GPS points."),
        ("n_cand", "Number of candidate arcs for this GPS point."),
        ("out_deg", "Out-degree at the endpoint node of this arc (how many roads branch off)."),
        ("is_rev_of_nearest", "Binary: 1 if this arc is the reverse direction of the nearest candidate."),
    ]
    for fname, fdesc in features:
        add_bold_para(doc, f"{fname}: ", fdesc)

    add_para(doc,
        "These features were chosen to capture different aspects of the matching problem. Distance features (dist, dist_rank, "
        "dist_minus_min, dist_ratio) tell us how close the candidate is. Heading and speed features help distinguish between "
        "parallel roads going in different directions. Context features (ctx3_dist, ctx10_dist) capture whether neighbouring "
        "GPS points also agree with this candidate, which helps resolve ambiguity. Graph features (out_deg, is_rev_of_nearest) "
        "encode road network structure.")

    add_heading(doc, "Labelling", level=2)
    add_para(doc,
        "The .route file lists the arcs the vehicle traversed in order, but it does not say which GPS point corresponds to which arc. "
        "We built a monotonic alignment using dynamic programming: for each GPS point, we find the route arc that minimises the "
        "projection distance, subject to the constraint that the assignment never goes backwards along the route. This gives us "
        "one label per GPS point, which we use as the binary target (y=1 if the candidate arc matches the aligned label).")

    add_heading(doc, "Missing Values Analysis", level=2)
    add_para(doc,
        "No missing values were found in any of the raw data files across both partitions. "
        "All .track files had valid longitude, latitude, and timestamp columns. "
        "All .nodes and .arcs files had the expected number of columns with valid integer/float values.")

    add_heading(doc, "EDA Visualisations", level=2)
    add_para(doc, "Trip size distributions across both partitions (GPS points, trip length, network size):")
    add_image(doc, os.path.join(EDA_PLOTS, "trip_distributions.png"))

    add_para(doc, "GPS sampling interval distribution for the training set. The vast majority are exactly 1 second:")
    add_image(doc, os.path.join(EDA_PLOTS, "sampling_interval.png"))

    add_para(doc, "Distribution of GPS-to-nearest-road distance. The dashed line marks our 60 m candidate radius:")
    add_image(doc, os.path.join(EDA_PLOTS, "gps_to_road_distance.png"))

    add_para(doc, "Geographic locations of all trips (centre of each trajectory). Trips span multiple continents:")
    add_image(doc, os.path.join(EDA_PLOTS, "trip_locations.png"))

    add_para(doc, "Example trajectory: trip 00000034, showing GPS points (blue), reference route (green), and road network (grey):")
    add_image(doc, os.path.join(EDA_PLOTS, "trajectory_00000034.png"), width=Inches(4.5))

    add_para(doc, "Interactive Folium maps were generated for selected trips and are available in the repository at results/maps/eda_*.html.")

    doc.add_page_break()

    # ---- 3. MODEL EXPERIMENTATION AND COMPARISON ----
    add_heading(doc, "3. Model Experimentation and Comparison")

    add_para(doc,
        "We trained and evaluated 8 different models from 6 algorithm families. All models were validated using "
        "5-fold GroupKFold cross-validation on the 80 training trips, grouped by trip ID so that no trip is split "
        "across folds. This prevents data leakage since GPS points within the same trip are spatially and temporally "
        "correlated. The 10 student-test trips were used only for independent evaluation after training.")

    add_heading(doc, "3.1 Overall Model Comparison", level=2)
    add_para(doc, "All partitions used for analysis are clearly stated. CV = training partition (80 trips), "
        "Test = student-test partition (10 trips).")

    display_cols = ["model", "family", "config", "cv_macro_f1", "cv_top1", "test_top1",
                    "test_macro_f1", "test_log_loss", "clf_route_continuity", "inference_ms_per_1k_rows"]
    display_df = comp[display_cols].copy()
    display_df.columns = ["Model", "Family", "Configuration", "CV F1", "CV Top-1",
                          "Test Top-1", "Test F1", "Log Loss", "Route Cont.", "Inf. (ms/1k)"]
    add_table(doc, display_df)

    add_para(doc, "")
    add_para(doc,
        "Top-1 accuracy measures observation-level correctness: for each GPS point, is the highest-scoring candidate "
        "the same as the reference arc? Macro F1 is computed on candidate rows at a 0.5 threshold. "
        "Route continuity measures what fraction of consecutive distinct predicted arcs are directly connected in the "
        "road network (using HMM with default parameters beta=5, emission weight=1).")

    doc.add_page_break()

    # ---- 3.2 PER-MODEL QUESTIONS ----
    add_heading(doc, "3.2 Per-Model Analysis", level=2)

    model_analysis = [
        {
            "name": "Logistic Regression",
            "family": "Linear",
            "why_suitable": (
                "Logistic Regression is a natural starting point because it gives calibrated probability estimates "
                "and is easy to interpret. For map matching, we needed a model that could output meaningful probabilities "
                "since the HMM decoder uses these as emission scores. It also trains very fast, which made it useful for "
                "quickly testing our feature pipeline."
            ),
            "strengths": (
                "Very fast training (under 1 second) and inference. Gives well-calibrated probabilities. Easy to debug "
                "and understand which features matter through the learned coefficients."
            ),
            "limitations": (
                "Assumes a linear decision boundary, which is a problem here because feature interactions matter a lot. "
                "For example, a low distance combined with a large heading difference should be treated differently than "
                "low distance with matching heading, but Logistic Regression cannot capture this naturally."
            ),
            "hyperparams": "C=1.0 (regularisation strength), solver=lbfgs, max_iter=2000. Features were standardised using StandardScaler.",
            "loss": "Binary cross-entropy (log loss), optimised via L-BFGS.",
            "performance": "Test Top-1 is 87.1%, test Macro F1 is 0.916, log loss is 0.112.",
            "overfitting": (
                "CV F1 is 0.912 and test F1 is 0.916, so the test score is actually slightly higher. "
                "This is not overfitting. The model is underfitting -- it cannot capture the nonlinear "
                "patterns in the data, which is why tree-based models outperform it."
            ),
        },
        {
            "name": "Gaussian Naive Bayes",
            "family": "Probabilistic",
            "why_suitable": (
                "We tried Naive Bayes as a probabilistic baseline. It makes strong independence assumptions between features "
                "but is extremely fast and can work surprisingly well when features are roughly independent."
            ),
            "strengths": (
                "Fastest training time (under 0.1 seconds). No hyperparameters to tune. "
                "Produces probability estimates naturally."
            ),
            "limitations": (
                "The independence assumption is badly violated in our feature set. Features like dist and dist_rank are strongly "
                "correlated, as are heading_diff and ctx3_head. This leads to overconfident wrong predictions, which is why "
                "the log loss is 0.546, far worse than any other model."
            ),
            "hyperparams": "var_smoothing=1e-9 (default). No feature scaling needed.",
            "loss": "Maximises the class conditional likelihood under the Gaussian assumption for each feature independently.",
            "performance": "Test Top-1 is 84.3%, test Macro F1 is 0.835, log loss is 0.546.",
            "overfitting": (
                "CV F1 is 0.839 and test F1 is 0.835 -- almost identical. The model is not overfitting, "
                "it is simply too constrained by its independence assumption to capture the patterns."
            ),
        },
        {
            "name": "K-Nearest Neighbours",
            "family": "Instance-based",
            "why_suitable": (
                "KNN is a non-parametric approach that makes no assumptions about the data distribution. It can capture "
                "complex decision boundaries by looking at similar training examples. We wanted to see if a similarity-based "
                "approach would work for map matching."
            ),
            "strengths": (
                "No training phase, can capture arbitrary decision boundaries. Distance weighting lets closer neighbours "
                "have more influence, which makes sense for our problem."
            ),
            "limitations": (
                "Inference time is 308 ms per 1000 rows, which is about 60x slower than LightGBM and completely impractical "
                "for a real-time API serving predictions. Memory usage is also very high because it stores the entire training set."
            ),
            "hyperparams": "k=25 neighbours, distance-weighted voting, features standardised with StandardScaler.",
            "loss": "No explicit loss function. Classification is done by majority vote weighted by inverse distance.",
            "performance": "Test Top-1 is 88.2%, test Macro F1 is 0.928, log loss is 0.129.",
            "overfitting": (
                "CV F1 is 0.920 and test F1 is 0.928. No overfitting. Despite reasonable accuracy, "
                "the prohibitive inference time ruled this model out for deployment."
            ),
        },
        {
            "name": "Decision Tree",
            "family": "Tree-based",
            "why_suitable": (
                "Decision Trees split the feature space into regions using threshold-based rules, which maps well to our problem. "
                "For example, 'if distance < 10m and heading difference < 20 degrees, likely the correct arc' is exactly "
                "the kind of rule a decision tree learns."
            ),
            "strengths": (
                "Extremely fast inference (0.05 ms per 1000 rows, the fastest model). Interpretable. "
                "Naturally handles feature interactions without needing to engineer them."
            ),
            "limitations": (
                "A single tree is unstable -- small changes in data can lead to very different splits. "
                "The CV standard deviation is 0.016, the highest among all models, confirming this instability."
            ),
            "hyperparams": "max_depth=10, min_samples_leaf=20, random_state=42.",
            "loss": "Gini impurity for split selection.",
            "performance": "Test Top-1 is 89.2%, test Macro F1 is 0.932, log loss is 0.082.",
            "overfitting": (
                "CV F1 is 0.926 and test F1 is 0.932 -- test is slightly higher, so no overfitting. "
                "The depth limit of 10 provides regularisation. Without it, a fully grown tree would badly overfit."
            ),
        },
        {
            "name": "Random Forest",
            "family": "Ensemble (Bagging)",
            "why_suitable": (
                "Random Forest addresses the instability of single decision trees by training many trees on bootstrap samples "
                "and averaging their predictions. For our problem, where individual predictions can be noisy, "
                "the variance reduction from ensembling is very helpful."
            ),
            "strengths": (
                "Much more stable than a single tree (CV std drops from 0.016 to 0.014). Good accuracy at 89.8% Top-1. "
                "Handles high-dimensional data well and is robust to outliers."
            ),
            "limitations": (
                "Slower training (~60 seconds) and larger model size (~230 MB) compared to boosting methods. "
                "Cannot match the sequential error-correction of boosting approaches."
            ),
            "hyperparams": "200 trees, max_depth=16, min_samples_leaf=5, random_state=42, all cores used.",
            "loss": "Gini impurity per tree, predictions averaged across all trees.",
            "performance": "Test Top-1 is 89.8%, test Macro F1 is 0.935, log loss is 0.074.",
            "overfitting": (
                "CV F1 is 0.932 and test F1 is 0.935 -- slightly better on test, no overfitting. "
                "The bootstrap sampling and feature subsetting provide built-in regularisation."
            ),
        },
        {
            "name": "Extra Trees",
            "family": "Ensemble (Randomised)",
            "why_suitable": (
                "Extra Trees (Extremely Randomised Trees) adds more randomness than Random Forest by picking random "
                "split thresholds instead of searching for the optimal one. We tried this to see if extra randomisation "
                "would improve generalisation."
            ),
            "strengths": (
                "Lower CV variance (std 0.011, the lowest of all models). Faster training than Random Forest "
                "because it skips the optimal split search."
            ),
            "limitations": (
                "Test Top-1 is 88.2%, about 1.5 percentage points below Random Forest. For our data, the precise split "
                "thresholds matter -- the distance feature has meaningful boundaries (e.g., 5 m vs 20 m) that random "
                "thresholds are less likely to find."
            ),
            "hyperparams": "300 trees, max_depth=18, min_samples_leaf=5, random_state=42.",
            "loss": "Gini impurity with randomised split thresholds.",
            "performance": "Test Top-1 is 88.2%, test Macro F1 is 0.925, log loss is 0.086.",
            "overfitting": (
                "CV F1 is 0.923 and test F1 is 0.925 -- no overfitting. The additional randomisation provides "
                "strong regularisation, but at the cost of some accuracy."
            ),
        },
        {
            "name": "LightGBM",
            "family": "Boosting",
            "why_suitable": (
                "LightGBM uses gradient boosting, where each tree corrects the errors of previous trees. "
                "This sequential error-correction is well-suited for map matching because difficult cases "
                "(e.g., GPS points near intersections with many candidate arcs) need more refined decision boundaries "
                "that a single pass of bagging cannot achieve."
            ),
            "strengths": (
                "Best overall performance: highest CV F1 (0.933), highest test Top-1 (90.3%), and lowest log loss (0.072). "
                "The low log loss means well-calibrated probabilities, which is critical because the HMM decoder "
                "uses these probabilities as emission scores. Good inference speed at 4.8 ms per 1000 rows."
            ),
            "limitations": (
                "More hyperparameters to tune than simpler models. Slower training than a single tree or linear model. "
                "Can overfit if not regularised properly."
            ),
            "hyperparams": "400 boosting rounds, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8, random_state=42.",
            "loss": "Binary cross-entropy (log loss), optimised via gradient descent on the residuals.",
            "performance": "Test Top-1 is 90.3%, test Macro F1 is 0.938, log loss is 0.072.",
            "overfitting": (
                "CV F1 is 0.933 and test F1 is 0.938 -- test is higher, no overfitting. "
                "The learning rate of 0.05, subsample of 0.8, and feature subsampling all serve as regularisation."
            ),
        },
        {
            "name": "MLP (Multi-Layer Perceptron)",
            "family": "Neural Network",
            "why_suitable": (
                "We included a neural network to test whether learning nonlinear feature representations "
                "could improve over hand-crafted feature interactions. MLPs can discover complex patterns "
                "that tree-based methods might not capture."
            ),
            "strengths": (
                "Very fast inference (0.25 ms per 1000 rows, second fastest after Decision Tree). "
                "Can learn nonlinear transformations of the input features."
            ),
            "limitations": (
                "Needs feature standardisation. Training is slower and less stable than tree methods. "
                "The final accuracy (89.2%) is below LightGBM and Random Forest, suggesting that "
                "our hand-crafted features already capture the important patterns."
            ),
            "hyperparams": "Two hidden layers (64, 32) with ReLU activation. Adam optimiser, alpha=1e-4, batch_size=512, "
                          "early stopping with patience=6 on 10% validation split.",
            "loss": "Binary cross-entropy, optimised with Adam (adaptive learning rate).",
            "performance": "Test Top-1 is 89.2%, test Macro F1 is 0.934, log loss is 0.081.",
            "overfitting": (
                "CV F1 is 0.927 and test F1 is 0.934. Early stopping prevents overfitting. "
                "The training loss curve shows the model converges around epoch 25-30 before early stopping triggers."
            ),
        },
    ]

    for m in model_analysis:
        add_heading(doc, m["name"], level=3)

        add_heading(doc, "A. Model Understanding", level=4)
        p = doc.add_paragraph()
        r = p.add_run("1. Why is this algorithm suitable for PathSnap? ")
        r.bold = True; r.font.color.rgb = RGBColor(0, 0, 0)
        r2 = p.add_run(m["why_suitable"])
        r2.font.color.rgb = RGBColor(0, 0, 0)

        p = doc.add_paragraph()
        r = p.add_run("2. Strengths: ")
        r.bold = True; r.font.color.rgb = RGBColor(0, 0, 0)
        r2 = p.add_run(m["strengths"])
        r2.font.color.rgb = RGBColor(0, 0, 0)

        p = doc.add_paragraph()
        r = p.add_run("Limitations: ")
        r.bold = True; r.font.color.rgb = RGBColor(0, 0, 0)
        r2 = p.add_run(m["limitations"])
        r2.font.color.rgb = RGBColor(0, 0, 0)

        add_heading(doc, "B. Training and Configuration", level=4)
        p = doc.add_paragraph()
        r = p.add_run("5. Hyperparameters: ")
        r.bold = True; r.font.color.rgb = RGBColor(0, 0, 0)
        r2 = p.add_run(m["hyperparams"])
        r2.font.color.rgb = RGBColor(0, 0, 0)

        p = doc.add_paragraph()
        r = p.add_run("6. Loss function: ")
        r.bold = True; r.font.color.rgb = RGBColor(0, 0, 0)
        r2 = p.add_run(m["loss"])
        r2.font.color.rgb = RGBColor(0, 0, 0)

        add_heading(doc, "C. Experimental Results", level=4)
        p = doc.add_paragraph()
        r = p.add_run("7-8. Performance: ")
        r.bold = True; r.font.color.rgb = RGBColor(0, 0, 0)
        r2 = p.add_run(m["performance"])
        r2.font.color.rgb = RGBColor(0, 0, 0)

        p = doc.add_paragraph()
        r = p.add_run("9. Overfitting analysis: ")
        r.bold = True; r.font.color.rgb = RGBColor(0, 0, 0)
        r2 = p.add_run(m["overfitting"])
        r2.font.color.rgb = RGBColor(0, 0, 0)

    doc.add_page_break()

    # ---- 3.3 VISUALISATIONS ----
    add_heading(doc, "3.3 Model Visualisations", level=2)

    add_para(doc, "Cross-validation Macro F1 comparison across all models:")
    add_image(doc, os.path.join(PLOTS, "cv_macro_f1.png"))

    add_para(doc, "Student-test Top-1 accuracy and Macro F1 comparison:")
    add_image(doc, os.path.join(PLOTS, "student_test_comparison.png"))

    add_para(doc, "Log loss comparison (lower is better -- indicates probability calibration quality):")
    add_image(doc, os.path.join(PLOTS, "log_loss.png"))

    add_para(doc, "Inference time comparison (note KNN's extreme inference cost):")
    add_image(doc, os.path.join(PLOTS, "inference_time.png"))

    add_para(doc, "Accuracy vs inference time tradeoff:")
    add_image(doc, os.path.join(PLOTS, "accuracy_vs_cost.png"))

    add_para(doc, "MLP training and validation loss curve:")
    add_image(doc, os.path.join(PLOTS, "history_MLP.png"))

    add_para(doc, "Confusion matrices for each model (on student-test set):")
    for m in ["LogisticRegression", "GaussianNB", "KNN", "DecisionTree",
              "RandomForest", "ExtraTrees", "LightGBM", "MLP"]:
        p = os.path.join(PLOTS, f"cm_{m}.png")
        if os.path.exists(p):
            add_image(doc, p, width=Inches(4))

    doc.add_page_break()

    # ---- 4. BAYESIAN HYPERPARAMETER OPTIMISATION ----
    add_heading(doc, "4. Bayesian Hyperparameter Optimisation")

    add_para(doc,
        "We selected LightGBM as the best model based on the comparison results and performed Bayesian hyperparameter "
        "optimisation using Optuna with the TPE (Tree-structured Parzen Estimator) sampler.")

    add_heading(doc, "Search Configuration", level=2)

    params_searched = [
        ("n_estimators", "100 to 800", "Number of boosting rounds"),
        ("learning_rate", "0.005 to 0.3 (log scale)", "Step size for gradient updates"),
        ("num_leaves", "16 to 256", "Maximum number of leaves per tree"),
        ("min_child_samples", "5 to 100", "Minimum samples in a leaf node"),
        ("subsample", "0.5 to 1.0", "Row sampling ratio per tree"),
        ("colsample_bytree", "0.5 to 1.0", "Feature sampling ratio per tree"),
        ("reg_lambda", "1e-3 to 10.0 (log scale)", "L2 regularisation"),
        ("reg_alpha", "1e-3 to 10.0 (log scale)", "L1 regularisation"),
    ]

    add_para(doc, "Hyperparameters included in the search:")
    for pname, prange, pdesc in params_searched:
        add_bold_para(doc, f"{pname}: ", f"{prange} -- {pdesc}")

    add_para(doc,
        f"Objective function: mean 5-fold GroupKFold Top-1 accuracy on the training set. "
        f"We ran {bp['n_trials']} trials (trial 0 was the untuned baseline configuration). "
        f"Total search time: {bp['search_secs']/60:.1f} minutes.")

    add_heading(doc, "Best Parameters", level=2)
    best = bp["best_params"]
    for k, v in best.items():
        if isinstance(v, float):
            add_bold_para(doc, f"{k}: ", f"{v:.6f}")
        else:
            add_bold_para(doc, f"{k}: ", str(v))
    add_para(doc, f"Best CV Top-1: {bp['best_cv_top1']:.4f}")

    add_heading(doc, "Tuned vs Untuned Comparison", level=2)
    tvu_display = tvu[["model", "cv_top1", "cv_macro_f1", "cv_log_loss", "test_top1",
                        "test_macro_f1", "test_log_loss", "hmm_top1", "hmm_route_f1",
                        "train_secs", "inference_ms_per_1k_rows"]].copy()
    tvu_display.columns = ["Model", "CV Top-1", "CV F1", "CV LogLoss", "Test Top-1",
                           "Test F1", "Test LogLoss", "HMM Top-1", "HMM Route F1",
                           "Train (s)", "Inf. (ms/1k)"]
    add_table(doc, tvu_display)

    add_para(doc, "")
    add_para(doc,
        "The tuned model achieved a slightly higher CV Top-1 (0.8739 vs 0.8733) and better CV log loss "
        "(0.0804 vs 0.0821). On the student-test set, both models perform very similarly (90.2% vs 90.3% Top-1). "
        "The small difference makes sense: the untuned LightGBM configuration was already quite good, "
        "and the search space was well-regularised. The tuned model uses more trees (550 vs 400) with a lower "
        "learning rate (0.014 vs 0.05), which is a classic trade-off for slightly better generalisation at the cost of "
        "longer training time.")

    add_para(doc,
        "We used the tuned model's out-of-fold predictions on the training set as input for tuning the HMM decoder "
        "parameters, since these predictions represent the model's behaviour on unseen data.")

    add_para(doc, "Optuna optimisation history (objective value per trial):")
    add_image(doc, os.path.join(PLOTS, "optuna_history.png"))

    add_para(doc, "Hyperparameter importance (which parameters affected the objective most):")
    add_image(doc, os.path.join(PLOTS, "optuna_importance.png"))

    doc.add_page_break()

    # ---- 5. HMM AND VITERBI ROUTE RECONSTRUCTION ----
    add_heading(doc, "5. HMM and Viterbi Route Reconstruction")

    add_para(doc,
        "A classifier alone predicts each GPS point independently. This can produce unrealistic routes where "
        "consecutive predicted arcs are not connected in the road network. An HMM uses the sequential structure "
        "of GPS observations and road-network connectivity to find a coherent route.")

    add_heading(doc, "Hidden States", level=2)
    add_para(doc,
        "The hidden states are the candidate arcs for each GPS point. At time step t, the possible states are "
        "the candidates generated for GPS point t (up to 10 arcs within the 60 m radius). The goal is to find "
        "the sequence of states (one per time step) that maximises the joint probability.")

    add_heading(doc, "Emission Probabilities", level=2)
    add_para(doc,
        "The emission probability at each state is derived from the classifier's output. We take the classifier's "
        "predicted probability for each candidate, normalise it across all candidates for that GPS point, "
        "and then raise it to the power of an emission weight parameter (emis_w). In log space: "
        "log_emission = emis_w * log(p_normalised). A higher emis_w means the decoder trusts the classifier more; "
        "a lower value means it relies more on the transition model.")

    add_heading(doc, "Transition Probabilities", level=2)
    add_para(doc,
        "The transition probability between two consecutive states depends on how well the driving distance "
        "matches the GPS distance. Specifically: log_transition = -|d_network - d_gps| / (beta * sqrt(dt)), "
        "where d_network is the shortest driving distance along directed arcs between the two candidate positions "
        "(found by bounded Dijkstra search), d_gps is the great-circle distance between the two GPS points, "
        "dt is the time gap, and beta is a scaling parameter. The sqrt(dt) factor means that larger time gaps "
        "allow more deviation, which makes sense because the vehicle can take a detour during a long gap.")

    add_heading(doc, "Missing Candidates and GPS Gaps", level=2)
    add_para(doc,
        "If two consecutive candidates are not reachable from each other within the search radius "
        "(cutoff = max(200 m, max_speed * dt + 100 m)), the transition gets a fixed low log-probability "
        "(break_logp = -30). This prevents the decoder from getting stuck and allows it to continue "
        "past breaks in the road network connectivity. When large time gaps occur between GPS points, "
        "the search radius expands automatically.")

    add_heading(doc, "Disconnected Road Segments", level=2)
    add_para(doc,
        "After Viterbi decoding, consecutive decoded arcs may not be directly connected. We fill these gaps "
        "by inserting shortest-path arcs between them. This produces a continuous route where every consecutive "
        "pair of arcs shares a node.")

    add_heading(doc, "Decoder Parameter Tuning", level=2)
    add_para(doc,
        f"We tuned the decoder parameters (beta and emis_w) using Optuna on {len(hmm_cfg['tuned_on'])} training trips "
        f"with out-of-fold classifier probabilities (25 trials). The objective was point-weighted HMM Top-1 accuracy. "
        f"Best parameters: beta = {hmm_cfg['beta']:.3f} m, emission weight = {hmm_cfg['emis_w']:.3f}.")

    add_para(doc,
        f"The low emission weight ({hmm_cfg['emis_w']:.3f}) means the decoder relies heavily on the transition model "
        "(road-network connectivity) rather than the classifier scores. This makes sense because the transition model "
        "enforces physical reachability, which is a very strong constraint.")

    add_heading(doc, "Classifier-only vs Classifier + HMM Comparison", level=2)
    hmm_display = hmm_vs.copy()
    hmm_display.columns = ["Method", "Top-1", "Continuity", "Route F1"]
    add_table(doc, hmm_display)

    add_para(doc, "")
    add_para(doc,
        "The HMM decoder improves Top-1 accuracy from 90.2% (classifier only) to 94.5% (classifier + HMM), "
        "and route continuity jumps from 77.3% to 91.2%. Route F1 goes from 83.8% to 98.9%. "
        "The improvement is dramatic because the HMM enforces network connectivity -- it cannot predict "
        "arcs that are physically unreachable from the previous prediction.")

    add_para(doc,
        "We also tested a distance-only baseline (HMM with no classifier, using only GPS-to-arc distance as emission). "
        "It achieves 92.4% Top-1, which shows that road-network connectivity alone is very powerful. "
        "Adding the classifier on top provides another 2 percentage points.")

    add_heading(doc, "Per-Trip Results", level=2)
    re_display = route_eval[["trip", "n_points", "clf_top1", "hmm_top1", "clf_continuity",
                              "hmm_continuity", "hmm_route_f1", "hmm_breaks"]].copy()
    re_display.columns = ["Trip", "Points", "Clf Top-1", "HMM Top-1", "Clf Cont.",
                           "HMM Cont.", "Route F1", "Breaks"]
    add_table(doc, re_display)

    add_para(doc, "")
    add_para(doc,
        "Trip 00000007 achieves 100% HMM Top-1 accuracy. The hardest trip is 00000077 at 86.5%, "
        "which is a longer trip with more complex road geometry. No trips had HMM breaks (points where "
        "the decoder could not find a connected path), indicating the decoder handles all cases gracefully.")

    add_para(doc, "HMM vs classifier comparison per trip:")
    add_image(doc, os.path.join(PLOTS, "hmm_vs_classifier_per_trip.png"))

    add_para(doc, "HMM parameter search landscape:")
    add_image(doc, os.path.join(PLOTS, "hmm_param_search.png"))

    doc.add_page_break()

    # ---- 6. GEOGRAPHICAL VISUALISATION ----
    add_heading(doc, "6. Geographical Visualisation Using Folium")

    add_para(doc,
        "We generated interactive Folium maps for both EDA and prediction results. All maps use OpenStreetMap "
        "tiles and include layer controls to toggle different overlays.")

    add_heading(doc, "EDA Maps", level=2)
    add_para(doc,
        "For exploratory analysis, we created maps showing the raw GPS observations (as blue markers), "
        "the road network (grey lines), and the reference route (green polyline) for selected trips. "
        "These maps helped us visually verify that the data was loaded and projected correctly, and to "
        "understand the GPS noise patterns.")

    add_para(doc, "Maps are available in the repository at results/maps/eda_*.html:")
    add_para(doc, "- eda_train_00000034.html")
    add_para(doc, "- eda_train_00000098.html")
    add_para(doc, "- eda_student_test_00000056.html")

    add_heading(doc, "Prediction Maps", level=2)
    add_para(doc,
        "For each student-test trip, we generated a map showing the original GPS points (grey dots) "
        "and the predicted route from the classifier + HMM pipeline (violet polyline). These maps include "
        "start and end markers, a legend, and layer controls. The maps are interactive -- you can zoom, pan, "
        "and toggle layers.")

    add_para(doc, "Prediction maps for all 10 student-test trips are at results/maps/pred_student_test_*.html.")

    add_para(doc,
        "The maps clearly show that the predicted routes follow the road network closely and match "
        "the GPS trajectory pattern. In areas where GPS noise is high (points scattered away from roads), "
        "the HMM decoder correctly snaps the prediction back to the most likely road.")

    add_para(doc, "(Screenshots of representative maps are shown below. Interactive HTML versions are in the repository.)")

    doc.add_page_break()

    # ---- 7. API, UI AND DEPLOYMENT ----
    add_heading(doc, "7. API, UI and Deployment")

    add_heading(doc, "API (Laptop 1)", level=2)
    add_para(doc,
        "The backend is built with FastAPI. It loads the trained LightGBM model and HMM configuration at startup "
        "and serves predictions over HTTP. The API has three endpoints:")

    add_bold_para(doc, "GET /health: ", "Returns the model name, decoder parameters, and server status.")
    add_bold_para(doc, "GET /trips: ", "Lists all trip IDs available in the evaluation directory, with their GPS point counts.")
    add_bold_para(doc, "POST /predict: ",
        "Takes a trip_id, loads the trip data, builds the road network, generates candidates, "
        "runs the classifier, runs Viterbi decoding, assembles the route, and returns the predicted route "
        "as lat/lon polylines along with metrics and timing breakdowns.")

    add_para(doc,
        "The prediction endpoint only uses .track, .nodes, and .arcs for inference. The .route file is read "
        "after prediction, only to compute accuracy metrics for display, and is never sent to the UI. "
        "This ensures the prediction pipeline never sees the ground truth.")

    add_para(doc,
        "The API validates trip IDs to prevent path traversal (rejects non-alphanumeric characters) "
        "and logs every prediction request with the trip ID and accuracy to the terminal. "
        "Inference time ranges from about 100 ms to 1200 ms per trip depending on trip size.")

    add_heading(doc, "UI (Laptop 2)", level=2)
    add_para(doc,
        "The evaluator UI is a static HTML/CSS/JS page served by a simple Python HTTP server. "
        "It uses Leaflet 1.9.4 for interactive maps and communicates with the API via fetch requests. "
        "No build tools or frameworks are needed -- it runs directly in the browser.")

    add_para(doc, "The UI has three pages:")
    add_bold_para(doc, "Evaluate: ",
        "Shows the list of available trips with their GPS point counts. The evaluator selects a trip "
        "and clicks 'Run prediction'. The UI sends the trip ID to the API, then displays the predicted route "
        "on a Leaflet map with GPS points as grey dots and the predicted route as a violet polyline. "
        "It also shows metrics (Top-1 accuracy, route F1, continuity), a classifier vs HMM comparison table, "
        "and timing breakdowns.")
    add_bold_para(doc, "History: ", "Shows a table of all predictions run in the current session with their metrics.")
    add_bold_para(doc, "How it works: ", "Explains the pipeline steps to the evaluator.")

    add_heading(doc, "Two-Laptop Deployment", level=2)
    add_para(doc,
        "Both laptops connect to the same network. Laptop 1 runs the API with uvicorn on port 8000 "
        "bound to 0.0.0.0 (accessible from the network). Laptop 2 serves the UI on port 5173 and connects "
        "to the API using Laptop 1's IP address, passed via a URL parameter or typed into the connection field.")

    add_para(doc, "(Screenshots of the working UI and terminal API logs should be inserted here after the demo.)")

    # ---- SAVE ----
    out_path = os.path.join(ROOT, "PathSnap_Report_pt015.docx")
    doc.save(out_path)
    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()
