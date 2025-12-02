# Figure Placement & Content Guide

`docs/ABSTRACT_AND_RELATED.md` references a set of figures stored in `figures/`. Populate each placeholder with your actual screenshots/plots before regenerating `docs/AutoML_Benchmark_Paper.docx`. Use this guide as a checklist so you know exactly what to capture and where it appears in the paper.

---

## Section 3 — Methods

### Figure 0 – `figures/method_flow.png`
- **What to capture:** A high-level flow diagram illustrating dataset discovery → guardrails → CV training → analysis → dashboards/serving.
- **Where it shows up:** Immediately after the Section 3 heading, before 3.1.
- **How to create:** Use PowerPoint, draw.io, or any diagram tool. Save as PNG with the exact filename.

### Figure 7 – `figures/data_collection_code_placeholder.png`
- **What to capture:** Screenshot of the `DataCollector` class (in `Project/utils/io.py` or helper scripts) showing `import_from_local_file`, `import_from_mysql`, etc.
- **Where it shows up:** Section 3.4 “Data Collection (multi-source ingestion).”
- **Tip:** Highlight the method signatures so readers can see the input types and normalization logic.

---

## Section 4.1 — Classifier Visualisations

These plots summarize model metrics. You can export them from Excel, Matplotlib, or any plotting tool.

1. **Figure 4 – `figures/classifier_accuracy_placeholder.png`**  
   - Bar/line chart of Accuracy for the top five classifiers (Random Forest, Decision Tree, etc.).
2. **Figure 5 – `figures/classifier_precision_placeholder.png`**  
   - Bar/line chart of Precision for the same classifiers.  
3. **Figure 6 – `figures/classifier_histogram_placeholder.png`**  
   - Combined histogram showing Accuracy, Precision, Recall, and F1 side-by-side.

Place these images under Section 4.1 in the Markdown; they already have slots.

---

## Section 4.2 — Module Outputs (Modules 4.4–4.9)

Provide screenshots of the supporting code/EDA cells. Recommended sources:

| Figure | Filename | Suggested Source | Notes |
|--------|----------|------------------|-------|
| Figure 8 | `figures/data_processing_code_placeholder.png` | Python script (≈253 lines) handling cleaning/encoding | Capture the section that treats missing values and type conversions. |
| Figure 9 | `figures/initial_exploration_code_placeholder.png` | Notebook/EDA script (~25 lines) | Show initial `.head()`/`.describe()` outputs or Plotly setup. |
| Figure 10 | `figures/vif_code_placeholder.png` | VIF function (Section 3.3) | Highlight how multicollinearity is removed. |
| Figure 11 | `figures/new_feature_generation_code_placeholder.png` | Featuretools snippet | Showcase how new features are synthesized. |
| Figure 12 | `figures/plotly_exploration_placeholder.png` | Plotly visualization (map, scatter, etc.) | Example EDA result, e.g., lat/long choropleth. |

Save each screenshot with the exact filename.

---

## After Updating Images

1. Replace placeholder PNGs in `figures/` with your actual images (same filenames).  
2. Regenerate the Word document:  
   ```bash
   pandoc docs/ABSTRACT_AND_RELATED.md -o docs/AutoML_Benchmark_Paper.docx
   ```  
3. Open `docs/AutoML_Benchmark_Paper.docx` to confirm the new figures appear correctly.  
4. Repeat whenever you update the Markdown or figures.

This workflow ensures your `.md` and `.docx` stay in sync, and you always know what content each figure should include.
