import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def write_pseudobulk_and_metadata(
    adata,
    celltype_var="celltype",
    Individual_var="individual",
    Group_var="disease",
    outdir="Pseudobulk",
):
    """Generate pseudobulk count and metadata CSVs from an AnnData object."""
    os.makedirs(outdir, exist_ok=True)
    cell_types = adata.obs[celltype_var].unique().tolist()
    for celltype in cell_types:
        bdata = adata[adata.obs[celltype_var] == celltype, :].copy()
        bdata.X = bdata.layers["counts"]
        bdata.obs["clustersample"] = (
            bdata.obs[Individual_var].astype(str) + "_" + bdata.obs[celltype_var].astype(str)
        ).astype("category")
        res = pd.DataFrame(
            columns=bdata.var_names,
            index=bdata.obs["clustersample"].cat.categories,
        )
        for clust in bdata.obs["clustersample"].cat.categories:
            res.loc[clust] = bdata[bdata.obs["clustersample"] == clust, :].X.sum(0)
        res.T.to_csv(os.path.join(outdir, f"{celltype}_pseudobulk_counts.csv"))
        metadata = (
            bdata.obs
            .drop_duplicates("clustersample")[
                ["clustersample", celltype_var, Individual_var, Group_var]
            ]
            .rename(columns={"clustersample": "Sample"})
        )
        metadata.to_csv(os.path.join(outdir, f"{celltype}_metadata.csv"), index=False)


def _detect_cols(df):
    cols_lower = {c.lower(): c for c in df.columns}
    for k in ("logfc", "log2fc", "log2foldchange"):
        if k in cols_lower:
            logfc_col = cols_lower[k]
            break
    else:
        raise ValueError("Could not find a logFC column")
    for k in ("fdr", "adj.p.val", "padj", "fdr_bh", "qvalue", "pvalue", "p.value"):
        if k in cols_lower:
            fdr_col = cols_lower[k]
            break
    else:
        raise ValueError("Could not find an FDR/adj.P.Val/Padj column")
    return logfc_col, fdr_col


def run_edger_batch(
    folder=".",
    rscript_path="EdgeR_pipeline.R",
    counts_pattern="*_pseudobulk_counts.csv",
    metadata_suffix="_metadata.csv",
    sample_col="Sample",
    group_col="disease",
    donor_col=None,
    relevel=None,
    contrast_groups=None,
    output_dir="edger_results",
    fdr_thresh=0.05,
    lfc_thresh_abs=0.0,
):
    import glob
    import re
    import subprocess

    os.makedirs(output_dir, exist_ok=True)
    runs = []
    counts_files = sorted(glob.glob(os.path.join(folder, counts_pattern)))
    for cf in counts_files:
        celltype = os.path.basename(cf).replace("_pseudobulk_counts.csv", "")
        mf = cf.replace("_pseudobulk_counts.csv", metadata_suffix)
        if not os.path.exists(mf):
            runs.append({"celltype": celltype, "status": "metadata_missing"})
            continue
        cf_abs, mf_abs = os.path.abspath(cf), os.path.abspath(mf)
        cmd = ["Rscript", rscript_path, cf_abs, mf_abs, sample_col, group_col]
        if donor_col:
            cmd.append(donor_col)
        env = os.environ.copy()
        if relevel:
            env["RELEVEL"] = str(relevel)
        if contrast_groups:
            env["CONTRAST_GROUPS"] = ",".join(contrast_groups)
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=folder)
        contrast_label = "unknown"
        for line in result.stdout.splitlines():
            if line.startswith("CONTRAST_LABEL="):
                contrast_label = line.split("=", 1)[1].strip()
                break
        default_file = os.path.join(folder, f"{contrast_label}_edgeR_results.csv")
        target = os.path.join(output_dir, f"{contrast_label}_{celltype}_edgeR_results.csv")
        up_n = down_n = None
        if os.path.exists(default_file):
            os.replace(default_file, target)
            try:
                df = pd.read_csv(target)
                logfc_col, fdr_col = _detect_cols(df)
                sig = df[df[fdr_col] <= fdr_thresh]
                if lfc_thresh_abs == 0:
                    up_n = int((sig[logfc_col] >= 0).sum())
                    down_n = int((sig[logfc_col] < 0).sum())
                else:
                    up_n = int(((sig[logfc_col].abs() >= lfc_thresh_abs) & (sig[logfc_col] > 0)).sum())
                    down_n = int(((sig[logfc_col].abs() >= lfc_thresh_abs) & (sig[logfc_col] < 0)).sum())
            except Exception:
                pass
        runs.append({"celltype": celltype, "status": "ok", "up_n": up_n, "down_n": down_n, "result_file": target})
    return pd.DataFrame(runs)


def plot_deg_counts(summary_df, outfile="deg_counts.png", figsize=(8,6), palette=("firebrick", "royalblue")):
    """Create a dot plot summarising up/down regulated gene counts."""
    df = summary_df.dropna(subset=["up_n", "down_n"]).copy().sort_values("celltype")
    plot_df = df.melt(id_vars="celltype", value_vars=["up_n", "down_n"],
                      var_name="Direction", value_name="Count")
    plot_df["Direction"] = plot_df["Direction"].map({"up_n": "Upregulated", "down_n": "Downregulated"})
    sns.set_theme(style="white")
    plt.figure(figsize=figsize)
    ax = sns.scatterplot(data=plot_df, x="celltype", y="Count", hue="Direction",
                         palette=palette, s=120, edgecolor="black", linewidth=0.6)
    ax.set_xlabel("")
    ax.set_ylabel("Number of DE genes", fontsize=14, weight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(ax.get_yticks(), fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=12, title="")
    plt.tight_layout()
    plt.savefig(outfile)
    return outfile
