from __future__ import annotations
import numpy as np; import pandas as pd
FEATURE_NAMES = ["ocr_confidence","page_count","document_area","text_density","edge_density","entropy","aspect_ratio","sharpness","brightness","contrast","num_tables","num_signatures","has_barcode","document_type"]
CATEGORICAL_FEATURES = ["document_type"]
NUMERICAL_FEATURES = ["ocr_confidence","page_count","document_area","text_density","edge_density","entropy","aspect_ratio","sharpness","brightness","contrast","num_tables","num_signatures","has_barcode"]
TARGET_NAME = "is_valid"
def make_synthetic(n=10000,seed=42):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        "ocr_confidence": rng.beta(8,2,size=n).round(3),
        "page_count": rng.poisson(lam=3,size=n).clip(1,20),
        "document_area": rng.lognormal(mean=6.5,sigma=0.3,size=n).clip(100,5000).astype(int),
        "text_density": rng.beta(5,3,size=n).round(3),
        "edge_density": rng.beta(3,4,size=n).round(3),
        "entropy": rng.uniform(1,8,size=n).round(2),
        "aspect_ratio": rng.uniform(0.5,2.5,size=n).round(2),
        "sharpness": rng.uniform(0,100,size=n).round(1),
        "brightness": rng.normal(128,30,size=n).clip(0,255).astype(int),
        "contrast": rng.uniform(0,100,size=n).round(1),
        "num_tables": rng.poisson(lam=1.5,size=n).clip(0,10).astype(int),
        "num_signatures": rng.poisson(lam=0.8,size=n).clip(0,5).astype(int),
        "has_barcode": rng.choice([0,1],size=n,p=[0.6,0.4]),
        "document_type": rng.choice(["invoice","receipt","contract","id_card","bank_statement","form"],size=n,p=[0.25,0.20,0.15,0.15,0.15,0.10]),
    })
    ocr=df["ocr_confidence"]; edge=df["edge_density"]; ent=np.clip(df["entropy"]/8,0,1)
    sharp=df["sharpness"]/100; bright=np.clip((df["brightness"]-128)/128,-1,1)
    tables=np.clip(df["num_tables"]/10,0,1); sigs=np.clip(df["num_signatures"]/5,0,1)
    barcode=df["has_barcode"]; doc_type=df["document_type"].map({"invoice":0,"receipt":0.2,"contract":0.4,"id_card":0.6,"bank_statement":0.8,"form":1}).values
    log_odds = 3.0 + 1.5*ocr - 0.3*edge - 0.2*ent + 0.3*sharp + 0.1*bright + 0.2*tables + 0.3*sigs + 0.2*barcode + 0.1*doc_type + rng.normal(0,0.5,size=n)
    prob=1/(1+np.exp(-log_odds)); y=(prob>np.percentile(prob,75)).astype(np.float64)
    return {"X":df,"y":y,"features":FEATURE_NAMES,"df":df.assign(is_valid=y),"categorical_features":CATEGORICAL_FEATURES,"numerical_features":NUMERICAL_FEATURES,"n_samples":n,"n_features":len(FEATURE_NAMES),"positive_rate":y.mean()}
