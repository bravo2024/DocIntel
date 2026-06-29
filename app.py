from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from src.data import make_synthetic, TARGET_NAME
from src.model import train_all_models, cross_validate
from src.visualizations import *
st.set_page_config(page_title="DocIntel | Hyperscience IDP", layout="wide", page_icon="\U0001f4c4")
with st.sidebar:
    st.header("\u2699 Config"); n=st.slider("Samples",2000,20000,10000,1000); tau=st.slider("Threshold",0.05,0.95,0.50,0.05)
    st.caption("Hyperscience | Intelligent Document Processing")
data=make_synthetic(n=n); b=train_all_models(data)
y_test=b["y_test"]; y_probas={n:b["results"][n]["y_proba"] for n in b["results"]}
best=max(b["results"],key=lambda n: b["results"][n]["metrics"].get("roc_auc",0))
c1,c2,c3,c4=st.columns(4)
c1.metric("Samples",f"{n:,}"); c2.metric("Valid Rate",f"{data['positive_rate']:.1%}")
c3.metric("Best AUC",f"{b['results'][best]['metrics']['roc_auc']:.4f}"); c4.metric("Best",best)
t1,t2,t3,t4=st.tabs(["\U0001f4ca Explorer","\U0001f52c Model Lab","\U0001f4d0 Document Analytics","\U0001f4ca Extraction Quality"])
with t1:
    st.dataframe(data["df"].head(50),use_container_width=True,height=200)
    fig,ax=plt.subplots(figsize=(5,3)); _style()
    ax.bar(["Invalid","Valid"],[1-data["positive_rate"],data["positive_rate"]],color=["#f43f5e","#22c55e"])
    for i,v in enumerate([1-data["positive_rate"],data["positive_rate"]]): ax.text(i,v+.01,f"{v:.1%}",ha="center",color="white")
    ax.set_title("Document Validity Distribution",color="white"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
with t2:
    rows=[{**{"Model":n},**{k:f"{v:.4f}" for k,v in r["metrics"].items() if k!="confusion_matrix"}} for n,r in b["results"].items()]
    st.dataframe(pd.DataFrame(rows).set_index("Model"),use_container_width=True)
    col_a,col_b=st.columns(2)
    with col_a: st.pyplot(plot_roc_curve(y_test,y_probas))
    with col_b: st.pyplot(plot_calibration_curve(y_test,y_probas))
    st.pyplot(plot_confusion_matrix(y_test,b["results"]["XGBoost"]["y_pred"],"XGBoost"))
    cv=cross_validate(data); cvr=[{"Model":n,"AUC":f"{s['roc_auc']['mean']:.4f}","\u00b1":f"\u00b1{s['roc_auc']['std']:.4f}"} for n,s in cv.items()]
    st.dataframe(pd.DataFrame(cvr).set_index("Model"),use_container_width=True)
with t3:
    st.subheader("Document Type Composition")
    doc_dist=data["df"]["document_type"].value_counts()
    fig,ax=plt.subplots(figsize=(6,4)); _style()
    colors=["#22d3ee","#f97316","#22c55e","#a78bfa","#fbbf24","#f43f5e"]
    ax.pie(doc_dist.values,labels=doc_dist.index,autopct="%1.1f%%",colors=colors,textprops={"color":"white"})
    ax.set_title("Document Types",color="white")
    st.pyplot(fig)
    col_a,col_b=st.columns(2)
    with col_a:
        ocr_valid=data["df"].groupby("document_type")["ocr_confidence"].mean()
        fig,ax=plt.subplots(figsize=(5,3)); _style()
        ax.bar(ocr_valid.index,ocr_valid.values,color="#22d3ee")
        ax.set_title("Avg OCR Confidence by Type",color="white"); ax.tick_params(axis="x",rotation=45,labelsize=8); ax.grid(True,alpha=.2)
        st.pyplot(fig)
    with col_b:
        pages_valid=data["df"].groupby("document_type")["page_count"].mean()
        fig,ax=plt.subplots(figsize=(5,3)); _style()
        ax.bar(pages_valid.index,pages_valid.values,color="#f97316")
        ax.set_title("Avg Page Count by Type",color="white"); ax.tick_params(axis="x",rotation=45,labelsize=8); ax.grid(True,alpha=.2)
        st.pyplot(fig)
with t4:
    st.subheader("Extraction Quality Metrics")
    st.latex(r"\text{F1}_{\text{extract}} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}")
    col_a,col_b=st.columns(2)
    with col_a:
        fig,ax=plt.subplots(figsize=(5,4)); _style()
        valid=data["df"][data["df"]["is_valid"]==1]["sharpness"]
        invalid=data["df"][data["df"]["is_valid"]==0]["sharpness"]
        ax.hist(valid,bins=30,alpha=0.5,color="#22c55e",label="Valid",density=True)
        ax.hist(invalid,bins=30,alpha=0.5,color="#f43f5e",label="Invalid",density=True)
        ax.set_title("Sharpness Distribution",color="white"); ax.legend(fontsize=8); ax.grid(True,alpha=.2)
        st.pyplot(fig)
    with col_b:
        fig,ax=plt.subplots(figsize=(5,4)); _style()
        valid=data["df"][data["df"]["is_valid"]==1]["edge_density"]
        invalid=data["df"][data["df"]["is_valid"]==0]["edge_density"]
        ax.hist(valid,bins=30,alpha=0.5,color="#22c55e",label="Valid",density=True)
        ax.hist(invalid,bins=30,alpha=0.5,color="#f43f5e",label="Invalid",density=True)
        ax.set_title("Edge Density Distribution",color="white"); ax.legend(fontsize=8); ax.grid(True,alpha=.2)
        st.pyplot(fig)
    st.subheader("OCR Confidence Breakdown")
    ocr_bins=pd.cut(data["df"]["ocr_confidence"],bins=[0,0.7,0.9,0.95,1],labels=["Low","Medium","High","Very High"])
    valid_by_ocr=data["df"].groupby(ocr_bins,observed=True)["is_valid"].mean()
    fig,ax=plt.subplots(figsize=(5,3)); _style()
    ax.bar(range(4),valid_by_ocr.values,color=["#f43f5e","#fbbf24","#22c55e","#22d3ee"])
    ax.set_xticks(range(4)); ax.set_xticklabels(valid_by_ocr.index); ax.set_title("Validity Rate by OCR Confidence",color="white"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
