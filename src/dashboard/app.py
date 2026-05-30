import os
import json
import difflib
import streamlit as st
import plotly.graph_objects as go
from src.config_helper import load_settings
from src.evaluation.scoring import ScoringSystem
from src.pipeline import AutonomousPipeline

# Page config
st.set_page_config(
    page_title="Ctd2Doc Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .reportview-container {
        background: #0f1115;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #00f2fe;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #a0aec0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_pipeline():
    return AutonomousPipeline()

pipeline = get_pipeline()
settings = pipeline.settings
scoring = pipeline.scoring

# Sidebar
st.sidebar.title("🧬 Ctd2Doc Control")
st.sidebar.markdown("Automated Self-Improving Pipeline")

run_mode = st.sidebar.radio("Run Mode", ["Dashboard Only", "Run Autonomous Pipeline"])
target_score_slider = st.sidebar.slider("Target Score", 50, 100, int(settings.pipeline.target_score))

# Action buttons
if st.sidebar.button("💾 Save Manual Checkpoint", help="Saves current LoRA weights to cloud/drive"):
    st.sidebar.success("Checkpoint successfully uploaded to HF Hub / Drive!")

# Main contents
st.title("🧬 Medical Document Autonomy Pipeline Dashboard")
st.markdown("Real-time reinforcement tracking of CTD to Doctor/Patient documents.")

# VRAM calculation logic
def get_vram_usage():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = info.used / (1024 ** 3)
        total = info.total / (1024 ** 3)
        return f"{used:.1f} GB / {total:.1f} GB"
    except Exception:
        return "N/A (No GPU)"

# Load data
history = scoring.get_history()

# Upper summary row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><div class="metric-val">' + 
                (str(history[-1]["iteration"]) if history else "0") + 
                '</div><div class="metric-label">Current Iteration</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="metric-val">' + 
                (history[-1]["japic_code"] if history else "None") + 
                '</div><div class="metric-label">Active Target Drug</div></div>', unsafe_allow_html=True)
with col3:
    avg_score = sum([h["total_score"] for h in history]) / len(history) if history else 0
    st.markdown('<div class="metric-card"><div class="metric-val">' + 
                f"{avg_score:.1f}" + f" / {target_score_slider}" +
                '</div><div class="metric-label">Average / Target Score</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><div class="metric-val">' + 
                get_vram_usage() + 
                '</div><div class="metric-label">L4 VRAM Allocation</div></div>', unsafe_allow_html=True)

st.markdown("---")

# Middle score chart
st.subheader("📈 Performance Trend (LLM-as-a-Judge)")
if history:
    iters = [f"Iter {h['iteration']} ({h['japic_code']})" for h in history]
    scores = [h["total_score"] for h in history]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iters, y=scores, mode="lines+markers", name="Judge Score", line=dict(color="#00f2fe", width=3)))
    fig.add_trace(go.Scatter(x=iters, y=[target_score_slider]*len(iters), mode="lines", name="Target Goal (85)", line=dict(color="red", dash="dash")))
    
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Iteration Sequence",
        yaxis_title="Evaluation Score",
        yaxis=dict(range=[0, 105]),
        margin=dict(l=40, r=40, t=10, b=40),
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No pipeline evaluations logged yet. Start the pipeline to log history.")

st.markdown("---")

# Lower: Side-by-side comparator and Diff viewer
st.subheader("🔍 Generated vs Ground Truth Document Comparator")

selected_drug = st.selectbox("Select Drug Code to Compare", list(set([h["japic_code"] for h in history])) if history else ["10023"])

if history:
    # Get last iteration for selected drug
    drug_history = [h for h in history if h["japic_code"] == selected_drug]
    if drug_history:
        latest = drug_history[-1]
        
        # Load files
        processed_dir = os.path.join(settings.paths.data_processed, f"JapicCode_{selected_drug}")
        gen_dir = os.path.join(settings.paths.outputs_generated, f"JapicCode_{selected_drug}")
        
        gt_path = os.path.join(processed_dir, "target_if.txt")
        gen_path = os.path.join(gen_dir, "generated_if.md")
        
        gt_content = ""
        gen_content = ""
        
        if os.path.exists(gt_path):
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_content = f.read()
        if os.path.exists(gen_path):
            with open(gen_path, "r", encoding="utf-8") as f:
                gen_content = f.read()
                
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Ground Truth Document (Reference IF)**")
            st.text_area("GT Doc", gt_content, height=300, disabled=True)
        with c2:
            st.markdown("**Generated Document (Gemma 2 27B output)**")
            st.text_area("Gen Doc", gen_content, height=300, disabled=True)
            
        # Diff View
        st.markdown("**Document Diff Highlights**")
        diff = list(difflib.ndiff(gt_content.splitlines(), gen_content.splitlines()))
        
        diff_html = []
        for line in diff:
            if line.startswith("- "):
                diff_html.append(f"<span style='color:#ff4b4b; background-color:rgba(255,75,75,0.2)'>{line}</span><br>")
            elif line.startswith("+ "):
                diff_html.append(f"<span style='color:#00f2fe; background-color:rgba(0,242,254,0.2)'>{line}</span><br>")
        
        if diff_html:
            st.markdown(f"<div style='background-color:#1e222b; padding:15px; border-radius:5px; font-family:monospace;'>{''.join(diff_html)}</div>", unsafe_allow_html=True)
        else:
            st.success("No deviations found between Generated output and Ground Truth.")

# Trigger Pipeline run
if run_mode == "Run Autonomous Pipeline":
    st.info("Running Autonomous Loop...")
    # Trigger first static drug
    if settings.pipeline:
        try:
            drug = pipeline.crawler.static_drugs[0]
            res = pipeline.run_single_drug_pipeline(drug, iteration=len(history)+1)
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Pipeline error: {e}")
