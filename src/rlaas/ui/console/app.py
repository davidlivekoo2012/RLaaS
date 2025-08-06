"""
RLaaS Web Console - Streamlit Application.

This is the main web console for the RLaaS platform providing:
- Dashboard and monitoring
- Optimization configuration and results
- Model management interface
- Training job management
- System administration
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import httpx
from datetime import datetime, timedelta
import json
import time

# Page configuration
st.set_page_config(
    page_title="RLaaS Console",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-running {
        color: #28a745;
    }
    .status-failed {
        color: #dc3545;
    }
    .status-pending {
        color: #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = st.secrets.get("api_base_url", "http://localhost:8000")

class RLaaSAPI:
    """API client for RLaaS platform."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    async def get_health(self):
        """Get system health status."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/health")
                return response.json()
            except Exception as e:
                return {"status": "error", "error": str(e)}
    
    async def list_optimizations(self):
        """List optimization jobs."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/api/v1/optimization/jobs")
                return response.json()
            except Exception as e:
                return []
    
    async def start_optimization(self, config):
        """Start new optimization."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/optimization/optimize",
                    json=config
                )
                return response.json()
            except Exception as e:
                return {"error": str(e)}
    
    async def get_optimization_status(self, optimization_id):
        """Get optimization status."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/api/v1/optimization/optimize/{optimization_id}"
                )
                return response.json()
            except Exception as e:
                return {"error": str(e)}
    
    async def list_models(self):
        """List deployed models."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/api/v1/inference/models")
                return response.json()
            except Exception as e:
                return []
    
    async def list_training_jobs(self):
        """List training jobs."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/api/v1/training/jobs")
                return response.json()
            except Exception as e:
                return []

# Initialize API client
api = RLaaSAPI(API_BASE_URL)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 RLaaS Console</h1>', unsafe_allow_html=True)
    st.markdown("**Reinforcement Learning as a Service Platform**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        [
            "Dashboard",
            "Optimization",
            "Training",
            "Models",
            "Data",
            "Monitoring",
            "Settings"
        ]
    )
    
    # Route to selected page
    if page == "Dashboard":
        show_dashboard()
    elif page == "Optimization":
        show_optimization()
    elif page == "Training":
        show_training()
    elif page == "Models":
        show_models()
    elif page == "Data":
        show_data()
    elif page == "Monitoring":
        show_monitoring()
    elif page == "Settings":
        show_settings()

def show_dashboard():
    """Show main dashboard."""
    
    st.header("📊 System Dashboard")
    
    # System health check
    with st.spinner("Checking system health..."):
        health_data = asyncio.run(api.get_health())
    
    # Health status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = health_data.get("status", "unknown")
        color = "🟢" if status == "healthy" else "🔴"
        st.metric("System Status", f"{color} {status.title()}")
    
    with col2:
        version = health_data.get("version", "unknown")
        st.metric("Version", version)
    
    with col3:
        uptime = health_data.get("uptime", 0)
        uptime_str = f"{uptime/3600:.1f}h" if uptime > 3600 else f"{uptime:.0f}s"
        st.metric("Uptime", uptime_str)
    
    with col4:
        env = health_data.get("environment", "unknown")
        st.metric("Environment", env.title())
    
    # Component health
    st.subheader("Component Health")
    
    if "checks" in health_data:
        health_df = pd.DataFrame([
            {"Component": comp, "Status": info["status"], "Message": info.get("message", "")}
            for comp, info in health_data["checks"].items()
        ])
        
        # Color code status
        def color_status(val):
            if val == "healthy":
                return "background-color: #d4edda"
            elif val == "unhealthy":
                return "background-color: #f8d7da"
            else:
                return "background-color: #fff3cd"
        
        st.dataframe(
            health_df.style.applymap(color_status, subset=["Status"]),
            use_container_width=True
        )
    
    # Recent activity
    st.subheader("Recent Activity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Recent Optimizations**")
        optimizations = asyncio.run(api.list_optimizations())
        if optimizations:
            opt_df = pd.DataFrame(optimizations[:5])  # Last 5
            st.dataframe(opt_df[["job_id", "status", "algorithm"]], use_container_width=True)
        else:
            st.info("No recent optimizations")
    
    with col2:
        st.write("**Recent Training Jobs**")
        training_jobs = asyncio.run(api.list_training_jobs())
        if training_jobs:
            train_df = pd.DataFrame(training_jobs[:5])  # Last 5
            st.dataframe(train_df[["job_id", "status", "algorithm"]], use_container_width=True)
        else:
            st.info("No recent training jobs")

def show_optimization():
    """Show optimization interface."""
    
    st.header("🎯 Multi-Objective Optimization")
    
    tab1, tab2, tab3 = st.tabs(["New Optimization", "Running Jobs", "Results"])
    
    with tab1:
        st.subheader("Start New Optimization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            problem_type = st.selectbox(
                "Problem Type",
                ["5g", "recommendation"],
                help="Select the type of optimization problem"
            )
            
            algorithm = st.selectbox(
                "Algorithm",
                ["nsga3", "moead"],
                help="Multi-objective optimization algorithm"
            )
            
            mode = st.selectbox(
                "Mode",
                ["normal", "emergency", "revenue_focused", "user_experience"],
                help="Optimization mode affects objective weights"
            )
        
        with col2:
            population_size = st.slider("Population Size", 50, 500, 100)
            generations = st.slider("Generations", 100, 2000, 500)
            timeout = st.number_input("Timeout (seconds)", min_value=60, value=3600)
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            weights_json = st.text_area(
                "Custom Weights (JSON)",
                placeholder='{"latency": 0.4, "throughput": 0.3, "energy": 0.3}',
                help="Custom objective weights as JSON"
            )
            
            constraints_json = st.text_area(
                "Constraints (JSON)",
                placeholder='{"max_power": 10.0, "min_throughput": 100.0}',
                help="Problem constraints as JSON"
            )
        
        if st.button("Start Optimization", type="primary"):
            config = {
                "problem_type": problem_type,
                "algorithm": algorithm,
                "mode": mode,
                "population_size": population_size,
                "generations": generations,
                "timeout": timeout
            }
            
            # Add weights if provided
            if weights_json.strip():
                try:
                    config["weights"] = json.loads(weights_json)
                except json.JSONDecodeError:
                    st.error("Invalid JSON format for weights")
                    return
            
            # Add constraints if provided
            if constraints_json.strip():
                try:
                    config["constraints"] = json.loads(constraints_json)
                except json.JSONDecodeError:
                    st.error("Invalid JSON format for constraints")
                    return
            
            with st.spinner("Starting optimization..."):
                result = asyncio.run(api.start_optimization(config))
            
            if "error" in result:
                st.error(f"Failed to start optimization: {result['error']}")
            else:
                st.success(f"Optimization started! ID: {result['optimization_id']}")
                st.info("Check the 'Running Jobs' tab to monitor progress")
    
    with tab2:
        st.subheader("Running Optimization Jobs")
        
        if st.button("Refresh"):
            st.rerun()
        
        optimizations = asyncio.run(api.list_optimizations())
        
        if optimizations:
            for opt in optimizations:
                if opt["status"] in ["running", "pending"]:
                    with st.container():
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        
                        with col1:
                            st.write(f"**{opt['job_id']}**")
                            st.write(f"Algorithm: {opt['algorithm']}")
                        
                        with col2:
                            status_class = f"status-{opt['status']}"
                            st.markdown(f'<span class="{status_class}">{opt["status"].title()}</span>', 
                                      unsafe_allow_html=True)
                        
                        with col3:
                            if "progress" in opt:
                                st.progress(opt["progress"])
                        
                        with col4:
                            if st.button(f"Cancel", key=f"cancel_{opt['job_id']}"):
                                # Cancel optimization logic here
                                st.info("Cancellation requested")
                        
                        st.divider()
        else:
            st.info("No running optimizations")
    
    with tab3:
        st.subheader("Optimization Results")
        
        # Mock results visualization
        if st.button("Load Sample Results"):
            # Generate sample Pareto frontier
            n_points = 50
            
            # Sample 3-objective optimization results
            latency = np.random.uniform(1, 10, n_points)
            throughput = np.random.uniform(100, 1000, n_points)
            energy = np.random.uniform(0.1, 1.0, n_points)
            
            # Create 3D scatter plot
            fig = go.Figure(data=go.Scatter3d(
                x=latency,
                y=throughput,
                z=energy,
                mode='markers',
                marker=dict(
                    size=8,
                    color=energy,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Energy")
                ),
                text=[f"Point {i+1}" for i in range(n_points)],
                hovertemplate="<b>%{text}</b><br>" +
                            "Latency: %{x:.2f}ms<br>" +
                            "Throughput: %{y:.2f}Mbps<br>" +
                            "Energy: %{z:.2f}<br>" +
                            "<extra></extra>"
            ))
            
            fig.update_layout(
                title="Pareto Frontier - 3D Visualization",
                scene=dict(
                    xaxis_title="Latency (ms)",
                    yaxis_title="Throughput (Mbps)",
                    zaxis_title="Energy Consumption"
                ),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Best solution details
            st.subheader("Best Solution")
            best_idx = np.argmin(latency + (1000 - throughput)/1000 + energy)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Latency", f"{latency[best_idx]:.2f} ms")
            with col2:
                st.metric("Throughput", f"{throughput[best_idx]:.2f} Mbps")
            with col3:
                st.metric("Energy", f"{energy[best_idx]:.3f}")

def show_training():
    """Show training interface."""
    
    st.header("🎓 Training Management")
    
    tab1, tab2 = st.tabs(["Training Jobs", "New Training"])
    
    with tab1:
        st.subheader("Training Jobs")
        
        training_jobs = asyncio.run(api.list_training_jobs())
        
        if training_jobs:
            df = pd.DataFrame(training_jobs)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No training jobs found")
    
    with tab2:
        st.subheader("Start New Training")
        
        training_type = st.selectbox(
            "Training Type",
            ["reinforcement_learning", "deep_learning", "optimization"]
        )
        
        algorithm = st.selectbox(
            "Algorithm",
            ["sac", "ppo", "dqn"] if training_type == "reinforcement_learning" else ["adam", "sgd"]
        )
        
        st.info("Training job creation interface - Implementation in progress")

def show_models():
    """Show model management interface."""
    
    st.header("🤖 Model Management")
    
    tab1, tab2 = st.tabs(["Deployed Models", "Model Registry"])
    
    with tab1:
        st.subheader("Deployed Models")
        
        models = asyncio.run(api.list_models())
        
        if models:
            for model in models:
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**{model['name']}**")
                        st.write(f"Version: {model['version']}")
                    
                    with col2:
                        status_color = "🟢" if model['status'] == 'running' else "🔴"
                        st.write(f"{status_color} {model['status'].title()}")
                    
                    with col3:
                        if 'metrics' in model:
                            st.metric("RPS", f"{model['metrics'].get('requests_per_second', 0):.1f}")
                    
                    with col4:
                        if st.button(f"Details", key=f"model_{model['model_id']}"):
                            st.json(model)
                    
                    st.divider()
        else:
            st.info("No deployed models found")
    
    with tab2:
        st.subheader("Model Registry")
        st.info("Model registry interface - Implementation in progress")

def show_data():
    """Show data management interface."""
    
    st.header("📊 Data Management")
    
    tab1, tab2, tab3 = st.tabs(["Datasets", "Streaming", "Features"])
    
    with tab1:
        st.subheader("Data Lake")
        st.info("Data lake interface - Implementation in progress")
    
    with tab2:
        st.subheader("Stream Processing")
        st.info("Stream processing interface - Implementation in progress")
    
    with tab3:
        st.subheader("Feature Store")
        st.info("Feature store interface - Implementation in progress")

def show_monitoring():
    """Show monitoring interface."""
    
    st.header("📈 System Monitoring")
    
    # Generate sample metrics
    if st.button("Load Sample Metrics"):
        # Time series data
        dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='H')
        
        # CPU usage
        cpu_usage = 20 + 30 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24) + np.random.normal(0, 5, len(dates))
        cpu_usage = np.clip(cpu_usage, 0, 100)
        
        # Memory usage
        memory_usage = 40 + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24 + 1) + np.random.normal(0, 3, len(dates))
        memory_usage = np.clip(memory_usage, 0, 100)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Usage', 'Memory Usage', 'Request Rate', 'Response Time'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # CPU usage
        fig.add_trace(
            go.Scatter(x=dates, y=cpu_usage, name="CPU %", line=dict(color='blue')),
            row=1, col=1
        )
        
        # Memory usage
        fig.add_trace(
            go.Scatter(x=dates, y=memory_usage, name="Memory %", line=dict(color='green')),
            row=1, col=2
        )
        
        # Request rate
        request_rate = 100 + 50 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24) + np.random.normal(0, 10, len(dates))
        request_rate = np.clip(request_rate, 0, None)
        
        fig.add_trace(
            go.Scatter(x=dates, y=request_rate, name="Requests/sec", line=dict(color='orange')),
            row=2, col=1
        )
        
        # Response time
        response_time = 50 + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24 + 2) + np.random.normal(0, 5, len(dates))
        response_time = np.clip(response_time, 0, None)
        
        fig.add_trace(
            go.Scatter(x=dates, y=response_time, name="Response Time (ms)", line=dict(color='red')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="System Metrics")
        st.plotly_chart(fig, use_container_width=True)

def show_settings():
    """Show settings interface."""
    
    st.header("⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["System", "API", "User"])
    
    with tab1:
        st.subheader("System Configuration")
        
        st.text_input("API Base URL", value=API_BASE_URL)
        st.selectbox("Theme", ["Light", "Dark"])
        st.slider("Refresh Interval (seconds)", 5, 60, 30)
    
    with tab2:
        st.subheader("API Configuration")
        
        st.text_input("API Key", type="password")
        st.number_input("Request Timeout", value=30)
        st.checkbox("Enable API Caching")
    
    with tab3:
        st.subheader("User Preferences")
        
        st.text_input("Username")
        st.selectbox("Language", ["English", "Chinese"])
        st.multiselect("Notifications", ["Email", "SMS", "Push"])

if __name__ == "__main__":
    main()
