# ======= MAIN CONTENT =======
if selected_section == "Overview":
    # Page header with title and description
    st.markdown("""
    <div class="dashboard-header">
        <h1>Aegis Alliance Dashboard</h1>
        <p>Trust & Transparency Layer: Real-time insights into system performance, privacy, and federation status</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Overview metrics with enhanced styling
    st.markdown("""
    <div class="metric-container">
        <div class="metric-row">
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Custom styled metrics
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon" style="background-color: rgba(62, 207, 142, 0.2);">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M8 16L10.879 13.121C11.3395 12.6605 12.0875 12.62 12.5979 13.0229L13.5 13.75L18 10" stroke="#3ECF8E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="#3ECF8E" stroke-width="2"/>
                </svg>
            </div>
            <div class="metric-content">
                <div class="metric-label">Transactions Processed</div>
                <div class="metric-value">12,456</div>
                <div class="metric-delta positive">+845 today</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon" style="background-color: rgba(110, 86, 207, 0.2);">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M21 21H4.6C4.03995 21 3.75992 21 3.54601 20.891C3.35785 20.7951 3.20487 20.6422 3.10899 20.454C3 20.2401 3 19.9601 3 19.4V3" stroke="#6E56CF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M7 14.5L11.2929 10.2071C11.6834 9.81658 12.3166 9.81658 12.7071 10.2071L14.5 12L17.5 9" stroke="#6E56CF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
            <div class="metric-content">
                <div class="metric-label">Fraud Detection Rate</div>
                <div class="metric-value">89.2%</div>
                <div class="metric-delta positive">+1.3% vs. last week</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon" style="background-color: rgba(236, 72, 153, 0.2);">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="#EC4899" stroke-width="2"/>
                    <path d="M12 8V12L15 15" stroke="#EC4899" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
            <div class="metric-content">
                <div class="metric-label">Privacy Budget (ε)</div>
                <div class="metric-value">{epsilon:.1f}</div>
                <div class="metric-delta neutral">Currently active</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon" style="background-color: rgba(37, 99, 235, 0.2);">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M2 12H22" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M12 2C14.5013 4.73835 15.9228 8.29203 16 12C15.9228 15.708 14.5013 19.2616 12 22C9.49872 19.2616 8.07725 15.708 8 12C8.07725 8.29203 9.49872 4.73835 12 2Z" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
            <div class="metric-content">
                <div class="metric-label">ZK Proof Verification</div>
                <div class="metric-value">99.7%</div>
                <div class="metric-delta positive">+0.2% vs. last week</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Overview charts with enhanced card styling
    st.markdown("""
    <div class="section-title">
        <h2>Performance vs. Privacy Trade-off</h2>
        <div class="badge badge-primary">Real-time Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate dummy data for demonstration
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    accuracies = [0.82, 0.86, 0.89, 0.92, 0.94, 0.95]
    privacy_loss = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95]
    
    # Create a two-panel chart with enhanced styling
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <h3>Accuracy vs. Privacy Budget</h3>
            </div>
        """, unsafe_allow_html=True)
        
        chart_data = pd.DataFrame({
            'Privacy Budget (ε)': epsilons,
            'Model Accuracy': accuracies
        })
        
        chart = alt.Chart(chart_data).mark_line(
            point=True,
            color='#6E56CF',
            strokeWidth=3
        ).encode(
            x=alt.X('Privacy Budget (ε)', title='Privacy Budget (ε)'),
            y=alt.Y('Model Accuracy', scale=alt.Scale(domain=[0.80, 1.0]), title='Model Accuracy'),
            tooltip=['Privacy Budget (ε)', 'Model Accuracy']
        ).properties(
            height=250
        )
        
        st.altair_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <h3>Privacy Loss vs. Privacy Budget</h3>
            </div>
        """, unsafe_allow_html=True)
        
        chart_data = pd.DataFrame({
            'Privacy Budget (ε)': epsilons,
            'Privacy Loss': privacy_loss
        })
        
        chart = alt.Chart(chart_data).mark_line(
            point=True,
            color='#EC4899',
            strokeWidth=3
        ).encode(
            x=alt.X('Privacy Budget (ε)', title='Privacy Budget (ε)'),
            y=alt.Y('Privacy Loss', scale=alt.Scale(domain=[0, 1.0]), title='Privacy Loss'),
            tooltip=['Privacy Budget (ε)', 'Privacy Loss']
        ).properties(
            height=250
        )
        
        st.altair_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # System architecture diagram with enhanced styling
    st.markdown("""
    <div class="section-title">
        <h2>System Architecture</h2>
        <div class="badge badge-secondary">System Design</div>
    </div>
    <div class="architecture-container">
    """, unsafe_allow_html=True)
    
    # Placeholder for architecture diagram
    architecture = """
    digraph G {
        rankdir=LR;
        bgcolor="transparent";
        node [shape=box, style=filled, color="#2A3140", fontcolor="#E1E7EF", fontname="Arial"];
        edge [color="#6E56CF", penwidth=1.5];
        
        Bank1 [label="Bank A Data"];
        Bank2 [label="Bank B Data"];
        Bank3 [label="Bank C Data"];
        
        Oracle [label="Oracle Engine\n(XGBoost)", color="#1F2937"];
        Adaptive [label="Adaptive Intervention\n(Policy Engine)", color="#1F2937"];
        Federated [label="Zero-Knowledge Fabric\n(Federated Learning)", color="#1F2937"];
        Trust [label="Trust & Transparency\n(Audit & Verification)", color="#1F2937"];
        
        {Bank1, Bank2, Bank3} -> Federated;
        Federated -> Oracle;
        Oracle -> Adaptive;
        {Oracle, Adaptive, Federated} -> Trust;
    }
    """
    
    st.graphviz_chart(architecture)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Recent activity with enhanced styling
    st.markdown("""
    <div class="section-title">
        <h2>Recent Activity</h2>
        <div class="badge badge-info">Live Updates</div>
    </div>
    <div class="activity-timeline">
    """, unsafe_allow_html=True)
    
    # Dummy activity data
    activities = [
        {"time": "2 mins ago", "activity": "Bank A completed model training", "type": "training", "icon": "🔄"},
        {"time": "15 mins ago", "activity": "New transactions processed: 145", "type": "transaction", "icon": "💼"},
        {"time": "1 hour ago", "activity": "Privacy budget updated to ε=1.2", "type": "config", "icon": "⚙️"},
        {"time": "3 hours ago", "activity": "Bank C joined the federation", "type": "federation", "icon": "🏦"},
        {"time": "5 hours ago", "activity": "System audit completed", "type": "audit", "icon": "📋"}
    ]
    
    for activity in activities:
        # Different formatting based on activity type
        icon_colors = {
            "training": "#6E56CF",
            "transaction": "#3ECF8E",
            "config": "#F59E0B",
            "federation": "#2563EB",
            "audit": "#EC4899"
        }
        
        st.markdown(f"""
        <div class="activity-item">
            <div class="activity-icon" style="background-color: {icon_colors[activity['type']]};">
                {activity['icon']}
            </div>
            <div class="activity-content">
                <div class="activity-text">{activity['activity']}</div>
                <div class="activity-time">{activity['time']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # System status with enhanced styling
    st.markdown("""
    <div class="section-title">
        <h2>System Status</h2>
        <div class="badge badge-success">Operational</div>
    </div>
    <div class="status-card">
        <div class="status-icon pulse">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="#3ECF8E" stroke-width="2"/>
                <path d="M12 8V12" stroke="#3ECF8E" stroke-width="2" stroke-linecap="round"/>
                <path d="M12 16H12.01" stroke="#3ECF8E" stroke-width="2" stroke-linecap="round"/>
            </svg>
        </div>
        <div class="status-content">
            <p>The Aegis Alliance is currently using a privacy budget of <span class="highlight">ε = {epsilon:.1f}</span>. The system is fully operational with all 3 banks participating in the federation. All privacy guarantees are being maintained while achieving optimal model performance.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
