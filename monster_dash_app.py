import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import os
import pandas as pd
import random

# --- [1. معالجة التوافقية (Compatibility Shield) ---
# محاولة استيراد المكتبات الثقيلة، وإذا فشلت، التطبيق سيستمر بالعمل عبر "Safe Mode"
try:
    import numpy as np
    import tensorflow as tf
    HAS_AI_LIBS = True
except ImportError:
    HAS_AI_LIBS = False
    # تعريف بديل بسيط لـ numpy لضمان عمل العمليات الرياضية الأساسية
    class MockNP:
        def abs(self, x): return [abs(i) for i in x] if isinstance(x, list) else abs(x)
        def random_normal(self, loc, scale, size): return [random.gauss(loc, scale) for _ in range(size)]
        def linspace(self, start, stop, num): return [start + (stop-start)*i/(num-1) for i in range(num)]
    np = MockNP()

# --- [2. استيراد المحرك المحدث - Phase 12] ---
import monster_core as mc

# استيراد الأدوات الإضافية مع معالجة الأخطاء
try:
    from src.sovereign_brain import build_sovereign_model
    from src.sensor_integrity import SensorShield
    from wavelet_surgeon import apply_wavelet_denoising, extract_echo_features
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False

# --- [3. تهيئة البيانات والأسطول السوري الشامل] ---
def generate_syrian_fleet_assets():
    """توليد 50 موقعاً استراتيجياً في سوريا لتمثيل أسطول الطاقة الوطني"""
    main_sites = {
        'Site': ['Homs Refinery', 'Baniyas Terminal', 'Deir ez-Zor Field', 'Damascus Hub', 'Al-Shaer Field', 'Tiyas Station'],
        'lat': [34.7324, 35.1856, 35.3359, 33.5138, 34.8500, 34.5200],
        'lon': [36.7137, 35.9479, 40.1401, 36.2765, 38.1500, 37.6500]
    }
    df = pd.DataFrame(main_sites)
    
    # إضافة مواقع عشوائية لتكملة الـ 50
    random.seed(42)
    extra_data = []
    for i in range(44):
        extra_data.append({
            'Site': f"Energy Asset-{i+7:02d}",
            'lat': random.uniform(33.0, 37.0),
            'lon': random.uniform(36.0, 41.5)
        })
    
    return pd.concat([df, pd.DataFrame(extra_data)], ignore_index=True)

fleet_df = generate_syrian_fleet_assets()

# محاولة تحميل النموذج إذا توفرت المكتبات
monster_brain = None
shield = None
if HAS_AI_LIBS and TOOLS_AVAILABLE:
    try:
        monster_brain = build_sovereign_model()
        shield = SensorShield()
    except: pass

# --- [4. تهيئة التطبيق] ---
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.CYBORG], 
                update_title="🔄 Synchronizing...")
app.title = "Acoustic Monster - Sovereign Command"

# --- [5. بناء الواجهة (The Sovereign UI)] ---
app.layout = dbc.Container([
    # نافذة الدرون التفاعلية
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(id="drone-title", className="text-info")),
        dbc.ModalBody(id="drone-hud-content"),
        dbc.ModalFooter(dbc.Button("Emergency Return", id="close-drone", color="outline-danger"))
    ], id="drone-modal", is_open=False, size="lg", backdrop="static"),

    # شريط الحالة العلوي
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("🚀 ACOUSTIC MONSTER", className="display-4 text-primary mb-0 fw-bold"),
                html.P(f"Phase 12: Syrian Fleet Orchestration | Status: {'AI-ENABLED' if HAS_AI_LIBS else 'SAFE-MODE'}", 
                       className="text-muted")
            ], className="py-3 border-bottom border-primary mb-4")
        ], width=12)
    ]),

    dbc.Row([
        # لوحة التحكم الجانبية
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🛠️ Asset Environment Simulation"),
                dbc.CardBody([
                    html.Label("Target Pressure (Bar)"),
                    dcc.Slider(id='p-slider', min=20, max=100, value=55, marks={20:'20', 60:'60', 100:'100'}),
                    html.Label("Oil Temp (°C)"),
                    dcc.Slider(id='t-slider', min=10, max=60, value=35, marks={10:'10', 35:'35', 60:'60'}),
                    html.Label("Viscosity Index"),
                    dcc.Slider(id='v-slider', min=5, max=500, value=150),
                    html.Hr(),
                    dbc.Checklist(options=[{"label": "🛡️ SensorShield Boost", "value": "stress"}], id="stress-toggle", switch=True),
                    html.Hr(),
                    dbc.Button("⚡ Run AI Optimization", id="opt-btn", color="primary", className="w-100 mb-2"),
                    dbc.Button("🛰️ Launch Drone Recon", id="drone-btn", color="danger", className="w-100"),
                    html.Div(id="opt-output", className="mt-3 p-2 bg-dark rounded border border-info small text-center")
                ])
            ], className="shadow-lg")
        ], width=3),

        # منطقة البيانات الكبرى
        dbc.Col([
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader("Fleet Blockage Avg"), dbc.CardBody(html.H2(id="block-val", className="text-warning"))]), width=3),
                dbc.Col(dbc.Card([dbc.CardHeader("Confidence Level"), dbc.CardBody(html.H2(id="uncert-val", className="text-info"))]), width=3),
                dbc.Col(dbc.Card([dbc.CardHeader("Economic Risk (ABS)"), dbc.CardBody(html.H2(id="cost-val", className="text-danger"))]), width=3),
                dbc.Col(dbc.Card([dbc.CardHeader("Sensor Integrity"), dbc.CardBody(html.Div(id="health-list", className="small"))]), width=3),
            ], className="mb-4 text-center"),

            dbc.Alert(id="decision-alert", className="mb-4 text-center h4 p-3 border-start border-5"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔬 Signal Wavelet Surgeon"),
                        dbc.CardBody([dcc.Graph(id='wavelet-main-graph', config={'displayModeBar': False})])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🌐 Syria Sovereign Asset Fleet (50 Locations)"),
                        dbc.CardBody([dcc.Graph(id='syria-map', config={'displayModeBar': False})])
                    ])
                ], width=6),
            ])
        ], width=9)
    ])
], fluid=True, className="p-4")

# --- [6. منطق العمليات المطور (The Nervous System)] ---

@app.callback(
    [Output('block-val', 'children'), Output('uncert-val', 'children'),
     Output('cost-val', 'children'), Output('decision-alert', 'children'),
     Output('decision-alert', 'color'), Output('health-list', 'children'),
     Output('wavelet-main-graph', 'figure'), Output('syria-map', 'figure')],
    [Input('p-slider', 'value'), Input('t-slider', 'value'), 
     Input('v-slider', 'value'), Input('stress-toggle', 'value')]
)
def update_fleet_intelligence(p, t, v, stress):
    is_stress = True if stress and "stress" in stress else False
    
    # 1. التنبؤ وتصحيح القيم المالية
    final_val, health_reports, uncert = mc.get_multi_sensor_prediction(p, t, v, is_stress, monster_brain, shield)
    risk_cost, _, _ = mc.calculate_financial_impact(final_val, p)
    decision, color = mc.autonomous_manager(final_val, risk_cost)
    
    # 2. توليد بيانات الأسطول (Simulated Distribution)
    fleet_severities = [min(1.0, max(0.0, final_val + random.uniform(-0.15, 0.15))) for _ in range(50)]
    
    # 3. معالجة الـ Wavelet (محاكاة بصرية في حال غياب المكتبات)
    wavelet_fig = go.Figure()
    if HAS_AI_LIBS and TOOLS_AVAILABLE:
        try:
            t_axis = np.linspace(0, 0.1, 500)
            raw_sig = np.sin(2 * np.pi * 10 * t_axis) + np.random.normal(0, (0.4 if is_stress else 0.05), 500)
            denoised = apply_wavelet_denoising(raw_sig)
            scalogram = np.abs(extract_echo_features(denoised))
            wavelet_fig.add_trace(go.Heatmap(z=scalogram, colorscale='Viridis', showscale=False))
        except: pass
    else:
        # شكل عشوائي جمالي في حال عدم وجود مكتبات
        z_data = [[random.random() for _ in range(20)] for _ in range(20)]
        wavelet_fig.add_trace(go.Heatmap(z=z_data, colorscale='Viridis', showscale=False))
    
    wavelet_fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)')

    # 4. تحديث خريطة الأسطول
    fleet_colors = ['#ff0000' if s > 0.6 else '#ffa500' if s > 0.3 else '#00ff00' for s in fleet_severities]
    
    map_fig = go.Figure(go.Scattermapbox(
        lat=fleet_df['lat'], lon=fleet_df['lon'],
        mode='markers',
        marker=go.scattermapbox.Marker(size=12, color=fleet_colors, opacity=0.8),
        text=fleet_df['Site'], hoverinfo='text'
    ))

    map_fig.update_layout(
        mapbox_style="carto-darkmatter", mapbox_zoom=5.2,
        mapbox_center={"lat": 35.0, "lon": 38.5}, height=350,
        margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor='rgba(0,0,0,0)'
    )

    health_display = [html.Div(f"{'✅' if v else '❌'} {k}", 
                               className="text-success" if v else "text-danger") for k, v in health_reports.items()]

    return (
        f"{final_val*100:.1f}%", f"±{uncert*10:.2f}%", f"${risk_cost:,.0f}",
        decision, color, health_display, wavelet_fig, map_fig
    )

@app.callback(
    [Output("drone-modal", "is_open"), Output("drone-title", "children"), Output("drone-hud-content", "children")],
    [Input("drone-btn", "n_clicks"), Input("close-drone", "n_clicks")],
    [State("drone-modal", "is_open"), State("block-val", "children")]
)
def manage_drone_mission(n1, n2, is_open, severity):
    if not n1 and not n2: return no_update, no_update, no_update
    target_site = fleet_df.iloc[0]['Site']
    
    hud = html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.P("📡 SATELLITE LINK: STABLE", className="text-success mb-1 small"),
                    html.P(f"📍 GPS: 34.73° N, 36.71° E", className="text-info mb-3"),
                    html.H5(f"Telemetry: {severity} Blockage Detected", className="text-warning"),
                    html.P("Status: Drone scanning pipe thermal signature...", className="text-white small")
                ])
            ], width=7),
            dbc.Col([
                html.Label("ENERGY CORE", className="small"),
                dbc.Progress(value=88, label="88%", color="success", striped=True, animated=True, className="mb-3"),
                html.Label("DATA UPLINK", className="small"),
                dbc.Progress(value=95, color="info", className="mb-2")
            ], width=5)
        ])
    ], className="p-3 bg-black border border-primary rounded")
    
    return not is_open, f"🛰️ MISSION HUD: {target_site}", hud

@app.callback(Output('opt-output', 'children'), Input('opt-btn', 'n_clicks'), [State('p-slider', 'value'), State('v-slider', 'value')])
def run_ai_optimizer(n, p, v):
    if not n: return "Ready for Optimization"
    res = mc.genetic_optimization(p, v)
    return html.Div([html.Span(f"Optimal Pressure: {res} Bar", className="text-info fw-bold")])

if __name__ == '__main__':
    app.run(debug=True, port=8050)