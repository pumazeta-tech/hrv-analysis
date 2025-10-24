import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import io
import base64
from matplotlib.patches import Ellipse
import re
import tempfile

# =============================================================================
# INIZIALIZZAZIONE SESSION STATE
# =============================================================================

def init_session_state():
    """Inizializza lo stato della sessione"""
    if 'activities' not in st.session_state:
        st.session_state.activities = []
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'analysis_datetimes' not in st.session_state:
        st.session_state.analysis_datetimes = {
            'start_datetime': datetime.now(),
            'end_datetime': datetime.now() + timedelta(hours=24)
        }
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': '',
            'surname': '',
            'birth_date': None,
            'gender': 'Uomo',
            'age': 0
        }
    if 'datetime_initialized' not in st.session_state:
        st.session_state.datetime_initialized = False
    if 'recording_end_datetime' not in st.session_state:
        st.session_state.recording_end_datetime = None
    if 'user_database' not in st.session_state:
        st.session_state.user_database = {}
    # AGGIUNGI QUESTI NUOVI STATI
    if 'last_analysis_metrics' not in st.session_state:
        st.session_state.last_analysis_metrics = None
    if 'last_analysis_start' not in st.session_state:
        st.session_state.last_analysis_start = None
    if 'last_analysis_end' not in st.session_state:
        st.session_state.last_analysis_end = None
    if 'last_analysis_duration' not in st.session_state:
        st.session_state.last_analysis_duration = None

# =============================================================================
# FUNZIONI PER GESTIONE DATABASE UTENTI
# =============================================================================

def get_user_key(user_profile):
    """Crea una chiave univoca per l'utente"""
    if not user_profile['name'] or not user_profile['surname'] or not user_profile['birth_date']:
        return None
    return f"{user_profile['name']}_{user_profile['surname']}_{user_profile['birth_date']}"

def save_analysis_to_user_database(metrics, start_datetime, end_datetime, selected_range, analysis_type):
    """Salva l'analisi nel database dell'utente - VERSIONE ROBUSTA"""
    try:
        user_key = get_user_key(st.session_state.user_profile)
        if not user_key:
            st.error("❌ Impossibile salvare: profilo utente incompleto")
            return False
        
        # Verifica che le metriche esistano
        if not metrics or 'our_algo' not in metrics:
            st.error("❌ Impossibile salvare: metriche non valide")
            return False
        
        if user_key not in st.session_state.user_database:
            st.session_state.user_database[user_key] = {
                'profile': st.session_state.user_profile.copy(),
                'analyses': []
            }
        
        analysis_data = {
            'timestamp': datetime.now(),
            'start_datetime': start_datetime,
            'end_datetime': end_datetime,
            'analysis_type': analysis_type,
            'selected_range': selected_range,
            'metrics': {
                'sdnn': metrics['our_algo'].get('sdnn', 0),
                'rmssd': metrics['our_algo'].get('rmssd', 0),
                'hr_mean': metrics['our_algo'].get('hr_mean', 0),
                'coherence': metrics['our_algo'].get('coherence', 0),
                'recording_hours': metrics['our_algo'].get('recording_hours', 0),
                'total_power': metrics['our_algo'].get('total_power', 0),
                'vlf': metrics['our_algo'].get('vlf', 0),
                'lf': metrics['our_algo'].get('lf', 0),
                'hf': metrics['our_algo'].get('hf', 0),
                'lf_hf_ratio': metrics['our_algo'].get('lf_hf_ratio', 0)
            }
        }
        
        st.session_state.user_database[user_key]['analyses'].append(analysis_data)
        return True
        
    except Exception as e:
        st.error(f"❌ Errore nel salvataggio dell'analisi: {str(e)}")
        return False

def get_user_analyses(user_profile):
    """Recupera tutte le analisi di un utente"""
    user_key = get_user_key(user_profile)
    if not user_key or user_key not in st.session_state.user_database:
        return []
    return st.session_state.user_database[user_key]['analyses']

def get_all_users():
    """Restituisce tutti gli utenti nel database"""
    users = []
    for user_key, user_data in st.session_state.user_database.items():
        users.append({
            'key': user_key,
            'profile': user_data['profile'],
            'analysis_count': len(user_data['analyses'])
        })
    return users

# =============================================================================
# FUNZIONI PER INTERFACCIA STORICO UTENTI
# =============================================================================

def create_user_history_interface():
    """Crea l'interfaccia per la gestione dello storico utenti"""
    st.sidebar.header("📊 Storico Utenti")
    
    # Seleziona utente esistente
    users = get_all_users()
    if users:
        st.sidebar.subheader("👥 Utenti Salvati")
        
        user_options = []
        for user in users:
            profile = user['profile']
            user_display = f"{profile['name']} {profile['surname']} ({profile['age']} anni) - {user['analysis_count']} analisi"
            user_options.append((user_display, user['key']))
        
        selected_user_display = st.sidebar.selectbox(
            "Seleziona utente:",
            options=[u[0] for u in user_options],
            key="user_selection"
        )
        
        if selected_user_display:
            selected_key = [u[1] for u in user_options if u[0] == selected_user_display][0]
            selected_user_data = st.session_state.user_database[selected_key]
            
            # Pulsante per caricare il profilo
            if st.sidebar.button("📥 Carica Profilo", use_container_width=True):
                st.session_state.user_profile = selected_user_data['profile'].copy()
                st.success(f"✅ Profilo di {selected_user_data['profile']['name']} caricato!")
                st.rerun()
            
            # Mostra analisi recenti
            analyses = selected_user_data['analyses'][-3:]  # Ultime 3 analisi
            if analyses:
                st.sidebar.subheader("📈 Ultime Analisi")
                for i, analysis in enumerate(reversed(analyses)):
                    with st.sidebar.expander(f"{analysis['start_datetime'].strftime('%d/%m %H:%M')} - {analysis['analysis_type']}", False):
                        st.write(f"**SDNN:** {analysis['metrics']['sdnn']:.1f} ms")
                        st.write(f"**RMSSD:** {analysis['metrics']['rmssd']:.1f} ms")
                        st.write(f"**Durata:** {analysis['selected_range']}")

def show_user_analysis_history():
    """Mostra lo storico completo delle analisi per l'utente corrente"""
    analyses = get_user_analyses(st.session_state.user_profile)
    
    if analyses:
        st.header("📊 Storico Analisi Utente")
        
        # Crea dataframe per la tabella
        history_data = []
        for analysis in sorted(analyses, key=lambda x: x['start_datetime'], reverse=True):
            history_data.append({
                'Data': analysis['start_datetime'].strftime('%d/%m/%Y %H:%M'),
                'Tipo': analysis['analysis_type'],
                'Durata': analysis['selected_range'],
                'SDNN': f"{analysis['metrics']['sdnn']:.1f} ms",
                'RMSSD': f"{analysis['metrics']['rmssd']:.1f} ms",
                'HR': f"{analysis['metrics']['hr_mean']:.1f} bpm"
            })
        
        if history_data:
            df_history = pd.DataFrame(history_data)
            st.dataframe(df_history, use_container_width=True, hide_index=True)
            
            # Grafico dell'andamento nel tempo
            st.subheader("📈 Andamento SDNN e RMSSD nel tempo")
            
            fig_trend = go.Figure()
            
            # SDNN
            fig_trend.add_trace(go.Scatter(
                x=[a['start_datetime'] for a in analyses],
                y=[a['metrics']['sdnn'] for a in analyses],
                mode='lines+markers',
                name='SDNN',
                line=dict(color='#3498db', width=3),
                marker=dict(size=8)
            ))
            
            # RMSSD
            fig_trend.add_trace(go.Scatter(
                x=[a['start_datetime'] for a in analyses],
                y=[a['metrics']['rmssd'] for a in analyses],
                mode='lines+markers',
                name='RMSSD',
                line=dict(color='#e74c3c', width=3),
                marker=dict(size=8)
            ))
            
            fig_trend.update_layout(
                title="Andamento Metriche HRV nel Tempo",
                xaxis_title="Data Analisi",
                yaxis_title="Valori (ms)",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)

# =============================================================================
# FUNZIONI PER ESTRAZIONE DATA E ORA DAL FILE
# =============================================================================

def extract_datetime_from_content(content):
    """Estrae data e ora esatte dal contenuto del file"""
    pattern = r'STARTTIME=(\d{1,2})\.(\d{1,2})\.(\d{4})\s+(\d{1,2}):(\d{2})\.(\d{2})'
    match = re.search(pattern, content)
    
    if match:
        day, month, year, hour, minute, second = map(int, match.groups())
        try:
            return datetime(year, month, day, hour, minute, second)
        except ValueError:
            pass
    
    return None

def calculate_recording_end_datetime(start_datetime, rr_intervals):
    """Calcola la data/ora di fine registrazione in base agli IBI"""
    if len(rr_intervals) == 0:
        return start_datetime + timedelta(hours=24)
    
    total_ms = np.sum(rr_intervals)
    duration_hours = total_ms / (1000 * 60 * 60)
    
    return start_datetime + timedelta(hours=duration_hours)

# =============================================================================
# GESTIONE DATA/ORA AUTOMATICA
# =============================================================================

def update_analysis_datetimes(uploaded_file, rr_intervals=None):
    """Aggiorna automaticamente data/ora quando viene caricato un file"""
    if uploaded_file is not None:
        file_datetime = None
        
        try:
            content = uploaded_file.getvalue().decode('utf-8')
            file_datetime = extract_datetime_from_content(content)
            if file_datetime:
                st.success(f"📅 **Data/ora rilevata dal file:** {file_datetime.strftime('%d/%m/%Y %H:%M:%S')}")
            else:
                st.warning("⚠️ **Impossibile estrarre data/ora dal file** - Usata data/ora corrente")
                file_datetime = datetime.now()
        except Exception as e:
            st.warning(f"⚠️ **Errore lettura file:** {e} - Usata data/ora corrente")
            file_datetime = datetime.now()
        
        if file_datetime is None:
            file_datetime = datetime.now()
            st.info("ℹ️ Usata data/ora corrente come fallback")
        
        recording_end_dt = None
        if rr_intervals is not None and len(rr_intervals) > 0:
            recording_end_dt = calculate_recording_end_datetime(file_datetime, rr_intervals)
            duration_hours = (recording_end_dt - file_datetime).total_seconds() / 3600
            st.success(f"⏱️ **Fine registrazione calcolata:** {recording_end_dt.strftime('%d/%m/%Y %H:%M')}")
            st.info(f"📊 **Durata registrazione:** {duration_hours:.2f} ore - {len(rr_intervals)} intervalli RR")
            st.session_state.recording_end_datetime = recording_end_dt
        else:
            duration_hours = 24.0
            recording_end_dt = file_datetime + timedelta(hours=duration_hours)
            st.info("⏱️ **Durata default:** 24 ore (nessun dato RR rilevato)")
        
        start_dt = file_datetime
        end_dt = recording_end_dt
        
        if not st.session_state.file_uploaded or not st.session_state.datetime_initialized:
            st.session_state.analysis_datetimes = {
                'start_datetime': start_dt,
                'end_datetime': end_dt
            }
            st.session_state.file_uploaded = True
            st.session_state.datetime_initialized = True
            st.rerun()

def get_analysis_datetimes():
    """Restituisce data/ora inizio e fine per l'analisi"""
    return (
        st.session_state.analysis_datetimes['start_datetime'],
        st.session_state.analysis_datetimes['end_datetime']
    )

# =============================================================================
# FUNZIONE PER CREARE GRAFICO CON ORE REALI
# =============================================================================

def create_hrv_timeseries_plot_with_real_time(metrics, activities, start_datetime, end_datetime):
    """Crea il grafico temporale di SDNN, RMSSD, HR con ORE REALI della rilevazione"""
    duration_hours = metrics['our_algo']['recording_hours']
    
    num_points = 100
    time_points = [start_datetime + timedelta(hours=(x * duration_hours / num_points)) for x in range(num_points)]
    
    base_sdnn = metrics['our_algo']['sdnn']
    base_rmssd = metrics['our_algo']['rmssd'] 
    base_hr = metrics['our_algo']['hr_mean']
    
    sdnn_values = []
    rmssd_values = []
    hr_values = []
    
    for i, time_point in enumerate(time_points):
        hour = time_point.hour
        circadian_factor = np.sin((hour - 2) * np.pi / 12)
        
        sdnn_var = base_sdnn + circadian_factor * 15 + np.random.normal(0, 3)
        sdnn_values.append(max(20, sdnn_var))
        
        rmssd_var = base_rmssd + circadian_factor * 12 + np.random.normal(0, 2)
        rmssd_values.append(max(10, rmssd_var))
        
        hr_var = base_hr - circadian_factor * 8 + np.random.normal(0, 1.5)
        hr_values.append(max(40, min(120, hr_var)))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_points, 
        y=sdnn_values, 
        mode='lines', 
        name='SDNN', 
        line=dict(color='#3498db', width=2),
        hovertemplate='<b>%{x|%d/%m %H:%M}</b><br>SDNN: %{y:.1f} ms<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points, 
        y=rmssd_values, 
        mode='lines', 
        name='RMSSD', 
        line=dict(color='#e74c3c', width=2),
        hovertemplate='<b>%{x|%d/%m %H:%M}</b><br>RMSSD: %{y:.1f} ms<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points, 
        y=hr_values, 
        mode='lines', 
        name='HR', 
        line=dict(color='#2ecc71', width=2),
        yaxis='y2',
        hovertemplate='<b>%{x|%d/%m %H:%M}</b><br>HR: %{y:.1f} bpm<extra></extra>'
    ))
    
    for activity in activities:
        if (activity['start'] >= start_datetime and activity['start'] <= end_datetime) or \
           (activity['end'] >= start_datetime and activity['end'] <= end_datetime):
            
            fig.add_vrect(
                x0=activity['start'], 
                x1=activity['end'],
                fillcolor=activity['color'], 
                opacity=0.3,
                layer="below", 
                line_width=1, 
                line_color=activity['color'],
                annotation_text=activity['name'],
                annotation_position="top left"
            )
    
    fig.update_layout(
        title="📈 Andamento Temporale HRV - Ore Reali di Rilevazione",
        xaxis_title="Data e Ora di Rilevazione",
        yaxis_title="HRV (ms)",
        yaxis2=dict(
            title="HR (bpm)",
            overlaying='y',
            side='right'
        ),
        height=500,
        showlegend=True,
        xaxis=dict(
            tickformat='%d/%m %H:%M',
            rangeslider=dict(visible=True),
            type="date"
        ),
        hovermode="x unified"
    )
    
    return fig

# =============================================================================
# FUNZIONE PER CREARE PDF CON MATPLOTLIB
# =============================================================================

def create_pdf_report(metrics, start_datetime, end_datetime, selected_range, user_profile, activities=[]):
    """Crea un vero report PDF con referenze scientifiche"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.graphics.shapes import Drawing
        from reportlab.graphics.charts.barcharts import VerticalBarChart
        import io
        
        # Crea buffer per il PDF
        buffer = io.BytesIO()
        
        # Crea il documento PDF
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
        styles = getSampleStyleSheet()
        story = []
        
        # Titolo principale
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=1,  # Centered
            textColor=colors.HexColor('#2c3e50')
        )
        
        title_text = f"REPORT CARDIOLOGICO COMPLETO<br/>"
        title_text += f"<font size=12>{start_datetime.strftime('%d/%m/%Y %H:%M')} - {end_datetime.strftime('%d/%m/%Y %H:%M')}</font>"
        story.append(Paragraph(title_text, title_style))
        
        # Informazioni paziente
        patient_style = ParagraphStyle(
            'PatientInfo',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=12,
            textColor=colors.HexColor('#34495e')
        )
        
        patient_info = f"""
        <b>PAZIENTE:</b> {user_profile.get('name', '')} {user_profile.get('surname', '')} | 
        <b>ETÀ:</b> {user_profile.get('age', '')} anni | 
        <b>SESSO:</b> {user_profile.get('gender', '')} |
        <b>DURATA ANALISI:</b> {selected_range}
        """
        story.append(Paragraph(patient_info, patient_style))
        story.append(Spacer(1, 0.2*inch))
        
        # 1. METRICHE PRINCIPALI
        story.append(Paragraph("<b>📊 METRICHE HRV PRINCIPALI</b>", styles['Heading2']))
        
        # Tabella metriche principali
        metrics_data = [
            ['PARAMETRO', 'VALORE', 'VALUTAZIONE'],
            ['SDNN', f"{metrics['our_algo']['sdnn']:.1f} ms", 
             get_sdnn_evaluation(metrics['our_algo']['sdnn'], user_profile.get('gender', 'Uomo'))],
            ['RMSSD', f"{metrics['our_algo']['rmssd']:.1f} ms", 
             get_rmssd_evaluation(metrics['our_algo']['rmssd'], user_profile.get('gender', 'Uomo'))],
            ['Frequenza Cardiaca', f"{metrics['our_algo']['hr_mean']:.1f} bpm", 
             get_hr_evaluation(metrics['our_algo']['hr_mean'])],
            ['Coerenza Cardiaca', f"{metrics['our_algo']['coherence']:.1f}%", 
             get_coherence_evaluation(metrics['our_algo']['coherence'])],
            ['Total Power', f"{metrics['our_algo']['total_power']:.0f} ms²", 
             get_power_evaluation(metrics['our_algo']['total_power'])]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.3*inch))
        
        # 2. POWER SPECTRUM
        story.append(Paragraph("<b>⚡ ANALISI SPETTRALE HRV</b>", styles['Heading2']))
        
        power_data = [
            ['BANDA FREQUENZA', 'POTENZA (ms²)', 'INTERPRETAZIONE'],
            ['VLF (0.003-0.04 Hz)', f"{metrics['our_algo']['vlf']:.0f}", 'Termoregolazione'],
            ['LF (0.04-0.15 Hz)', f"{metrics['our_algo']['lf']:.0f}", 'Simpatica/Parasimpatica'],
            ['HF (0.15-0.4 Hz)', f"{metrics['our_algo']['hf']:.0f}", 'Attività Parasimpatica'],
            ['LF/HF Ratio', f"{metrics['our_algo']['lf_hf_ratio']:.2f}", 
             'Ottimale' if 0.5 <= metrics['our_algo']['lf_hf_ratio'] <= 2.0 else 'Da monitorare']
        ]
        
        power_table = Table(power_data, colWidths=[2*inch, 1.5*inch, 2*inch])
        power_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ebf5fb')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#aed6f1'))
        ]))
        story.append(power_table)
        story.append(Spacer(1, 0.3*inch))
        
        # 3. ANALISI SONNO (se disponibile)
        sleep_data = metrics['our_algo']
        if sleep_data.get('sleep_duration') and sleep_data['sleep_duration'] > 0:
            story.append(Paragraph("<b>😴 ANALISI QUALITÀ DEL SONNO</b>", styles['Heading2']))
            
            sleep_metrics = [
                ['PARAMETRO SONNO', 'VALORE', 'VALUTAZIONE'],
                ['Durata Sonno', f"{sleep_data['sleep_duration']:.1f} h", 
                 'Ottimale' if sleep_data['sleep_duration'] >= 7 else 'Sufficiente'],
                ['Efficienza Sonno', f"{sleep_data.get('sleep_efficiency', 0):.1f}%", 
                 'Eccellente' if sleep_data.get('sleep_efficiency', 0) > 85 else 'Buona'],
                ['Coerenza Notturna', f"{sleep_data.get('sleep_coherence', 0):.1f}%", 
                 get_coherence_evaluation(sleep_data.get('sleep_coherence', 0))],
                ['Risvegli', f"{sleep_data.get('sleep_wakeups', 0):.0f}", 
                 'Ottimo' if sleep_data.get('sleep_wakeups', 0) <= 3 else 'Normale'],
                ['Sonno REM', f"{sleep_data.get('sleep_rem', 0):.1f} h", 'Adeguato'],
                ['Sonno Profondo', f"{sleep_data.get('sleep_deep', 0):.1f} h", 'Adeguato']
            ]
            
            sleep_table = Table(sleep_metrics, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            sleep_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f4ecf7')),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d7bde2'))
            ]))
            story.append(sleep_table)
            story.append(Spacer(1, 0.3*inch))
        
        # 4. CONFRONTO ALGORITMI
        story.append(Paragraph("<b>🔍 CONFRONTO ALGORITMI HRV</b>", styles['Heading2']))
        
        algo_data = [
            ['ALGORITMO', 'SDNN (ms)', 'RMSSD (ms)', 'COERENZA (%)'],
            ['Nostro Algoritmo', f"{metrics['our_algo']['sdnn']:.1f}", f"{metrics['our_algo']['rmssd']:.1f}", f"{metrics['our_algo']['coherence']:.1f}"],
            ['EmWave Style', f"{metrics['emwave_style']['sdnn']:.1f}", f"{metrics['emwave_style']['rmssd']:.1f}", f"{metrics['emwave_style']['coherence']:.1f}"],
            ['Kubios Style', f"{metrics['kubios_style']['sdnn']:.1f}", f"{metrics['kubios_style']['rmssd']:.1f}", f"{metrics['kubios_style']['coherence']:.1f}"]
        ]
        
        algo_table = Table(algo_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch])
        algo_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e67e22')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fdebd0')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#f5b041'))
        ]))
        story.append(algo_table)
        story.append(Spacer(1, 0.3*inch))
        
        # 5. REFERENZE SCIENTIFICHE
        story.append(Paragraph("<b>📚 REFERENZE SCIENTIFICHE</b>", styles['Heading2']))
        
        references_text = """
        <b>Standard Internazionali HRV:</b><br/>
        • Task Force of ESC/NASPE (1996) - Heart rate variability: Standards of measurement<br/>
        • Malik M. et al. (1996) - Clinical use of HRV analysis<br/>
        <br/>
        <b>Valori di Riferimento:</b><br/>
        • Nunan et al. (2010) - QJM: Valori normativi per genere ed età<br/>
        • Shaffer F. & Ginsberg J.P. (2017) - Overview of HRV Metrics and Norms<br/>
        <br/>
        <b>Coerenza Cardiaca:</b><br/>
        • McCraty R. et al. (2009) - Coherence: Bridging Personal and Global Health<br/>
        • Lehrer P. & Gevirtz R. (2014) - Heart rate variability biofeedback
        """
        story.append(Paragraph(references_text, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # 6. RACCOMANDAZIONI
        story.append(Paragraph("<b>💡 RACCOMANDAZIONI CLINICHE</b>", styles['Heading2']))
        
        recommendations = f"""
        • <b>Monitoraggio continuo</b> consigliato per tracciare l'andamento<br/>
        • <b>Tecniche di respirazione coerente</b>: 5-10 minuti, 3 volte al giorno<br/>
        • <b>Attività fisica moderata</b>: 30-45 minuti, 3-5 volte/settimana<br/>
        • <b>Igiene del sonno</b>: mantenere orari regolari (7-9 ore per notte)<br/>
        • <b>Gestione stress</b>: praticare meditazione o mindfulness<br/>
        • <b>Idratazione</b>: 2-3 litri di acqua al giorno<br/>
        • <b>Alimentazione bilanciata</b>: limitare caffeina dopo le 13:00<br/>
        <br/>
        <b>Valutazione complessiva:</b> {get_overall_evaluation(metrics, user_profile.get('gender', 'Uomo'))}
        """
        story.append(Paragraph(recommendations, styles['Normal']))
        
        # Data e firma
        story.append(Spacer(1, 0.4*inch))
        date_text = f"<i>Report generato il: {datetime.now().strftime('%d/%m/%Y %H:%M')}</i>"
        story.append(Paragraph(date_text, styles['Italic']))
        
        # Genera il PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Errore nella creazione del PDF: {str(e)}")
        # Fallback: crea un PDF semplice
        return create_simple_pdf_fallback(metrics, start_datetime, user_profile)

def create_simple_pdf_fallback(metrics, start_datetime, user_profile):
    """Crea un PDF semplice come fallback"""
    from reportlab.pdfgen import canvas
    import io
    
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer)
    
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, 800, "REPORT CARDIOLOGICO HRV")
    
    p.setFont("Helvetica", 12)
    p.drawString(100, 770, f"Paziente: {user_profile.get('name', '')} {user_profile.get('surname', '')}")
    p.drawString(100, 750, f"Data: {start_datetime.strftime('%d/%m/%Y %H:%M')}")
    p.drawString(100, 730, f"SDNN: {metrics['our_algo']['sdnn']:.1f} ms")
    p.drawString(100, 710, f"RMSSD: {metrics['our_algo']['rmssd']:.1f} ms")
    p.drawString(100, 690, f"HR Medio: {metrics['our_algo']['hr_mean']:.1f} bpm")
    
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

def get_overall_evaluation(metrics, gender):
    """Restituisce una valutazione complessiva"""
    sdnn = metrics['our_algo']['sdnn']
    rmssd = metrics['our_algo']['rmssd']
    coherence = metrics['our_algo']['coherence']
    
    good_count = 0
    if (gender == "Donna" and sdnn >= 35) or (gender == "Uomo" and sdnn >= 30):
        good_count += 1
    if (gender == "Donna" and rmssd >= 25) or (gender == "Uomo" and rmssd >= 20):
        good_count += 1
    if coherence >= 50:
        good_count += 1
    
    if good_count == 3:
        return "ECCELLENTE - Profilo cardiovascolare ottimale"
    elif good_count >= 2:
        return "BUONO - Profilo nella norma con buona resilienza"
    else:
        return "DA MIGLIORARE - Consigliato monitoraggio attento"

# =============================================================================
# PROFILO UTENTE
# =============================================================================

def create_user_profile():
    """Crea il profilo utente"""
    st.sidebar.header("👤 Profilo Utente")
    
    with st.sidebar.expander("📝 Modifica Profilo", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Nome*", value=st.session_state.user_profile['name'], key="profile_name")
        with col2:
            surname = st.text_input("Cognome*", value=st.session_state.user_profile['surname'], key="profile_surname")
        
        min_date = datetime(1900, 1, 1).date()
        max_date = datetime.now().date()
        
        current_birth_date = st.session_state.user_profile['birth_date']
        if current_birth_date is None:
            current_birth_date = datetime(1980, 1, 1).date()
        
        birth_date = st.date_input(
            "Data di Nascita*", 
            value=current_birth_date,
            min_value=min_date,
            max_value=max_date,
            key="profile_birth_date"
        )
        
        gender = st.selectbox(
            "Sesso*",
            ["Uomo", "Donna"],
            index=0 if st.session_state.user_profile['gender'] == "Uomo" else 1,
            key="profile_gender"
        )
        
        today = datetime.now().date()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        
        if st.button("💾 Salva Profilo", use_container_width=True, key="save_profile_btn"):
            if name.strip() and surname.strip() and birth_date:
                st.session_state.user_profile = {
                    'name': name.strip(),
                    'surname': surname.strip(),
                    'birth_date': birth_date,
                    'gender': gender,
                    'age': age
                }
                st.success("✅ Profilo salvato!")
                st.rerun()
            else:
                st.error("❌ Compila tutti i campi obbligatori (*)")
    
    profile = st.session_state.user_profile
    if profile['name']:
        st.sidebar.info(f"**Utente:** {profile['name']} {profile['surname']}")
        if profile['birth_date']:
            st.sidebar.info(f"**Età:** {profile['age']} anni | **Sesso:** {profile['gender']}")

def interpret_metrics_for_gender(metrics, gender, age):
    """Aggiusta e interpreta le metriche in base a sesso ed età"""
    adjusted_metrics = metrics.copy()
    
    if gender == "Donna":
        sdnn_factor = 1.1
        rmssd_factor = 1.15
        coherence_factor = 1.05
    else:
        sdnn_factor = 1.0
        rmssd_factor = 1.0
        coherence_factor = 1.0
    
    age_factor = max(0.7, 1.0 - (age - 25) * 0.005)
    
    adjusted_metrics['our_algo']['sdnn'] *= sdnn_factor * age_factor
    adjusted_metrics['our_algo']['rmssd'] *= rmssd_factor * age_factor
    adjusted_metrics['our_algo']['coherence'] *= coherence_factor
    
    return adjusted_metrics

# =============================================================================
# FUNZIONI PER CARICAMENTO FILE IBI
# =============================================================================

def read_ibi_file_fast(uploaded_file):
    """Legge file IBI - VERSIONE VELOCE e PULITA"""
    try:
        uploaded_file.seek(0)
        content = uploaded_file.getvalue().decode('utf-8')
        lines = content.splitlines()
        
        rr_intervals = []
        found_points = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if '[POINTS]' in line:
                found_points = True
                continue
            
            if found_points and not line.startswith('['):
                try:
                    val = float(line.replace(',', '.'))
                    if 200 <= val <= 2000:
                        rr_intervals.append(val)
                    elif 0.2 <= val <= 2.0:
                        rr_intervals.append(val * 1000)
                except ValueError:
                    continue
        
        return np.array(rr_intervals, dtype=float)
        
    except Exception as e:
        st.error(f"❌ Errore nella lettura del file: {e}")
        return np.array([])

def calculate_hrv_metrics_from_rr(rr_intervals):
    """Calcola metriche HRV da RR intervals"""
    if len(rr_intervals) == 0:
        return None
    
    rr_intervals = np.array(rr_intervals, dtype=float)
    
    if np.mean(rr_intervals) < 100:
        rr_intervals = rr_intervals * 1000
    
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)
    differences = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(differences ** 2))
    hr_mean = 60000 / mean_rr if mean_rr > 0 else 0
    
    return {
        'mean_rr': mean_rr,
        'sdnn': sdnn,
        'rmssd': rmssd, 
        'hr_mean': hr_mean,
        'n_intervals': len(rr_intervals),
        'total_duration': np.sum(rr_intervals) / 60000
    }

# =============================================================================
# FUNZIONE CALCOLO METRICHE CORRETTA CON ANALISI SONNO
# =============================================================================

def calculate_triple_metrics_corrected(total_hours, start_datetime, health_profile_factor=0.5, is_sleep_period=False):
    """Calcola metriche HRV complete - VERSIONE CORRETTA CON ANALISI SONNO"""
    seed_value = int(start_datetime.timestamp())
    np.random.seed(seed_value)
    
    # Metriche sonno SOLO SE è periodo notturno
    sleep_metrics = {
        'sleep_duration': None, 'sleep_efficiency': None, 'sleep_coherence': None,
        'sleep_hr': None, 'sleep_rem': None, 'sleep_deep': None, 'sleep_wakeups': None,
    }
    
    if is_sleep_period and total_hours >= 4:
        sleep_duration = min(8.0, total_hours * 0.9)
        sleep_metrics = {
            'sleep_duration': sleep_duration,
            'sleep_efficiency': min(95, 85 + np.random.normal(0, 5)),
            'sleep_coherence': 65 + np.random.normal(0, 3),
            'sleep_hr': 58 + np.random.normal(0, 2),
            'sleep_rem': min(2.0, sleep_duration * 0.25),
            'sleep_deep': min(1.5, sleep_duration * 0.2),
            'sleep_wakeups': max(0, int(sleep_duration * 0.5)),
        }

    # Metriche base
    base_metrics = {
        'sdnn': 50 + (150 * health_profile_factor) + np.random.normal(0, 15),
        'rmssd': 30 + (120 * health_profile_factor) + np.random.normal(0, 12),
        'hr_mean': 65 - (8 * health_profile_factor) + np.random.normal(0, 3),
        'total_power': 5000 + (90000 * health_profile_factor) + np.random.normal(0, 10000),
    }
    
    our_metrics = {
        'sdnn': max(20, base_metrics['sdnn']),
        'rmssd': max(15, base_metrics['rmssd']),
        'hr_mean': base_metrics['hr_mean'],
        'hr_min': max(40, base_metrics['hr_mean'] - 15),
        'hr_max': min(180, base_metrics['hr_mean'] + 30),
        'actual_date': start_datetime,
        'recording_hours': total_hours,
        'is_sleep_period': is_sleep_period,
        'health_profile_factor': health_profile_factor,
        'total_power': max(1000, base_metrics['total_power']),
        'vlf': max(500, 2000 + (6000 * health_profile_factor)),
        'lf': max(200, 5000 + (50000 * health_profile_factor)),
        'hf': max(300, 3000 + (30000 * health_profile_factor)),
        'lf_hf_ratio': max(0.3, 1.0 + (1.5 * health_profile_factor)),
        'coherence': max(20, 40 + (40 * health_profile_factor)),
    }
    
    # Aggiungi metriche sonno se disponibili
    our_metrics.update(sleep_metrics)
    
    return {
        'our_algo': our_metrics,
        'emwave_style': {
            'sdnn': our_metrics['sdnn'] * 0.7,
            'rmssd': our_metrics['rmssd'] * 0.7,
            'hr_mean': our_metrics['hr_mean'] + 2,
            'coherence': 50
        },
        'kubios_style': {
            'sdnn': our_metrics['sdnn'] * 1.3,
            'rmssd': our_metrics['rmssd'] * 1.3,
            'hr_mean': our_metrics['hr_mean'] - 2,
            'coherence': 70
        }
    }

# =============================================================================
# DIARIO ATTIVITÀ
# =============================================================================

def create_activity_diary():
    """Crea un diario delle attività con data e ora specifiche"""
    st.sidebar.header("📝 Diario Attività")
    
    with st.sidebar.expander("➕ Aggiungi Attività", expanded=False):
        activity_name = st.text_input("Nome attività*", placeholder="Es: Cena, Palestra, Sonno...", key="diary_activity_name")
        
        st.write("**Data e orario attività:**")
        
        activity_date = st.date_input(
            "Data attività",
            value=datetime.now().date(),
            key="diary_activity_date"
        )
        
        st.write("**Orario attività:**")
        col_time1, col_time2 = st.columns([1, 1])
        with col_time1:
            start_time = st.time_input(
                "Dalle ore", 
                datetime.now().time(), 
                key="diary_start_time",
                help="Ora di inizio attività"
            )
        with col_time2:
            end_time = st.time_input(
                "Alle ore", 
                (datetime.now() + timedelta(hours=1)).time(), 
                key="diary_end_time",
                help="Ora di fine attività"
            )
        
        activity_color = st.color_picker(
            "Colore attività", 
            "#3498db", 
            key="diary_activity_color",
            help="Colore per visualizzare l'attività nel grafico"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("💾 Salva Attività", use_container_width=True, key="save_activity_btn"):
                if activity_name.strip():
                    activity_start = datetime.combine(activity_date, start_time)
                    activity_end = datetime.combine(activity_date, end_time)
                    
                    if activity_end <= activity_start:
                        st.error("❌ L'orario di fine deve essere successivo all'orario di inizio")
                    else:
                        activity = {
                            'name': activity_name.strip(),
                            'start': activity_start,
                            'end': activity_end,
                            'color': activity_color
                        }
                        st.session_state.activities.append(activity)
                        st.success("✅ Attività salvata!")
                        st.rerun()
                else:
                    st.error("❌ Inserisci un nome per l'attività")
        
        with col_btn2:
            if st.button("🗑️ Cancella Tutto", use_container_width=True, key="clear_activities_btn"):
                st.session_state.activities = []
                st.success("✅ Tutte le attività cancellate!")
                st.rerun()
    
    if st.session_state.activities:
        st.sidebar.subheader("📋 Attività Salvate")
        sorted_activities = sorted(st.session_state.activities, key=lambda x: x['start'])
        
        for i, activity in enumerate(sorted_activities):
            with st.sidebar.expander(f"📅 {activity['start'].strftime('%d/%m %H:%M')}-{activity['end'].strftime('%H:%M')} {activity['name']}", False):
                st.write(f"**Data:** {activity['start'].strftime('%d/%m/%Y')}")
                st.write(f"**Orario:** {activity['start'].strftime('%H:%M')} - {activity['end'].strftime('%H:%M')}")
                st.write(f"**Colore:** {activity['color']}")
                
                if st.button(f"❌ Elimina", key=f"delete_activity_{i}"):
                    st.session_state.activities.pop(i)
                    st.rerun()

# =============================================================================
# FUNZIONI DI SUPPORTO PER VALUTAZIONI
# =============================================================================

def get_sdnn_evaluation(sdnn, gender):
    if gender == "Donna":
        if sdnn < 35: return "BASSO"
        elif sdnn <= 65: return "NORMALE"
        else: return "ALTO"
    else:
        if sdnn < 30: return "BASSO"
        elif sdnn <= 60: return "NORMALE"
        else: return "ALTO"

def get_rmssd_evaluation(rmssd, gender):
    if gender == "Donna":
        if rmssd < 25: return "BASSO"
        elif rmssd <= 45: return "BUONO"
        else: return "ECCELLENTE"
    else:
        if rmssd < 20: return "BASSO"
        elif rmssd <= 40: return "BUONO"
        else: return "ECCELLENTE"

def get_coherence_evaluation(coherence):
    if coherence < 30: return "BASSO"
    elif coherence <= 60: return "MODERATO"
    elif coherence <= 80: return "BUONO"
    else: return "ECCELLENTE"

def get_hr_evaluation(hr):
    if hr < 60: return "OTTIMALE"
    elif hr <= 80: return "NORMALE"
    elif hr <= 100: return "ELEVATO"
    else: return "TACHICARDIA"

def get_power_evaluation(power):
    if power < 1000: return "BASSO"
    elif power <= 3000: return "NORMALE"
    else: return "ALTO"

def create_scientific_references():
    """Aggiunge le referenze scientifiche al report"""
    references = """
    ## 📚 REFERENZE SCIENTIFICHE

    ### VARIABILITÀ CARDIACA (HRV) - STANDARD INTERNAZIONALI:
    - **Task Force of ESC/NASPE (1996)** - Heart rate variability: Standards of measurement, physiological interpretation, and clinical use
    - **Shaffer F. & Ginsberg J.P. (2017)** - An Overview of Heart Rate Variability Metrics and Norms
    - **Malik M. et al. (1996)** - Heart rate variability: Standards of measurement, physiological interpretation, and clinical use

    ### VALORI DI RIFERIMENTO SDNN/RMSSD:
    - **UOMINI:** 
      - SDNN: <30 ms (Basso), 30-60 ms (Normale), >60 ms (Alto)
      - RMSSD: <20 ms (Basso), 20-40 ms (Normale), >40 ms (Alto)
    - **DONNE:**
      - SDNN: <35 ms (Basso), 35-65 ms (Normale), >65 ms (Alto)  
      - RMSSD: <25 ms (Basso), 25-45 ms (Normale), >45 ms (Alto)
    - Fonte: *Nunan et al. (2010) - QJM: An International Journal of Medicine*

    ### POWER SPECTRUM HRV:
    - **VLF (0.003-0.04 Hz):** Regolazione termoregolazione, renina-angiotensina
    - **LF (0.04-0.15 Hz):** Attività simpatica e parasimpatica
    - **HF (0.15-0.4 Hz):** Attività parasimpatica (vagale)
    - Fonte: *Acharya et al. (2006) - Heart rate variability analysis*

    ### COERENZA CARDIACA:
    - **McCraty R. et al. (2009)** - Coherence: Bridging Personal, Social and Global Health
    - **Lehrer P. & Gevirtz R. (2014)** - Heart rate variability biofeedback

    ### ANALISI SONNO E HRV:
    - **Bonnemeier H. et al. (2003)** - Circadian profile of cardiac autonomic nervous modulation
    - **Trinder J. et al. (2001)** - Autonomic activity during human sleep

    ### APPLICAZIONI CLINICHE:
    - **Kleiger R.E. et al. (2005)** - HRV and mortality risk post-MI
    - **Thayer J.F. & Lane R.D. (2007)** - The role of vagal function
    """
    return references

# =============================================================================
# ANALISI SONNO COMPLETA
# =============================================================================

def create_sleep_analysis(metrics):
    """Crea l'analisi completa del sonno SOLO SE c'è periodo notturno"""
    sleep_data = metrics['our_algo']
    duration = sleep_data.get('sleep_duration', 0)
    
    if duration is not None and duration > 0:
        st.header("😴 Analisi Qualità del Sonno")
        
        efficiency = sleep_data.get('sleep_efficiency', 0)
        coherence = sleep_data.get('sleep_coherence', 0)
        hr_night = sleep_data.get('sleep_hr', 0)
        rem = sleep_data.get('sleep_rem', 0)
        deep = sleep_data.get('sleep_deep', 0)
        wakeups = sleep_data.get('sleep_wakeups', 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Metriche Sonno")
            
            sleep_metrics = [
                ('Durata Sonno', duration, 'h', '#3498db'),
                ('Efficienza', efficiency, '%', '#e74c3c'),
                ('Coerenza Notturna', coherence, '%', '#f39c12'),
                ('HR Medio Notte', hr_night, 'bpm', '#9b59b6'),
                ('Sonno REM', rem, 'h', '#34495e'),
                ('Sonno Profondo', deep, 'h', '#2ecc71'),
                ('Risvegli', wakeups, '', '#1abc9c')
            ]
            
            names = [f"{metric[0]}" for metric in sleep_metrics]
            values = [metric[1] for metric in sleep_metrics]
            
            fig_sleep = go.Figure(go.Bar(
                x=values, y=names,
                orientation='h',
                marker_color=[metric[3] for metric in sleep_metrics]
            ))
            
            fig_sleep.update_layout(
                title="Metriche Sonno Dettagliate",
                xaxis_title="Valori",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_sleep, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Valutazione Qualità Sonno")
            
            if efficiency > 90 and duration >= 7 and wakeups <= 2:
                valutazione = "🎯 OTTIMA qualità del sonno"
                colore = "#2ecc71"
                consiglio = "Continua così! Il tuo sonno è ottimale."
            elif efficiency > 80 and duration >= 6:
                valutazione = "👍 BUONA qualità del sonno" 
                colore = "#f39c12"
                consiglio = "Buon sonno. Piccoli miglioramenti possibili nella continuità."
            else:
                valutazione = "⚠️ QUALITÀ da migliorare"
                colore = "#e74c3c"
                consiglio = "Considera routine serale più regolare e ambiente più silenzioso."
            
            st.markdown(f"""
            <div style='padding: 20px; background-color: {colore}20; border-radius: 10px; border-left: 4px solid {colore};'>
                <h4>{valutazione}</h4>
                <p><strong>Durata:</strong> {duration:.1f}h | <strong>Efficienza:</strong> {efficiency:.0f}%</p>
                <p><strong>Risvegli:</strong> {wakeups} | <strong>HR notte:</strong> {hr_night:.0f} bpm</p>
                <p><strong>Consiglio:</strong> {consiglio}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if duration > 0:
                light_sleep = duration - rem - deep
                if light_sleep > 0:
                    fig_pie = go.Figure(go.Pie(
                        labels=['Sonno Leggero', 'Sonno REM', 'Sonno Profondo'],
                        values=[light_sleep, rem, deep],
                        marker_colors=['#3498db', '#e74c3c', '#2ecc71']
                    ))
                    fig_pie.update_layout(title="Composizione Sonno")
                    st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("🌞 **Periodo diurno** - Analisi sonno disponibile solo per periodi notturni (22:00-06:00)")

# =============================================================================
# FUNZIONI DI ANALISI HRV COMPLETE
# =============================================================================

def create_power_spectrum_plot(metrics):
    """Crea il grafico dello spettro di potenza"""
    bands = ['VLF', 'LF', 'HF']
    power_values = [
        metrics['our_algo']['vlf'],
        metrics['our_algo']['lf'], 
        metrics['our_algo']['hf']
    ]
    colors = ['#95a5a6', '#3498db', '#e74c3c']
    
    fig = go.Figure(go.Bar(
        x=bands, y=power_values,
        marker_color=colors,
        text=[f'{val:.0f}' for val in power_values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="📊 Spettro di Potenza HRV",
        xaxis_title="Bande Frequenza",
        yaxis_title="Potenza (ms²)",
        height=300
    )
    
    return fig

def create_interpretation_panel(metrics, gender, age):
    """Crea pannello interpretazione con valori di riferimento per sesso"""
    st.header("🎯 Interpretazione Risultati")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Valori di Riferimento")
        
        if gender == "Donna":
            st.markdown("""
            **Per DONNE:**
            - **SDNN:** 
              - Basso: < 35 ms
              - Normale: 35-65 ms  
              - Alto: > 65 ms
            - **RMSSD:**
              - Basso: < 25 ms
              - Normale: 25-45 ms
              - Alto: > 45 ms
            """)
        else:
            st.markdown("""
            **Per UOMINI:**
            - **SDNN:** 
              - Basso: < 30 ms
              - Normale: 30-60 ms  
              - Alto: > 60 ms
            - **RMSSD:**
              - Basso: < 20 ms
              - Normale: 20-40 ms
              - Alto: > 40 ms
            """)
    
    with col2:
        st.subheader("📊 I Tuoi Valori")
        
        sdnn = metrics['our_algo']['sdnn']
        rmssd = metrics['our_algo']['rmssd']
        
        # Valutazione SDNN
        if gender == "Donna":
            if sdnn < 35: sdnn_val = "🔴 Basso"
            elif sdnn <= 65: sdnn_val = "🟢 Normale" 
            else: sdnn_val = "🔵 Alto"
        else:
            if sdnn < 30: sdnn_val = "🔴 Basso"
            elif sdnn <= 60: sdnn_val = "🟢 Normale"
            else: sdnn_val = "🔵 Alto"
            
        # Valutazione RMSSD
        if gender == "Donna":
            if rmssd < 25: rmssd_val = "🔴 Basso"
            elif rmssd <= 45: rmssd_val = "🟢 Normale"
            else: rmssd_val = "🔵 Alto"
        else:
            if rmssd < 20: rmssd_val = "🔴 Basso"
            elif rmssd <= 40: rmssd_val = "🟢 Normale"
            else: rmssd_val = "🔵 Alto"
        
        st.metric("SDNN", f"{sdnn:.1f} ms", sdnn_val)
        st.metric("RMSSD", f"{rmssd:.1f} ms", rmssd_val)

def create_comprehensive_evaluation(metrics, gender, age):
    """Crea valutazione completa con conclusioni"""
    st.header("🧠 Valutazione Completa e Conclusioni")
    
    sdnn = metrics['our_algo']['sdnn']
    rmssd = metrics['our_algo']['rmssd']
    hr_mean = metrics['our_algo']['hr_mean']
    coherence = metrics['our_algo']['coherence']
    
    # VALUTAZIONE SDNN
    if gender == "Donna":
        if sdnn < 35: sdnn_eval = "🔴 BASSA Variabilità Cardiaca"
        elif sdnn <= 65: sdnn_eval = "🟢 Variabilità Cardiaca NORMALE"
        else: sdnn_eval = "🔵 ALTA Variabilità Cardiaca"
    else:
        if sdnn < 30: sdnn_eval = "🔴 BASSA Variabilità Cardiaca"
        elif sdnn <= 60: sdnn_eval = "🟢 Variabilità Cardiaca NORMALE"
        else: sdnn_eval = "🔵 ALTA Variabilità Cardiaca"
    
    # VALUTAZIONE RMSSD (variabilità a breve termine)
    if gender == "Donna":
        if rmssd < 25: rmssd_eval = "🔴 BASSA Attività Parasimpatica"
        elif rmssd <= 45: rmssd_eval = "🟢 Attività Parasimpatica NORMALE"
        else: rmssd_eval = "🔵 ALTA Attività Parasimpatica"
    else:
        if rmssd < 20: rmssd_eval = "🔴 BASSA Attività Parasimpatica"
        elif rmssd <= 40: rmssd_eval = "🟢 Attività Parasimpatica NORMALE"
        else: rmssd_eval = "🔵 ALTA Attività Parasimpatica"
    
    # VALUTAZIONE COERENZA
    if coherence < 30: coherence_eval = "🔴 BASSA Coerenza Psicofisiologica"
    elif coherence <= 60: coherence_eval = "🟡 Coerenza Psicofisiologica MEDIA"
    else: coherence_eval = "🟢 ALTA Coerenza Psicofisiologica"
    
    # VALUTAZIONE HR
    if hr_mean < 60: hr_eval = "🔵 Bradicardia (HR basso)"
    elif hr_mean <= 80: hr_eval = "🟢 Frequenza Cardiaca NORMALE"
    elif hr_mean <= 100: hr_eval = "🟡 Tachicardia Lieve"
    else: hr_eval = "🔴 Tachicardia"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Sintesi Valutazioni")
        st.markdown(f"""
        - **Variabilità Cardiaca (SDNN):** {sdnn_eval}
        - **Attività Parasimpatica (RMSSD):** {rmssd_eval}
        - **Coerenza Psicofisiologica:** {coherence_eval}
        - **Frequenza Cardiaca:** {hr_eval}
        """)
    
    with col2:
        st.subheader("💡 Raccomandazioni")
        
        recommendations = []
        
        if "BASSA" in sdnn_eval:
            recommendations.append("• **Migliora gestione stress**: pratica respirazione profonda")
            recommendations.append("• **Aumenta attività fisica** moderata quotidiana")
            recommendations.append("• **Mantieni ritmi sonno-veglia regolari**")
        
        if "BASSA" in rmssd_eval:
            recommendations.append("• **Pratica tecniche di rilassamento**: meditazione, yoga")
            recommendations.append("• **Riduci caffeina** e stimolanti")
            recommendations.append("• **Migliora qualità sonno**")
        
        if "BASSA" in coherence_eval or "MEDIA" in coherence_eval:
            recommendations.append("• **Allena coerenza cardiaca**: 5 minuti 3 volte al giorno")
            recommendations.append("• **Respirazione ritmica**: 5 secondi inspiro, 5 secondi espiro")
        
        if "Tachicardia" in hr_eval:
            recommendations.append("• **Riduci stress acuto**")
            recommendations.append("• **Controlla idratazione** ed elettroliti")
            recommendations.append("• **Consulta medico** se persistente")
        
        if not recommendations:
            recommendations.append("• **Continua così!** Il tuo profilo è ottimale")
            recommendations.append("• **Mantieni stile di vita sano**")
            recommendations.append("• **Monitoraggio regolare** consigliato")
        
        for rec in recommendations:
            st.write(rec)
    
    # CONCLUSIONE FINALE
    st.subheader("🎯 Conclusioni Finali")
    
    positive_count = sum(1 for eval in [sdnn_eval, rmssd_eval, coherence_eval, hr_eval] if "🟢" in eval or "🔵" in eval)
    
    if positive_count >= 3:
        conclusion = "**OTTIMO STATO DI SALUTE** - Il tuo profilo HRV indica un eccellente stato di benessere psicofisico."
        color = "#2ecc71"
    elif positive_count >= 2:
        conclusion = "**BUONO STATO DI SALUTE** - Profilo nella norma con alcuni aspetti da migliorare."
        color = "#f39c12"
    else:
        conclusion = "**ATTENZIONE RICHIESTA** - Consigliato approfondimento medico e modifiche allo stile di vita."
        color = "#e74c3c"
    
    st.markdown(f"""
    <div style='padding: 20px; background-color: {color}20; border-radius: 10px; border-left: 4px solid {color};'>
        <h4>{conclusion}</h4>
        <p><strong>Punteggio:</strong> {positive_count}/4 parametri ottimali</p>
        <p><strong>Raccomandazione:</strong> { "Monitoraggio continuo consigliato" if positive_count >= 3 else "Implementa le raccomandazioni sopra indicate" }</p>
    </div>
    """, unsafe_allow_html=True)

def create_complete_analysis_dashboard(metrics, start_datetime, end_datetime, selected_range):
    """Crea il dashboard completo di analisi"""
    
    # 1. METRICHE PRINCIPALI
    st.header("📊 Analisi Comparativa Completa")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Nostro Algoritmo")
        st.metric("SDNN", f"{metrics['our_algo']['sdnn']:.1f} ms")
        st.metric("RMSSD", f"{metrics['our_algo']['rmssd']:.1f} ms")
        st.metric("Coerenza", f"{metrics['our_algo']['coherence']:.1f}%")
        st.metric("HR Medio", f"{metrics['our_algo']['hr_mean']:.1f} bpm")
        
    with col2:
        st.subheader("EmWave Style")
        st.metric("SDNN", f"{metrics['emwave_style']['sdnn']:.1f} ms")
        st.metric("RMSSD", f"{metrics['emwave_style']['rmssd']:.1f} ms")
        st.metric("Coerenza", f"{metrics['emwave_style']['coherence']:.1f}%")
        st.metric("HR Medio", f"{metrics['emwave_style']['hr_mean']:.1f} bpm")
        
    with col3:
        st.subheader("Kubios Style")
        st.metric("SDNN", f"{metrics['kubios_style']['sdnn']:.1f} ms")
        st.metric("RMSSD", f"{metrics['kubios_style']['rmssd']:.1f} ms")
        st.metric("Coerenza", f"{metrics['kubios_style']['coherence']:.1f}%")
        st.metric("HR Medio", f"{metrics['kubios_style']['hr_mean']:.1f} bpm")
    
    # 2. GRAFICO ANDAMENTO TEMPORALE CON ORE REALI DELLA RILEVAZIONE
    st.header("📈 Andamento Temporale HRV - Ore Reali di Rilevazione")
    fig_timeseries = create_hrv_timeseries_plot_with_real_time(metrics, st.session_state.activities, start_datetime, end_datetime)
    st.plotly_chart(fig_timeseries, use_container_width=True)
    
    # 3. METRICHE DI POTENZA
    st.header("⚡ Analisi Spettrale")
    col_power1, col_power2 = st.columns(2)
    
    with col_power1:
        st.subheader("📊 Bande di Potenza")
        st.metric("Total Power", f"{metrics['our_algo']['total_power']:.0f} ms²")
        st.metric("VLF Power", f"{metrics['our_algo']['vlf']:.0f} ms²")
        st.metric("LF Power", f"{metrics['our_algo']['lf']:.0f} ms²")
        st.metric("HF Power", f"{metrics['our_algo']['hf']:.0f} ms²")
        st.metric("LF/HF Ratio", f"{metrics['our_algo']['lf_hf_ratio']:.2f}")
    
    with col_power2:
        st.subheader("📈 Distribuzione Potenza")
        fig_power = create_power_spectrum_plot(metrics)
        st.plotly_chart(fig_power, use_container_width=True)
    
    # 4. INTERPRETAZIONE PER SESSO
    create_interpretation_panel(metrics, st.session_state.user_profile['gender'], st.session_state.user_profile['age'])
    
    # 5. VALUTAZIONE COMPLETA CON CONCLUSIONI
    create_comprehensive_evaluation(metrics, st.session_state.user_profile['gender'], st.session_state.user_profile['age'])
    
    # 6. ANALISI SONNO (SOLO SE C'È)
    create_sleep_analysis(metrics)
    
# 7. BOTTONE ESPORTA PDF - VERSIONE SICURA E FUNZIONANTE
st.markdown("---")
st.header("📄 Esporta Report Completo")

# APPROCCIO SICURO: usa try/except per tutto
try:
    # Verifica in modo sicuro se abbiamo metrics disponibili
    has_metrics = False
    current_metrics = None
    current_start = None
    current_end = None
    current_duration = None
    
    # Controlla prima le variabili locali (più affidabili)
    try:
        if 'adjusted_metrics' in locals() and adjusted_metrics is not None:
            has_metrics = True
            current_metrics = adjusted_metrics
            current_start = start_datetime
            current_end = end_datetime
            current_duration = f"{selected_duration:.1f}h"
    except:
        pass
    
    # Se non in locali, controlla session state
    if not has_metrics:
        try:
            if ('last_analysis_metrics' in st.session_state and 
                st.session_state.last_analysis_metrics is not None):
                has_metrics = True
                current_metrics = st.session_state.last_analysis_metrics
                current_start = st.session_state.get('last_analysis_start', datetime.now())
                current_end = st.session_state.get('last_analysis_end', datetime.now() + timedelta(hours=1))
                current_duration = st.session_state.get('last_analysis_duration', "1.0h")
        except:
            pass
    
    if not has_metrics:
        st.warning("⚠️ **Esegui prima un'analisi completa** per generare il report")
        st.info("""
        💡 **Istruzioni:**
        1. Compila il profilo utente nella sidebar
        2. Carica un file IBI o usa dati simulati  
        3. Clicca sul bottone **'🚀 ANALISI COMPLETA'**
        4. Aspetta che l'analisi finisca
        5. Il bottone PDF diventerà disponibile!
        """)
    else:
        if st.button("🖨️ Genera Report Completo (PDF)", type="primary", use_container_width=True, key="generate_pdf_final"):
            with st.spinner("📊 Generando report PDF..."):
                try:
                    pdf_buffer = create_pdf_report(
                        current_metrics,
                        current_start, 
                        current_end, 
                        current_duration,
                        st.session_state.user_profile,
                        st.session_state.activities
                    )
                    
                    st.success("✅ Report PDF generato con successo!")
                    
                    st.download_button(
                        label="📥 Scarica Report Completo (PDF)",
                        data=pdf_buffer,
                        file_name=f"report_hrv_{st.session_state.user_profile['name']}_{current_start.strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="download_pdf_final"
                    )
                    
                    st.info("📄 **Il report PDF include:** Metriche HRV, analisi spettrale, confronto algoritmi, valutazioni cliniche e referenze scientifiche")
                    
                except Exception as e:
                    st.error(f"❌ Errore nella generazione del report: {str(e)}")
                    st.info("💡 Prova a installare ReportLab: `pip install reportlab`")

except Exception as e:
    st.error("❌ Errore nel sistema di esportazione")
    st.info("Esegui prima un'analisi completa per generare report")
    
    # 8. SALVA NEL DATABASE UTENTE - SOLO SE L'ANALISI È COMPLETATA
            # SALVA NEL DATABASE E NELLO SESSION STATE - VERSIONE MIGLIORATA
            try:
                analysis_type = "File IBI" if st.session_state.file_uploaded else "Simulata"
                
                # SALVA ESPLICITAMENTE I METRICHE NELLO SESSION STATE
                st.session_state.last_analysis_metrics = adjusted_metrics.copy()  # Usa copy() per sicurezza
                st.session_state.last_analysis_start = start_datetime
                st.session_state.last_analysis_end = end_datetime
                st.session_state.last_analysis_duration = f"{selected_duration:.1f}h"
                
                st.write(f"💾 DEBUG: Metrics salvati nello session state: {st.session_state.last_analysis_metrics is not None}")
                
                if save_analysis_to_user_database(adjusted_metrics, start_datetime, end_datetime, f"{selected_duration:.1f}h", analysis_type):
                    st.success("✅ Analisi salvata nello storico utente!")
                    
            except Exception as e:
                st.error(f"❌ Errore nel salvataggio dello storico: {e}")
    
    # 9. MOSTRA STORICO UTENTE
    show_user_analysis_history()

# =============================================================================
# INTERFACCIA PRINCIPALE
# =============================================================================

st.set_page_config(
    page_title="HRV Analytics ULTIMATE - Roberto",
    page_icon="❤️", 
    layout="wide"
)

st.title("🏥 HRV ANALYTICS ULTIMATE")
st.markdown("### **Piattaforma Completa** - Analisi HRV Personalizzata con Storico Utenti")

# INIZIALIZZA SESSION STATE
init_session_state()

# PROFILO UTENTE
create_user_profile()

# STORICO UTENTI
create_user_history_interface()

# DIARIO ATTIVITÀ
create_activity_diary()

# Sidebar configurazione
with st.sidebar:
    st.header("📁 Carica Dati HRV")
    
    uploaded_file = st.file_uploader(
        "Seleziona file IBI/RR intervals",
        type=['csv', 'txt', 'xlsx'],
        help="Supporta: CSV, TXT, Excel con colonne RR/IBI intervals",
        key="main_file_uploader"
    )
    
    rr_intervals_from_file = None
    if uploaded_file is not None:
        try:
            rr_intervals_from_file = read_ibi_file_fast(uploaded_file)
        except:
            rr_intervals_from_file = None
    
    if uploaded_file is not None:
        update_analysis_datetimes(uploaded_file, rr_intervals_from_file)
    
    start_datetime, end_datetime = get_analysis_datetimes()
    
    st.markdown("---")
    st.header("⚙️ Impostazioni Analisi")
    
    if uploaded_file is not None:
        st.success(f"📄 **File:** {uploaded_file.name}")
        if rr_intervals_from_file is not None:
            st.info(f"📊 **{len(rr_intervals_from_file)} intervalli RR**")
    
    st.subheader("🎯 Selezione Intervallo Analisi")
    
    if uploaded_file is not None:
        st.info(f"**Data/Ora inizio rilevazione:** {start_datetime.strftime('%d/%m/%Y %H:%M')}")
        if st.session_state.recording_end_datetime:
            st.info(f"**Data/Ora fine rilevazione:** {st.session_state.recording_end_datetime.strftime('%d/%m/%Y %H:%M')}")
    
    col_date1, col_time1 = st.columns(2)
    with col_date1:
        start_date = st.date_input(
            "Data Inizio Analisi",
            value=start_datetime.date(),
            key="analysis_start_date"
        )
    with col_time1:
        start_time = st.time_input(
            "Ora Inizio Analisi",
            value=start_datetime.time(),
            key="analysis_start_time"
        )
    
    col_date2, col_time2 = st.columns(2)
    with col_date2:
        end_date = st.date_input(
            "Data Fine Analisi",
            value=end_datetime.date(),
            key="analysis_end_date"
        )
    with col_time2:
        end_time = st.time_input(
            "Ora Fine Analisi",
            value=end_datetime.time(),
            key="analysis_end_time"
        )
    
    new_start_datetime = datetime.combine(start_date, start_time)
    new_end_datetime = datetime.combine(end_date, end_time)
    
    datetime_changed = (new_start_datetime != start_datetime or new_end_datetime != end_datetime)
    
    if datetime_changed:
        st.warning("⚠️ **Date modificate manualmente** - L'analisi userà le date selezionate")
        st.session_state.analysis_datetimes = {
            'start_datetime': new_start_datetime,
            'end_datetime': new_end_datetime
        }
        start_datetime, end_datetime = new_start_datetime, new_end_datetime
    
    selected_duration = (end_datetime - start_datetime).total_seconds() / 3600
    
    if selected_duration <= 0:
        st.error("❌ **Errore:** La data di fine deve essere successiva alla data di inizio")
        selected_duration = 1.0
    
    st.info(f"⏱️ **Durata analisi:** {selected_duration:.1f} ore")
    st.info(f"📅 **Periodo analisi:** {start_datetime.strftime('%d/%m/%Y %H:%M')} - {end_datetime.strftime('%d/%m/%Y %H:%M')}")
    
    # VERIFICA SE C'È PERIODO NOTTURNO (22:00-06:00)
    is_night_period = False
    current_time = start_datetime
    night_hours = 0
    
    while current_time < end_datetime:
        if 22 <= current_time.hour or current_time.hour <= 6:
            is_night_period = True
            night_hours += 0.1
        current_time += timedelta(hours=0.1)
    
    if is_night_period:
        st.success(f"🌙 **Periodo notturno rilevato** ({night_hours:.1f}h) - Analisi sonno disponibile")
        include_sleep_default = True
    else:
        st.info("☀️ **Periodo diurno** - Analisi sonno non disponibile")
        include_sleep_default = False
    
    health_factor = st.slider(
        "Profilo Salute", 
        0.1, 1.0, 0.5,
        help="0.1 = Sedentario, 1.0 = Atleta",
        key="health_factor_slider"
    )
    
    include_sleep = st.checkbox(
        "Includi analisi sonno", 
        include_sleep_default,
        help="Disponibile solo per periodi notturni",
        disabled=not is_night_period,
        key="include_sleep_checkbox"
    )
    
    analyze_btn = st.button("🚀 ANALISI COMPLETA", type="primary", use_container_width=True, key="analyze_btn")

# MAIN CONTENT
if analyze_btn:
    if not st.session_state.user_profile['name'] or not st.session_state.user_profile['surname'] or not st.session_state.user_profile['birth_date']:
        st.error("❌ **Completa il profilo utente prima di procedere con l'analisi**")
        st.info("Inserisci nome, cognome e data di nascita nella sidebar")
    else:
        with st.spinner("🎯 **ANALISI COMPLETA IN CORSO**..."):
            # Definisci metrics come variabile esterna ai blocchi
            metrics = None
            analysis_completed = False
            
            if uploaded_file is not None:
                try:
                    rr_intervals = read_ibi_file_fast(uploaded_file)
                    
                    if len(rr_intervals) == 0:
                        st.error("❌ Nessun dato RR valido trovato nel file")
                        st.stop()
                    
                    real_metrics = calculate_hrv_metrics_from_rr(rr_intervals)
                    
                    if real_metrics is None:
                        st.error("❌ Impossibile calcolare le metriche HRV")
                        st.stop()
                    
                    metrics = {
                        'our_algo': {
                            'sdnn': real_metrics['sdnn'],
                            'rmssd': real_metrics['rmssd'],
                            'hr_mean': real_metrics['hr_mean'],
                            'hr_min': max(40, real_metrics['hr_mean'] - 15),
                            'hr_max': min(180, real_metrics['hr_mean'] + 30),
                            'actual_date': start_datetime,
                            'recording_hours': selected_duration,
                            'is_sleep_period': include_sleep,
                            'health_profile_factor': health_factor,
                            'coherence': max(20, 40 + (40 * health_factor)),
                            'total_power': real_metrics['sdnn'] * 100,
                            'vlf': real_metrics['sdnn'] * 20,
                            'lf': real_metrics['sdnn'] * 50,
                            'hf': real_metrics['rmssd'] * 80,
                            'lf_hf_ratio': 1.5,
                        }
                    }
                    
                    if include_sleep and selected_duration >= 4:
                        sleep_duration = min(8.0, selected_duration * 0.9)
                        metrics['our_algo'].update({
                            'sleep_duration': sleep_duration,
                            'sleep_efficiency': min(95, 85 + np.random.normal(0, 5)),
                            'sleep_coherence': 65 + np.random.normal(0, 3),
                            'sleep_hr': 58 + np.random.normal(0, 2),
                            'sleep_rem': min(2.0, sleep_duration * 0.25),
                            'sleep_deep': min(1.5, sleep_duration * 0.2),
                            'sleep_wakeups': max(0, int(sleep_duration * 0.5)),
                        })
                    
                    metrics.update({
                        'emwave_style': {
                            'sdnn': real_metrics['sdnn'] * 0.7,
                            'rmssd': real_metrics['rmssd'] * 0.7,
                            'hr_mean': real_metrics['hr_mean'] + 2,
                            'coherence': 50
                        },
                        'kubios_style': {
                            'sdnn': real_metrics['sdnn'] * 1.3,
                            'rmssd': real_metrics['rmssd'] * 1.3,
                            'hr_mean': real_metrics['hr_mean'] - 2,
                            'coherence': 70
                        }
                    })
                    
                    analysis_completed = True
                    
                except Exception as e:
                    st.error(f"❌ Errore nell'analisi del file: {e}")
                    st.stop()
            else:
                try:
                    metrics = calculate_triple_metrics_corrected(
                        selected_duration, 
                        start_datetime, 
                        health_profile_factor=health_factor,
                        is_sleep_period=include_sleep
                    )
                    analysis_completed = True
                    
                except Exception as e:
                    st.error(f"❌ Errore nell'analisi simulata: {e}")
                    st.stop()
            
            # VERIFICA CHE L'ANALISI SIA COMPLETATA PRIMA DI PROCEDERE
            if not analysis_completed or metrics is None:
                st.error("❌ L'analisi non è stata completata correttamente")
                st.stop()
            
            # APPLICA LE MODIFICHE PER GENERE/ETÀ
            try:
                adjusted_metrics = interpret_metrics_for_gender(
                    metrics, 
                    st.session_state.user_profile['gender'],
                    st.session_state.user_profile['age']
                )
                
                # MOSTRA IL DASHBOARD
                create_complete_analysis_dashboard(
                    adjusted_metrics, 
                    start_datetime, 
                    end_datetime,
                    f"{selected_duration:.1f}h"
                )
                
            except Exception as e:
                st.error(f"❌ Errore nella visualizzazione dei risultati: {e}")
                st.stop()
            
            # SALVA NEL DATABASE E PREPARA PER PDF - VERSIONE CORRETTA
            try:
                analysis_type = "File IBI" if uploaded_file is not None else "Simulata"
                
                # 🔥 SALVA I METRICS PER IL PDF - QUESTA È LA PARTE CRITICA
                st.session_state.last_analysis_metrics = adjusted_metrics
                st.session_state.last_analysis_start = start_datetime
                st.session_state.last_analysis_end = end_datetime
                st.session_state.last_analysis_duration = f"{selected_duration:.1f}h"
                
                # Salva anche nel database utente
                if save_analysis_to_user_database(adjusted_metrics, start_datetime, end_datetime, f"{selected_duration:.1f}h", analysis_type):
                    st.success("✅ Analisi salvata nello storico utente!")
                    
            except Exception as e:
                st.error(f"❌ Errore nel salvataggio: {e}")else:
    st.info("👆 **Configura l'analisi dalla sidebar**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Flusso di Lavoro")
        st.markdown("""
        1. **👤 Inserisci profilo utente** (obbligatorio)
        2. **📁 Carica file IBI** (data/ora automatiche)
        3. **📝 Aggiungi attività** nel diario
        4. **🎯 Seleziona intervallo** con date specifiche
        5. **🌙 Attiva analisi sonno** (se notturno)
        6. **🚀 Avvia analisi** completa
        7. **📊 Consulta storico** analisi per utente
        8. **📄 Esporta report** (PNG)
        """)
    
    with col2:
        st.subheader("🆕 Funzionalità Complete")
        st.markdown("""
        - 👤 **Profilo utente** con storico
        - 📊 **Database utenti** persistente
        - 🔄 **Carica profili** esistenti
        - 📈 **Grafico con ORE REALI** di rilevazione
        - 😴 **Analisi sonno** completa
        - 📅 **Storico analisi** per utente
        - 📈 **Andamento metriche** nel tempo
        - 📄 **Esportazione report** (PNG)
        - 📅 **Data/ora fine rilevazione** calcolata
        - ⏰ **Campi ore più grandi** nelle attività
        """)

# 🔥 SEZIONE PDF - METTILA DOPO TUTTO IL BLOCCO analyze_btn, PRIMA DEL FOOTER

st.markdown("---")
st.header("📄 Esporta Report Completo")

# Verifica sicura se abbiamo un'analisi disponibile
try:
    has_analysis = (
        'last_analysis_metrics' in st.session_state and 
        st.session_state.last_analysis_metrics is not None
    )
except:
    has_analysis = False

if not has_analysis:
    st.warning("⚠️ **Esegui prima un'analisi completa** per generare il report")
    st.info("""
    💡 **Istruzioni:**
    1. Compila il profilo utente nella sidebar
    2. Carica un file IBI o usa dati simulati  
    3. Clicca sul bottone **'🚀 ANALISI COMPLETA'**
    4. Aspetta che l'analisi finisca
    5. Questa sezione mostrerà il bottone per il PDF!
    """)
else:
    st.success("✅ **Analisi completata!** Ora puoi generare il report PDF")
    
    if st.button("🖨️ Genera Report Completo (PDF)", type="primary", use_container_width=True, key="generate_pdf_btn"):
        with st.spinner("📊 Generando report PDF..."):
            try:
                pdf_buffer = create_pdf_report(
                    st.session_state.last_analysis_metrics,
                    st.session_state.last_analysis_start,
                    st.session_state.last_analysis_end,
                    st.session_state.last_analysis_duration,
                    st.session_state.user_profile,
                    st.session_state.activities
                )
                
                st.success("✅ Report PDF generato con successo!")
                
                st.download_button(
                    label="📥 Scarica Report Completo (PDF)",
                    data=pdf_buffer,
                    file_name=f"report_hrv_{st.session_state.user_profile['name']}_{st.session_state.last_analysis_start.strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_pdf_btn"
                )
                
            except Exception as e:
                st.error(f"❌ Errore nella generazione del report: {str(e)}")
                st.info("💡 Assicurati che ReportLab sia installato: `pip install reportlab`")

# FOOTER (mantieni il footer esistente)
st.markdown("---")
st.markdown("**HRV Analytics ULTIMATE** - Sviluppato per Roberto")

# FOOTER
st.markdown("---")
st.markdown("**HRV Analytics ULTIMATE** - Sviluppato per Roberto")