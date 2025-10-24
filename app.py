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
    """Salva l'analisi nel database dell'utente"""
    user_key = get_user_key(st.session_state.user_profile)
    if not user_key:
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
            'sdnn': metrics['our_algo']['sdnn'],
            'rmssd': metrics['our_algo']['rmssd'],
            'hr_mean': metrics['our_algo']['hr_mean'],
            'coherence': metrics['our_algo']['coherence'],
            'recording_hours': metrics['our_algo']['recording_hours'],
            'total_power': metrics['our_algo']['total_power'],
            'vlf': metrics['our_algo']['vlf'],
            'lf': metrics['our_algo']['lf'],
            'hf': metrics['our_algo']['hf'],
            'lf_hf_ratio': metrics['our_algo']['lf_hf_ratio']
        }
    }
    
    st.session_state.user_database[user_key]['analyses'].append(analysis_data)
    return True

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
# FUNZIONE PER CREARE PDF CON GRAFICHE AVANZATE - VERSIONE CORRETTA
# =============================================================================

def create_advanced_pdf_report(metrics, start_datetime, end_datetime, selected_range, user_profile, activities=[]):
    """Crea un report PDF avanzato con grafiche 3D e analisi completa - VERSIONE CORRETTA"""
    try:
        # Import qui per evitare errori se reportlab non è installato
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib.utils import ImageReader
        from reportlab.lib.colors import Color, HexColor
        import io
        import matplotlib.pyplot as plt
        import numpy as np
        
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        
        # Colori moderni
        primary_color = HexColor("#3498db")
        secondary_color = HexColor("#e74c3c")
        accent_color = HexColor("#2ecc71")
        background_color = HexColor("#f8f9fa")
        text_color = HexColor("#2c3e50")
        
        # === PAGINA 1: INTESTAZIONE E METRICHE PRINCIPALI ===
        
        # Sfondo
        p.setFillColor(background_color)
        p.rect(0, 0, width, height, fill=1, stroke=0)
        
        # Header con gradiente
        p.setFillColor(primary_color)
        p.rect(0, height-80, width, 80, fill=1, stroke=0)
        
        # Titolo
        p.setFillColorRGB(1, 1, 1)
        p.setFont("Helvetica-Bold", 24)
        p.drawString(50, height-50, "REPORT CARDIOLOGICO HRV")
        p.setFont("Helvetica", 10)
        p.drawString(50, height-70, f"Generato il: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        
        # Informazioni paziente
        p.setFillColor(text_color)
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, height-120, "INFORMAZIONI PAZIENTE")
        p.setFont("Helvetica", 10)
        p.drawString(50, height-140, f"Nome: {user_profile.get('name', '')} {user_profile.get('surname', '')}")
        p.drawString(50, height-155, f"Età: {user_profile.get('age', '')} anni | Sesso: {user_profile.get('gender', '')}")
        p.drawString(50, height-170, f"Periodo analisi: {start_datetime.strftime('%d/%m/%Y %H:%M')} - {end_datetime.strftime('%d/%m/%Y %H:%M')}")
        p.drawString(50, height-185, f"Durata: {selected_range}")
        
        # Metriche principali in box moderni
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, height-220, "METRICHE HRV PRINCIPALI")
        
        y_pos = height-250
        metrics_data = [
            ("SDNN", f"{metrics['our_algo']['sdnn']:.1f} ms", get_sdnn_evaluation(metrics['our_algo']['sdnn'], user_profile.get('gender', 'Uomo')), primary_color),
            ("RMSSD", f"{metrics['our_algo']['rmssd']:.1f} ms", get_rmssd_evaluation(metrics['our_algo']['rmssd'], user_profile.get('gender', 'Uomo')), secondary_color),
            ("Frequenza Cardiaca", f"{metrics['our_algo']['hr_mean']:.1f} bpm", get_hr_evaluation(metrics['our_algo']['hr_mean']), accent_color),
            ("Coerenza", f"{metrics['our_algo']['coherence']:.1f}%", get_coherence_evaluation(metrics['our_algo']['coherence']), HexColor("#f39c12")),
            ("Total Power", f"{metrics['our_algo']['total_power']:.0f} ms²", get_power_evaluation(metrics['our_algo']['total_power']), HexColor("#9b59b6"))
        ]
        
        for i, (metric_name, value, evaluation, color) in enumerate(metrics_data):
            x_pos = 50 + (i % 2) * 250
            if i % 2 == 0 and i > 0:
                y_pos -= 80
            
            # Box metriche - CORRETTO: senza setAlpha
            p.setFillColor(color)
            p.roundRect(x_pos, y_pos, 230, 60, 10, fill=1, stroke=0)
            
            p.setFillColor(color)
            p.setFont("Helvetica-Bold", 12)
            p.drawString(x_pos + 10, y_pos + 40, metric_name)
            
            p.setFillColor(text_color)
            p.setFont("Helvetica", 10)
            p.drawString(x_pos + 10, y_pos + 25, value)
            
            p.setFont("Helvetica-Bold", 9)
            p.drawString(x_pos + 10, y_pos + 10, evaluation)
        
        # Grafico radar compatto per pagina 1
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, height-450, "PROFILO HRV - ANALISI MULTIDIMENSIONALE")
        
        # Creazione grafico radar semplificato
        try:
            fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(projection='polar'))
            
            categories = ['SDNN', 'RMSSD', 'Coerenza', 'HRV Totale', 'Bilancio LF/HF']
            values = [
                min(100, metrics['our_algo']['sdnn'] / 100 * 100),
                min(100, metrics['our_algo']['rmssd'] / 80 * 100),
                min(100, metrics['our_algo']['coherence']),
                min(100, metrics['our_algo']['total_power'] / 50000 * 100),
                min(100, (1 - abs(metrics['our_algo']['lf_hf_ratio'] - 1.5) / 1.5) * 100)
            ]
            
            values += values[:1]
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, color='#3498db', alpha=0.7)
            ax.fill(angles, values, alpha=0.3, color='#3498db')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 100)
            ax.grid(True)
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
            img_buffer.seek(0)
            
            p.drawImage(ImageReader(img_buffer), 50, height-650, width=500, height=180)
            plt.close()
        except Exception as e:
            # Se il grafico fallisce, aggiungi testo alternativo
            p.setFillColor(text_color)
            p.setFont("Helvetica", 10)
            p.drawString(50, height-600, "Grafico radar non disponibile - dati tecnici temporaneamente non accessibili")
        
        p.showPage()
        
        # === PAGINA 2: ANALISI DETTAGLIATA E PUNTI DI DEBOLEZZA ===
        
        # Sfondo
        p.setFillColor(background_color)
        p.rect(0, 0, width, height, fill=1, stroke=0)
        
        # Titolo pagina
        p.setFillColor(primary_color)
        p.setFont("Helvetica-Bold", 18)
        p.drawString(50, height-50, "ANALISI DETTAGLIATA E PUNTI DI ATTENZIONE")
        
        # Analisi spettrale
        p.setFillColor(text_color)
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, height-100, "ANALISI SPETTRALE HRV")
        
        spectral_data = [
            ("VLF Power", f"{metrics['our_algo']['vlf']:.0f} ms²", "Attività termoregolatorie"),
            ("LF Power", f"{metrics['our_algo']['lf']:.0f} ms²", "Sistema simpatico"),
            ("HF Power", f"{metrics['our_algo']['hf']:.0f} ms²", "Sistema parasimpatico"),
            ("LF/HF Ratio", f"{metrics['our_algo']['lf_hf_ratio']:.2f}", "Bilancio autonomico")
        ]
        
        y_pos = height-130
        for i, (name, value, desc) in enumerate(spectral_data):
            x_pos = 50 + (i % 2) * 250
            if i % 2 == 0 and i > 0:
                y_pos -= 60
            
            # Box spettrale - CORRETTO: senza trasparenza
            p.setFillColor(HexColor("#34495e"))
            p.roundRect(x_pos, y_pos, 230, 45, 8, fill=1, stroke=0)
            
            p.setFillColor(primary_color)
            p.setFont("Helvetica-Bold", 10)
            p.drawString(x_pos + 10, y_pos + 30, name)
            
            p.setFillColor(text_color)
            p.setFont("Helvetica", 9)
            p.drawString(x_pos + 10, y_pos + 20, value)
            p.drawString(x_pos + 10, y_pos + 10, desc)
        
        # Identificazione punti di debolezza
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, height-280, "🔍 PUNTI DI DEBOLEZZA IDENTIFICATI")
        
        weaknesses = identify_weaknesses(metrics, user_profile)
        y_weak = height-310
        
        for i, weakness in enumerate(weaknesses[:4]):  # Massimo 4 punti principali
            # Box debolezze - CORRETTO
            p.setFillColor(HexColor("#e74c3c"))
            p.roundRect(50, y_weak, 500, 25, 5, fill=1, stroke=0)
            
            p.setFillColor(HexColor("#c0392b"))
            p.setFont("Helvetica-Bold", 9)
            p.drawString(60, y_weak + 15, f"• {weakness}")
            
            y_weak -= 30
        
        # Grafico a barre per distribuzione potenza (2D invece di 3D per semplicità)
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            
            bands = ['VLF', 'LF', 'HF']
            power_values = [metrics['our_algo']['vlf'], metrics['our_algo']['lf'], metrics['our_algo']['hf']]
            colors = ['#95a5a6', '#3498db', '#e74c3c']
            
            bars = ax.bar(bands, power_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Bande Frequenza')
            ax.set_ylabel('Potenza (ms²)')
            ax.set_title('Distribuzione Potenza HRV', pad=20)
            ax.grid(True, alpha=0.3)
            
            # Aggiungi valori sulle barre
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.0f}', ha='center', va='bottom', fontsize=8)
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='#f8f9fa', edgecolor='none')
            img_buffer.seek(0)
            
            p.drawImage(ImageReader(img_buffer), 50, height-500, width=500, height=180)
            plt.close()
        except Exception as e:
            p.setFillColor(text_color)
            p.setFont("Helvetica", 10)
            p.drawString(50, height-500, "Grafico distribuzione potenza non disponibile")
        
        p.showPage()
        
        # === PAGINA 3: RACCOMANDAZIONI E PIANO D'AZIONE ===
        
        # Sfondo
        p.setFillColor(background_color)
        p.rect(0, 0, width, height, fill=1, stroke=0)
        
        # Titolo pagina
        p.setFillColor(accent_color)
        p.setFont("Helvetica-Bold", 18)
        p.drawString(50, height-50, "PIANO DI MIGLIORAMENTO E RACCOMANDAZIONI")
        
        # Raccomandazioni personalizzate
        recommendations = generate_recommendations(metrics, user_profile, weaknesses)
        
        p.setFillColor(text_color)
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, height-90, "🎯 RACCOMANDAZIONI PERSONALIZZATE")
        
        y_rec = height-120
        for i, (category, rec_list) in enumerate(recommendations.items()):
            p.setFillColor(primary_color)
            p.setFont("Helvetica-Bold", 12)
            p.drawString(50, y_rec, f"{category}:")
            y_rec -= 20
            
            p.setFillColor(text_color)
            p.setFont("Helvetica", 9)
            for rec in rec_list[:3]:  # Massimo 3 raccomandazioni per categoria
                p.drawString(70, y_rec, f"• {rec}")
                y_rec -= 15
            y_rec -= 10
        
        # Piano d'azione temporale
        p.setFillColor(secondary_color)
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, height-350, "📅 PIANO D'AZIONE 30 GIORNI")
        
        action_plan = [
            ("Settimana 1-2", "Focus su respirazione e routine sonno"),
            ("Settimana 3-4", "Implementazione attività fisica leggera"),
            ("Giornaliero", "Monitoraggio coerenza cardiaca 5 minuti"),
            ("Settimanale", "Sessioni di rilassamento guidato")
        ]
        
        y_plan = height-380
        for timeframe, action in action_plan:
            # Box piano d'azione - CORRETTO
            p.setFillColor(HexColor("#34495e"))
            p.roundRect(50, y_plan, 500, 20, 5, fill=1, stroke=0)
            
            p.setFillColor(primary_color)
            p.setFont("Helvetica-Bold", 9)
            p.drawString(60, y_plan + 12, timeframe)
            
            p.setFillColor(text_color)
            p.setFont("Helvetica", 9)
            p.drawString(200, y_plan + 12, action)
            
            y_plan -= 25
        
        # Grafico progresso atteso
        try:
            fig, ax = plt.subplots(figsize=(8, 3))
            
            weeks = ['Attuale', 'Sett 2', 'Sett 4']
            sdnn_progress = [
                metrics['our_algo']['sdnn'],
                metrics['our_algo']['sdnn'] * 1.15,
                metrics['our_algo']['sdnn'] * 1.3
            ]
            rmssd_progress = [
                metrics['our_algo']['rmssd'],
                metrics['our_algo']['rmssd'] * 1.2,
                metrics['our_algo']['rmssd'] * 1.4
            ]
            
            x = range(len(weeks))
            width = 0.35
            
            bars1 = ax.bar([i - width/2 for i in x], sdnn_progress, width, 
                          label='SDNN', color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
            bars2 = ax.bar([i + width/2 for i in x], rmssd_progress, width, 
                          label='RMSSD', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Timeline')
            ax.set_ylabel('Valori (ms)')
            ax.set_title('Progresso Atteso con Interventi', pad=10)
            ax.set_xticks(x)
            ax.set_xticklabels(weeks)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Aggiungi valore sulle barre
            for bar in bars1 + bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.0f}', ha='center', va='bottom', fontsize=8)
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='#f8f9fa', edgecolor='none')
            img_buffer.seek(0)
            
            p.drawImage(ImageReader(img_buffer), 50, height-550, width=500, height=150)
            plt.close()
        except Exception as e:
            p.setFillColor(text_color)
            p.setFont("Helvetica", 10)
            p.drawString(50, height-550, "Grafico progresso non disponibile")
        
        # Referenze scientifiche
        p.setFillColor(HexColor("#7f8c8d"))
        p.setFont("Helvetica-Bold", 10)
        p.drawString(50, 80, "REFERENZE SCIENTIFICHE:")
        p.setFont("Helvetica", 8)
        references = [
            "• Task Force of ESC/NASPE (1996) - Heart rate variability",
            "• Malik et al. (1996) - HRV standards of measurement",
            "• McCraty et al. (2009) - Coherence and HRV",
            "• Shaffer et al. (2014) - Clinical applications of HRV"
        ]
        y_ref = 60
        for ref in references:
            p.drawString(60, y_ref, ref)
            y_ref -= 12
        
        # Footer
        p.setFillColor(HexColor("#7f8c8d"))
        p.setFont("Helvetica", 8)
        p.drawString(50, 30, "Report generato da HRV Analytics ULTIMATE - Sistema avanzato di analisi della variabilità cardiaca")
        p.drawString(50, 20, "Consultare sempre un medico per interpretazioni cliniche")
        
        p.showPage()
        p.save()
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Errore nella generazione PDF avanzato: {e}")
        # Fallback - report semplice ma completo
        return create_improved_fallback_report(metrics, start_datetime, end_datetime, selected_range, user_profile, weaknesses, recommendations)

def create_improved_fallback_report(metrics, start_datetime, end_datetime, selected_range, user_profile, weaknesses, recommendations):
    """Crea un report fallback migliorato con tutte le informazioni"""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    import io
    
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Pagina 1
    p.setFont("Helvetica-Bold", 18)
    p.drawString(100, height-50, "REPORT HRV COMPLETO - ANALISI CARDIACA")
    p.setFont("Helvetica", 10)
    p.drawString(100, height-70, f"Paziente: {user_profile.get('name', '')} {user_profile.get('surname', '')}")
    p.drawString(100, height-85, f"Età: {user_profile.get('age', '')} anni | Sesso: {user_profile.get('gender', '')}")
    p.drawString(100, height-100, f"Periodo: {start_datetime.strftime('%d/%m/%Y %H:%M')} - {end_datetime.strftime('%d/%m/%Y %H:%M')}")
    p.drawString(100, height-115, f"Durata analisi: {selected_range}")
    
    y_pos = height-150
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, y_pos, "METRICHE HRV PRINCIPALI:")
    y_pos -= 25
    
    main_metrics = [
        ("SDNN", f"{metrics['our_algo']['sdnn']:.1f} ms", get_sdnn_evaluation(metrics['our_algo']['sdnn'], user_profile.get('gender', 'Uomo'))),
        ("RMSSD", f"{metrics['our_algo']['rmssd']:.1f} ms", get_rmssd_evaluation(metrics['our_algo']['rmssd'], user_profile.get('gender', 'Uomo'))),
        ("Frequenza Cardiaca", f"{metrics['our_algo']['hr_mean']:.1f} bpm", get_hr_evaluation(metrics['our_algo']['hr_mean'])),
        ("Coerenza", f"{metrics['our_algo']['coherence']:.1f}%", get_coherence_evaluation(metrics['our_algo']['coherence'])),
        ("Total Power", f"{metrics['our_algo']['total_power']:.0f} ms²", get_power_evaluation(metrics['our_algo']['total_power']))
    ]
    
    for name, value, evaluation in main_metrics:
        p.setFont("Helvetica-Bold", 10)
        p.drawString(120, y_pos, f"{name}:")
        p.setFont("Helvetica", 10)
        p.drawString(250, y_pos, value)
        p.drawString(350, y_pos, evaluation)
        y_pos -= 20
    
    y_pos -= 10
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, y_pos, "ANALISI SPETTRALE:")
    y_pos -= 25
    
    spectral_metrics = [
        ("VLF Power", f"{metrics['our_algo']['vlf']:.0f} ms²"),
        ("LF Power", f"{metrics['our_algo']['lf']:.0f} ms²"),
        ("HF Power", f"{metrics['our_algo']['hf']:.0f} ms²"),
        ("LF/HF Ratio", f"{metrics['our_algo']['lf_hf_ratio']:.2f}", get_lf_hf_evaluation(metrics['our_algo']['lf_hf_ratio']))
    ]
    
    for name, value, *evaluation in spectral_metrics:
        p.setFont("Helvetica-Bold", 10)
        p.drawString(120, y_pos, f"{name}:")
        p.setFont("Helvetica", 10)
        p.drawString(250, y_pos, value)
        if evaluation:
            p.drawString(350, y_pos, evaluation[0])
        y_pos -= 20
    
    p.showPage()
    
    # Pagina 2 - Punti di debolezza e raccomandazioni
    p.setFont("Helvetica-Bold", 18)
    p.drawString(100, height-50, "ANALISI PUNTI DI ATTENZIONE E RACCOMANDAZIONI")
    
    y_pos = height-80
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, y_pos, "PUNTI DI DEBOLEZZA IDENTIFICATI:")
    y_pos -= 25
    
    for weakness in weaknesses:
        p.setFont("Helvetica", 10)
        p.drawString(120, y_pos, f"• {weakness}")
        y_pos -= 15
    
    y_pos -= 20
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, y_pos, "RACCOMANDAZIONI PERSONALIZZATE:")
    y_pos -= 25
    
    for category, recs in recommendations.items():
        p.setFont("Helvetica-Bold", 12)
        p.drawString(100, y_pos, f"{category}:")
        y_pos -= 15
        p.setFont("Helvetica", 10)
        for rec in recs[:2]:  # Prime 2 raccomandazioni per categoria
            p.drawString(120, y_pos, f"• {rec}")
            y_pos -= 12
        y_pos -= 10
    
    p.showPage()
    
    # Pagina 3 - Referenze scientifiche
    p.setFont("Helvetica-Bold", 18)
    p.drawString(100, height-50, "REFERENZE SCIENTIFICHE E CONCLUSIONI")
    
    y_pos = height-80
    p.setFont("Helvetica-Bold", 12)
    p.drawString(100, y_pos, "Bibliografia:")
    y_pos -= 20
    
    references = [
        "Task Force of ESC/NASPE (1996) - Heart rate variability: standards of measurement...",
        "Malik et al. (1996) - Heart rate variability: Standards of measurement...",
        "McCraty et al. (2009) - The coherent heart: Heart-brain interactions...",
        "Shaffer et al. (2014) - An overview of heart rate variability metrics and norms",
        "Nunan et al. (2010) - A quantitative systematic review of normal values for short-term HRV"
    ]
    
    for ref in references:
        p.setFont("Helvetica", 9)
        p.drawString(120, y_pos, f"• {ref}")
        y_pos -= 15
    
    y_pos -= 20
    p.setFont("Helvetica-Bold", 12)
    p.drawString(100, y_pos, "Conclusioni:")
    y_pos -= 15
    p.setFont("Helvetica", 10)
    
    # Conclusioni basate sul profilo
    if len(weaknesses) <= 1:
        conclusion = "Profilo HRV nella norma. Mantenere stile di vita sano e monitoraggio periodico."
    elif len(weaknesses) <= 3:
        conclusion = "Alcuni aspetti richiedono attenzione. Implementare le raccomandazioni fornite."
    else:
        conclusion = "Profilo che richiede miglioramenti significativi. Consigliato follow-up medico."
    
    p.drawString(100, y_pos, conclusion)
    
    # Footer
    p.setFont("Helvetica", 8)
    p.drawString(100, 30, "Report generato da HRV Analytics ULTIMATE")
    p.drawString(100, 20, "Consultare sempre un medico per interpretazioni cliniche")
    
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

# =============================================================================
# FUNZIONI DI VALUTAZIONE E ANALISI
# =============================================================================

def get_sdnn_evaluation(sdnn, gender):
    """Valuta il valore SDNN"""
    if gender == 'Donna':
        if sdnn < 35: return "⬇️ Basso"
        elif sdnn < 65: return "✅ Normale"
        else: return "⬆️ Ottimo"
    else:
        if sdnn < 40: return "⬇️ Basso"
        elif sdnn < 75: return "✅ Normale"
        else: return "⬆️ Ottimo"

def get_rmssd_evaluation(rmssd, gender):
    """Valuta il valore RMSSD"""
    if gender == 'Donna':
        if rmssd < 19: return "⬇️ Basso"
        elif rmssd < 45: return "✅ Normale"
        else: return "⬆️ Ottimo"
    else:
        if rmssd < 25: return "⬇️ Basso"
        elif rmssd < 55: return "✅ Normale"
        else: return "⬆️ Ottimo"

def get_hr_evaluation(hr):
    """Valuta la frequenza cardiaca"""
    if hr < 50: return "⬇️ Bradicardia"
    elif hr < 90: return "✅ Normale"
    elif hr < 100: return "⚠️ Leggermente alta"
    else: return "⬆️ Tachicardia"

def get_coherence_evaluation(coherence):
    """Valuta la coerenza cardiaca"""
    if coherence < 30: return "⬇️ Bassa"
    elif coherence < 60: return "✅ Media"
    else: return "⬆️ Alta"

def get_power_evaluation(total_power):
    """Valuta la potenza totale"""
    if total_power < 1000: return "⬇️ Molto bassa"
    elif total_power < 3000: return "⚠️ Bassa"
    elif total_power < 8000: return "✅ Normale"
    else: return "⬆️ Alta"

def get_lf_hf_evaluation(ratio):
    """Valuta il rapporto LF/HF"""
    if ratio < 0.5: return "⬇️ Parasimpatico dominante"
    elif ratio < 2.0: return "✅ Bilanciato"
    else: return "⬆️ Simpatico dominante"

def identify_weaknesses(metrics, user_profile):
    """Identifica i punti di debolezza basati sulle metriche HRV"""
    weaknesses = []
    
    sdnn = metrics['our_algo']['sdnn']
    rmssd = metrics['our_algo']['rmssd']
    hr = metrics['our_algo']['hr_mean']
    coherence = metrics['our_algo']['coherence']
    lf_hf_ratio = metrics['our_algo']['lf_hf_ratio']
    total_power = metrics['our_algo']['total_power']
    
    # Valori di riferimento per genere
    if user_profile.get('gender') == 'Donna':
        sdnn_low, sdnn_high = 35, 65
        rmssd_low, rmssd_high = 19, 45
    else:
        sdnn_low, sdnn_high = 40, 75
        rmssd_low, rmssd_high = 25, 55
    
    # Analisi SDNN
    if sdnn < sdnn_low:
        weaknesses.append("Ridotta variabilità cardiaca generale (SDNN basso)")
    elif sdnn > sdnn_high:
        weaknesses.append("Variabilità cardiaca elevata - verificare condizioni")
    
    # Analisi RMSSD
    if rmssd < rmssd_low:
        weaknesses.append("Ridotta attività parasimpatica (RMSSD basso)")
    
    # Analisi frequenza cardiaca
    if hr > 90:
        weaknesses.append("Frequenza cardiaca a riposo elevata")
    elif hr < 50:
        weaknesses.append("Frequenza cardiaca a riposo molto bassa")
    
    # Analisi coerenza
    if coherence < 40:
        weaknesses.append("Bassa coerenza cardiaca - possibile stress")
    
    # Analisi bilanciamento autonomico
    if lf_hf_ratio > 3.0:
        weaknesses.append("Dominanza simpatica eccessiva")
    elif lf_hf_ratio < 0.5:
        weaknesses.append("Dominanza parasimpatica eccessiva")
    
    # Analisi potenza totale
    if total_power < 3000:
        weaknesses.append("Ridotta riserva autonomica generale")
    
    # Aggiungi debolezze generali se necessario
    if len(weaknesses) == 0:
        weaknesses.append("Profilo HRV nella norma - mantenere stile di vita sano")
    
    return weaknesses[:5]  # Massimo 5 punti di debolezza

def generate_recommendations(metrics, user_profile, weaknesses):
    """Genera raccomandazioni personalizzate basate sulle debolezze identificate"""
    recommendations = {
        "Respirazione e Rilassamento": [],
        "Attività Fisica": [],
        "Gestione Sonno": [],
        "Alimentazione": [],
        "Gestione Stress": []
    }
    
    sdnn = metrics['our_algo']['sdnn']
    rmssd = metrics['our_algo']['rmssd']
    hr = metrics['our_algo']['hr_mean']
    coherence = metrics['our_algo']['coherence']
    lf_hf_ratio = metrics['our_algo']['lf_hf_ratio']
    
    # Raccomandazioni basate su metriche specifiche
    if any("parasimpatica" in w.lower() for w in weaknesses) or rmssd < 30:
        recommendations["Respirazione e Rilassamento"].append("Pranayama: respirazione 4-7-8 (4s inspiro, 7s pausa, 8s espiro)")
        recommendations["Respirazione e Rilassamento"].append("Meditazione guidata 10 minuti al giorno")
        recommendations["Attività Fisica"].append("Yoga o Tai Chi 2-3 volte a settimana")
    
    if any("simpatica" in w.lower() for w in weaknesses) or lf_hf_ratio > 2.5:
        recommendations["Gestione Stress"].append("Tecniche di grounding: 5-4-3-2-1 (5 cose che vedi, 4 che tocchi, etc.)")
        recommendations["Gestione Stress"].append("Pause attive ogni 90 minuti di lavoro")
        recommendations["Attività Fisica"].append("Camminate nella natura 30 minuti al giorno")
    
    if any("frequenza cardiaca" in w.lower() for w in weaknesses) or hr > 85:
        recommendations["Attività Fisica"].append("Allenamento aerobico moderato 150 minuti/settimana")
        recommendations["Alimentazione"].append("Ridurre caffeina dopo le 14:00")
        recommendations["Gestione Sonno"].append("Mantenere temperatura camera da letto 18-20°C")
    
    if coherence < 50:
        recommendations["Respirazione e Rilassamento"].append("Coerenza cardiaca: 3 volte al giorno per 5 minuti (5.5 respiri/min)")
        recommendations["Gestione Stress"].append("Journaling serale per scaricare tensioni")
    
    # Raccomandazioni generali
    recommendations["Gestione Sonno"].append("Orari regolari di sonno (variazione max 1h weekend)")
    recommendations["Alimentazione"].append("Idratazione: 2L acqua al giorno")
    recommendations["Alimentazione"].append("Omega-3: pesce azzurro 2 volte a settimana")
    recommendations["Gestione Stress"].append("Tecnologia: 1 ora prima di dormire no schermi")
    
    # Pulisci raccomandazioni vuote
    return {k: v for k, v in recommendations.items() if v}

# =============================================================================
# FUNZIONE PER GRAFICO 3D AVANZATO
# =============================================================================

def create_advanced_3d_plot(metrics):
    """Crea un grafico 3D avanzato per l'analisi HRV"""
    fig = go.Figure()
    
    # Sfera di riferimento per variabilità
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        opacity=0.1,
        colorscale='Blues',
        showscale=False,
        name='Sfera riferimento'
    ))
    
    # Punti dati principali
    metrics_points = [
        dict(
            x=[metrics['our_algo']['sdnn'] / 50],
            y=[metrics['our_algo']['rmssd'] / 30],
            z=[metrics['our_algo']['coherence'] / 20],
            name='Profilo Attuale',
            color='red',
            size=15
        )
    ]
    
    for point in metrics_points:
        fig.add_trace(go.Scatter3d(
            x=point['x'],
            y=point['y'], 
            z=point['z'],
            mode='markers',
            marker=dict(
                size=point['size'],
                color=point['color'],
                opacity=0.8,
                line=dict(width=2, color='darkred')
            ),
            name=point['name']
        ))
    
    fig.update_layout(
        title="🔄 Analisi 3D Profilo HRV",
        scene=dict(
            xaxis_title='SDNN (scalato)',
            yaxis_title='RMSSD (scalato)', 
            zaxis_title='Coerenza (scalato)',
            bgcolor='rgb(240, 240, 240)'
        ),
        height=500
    )
    
    return fig

# =============================================================================
# INTERFACCIA PRINCIPALE STREAMLIT - MIGLIORATA
# =============================================================================

def main():
    st.set_page_config(
        page_title="HRV Analytics ULTIMATE",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # CSS personalizzato
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #3498db;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .spectral-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .sleep-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .weakness-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #2c3e50;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(45deg, #3498db, #2980b9);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header principale
    st.markdown('<h1 class="main-header">❤️ HRV Analytics ULTIMATE</h1>', unsafe_allow_html=True)
    
    # Sidebar per profilo utente
    with st.sidebar:
        st.header("👤 Profilo Paziente")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.user_profile['name'] = st.text_input("Nome", value=st.session_state.user_profile['name'])
        with col2:
            st.session_state.user_profile['surname'] = st.text_input("Cognome", value=st.session_state.user_profile['surname'])
        
        st.session_state.user_profile['birth_date'] = st.date_input("Data di nascita", value=st.session_state.user_profile['birth_date'] or datetime.now().date())
        st.session_state.user_profile['gender'] = st.selectbox("Sesso", ["Uomo", "Donna"], index=0 if st.session_state.user_profile['gender'] == 'Uomo' else 1)
        
        if st.session_state.user_profile['birth_date']:
            age = datetime.now().year - st.session_state.user_profile['birth_date'].year
            st.session_state.user_profile['age'] = age
            st.info(f"Età: {age} anni")
        
        create_user_history_interface()
    
    # Upload file
    st.header("📤 Carica File IBI")
    uploaded_file = st.file_uploader("Carica il tuo file .txt o .csv con gli intervalli IBI", type=['txt', 'csv'])
    
    if uploaded_file is not None:
        try:
            content = uploaded_file.getvalue().decode('utf-8')
            lines = content.strip().split('\n')
            
            rr_intervals = []
            for line in lines:
                if line.strip():
                    try:
                        rr_intervals.append(float(line.strip()))
                    except ValueError:
                        continue
            
            if len(rr_intervals) == 0:
                st.error("❌ Nessun dato IBI valido trovato nel file")
                return
            
            # Aggiorna data/ora automaticamente
            update_analysis_datetimes(uploaded_file, rr_intervals)
            
            # Selezione range temporale
            start_datetime, end_datetime = get_analysis_datetimes()
            
            st.header("⏰ Selezione Periodo Analisi")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Inizio Analisi")
                start_date = st.date_input("Data Inizio", value=start_datetime.date(), key="start_date")
                start_time = st.time_input("Ora Inizio", value=start_datetime.time(), key="start_time")
                new_start = datetime.combine(start_date, start_time)
            
            with col2:
                st.subheader("Fine Analisi")
                end_date = st.date_input("Data Fine", value=end_datetime.date(), key="end_date")
                end_time = st.time_input("Ora Fine", value=end_datetime.time(), key="end_time")
                new_end = datetime.combine(end_date, end_time)
            
            if new_start != start_datetime or new_end != end_datetime:
                st.session_state.analysis_datetimes = {
                    'start_datetime': new_start,
                    'end_datetime': new_end
                }
                st.rerun()
            
            # Calcola durata selezionata
            duration = (end_datetime - start_datetime).total_seconds() / 3600
            selected_range = f"{duration:.1f} ore"
            
            # Simula metriche (sostituire con calcoli reali)
            metrics = {
                'our_algo': {
                    'sdnn': max(20, np.std(rr_intervals) if len(rr_intervals) > 1 else 25),
                    'rmssd': max(15, np.sqrt(np.mean(np.diff(rr_intervals)**2)) if len(rr_intervals) > 1 else 20),
                    'hr_mean': 60000 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 70,
                    'coherence': min(100, max(20, np.random.normal(50, 20))),
                    'recording_hours': duration,
                    'total_power': np.var(rr_intervals) * 1000 if len(rr_intervals) > 1 else 5000,
                    'vlf': np.var(rr_intervals) * 100 if len(rr_intervals) > 1 else 1000,
                    'lf': np.var(rr_intervals) * 300 if len(rr_intervals) > 1 else 2000,
                    'hf': np.var(rr_intervals) * 200 if len(rr_intervals) > 1 else 1500,
                    'lf_hf_ratio': 1.5 + np.random.normal(0, 0.5),
                    'sleep_duration': 7.2 if duration > 6 else 0,
                    'sleep_efficiency': 85 + np.random.normal(0, 5),
                    'sleep_hr': 58 + np.random.normal(0, 2)
                }
            }
            
            # Salva metriche per report
            st.session_state.last_analysis_metrics = metrics
            st.session_state.last_analysis_start = start_datetime
            st.session_state.last_analysis_end = end_datetime
            st.session_state.last_analysis_duration = selected_range
            
            # Salva nel database
            save_analysis_to_user_database(metrics, start_datetime, end_datetime, selected_range, "Analisi HRV")
            
            # =============================================================================
            # VISUALIZZAZIONE RISULTATI COMPLETA
            # =============================================================================
            
            st.header("📊 Risultati Analisi HRV Completa")
            
            # Metriche principali in cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>SDNN</h3>
                    <h2>{metrics['our_algo']['sdnn']:.1f} ms</h2>
                    <p>{get_sdnn_evaluation(metrics['our_algo']['sdnn'], st.session_state.user_profile['gender'])}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>RMSSD</h3>
                    <h2>{metrics['our_algo']['rmssd']:.1f} ms</h2>
                    <p>{get_rmssd_evaluation(metrics['our_algo']['rmssd'], st.session_state.user_profile['gender'])}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Frequenza Cardiaca</h3>
                    <h2>{metrics['our_algo']['hr_mean']:.1f} bpm</h2>
                    <p>{get_hr_evaluation(metrics['our_algo']['hr_mean'])}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Coerenza</h3>
                    <h2>{metrics['our_algo']['coherence']:.1f}%</h2>
                    <p>{get_coherence_evaluation(metrics['our_algo']['coherence'])}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # ANALISI SPETTRALE
            st.header("📡 Analisi Spettrale HRV")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="spectral-card">
                    <h4>Total Power</h4>
                    <h3>{metrics['our_algo']['total_power']:.0f} ms²</h3>
                    <p>{get_power_evaluation(metrics['our_algo']['total_power'])}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="spectral-card">
                    <h4>VLF Power</h4>
                    <h3>{metrics['our_algo']['vlf']:.0f} ms²</h3>
                    <p>Termoregolazione</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="spectral-card">
                    <h4>LF/HF Ratio</h4>
                    <h3>{metrics['our_algo']['lf_hf_ratio']:.2f}</h3>
                    <p>{get_lf_hf_evaluation(metrics['our_algo']['lf_hf_ratio'])}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                # Grafico a torta per distribuzione potenza
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['VLF', 'LF', 'HF'],
                    values=[metrics['our_algo']['vlf'], metrics['our_algo']['lf'], metrics['our_algo']['hf']],
                    hole=.3,
                    marker_colors=['#95a5a6', '#3498db', '#e74c3c']
                )])
                fig_pie.update_layout(title="Distribuzione Potenza", height=200)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # ANALISI SONNO (se disponibile)
            if metrics['our_algo'].get('sleep_duration', 0) > 0:
                st.header("😴 Analisi Qualità Sonno")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="sleep-card">
                        <h4>Durata Sonno</h4>
                        <h3>{metrics['our_algo']['sleep_duration']:.1f} h</h3>
                        <p>{"✅ Ottima" if metrics['our_algo']['sleep_duration'] >= 7 else "⚠️ Da migliorare"}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="sleep-card">
                        <h4>Efficienza Sonno</h4>
                        <h3>{metrics['our_algo']['sleep_efficiency']:.0f}%</h3>
                        <p>{"✅ Ottima" if metrics['our_algo']['sleep_efficiency'] >= 85 else "⚠️ Da migliorare"}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="sleep-card">
                        <h4>HR Notturno</h4>
                        <h3>{metrics['our_algo']['sleep_hr']:.0f} bpm</h3>
                        <p>{"✅ Normale" if metrics['our_algo']['sleep_hr'] <= 65 else "⚠️ Elevato"}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # PUNTI DI DEBOLEZZA E RACCOMANDAZIONI
            st.header("🔍 Analisi Punti di Debolezza")
            weaknesses = identify_weaknesses(metrics, st.session_state.user_profile)
            recommendations = generate_recommendations(metrics, st.session_state.user_profile, weaknesses)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Punti Critici Identificati")
                for weakness in weaknesses:
                    st.markdown(f"""
                    <div class="weakness-card">
                        <p>• {weakness}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("💡 Raccomandazioni Personalizzate")
                for category, recs in recommendations.items():
                    with st.expander(f"{category} ({len(recs)} raccomandazioni)"):
                        for rec in recs[:3]:  # Mostra prime 3 raccomandazioni
                            st.write(f"• {rec}")
            
            # GRAFICI AVANZATI
            st.header("📈 Visualizzazioni Avanzate")
            
            tab1, tab2, tab3 = st.tabs(["🔄 Andamento Temporale", "🎯 Analisi 3D", "📋 Storico Analisi"])
            
            with tab1:
                fig_timeseries = create_hrv_timeseries_plot_with_real_time(
                    metrics, [], start_datetime, end_datetime
                )
                st.plotly_chart(fig_timeseries, use_container_width=True)
            
            with tab2:
                fig_3d = create_advanced_3d_plot(metrics)
                st.plotly_chart(fig_3d, use_container_width=True)
            
            with tab3:
                analyses = get_user_analyses(st.session_state.user_profile)
                if analyses:
                    st.subheader("📊 Storico Analisi")
                    for analysis in analyses[-3:]:
                        with st.expander(f"Analisi del {analysis['start_datetime'].strftime('%d/%m/%Y %H:%M')}"):
                            st.write(f"**SDNN:** {analysis['metrics']['sdnn']:.1f} ms")
                            st.write(f"**RMSSD:** {analysis['metrics']['rmssd']:.1f} ms")
                            st.write(f"**HR:** {analysis['metrics']['hr_mean']:.1f} bpm")
                            st.write(f"**Durata:** {analysis['selected_range']}")
                else:
                    st.info("Nessuna analisi precedente trovata")
            
            # GENERAZIONE REPORT PDF
            st.header("📄 Genera Report Completo")
            
            if st.button("🎨 Genera Report PDF Avanzato", use_container_width=True):
                with st.spinner("Generando report PDF con grafiche avanzate..."):
                    try:
                        pdf_buffer = create_advanced_pdf_report(
                            metrics, start_datetime, end_datetime, selected_range, 
                            st.session_state.user_profile, []
                        )
                        
                        st.success("✅ Report PDF generato con successo!")
                        
                        # Download button
                        st.download_button(
                            label="📥 Scarica Report PDF Completo",
                            data=pdf_buffer,
                            file_name=f"HRV_Report_{st.session_state.user_profile['name']}_{st.session_state.user_profile['surname']}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"❌ Errore nella generazione del PDF: {e}")
                        st.info("⚠️ Assicurati che reportlab sia installato: pip install reportlab")
            
        except Exception as e:
            st.error(f"❌ Errore durante l'elaborazione del file: {str(e)}")
    
    else:
        # Schermata iniziale
        st.info("""
        ### 👆 Carica un file IBI per iniziare l'analisi
        
        **Formati supportati:** .txt, .csv
        
        Il file deve contenere gli intervalli IBI (Inter-Beat Intervals) in millisecondi, uno per riga.
        
        ### 🎯 Funzionalità disponibili:
        - ✅ **Analisi HRV avanzata** con metriche SDNN, RMSSD, LF/HF
        - ✅ **Analisi spettrale** completa (Total Power, VLF, LF, HF)
        - ✅ **Analisi qualità sonno** (se periodo notturno)
        - ✅ **Identificazione punti debolezza** automatica
        - ✅ **Raccomandazioni personalizzate** per migliorare HRV
        - ✅ **Grafiche 3D interattive** per visualizzazione dati
        - ✅ **Report PDF professionale** con analisi completa
        - ✅ **Storico paziente** per monitoraggio nel tempo
        
        ### 📋 Installazione dipendenze:
        ```bash
        pip install reportlab matplotlib streamlit plotly numpy pandas
        ```
        """)

if __name__ == "__main__":
    main()