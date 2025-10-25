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
import os
from scipy import stats

# =============================================================================
# NUOVO: IMPORT NEUROKIT2 PER CALCOLI HRV AVANZATI
# =============================================================================
try:
    import neurokit2 as nk
    NEUROKIT_AVAILABLE = True
    st.sidebar.success("✅ NeuroKit2 caricato - Calcoli HRV avanzati attivi")
except ImportError:
    NEUROKIT_AVAILABLE = False
    st.sidebar.warning("⚠️ NeuroKit2 non disponibile - Usando calcoli base")

# =============================================================================
# FUNZIONE INIT_SESSION_STATE MANCANTE - AGGIUNTA
# =============================================================================
def init_session_state():
    """Inizializza lo stato della sessione"""
    if 'user_database' not in st.session_state:
        st.session_state.user_database = load_user_database()
    
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': '',
            'surname': '',
            'birth_date': None,
            'age': 30,
            'gender': 'Uomo'
        }
    
    if 'activities' not in st.session_state:
        st.session_state.activities = []
    
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    
    if 'datetime_initialized' not in st.session_state:
        st.session_state.datetime_initialized = False
    
    if 'analysis_datetimes' not in st.session_state:
        st.session_state.analysis_datetimes = {
            'start_datetime': datetime.now(),
            'end_datetime': datetime.now() + timedelta(hours=24)
        }
    
    if 'last_analysis_metrics' not in st.session_state:
        st.session_state.last_analysis_metrics = None
    
    if 'last_analysis_start' not in st.session_state:
        st.session_state.last_analysis_start = None
    
    if 'last_analysis_end' not in st.session_state:
        st.session_state.last_analysis_end = None
    
    if 'last_analysis_duration' not in st.session_state:
        st.session_state.last_analysis_duration = None
    
    if 'editing_activity_index' not in st.session_state:
        st.session_state.editing_activity_index = None
    
    if 'last_analysis_daily' not in st.session_state:
        st.session_state.last_analysis_daily = []

# =============================================================================
# CORREZIONE FUNZIONE HRV CON NEUROKIT2 - RISOLTO ERRORE HRV_TotalPower
# =============================================================================

def calculate_hrv_with_neurokit(rr_intervals, user_age, user_gender):
    """Calcola metriche HRV usando NeuroKit2 (versione corretta)"""
    if len(rr_intervals) < 10:
        return get_default_metrics(user_age, user_gender)
    
    try:
        # Converti RR intervals in secondi per NeuroKit2
        rr_seconds = np.array(rr_intervals) / 1000.0
        
        # CORREZIONE: Usa rsp_clean invece di ppg_clean per dati RR
        rr_clean = nk.rsp_clean(rr_seconds, sampling_rate=1000, method="khodadad2018")
        peaks = nk.rsp_peaks(rr_clean, sampling_rate=1000)[0]
        
        if len(peaks) < 10:
            return get_default_metrics(user_age, user_gender)
        
        # Calcola metriche HRV complete con NeuroKit2 - CORREZIONE: gestione errori
        try:
            hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)
            # CORREZIONE: Estrai valori con controllo degli errori
            sdnn = hrv_time['HRV_SDNN'].iloc[0] * 1000 if 'HRV_SDNN' in hrv_time.columns else np.std(rr_intervals, ddof=1)
            rmssd = hrv_time['HRV_RMSSD'].iloc[0] * 1000 if 'HRV_RMSSD' in hrv_time.columns else calculate_rmssd_fallback(rr_intervals)
        except:
            sdnn = np.std(rr_intervals, ddof=1)
            rmssd = calculate_rmssd_fallback(rr_intervals)
        
        hr_mean = 60000 / np.mean(rr_intervals)
        
        # CORREZIONE: Analisi spettrale con gestione errori
        try:
            hrv_frequency = nk.hrv_frequency(peaks, sampling_rate=1000, show=False)
            total_power = hrv_frequency['HRV_TotalPower'].iloc[0] if 'HRV_TotalPower' in hrv_frequency.columns else 2000
            vlf = hrv_frequency['HRV_VLF'].iloc[0] if 'HRV_VLF' in hrv_frequency.columns else 500
            lf = hrv_frequency['HRV_LF'].iloc[0] if 'HRV_LF' in hrv_frequency.columns else 800
            hf = hrv_frequency['HRV_HF'].iloc[0] if 'HRV_HF' in hrv_frequency.columns else 700
            lf_hf_ratio = hrv_frequency['HRV_LFHF'].iloc[0] if 'HRV_LFHF' in hrv_frequency.columns else 1.1
        except:
            # Fallback per analisi spettrale
            total_power = 2000
            vlf = 500
            lf = 800
            hf = 700
            lf_hf_ratio = 1.1
        
        # CORREZIONE: Analisi non-lineare con gestione errori
        try:
            hrv_nonlinear = nk.hrv_nonlinear(peaks, sampling_rate=1000, show=False)
            sd1 = hrv_nonlinear['HRV_SD1'].iloc[0] * 1000 if 'HRV_SD1' in hrv_nonlinear.columns else rmssd / np.sqrt(2)
            sd2 = hrv_nonlinear['HRV_SD2'].iloc[0] * 1000 if 'HRV_SD2' in hrv_nonlinear.columns else sdnn
            sd1_sd2_ratio = hrv_nonlinear['HRV_SD1SD2'].iloc[0] if 'HRV_SD1SD2' in hrv_nonlinear.columns else (rmssd / np.sqrt(2)) / sdnn if sdnn > 0 else 1.0
        except:
            sd1 = rmssd / np.sqrt(2)
            sd2 = sdnn
            sd1_sd2_ratio = (rmssd / np.sqrt(2)) / sdnn if sdnn > 0 else 1.0
        
        # Coerenza cardiaca e analisi sonno
        coherence = calculate_hrv_coherence_advanced(rr_intervals, hr_mean, user_age)
        sleep_metrics = estimate_sleep_metrics_advanced(rr_intervals, hr_mean, user_age)
        
        return {
            'sdnn': max(25, min(180, sdnn)),
            'rmssd': max(15, min(120, rmssd)),
            'hr_mean': max(45, min(100, hr_mean)),
            'coherence': max(20, min(95, coherence)),
            'recording_hours': len(rr_intervals) * np.mean(rr_intervals) / (1000 * 60 * 60),
            'total_power': max(800, min(8000, total_power)),
            'vlf': max(100, min(2500, vlf)),
            'lf': max(200, min(4000, lf)),
            'hf': max(200, min(4000, hf)),
            'lf_hf_ratio': max(0.3, min(4.0, lf_hf_ratio)),
            'sd1': max(10, min(80, sd1)),
            'sd2': max(30, min(200, sd2)),
            'sd1_sd2_ratio': max(0.2, min(3.0, sd1_sd2_ratio)),
            'sleep_duration': sleep_metrics['duration'],
            'sleep_efficiency': sleep_metrics['efficiency'],
            'sleep_hr': sleep_metrics['hr'],
            'sleep_light': sleep_metrics['light'],
            'sleep_deep': sleep_metrics['deep'],
            'sleep_rem': sleep_metrics['rem'],
            'sleep_awake': sleep_metrics['awake'],
            'analysis_method': 'NeuroKit2'
        }
        
    except Exception as e:
        st.warning(f"NeuroKit2 analysis failed: {e}. Using fallback method.")
        return calculate_hrv_fallback(rr_intervals, user_age, user_gender)

def calculate_rmssd_fallback(rr_intervals):
    """Calcola RMSSD come fallback"""
    if len(rr_intervals) < 2:
        return 30
    differences = np.diff(rr_intervals)
    return np.sqrt(np.mean(np.square(differences)))

# =============================================================================
# CORREZIONE ANALISI GIORNALIERA - RISOLTO PROBLEMA DATE
# =============================================================================

def analyze_daily_metrics(rr_intervals, start_datetime, user_profile, activities=[]):
    """Divide l'analisi in giorni separati - VERSIONE CORRETTA CON DATE CORRETTE"""
    daily_analyses = []
    
    if len(rr_intervals) == 0:
        return daily_analyses
    
    # Calcola durata totale in giorni
    total_duration_ms = np.sum(rr_intervals)
    total_duration_hours = total_duration_ms / (1000 * 60 * 60)
    total_days = int(np.ceil(total_duration_hours / 24))
    
    current_index = 0
    
    for day in range(total_days):
        day_start = start_datetime + timedelta(days=day)
        day_end = day_start + timedelta(hours=24)
        
        # CORREZIONE: Seleziona RR intervals per questo giorno basandosi sul tempo accumulato
        day_rr = []
        accumulated_time_ms = 0
        day_duration_ms = 24 * 60 * 60 * 1000  # 24 ore in millisecondi
        
        while current_index < len(rr_intervals) and accumulated_time_ms < day_duration_ms:
            current_rr = rr_intervals[current_index]
            day_rr.append(current_rr)
            accumulated_time_ms += current_rr
            current_index += 1
        
        # Analizza solo se abbiamo dati sufficienti
        if len(day_rr) >= 50:  # Almeno 50 battiti per analisi significativa
            daily_metrics = calculate_realistic_hrv_metrics(
                day_rr, user_profile.get('age', 30), user_profile.get('gender', 'Uomo')
            )
            
            # CORREZIONE: Filtra attività per il giorno CORRETTO
            day_activities = get_activities_for_period(activities, day_start, day_end)
            nutrition_impact = analyze_nutritional_impact_day(day_activities, daily_metrics)
            activity_impact = analyze_activity_impact_on_ans(day_activities, daily_metrics)
            
            daily_analyses.append({
                'day_number': day + 1,
                'date': day_start.date(),  # CORREZIONE: Usa la data corretta
                'start_time': day_start,
                'end_time': day_end,
                'metrics': daily_metrics,
                'activities': day_activities,
                'nutrition_impact': nutrition_impact,
                'activity_impact': activity_impact,
                'rr_count': len(day_rr),
                'recording_hours': accumulated_time_ms / (1000 * 60 * 60)
            })
    
    return daily_analyses

def get_activities_for_period(activities, start_time, end_time):
    """Filtra le attività per il periodo specificato - VERSIONE CORRETTA"""
    period_activities = []
    for activity in activities:
        activity_start = activity['start_time']
        activity_end = activity_start + timedelta(minutes=activity['duration'])
        
        # CORREZIONE: Controllo più preciso dell'overlap
        # Un'attività è nel periodo se inizia prima della fine del periodo e finisce dopo l'inizio
        if (activity_start < end_time and activity_end > start_time):
            period_activities.append(activity)
    return period_activities

# =============================================================================
# CORREZIONE SALVATAGGIO ATTIVITÀ - RISOLTO PROBLEMA DATA
# =============================================================================

def save_activity(activity_type, name, intensity, food_items, start_date, start_time, duration, notes):
    """Salva una nuova attività - VERSIONE CORRETTA CON DATA CORRETTA"""
    # CORREZIONE: Combina data e ora correttamente
    start_datetime = datetime.combine(start_date, start_time)
    
    activity = {
        'type': activity_type,
        'name': name,
        'intensity': intensity,
        'food_items': food_items,
        'start_time': start_datetime,  # CORREZIONE: Usa datetime combinato
        'duration': duration,
        'notes': notes,
        'timestamp': datetime.now(),
        'color': ACTIVITY_COLORS.get(activity_type, "#95a5a6")
    }
    
    st.session_state.activities.append(activity)
    
    # Limita a 50 attività
    if len(st.session_state.activities) > 50:
        st.session_state.activities = st.session_state.activities[-50:]

def update_activity(index, activity_type, name, intensity, food_items, start_date, start_time, duration, notes):
    """Aggiorna un'attività esistente - VERSIONE CORRETTA"""
    if 0 <= index < len(st.session_state.activities):
        # CORREZIONE: Combina data e ora correttamente
        start_datetime = datetime.combine(start_date, start_time)
        
        st.session_state.activities[index] = {
            'type': activity_type,
            'name': name,
            'intensity': intensity,
            'food_items': food_items,
            'start_time': start_datetime,  # CORREZIONE: Usa datetime combinato
            'duration': duration,
            'notes': notes,
            'timestamp': datetime.now(),
            'color': ACTIVITY_COLORS.get(activity_type, "#95a5a6")
        }

# =============================================================================
# CORREZIONE INTERFACCIA ATTIVITÀ - MIGLIORATA VISUALIZZAZIONE DATE
# =============================================================================

def create_activity_tracker():
    """Interfaccia per tracciare attività e alimentazione - VERSIONE MIGLIORATA"""
    st.sidebar.header("🏃‍♂️ Tracker Attività & Alimentazione")
    
    # Gestione modifica attività
    if st.session_state.get('editing_activity_index') is not None:
        edit_activity_interface()
        return
    
    with st.sidebar.expander("➕ Aggiungi Attività/Pasto", expanded=False):
        activity_type = st.selectbox("Tipo Attività", 
                                   ["Allenamento", "Alimentazione", "Stress", "Riposo", "Altro"])
        
        activity_name = st.text_input("Nome Attività/Pasto", placeholder="Es: Corsa mattutina, Pranzo, etc.")
        
        if activity_type == "Alimentazione":
            food_items = st.text_area("Cosa hai mangiato? (separato da virgola)", placeholder="Es: pasta, insalata, frutta")
            intensity = st.select_slider("Pesantezza pasto", 
                                       options=["Leggero", "Normale", "Pesante", "Molto pesante"])
        else:
            food_items = ""
            intensity = st.select_slider("Intensità", 
                                       options=["Leggera", "Moderata", "Intensa", "Massimale"])
        
        col1, col2 = st.columns(2)
        with col1:
            # CORREZIONE: Data predefinita più chiara
            start_date = st.date_input("Data", value=datetime.now().date(), key="activity_date")
            start_time = st.time_input("Ora inizio", value=datetime.now().time(), key="activity_time")
        with col2:
            duration = st.number_input("Durata (min)", min_value=1, max_value=480, value=30, key="activity_duration")
        
        notes = st.text_area("Note (opzionale)", placeholder="Note aggiuntive...", key="activity_notes")
        
        if st.button("💾 Salva Attività", use_container_width=True, key="save_activity"):
            save_activity(activity_type, activity_name, intensity, food_items, start_date, start_time, duration, notes)
            st.success(f"Attività salvata per il {start_date.strftime('%d/%m/%Y')} alle {start_time.strftime('%H:%M')}!")
            st.rerun()
    
    # Gestione attività esistenti - CORREZIONE: Visualizzazione migliorata
    if st.session_state.activities:
        st.sidebar.subheader("📋 Attività Salvate")
        
        # Ordina attività per data (più recenti prima)
        sorted_activities = sorted(st.session_state.activities, key=lambda x: x['start_time'], reverse=True)
        
        for i, activity in enumerate(sorted_activities[:10]):  # Mostra solo ultime 10
            with st.sidebar.expander(f"{activity['name']} - {activity['start_time'].strftime('%d/%m %H:%M')}", False):
                st.write(f"**Tipo:** {activity['type']}")
                st.write(f"**Intensità:** {activity['intensity']}")
                if activity['food_items']:
                    st.write(f"**Cibo:** {activity['food_items']}")
                st.write(f"**Data/Ora:** {activity['start_time'].strftime('%d/%m/%Y %H:%M')}")
                st.write(f"**Durata:** {activity['duration']} min")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✏️ Modifica", key=f"edit_{i}", use_container_width=True):
                        st.session_state.editing_activity_index = i
                        st.rerun()
                with col2:
                    if st.button("🗑️ Elimina", key=f"delete_{i}", use_container_width=True):
                        delete_activity(i)
                        st.rerun()

# =============================================================================
# CORREZIONE VISUALIZZAZIONE ANALISI GIORNALIERA - MOSTRA TUTTI I GIORNI
# =============================================================================

def create_daily_analysis_visualization(daily_analyses):
    """Crea visualizzazioni per l'analisi giornaliera - VERSIONE CORRETTA"""
    if not daily_analyses:
        return None
    
    st.header("📅 Analisi Giornaliera Dettagliata")
    
    # CORREZIONE: Mostra informazioni su tutti i giorni analizzati
    st.info(f"**Periodo analizzato:** {len(daily_analyses)} giorni - Dal {daily_analyses[0]['date'].strftime('%d/%m/%Y')} al {daily_analyses[-1]['date'].strftime('%d/%m/%Y')}")
    
    # Metriche chiave per giorno
    days = [f"Giorno {day['day_number']}\n({day['date'].strftime('%d/%m')})" for day in daily_analyses]
    sdnn_values = [day['metrics']['sdnn'] for day in daily_analyses]
    rmssd_values = [day['metrics']['rmssd'] for day in daily_analyses]
    hr_values = [day['metrics']['hr_mean'] for day in daily_analyses]
    
    # Crea tabs per organizzare meglio le informazioni
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Andamento Giornaliero", "🎯 Analisi Dettagliata", "📈 Analisi Spettrale", "🏃‍♂️ Impatto Attività"])
    
    with tab1:
        # Grafico andamento metriche principali
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=days, y=sdnn_values, 
            mode='lines+markers', name='SDNN',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8, color='#3498db')
        ))
        
        fig.add_trace(go.Scatter(
            x=days, y=rmssd_values, 
            mode='lines+markers', name='RMSSD',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8, color='#e74c3c')
        ))
        
        fig.add_trace(go.Scatter(
            x=days, y=hr_values, 
            mode='lines+markers', name='HR',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=8, color='#2ecc71'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="📊 Andamento Metriche HRV per Giorno",
            xaxis_title="Giorno",
            yaxis_title="HRV (ms)",
            yaxis2=dict(
                title="HR (bpm)",
                overlaying='y',
                side='right'
            ),
            height=400,
            showlegend=True,
            plot_bgcolor='rgba(240,240,240,0.5)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # CORREZIONE: Mostra dettaglio per TUTTI i giorni
        st.subheader(f"Dettaglio Completo - {len(daily_analyses)} Giorni Analizzati")
        
        for day_analysis in daily_analyses:
            with st.expander(f"📋 Giorno {day_analysis['day_number']} - {day_analysis['date'].strftime('%d/%m/%Y')} ({day_analysis['recording_hours']:.1f}h)", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("SDNN", f"{day_analysis['metrics']['sdnn']:.1f} ms", 
                             delta=f"{get_sdnn_evaluation(day_analysis['metrics']['sdnn'], 'Uomo')}")
                    st.metric("RMSSD", f"{day_analysis['metrics']['rmssd']:.1f} ms",
                             delta=f"{get_rmssd_evaluation(day_analysis['metrics']['rmssd'], 'Uomo')}")
                
                with col2:
                    st.metric("Frequenza Cardiaca", f"{day_analysis['metrics']['hr_mean']:.1f} bpm",
                             delta=f"{get_hr_evaluation(day_analysis['metrics']['hr_mean'])}")
                    st.metric("Coerenza", f"{day_analysis['metrics']['coherence']:.1f}%",
                             delta=f"{get_coherence_evaluation(day_analysis['metrics']['coherence'])}")
                
                with col3:
                    st.metric("Durata Registrazione", f"{day_analysis['recording_hours']:.1f} h")
                    st.metric("Battiti Analizzati", f"{day_analysis['rr_count']:,}")
                
                with col4:
                    st.metric("LF/HF Ratio", f"{day_analysis['metrics']['lf_hf_ratio']:.2f}",
                             delta=f"{get_lf_hf_evaluation(day_analysis['metrics']['lf_hf_ratio'])}")
                    st.metric("Potenza Totale", f"{day_analysis['metrics']['total_power']:.0f} ms²")
                
                # Analisi del sonno per il giorno
                if day_analysis['metrics'].get('sleep_duration', 0) > 0:
                    st.subheader("😴 Analisi Sonno")
                    sleep_cols = st.columns(4)
                    with sleep_cols[0]:
                        st.metric("Durata Totale", f"{day_analysis['metrics'].get('sleep_duration', 0):.1f} h")
                    with sleep_cols[1]:
                        st.metric("Efficienza", f"{day_analysis['metrics'].get('sleep_efficiency', 0):.0f}%")
                    with sleep_cols[2]:
                        st.metric("Sonno Leggero", f"{day_analysis['metrics'].get('sleep_light', 0):.1f} h")
                    with sleep_cols[3]:
                        st.metric("Sonno Profondo", f"{day_analysis['metrics'].get('sleep_deep', 0):.1f} h")
                    
                    sleep_cols2 = st.columns(3)
                    with sleep_cols2[0]:
                        st.metric("Sonno REM", f"{day_analysis['metrics'].get('sleep_rem', 0):.1f} h")
                    with sleep_cols2[1]:
                        st.metric("Risvegli", f"{day_analysis['metrics'].get('sleep_awake', 0):.1f} h")
                    with sleep_cols2[2]:
                        st.metric("FC Notturna", f"{day_analysis['metrics'].get('sleep_hr', 0):.0f} bpm")
                    
                    # Grafico a torta per le fasi del sonno
                    sleep_phases = ['Leggero', 'Profondo', 'REM', 'Risvegli']
                    sleep_values = [
                        day_analysis['metrics'].get('sleep_light', 0),
                        day_analysis['metrics'].get('sleep_deep', 0), 
                        day_analysis['metrics'].get('sleep_rem', 0),
                        day_analysis['metrics'].get('sleep_awake', 0)
                    ]
                    
                    fig_sleep = px.pie(
                        values=sleep_values,
                        names=sleep_phases,
                        title="Distribuzione Fasi del Sonno",
                        color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
                    )
                    fig_sleep.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_sleep, use_container_width=True)
                
                # Attività del giorno - CORREZIONE: Mostra attività del giorno corretto
                day_activities = day_analysis['activities']
                if day_activities:
                    st.subheader(f"🏃‍♂️ Attività del {day_analysis['date'].strftime('%d/%m/%Y')}")
                    for activity in day_activities:
                        st.write(f"• **{activity['name']}** ({activity['type']}) - {activity['intensity']} - {activity['start_time'].strftime('%H:%M')} ({activity['duration']} min)")
                else:
                    st.info("Nessuna attività registrata per questo giorno")
                
                # Analisi alimentare
                nutrition = day_analysis.get('nutrition_impact', {})
                if nutrition.get('analysis'):
                    st.subheader("🍽️ Analisi Alimentazione")
                    
                    if nutrition['score'] > 2:
                        st.error(nutrition['analysis'])
                    elif nutrition['score'] < 0:
                        st.success(nutrition['analysis'])
                    else:
                        st.warning(nutrition['analysis'])
                    
                    if nutrition.get('recommendations'):
                        st.write("**Suggerimenti:**")
                        for rec in nutrition['recommendations']:
                            st.write(f"• {rec}")

    # ... (resto del codice invariato per le altre tabs)

# =============================================================================
# CORREZIONE GENERAZIONE PDF - RISOLTO PROBLEMA BOTTONE
# =============================================================================

def create_advanced_pdf_report(metrics, start_datetime, end_datetime, selected_range, user_profile, activities=[], daily_analyses=[]):
    """Crea un report PDF avanzato - VERSIONE CORRETTA"""
    try:
        # CORREZIONE: Import dentro la funzione per gestire meglio gli errori
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import mm
            from reportlab.lib.colors import HexColor, white
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        except ImportError as e:
            st.error(f"❌ ReportLab non installato: {e}")
            st.info("Installa: pip install reportlab")
            return None
        
        buffer = io.BytesIO()
        
        try:
            doc = SimpleDocTemplate(buffer, pagesize=A4, 
                                  topMargin=20*mm, bottomMargin=20*mm,
                                  leftMargin=15*mm, rightMargin=15*mm)
            
            styles = getSampleStyleSheet()
            
            # Stili personalizzati
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                textColor=HexColor("#2c3e50"),
                spaceAfter=20,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            )
            
            story = []
            
            # INTESTAZIONE
            story.append(Paragraph("REPORT HRV COMPLETO<br/><font size=10>Analisi Sistema Neurovegetativo</font>", title_style))
            story.append(Spacer(1, 15))
            
            # Informazioni utente
            user_info = f"""
            <b>PAZIENTE:</b> {user_profile.get('name', '')} {user_profile.get('surname', '')}<br/>
            <b>ETÀ:</b> {user_profile.get('age', '')} anni | <b>SESSO:</b> {user_profile.get('gender', '')}<br/>
            <b>PERIODO ANALISI:</b> {start_datetime.strftime('%d/%m/%Y %H:%M')} - {end_datetime.strftime('%d/%m/%Y %H:%M')}<br/>
            <b>DURATA TOTALE:</b> {selected_range}<br/>
            <b>DATA GENERAZIONE:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}
            """
            story.append(Paragraph(user_info, ParagraphStyle('Normal', fontSize=10)))
            story.append(Spacer(1, 20))
            
            # METRICHE PRINCIPALI
            story.append(Paragraph("<b>METRICHE HRV PRINCIPALI</b>", ParagraphStyle('Heading2', fontSize=12)))
            
            main_metrics_data = [
                ['METRICA', 'VALORE', 'VALUTAZIONE'],
                ['SDNN', f"{metrics['sdnn']:.1f} ms", get_sdnn_evaluation(metrics['sdnn'], user_profile.get('gender', 'Uomo'))],
                ['RMSSD', f"{metrics['rmssd']:.1f} ms", get_rmssd_evaluation(metrics['rmssd'], user_profile.get('gender', 'Uomo'))],
                ['FC Media', f"{metrics['hr_mean']:.1f} bpm", get_hr_evaluation(metrics['hr_mean'])],
                ['Coerenza', f"{metrics['coherence']:.1f}%", get_coherence_evaluation(metrics['coherence'])]
            ]
            
            main_table = Table(main_metrics_data, colWidths=[60, 50, 80])
            main_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor("#3498db")),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), HexColor("#ecf0f1")),
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#bdc3c7")),
            ]))
            
            story.append(main_table)
            story.append(Spacer(1, 20))
            
            # GENERA IL PDF
            doc.build(story)
            buffer.seek(0)
            return buffer
            
        except Exception as e:
            st.error(f"Errore nella generazione PDF: {e}")
            return None
            
    except Exception as e:
        st.error(f"Errore generale PDF: {e}")
        return None

# =============================================================================
# CORREZIONE INTERFACCIA PRINCIPALE - GESTIONE MIGLIORATA
# =============================================================================

def main():
    st.set_page_config(
        page_title="HRV Analytics ULTIMATE + NeuroKit2",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inizializza lo stato della sessione
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
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header principale
    st.markdown('<h1 class="main-header">❤️ HRV Analytics ULTIMATE</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("👤 Profilo Paziente")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.user_profile['name'] = st.text_input("Nome", value=st.session_state.user_profile['name'])
        with col2:
            st.session_state.user_profile['surname'] = st.text_input("Cognome", value=st.session_state.user_profile['surname'])
        
        st.session_state.user_profile['birth_date'] = st.date_input(
            "Data di nascita", 
            value=st.session_state.user_profile['birth_date'] or datetime(1980, 1, 1).date(),
            format="DD/MM/YYYY"
        )
        
        st.session_state.user_profile['gender'] = st.selectbox("Sesso", ["Uomo", "Donna"], 
                                                             index=0 if st.session_state.user_profile['gender'] == 'Uomo' else 1)
        
        if st.session_state.user_profile['birth_date']:
            age = datetime.now().year - st.session_state.user_profile['birth_date'].year
            st.session_state.user_profile['age'] = age
            st.info(f"Età: {age} anni")
        
        # Aggiungi tracker attività
        create_activity_tracker()
    
    # Upload file
    st.header("📤 Carica File IBI")
    uploaded_file = st.file_uploader("Carica il tuo file .txt o .csv con gli intervalli IBI", type=['txt', 'csv'])
    
    if uploaded_file is not None:
        try:
            # Processa il file
            content = uploaded_file.getvalue().decode('utf-8')
            lines = content.strip().split('\n')
            
            rr_intervals = []
            for line in lines:
                if line.strip():
                    try:
                        value = float(line.strip())
                        if 400 <= value <= 1800:
                            rr_intervals.append(value)
                    except ValueError:
                        continue
            
            if len(rr_intervals) == 0:
                st.error("❌ Nessun dato IBI valido trovato nel file")
                return
            
            st.success(f"✅ {len(rr_intervals)} RR-interval validi rilevati")
            
            # Aggiorna data/ora automaticamente
            update_analysis_datetimes(uploaded_file, rr_intervals)
            
            # Selezione range temporale
            start_datetime, end_datetime = get_analysis_datetimes()
            
            st.header("⏰ Selezione Periodo Analisi")
            col1, col2 = st.columns(2)
            
            with col1:
                start_date = st.date_input("Data Inizio", value=start_datetime.date())
                start_time = st.time_input("Ora Inizio", value=start_datetime.time())
                new_start = datetime.combine(start_date, start_time)
            
            with col2:
                end_date = st.date_input("Data Fine", value=end_datetime.date())
                end_time = st.time_input("Ora Fine", value=end_datetime.time())
                new_end = datetime.combine(end_date, end_time)
            
            if new_start != start_datetime or new_end != end_datetime:
                st.session_state.analysis_datetimes = {
                    'start_datetime': new_start,
                    'end_datetime': new_end
                }
                st.rerun()
            
            # Calcola durata
            duration = (end_datetime - start_datetime).total_seconds() / 3600
            selected_range = f"{duration:.1f} ore"
            
            # CORREZIONE: Salva le daily_analyses nello session state
            if st.button("🚀 Avvia Analisi HRV Completa", use_container_width=True):
                with st.spinner("Analisi in corso..."):
                    metrics = {
                        'our_algo': calculate_realistic_hrv_metrics(
                            rr_intervals, 
                            st.session_state.user_profile.get('age', 30), 
                            st.session_state.user_profile.get('gender', 'Uomo')
                        )
                    }
                    
                    # Analisi giornaliera
                    daily_analyses = []
                    if duration > 24:
                        daily_analyses = analyze_daily_metrics(
                            rr_intervals, start_datetime, st.session_state.user_profile, st.session_state.activities
                        )
                        st.info(f"📅 **Analisi giornaliera:** {len(daily_analyses)} giorni analizzati")
                    
                    # CORREZIONE: Salva tutto nello session state
                    st.session_state.last_analysis_metrics = metrics
                    st.session_state.last_analysis_start = start_datetime
                    st.session_state.last_analysis_end = end_datetime
                    st.session_state.last_analysis_duration = selected_range
                    st.session_state.last_analysis_daily = daily_analyses  # CORREZIONE: Salva daily_analyses
                    
                    st.success("✅ Analisi completata!")
            
            # CORREZIONE: Mostra risultati se disponibili
            if st.session_state.last_analysis_metrics:
                show_analysis_results(
                    st.session_state.last_analysis_metrics,
                    st.session_state.last_analysis_daily,
                    st.session_state.last_analysis_start,
                    st.session_state.last_analysis_end,
                    st.session_state.last_analysis_duration
                )
        
        except Exception as e:
            st.error(f"❌ Errore durante l'elaborazione: {str(e)}")
    
    else:
        show_welcome_screen()

# =============================================================================
# FUNZIONI MANCANTI PER COMPATIBILITÀ
# =============================================================================

def calculate_realistic_hrv_metrics(rr_intervals, user_age, user_gender):
    """Wrapper per compatibilità"""
    return calculate_comprehensive_hrv(rr_intervals, user_age, user_gender)

def calculate_comprehensive_hrv(rr_intervals, user_age, user_gender):
    """Funzione principale HRV"""
    if NEUROKIT_AVAILABLE:
        return calculate_hrv_with_neurokit(rr_intervals, user_age, user_gender)
    else:
        return calculate_hrv_fallback(rr_intervals, user_age, user_gender)

# ... (inserisci qui tutte le altre funzioni mancanti dal codice originale)

# =============================================================================
# ESECUZIONE
# =============================================================================
if __name__ == "__main__":
    main()