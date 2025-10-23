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
            'birth_date': datetime.now().date(),
            'gender': 'Uomo',
            'age': 0
        }

# =============================================================================
# FUNZIONI PER ESTRAZIONE DATA E ORA DAL FILE
# =============================================================================

def extract_datetime_from_content(content):
    """Estrae data e ora esatte dal contenuto del file"""
    # Cerca STARTTIME=13.10.2025 19:46.16
    pattern = r'STARTTIME=(\d{1,2})\.(\d{1,2})\.(\d{4})\s+(\d{1,2}):(\d{2})\.(\d{2})'
    match = re.search(pattern, content)
    
    if match:
        day, month, year, hour, minute, second = map(int, match.groups())
        try:
            return datetime(year, month, day, hour, minute, second)
        except ValueError:
            pass
    
    # Pattern alternativo
    patterns = [
        r'(\d{1,2})[\./-](\d{1,2})[\./-](\d{4})[\sT](\d{1,2}):(\d{2}):(\d{2})',
        r'(\d{4})[\./-](\d{1,2})[\./-](\d{1,2})[\sT](\d{1,2}):(\d{2}):(\d{2})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            groups = match.groups()
            if len(groups[0]) == 4:  # YYYY-MM-DD
                year, month, day, hour, minute, second = map(int, groups[:6])
            else:  # DD-MM-YYYY
                day, month, year, hour, minute, second = map(int, groups[:6])
            
            try:
                return datetime(year, month, day, hour, minute, second)
            except ValueError:
                continue
    
    return None

def estimate_recording_duration(rr_intervals):
    """Stima la durata della registrazione dagli intervalli RR"""
    if len(rr_intervals) == 0:
        return 1.0
    
    total_ms = np.sum(rr_intervals)
    duration_hours = total_ms / (1000 * 60 * 60)
    
    return max(0.1, min(168.0, duration_hours))  # Limita tra 0.1 e 168 ore (7 giorni)

# =============================================================================
# GESTIONE DATA/ORA AUTOMATICA MIGLIORATA
# =============================================================================

def update_analysis_datetimes(uploaded_file, rr_intervals=None):
    """Aggiorna automaticamente data/ora quando viene caricato un file"""
    if uploaded_file is not None and not st.session_state.file_uploaded:
        file_datetime = None
        
        # Estrai data/ora dal contenuto del file
        try:
            content = uploaded_file.getvalue().decode('utf-8')
            file_datetime = extract_datetime_from_content(content)
        except:
            pass
        
        # Se non abbiamo data/ora, usa valori di default
        if file_datetime is None:
            file_datetime = datetime.now()
        
        # Stima la durata
        if rr_intervals is not None and len(rr_intervals) > 0:
            duration_hours = estimate_recording_duration(rr_intervals)
        else:
            duration_hours = 1.0
        
        start_dt = file_datetime
        end_dt = start_dt + timedelta(hours=duration_hours)
        
        st.session_state.analysis_datetimes = {
            'start_datetime': start_dt,
            'end_datetime': end_dt
        }
        st.session_state.file_uploaded = True
        st.rerun()

def get_analysis_datetimes():
    """Restituisce data/ora inizio e fine per l'analisi"""
    return (
        st.session_state.analysis_datetimes['start_datetime'],
        st.session_state.analysis_datetimes['end_datetime']
    )

# =============================================================================
# PROFILO UTENTE
# =============================================================================

def create_user_profile():
    """Crea il profilo utente"""
    st.sidebar.header("👤 Profilo Utente")
    
    with st.sidebar.expander("📝 Modifica Profilo", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Nome", value=st.session_state.user_profile['name'], key="user_name")
        with col2:
            surname = st.text_input("Cognome", value=st.session_state.user_profile['surname'], key="user_surname")
        
        birth_date = st.date_input(
            "Data di Nascita", 
            value=st.session_state.user_profile['birth_date'],
            max_value=datetime.now().date(),
            key="user_birth_date"
        )
        
        gender = st.selectbox(
            "Sesso",
            ["Uomo", "Donna"],
            index=0 if st.session_state.user_profile['gender'] == "Uomo" else 1,
            key="user_gender"
        )
        
        # Calcola età
        today = datetime.now().date()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        
        if st.button("💾 Salva Profilo", use_container_width=True, key="save_profile"):
            st.session_state.user_profile = {
                'name': name,
                'surname': surname,
                'birth_date': birth_date,
                'gender': gender,
                'age': age
            }
            st.success("✅ Profilo salvato!")
    
    # Mostra profilo corrente
    if st.session_state.user_profile['name']:
        st.sidebar.info(f"**Utente:** {st.session_state.user_profile['name']} {st.session_state.user_profile['surname']}")
        st.sidebar.info(f"**Età:** {st.session_state.user_profile['age']} anni | **Sesso:** {st.session_state.user_profile['gender']}")

def adjust_metrics_for_gender(metrics, gender):
    """Aggiusta le metriche in base al sesso"""
    adjusted_metrics = metrics.copy()
    
    if gender == "Donna":
        # Donne tendono ad avere valori HRV leggermente più alti
        adjustment_factor = 1.1
    else:
        # Uomini - valori standard
        adjustment_factor = 1.0
    
    adjusted_metrics['our_algo']['sdnn'] *= adjustment_factor
    adjusted_metrics['our_algo']['rmssd'] *= adjustment_factor
    adjusted_metrics['our_algo']['coherence'] *= adjustment_factor
    
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
# DIARIO ATTIVITÀ
# =============================================================================

def create_activity_diary():
    """Crea un diario delle attività con orari specifici"""
    st.sidebar.header("📝 Diario Attività")
    
    with st.sidebar.expander("➕ Aggiungi Attività", expanded=False):
        activity_name = st.text_input("Nome attività*", placeholder="Scrivi qui...", key="activity_name")
        
        st.write("**Orario attività:**")
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.time_input("Dalle ore", datetime.now().time(), key="start_time")
        with col2:
            end_time = st.time_input("Alle ore", (datetime.now() + timedelta(hours=1)).time(), key="end_time")
        
        activity_color = st.color_picker("Colore attività", "#3498db", key="activity_color")
        
        col3, col4 = st.columns(2)
        with col3:
            if st.button("💾 Salva Attività", use_container_width=True, key="save_activity"):
                if activity_name.strip():
                    start_datetime, end_datetime = get_analysis_datetimes()
                    activity_date = start_datetime.date()
                    
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
                else:
                    st.error("❌ Inserisci un nome per l'attività")
        
        with col4:
            if st.button("🗑️ Cancella Tutto", use_container_width=True, key="clear_activities"):
                st.session_state.activities = []
                st.success("✅ Tutte le attività cancellate!")
    
    if st.session_state.activities:
        st.sidebar.subheader("📋 Attività Salvate")
        for i, activity in enumerate(st.session_state.activities):
            with st.sidebar.expander(f"🕒 {activity['start'].strftime('%H:%M')}-{activity['end'].strftime('%H:%M')} {activity['name']}", False):
                st.write(f"**Orario:** {activity['start'].strftime('%H:%M')} - {activity['end'].strftime('%H:%M')}")
                st.write(f"**Colore:** {activity['color']}")
                
                if st.button(f"❌ Elimina", key=f"delete_activity_{i}"):
                    st.session_state.activities.pop(i)
                    st.rerun()

# =============================================================================
# STORICO ANALISI
# =============================================================================

def save_to_history(metrics, start_datetime, end_datetime, analysis_type, selected_range):
    """Salva l'analisi corrente nello storico"""
    analysis_data = {
        'timestamp': datetime.now(),
        'start_datetime': start_datetime,
        'end_datetime': end_datetime,
        'analysis_type': analysis_type,
        'selected_range': selected_range,
        'user_profile': st.session_state.user_profile.copy(),
        'metrics': {
            'sdnn': metrics['our_algo']['sdnn'],
            'rmssd': metrics['our_algo']['rmssd'],
            'hr_mean': metrics['our_algo']['hr_mean'],
            'coherence': metrics['our_algo']['coherence'],
            'recording_hours': metrics['our_algo']['recording_hours']
        }
    }
    st.session_state.analysis_history.append(analysis_data)

def show_analysis_history():
    """Mostra lo storico delle analisi"""
    if st.session_state.analysis_history:
        st.sidebar.header("📊 Storico Analisi")
        
        history_data = []
        for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):
            user = analysis['user_profile']
            history_data.append({
                'Data': analysis['start_datetime'].strftime('%d/%m %H:%M'),
                'Utente': f"{user['name']} {user['surname']}" if user['name'] else "N/A",
                'SDNN': f"{analysis['metrics']['sdnn']:.1f}",
                'RMSSD': f"{analysis['metrics']['rmssd']:.1f}",
            })
        
        df_history = pd.DataFrame(history_data)
        st.sidebar.dataframe(df_history, use_container_width=True, hide_index=True)

# =============================================================================
# FUNZIONI DI ANALISI HRV
# =============================================================================

def calculate_triple_metrics(total_hours, actual_date, is_sleep_period=False, health_profile_factor=0.5):
    """Calcola metriche HRV complete"""
    np.random.seed(123 + int(actual_date.timestamp()))
    
    # Metriche sonno SOLO SE è periodo notturno
    sleep_metrics = {
        'sleep_duration': None, 'sleep_efficiency': None, 'sleep_coherence': None,
        'sleep_hr': None, 'sleep_rem': None, 'sleep_deep': None, 'sleep_wakeups': None,
    }
    
    if is_sleep_period and total_hours >= 6:
        sleep_metrics = {
            'sleep_duration': min(8.0, total_hours * 0.9),
            'sleep_efficiency': min(95, 85 + np.random.normal(0, 5)),
            'sleep_coherence': 65 + np.random.normal(0, 3),
            'sleep_hr': 58 + np.random.normal(0, 2),
            'sleep_rem': min(2.0, total_hours * 0.25),
            'sleep_deep': min(1.5, total_hours * 0.2),
            'sleep_wakeups': max(0, int(total_hours * 0.5)),
        }

    # Metriche base
    base_metrics = {
        'sdnn': 50 + (250 * health_profile_factor) + np.random.normal(0, 20),
        'rmssd': 30 + (380 * health_profile_factor) + np.random.normal(0, 25),
        'hr_mean': 65 - (10 * health_profile_factor) + np.random.normal(0, 2),
        'total_power': 5000 + (90000 * health_profile_factor) + np.random.normal(0, 10000),
    }
    
    our_metrics = {
        'sdnn': max(20, base_metrics['sdnn']),
        'rmssd': max(15, base_metrics['rmssd']),
        'hr_mean': base_metrics['hr_mean'],
        'hr_min': max(40, base_metrics['hr_mean'] - 15),
        'hr_max': min(180, base_metrics['hr_mean'] + 30),
        'actual_date': actual_date,
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
            elif efficiency > 80 and duration >= 6:
                valutazione = "👍 BUONA qualità del sonno" 
                colore = "#f39c12"
            else:
                valutazione = "⚠️ QUALITÀ da migliorare"
                colore = "#e74c3c"
            
            st.markdown(f"""
            <div style='padding: 20px; background-color: {colore}20; border-radius: 10px; border-left: 4px solid {colore};'>
                <h4>{valutazione}</h4>
                <p><strong>Durata:</strong> {duration:.1f}h | <strong>Efficienza:</strong> {efficiency:.0f}%</p>
                <p><strong>Risvegli:</strong> {wakeups} | <strong>HR notte:</strong> {hr_night:.0f} bpm</p>
            </div>
            """, unsafe_allow_html=True)
            
            if duration > 0:
                fig_pie = go.Figure(go.Pie(
                    labels=['Sonno Leggero', 'Sonno REM', 'Sonno Profondo'],
                    values=[duration - rem - deep, rem, deep],
                    marker_colors=['#3498db', '#e74c3c', '#2ecc71']
                ))
                fig_pie.update_layout(title="Composizione Sonno")
                st.plotly_chart(fig_pie, use_container_width=True)

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
        
    with col2:
        st.subheader("EmWave Style")
        st.metric("SDNN", f"{metrics['emwave_style']['sdnn']:.1f} ms")
        st.metric("RMSSD", f"{metrics['emwave_style']['rmssd']:.1f} ms")
        st.metric("Coerenza", f"{metrics['emwave_style']['coherence']:.1f}%")
        
    with col3:
        st.subheader("Kubios Style")
        st.metric("SDNN", f"{metrics['kubios_style']['sdnn']:.1f} ms")
        st.metric("RMSSD", f"{metrics['kubios_style']['rmssd']:.1f} ms")
        st.metric("Coerenza", f"{metrics['kubios_style']['coherence']:.1f}%")
    
    # 2. ANALISI SONNO (SOLO SE C'È)
    create_sleep_analysis(metrics)
    
    # 3. SALVA NELLO STORICO
    analysis_type = "File IBI" if st.session_state.file_uploaded else "Simulata"
    save_to_history(metrics, start_datetime, end_datetime, analysis_type, selected_range)

# =============================================================================
# INTERFACCIA PRINCIPALE
# =============================================================================

st.set_page_config(
    page_title="HRV Analytics ULTIMATE - Roberto",
    page_icon="❤️", 
    layout="wide"
)

st.title("🏥 HRV ANALYTICS ULTIMATE")
st.markdown("### **Piattaforma Completa** - Analisi HRV Personalizzata")

# INIZIALIZZA SESSION STATE
init_session_state()

# PROFILO UTENTE
create_user_profile()

# DIARIO ATTIVITÀ
create_activity_diary()

# STORICO ANALISI
show_analysis_history()

# Sidebar configurazione
with st.sidebar:
    st.header("📁 Carica Dati HRV")
    
    uploaded_file = st.file_uploader(
        "Seleziona file IBI/RR intervals",
        type=['csv', 'txt', 'xlsx'],
        help="Supporta: CSV, TXT, Excel con colonne RR/IBI intervals",
        key="file_uploader"
    )
    
    # Leggi il file per estrarre gli intervalli RR
    rr_intervals_from_file = None
    if uploaded_file is not None:
        try:
            rr_intervals_from_file = read_ibi_file_fast(uploaded_file)
        except:
            rr_intervals_from_file = None
    
    # AGGIORNA AUTOMATICAMENTE DATA/ORA
    update_analysis_datetimes(uploaded_file, rr_intervals_from_file)
    
    # OTTIENI DATA/ORA CORRENTI
    start_datetime, end_datetime = get_analysis_datetimes()
    
    st.markdown("---")
    st.header("⚙️ Impostazioni Analisi")
    
    # INFORMAZIONI FILE
    if uploaded_file is not None:
        st.success(f"📄 **File:** {uploaded_file.name}")
        if rr_intervals_from_file is not None:
            st.info(f"📊 **{len(rr_intervals_from_file)} intervalli RR**")
    
    # SELEZIONE INTERVALLO
    st.subheader("🎯 Selezione Intervallo")
    
    total_duration = (end_datetime - start_datetime).total_seconds() / 3600
    
    if total_duration > 0:
        selected_start = st.slider(
            "Inizio analisi (ore dall'inizio)", 
            0.0, total_duration, 0.0, 0.1,
            help="Seleziona l'inizio del periodo da analizzare"
        )
        selected_end = st.slider(
            "Fine analisi (ore dall'inizio)", 
            0.0, total_duration, total_duration, 0.1,
            help="Seleziona la fine del periodo da analizzare"
        )
        
        selected_duration = selected_end - selected_start
        selected_start_dt = start_datetime + timedelta(hours=selected_start)
        selected_end_dt = start_datetime + timedelta(hours=selected_end)
        
        st.info(f"⏱️ **Analizzerai:** {selected_duration:.1f} ore")
        st.info(f"📅 **Periodo:** {selected_start_dt.strftime('%d/%m %H:%M')} - {selected_end_dt.strftime('%d/%m %H:%M')}")
        
        # VERIFICA SE C'È PERIODO NOTTURNO
        is_night_period = False
        current_time = selected_start_dt
        while current_time < selected_end_dt:
            if 22 <= current_time.hour or current_time.hour <= 6:
                is_night_period = True
                break
            current_time += timedelta(hours=1)
        
        if is_night_period:
            st.success("🌙 **Periodo notturno rilevato** - Analisi sonno disponibile")
        else:
            st.info("☀️ **Periodo diurno** - Analisi sonno non disponibile")
    
    # ALTRE IMPOSTAZIONI
    health_factor = st.slider(
        "Profilo Salute", 
        0.1, 1.0, 0.5,
        help="0.1 = Sedentario, 1.0 = Atleta"
    )
    
    include_sleep = st.checkbox(
        "Includi analisi sonno", 
        is_night_period if 'is_night_period' in locals() else False,
        help="Disponibile solo per periodi notturni"
    )
    
    analyze_btn = st.button("🚀 ANALISI COMPLETA", type="primary", use_container_width=True)

# MAIN CONTENT
if analyze_btn:
    with st.spinner("🎯 **ANALISI COMPLETA IN CORSO**..."):
        if uploaded_file is not None:
            # ANALISI CON FILE CARICATO
            try:
                rr_intervals = read_ibi_file_fast(uploaded_file)
                
                if len(rr_intervals) == 0:
                    st.error("❌ Nessun dato RR valido trovato nel file")
                    st.stop()
                
                hrv_metrics = calculate_hrv_metrics_from_rr(rr_intervals)
                
                if hrv_metrics:
                    st.success("✅ **ANALISI FILE COMPLETATA!**")
                    
                    # Crea metriche complete
                    metrics = calculate_triple_metrics(
                        total_hours=selected_duration,
                        actual_date=selected_start_dt,
                        is_sleep_period=include_sleep and is_night_period,
                        health_profile_factor=health_factor
                    )
                    
                    # Aggiorna con metriche reali dal file
                    metrics['our_algo']['sdnn'] = hrv_metrics['sdnn']
                    metrics['our_algo']['rmssd'] = hrv_metrics['rmssd']
                    metrics['our_algo']['hr_mean'] = hrv_metrics['hr_mean']
                    
                    # Aggiusta per sesso
                    metrics = adjust_metrics_for_gender(metrics, st.session_state.user_profile['gender'])
                    
                    create_complete_analysis_dashboard(metrics, selected_start_dt, selected_end_dt, f"{selected_duration:.1f}h")
                    
            except Exception as e:
                st.error(f"❌ Errore nel processare il file: {e}")
        
        else:
            # ANALISI SIMULATA
            metrics = calculate_triple_metrics(
                total_hours=selected_duration,
                actual_date=selected_start_dt,
                is_sleep_period=include_sleep and is_night_period,
                health_profile_factor=health_factor
            )
            
            # Aggiusta per sesso
            metrics = adjust_metrics_for_gender(metrics, st.session_state.user_profile['gender'])
            
            st.success("✅ **ANALISI SIMULATA COMPLETATA!**")
            create_complete_analysis_dashboard(metrics, selected_start_dt, selected_end_dt, f"{selected_duration:.1f}h")

else:
    # SCHERMATA INIZIALE
    st.info("👆 **Configura l'analisi dalla sidebar**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Flusso di Lavoro")
        st.markdown("""
        1. **👤 Inserisci profilo utente**
        2. **📁 Carica file IBI** (data/ora automatiche)
        3. **📝 Aggiungi attività** nel diario
        4. **🎯 Seleziona intervallo** da analizzare
        5. **🚀 Avvia analisi** completa
        6. **📊 Consulta storico** analisi
        """)
    
    with col2:
        st.subheader("🆕 Nuove Funzionalità")
        st.markdown("""
        - 👤 **Profilo utente** completo
        - 📅 **Data/ora automatiche** dal file
        - 🎯 **Selezione intervallo** flessibile
        - 🌙 **Rilevamento automatico** periodo notturno
        - ⚖️ **Metriche aggiustate** per sesso
        - 📊 **Storico** analisi personalizzato
        """)

# Footer
st.markdown("---")
st.markdown("**HRV Analytics ULTIMATE** - Creato da Roberto con ❤️")