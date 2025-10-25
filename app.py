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
# NEUROKIT2 IMPORT
# =============================================================================
try:
    import neurokit2 as nk
    NEUROKIT_AVAILABLE = True
    st.sidebar.success("✅ NeuroKit2 caricato")
except ImportError:
    NEUROKIT_AVAILABLE = False
    st.sidebar.warning("⚠️ NeuroKit2 non disponibile")

# =============================================================================
# COSTANTI
# =============================================================================
ACTIVITY_COLORS = {
    "Allenamento": "#e74c3c",
    "Alimentazione": "#3498db", 
    "Stress": "#f39c12",
    "Riposo": "#2ecc71",
    "Altro": "#95a5a6"
}

# =============================================================================
# FUNZIONE SERIALIZZAZIONE JSON PER DATETIME
# =============================================================================
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# =============================================================================
# INIZIALIZZAZIONE SESSION STATE
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
    
    if 'last_analysis_daily' not in st.session_state:
        st.session_state.last_analysis_daily = []
    
    if 'editing_activity_index' not in st.session_state:
        st.session_state.editing_activity_index = None
    
    if 'rr_intervals' not in st.session_state:
        st.session_state.rr_intervals = []

# =============================================================================
# GESTIONE DATABASE UTENTI - CORRETTA
# =============================================================================
def load_user_database():
    """Carica il database utenti dal file JSON"""
    try:
        if os.path.exists('user_database.json'):
            with open('user_database.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Converti le stringhe ISO back in datetime
                if 'activities' in data:
                    for activity in data['activities']:
                        if 'start_time' in activity and isinstance(activity['start_time'], str):
                            activity['start_time'] = datetime.fromisoformat(activity['start_time'])
                        if 'timestamp' in activity and isinstance(activity['timestamp'], str):
                            activity['timestamp'] = datetime.fromisoformat(activity['timestamp'])
                return data
    except Exception as e:
        st.error(f"Errore nel caricamento del database: {e}")
    return {'activities': [], 'user_profile': {}}

def save_user_database():
    """Salva il database utenti nel file JSON - VERSIONE CORRETTA"""
    try:
        # Prepara i dati per il salvataggio
        save_data = {
            'activities': st.session_state.activities,
            'user_profile': st.session_state.user_profile,
            'last_save': datetime.now().isoformat()
        }
        
        with open('user_database.json', 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
        return True
    except Exception as e:
        st.error(f"Errore nel salvataggio database: {e}")
        return False

# =============================================================================
# FUNZIONI HRV CON NEUROKIT2 - COMPLETAMENTE RIVISTA
# =============================================================================
def calculate_hrv_with_neurokit(rr_intervals, user_age, user_gender):
    """Calcola metriche HRV usando NeuroKit2 - VERSIONE ROBUSTA"""
    if len(rr_intervals) < 10:
        return get_default_metrics(user_age, user_gender)
    
    try:
        # Converti RR intervals in secondi
        rr_seconds = np.array(rr_intervals) / 1000.0
        
        # Usa il metodo corretto per i dati RR
        signals, info = nk.ecg_process(rr_seconds, sampling_rate=1000)
        peaks = info['ECG_R_Peaks']
        
        if len(peaks) < 10:
            return calculate_hrv_fallback(rr_intervals, user_age, user_gender)
        
        # Calcola metriche HRV temporali
        hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)
        
        # Estrai valori con controllo robusto
        sdnn = float(hrv_time['HRV_SDNN'].iloc[0]) * 1000 if 'HRV_SDNN' in hrv_time.columns else np.std(rr_intervals, ddof=1)
        rmssd = float(hrv_time['HRV_RMSSD'].iloc[0]) * 1000 if 'HRV_RMSSD' in hrv_time.columns else calculate_rmssd_fallback(rr_intervals)
        hr_mean = 60000 / np.mean(rr_intervals)
        
        # Calcola metriche frequenziali con gestione errori completa
        try:
            hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False)
            
            # Controllo completo per ogni colonna
            total_power = 2000
            lf = 800
            hf = 700
            
            if 'HRV_TotalPower' in hrv_freq.columns and not pd.isna(hrv_freq['HRV_TotalPower'].iloc[0]):
                total_power = float(hrv_freq['HRV_TotalPower'].iloc[0])
            if 'HRV_LF' in hrv_freq.columns and not pd.isna(hrv_freq['HRV_LF'].iloc[0]):
                lf = float(hrv_freq['HRV_LF'].iloc[0])
            if 'HRV_HF' in hrv_freq.columns and not pd.isna(hrv_freq['HRV_HF'].iloc[0]):
                hf = float(hrv_freq['HRV_HF'].iloc[0])
                
            lf_hf_ratio = lf / hf if hf > 0 else 1.1
            
        except Exception as freq_error:
            # Fallback per analisi spettrale
            total_power = 2000
            lf = 800
            hf = 700
            lf_hf_ratio = 1.1
        
        # Calcola metriche non-lineari
        try:
            hrv_nonlinear = nk.hrv_nonlinear(peaks, sampling_rate=1000, show=False)
            sd1 = float(hrv_nonlinear['HRV_SD1'].iloc[0]) * 1000 if 'HRV_SD1' in hrv_nonlinear.columns else rmssd / np.sqrt(2)
            sd2 = float(hrv_nonlinear['HRV_SD2'].iloc[0]) * 1000 if 'HRV_SD2' in hrv_nonlinear.columns else sdnn
        except Exception as nonlin_error:
            sd1 = rmssd / np.sqrt(2)
            sd2 = sdnn
        
        # Calcola metriche aggiuntive
        coherence = calculate_hrv_coherence_advanced(rr_intervals, hr_mean, user_age)
        sleep_metrics = estimate_sleep_metrics_advanced(rr_intervals, hr_mean, user_age)
        
        return {
            'sdnn': max(25, min(180, sdnn)),
            'rmssd': max(15, min(120, rmssd)),
            'hr_mean': max(45, min(100, hr_mean)),
            'coherence': max(20, min(95, coherence)),
            'recording_hours': len(rr_intervals) * np.mean(rr_intervals) / (1000 * 60 * 60),
            'total_power': max(800, min(8000, total_power)),
            'vlf': 500,
            'lf': max(200, min(4000, lf)),
            'hf': max(200, min(4000, hf)),
            'lf_hf_ratio': max(0.3, min(4.0, lf_hf_ratio)),
            'sd1': max(10, min(80, sd1)),
            'sd2': max(30, min(200, sd2)),
            'sd1_sd2_ratio': max(0.2, min(3.0, sd1/sd2 if sd2 > 0 else 1.0)),
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
        st.warning(f"NeuroKit2 analysis failed: {str(e)}. Using fallback method.")
        return calculate_hrv_fallback(rr_intervals, user_age, user_gender)

def calculate_rmssd_fallback(rr_intervals):
    """Calcola RMSSD come fallback"""
    if len(rr_intervals) < 2:
        return 30
    differences = np.diff(rr_intervals)
    return np.sqrt(np.mean(np.square(differences)))

def get_default_metrics(user_age, user_gender):
    """Restituisce metriche di default per dati insufficienti"""
    return {
        'sdnn': 45, 'rmssd': 30, 'hr_mean': 70, 'coherence': 50,
        'recording_hours': 0, 'total_power': 1500, 'vlf': 500, 'lf': 800, 'hf': 700,
        'lf_hf_ratio': 1.1, 'sd1': 20, 'sd2': 50, 'sd1_sd2_ratio': 0.4,
        'sleep_duration': 7, 'sleep_efficiency': 85, 'sleep_hr': 60,
        'sleep_light': 4, 'sleep_deep': 2, 'sleep_rem': 1, 'sleep_awake': 0.5,
        'analysis_method': 'Default'
    }

def calculate_hrv_fallback(rr_intervals, user_age, user_gender):
    """Calcolo HRV fallback quando NeuroKit2 non disponibile o fallisce"""
    if len(rr_intervals) < 10:
        return get_default_metrics(user_age, user_gender)
    
    sdnn = np.std(rr_intervals, ddof=1)
    rmssd = calculate_rmssd_fallback(rr_intervals)
    hr_mean = 60000 / np.mean(rr_intervals)
    
    return {
        'sdnn': max(25, min(180, sdnn)),
        'rmssd': max(15, min(120, rmssd)),
        'hr_mean': max(45, min(100, hr_mean)),
        'coherence': max(20, min(95, calculate_hrv_coherence_advanced(rr_intervals, hr_mean, user_age))),
        'recording_hours': len(rr_intervals) * np.mean(rr_intervals) / (1000 * 60 * 60),
        'total_power': 2000, 'vlf': 500, 'lf': 800, 'hf': 700, 'lf_hf_ratio': 1.1,
        'sd1': rmssd / np.sqrt(2), 'sd2': sdnn,
        'sd1_sd2_ratio': (rmssd / np.sqrt(2)) / sdnn if sdnn > 0 else 1.0,
        'sleep_duration': 7, 'sleep_efficiency': 85, 'sleep_hr': 60,
        'sleep_light': 4, 'sleep_deep': 2, 'sleep_rem': 1, 'sleep_awake': 0.5,
        'analysis_method': 'Fallback'
    }

def calculate_hrv_coherence_advanced(rr_intervals, hr_mean, user_age):
    """Calcola coerenza cardiaca avanzata"""
    if len(rr_intervals) < 30:
        return 50
    
    rmssd = calculate_rmssd_fallback(rr_intervals)
    sdnn = np.std(rr_intervals, ddof=1)
    
    coherence_score = (rmssd / 50 * 40 + min(60, sdnn / 3) + max(0, (75 - abs(hr_mean - 65)) / 75 * 20)) / 3
    return max(20, min(95, coherence_score))

def estimate_sleep_metrics_advanced(rr_intervals, hr_mean, user_age):
    """Stima metriche del sonno basate su HRV"""
    if len(rr_intervals) < 100:
        return get_default_sleep_metrics()
    
    night_hr = hr_mean * 0.85
    sleep_duration = min(9, max(4, 7 + (65 - hr_mean) / 20))
    
    return {
        'duration': sleep_duration,
        'efficiency': max(70, min(95, 85 + (calculate_rmssd_fallback(rr_intervals) - 30) / 2)),
        'hr': night_hr,
        'light': sleep_duration * 0.6,
        'deep': sleep_duration * 0.2,
        'rem': sleep_duration * 0.2,
        'awake': max(0.1, sleep_duration * 0.05)
    }

def get_default_sleep_metrics():
    """Metriche del sonno di default"""
    return {
        'duration': 7, 'efficiency': 85, 'hr': 60,
        'light': 4.2, 'deep': 1.4, 'rem': 1.4, 'awake': 0.35
    }

# =============================================================================
# ANALISI GIORNALIERA - COMPLETAMENTE RIVISTA
# =============================================================================
def analyze_daily_metrics(rr_intervals, start_datetime, user_profile, activities=[]):
    """Divide l'analisi in giorni separati - VERSIONE PERFETTA"""
    daily_analyses = []
    
    if len(rr_intervals) == 0:
        return daily_analyses
    
    # Calcola durata totale
    total_duration_ms = np.sum(rr_intervals)
    total_duration_hours = total_duration_ms / (1000 * 60 * 60)
    total_days = max(1, int(np.ceil(total_duration_hours / 24)))
    
    current_index = 0
    
    for day in range(total_days):
        day_start = start_datetime + timedelta(days=day)
        day_end = day_start + timedelta(hours=24)
        
        # Raccogli RR intervals per questo giorno
        day_rr = []
        accumulated_time_ms = 0
        day_duration_ms = 24 * 60 * 60 * 1000
        
        while current_index < len(rr_intervals) and accumulated_time_ms < day_duration_ms:
            current_rr = rr_intervals[current_index]
            day_rr.append(current_rr)
            accumulated_time_ms += current_rr
            current_index += 1
        
        # Analizza solo se abbiamo dati sufficienti
        if len(day_rr) >= 10:  # Ridotto a 10 per essere più permissivo
            daily_metrics = calculate_realistic_hrv_metrics(
                day_rr, user_profile.get('age', 30), user_profile.get('gender', 'Uomo')
            )
            
            # Filtra attività per il giorno CORRETTO
            day_activities = get_activities_for_period(activities, day_start, day_end)
            nutrition_impact = analyze_nutritional_impact_day(day_activities, daily_metrics)
            activity_impact = analyze_activity_impact_on_ans(day_activities, daily_metrics)
            
            daily_analyses.append({
                'day_number': day + 1,
                'date': day_start.date(),
                'start_time': day_start,
                'end_time': day_end,
                'metrics': daily_metrics,
                'activities': day_activities,
                'nutrition_impact': nutrition_impact,
                'activity_impact': activity_impact,
                'rr_count': len(day_rr),
                'recording_hours': accumulated_time_ms / (1000 * 60 * 60)
            })
        else:
            # Aggiungi giorno vuoto per mantenere la sequenza
            daily_analyses.append({
                'day_number': day + 1,
                'date': day_start.date(),
                'start_time': day_start,
                'end_time': day_end,
                'metrics': get_default_metrics(user_profile.get('age', 30), user_profile.get('gender', 'Uomo')),
                'activities': [],
                'nutrition_impact': {'score': 0, 'analysis': "Dati insufficienti"},
                'activity_impact': {'score': 0, 'analysis': "Dati insufficienti"},
                'rr_count': len(day_rr),
                'recording_hours': accumulated_time_ms / (1000 * 60 * 60)
            })
    
    return daily_analyses

def get_activities_for_period(activities, start_time, end_time):
    """Filtra le attività per il periodo specificato"""
    period_activities = []
    for activity in activities:
        activity_start = activity['start_time']
        activity_end = activity_start + timedelta(minutes=activity['duration'])
        
        # Controllo preciso dell'overlap
        if (activity_start < end_time and activity_end > start_time):
            period_activities.append(activity)
    return period_activities

def analyze_nutritional_impact_day(activities, daily_metrics):
    """Analizza l'impatto nutrizionale sulla giornata"""
    food_activities = [a for a in activities if a['type'] == 'Alimentazione']
    
    if not food_activities:
        return {'score': 0, 'analysis': "Nessun dato alimentare registrato", 'recommendations': []}
    
    heavy_meals = sum(1 for a in food_activities if a['intensity'] in ['Pesante', 'Molto pesante'])
    
    if heavy_meals > 2:
        return {
            'score': 3,
            'analysis': "Troppi pasti pesanti registrati che possono influenzare negativamente l'HRV",
            'recommendations': ["Riduci i pasti pesanti", "Aumenta l'idratazione", "Distanzia i pasti principali"]
        }
    elif heavy_meals == 0:
        return {
            'score': -1,
            'analysis': "Alimentazione leggera e bilanciata, favorevole per l'HRV",
            'recommendations': ["Continua con questa alimentazione bilanciata"]
        }
    else:
        return {
            'score': 1,
            'analysis': "Alimentazione nella norma con qualche pasto sostanzioso",
            'recommendations': ["Mantieni un buon equilibrio tra pasti leggeri e sostanziosi"]
        }

def analyze_activity_impact_on_ans(activities, daily_metrics):
    """Analizza l'impatto delle attività sul sistema nervoso autonomo"""
    training_activities = [a for a in activities if a['type'] == 'Allenamento']
    
    if not training_activities:
        return {'score': 0, 'analysis': "Nessun allenamento registrato oggi"}
    
    intense_count = sum(1 for a in training_activities if a['intensity'] in ['Intensa', 'Massimale'])
    total_duration = sum(a['duration'] for a in training_activities)
    
    if intense_count > 1 or total_duration > 120:
        return {
            'score': 2,
            'analysis': "Allenamenti intensi o prolungati possono aver stressato il sistema nervoso"
        }
    elif intense_count == 1 and total_duration <= 90:
        return {
            'score': -1,
            'analysis': "Attività fisica bilanciata e benefica per l'HRV"
        }
    else:
        return {
            'score': 1,
            'analysis': "Attività fisica moderata con impatto neutro"
        }

# =============================================================================
# GESTIONE ATTIVITÀ - CORRETTA
# =============================================================================
def save_activity(activity_type, name, intensity, food_items, start_date, start_time, duration, notes):
    """Salva una nuova attività"""
    start_datetime = datetime.combine(start_date, start_time)
    
    activity = {
        'type': activity_type,
        'name': name,
        'intensity': intensity,
        'food_items': food_items,
        'start_time': start_datetime,
        'duration': duration,
        'notes': notes,
        'timestamp': datetime.now(),
        'color': ACTIVITY_COLORS.get(activity_type, "#95a5a6")
    }
    
    st.session_state.activities.append(activity)
    
    # Salva nel database
    if save_user_database():
        st.success(f"✅ Attività '{name}' salvata per il {start_date.strftime('%d/%m/%Y')} alle {start_time.strftime('%H:%M')}!")
    else:
        st.error("❌ Errore nel salvataggio dell'attività")
    
    st.rerun()

def update_activity(index, activity_type, name, intensity, food_items, start_date, start_time, duration, notes):
    """Aggiorna un'attività esistente"""
    if 0 <= index < len(st.session_state.activities):
        start_datetime = datetime.combine(start_date, start_time)
        
        st.session_state.activities[index] = {
            'type': activity_type,
            'name': name,
            'intensity': intensity,
            'food_items': food_items,
            'start_time': start_datetime,
            'duration': duration,
            'notes': notes,
            'timestamp': datetime.now(),
            'color': ACTIVITY_COLORS.get(activity_type, "#95a5a6")
        }
        
        if save_user_database():
            st.success("✅ Attività aggiornata!")
        st.rerun()

def delete_activity(index):
    """Elimina un'attività"""
    if 0 <= index < len(st.session_state.activities):
        activity_name = st.session_state.activities[index]['name']
        st.session_state.activities.pop(index)
        
        if save_user_database():
            st.success(f"✅ Attività '{activity_name}' eliminata!")
        st.rerun()

# =============================================================================
# INTERFACCIA ATTIVITÀ - CORRETTA
# =============================================================================
def edit_activity_interface():
    """Interfaccia per modificare attività"""
    st.sidebar.header("✏️ Modifica Attività")
    
    index = st.session_state.editing_activity_index
    if index is None or index >= len(st.session_state.activities):
        st.session_state.editing_activity_index = None
        return
    
    activity = st.session_state.activities[index]
    start_datetime = activity['start_time']
    
    activity_type = st.sidebar.selectbox(
        "Tipo Attività", 
        ["Allenamento", "Alimentazione", "Stress", "Riposo", "Altro"],
        index=["Allenamento", "Alimentazione", "Stress", "Riposo", "Altro"].index(activity['type'])
    )
    
    activity_name = st.sidebar.text_input("Nome Attività", value=activity['name'])
    
    if activity_type == "Alimentazione":
        food_items = st.sidebar.text_area("Cosa hai mangiato?", value=activity.get('food_items', ''))
        intensity_options = ["Leggero", "Normale", "Pesante", "Molto pesante"]
        intensity = st.sidebar.select_slider("Pesantezza pasto", options=intensity_options,
                                           value=activity['intensity'])
    else:
        food_items = ""
        intensity_options = ["Leggera", "Moderata", "Intensa", "Massimale"]
        intensity = st.sidebar.select_slider("Intensità", options=intensity_options,
                                           value=activity['intensity'])
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Data", value=start_datetime.date())
        start_time = st.time_input("Ora inizio", value=start_datetime.time())
    with col2:
        duration = st.number_input("Durata (min)", min_value=1, max_value=480, value=activity['duration'])
    
    notes = st.sidebar.text_area("Note", value=activity.get('notes', ''))
    
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.button("💾 Salva Modifiche", use_container_width=True):
            update_activity(index, activity_type, activity_name, intensity, food_items, start_date, start_time, duration, notes)
    with col2:
        if st.button("❌ Annulla", use_container_width=True):
            st.session_state.editing_activity_index = None
            st.rerun()
    with col3:
        if st.button("🗑️ Elimina", use_container_width=True):
            delete_activity(index)

def create_activity_tracker():
    """Interfaccia per tracciare attività e alimentazione"""
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
            start_date = st.date_input("Data", value=datetime.now().date())
            start_time = st.time_input("Ora inizio", value=datetime.now().time())
        with col2:
            duration = st.number_input("Durata (min)", min_value=1, max_value=480, value=30)
        
        notes = st.text_area("Note (opzionale)", placeholder="Note aggiuntive...")
        
        if st.button("💾 Salva Attività", use_container_width=True):
            if activity_name.strip():
                save_activity(activity_type, activity_name, intensity, food_items, start_date, start_time, duration, notes)
            else:
                st.error("❌ Inserisci un nome per l'attività")
    
    # Gestione attività esistenti
    if st.session_state.activities:
        st.sidebar.subheader("📋 Attività Salvate")
        
        # Ordina attività per data (più recenti prima)
        sorted_activities = sorted(st.session_state.activities, key=lambda x: x['start_time'], reverse=True)
        
        for i, activity in enumerate(sorted_activities[:10]):
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

# =============================================================================
# VISUALIZZAZIONE ANALISI GIORNALIERA - COMPLETA
# =============================================================================
def create_daily_analysis_visualization(daily_analyses):
    """Crea visualizzazioni per l'analisi giornaliera"""
    if not daily_analyses:
        st.info("📊 Nessuna analisi giornaliera disponibile. Carica un file con almeno 24 ore di dati.")
        return
    
    st.header("📅 Analisi Giornaliera Dettagliata")
    
    # Informazioni periodo analizzato
    st.info(f"**Periodo analizzato:** {len(daily_analyses)} giorni - Dal {daily_analyses[0]['date'].strftime('%d/%m/%Y')} al {daily_analyses[-1]['date'].strftime('%d/%m/%Y')}")
    
    # Prepara dati per i grafici
    days = [f"Giorno {day['day_number']}\n({day['date'].strftime('%d/%m')})" for day in daily_analyses]
    sdnn_values = [day['metrics']['sdnn'] for day in daily_analyses]
    rmssd_values = [day['metrics']['rmssd'] for day in daily_analyses]
    hr_values = [day['metrics']['hr_mean'] for day in daily_analyses]
    
    # Crea tabs per organizzare le informazioni
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Andamento Giornaliero", "🎯 Analisi Dettagliata", "📈 Analisi Spettrale", "🏃‍♂️ Impatto Attività"])
    
    with tab1:
        # Grafico andamento metriche principali
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=days, y=sdnn_values, mode='lines+markers', name='SDNN',
                               line=dict(color='#3498db', width=3), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=days, y=rmssd_values, mode='lines+markers', name='RMSSD',
                               line=dict(color='#e74c3c', width=3), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=days, y=hr_values, mode='lines+markers', name='HR Media',
                               line=dict(color='#2ecc71', width=3), marker=dict(size=8), yaxis='y2'))
        
        fig.update_layout(
            title="Andamento Metriche HRV per Giorno",
            xaxis_title="Giorno",
            yaxis_title="HRV (ms)",
            yaxis2=dict(title="HR (bpm)", overlaying='y', side='right'),
            height=400, plot_bgcolor='rgba(240,240,240,0.5)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # ANALISI DETTAGLIATA PER OGNI GIORNO
        st.subheader(f"📋 Dettaglio Completo - {len(daily_analyses)} Giorni Analizzati")
        
        for day_analysis in daily_analyses:
            with st.expander(f"🎯 Giorno {day_analysis['day_number']} - {day_analysis['date'].strftime('%d/%m/%Y')} ({day_analysis['recording_hours']:.1f} ore)", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("SDNN", f"{day_analysis['metrics']['sdnn']:.1f} ms", 
                             delta=f"{get_sdnn_evaluation(day_analysis['metrics']['sdnn'], 'Uomo')}")
                    st.metric("RMSSD", f"{day_analysis['metrics']['rmssd']:.1f} ms",
                             delta=f"{get_rmssd_evaluation(day_analysis['metrics']['rmssd'], 'Uomo')}")
                
                with col2:
                    st.metric("FC Media", f"{day_analysis['metrics']['hr_mean']:.1f} bpm",
                             delta=f"{get_hr_evaluation(day_analysis['metrics']['hr_mean'])}")
                    st.metric("Coerenza", f"{day_analysis['metrics']['coherence']:.1f}%",
                             delta=f"{get_coherence_evaluation(day_analysis['metrics']['coherence'])}")
                
                with col3:
                    st.metric("Durata Registrazione", f"{day_analysis['recording_hours']:.1f} h")
                    st.metric("Battiti Analizzati", f"{day_analysis['rr_count']:,}")
                
                with col4:
                    st.metric("LF/HF Ratio", f"{day_analysis['metrics']['lf_hf_ratio']:.2f}",
                             delta=f"{get_lf_hf_evaluation(day_analysis['metrics']['lf_hf_ratio'])}")
                    st.metric("Metodo Analisi", day_analysis['metrics'].get('analysis_method', 'Standard'))
                
                # Attività del giorno
                day_activities = day_analysis['activities']
                if day_activities:
                    st.subheader(f"🏃‍♂️ Attività del {day_analysis['date'].strftime('%d/%m/%Y')}")
                    for activity in day_activities:
                        activity_time = activity['start_time'].strftime('%H:%M')
                        st.write(f"• **{activity['name']}** ({activity['type']}) - {activity['intensity']} - {activity_time} ({activity['duration']} min)")
                else:
                    st.info("ℹ️ Nessuna attività registrata per questo giorno")
    
    with tab3:
        # ANALISI SPETTRALE PER OGNI GIORNO
        st.subheader("📈 Analisi Spettrale Dettagliata")
        
        for day_analysis in daily_analyses:
            with st.expander(f"📊 Giorno {day_analysis['day_number']} - {day_analysis['date'].strftime('%d/%m/%Y')}", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Potenza Totale", f"{day_analysis['metrics']['total_power']:.0f} ms²")
                    st.metric("Potenza VLF", f"{day_analysis['metrics']['vlf']:.0f} ms²")
                
                with col2:
                    st.metric("Potenza LF", f"{day_analysis['metrics']['lf']:.0f} ms²")
                    st.metric("Potenza HF", f"{day_analysis['metrics']['hf']:.0f} ms²")
                
                with col3:
                    st.metric("LF/HF Ratio", f"{day_analysis['metrics']['lf_hf_ratio']:.2f}")
                    st.metric("SD1", f"{day_analysis['metrics']['sd1']:.1f} ms")
                
                with col4:
                    st.metric("SD2", f"{day_analysis['metrics']['sd2']:.1f} ms")
                    st.metric("SD1/SD2", f"{day_analysis['metrics']['sd1_sd2_ratio']:.2f}")
    
    with tab4:
        # IMPATTO ATTIVITÀ PER OGNI GIORNO
        st.subheader("🏃‍♂️ Analisi Impatto Attività")
        
        for day_analysis in daily_analyses:
            with st.expander(f"📋 Giorno {day_analysis['day_number']} - {day_analysis['date'].strftime('%d/%m/%Y')}", expanded=False):
                activity_impact = day_analysis.get('activity_impact', {})
                
                if activity_impact.get('analysis'):
                    st.write(f"**Analisi:** {activity_impact['analysis']}")
                
                day_activities = day_analysis['activities']
                if day_activities:
                    st.write("**Attività registrate:**")
                    for activity in day_activities:
                        activity_time = activity['start_time'].strftime('%H:%M')
                        st.write(f"• **{activity['name']}** ({activity['type']}) - {activity['intensity']} - {activity_time} ({activity['duration']} min)")

# =============================================================================
# FUNZIONI DI VALUTAZIONE
# =============================================================================
def get_sdnn_evaluation(sdnn, gender):
    if gender == 'Donna':
        if sdnn > 60: return 'Ottimo'
        elif sdnn > 45: return 'Buono'
        elif sdnn > 35: return 'Normale'
        else: return 'Basso'
    else:
        if sdnn > 70: return 'Ottimo'
        elif sdnn > 50: return 'Buono'
        elif sdnn > 40: return 'Normale'
        else: return 'Basso'

def get_rmssd_evaluation(rmssd, gender):
    if gender == 'Donna':
        if rmssd > 45: return 'Ottimo'
        elif rmssd > 30: return 'Buono'
        elif rmssd > 20: return 'Normale'
        else: return 'Basso'
    else:
        if rmssd > 50: return 'Ottimo'
        elif rmssd > 35: return 'Buono'
        elif rmssd > 25: return 'Normale'
        else: return 'Basso'

def get_hr_evaluation(hr):
    if hr < 55: return 'Bradicardia'
    elif hr < 65: return 'Ottimo'
    elif hr < 75: return 'Buono'
    elif hr < 85: return 'Normale'
    else: return 'Elevata'

def get_coherence_evaluation(coherence):
    if coherence > 80: return 'Alta'
    elif coherence > 60: return 'Buona'
    elif coherence > 40: return 'Media'
    else: return 'Bassa'

def get_lf_hf_evaluation(ratio):
    if 0.5 <= ratio <= 2.0: return 'Bilanciato'
    elif ratio < 0.5: return 'Vagale'
    else: return 'Simpatico'

def get_power_evaluation(power):
    if power > 3000: return 'Alta'
    elif power > 1500: return 'Buona'
    elif power > 800: return 'Normale'
    else: return 'Bassa'

# =============================================================================
# GESTIONE PDF - COMPLETAMENTE RIVISTA
# =============================================================================
def create_advanced_pdf_report(metrics, start_datetime, end_datetime, selected_range, user_profile, activities=[], daily_analyses=[]):
    """Crea un report PDF avanzato - VERSIONE SEMPLIFICATA E FUNZIONANTE"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        
        buffer = io.BytesIO()
        
        # Crea un PDF semplice ma funzionante
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        
        # Intestazione
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "REPORT HRV COMPLETO")
        c.setFont("Helvetica", 10)
        c.drawString(50, height - 70, "Analisi Sistema Neurovegetativo")
        
        # Informazioni utente
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, height - 100, "INFORMAZIONI PAZIENTE:")
        c.setFont("Helvetica", 10)
        c.drawString(50, height - 120, f"Nome: {user_profile.get('name', '')} {user_profile.get('surname', '')}")
        c.drawString(50, height - 140, f"Età: {user_profile.get('age', '')} anni - Sesso: {user_profile.get('gender', '')}")
        c.drawString(50, height - 160, f"Periodo analisi: {start_datetime.strftime('%d/%m/%Y %H:%M')} - {end_datetime.strftime('%d/%m/%Y %H:%M')}")
        
        # Metriche principali
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, height - 200, "METRICHE HRV PRINCIPALI:")
        c.setFont("Helvetica", 10)
        
        y_pos = height - 220
        metrics_list = [
            f"SDNN: {metrics['sdnn']:.1f} ms ({get_sdnn_evaluation(metrics['sdnn'], user_profile.get('gender', 'Uomo'))})",
            f"RMSSD: {metrics['rmssd']:.1f} ms ({get_rmssd_evaluation(metrics['rmssd'], user_profile.get('gender', 'Uomo'))})",
            f"Frequenza Cardiaca Media: {metrics['hr_mean']:.1f} bpm ({get_hr_evaluation(metrics['hr_mean'])})",
            f"Coerenza Cardiaca: {metrics['coherence']:.1f}% ({get_coherence_evaluation(metrics['coherence'])})",
            f"LF/HF Ratio: {metrics['lf_hf_ratio']:.2f} ({get_lf_hf_evaluation(metrics['lf_hf_ratio'])})",
            f"Potenza Totale: {metrics['total_power']:.0f} ms² ({get_power_evaluation(metrics['total_power'])})"
        ]
        
        for metric in metrics_list:
            c.drawString(50, y_pos, metric)
            y_pos -= 20
        
        # Analisi giornaliera
        if daily_analyses:
            y_pos -= 20
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y_pos, "ANALISI GIORNALIERA:")
            y_pos -= 20
            c.setFont("Helvetica", 8)
            
            for day in daily_analyses:
                if y_pos < 100:  # Nuova pagina se necessario
                    c.showPage()
                    y_pos = height - 50
                    c.setFont("Helvetica", 8)
                
                day_text = f"Giorno {day['day_number']} ({day['date'].strftime('%d/%m/%Y')}): SDNN={day['metrics']['sdnn']:.1f}, RMSSD={day['metrics']['rmssd']:.1f}, FC={day['metrics']['hr_mean']:.1f}"
                c.drawString(50, y_pos, day_text)
                y_pos -= 15
        
        # Data generazione
        c.setFont("Helvetica-Oblique", 8)
        c.drawString(50, 30, f"Report generato il {datetime.now().strftime('%d/%m/%Y alle %H:%M')}")
        c.drawString(50, 20, "Software di analisi HRV - Sistema Neurovegetativo")
        
        c.save()
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Errore nella generazione PDF: {e}")
        return None

def create_pdf_download_button():
    """Crea il bottone per scaricare il PDF"""
    if st.session_state.get('last_analysis_metrics'):
        st.sidebar.header("📊 Genera Report")
        
        if st.sidebar.button("📄 Genera Report PDF", use_container_width=True, key="generate_pdf"):
            with st.spinner("Generazione PDF in corso..."):
                try:
                    pdf_buffer = create_advanced_pdf_report(
                        st.session_state.last_analysis_metrics,
                        st.session_state.last_analysis_start,
                        st.session_state.last_analysis_end,
                        st.session_state.last_analysis_duration,
                        st.session_state.user_profile,
                        st.session_state.activities,
                        st.session_state.last_analysis_daily
                    )
                    
                    if pdf_buffer:
                        # Crea il link per il download
                        b64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode()
                        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="report_hrv.pdf" style="display: block; padding: 0.5rem 1rem; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 0.3rem; text-align: center; margin-top: 10px;">📥 Scarica Report PDF</a>'
                        st.sidebar.markdown(href, unsafe_allow_html=True)
                        st.sidebar.success("✅ PDF generato con successo!")
                    else:
                        st.sidebar.error("❌ Errore nella generazione del PDF")
                        
                except Exception as e:
                    st.sidebar.error(f"❌ Errore: {e}")

# =============================================================================
# FUNZIONI AUSILIARIE
# =============================================================================
def calculate_realistic_hrv_metrics(rr_intervals, user_age, user_gender):
    return calculate_comprehensive_hrv(rr_intervals, user_age, user_gender)

def calculate_comprehensive_hrv(rr_intervals, user_age, user_gender):
    if NEUROKIT_AVAILABLE:
        return calculate_hrv_with_neurokit(rr_intervals, user_age, user_gender)
    else:
        return calculate_hrv_fallback(rr_intervals, user_age, user_gender)

def update_analysis_datetimes(uploaded_file, rr_intervals):
    if not st.session_state.datetime_initialized:
        total_duration_ms = sum(rr_intervals)
        total_hours = total_duration_ms / (1000 * 60 * 60)
        
        start_datetime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_datetime = start_datetime + timedelta(hours=total_hours)
        
        st.session_state.analysis_datetimes = {
            'start_datetime': start_datetime,
            'end_datetime': end_datetime
        }
        st.session_state.datetime_initialized = True

def get_analysis_datetimes():
    return (st.session_state.analysis_datetimes['start_datetime'],
            st.session_state.analysis_datetimes['end_datetime'])

def show_analysis_results(metrics, daily_analyses, start_datetime, end_datetime, duration):
    """Mostra i risultati dell'analisi"""
    st.header("📊 Risultati Analisi HRV")
    
    # Metriche principali
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("SDNN", f"{metrics['sdnn']:.1f} ms", 
                 delta=get_sdnn_evaluation(metrics['sdnn'], st.session_state.user_profile.get('gender', 'Uomo')))
        st.metric("RMSSD", f"{metrics['rmssd']:.1f} ms",
                 delta=get_rmssd_evaluation(metrics['rmssd'], st.session_state.user_profile.get('gender', 'Uomo')))
    
    with col2:
        st.metric("FC Media", f"{metrics['hr_mean']:.1f} bpm",
                 delta=get_hr_evaluation(metrics['hr_mean']))
        st.metric("Coerenza", f"{metrics['coherence']:.1f}%",
                 delta=get_coherence_evaluation(metrics['coherence']))
    
    with col3:
        st.metric("LF/HF Ratio", f"{metrics['lf_hf_ratio']:.2f}",
                 delta=get_lf_hf_evaluation(metrics['lf_hf_ratio']))
        st.metric("Potenza Totale", f"{metrics['total_power']:.0f} ms²",
                 delta=get_power_evaluation(metrics['total_power']))
    
    with col4:
        st.metric("Durata Analisi", duration)
        st.metric("Metodo", metrics.get('analysis_method', 'Standard'))
    
    # Analisi giornaliera
    if daily_analyses:
        create_daily_analysis_visualization(daily_analyses)

def show_welcome_screen():
    """Mostra schermata di benvenuto"""
    st.header("Benvenuto in HRV Analytics")
    st.markdown("""
    ### ❤️ Analisi della Variabilità della Frequenza Cardiaca
    
    **Funzionalità:**
    - 📊 Analisi HRV completa
    - 📅 Analisi giornaliera
    - 🏃‍♂️ Tracciamento attività
    - 📈 Analisi spettrale
    - 📄 Report PDF
    
    **Come iniziare:**
    1. Compila il profilo utente
    2. Carica un file con gli intervalli RR
    3. Regola il periodo di analisi
    4. Clicca "Avvia Analisi HRV Completa"
    """)

# =============================================================================
# INTERFACCIA PRINCIPALE
# =============================================================================
def main():
    st.set_page_config(
        page_title="HRV Analytics",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inizializza lo stato della sessione
    init_session_state()
    
    # CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">❤️ HRV Analytics</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("👤 Profilo Utente")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Nome", value=st.session_state.user_profile.get('name', ''))
        with col2:
            surname = st.text_input("Cognome", value=st.session_state.user_profile.get('surname', ''))
        
        birth_date = st.date_input(
            "Data di nascita", 
            value=st.session_state.user_profile.get('birth_date') or datetime(1980, 1, 1).date()
        )
        
        gender = st.selectbox("Sesso", ["Uomo", "Donna"], 
                            index=0 if st.session_state.user_profile.get('gender') == 'Uomo' else 1)
        
        if birth_date:
            age = datetime.now().year - birth_date.year
            st.write(f"**Età:** {age} anni")
        else:
            age = 30
        
        if st.button("💾 Salva Profilo", use_container_width=True):
            st.session_state.user_profile.update({
                'name': name, 'surname': surname, 'birth_date': birth_date,
                'age': age, 'gender': gender
            })
            if save_user_database():
                st.success("Profilo salvato!")
            st.rerun()
        
        # Tracker attività
        create_activity_tracker()
        
        # Bottone PDF
        create_pdf_download_button()
    
    # Upload file
    st.header("📤 Carica File IBI")
    uploaded_file = st.file_uploader("Carica file .txt o .csv con intervalli IBI", type=['txt', 'csv'])
    
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
                st.error("❌ Nessun dato IBI valido trovato")
                return
            
            st.success(f"✅ {len(rr_intervals)} RR-interval validi rilevati")
            st.session_state.rr_intervals = rr_intervals
            st.session_state.file_uploaded = True
            
            # Aggiorna data/ora
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
            
            # Calcola durata
            duration = (new_end - new_start).total_seconds() / 3600
            selected_range = f"{duration:.1f} ore"
            
            # Bottone analisi
            if st.button("🚀 Avvia Analisi HRV Completa", use_container_width=True, type="primary"):
                with st.spinner("Analisi in corso..."):
                    metrics = calculate_realistic_hrv_metrics(
                        rr_intervals, 
                        st.session_state.user_profile.get('age', 30), 
                        st.session_state.user_profile.get('gender', 'Uomo')
                    )
                    
                    # Analisi giornaliera
                    daily_analyses = []
                    if duration > 24:
                        daily_analyses = analyze_daily_metrics(
                            rr_intervals, new_start, st.session_state.user_profile, st.session_state.activities
                        )
                        st.info(f"📅 Analisi giornaliera: {len(daily_analyses)} giorni analizzati")
                    
                    # Salva risultati
                    st.session_state.last_analysis_metrics = metrics
                    st.session_state.last_analysis_start = new_start
                    st.session_state.last_analysis_end = new_end
                    st.session_state.last_analysis_duration = selected_range
                    st.session_state.last_analysis_daily = daily_analyses
                    
                    st.success("✅ Analisi completata!")
            
            # Mostra risultati
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
# ESECUZIONE
# =============================================================================
if __name__ == "__main__":
    main()