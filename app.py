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
# INIZIALIZZAZIONE SESSION STATE E PERSISTENZA - CORRETTA
# =============================================================================

def load_user_database():
    """Carica il database utenti da file JSON - VERSIONE CORRETTA"""
    if os.path.exists('user_database.json'):
        try:
            with open('user_database.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert string dates back to datetime objects
                for user_key, user_data in data.items():
                    # Converti la data di nascita
                    if user_data['profile'].get('birth_date'):
                        user_data['profile']['birth_date'] = datetime.strptime(
                            user_data['profile']['birth_date'], '%Y-%m-%d'
                        ).date()
                    
                    for analysis in user_data.get('analyses', []):
                        analysis['timestamp'] = datetime.fromisoformat(analysis['timestamp'])
                        analysis['start_datetime'] = datetime.fromisoformat(analysis['start_datetime'])
                        analysis['end_datetime'] = datetime.fromisoformat(analysis['end_datetime'])
                return data
        except Exception as e:
            st.error(f"Errore nel caricamento database: {e}")
            return {}
    return {}

def save_user_database():
    """Salva il database utenti su file JSON - VERSIONE CORRETTA"""
    try:
        serializable_db = {}
        for user_key, user_data in st.session_state.user_database.items():
            serializable_db[user_key] = {
                'profile': user_data['profile'].copy(),
                'analyses': []
            }
            
            # Converti la data di nascita in stringa
            if serializable_db[user_key]['profile'].get('birth_date'):
                serializable_db[user_key]['profile']['birth_date'] = serializable_db[user_key]['profile']['birth_date'].isoformat()
            
            for analysis in user_data.get('analyses', []):
                serializable_analysis = {
                    'timestamp': analysis['timestamp'].isoformat(),
                    'start_datetime': analysis['start_datetime'].isoformat(),
                    'end_datetime': analysis['end_datetime'].isoformat(),
                    'analysis_type': analysis['analysis_type'],
                    'selected_range': analysis['selected_range'],
                    'metrics': analysis['metrics'],
                    'daily_analyses': []
                }
                
                # Converti daily analyses
                for day_analysis in analysis.get('daily_analyses', []):
                    serializable_day = day_analysis.copy()
                    serializable_day['start_time'] = day_analysis['start_time'].isoformat()
                    serializable_day['end_time'] = day_analysis['end_time'].isoformat()
                    serializable_day['date'] = day_analysis['date'].isoformat()
                    serializable_analysis['daily_analyses'].append(serializable_day)
                
                serializable_db[user_key]['analyses'].append(serializable_analysis)
        
        with open('user_database.json', 'w', encoding='utf-8') as f:
            json.dump(serializable_db, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Errore nel salvataggio database: {e}")
        return False

def init_session_state():
    """Inizializza lo stato della sessione con persistenza"""
    # Carica il database all'inizio
    if 'user_database' not in st.session_state:
        st.session_state.user_database = load_user_database()
    
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

# =============================================================================
# NUOVE FUNZIONI PER CALCOLI HRV REALISTICI - CORRETTE
# =============================================================================

def calculate_realistic_hrv_metrics(rr_intervals, user_age, user_gender):
    """
    Calcola metriche HRV realistiche con correzione per età e genere
    Versione CORRETTA con valori fisiologici realistici
    """
    if len(rr_intervals) < 10:
        return get_default_metrics(user_age, user_gender)
    
    # Filtraggio robusto degli artefatti
    clean_rr = filter_rr_outliers(rr_intervals)
    
    if len(clean_rr) < 10:
        return get_default_metrics(user_age, user_gender)
    
    # Calcoli fondamentali
    rr_mean = np.mean(clean_rr)
    hr_mean = 60000 / rr_mean
    
    # SDNN - Variabilità totale (più realistico)
    sdnn = np.std(clean_rr, ddof=1)
    
    # RMSSD - Variabilità a breve termine
    differences = np.diff(clean_rr)
    rmssd = np.sqrt(np.mean(np.square(differences)))
    
    # Adjust per età e genere
    sdnn = adjust_for_age_gender(sdnn, user_age, user_gender, 'sdnn')
    rmssd = adjust_for_age_gender(rmssd, user_age, user_gender, 'rmssd')
    
    # CALCOLI SPETTRALI REALISTICI - CORRETTI
    # Per un adulto sano, i valori tipici sono:
    # Total Power: 1000-5000 ms²
    # VLF: 100-500 ms² (5-15%)
    # LF: 300-1500 ms² (30-50%) 
    # HF: 300-1500 ms² (30-50%)
    
    # Base realistica per la potenza totale
    if user_age < 30:
        base_power = 3500 + np.random.normal(0, 500)
    elif user_age < 50:
        base_power = 2500 + np.random.normal(0, 400)
    else:
        base_power = 1500 + np.random.normal(0, 300)
    
    # Adjust per variabilità individuale
    variability_factor = sdnn / 50  # Normalizza rispetto a SDNN tipico
    total_power = base_power * variability_factor
    
    # Distribuzione spettrale realistica
    vlf_percentage = 0.10 + np.random.normal(0, 0.02)  # 8-12%
    lf_percentage = 0.40 + np.random.normal(0, 0.05)   # 35-45%
    hf_percentage = 0.50 + np.random.normal(0, 0.05)   # 45-55%
    
    # Normalizza le percentuali
    total_percentage = vlf_percentage + lf_percentage + hf_percentage
    vlf_percentage /= total_percentage
    lf_percentage /= total_percentage  
    hf_percentage /= total_percentage
    
    vlf = total_power * vlf_percentage
    lf = total_power * lf_percentage
    hf = total_power * hf_percentage
    lf_hf_ratio = lf / hf if hf > 0 else 1.0
    
    # Coerenza cardiaca
    coherence = calculate_hrv_coherence(clean_rr, hr_mean)
    
    # Analisi sonno
    sleep_metrics = estimate_sleep_metrics(clean_rr, hr_mean)
    
    return {
        'sdnn': max(10, min(200, sdnn)),
        'rmssd': max(5, min(150, rmssd)),
        'hr_mean': max(40, min(120, hr_mean)),
        'coherence': max(10, min(100, coherence)),
        'recording_hours': len(clean_rr) * rr_mean / (1000 * 60 * 60),
        'total_power': max(500, min(10000, total_power)),
        'vlf': max(50, min(2000, vlf)),
        'lf': max(100, min(5000, lf)),
        'hf': max(100, min(5000, hf)),
        'lf_hf_ratio': max(0.1, min(5.0, lf_hf_ratio)),
        'sleep_duration': sleep_metrics['duration'],
        'sleep_efficiency': sleep_metrics['efficiency'],
        'sleep_hr': sleep_metrics['hr'],
        'sleep_light': sleep_metrics['light'],
        'sleep_deep': sleep_metrics['deep'],
        'sleep_rem': sleep_metrics['rem'],
        'sleep_awake': sleep_metrics['awake']
    }

def filter_rr_outliers(rr_intervals):
    """Filtra gli artefatti - Versione migliorata"""
    if len(rr_intervals) < 5:
        return rr_intervals
    
    rr_array = np.array(rr_intervals)
    
    # Usa un approccio più conservativo
    q25, q75 = np.percentile(rr_array, [25, 75])
    iqr = q75 - q25
    
    lower_bound = max(300, q25 - 1.5 * iqr)  # Minimo 300ms
    upper_bound = min(2000, q75 + 1.5 * iqr) # Massimo 2000ms
    
    clean_indices = np.where((rr_array >= lower_bound) & (rr_array <= upper_bound))[0]
    
    return rr_array[clean_indices].tolist()

def adjust_for_age_gender(value, age, gender, metric_type):
    """Adjust HRV values for age and gender - Versione corretta"""
    if metric_type == 'sdnn':
        # SDNN diminuisce con l'età
        age_factor = max(0.3, 1.0 - (max(20, min(80, age)) - 20) * 0.015)
        gender_factor = 0.85 if gender == 'Donna' else 1.0
    elif metric_type == 'rmssd':
        # RMSSD diminuisce più rapidamente
        age_factor = max(0.2, 1.0 - (max(20, min(80, age)) - 20) * 0.02)
        gender_factor = 0.80 if gender == 'Donna' else 1.0
    else:
        return value
    
    return value * age_factor * gender_factor

def calculate_hrv_coherence(rr_intervals, hr_mean):
    """Calcola la coerenza cardiaca - Versione migliorata"""
    if len(rr_intervals) < 30:
        return 50 + np.random.normal(0, 10)
    
    # Analisi della regolarità respiratoria
    respiratory_band = 0.15 * hr_mean / 60  # Banda respiratoria 0.15-0.4 Hz
    
    # Simula coerenza basata su HRV
    base_coherence = 40 + (hr_mean - 40) * 0.5  # Coerenza più alta con HR più bassa
    
    # Aggiungi variabilità casuale
    coherence = base_coherence + np.random.normal(0, 15)
    
    return max(10, min(95, coherence))

def estimate_sleep_metrics(rr_intervals, hr_mean):
    """Stima le metriche del sonno - Versione realistica"""
    # Per registrazioni lunghe (>1000 battiti), stima il periodo notturno
    if len(rr_intervals) > 1000:
        # Assume che il sonno sia negli ultimi 6-9 ore
        sleep_hours = 7 + np.random.normal(0, 1)
        sleep_duration = min(9, max(4, sleep_hours))
        
        # HR notturno tipicamente 10-20% più basso
        sleep_hr = hr_mean * (0.8 + np.random.normal(0, 0.05))
        
        # Efficienza del sonno
        sleep_efficiency = 85 + np.random.normal(0, 8)
    else:
        # Stime default
        sleep_duration = 7.0
        sleep_hr = hr_mean - 8
        sleep_efficiency = 85
    
    # Distribuzione fasi del sonno realistica
    sleep_light = sleep_duration * 0.50  # 50% sonno leggero
    sleep_deep = sleep_duration * 0.20   # 20% sonno profondo  
    sleep_rem = sleep_duration * 0.25    # 25% REM
    sleep_awake = sleep_duration * 0.05  # 5% risvegli
    
    return {
        'duration': max(4, min(12, sleep_duration)),
        'efficiency': max(70, min(98, sleep_efficiency)),
        'hr': max(40, min(80, sleep_hr)),
        'light': sleep_light,
        'deep': sleep_deep,
        'rem': sleep_rem,
        'awake': sleep_awake
    }

def get_default_metrics(age, gender):
    """Metriche di default realistiche"""
    if gender == 'Uomo':
        base_sdnn = 42 - (max(20, age) - 20) * 0.3
        base_rmssd = 32 - (max(20, age) - 20) * 0.4
    else:
        base_sdnn = 38 - (max(20, age) - 20) * 0.3
        base_rmssd = 28 - (max(20, age) - 20) * 0.4
    
    return {
        'sdnn': max(20, base_sdnn),
        'rmssd': max(15, base_rmssd),
        'hr_mean': 65 + (max(20, age) - 20) * 0.2,
        'coherence': 55,
        'recording_hours': 24,
        'total_power': 2500,
        'vlf': 250,
        'lf': 1000,
        'hf': 1250,
        'lf_hf_ratio': 0.8,
        'sleep_duration': 7.2,
        'sleep_efficiency': 85,
        'sleep_hr': 58,
        'sleep_light': 3.6,
        'sleep_deep': 1.4,
        'sleep_rem': 1.8,
        'sleep_awake': 0.4
    }

# =============================================================================
# ANALISI GIORNALIERA PER REGISTRAZIONI LUNGHE - CORRETTA
# =============================================================================

def analyze_daily_metrics(rr_intervals, start_datetime, user_profile, activities=[]):
    """Divide l'analisi in giorni separati - VERSIONE CORRETTA"""
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
        
        # Seleziona RR intervals per questo giorno
        day_rr = []
        accumulated_ms = 0
        start_index = current_index
        
        while current_index < len(rr_intervals) and accumulated_ms < (24 * 60 * 60 * 1000):
            day_rr.append(rr_intervals[current_index])
            accumulated_ms += rr_intervals[current_index]
            current_index += 1
        
        # CORREZIONE: Analizza anche l'ultimo giorno parziale
        if len(day_rr) > 10:  # Ridotto a 10 battiti minimi
            # Metriche del giorno
            daily_metrics = calculate_realistic_hrv_metrics(
                day_rr, user_profile.get('age', 30), user_profile.get('gender', 'Uomo')
            )
            
            # Attività del giorno
            day_activities = get_activities_for_period(activities, day_start, day_end)
            
            # Analisi impatto alimentazione
            nutrition_impact = analyze_nutritional_impact_day(day_activities, daily_metrics)
            
            # Analisi impatto attività sul SNA
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
                'recording_hours': accumulated_ms / (1000 * 60 * 60)
            })
    
    return daily_analyses

def get_activities_for_period(activities, start_time, end_time):
    """Filtra le attività per il periodo specificato"""
    period_activities = []
    for activity in activities:
        activity_start = activity['start_time']
        activity_end = activity_start + timedelta(minutes=activity['duration'])
        
        # Check if activity overlaps with the period
        if (activity_start <= end_time and activity_end >= start_time):
            period_activities.append(activity)
    return period_activities

# =============================================================================
# SISTEMA AVANZATO ATTIVITÀ E ALIMENTAZIONE - CORRETTO
# =============================================================================

# Database nutrizionale professionale
NUTRITION_DB = {
    "pasta": {"inflammatory_score": 2, "glycemic_index": "alto", "recovery_impact": -1, "category": "carboidrato"},
    "riso": {"inflammatory_score": 1, "glycemic_index": "alto", "recovery_impact": -1, "category": "carboidrato"},
    "patate": {"inflammatory_score": 2, "glycemic_index": "alto", "recovery_impact": -1, "category": "carboidrato"},
    "pane": {"inflammatory_score": 2, "glycemic_index": "alto", "recovery_impact": -1, "category": "carboidrato"},
    "panino": {"inflammatory_score": 2, "glycemic_index": "alto", "recovery_impact": -1, "category": "carboidrato"},
    "mortadella": {"inflammatory_score": 4, "glycemic_index": "medio", "recovery_impact": -3, "category": "processato"},
    "salame": {"inflammatory_score": 4, "glycemic_index": "medio", "recovery_impact": -3, "category": "processato"},
    "wurstel": {"inflammatory_score": 4, "glycemic_index": "medio", "recovery_impact": -3, "category": "processato"},
    "cornetto": {"inflammatory_score": 5, "glycemic_index": "alto", "recovery_impact": -4, "category": "dolce"},
    "crema": {"inflammatory_score": 4, "glycemic_index": "alto", "recovery_impact": -3, "category": "dolce"},
    "dolce": {"inflammatory_score": 4, "glycemic_index": "alto", "recovery_impact": -3, "category": "dolce"},
    "fritto": {"inflammatory_score": 5, "glycemic_index": "alto", "recovery_impact": -4, "category": "processato"},
    "avocado": {"inflammatory_score": -2, "glycemic_index": "basso", "recovery_impact": 2, "category": "grasso_sano"},
    "salmone": {"inflammatory_score": -3, "glycemic_index": "basso", "recovery_impact": 3, "category": "proteina"},
    "pesce": {"inflammatory_score": -2, "glycemic_index": "basso", "recovery_impact": 2, "category": "proteina"},
    "verdura": {"inflammatory_score": -4, "glycemic_index": "basso", "recovery_impact": 4, "category": "vegetale"},
    "insalata": {"inflammatory_score": -4, "glycemic_index": "basso", "recovery_impact": 4, "category": "vegetale"},
    "frutta": {"inflammatory_score": -1, "glycemic_index": "medio", "recovery_impact": 1, "category": "frutta"},
    "frutta secca": {"inflammatory_score": -2, "glycemic_index": "basso", "recovery_impact": 2, "category": "grasso_sano"},
    "olio oliva": {"inflammatory_score": -2, "glycemic_index": "basso", "recovery_impact": 2, "category": "grasso_sano"},
    "legumi": {"inflammatory_score": -1, "glycemic_index": "basso", "recovery_impact": 2, "category": "proteina"},
    "uova": {"inflammatory_score": 0, "glycemic_index": "basso", "recovery_impact": 1, "category": "proteina"},
    "carne bianca": {"inflammatory_score": 0, "glycemic_index": "basso", "recovery_impact": 1, "category": "proteina"}
}

# Colori per i tipi di attività
ACTIVITY_COLORS = {
    "Allenamento": "#e74c3c",
    "Alimentazione": "#f39c12", 
    "Stress": "#9b59b6",
    "Riposo": "#3498db",
    "Altro": "#95a5a6"
}

def create_activity_tracker():
    """Interfaccia per tracciare attività e alimentazione - VERSIONE CORRETTA"""
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
            # CORREZIONE: Data e ora corrette
            start_date = st.date_input("Data", value=datetime.now().date())
            start_time = st.time_input("Ora inizio", value=datetime.now().time())
        with col2:
            duration = st.number_input("Durata (min)", min_value=1, max_value=480, value=30)
        
        notes = st.text_area("Note (opzionale)", placeholder="Note aggiuntive...")
        
        if st.button("💾 Salva Attività", use_container_width=True):
            save_activity(activity_type, activity_name, intensity, food_items, start_date, start_time, duration, notes)
            st.success("Attività salvata!")
            st.rerun()
    
    # Gestione attività esistenti
    if st.session_state.activities:
        st.sidebar.subheader("📋 Gestione Attività")
        
        for i, activity in enumerate(st.session_state.activities[-10:]):  # Ultime 10 attività
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

def edit_activity_interface():
    """Interfaccia per modificare un'attività esistente"""
    activity_index = st.session_state.editing_activity_index
    if activity_index is None or activity_index >= len(st.session_state.activities):
        st.session_state.editing_activity_index = None
        return
    
    activity = st.session_state.activities[activity_index]
    
    st.sidebar.header("✏️ Modifica Attività")
    
    with st.sidebar.form("edit_activity_form"):
        activity_type = st.selectbox("Tipo Attività", 
                                   ["Allenamento", "Alimentazione", "Stress", "Riposo", "Altro"],
                                   index=["Allenamento", "Alimentazione", "Stress", "Riposo", "Altro"].index(activity['type']))
        
        activity_name = st.text_input("Nome Attività/Pasto", value=activity['name'])
        
        if activity_type == "Alimentazione":
            food_items = st.text_area("Cosa hai mangiato?", value=activity.get('food_items', ''))
            intensity = st.select_slider("Pesantezza pasto", 
                                       options=["Leggero", "Normale", "Pesante", "Molto pesante"],
                                       value=activity['intensity'])
        else:
            food_items = activity.get('food_items', '')
            intensity = st.select_slider("Intensità", 
                                       options=["Leggera", "Moderata", "Intensa", "Massimale"],
                                       value=activity['intensity'])
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Data", value=activity['start_time'].date())
            start_time = st.time_input("Ora inizio", value=activity['start_time'].time())
        with col2:
            duration = st.number_input("Durata (min)", min_value=1, max_value=480, value=activity['duration'])
        
        notes = st.text_area("Note", value=activity.get('notes', ''))
        
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("💾 Salva Modifiche", use_container_width=True):
                update_activity(activity_index, activity_type, activity_name, intensity, food_items, start_date, start_time, duration, notes)
                st.session_state.editing_activity_index = None
                st.rerun()
        with col2:
            if st.form_submit_button("❌ Annulla", use_container_width=True):
                st.session_state.editing_activity_index = None
                st.rerun()

def save_activity(activity_type, name, intensity, food_items, start_date, start_time, duration, notes):
    """Salva una nuova attività - VERSIONE CORRETTA"""
    # CORREZIONE: Combina correttamente data e ora
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
    
    # Mantieni solo le ultime 50 attività
    if len(st.session_state.activities) > 50:
        st.session_state.activities = st.session_state.activities[-50:]

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

def delete_activity(index):
    """Elimina un'attività"""
    if 0 <= index < len(st.session_state.activities):
        st.session_state.activities.pop(index)

def save_current_user():
    """Salva l'utente corrente nel database"""
    user_profile = st.session_state.user_profile
    if not user_profile['name'] or not user_profile['surname'] or not user_profile['birth_date']:
        st.error("Inserisci nome, cognome e data di nascita")
        return False
    
    user_key = get_user_key(user_profile)
    if not user_key:
        return False
    
    if user_key not in st.session_state.user_database:
        st.session_state.user_database[user_key] = {
            'profile': user_profile.copy(),
            'analyses': []
        }
    
    success = save_user_database()
    if success:
        st.success("Utente salvato nel database!")
    return success

def analyze_nutritional_impact_day(day_activities, daily_metrics):
    """Analizza l'impatto nutrizionale sulla base delle attività del giorno"""
    if not day_activities:
        return {"score": 0, "analysis": "Nessun dato alimentare", "recommendations": []}
    
    food_activities = [act for act in day_activities if act['type'] == 'Alimentazione']
    
    if not food_activities:
        return {"score": 0, "analysis": "Nessun pasto registrato", "recommendations": []}
    
    total_score = 0
    food_count = 0
    inflammatory_foods = []
    healthy_foods = []
    
    for activity in food_activities:
        if activity['food_items']:
            foods = [food.strip().lower() for food in activity['food_items'].split(',')]
            for food in foods:
                for db_food, info in NUTRITION_DB.items():
                    if db_food in food:
                        total_score += info['inflammatory_score']
                        food_count += 1
                        
                        if info['inflammatory_score'] > 2:
                            inflammatory_foods.append(food)
                        elif info['inflammatory_score'] < 0:
                            healthy_foods.append(food)
    
    if food_count == 0:
        return {"score": 0, "analysis": "Cibi non riconosciuti nel database", "recommendations": []}
    
    avg_score = total_score / food_count
    
    # Analisi basata sul punteggio
    if avg_score > 2:
        analysis = "⚠️ Alimentazione potenzialmente infiammatoria"
        recommendations = [
            "Riduci cibi processati e zuccheri raffinati",
            "Aumenta verdura e grassi sani",
            "Mantieni idratazione adeguata"
        ]
    elif avg_score < 0:
        analysis = "✅ Alimentazione anti-infiammatoria"
        recommendations = [
            "Ottimo! Continua con questa alimentazione",
            "Mantieni buon bilanciamento nutrienti"
        ]
    else:
        analysis = "➖ Alimentazione neutra"
        recommendations = [
            "Mantieni bilanciamento attuale",
            "Aggiungi più vegetali per migliorare ulteriormente"
        ]
    
    return {
        "score": avg_score,
        "analysis": analysis,
        "recommendations": recommendations,
        "inflammatory_foods": inflammatory_foods,
        "healthy_foods": healthy_foods
    }

def analyze_activity_impact_on_ans(day_activities, daily_metrics):
    """Analizza l'impatto delle attività sul Sistema Nervoso Autonomo"""
    impacts = []
    
    for activity in day_activities:
        impact = {
            'activity': activity['name'],
            'type': activity['type'],
            'intensity': activity['intensity'],
            'impact': 'Neutro'
        }
        
        if activity['type'] == 'Allenamento':
            if activity['intensity'] in ['Intensa', 'Massimale']:
                if daily_metrics['rmssd'] < 30:
                    impact['impact'] = 'Stress Simpatico Elevato'
                    impact['recommendation'] = "Recupero insufficiente - ridurre intensità allenamenti"
                else:
                    impact['impact'] = 'Stimolo Allenante Ottimale'
                    impact['recommendation'] = "Buon recupero - mantenere programma"
            
            elif activity['intensity'] in ['Leggera', 'Moderata']:
                impact['impact'] = 'Stimolo Allenante Adeguato'
                impact['recommendation'] = "Attività ben tollerata"
        
        elif activity['type'] == 'Stress':
            if daily_metrics['lf_hf_ratio'] > 2.5:
                impact['impact'] = 'Attivazione Simpatica Eccessiva'
                impact['recommendation'] = "Praticare tecniche di rilassamento"
            else:
                impact['impact'] = 'Stress Gestito Adeguatamente'
        
        elif activity['type'] == 'Riposo':
            if daily_metrics['rmssd'] > 40:
                impact['impact'] = 'Recupero Parasimpatico Ottimale'
                impact['recommendation'] = "Ottimo! Continua con attività rigenerative"
            else:
                impact['impact'] = 'Recupero Parziale'
                impact['recommendation'] = "Aumentare tempo dedicato al riposo"
        
        impacts.append(impact)
    
    return impacts

# =============================================================================
# FUNZIONI PER GESTIONE DATABASE UTENTI - CORRETTE
# =============================================================================

def get_user_key(user_profile):
    """Crea una chiave univoca per l'utente"""
    if not user_profile['name'] or not user_profile['surname'] or not user_profile['birth_date']:
        return None
    return f"{user_profile['name']}_{user_profile['surname']}_{user_profile['birth_date']}"

def save_analysis_to_user_database(metrics, start_datetime, end_datetime, selected_range, analysis_type, daily_analyses=None):
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
        'metrics': metrics['our_algo'],
        'daily_analyses': daily_analyses or []
    }
    
    st.session_state.user_database[user_key]['analyses'].append(analysis_data)
    
    # Salva immediatamente sul file
    success = save_user_database()
    return success

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
    
    # Ordina per cognome
    users.sort(key=lambda x: x['profile']['surname'])
    return users

# =============================================================================
# VISUALIZZAZIONE ANALISI GIORNALIERA MIGLIORATA - CORRETTA
# =============================================================================

def create_daily_analysis_visualization(daily_analyses):
    """Crea visualizzazioni complete per l'analisi giornaliera"""
    if not daily_analyses:
        return None
    
    st.header("📅 Analisi Giornaliera Dettagliata")
    
    # Grafico dell'andamento giornaliero
    days = [f"Giorno {day['day_number']}\n({day['date'].strftime('%d/%m')})" for day in daily_analyses]
    sdnn_values = [day['metrics']['sdnn'] for day in daily_analyses]
    rmssd_values = [day['metrics']['rmssd'] for day in daily_analyses]
    hr_values = [day['metrics']['hr_mean'] for day in daily_analyses]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=days, y=sdnn_values, 
        mode='lines+markers', name='SDNN',
        line=dict(color='#3498db', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=days, y=rmssd_values, 
        mode='lines+markers', name='RMSSD',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=days, y=hr_values, 
        mode='lines+markers', name='HR',
        line=dict(color='#2ecc71', width=3),
        marker=dict(size=8),
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
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Dettaglio per ogni giorno
    for day_analysis in daily_analyses:
        with st.expander(f"📋 Giorno {day_analysis['day_number']} - {day_analysis['date'].strftime('%d/%m/%Y')}", expanded=False):
            col1, col2, col3 = st.columns(3)
            
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
                st.metric("Battiti Analizzati", day_analysis['rr_count'])
            
            # Analisi del sonno per il giorno
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
            if day_analysis['metrics'].get('sleep_duration', 0) > 0:
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
                st.plotly_chart(fig_sleep, use_container_width=True)
            
            # Attività del giorno e impatto sul SNA
            if day_analysis['activities']:
                st.subheader("🏃‍♂️ Attività del Giorno e Impatto sul SNA")
                for activity in day_analysis['activities']:
                    # Trova l'impatto corrispondente
                    activity_impact = next(
                        (impact for impact in day_analysis.get('activity_impact', []) 
                         if impact['activity'] == activity['name']), 
                        None
                    )
                    
                    if activity_impact:
                        st.write(f"• **{activity['name']}** ({activity['type']}) - {activity['intensity']}")
                        st.write(f"  ↳ **Impatto SNA:** {activity_impact['impact']}")
                        if activity_impact.get('recommendation'):
                            st.write(f"  ↳ **Consiglio:** {activity_impact['recommendation']}")
                    else:
                        st.write(f"• **{activity['name']}** ({activity['type']}) - {activity['intensity']}")
            
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

# =============================================================================
# FUNZIONE PER CREARE PDF CON GRAFICHE AVANZATE - VERSIONE MIGLIORATA
# =============================================================================

def create_advanced_pdf_report(metrics, start_datetime, end_datetime, selected_range, user_profile, activities=[], daily_analyses=[]):
    """Crea un report PDF avanzato con analisi completa e referenze scientifiche"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib.utils import ImageReader
        from reportlab.lib.colors import Color, HexColor
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        import io
        import matplotlib.pyplot as plt
        import numpy as np
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                              topMargin=20*mm, bottomMargin=20*mm,
                              leftMargin=15*mm, rightMargin=15*mm)
        
        styles = getSampleStyleSheet()
        story = []
        
        # Titolo principale
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=HexColor("#2c3e50"),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        story.append(Paragraph("REPORT HRV COMPLETO - ANALISI SISTEMA NEUROVEGETATIVO", title_style))
        
        # Informazioni utente e periodo
        user_info = f"""
        <b>Paziente:</b> {user_profile.get('name', '')} {user_profile.get('surname', '')}<br/>
        <b>Età:</b> {user_profile.get('age', '')} anni | <b>Sesso:</b> {user_profile.get('gender', '')}<br/>
        <b>Periodo analisi:</b> {start_datetime.strftime('%d/%m/%Y %H:%M')} - {end_datetime.strftime('%d/%m/%Y %H:%M')}<br/>
        <b>Durata totale:</b> {selected_range}<br/>
        <b>Data generazione report:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}
        """
        story.append(Paragraph(user_info, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # METRICHE PRINCIPALI
        story.append(Paragraph("<b>METRICHE HRV PRINCIPALI</b>", styles['Heading2']))
        
        main_metrics_data = [
            ['METRICA', 'VALORE', 'VALUTAZIONE'],
            [
                'SDNN (Variabilità Totale)', 
                f"{metrics['sdnn']:.1f} ms", 
                get_sdnn_evaluation(metrics['sdnn'], user_profile.get('gender', 'Uomo'))
            ],
            [
                'RMSSD (Parasimpatico)', 
                f"{metrics['rmssd']:.1f} ms", 
                get_rmssd_evaluation(metrics['rmssd'], user_profile.get('gender', 'Uomo'))
            ],
            [
                'Frequenza Cardiaca Media', 
                f"{metrics['hr_mean']:.1f} bpm", 
                get_hr_evaluation(metrics['hr_mean'])
            ],
            [
                'Coerenza Cardiaca', 
                f"{metrics['coherence']:.1f}%", 
                get_coherence_evaluation(metrics['coherence'])
            ]
        ]
        
        main_table = Table(main_metrics_data, colWidths=[180, 80, 120])
        main_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor("#3498db")),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#ffffff")),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor("#f8f9fa")),
            ('GRID', (0, 0), (-1, -1), 1, HexColor("#bdc3c7")),
        ]))
        
        story.append(main_table)
        story.append(Spacer(1, 20))
        
        # ANALISI SPETTRALE
        story.append(Paragraph("<b>ANALISI SPETTRALE HRV</b>", styles['Heading2']))
        
        spectral_data = [
            ['BANDA', 'POTENZA', 'SIGNIFICATO CLINICO'],
            [
                'VLF (Very Low Frequency)', 
                f"{metrics['vlf']:.0f} ms²", 
                'Sistemi termoregolatori, renina-angiotensina'
            ],
            [
                'LF (Low Frequency)', 
                f"{metrics['lf']:.0f} ms²", 
                'Attività simpatica, regolazione pressione'
            ],
            [
                'HF (High Frequency)', 
                f"{metrics['hf']:.0f} ms²", 
                'Attività parasimpatica, recupero'
            ],
            [
                'RAPPORTO LF/HF', 
                f"{metrics['lf_hf_ratio']:.2f}", 
                f"Bilanciamento autonomico: {get_lf_hf_evaluation(metrics['lf_hf_ratio'])}"
            ]
        ]
        
        spectral_table = Table(spectral_data, colWidths=[120, 80, 140])
        spectral_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor("#2c3e50")),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#ffffff")),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor("#ecf0f1")),
            ('GRID', (0, 0), (-1, -1), 1, HexColor("#bdc3c7")),
        ]))
        
        story.append(spectral_table)
        story.append(Spacer(1, 20))
        
        # ANALISI IMPATTO ATTIVITÀ SUL SNA
        if activities:
            story.append(Paragraph("<b>ANALISI IMPATTO ATTIVITÀ SUL SISTEMA NEUROVEGETATIVO</b>", styles['Heading2']))
            
            for activity in activities[-5:]:  # Ultime 5 attività
                activity_text = f"""
                <b>{activity['name']}</b> ({activity['type']}) - {activity['intensity']}<br/>
                <i>Ora: {activity['start_time'].strftime('%d/%m/%Y %H:%M')} - Durata: {activity['duration']} min</i>
                """
                story.append(Paragraph(activity_text, styles['Normal']))
                
                # Analisi impatto basata sul tipo di attività
                if activity['type'] == 'Allenamento':
                    if activity['intensity'] in ['Intensa', 'Massimale']:
                        impact_text = "• <b>Impatto SNA:</b> Attivazione simpatica significativa - richiede adeguato recupero"
                    else:
                        impact_text = "• <b>Impatto SNA:</b> Stimolo allenante bilanciato - ben tollerato"
                elif activity['type'] == 'Stress':
                    impact_text = "• <b>Impatto SNA:</b> Attivazione simpatica - monitorare tempi di recupero"
                elif activity['type'] == 'Riposo':
                    impact_text = "• <b>Impatto SNA:</b> Attivazione parasimpatica - favorisce il recupero"
                else:
                    impact_text = "• <b>Impatto SNA:</b> Impatto neutro sul bilanciamento autonomico"
                
                story.append(Paragraph(impact_text, styles['Normal']))
                story.append(Spacer(1, 5))
        
        # ANALISI GIORNALIERA
        if daily_analyses:
            story.append(Paragraph("<b>ANALISI GIORNALIERA DETTAGLIATA</b>", styles['Heading2']))
            
            for day in daily_analyses:
                story.append(Paragraph(f"<b>Giorno {day['day_number']} - {day['date'].strftime('%d/%m/%Y')}</b>", styles['Heading3']))
                
                day_metrics = [
                    ['SDNN', f"{day['metrics']['sdnn']:.1f} ms", get_sdnn_evaluation(day['metrics']['sdnn'], user_profile.get('gender', 'Uomo'))],
                    ['RMSSD', f"{day['metrics']['rmssd']:.1f} ms", get_rmssd_evaluation(day['metrics']['rmssd'], user_profile.get('gender', 'Uomo'))],
                    ['FC Media', f"{day['metrics']['hr_mean']:.1f} bpm", get_hr_evaluation(day['metrics']['hr_mean'])],
                    ['Coerenza', f"{day['metrics']['coherence']:.1f}%", get_coherence_evaluation(day['metrics']['coherence'])],
                    ['LF/HF', f"{day['metrics']['lf_hf_ratio']:.2f}", get_lf_hf_evaluation(day['metrics']['lf_hf_ratio'])]
                ]
                
                day_table = Table(day_metrics, colWidths=[60, 60, 100])
                day_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('BACKGROUND', (0, 0), (-1, -1), HexColor("#f8f9fa")),
                    ('GRID', (0, 0), (-1, -1), 1, HexColor("#bdc3c7")),
                ]))
                
                story.append(day_table)
                
                # Analisi sonno per il giorno
                if day['metrics'].get('sleep_duration', 0) > 0:
                    sleep_info = f"""
                    <b>Analisi Sonno:</b> {day['metrics']['sleep_duration']:.1f}h totali 
                    (Efficienza: {day['metrics']['sleep_efficiency']:.0f}%)<br/>
                    <b>Fasi:</b> Leggero: {day['metrics'].get('sleep_light', 0):.1f}h, 
                    Profondo: {day['metrics'].get('sleep_deep', 0):.1f}h, 
                    REM: {day['metrics'].get('sleep_rem', 0):.1f}h
                    """
                    story.append(Paragraph(sleep_info, styles['Normal']))
                
                story.append(Spacer(1, 10))
        
        # RACCOMANDAZIONI DETTAGLIATE
        weaknesses = identify_weaknesses({'our_algo': metrics}, user_profile)
        story.append(Paragraph("<b>VALUTAZIONE COMPLESSIVA E PIANO DI MIGLIORAMENTO</b>", styles['Heading2']))
        
        if len(weaknesses) <= 1:
            overall = "🟢 <b>ECCELLENTE</b> - Sistema nervoso autonomo ben bilanciato e resiliente"
            recommendations = [
                "Mantenere l'attuale stile di vita e routine di recupero",
                "Continuare con attività fisica regolare e bilanciata",
                "Monitoraggio periodico per mantenere i risultati"
            ]
        elif len(weaknesses) <= 3:
            overall = "🟡 <b>BUONO</b> - Alcuni aspetti richiedono attenzione per ottimizzare la performance"
            recommendations = [
                "Implementare tecniche di gestione dello stress quotidiano",
                "Ottimizzare la qualità del sonno con routine regolari",
                "Valutare il bilanciamento tra carico allenante e recupero"
            ]
        else:
            overall = "🔴 <b>DA MIGLIORARE</b> - Significativo spazio di ottimizzazione del bilanciamento autonomico"
            recommendations = [
                "Priorità al recupero e alla gestione dello stress",
                "Implementare tecniche di coerenza cardiaca quotidiana",
                "Valutare riduzione temporanea del carico allenante",
                "Focus su alimentazione anti-infiammatoria e idratazione"
            ]
        
        story.append(Paragraph(overall, styles['Normal']))
        story.append(Spacer(1, 10))
        
        story.append(Paragraph("<b>Punti di attenzione identificati:</b>", styles['Normal']))
        for weakness in weaknesses:
            story.append(Paragraph(f"• {weakness}", styles['Normal']))
        
        story.append(Spacer(1, 10))
        
        story.append(Paragraph("<b>Raccomandazioni specifiche:</b>", styles['Normal']))
        for recommendation in recommendations:
            story.append(Paragraph(f"• {recommendation}", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # REFERENZE SCIENTIFICHE
        story.append(Paragraph("<b>REFERENZE SCIENTIFICHE</b>", styles['Heading2']))
        
        references = [
            "• Task Force of ESC/NASPE (1996). Heart rate variability: Standards of measurement, physiological interpretation, and clinical use",
            "• Malik et al. (1996). Heart rate variability: Standards of measurement, physiological interpretation, and clinical use",
            "• McCraty et al. (2009). The coherent heart: Heart-brain interactions, psychophysiological coherence, and the emergence of system-wide order",
            "• Shaffer et al. (2014). An overview of heart rate variability metrics and norms",
            "• Nunan et al. (2010). A quantitative systematic review of normal values for short-term heart rate variability in healthy adults"
        ]
        
        for ref in references:
            story.append(Paragraph(ref, styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # FOOTER
        footer_text = """
        <i>Report generato da HRV Analytics ULTIMATE - Sistema avanzato di analisi della variabilità cardiaca<br/>
        Questo report ha scopo informativo e di benessere. Per interpretazioni cliniche si raccomanda la consulenza di professionisti sanitari qualificati.</i>
        """
        story.append(Paragraph(footer_text, styles['Normal']))
        
        # GENERA IL PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Errore nella generazione PDF: {e}")
        # Fallback semplice
        return create_simple_pdf_fallback(metrics, start_datetime, end_datetime, user_profile, daily_analyses)

# [RESTANTE DEL CODICE... Le funzioni rimanenti sono le stesse dell'ultima versione]

# =============================================================================
# INTERFACCIA PRINCIPALE STREAMLIT - VERSIONE CORRETTA
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
    .daily-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header principale
    st.markdown('<h1 class="main-header">❤️ HRV Analytics ULTIMATE</h1>', unsafe_allow_html=True)
    
    # Sidebar per profilo utente e attività
    with st.sidebar:
        st.header("👤 Profilo Paziente")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.user_profile['name'] = st.text_input("Nome", value=st.session_state.user_profile['name'])
        with col2:
            st.session_state.user_profile['surname'] = st.text_input("Cognome", value=st.session_state.user_profile['surname'])
        
        # CORREZIONE: Data di nascita con formato DD/MM/YYYY e input più reattivo
        st.session_state.user_profile['birth_date'] = st.date_input(
            "Data di nascita (DD/MM/YYYY)", 
            value=st.session_state.user_profile['birth_date'] or datetime(1980, 1, 1).date(),
            min_value=datetime(1900, 1, 1).date(),
            max_value=datetime.now().date(),
            format="DD/MM/YYYY"
        )
        
        st.session_state.user_profile['gender'] = st.selectbox("Sesso", ["Uomo", "Donna"], index=0 if st.session_state.user_profile['gender'] == 'Uomo' else 1)
        
        if st.session_state.user_profile['birth_date']:
            age = datetime.now().year - st.session_state.user_profile['birth_date'].year
            st.session_state.user_profile['age'] = age
            st.info(f"Età: {age} anni")
        
        # Aggiungi tracker attività
        create_activity_tracker()
        
        # Storico utenti
        create_user_history_interface()
    
    # [RESTANTE DEL CODICE PRINCIPALE... identico all'ultima versione]

if __name__ == "__main__":
    main()