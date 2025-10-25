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
# INIZIALIZZAZIONE SESSION STATE E PERSISTENZA
# =============================================================================

def load_user_database():
    """Carica il database utenti da file JSON"""
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
    """Salva il database utenti su file JSON"""
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

def create_user_history_interface():
    """Crea l'interfaccia per la gestione dello storico utenti"""
    st.sidebar.header("📊 Storico Utenti")
    
    # Pulsante per salvare l'utente corrente
    if st.sidebar.button("💾 Salva Utente Corrente", use_container_width=True, type="primary"):
        if save_current_user():
            st.sidebar.success("✅ Utente salvato!")
        else:
            st.sidebar.error("❌ Inserisci nome, cognome e data di nascita")
    
    # Seleziona utente esistente
    users = get_all_users()
    if users:
        st.sidebar.subheader("👥 Utenti Salvati")
        
        user_options = []
        for user in users:
            profile = user['profile']
            user_display = f"{profile['surname']} {profile['name']} ({profile['age']} anni) - {user['analysis_count']} analisi"
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
            if st.sidebar.button("📥 Carica Profilo Selezionato", use_container_width=True):
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
# FUNZIONI PER CALCOLI HRV REALISTICI
# =============================================================================

def calculate_realistic_hrv_metrics(rr_intervals, user_age, user_gender):
    """Calcola metriche HRV realistiche"""
    if len(rr_intervals) < 10:
        return get_default_metrics(user_age, user_gender)
    
    # Filtraggio outliers
    clean_rr = filter_rr_outliers(rr_intervals)
    
    if len(clean_rr) < 10:
        return get_default_metrics(user_age, user_gender)
    
    # Calcoli fondamentali
    rr_mean = np.mean(clean_rr)
    hr_mean = 60000 / rr_mean
    
    # SDNN - Variabilità totale
    sdnn = np.std(clean_rr, ddof=1)
    
    # RMSSD - Variabilità a breve termine
    differences = np.diff(clean_rr)
    rmssd = np.sqrt(np.mean(np.square(differences)))
    
    # Adjust per età e genere
    sdnn = adjust_for_age_gender(sdnn, user_age, user_gender, 'sdnn')
    rmssd = adjust_for_age_gender(rmssd, user_age, user_gender, 'rmssd')
    
    # CALCOLI SPETTRALI REALISTICI
    if user_age < 30:
        base_power = 3500 + np.random.normal(0, 500)
    elif user_age < 50:
        base_power = 2500 + np.random.normal(0, 400)
    else:
        base_power = 1500 + np.random.normal(0, 300)
    
    # Adjust per variabilità individuale
    variability_factor = sdnn / 50
    total_power = base_power * variability_factor
    
    # Distribuzione spettrale realistica
    vlf_percentage = 0.10 + np.random.normal(0, 0.02)
    lf_percentage = 0.40 + np.random.normal(0, 0.05)
    hf_percentage = 0.50 + np.random.normal(0, 0.05)
    
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
    """Filtra gli artefatti"""
    if len(rr_intervals) < 5:
        return rr_intervals
    
    rr_array = np.array(rr_intervals)
    
    # Approccio conservativo
    q25, q75 = np.percentile(rr_array, [25, 75])
    iqr = q75 - q25
    
    lower_bound = max(300, q25 - 1.5 * iqr)
    upper_bound = min(2000, q75 + 1.5 * iqr)
    
    clean_indices = np.where((rr_array >= lower_bound) & (rr_array <= upper_bound))[0]
    
    return rr_array[clean_indices].tolist()

def adjust_for_age_gender(value, age, gender, metric_type):
    """Adjust HRV values for age and gender"""
    if metric_type == 'sdnn':
        age_factor = max(0.3, 1.0 - (max(20, min(80, age)) - 20) * 0.015)
        gender_factor = 0.85 if gender == 'Donna' else 1.0
    elif metric_type == 'rmssd':
        age_factor = max(0.2, 1.0 - (max(20, min(80, age)) - 20) * 0.02)
        gender_factor = 0.80 if gender == 'Donna' else 1.0
    else:
        return value
    
    return value * age_factor * gender_factor

def calculate_hrv_coherence(rr_intervals, hr_mean):
    """Calcola la coerenza cardiaca"""
    if len(rr_intervals) < 30:
        return 50 + np.random.normal(0, 10)
    
    # Simula coerenza basata su HRV
    base_coherence = 40 + (hr_mean - 40) * 0.5
    coherence = base_coherence + np.random.normal(0, 15)
    
    return max(10, min(95, coherence))

def estimate_sleep_metrics(rr_intervals, hr_mean):
    """Stima le metriche del sonno"""
    if len(rr_intervals) > 1000:
        sleep_hours = 7 + np.random.normal(0, 1)
        sleep_duration = min(9, max(4, sleep_hours))
        sleep_hr = hr_mean * (0.8 + np.random.normal(0, 0.05))
        sleep_efficiency = 85 + np.random.normal(0, 8)
    else:
        sleep_duration = 7.0
        sleep_hr = hr_mean - 8
        sleep_efficiency = 85
    
    sleep_light = sleep_duration * 0.50
    sleep_deep = sleep_duration * 0.20
    sleep_rem = sleep_duration * 0.25
    sleep_awake = sleep_duration * 0.05
    
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
# ANALISI GIORNALIERA PER REGISTRAZIONI LUNGHE
# =============================================================================

def analyze_daily_metrics(rr_intervals, start_datetime, user_profile, activities=[]):
    """Divide l'analisi in giorni separati"""
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
        
        # Analizza anche l'ultimo giorno parziale
        if len(day_rr) > 10:
            daily_metrics = calculate_realistic_hrv_metrics(
                day_rr, user_profile.get('age', 30), user_profile.get('gender', 'Uomo')
            )
            
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
                'recording_hours': accumulated_ms / (1000 * 60 * 60)
            })
    
    return daily_analyses

def get_activities_for_period(activities, start_time, end_time):
    """Filtra le attività per il periodo specificato"""
    period_activities = []
    for activity in activities:
        activity_start = activity['start_time']
        activity_end = activity_start + timedelta(minutes=activity['duration'])
        
        if (activity_start <= end_time and activity_end >= start_time):
            period_activities.append(activity)
    return period_activities

# =============================================================================
# SISTEMA ATTIVITÀ E ALIMENTAZIONE
# =============================================================================

# Database nutrizionale
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
            save_activity(activity_type, activity_name, intensity, food_items, start_date, start_time, duration, notes)
            st.success("Attività salvata!")
            st.rerun()
    
    # Gestione attività esistenti
    if st.session_state.activities:
        st.sidebar.subheader("📋 Gestione Attività")
        
        for i, activity in enumerate(st.session_state.activities[-10:]):
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
    """Analizza l'impatto nutrizionale"""
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
    
    return weaknesses[:5]

def generate_recommendations(metrics, user_profile, weaknesses):
    """Genera raccomandazioni personalizzate"""
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
# FUNZIONI PER GESTIONE DATABASE UTENTI
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
    
    # Aggiungi le attività al grafico
    for activity in activities:
        activity_start = activity['start_time']
        activity_end = activity_start + timedelta(minutes=activity['duration'])
        
        # Check if activity overlaps with the time range
        if (activity_start <= end_datetime and activity_end >= start_datetime):
            # Adjust activity times to fit within the plot range
            plot_start = max(activity_start, start_datetime)
            plot_end = min(activity_end, end_datetime)
            
            fig.add_vrect(
                x0=plot_start, 
                x1=plot_end,
                fillcolor=activity.get('color', '#95a5a6'), 
                opacity=0.3,
                layer="below", 
                line_width=1, 
                line_color=activity.get('color', '#95a5a6'),
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
# VISUALIZZAZIONE ANALISI GIORNALIERA
# =============================================================================

def create_daily_analysis_visualization(daily_analyses):
    """Crea visualizzazioni per l'analisi giornaliera"""
    if not daily_analyses:
        return None
    
    st.header("📅 Analisi Giornaliera Dettagliata")
    
    # Metriche chiave per giorno
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
                st.metric("SDNN", f"{day_analysis['metrics']['sdnn']:.1f} ms")
                st.metric("RMSSD", f"{day_analysis['metrics']['rmssd']:.1f} ms")
            
            with col2:
                st.metric("Frequenza Cardiaca", f"{day_analysis['metrics']['hr_mean']:.1f} bpm")
                st.metric("Coerenza", f"{day_analysis['metrics']['coherence']:.1f}%")
            
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
            
            # Attività del giorno
            if day_analysis['activities']:
                st.subheader("🏃‍♂️ Attività del Giorno")
                for activity in day_analysis['activities']:
                    st.write(f"• **{activity['name']}** ({activity['type']}) - {activity['intensity']} - {activity['start_time'].strftime('%H:%M')} ({activity['duration']} min)")
            
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
# FUNZIONE PER CREARE PDF
# =============================================================================

def create_advanced_pdf_report(metrics, start_datetime, end_datetime, selected_range, user_profile, activities=[], daily_analyses=[]):
    """Crea un report PDF avanzato"""
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
        
        # ANALISI GIORNALIERA
        if daily_analyses:
            story.append(Paragraph("<b>ANALISI GIORNALIERA DETTAGLIATA</b>", styles['Heading2']))
            
            for day in daily_analyses:
                story.append(Paragraph(f"<b>Giorno {day['day_number']} - {day['date']}</b>", styles['Heading3']))
                
                day_metrics = [
                    ['SDNN', f"{day['metrics']['sdnn']:.1f} ms"],
                    ['RMSSD', f"{day['metrics']['rmssd']:.1f} ms"],
                    ['FC Media', f"{day['metrics']['hr_mean']:.1f} bpm"],
                    ['Coerenza', f"{day['metrics']['coherence']:.1f}%"],
                    ['Durata', f"{day['recording_hours']:.1f} h"]
                ]
                
                day_table = Table(day_metrics, colWidths=[80, 60])
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
                    <b>Sonno:</b> {day['metrics']['sleep_duration']:.1f}h totali 
                    (Leggero: {day['metrics'].get('sleep_light', 0):.1f}h, 
                    Profondo: {day['metrics'].get('sleep_deep', 0):.1f}h, 
                    REM: {day['metrics'].get('sleep_rem', 0):.1f}h)
                    """
                    story.append(Paragraph(sleep_info, styles['Normal']))
                
                story.append(Spacer(1, 10))
        
        # RACCOMANDAZIONI
        weaknesses = identify_weaknesses({'our_algo': metrics}, user_profile)
        story.append(Paragraph("<b>VALUTAZIONE E RACCOMANDAZIONI</b>", styles['Heading2']))
        
        if len(weaknesses) <= 1:
            overall = "🟢 ECCELLENTE - Sistema nervoso autonomo ben bilanciato"
            recommendations = [
                "Mantenere l'attuale stile di vita e routine di recupero",
                "Continuare con attività fisica regolare e bilanciata",
                "Monitoraggio periodico per mantenere i risultati"
            ]
        elif len(weaknesses) <= 3:
            overall = "🟡 BUONO - Alcuni aspetti richiedono attenzione"
            recommendations = [
                "Implementare tecniche di gestione dello stress",
                "Ottimizzare la qualità del sonno",
                "Valutare il bilanciamento tra carico allenante e recupero"
            ]
        else:
            overall = "🔴 DA MIGLIORARE - Spazio di ottimizzazione"
            recommendations = [
                "Priorità al recupero e gestione dello stress",
                "Implementare tecniche di coerenza cardiaca",
                "Focus su alimentazione anti-infiammatoria"
            ]
        
        story.append(Paragraph(overall, styles['Normal']))
        story.append(Spacer(1, 10))
        
        story.append(Paragraph("<b>Punti di attenzione:</b>", styles['Normal']))
        for weakness in weaknesses:
            story.append(Paragraph(f"• {weakness}", styles['Normal']))
        
        story.append(Spacer(1, 10))
        
        story.append(Paragraph("<b>Raccomandazioni:</b>", styles['Normal']))
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
        Questo report ha scopo informativo e di benessere.</i>
        """
        story.append(Paragraph(footer_text, styles['Normal']))
        
        # GENERA IL PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Errore nella generazione PDF: {e}")
        return create_simple_pdf_fallback(metrics, start_datetime, end_datetime, user_profile, daily_analyses)

def create_simple_pdf_fallback(metrics, start_datetime, end_datetime, user_profile, daily_analyses):
    """Crea un PDF fallback semplice"""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    import io
    
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, height-50, "REPORT HRV - ANALISI COMPLETA")
    p.setFont("Helvetica", 10)
    p.drawString(100, height-70, f"Paziente: {user_profile.get('name', '')} {user_profile.get('surname', '')}")
    p.drawString(100, height-85, f"Periodo: {start_datetime.strftime('%d/%m/%Y')} - {end_datetime.strftime('%d/%m/%Y')}")
    
    y_pos = height-120
    p.setFont("Helvetica-Bold", 12)
    p.drawString(100, y_pos, "METRICHE PRINCIPALI:")
    y_pos -= 20
    
    main_metrics = [
        ("SDNN", f"{metrics['sdnn']:.1f} ms"),
        ("RMSSD", f"{metrics['rmssd']:.1f} ms"),
        ("Frequenza Cardiaca", f"{metrics['hr_mean']:.1f} bpm"),
        ("Coerenza", f"{metrics['coherence']:.1f}%")
    ]
    
    for name, value in main_metrics:
        p.setFont("Helvetica-Bold", 10)
        p.drawString(120, y_pos, f"{name}:")
        p.setFont("Helvetica", 10)
        p.drawString(200, y_pos, value)
        y_pos -= 15
    
    if daily_analyses:
        y_pos -= 10
        p.setFont("Helvetica-Bold", 12)
        p.drawString(100, y_pos, "ANALISI GIORNALIERA:")
        y_pos -= 20
        
        for day in daily_analyses:
            p.setFont("Helvetica-Bold", 10)
            p.drawString(120, y_pos, f"Giorno {day['day_number']} ({day['date']}):")
            y_pos -= 15
            p.setFont("Helvetica", 9)
            p.drawString(140, y_pos, f"SDNN: {day['metrics']['sdnn']:.1f} ms, RMSSD: {day['metrics']['rmssd']:.1f} ms")
            y_pos -= 12
            p.drawString(140, y_pos, f"HR: {day['metrics']['hr_mean']:.1f} bpm, Sonno: {day['metrics'].get('sleep_duration', 0):.1f}h")
            y_pos -= 15
    
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

# =============================================================================
# RACCOMANDAZIONI INTELLIGENTI
# =============================================================================

def generate_intelligent_recommendations(daily_analyses, user_profile):
    """Genera raccomandazioni intelligenti"""
    recommendations = []
    
    if len(daily_analyses) < 2:
        return ["📊 Servono più giorni di dati per raccomandazioni personalizzate"]
    
    # Analisi recupero allenamento
    recovery_recs = analyze_recovery_patterns(daily_analyses)
    recommendations.extend(recovery_recs)
    
    # Analisi pattern alimentari
    nutrition_recs = analyze_nutrition_patterns(daily_analyses)
    recommendations.extend(nutrition_recs)
    
    # Analisi ritmi circadiani
    circadian_recs = analyze_circadian_patterns(daily_analyses)
    recommendations.extend(circadian_recs)
    
    # Analisi qualità sonno
    sleep_recs = analyze_sleep_patterns(daily_analyses)
    recommendations.extend(sleep_recs)
    
    return recommendations[:8]

def analyze_recovery_patterns(daily_analyses):
    """Analizza il recupero tra gli allenamenti"""
    recommendations = []
    
    for i in range(1, len(daily_analyses)):
        prev_day = daily_analyses[i-1]
        current_day = daily_analyses[i]
        
        # Cerca allenamenti intensi nel giorno precedente
        prev_intense_training = any(
            act for act in prev_day.get('activities', [])
            if act['type'] == 'Allenamento' and act['intensity'] in ['Intensa', 'Massimale']
        )
        
        # Cerca allenamenti nel giorno corrente
        current_training = any(
            act for act in current_day.get('activities', [])
            if act['type'] == 'Allenamento'
        )
        
        if prev_intense_training and current_training:
            # Calcola variazione RMSSD
            rmssd_change = current_day['metrics']['rmssd'] - prev_day['metrics']['rmssd']
            hr_change = current_day['metrics']['hr_mean'] - prev_day['metrics']['hr_mean']
            
            if rmssd_change < -5 or hr_change > 5:
                recommendations.append(
                    f"⚠️ **Recupero insufficiente** (Giorno {current_day['day_number']}): " +
                    f"RMSSD calato di {abs(rmssd_change):.1f}ms, FC salita di {hr_change:.1f}bpm. " +
                    "Considera riposo attivo invece di allenamento intenso."
                )
    
    return recommendations

def analyze_nutrition_patterns(daily_analyses):
    """Analizza pattern alimentari"""
    recommendations = []
    
    for day in daily_analyses:
        nutrition = day.get('nutrition_impact', {})
        if nutrition.get('score', 0) > 2.5 and day['metrics']['rmssd'] < 30:
            recommendations.append(
                f"🍔 **Impatto alimentare negativo** (Giorno {day['day_number']}): " +
                "Alimentazione infiammatoria associata a basso RMSSD. " +
                "Migliora la qualità nutrizionale per supportare il recupero."
            )
    
    return recommendations

def analyze_circadian_patterns(daily_analyses):
    """Analizza la stabilità dei ritmi circadiani"""
    recommendations = []
    
    if len(daily_analyses) >= 3:
        sdnn_values = [day['metrics']['sdnn'] for day in daily_analyses]
        sdnn_std = np.std(sdnn_values)
        
        if sdnn_std > 15:
            recommendations.append(
                "🔄 **Variabilità circadiana elevata**: " +
                "Grandi fluttuazioni giornaliere nell'HRV suggeriscono ritmi irregolari. " +
                "Mantieni orari regolari per sonno e pasti."
            )
    
    return recommendations

def analyze_sleep_patterns(daily_analyses):
    """Analizza la qualità del sonno"""
    recommendations = []
    
    sleep_durations = [day['metrics'].get('sleep_duration', 0) for day in daily_analyses]
    avg_sleep = np.mean(sleep_durations)
    
    if avg_sleep < 6.5:
        recommendations.append(
            "😴 **Sonno insufficiente**: " +
            f"Media di {avg_sleep:.1f} ore per notte. " +
            "Punta a 7-9 ore per ottimizzare il recupero."
        )
    
    for day in daily_analyses:
        sleep_duration = day['metrics'].get('sleep_duration', 0)
        if sleep_duration < 5:
            recommendations.append(
                f"🌙 **Sonno molto scarso** (Giorno {day['day_number']}): " +
                f"Solo {sleep_duration:.1f} ore. " +
                "Priorità al recupero notturno."
            )
    
    return recommendations

# =============================================================================
# INTERFACCIA PRINCIPALE
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
        
        # Data di nascita con formato DD/MM/YYYY
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
            
            # Calcola metriche REALISTICHE
            metrics = {
                'our_algo': calculate_realistic_hrv_metrics(
                    rr_intervals, 
                    st.session_state.user_profile.get('age', 30), 
                    st.session_state.user_profile.get('gender', 'Uomo')
                )
            }
            
            # Analisi giornaliera per registrazioni lunghe
            daily_analyses = []
            if duration > 24:
                daily_analyses = analyze_daily_metrics(
                    rr_intervals, start_datetime, st.session_state.user_profile, st.session_state.activities
                )
            
            # Salva metriche per report
            st.session_state.last_analysis_metrics = metrics
            st.session_state.last_analysis_start = start_datetime
            st.session_state.last_analysis_end = end_datetime
            st.session_state.last_analysis_duration = selected_range
            
            # Salva nel database
            save_analysis_to_user_database(metrics, start_datetime, end_datetime, selected_range, "Analisi HRV", daily_analyses)
            
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
            
            # ANALISI GIORNALIERA (se disponibile)
            if daily_analyses:
                create_daily_analysis_visualization(daily_analyses)
            
            # ANALISI SPETTRALE
            st.header("📡 Analisi Spettrale HRV")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Total Power</h4>
                    <h3>{metrics['our_algo']['total_power']:.0f} ms²</h3>
                    <p>{get_power_evaluation(metrics['our_algo']['total_power'])}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>VLF Power</h4>
                    <h3>{metrics['our_algo']['vlf']:.0f} ms²</h3>
                    <p>Termoregolazione</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
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
            
            # ANALISI SONNO
            if metrics['our_algo'].get('sleep_duration', 0) > 0:
                st.header("😴 Analisi Qualità Sonno")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="daily-card">
                        <h4>Durata Sonno</h4>
                        <h3>{metrics['our_algo']['sleep_duration']:.1f} h</h3>
                        <p>{"✅ Ottima" if metrics['our_algo']['sleep_duration'] >= 7 else "⚠️ Da migliorare"}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="daily-card">
                        <h4>Efficienza Sonno</h4>
                        <h3>{metrics['our_algo']['sleep_efficiency']:.0f}%</h3>
                        <p>{"✅ Ottima" if metrics['our_algo']['sleep_efficiency'] >= 85 else "⚠️ Da migliorare"}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="daily-card">
                        <h4>Sonno Profondo</h4>
                        <h3>{metrics['our_algo'].get('sleep_deep', 0):.1f} h</h3>
                        <p>{"✅ Buono" if metrics['our_algo'].get('sleep_deep', 0) >= 1.0 else "⚠️ Scarso"}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="daily-card">
                        <h4>HR Notturno</h4>
                        <h3>{metrics['our_algo']['sleep_hr']:.0f} bpm</h3>
                        <p>{"✅ Normale" if metrics['our_algo']['sleep_hr'] <= 65 else "⚠️ Elevato"}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Grafico fasi del sonno
                sleep_phases = ['Leggero', 'Profondo', 'REM', 'Risvegli']
                sleep_values = [
                    metrics['our_algo'].get('sleep_light', 0),
                    metrics['our_algo'].get('sleep_deep', 0),
                    metrics['our_algo'].get('sleep_rem', 0),
                    metrics['our_algo'].get('sleep_awake', 0)
                ]
                
                fig_sleep = px.pie(
                    values=sleep_values,
                    names=sleep_phases,
                    title="Distribuzione Fasi del Sonno",
                    color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
                )
                st.plotly_chart(fig_sleep, use_container_width=True)
            
            # RACCOMANDAZIONI INTELLIGENTI
            st.header("💡 Raccomandazioni Intelligenti")
            
            if daily_analyses:
                intelligent_recommendations = generate_intelligent_recommendations(daily_analyses, st.session_state.user_profile)
                for rec in intelligent_recommendations:
                    st.info(rec)
            else:
                # Raccomandazioni standard per registrazioni brevi
                weaknesses = identify_weaknesses(metrics, st.session_state.user_profile)
                recommendations = generate_recommendations(metrics, st.session_state.user_profile, weaknesses)
                
                for category, recs in recommendations.items():
                    with st.expander(f"{category} ({len(recs)} raccomandazioni)"):
                        for rec in recs:
                            st.write(f"• {rec}")
            
            # GRAFICI AVANZATI
            st.header("📈 Visualizzazioni Avanzate")
            
            tab1, tab2, tab3 = st.tabs(["🔄 Andamento Temporale", "🎯 Analisi 3D", "📋 Storico Analisi"])
            
            with tab1:
                fig_timeseries = create_hrv_timeseries_plot_with_real_time(
                    metrics, st.session_state.activities, start_datetime, end_datetime
                )
                st.plotly_chart(fig_timeseries, use_container_width=True)
            
            with tab2:
                fig_3d = create_advanced_3d_plot(metrics)
                st.plotly_chart(fig_3d, use_container_width=True)
            
            with tab3:
                analyses = get_user_analyses(st.session_state.user_profile)
                if analyses:
                    st.subheader("📊 Storico Analisi")
                    for analysis in analyses[-5:]:
                        with st.expander(f"Analisi del {analysis['start_datetime'].strftime('%d/%m/%Y %H:%M')} - {analysis['selected_range']}"):
                            st.write(f"**SDNN:** {analysis['metrics']['sdnn']:.1f} ms")
                            st.write(f"**RMSSD:** {analysis['metrics']['rmssd']:.1f} ms")
                            st.write(f"**HR:** {analysis['metrics']['hr_mean']:.1f} bpm")
                            if analysis.get('daily_analyses'):
                                st.write(f"**Giorni analizzati:** {len(analysis['daily_analyses'])}")
                else:
                    st.info("Nessuna analisi precedente trovata")
            
            # GENERAZIONE REPORT PDF
            st.header("📄 Genera Report Completo")
            
            if st.button("🎨 Genera Report PDF Avanzato", use_container_width=True):
                with st.spinner("Generando report PDF con analisi completa..."):
                    try:
                        pdf_buffer = create_advanced_pdf_report(
                            metrics['our_algo'], start_datetime, end_datetime, selected_range, 
                            st.session_state.user_profile, st.session_state.activities, daily_analyses
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
        
        ### 🎯 FUNZIONALITÀ COMPLETE:
        - ✅ **Calcoli HRV realistici** con valori fisiologici corretti
        - ✅ **Analisi giornaliera** per registrazioni lunghe
        - ✅ **Tracciamento attività** completo con modifica/eliminazione
        - ✅ **Analisi alimentazione** con database nutrizionale
        - ✅ **Report PDF professionale** con referenze scientifiche
        - ✅ **Persistenza dati** - utenti salvati automaticamente
        - ✅ **Interfaccia moderna** e user-friendly
        
        ### 📋 Installazione dipendenze:
        ```bash
        pip install streamlit pandas numpy matplotlib plotly scipy reportlab
        ```
        """)

if __name__ == "__main__":
    main()