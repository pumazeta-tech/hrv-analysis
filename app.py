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

try:
    import neurokit2 as nk
    NEUROKIT_AVAILABLE = True
    st.sidebar.success("✅ NeuroKit2 caricato")
except ImportError:
    NEUROKIT_AVAILABLE = False
    st.sidebar.warning("⚠️ NeuroKit2 non disponibile")

# Costanti
ACTIVITY_COLORS = {
    "Allenamento": "#e74c3c",
    "Alimentazione": "#3498db", 
    "Stress": "#f39c12",
    "Riposo": "#2ecc71",
    "Altro": "#95a5a6"
}

def init_session_state():
    """Inizializza lo stato della sessione"""
    session_defaults = {
        'user_database': load_user_database(),
        'user_profile': {
            'name': '', 'surname': '', 'birth_date': None, 'age': 30, 'gender': 'Uomo'
        },
        'activities': [],
        'file_uploaded': False,
        'datetime_initialized': False,
        'analysis_datetimes': {
            'start_datetime': datetime.now(),
            'end_datetime': datetime.now() + timedelta(hours=24)
        },
        'last_analysis_metrics': None,
        'last_analysis_start': None,
        'last_analysis_end': None,
        'last_analysis_duration': None,
        'last_analysis_daily': [],
        'editing_activity_index': None
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def load_user_database():
    """Carica il database utenti"""
    try:
        if os.path.exists('user_database.json'):
            with open('user_database.json', 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_user_database():
    """Salva il database utenti"""
    try:
        with open('user_database.json', 'w', encoding='utf-8') as f:
            json.dump(st.session_state.user_database, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Errore salvataggio database: {e}")

def calculate_hrv_with_neurokit(rr_intervals, user_age, user_gender):
    """Calcola metriche HRV usando NeuroKit2"""
    if len(rr_intervals) < 10:
        return get_default_metrics(user_age, user_gender)
    
    try:
        rr_seconds = np.array(rr_intervals) / 1000.0
        rr_clean = nk.ppg_clean(rr_seconds, sampling_rate=1000)
        peaks = nk.ppg_findpeaks(rr_clean, sampling_rate=1000)['Peaks']
        
        if len(peaks) < 10:
            return get_default_metrics(user_age, user_gender)
        
        # Metriche temporali
        hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)
        sdnn = hrv_time['HRV_SDNN'].iloc[0] * 1000 if 'HRV_SDNN' in hrv_time.columns else np.std(rr_intervals)
        rmssd = hrv_time['HRV_RMSSD'].iloc[0] * 1000 if 'HRV_RMSSD' in hrv_time.columns else calculate_rmssd_fallback(rr_intervals)
        hr_mean = 60000 / np.mean(rr_intervals)
        
        # Metriche frequenziali
        hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False)
        total_power = hrv_freq['HRV_TotalPower'].iloc[0] if 'HRV_TotalPower' in hrv_freq.columns else 2000
        lf = hrv_freq['HRV_LF'].iloc[0] if 'HRV_LF' in hrv_freq.columns else 800
        hf = hrv_freq['HRV_HF'].iloc[0] if 'HRV_HF' in hrv_freq.columns else 700
        lf_hf_ratio = hrv_freq['HRV_LFHF'].iloc[0] if 'HRV_LFHF' in hrv_freq.columns else lf/hf if hf > 0 else 1.1
        
        # Metriche non-lineari
        hrv_nonlinear = nk.hrv_nonlinear(peaks, sampling_rate=1000, show=False)
        sd1 = hrv_nonlinear['HRV_SD1'].iloc[0] * 1000 if 'HRV_SD1' in hrv_nonlinear.columns else rmssd / np.sqrt(2)
        sd2 = hrv_nonlinear['HRV_SD2'].iloc[0] * 1000 if 'HRV_SD2' in hrv_nonlinear.columns else sdnn
        
        coherence = calculate_hrv_coherence_advanced(rr_intervals, hr_mean, user_age)
        sleep_metrics = estimate_sleep_metrics_advanced(rr_intervals, hr_mean, user_age)
        
        return {
            'sdnn': max(25, min(180, sdnn)),
            'rmssd': max(15, min(120, rmssd)),
            'hr_mean': max(45, min(100, hr_mean)),
            'coherence': max(20, min(95, coherence)),
            'recording_hours': len(rr_intervals) * np.mean(rr_intervals) / (1000 * 60 * 60),
            'total_power': max(800, min(8000, total_power)),
            'vlf': 500, 'lf': max(200, min(4000, lf)), 'hf': max(200, min(4000, hf)),
            'lf_hf_ratio': max(0.3, min(4.0, lf_hf_ratio)),
            'sd1': max(10, min(80, sd1)), 'sd2': max(30, min(200, sd2)),
            'sd1_sd2_ratio': max(0.2, min(3.0, sd1/sd2 if sd2 > 0 else 1.0)),
            'sleep_duration': sleep_metrics['duration'], 'sleep_efficiency': sleep_metrics['efficiency'],
            'sleep_hr': sleep_metrics['hr'], 'sleep_light': sleep_metrics['light'],
            'sleep_deep': sleep_metrics['deep'], 'sleep_rem': sleep_metrics['rem'],
            'sleep_awake': sleep_metrics['awake'], 'analysis_method': 'NeuroKit2'
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

def get_default_metrics(user_age, user_gender):
    """Metriche di default per dati insufficienti"""
    return {
        'sdnn': 45, 'rmssd': 30, 'hr_mean': 70, 'coherence': 50,
        'recording_hours': 0, 'total_power': 1500, 'vlf': 500, 'lf': 800, 'hf': 700,
        'lf_hf_ratio': 1.1, 'sd1': 20, 'sd2': 50, 'sd1_sd2_ratio': 0.4,
        'sleep_duration': 7, 'sleep_efficiency': 85, 'sleep_hr': 60,
        'sleep_light': 4, 'sleep_deep': 2, 'sleep_rem': 1, 'sleep_awake': 0.5,
        'analysis_method': 'Default'
    }

def calculate_hrv_fallback(rr_intervals, user_age, user_gender):
    """Calcolo HRV fallback"""
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
    """Stima metriche del sonno"""
    if len(rr_intervals) < 100:
        return get_default_sleep_metrics()
    
    night_hr = hr_mean * 0.85
    sleep_duration = min(9, max(4, 7 + (65 - hr_mean) / 20))
    
    return {
        'duration': sleep_duration,
        'efficiency': max(70, min(95, 85 + (rmssd - 30) / 2)),
        'hr': night_hr,
        'light': sleep_duration * 0.6,
        'deep': sleep_duration * 0.2,
        'rem': sleep_duration * 0.2,
        'awake': max(0.1, sleep_duration * 0.05)
    }

def get_default_sleep_metrics():
    """Metriche sonno di default"""
    return {
        'duration': 7, 'efficiency': 85, 'hr': 60,
        'light': 4.2, 'deep': 1.4, 'rem': 1.4, 'awake': 0.35
    }

def analyze_daily_metrics(rr_intervals, start_datetime, user_profile, activities=[]):
    """Divide l'analisi in giorni separati"""
    daily_analyses = []
    
    if len(rr_intervals) == 0:
        return daily_analyses
    
    total_duration_ms = np.sum(rr_intervals)
    total_duration_hours = total_duration_ms / (1000 * 60 * 60)
    total_days = int(np.ceil(total_duration_hours / 24))
    
    current_index = 0
    
    for day in range(total_days):
        day_start = start_datetime + timedelta(days=day)
        day_end = day_start + timedelta(hours=24)
        
        day_rr = []
        accumulated_time_ms = 0
        day_duration_ms = 24 * 60 * 60 * 1000
        
        while current_index < len(rr_intervals) and accumulated_time_ms < day_duration_ms:
            current_rr = rr_intervals[current_index]
            day_rr.append(current_rr)
            accumulated_time_ms += current_rr
            current_index += 1
        
        if len(day_rr) >= 50:
            daily_metrics = calculate_realistic_hrv_metrics(day_rr, user_profile.get('age', 30), user_profile.get('gender', 'Uomo'))
            day_activities = get_activities_for_period(activities, day_start, day_end)
            
            daily_analyses.append({
                'day_number': day + 1,
                'date': day_start.date(),
                'start_time': day_start,
                'end_time': day_end,
                'metrics': daily_metrics,
                'activities': day_activities,
                'nutrition_impact': analyze_nutritional_impact_day(day_activities, daily_metrics),
                'activity_impact': analyze_activity_impact_on_ans(day_activities, daily_metrics),
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
        
        if (activity_start < end_time and activity_end > start_time):
            period_activities.append(activity)
    return period_activities

def analyze_nutritional_impact_day(activities, daily_metrics):
    """Analizza impatto nutrizionale"""
    food_activities = [a for a in activities if a['type'] == 'Alimentazione']
    
    if not food_activities:
        return {'score': 0, 'analysis': "Nessun dato alimentare", 'recommendations': []}
    
    heavy_meals = sum(1 for a in food_activities if a['intensity'] in ['Pesante', 'Molto pesante'])
    
    if heavy_meals > 2:
        return {
            'score': 3,
            'analysis': "Troppi pasti pesanti che possono influenzare negativamente l'HRV",
            'recommendations': ["Riduci i pasti pesanti", "Aumenta l'idratazione"]
        }
    elif heavy_meals == 0:
        return {
            'score': -1,
            'analysis': "Alimentazione leggera e bilanciata",
            'recommendations': ["Continua così!"]
        }
    else:
        return {
            'score': 1,
            'analysis': "Alimentazione nella norma",
            'recommendations': ["Mantieni un buon equilibrio"]
        }

def analyze_activity_impact_on_ans(activities, daily_metrics):
    """Analizza impatto attività sul sistema nervoso autonomo"""
    training_activities = [a for a in activities if a['type'] == 'Allenamento']
    
    if not training_activities:
        return {'score': 0, 'analysis': "Nessun allenamento registrato"}
    
    intense_count = sum(1 for a in training_activities if a['intensity'] in ['Intensa', 'Massimale'])
    
    if intense_count > 1:
        return {
            'score': 2,
            'analysis': "Allenamenti intensi possono aver stressato il sistema nervoso"
        }
    else:
        return {
            'score': -1,
            'analysis': "Attività fisica bilanciata"
        }

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
    
    save_user_database()

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
        save_user_database()

def delete_activity(index):
    """Elimina un'attività"""
    if 0 <= index < len(st.session_state.activities):
        st.session_state.activities.pop(index)
        save_user_database()
        st.success("Attività eliminata!")
        st.rerun()

def edit_activity_interface():
    """Interfaccia per modificare attività"""
    st.sidebar.header("✏️ Modifica Attività")
    
    index = st.session_state.editing_activity_index
    activity = st.session_state.activities[index]
    
    start_datetime = activity['start_time']
    
    activity_type = st.sidebar.selectbox("Tipo Attività", ["Allenamento", "Alimentazione", "Stress", "Riposo", "Altro"], 
                                       index=["Allenamento", "Alimentazione", "Stress", "Riposo", "Altro"].index(activity['type']))
    
    activity_name = st.sidebar.text_input("Nome Attività", value=activity['name'])
    
    if activity_type == "Alimentazione":
        food_items = st.sidebar.text_area("Cibo consumato", value=activity.get('food_items', ''))
        intensity = st.sidebar.select_slider("Pesantezza pasto", 
                                           options=["Leggero", "Normale", "Pesante", "Molto pesante"],
                                           value=activity['intensity'])
    else:
        food_items = activity.get('food_items', '')
        intensity = st.sidebar.select_slider("Intensità", 
                                           options=["Leggera", "Moderata", "Intensa", "Massimale"],
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
            st.session_state.editing_activity_index = None
            st.rerun()
    with col2:
        if st.button("❌ Annulla", use_container_width=True):
            st.session_state.editing_activity_index = None
            st.rerun()
    with col3:
        if st.button("🗑️ Elimina", use_container_width=True):
            delete_activity(index)
            st.session_state.editing_activity_index = None
            st.rerun()

def create_activity_tracker():
    """Interfaccia per tracciare attività e alimentazione"""
    st.sidebar.header("🏃‍♂️ Tracker Attività")
    
    if st.session_state.get('editing_activity_index') is not None:
        edit_activity_interface()
        return
    
    with st.sidebar.expander("➕ Aggiungi Attività/Pasto", expanded=False):
        activity_type = st.selectbox("Tipo Attività", ["Allenamento", "Alimentazione", "Stress", "Riposo", "Altro"])
        
        activity_name = st.text_input("Nome Attività/Pasto", placeholder="Es: Corsa mattutina, Pranzo, etc.")
        
        if activity_type == "Alimentazione":
            food_items = st.text_area("Cibo consumato", placeholder="Es: pasta, insalata, frutta")
            intensity = st.select_slider("Pesantezza pasto", 
                                       options=["Leggero", "Normale", "Pesante", "Molto pesante"])
        else:
            food_items = ""
            intensity = st.select_slider("Intensità", 
                                       options=["Leggera", "Moderada", "Intensa", "Massimale"])
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Data", value=datetime.now().date())
            start_time = st.time_input("Ora inizio", value=datetime.now().time())
        with col2:
            duration = st.number_input("Durata (min)", min_value=1, max_value=480, value=30)
        
        notes = st.text_area("Note", placeholder="Note aggiuntive...")
        
        if st.button("💾 Salva Attività", use_container_width=True):
            save_activity(activity_type, activity_name, intensity, food_items, start_date, start_time, duration, notes)
            st.success(f"Attività salvata per il {start_date.strftime('%d/%m/%Y')}!")
            st.rerun()
    
    if st.session_state.activities:
        st.sidebar.subheader("📋 Attività Salvate")
        
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
                        st.rerun()

def create_daily_analysis_visualization(daily_analyses):
    """Crea visualizzazioni per l'analisi giornaliera"""
    if not daily_analyses:
        return None
    
    st.header("📅 Analisi Giornaliera Dettagliata")
    
    st.info(f"**Periodo analizzato:** {len(daily_analyses)} giorni - Dal {daily_analyses[0]['date'].strftime('%d/%m/%Y')} al {daily_analyses[-1]['date'].strftime('%d/%m/%Y')}")
    
    days = [f"Giorno {day['day_number']}\n({day['date'].strftime('%d/%m')})" for day in daily_analyses]
    sdnn_values = [day['metrics']['sdnn'] for day in daily_analyses]
    rmssd_values = [day['metrics']['rmssd'] for day in daily_analyses]
    hr_values = [day['metrics']['hr_mean'] for day in daily_analyses]
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Andamento Giornaliero", "🎯 Analisi Dettagliata", "📈 Analisi Spettrale", "🏃‍♂️ Impatto Attività"])
    
    with tab1:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=days, y=sdnn_values, mode='lines+markers', name='SDNN',
                               line=dict(color='#3498db', width=3), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=days, y=rmssd_values, mode='lines+markers', name='RMSSD',
                               line=dict(color='#e74c3c', width=3), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=days, y=hr_values, mode='lines+markers', name='HR',
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
                    st.metric("Potenza Totale", f"{day_analysis['metrics']['total_power']:.0f} ms²")
                
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
                    
                    sleep_phases = ['Leggero', 'Profondo', 'REM', 'Risvegli']
                    sleep_values = [
                        day_analysis['metrics'].get('sleep_light', 0),
                        day_analysis['metrics'].get('sleep_deep', 0), 
                        day_analysis['metrics'].get('sleep_rem', 0),
                        day_analysis['metrics'].get('sleep_awake', 0)
                    ]
                    
                    fig_sleep = px.pie(values=sleep_values, names=sleep_phases, title="Distribuzione Fasi del Sonno")
                    fig_sleep.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_sleep, use_container_width=True)
                
                day_activities = day_analysis['activities']
                if day_activities:
                    st.subheader(f"🏃‍♂️ Attività del {day_analysis['date'].strftime('%d/%m/%Y')}")
                    for activity in day_activities:
                        st.write(f"• **{activity['name']}** ({activity['type']}) - {activity['intensity']} - {activity['start_time'].strftime('%H:%M')} ({activity['duration']} min)")
                else:
                    st.info("Nessuna attività registrata per questo giorno")
                
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
    
    with tab3:
        st.subheader("Dettaglio Metriche Spettrali")
        
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
        st.subheader("Analisi Impatto Attività")
        
        for day_analysis in daily_analyses:
            with st.expander(f"🏃‍♂️ Giorno {day_analysis['day_number']} - {day_analysis['date'].strftime('%d/%m/%Y')}", expanded=False):
                activity_impact = day_analysis.get('activity_impact', {})
                
                if activity_impact.get('analysis'):
                    if activity_impact['score'] > 1:
                        st.error(activity_impact['analysis'])
                    elif activity_impact['score'] < 0:
                        st.success(activity_impact['analysis'])
                    else:
                        st.warning(activity_impact['analysis'])
                
                day_activities = day_analysis['activities']
                if day_activities:
                    st.write("**Attività registrate:**")
                    for activity in day_activities:
                        st.write(f"• **{activity['name']}** ({activity['type']}) - {activity['intensity']}")
                else:
                    st.info("Nessuna attività registrata")

def create_advanced_pdf_report(metrics, start_datetime, end_datetime, selected_range, user_profile, activities=[], daily_analyses=[]):
    """Crea un report PDF avanzato"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib.colors import HexColor, white
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        
        buffer = io.BytesIO()
        
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                              topMargin=20*mm, bottomMargin=20*mm,
                              leftMargin=15*mm, rightMargin=15*mm)
        
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=HexColor("#2c3e50"),
            spaceAfter=20,
            alignment=TA_CENTER
        )
        
        story = []
        
        story.append(Paragraph("REPORT HRV COMPLETO<br/><font size=10>Analisi Sistema Neurovegetativo</font>", title_style))
        story.append(Spacer(1, 10))
        
        user_info = f"<b>Paziente:</b> {user_profile.get('name', '')} {user_profile.get('surname', '')} | <b>Età:</b> {user_profile.get('age', '')} | <b>Sesso:</b> {user_profile.get('gender', '')}"
        story.append(Paragraph(user_info, styles['Normal']))
        
        period_info = f"<b>Periodo analizzato:</b> {start_datetime.strftime('%d/%m/%Y %H:%M')} - {end_datetime.strftime('%d/%m/%Y %H:%M')}"
        story.append(Paragraph(period_info, styles['Normal']))
        story.append(Spacer(1, 15))
        
        story.append(Paragraph("<b>METRICHE PRINCIPALI HRV</b>", styles['Heading2']))
        
        metrics_data = [
            ['Parametro', 'Valore', 'Valutazione'],
            ['SDNN', f"{metrics['sdnn']:.1f} ms", get_sdnn_evaluation(metrics['sdnn'], user_profile.get('gender', 'Uomo'))],
            ['RMSSD', f"{metrics['rmssd']:.1f} ms", get_rmssd_evaluation(metrics['rmssd'], user_profile.get('gender', 'Uomo'))],
            ['Frequenza Cardiaca Media', f"{metrics['hr_mean']:.1f} bpm", get_hr_evaluation(metrics['hr_mean'])],
            ['Coerenza Cardiaca', f"{metrics['coherence']:.1f}%", get_coherence_evaluation(metrics['coherence'])],
            ['LF/HF Ratio', f"{metrics['lf_hf_ratio']:.2f}", get_lf_hf_evaluation(metrics['lf_hf_ratio'])],
            ['Potenza Totale', f"{metrics['total_power']:.0f} ms²", get_power_evaluation(metrics['total_power'])]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[80*mm, 40*mm, 50*mm])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor("#34495e")),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor("#ecf0f1")),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, HexColor("#bdc3c7"))
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 15))
        
        if daily_analyses:
            story.append(Paragraph("<b>ANALISI GIORNALIERA</b>", styles['Heading2']))
            
            daily_data = [['Giorno', 'Data', 'SDNN', 'RMSSD', 'FC Media', 'Coerenza']]
            
            for day in daily_analyses:
                daily_data.append([
                    f"Giorno {day['day_number']}",
                    day['date'].strftime('%d/%m'),
                    f"{day['metrics']['sdnn']:.1f}",
                    f"{day['metrics']['rmssd']:.1f}",
                    f"{day['metrics']['hr_mean']:.1f}",
                    f"{day['metrics']['coherence']:.1f}%"
                ])
            
            daily_table = Table(daily_data, colWidths=[25*mm, 25*mm, 25*mm, 25*mm, 25*mm, 30*mm])
            daily_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor("#3498db")),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BACKGROUND', (0, 1), (-1, -1), HexColor("#f8f9fa")),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, HexColor("#dee2e6"))
            ]))
            
            story.append(daily_table)
            story.append(Spacer(1, 15))
        
        if activities:
            story.append(Paragraph("<b>ATTIVITÀ REGISTRATE</b>", styles['Heading2']))
            
            for activity in activities[-10:]:
                activity_text = f"<b>{activity['name']}</b> ({activity['type']}) - {activity['intensity']} - {activity['start_time'].strftime('%d/%m/%Y %H:%M')} ({activity['duration']} min)"
                if activity.get('notes'):
                    activity_text += f"<br/><i>Note: {activity['notes']}</i>"
                story.append(Paragraph(activity_text, styles['Normal']))
                story.append(Spacer(1, 5))
        
        story.append(Spacer(1, 15))
        story.append(Paragraph(f"<i>Report generato il {datetime.now().strftime('%d/%m/%Y alle %H:%M')}</i>", styles['Italic']))
        story.append(Paragraph("<i>Software di analisi HRV - Sistema Neurovegetativo</i>", styles['Italic']))
        
        doc.build(story)
        
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Errore nella creazione del PDF: {e}")
        return None

def create_pdf_download_button():
    """Crea il bottone per scaricare il PDF"""
    if st.session_state.get('last_analysis_metrics'):
        st.sidebar.header("📊 Report PDF")
        
        if st.sidebar.button("📄 Genera Report PDF Completo", use_container_width=True):
            with st.spinner("Generazione PDF in corso..."):
                pdf_buffer = create_advanced_pdf_report(
                    st.session_state.last_analysis_metrics,
                    st.session_state.last_analysis_start,
                    st.session_state.last_analysis_end,
                    "custom",
                    st.session_state.user_profile,
                    st.session_state.activities,
                    st.session_state.last_analysis_daily
                )
                
                if pdf_buffer:
                    st.sidebar.success("✅ PDF generato!")
                    
                    b64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode()
                    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="report_hrv_completo.pdf" style="display: inline-block; padding: 0.5rem 1rem; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 0.3rem; text-align: center;">📥 Scarica Report PDF</a>'
                    st.sidebar.markdown(href, unsafe_allow_html=True)
                else:
                    st.sidebar.error("❌ Errore nella generazione del PDF")

def get_sdnn_evaluation(sdnn, gender):
    """Valuta SDNN"""
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
    """Valuta RMSSD"""
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
    """Valuta frequenza cardiaca"""
    if hr < 55: return 'Bradicardia'
    elif hr < 65: return 'Ottimo'
    elif hr < 75: return 'Buono'
    elif hr < 85: return 'Normale'
    else: return 'Elevata'

def get_coherence_evaluation(coherence):
    """Valuta coerenza cardiaca"""
    if coherence > 80: return 'Alta'
    elif coherence > 60: return 'Buona'
    elif coherence > 40: return 'Media'
    else: return 'Bassa'

def get_lf_hf_evaluation(ratio):
    """Valuta rapporto LF/HF"""
    if 0.5 <= ratio <= 2.0: return 'Bilanciato'
    elif ratio < 0.5: return 'Vagale'
    else: return 'Simpatico'

def get_power_evaluation(power):
    """Valuta potenza totale"""
    if power > 3000: return 'Alta'
    elif power > 1500: return 'Buona'
    elif power > 800: return 'Normale'
    else: return 'Bassa'

def calculate_realistic_hrv_metrics(rr_intervals, user_age, user_gender):
    """Calcola metriche HRV realistiche"""
    if len(rr_intervals) < 10:
        return get_default_metrics(user_age, user_gender)
    
    if NEUROKIT_AVAILABLE:
        return calculate_hrv_with_neurokit(rr_intervals, user_age, user_gender)
    else:
        return calculate_hrv_fallback(rr_intervals, user_age, user_gender)

def main():
    """Funzione principale"""
    st.set_page_config(
        page_title="Analisi HRV Sistema Neurovegetativo",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">❤️ Analisi HRV - Sistema Neurovegetativo</h1>', unsafe_allow_html=True)
    
    init_session_state()
    create_activity_tracker()
    
    st.sidebar.header("👤 Profilo Utente")
    
    with st.sidebar.expander("Modifica Profilo", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Nome", value=st.session_state.user_profile.get('name', ''))
            birth_date = st.date_input("Data di nascita", 
                                     value=st.session_state.user_profile.get('birth_date') or datetime.now().date() - timedelta(days=30*365),
                                     max_value=datetime.now().date())
        with col2:
            surname = st.text_input("Cognome", value=st.session_state.user_profile.get('surname', ''))
            gender = st.selectbox("Sesso", ["Uomo", "Donna"], 
                                index=0 if st.session_state.user_profile.get('gender') == 'Uomo' else 1)
        
        if birth_date:
            age = (datetime.now().date() - birth_date).days // 365
            st.write(f"**Età calcolata:** {age} anni")
        else:
            age = st.session_state.user_profile.get('age', 30)
        
        if st.button("💾 Salva Profilo", use_container_width=True):
            st.session_state.user_profile.update({
                'name': name, 'surname': surname, 'birth_date': birth_date,
                'age': age, 'gender': gender
            })
            save_user_database()
            st.success("Profilo salvato!")
            st.rerun()
    
    st.sidebar.header("📁 Carica Dati")
    
    uploaded_file = st.sidebar.file_uploader("Carica file RR-intervals", type=['csv', 'txt'])
    
    if uploaded_file is not None and not st.session_state.file_uploaded:
        try:
            content = uploaded_file.getvalue().decode('utf-8').strip()
            rr_intervals = []
            
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    try:
                        rr_value = float(line)
                        if 200 <= rr_value <= 2000:
                            rr_intervals.append(rr_value)
                    except ValueError:
                        continue
            
            if rr_intervals:
                st.session_state.rr_intervals = rr_intervals
                st.session_state.file_uploaded = True
                st.sidebar.success(f"✅ File caricato: {len(rr_intervals)} intervalli RR")
            else:
                st.sidebar.error("❌ Nessun intervallo RR valido trovato")
                
        except Exception as e:
            st.sidebar.error(f"❌ Errore nel caricamento: {e}")
    
    if st.session_state.get('file_uploaded') and st.session_state.get('rr_intervals'):
        st.header("📊 Analisi HRV Avanzata")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Impostazioni Analisi")
            
            if not st.session_state.datetime_initialized:
                default_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                st.session_state.analysis_datetimes = {
                    'start_datetime': default_start,
                    'end_datetime': default_start + timedelta(hours=24)
                }
                st.session_state.datetime_initialized = True
            
            col1a, col1b = st.columns(2)
            with col1a:
                start_date = st.date_input("Data inizio analisi", 
                                         value=st.session_state.analysis_datetimes['start_datetime'].date())
                start_time = st.time_input("Ora inizio", 
                                         value=st.session_state.analysis_datetimes['start_datetime'].time())
            with col1b:
                end_date = st.date_input("Data fine analisi", 
                                       value=st.session_state.analysis_datetimes['end_datetime'].date())
                end_time = st.time_input("Ora fine", 
                                       value=st.session_state.analysis_datetimes['end_datetime'].time())
            
            start_datetime = datetime.combine(start_date, start_time)
            end_datetime = datetime.combine(end_date, end_time)
            
            if start_datetime >= end_datetime:
                st.error("⚠️ La data di fine deve essere successiva alla data di inizio")
            else:
                st.session_state.analysis_datetimes = {
                    'start_datetime': start_datetime,
                    'end_datetime': end_datetime
                }
        
        with col2:
            st.subheader("Azioni")
            if st.button("🔄 Analizza Dati", use_container_width=True, type="primary"):
                with st.spinner("Analisi in corso..."):
                    try:
                        rr_intervals = st.session_state.rr_intervals
                        
                        metrics = calculate_realistic_hrv_metrics(
                            rr_intervals, 
                            st.session_state.user_profile.get('age', 30),
                            st.session_state.user_profile.get('gender', 'Uomo')
                        )
                        
                        daily_analyses = analyze_daily_metrics(
                            rr_intervals,
                            start_datetime,
                            st.session_state.user_profile,
                            st.session_state.activities
                        )
                        
                        st.session_state.last_analysis_metrics = metrics
                        st.session_state.last_analysis_start = start_datetime
                        st.session_state.last_analysis_end = end_datetime
                        st.session_state.last_analysis_duration = len(rr_intervals) * np.mean(rr_intervals) / (1000 * 60 * 60)
                        st.session_state.last_analysis_daily = daily_analyses
                        
                        st.success(f"✅ Analisi completata! {len(daily_analyses)} giorni analizzati")
                        
                    except Exception as e:
                        st.error(f"❌ Errore nell'analisi: {e}")
        
        if st.session_state.get('last_analysis_metrics'):
            metrics = st.session_state.last_analysis_metrics
            daily_analyses = st.session_state.last_analysis_daily
            
            st.header("📈 Risultati Analisi")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("SDNN", f"{metrics['sdnn']:.1f} ms", 
                         delta=get_sdnn_evaluation(metrics['sdnn'], st.session_state.user_profile.get('gender', 'Uomo')))
                st.metric("RMSSD", f"{metrics['rmssd']:.1f} ms",
                         delta=get_rmssd_evaluation(metrics['rmssd'], st.session_state.user_profile.get('gender', 'Uomo')))
            
            with col2:
                st.metric("Frequenza Cardiaca Media", f"{metrics['hr_mean']:.1f} bpm",
                         delta=get_hr_evaluation(metrics['hr_mean']))
                st.metric("Coerenza Cardiaca", f"{metrics['coherence']:.1f}%",
                         delta=get_coherence_evaluation(metrics['coherence']))
            
            with col3:
                st.metric("LF/HF Ratio", f"{metrics['lf_hf_ratio']:.2f}",
                         delta=get_lf_hf_evaluation(metrics['lf_hf_ratio']))
                st.metric("Potenza Totale", f"{metrics['total_power']:.0f} ms²",
                         delta=get_power_evaluation(metrics['total_power']))
            
            with col4:
                st.metric("Durata Registrazione", f"{st.session_state.last_analysis_duration:.1f} h")
                st.metric("Metodo Analisi", metrics.get('analysis_method', 'Standard'))
            
            create_daily_analysis_visualization(daily_analyses)
    
    create_pdf_download_button()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Informazioni")
    st.sidebar.info("""
    Questo strumento analizza la Variabilità della Frequenza Cardiaca (HRV) per valutare l'attività del sistema neurovegetativo.
    
    **Funzionalità:**
    • Analisi HRV completa
    • Tracciamento attività
    • Analisi giornaliera
    • Report PDF
    """)

if __name__ == "__main__":
    main()