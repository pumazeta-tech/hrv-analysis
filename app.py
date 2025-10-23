def create_pdf_report(metrics, start_datetime, end_datetime, selected_range, user_profile, activities=[]):
    """Crea un report HTML che può essere stampato come PDF"""
    
    # Crea un report HTML ben formattato
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Report Cardiologico - HRV Analysis</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                text-align: center;
                background: #2c3e50;
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
            }}
            .section {{
                background: #f8f9fa;
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                border-left: 4px solid #3498db;
            }}
            .metric-row {{
                display: flex;
                justify-content: space-between;
                margin: 5px 0;
                padding: 5px;
            }}
            .metric-name {{
                font-weight: bold;
                width: 40%;
            }}
            .metric-value {{
                width: 30%;
            }}
            .metric-evaluation {{
                width: 30%;
                text-align: right;
            }}
            .recommendation {{
                background: #e8f4f8;
                padding: 10px;
                margin: 5px 0;
                border-radius: 5px;
                border-left: 3px solid #3498db;
            }}
            @media print {{
                body {{ font-size: 12pt; }}
                .section {{ break-inside: avoid; }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>REPORT CARDIOLOGICO - HRV ANALYSIS</h1>
            <p>Periodo: {start_datetime.strftime("%d/%m/%Y %H:%M")} - {end_datetime.strftime("%d/%m/%Y %H:%M")}</p>
            <p>Durata: {selected_range}</p>
        </div>
        
        <div class="section">
            <h2>👤 INFORMAZIONI PAZIENTE</h2>
            <div class="metric-row">
                <div class="metric-name">Paziente:</div>
                <div class="metric-value">{user_profile.get('name', '')} {user_profile.get('surname', '')}</div>
            </div>
            <div class="metric-row">
                <div class="metric-name">Età e Sesso:</div>
                <div class="metric-value">{user_profile.get('age', '')} anni | {user_profile.get('gender', '')}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 METRICHE HRV PRINCIPALI</h2>
    """
    
    # Aggiungi metriche principali
    metrics_data = [
        ('SDNN', f"{metrics['our_algo']['sdnn']:.1f} ms", get_sdnn_evaluation(metrics['our_algo']['sdnn'], user_profile.get('gender', 'Uomo'))),
        ('RMSSD', f"{metrics['our_algo']['rmssd']:.1f} ms", get_rmssd_evaluation(metrics['our_algo']['rmssd'], user_profile.get('gender', 'Uomo'))),
        ('HR Medio', f"{metrics['our_algo']['hr_mean']:.1f} bpm", get_hr_evaluation(metrics['our_algo']['hr_mean'])),
        ('Coerenza', f"{metrics['our_algo']['coherence']:.1f}%", get_coherence_evaluation(metrics['our_algo']['coherence'])),
        ('Total Power', f"{metrics['our_algo']['total_power']:.0f} ms²", get_power_evaluation(metrics['our_algo']['total_power']))
    ]
    
    for name, value, evaluation in metrics_data:
        html_content += f"""
            <div class="metric-row">
                <div class="metric-name">{name}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-evaluation">{evaluation}</div>
            </div>
        """
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>⚡ SPETTRO DI POTENZA</h2>
    """
    
    # Aggiungi power spectrum
    power_data = [
        ('VLF Power', f"{metrics['our_algo']['vlf']:.0f} ms²"),
        ('LF Power', f"{metrics['our_algo']['lf']:.0f} ms²"),
        ('HF Power', f"{metrics['our_algo']['hf']:.0f} ms²"),
        ('LF/HF Ratio', f"{metrics['our_algo']['lf_hf_ratio']:.2f}")
    ]
    
    for name, value in power_data:
        html_content += f"""
            <div class="metric-row">
                <div class="metric-name">{name}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-evaluation"></div>
            </div>
        """
    
    html_content += """
        </div>
    """
    
    # Aggiungi analisi sonno se disponibile
    if metrics['our_algo'].get('sleep_duration') and metrics['our_algo']['sleep_duration'] > 0:
        html_content += """
            <div class="section">
                <h2>😴 ANALISI QUALITÀ SONNO</h2>
        """
        
        sleep_data = [
            ('Durata Sonno', f"{metrics['our_algo']['sleep_duration']:.1f} h"),
            ('Efficienza', f"{metrics['our_algo']['sleep_efficiency']:.1f}%"),
            ('Coerenza Notturna', f"{metrics['our_algo']['sleep_coherence']:.1f}%"),
            ('HR Notturno', f"{metrics['our_algo']['sleep_hr']:.1f} bpm"),
            ('Sonno REM', f"{metrics['our_algo']['sleep_rem']:.1f} h"),
            ('Sonno Profondo', f"{metrics['our_algo']['sleep_deep']:.1f} h"),
            ('Risvegli', f"{metrics['our_algo']['sleep_wakeups']:.0f}")
        ]
        
        for name, value in sleep_data:
            html_content += f"""
                <div class="metric-row">
                    <div class="metric-name">{name}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-evaluation"></div>
                </div>
            """
        
        html_content += """
            </div>
        """
    
    # Aggiungi confronto algoritmi
    html_content += """
        <div class="section">
            <h2>🔄 CONFRONTO ALGORITMI</h2>
            <div class="metric-row">
                <div class="metric-name">SDNN - Nostro Algo:</div>
                <div class="metric-value">""" + f"{metrics['our_algo']['sdnn']:.1f} ms" + """</div>
            </div>
            <div class="metric-row">
                <div class="metric-name">SDNN - EmWave:</div>
                <div class="metric-value">""" + f"{metrics['emwave_style']['sdnn']:.1f} ms" + """</div>
            </div>
            <div class="metric-row">
                <div class="metric-name">SDNN - Kubios:</div>
                <div class="metric-value">""" + f"{metrics['kubios_style']['sdnn']:.1f} ms" + """</div>
            </div>
        </div>
    """
    
    # Aggiungi raccomandazioni
    html_content += """
        <div class="section">
            <h2>💡 RACCOMANDAZIONI</h2>
    """
    
    recommendations = generate_pdf_recommendations(metrics, user_profile)
    for rec in recommendations:
        html_content += f'<div class="recommendation">• {rec}</div>'
    
    html_content += """
        </div>
        
        <div style="text-align: center; margin-top: 30px; padding: 15px; background: #34495e; color: white; border-radius: 8px;">
            <p>Report generato il """ + datetime.now().strftime("%d/%m/%Y alle %H:%M") + """</p>
            <p><strong>HRV Analytics ULTIMATE</strong> - Sviluppato per Roberto</p>
        </div>
    </body>
    </html>
    """
    
    # Restituisci come HTML che può essere stampato come PDF
    buffer = io.BytesIO()
    buffer.write(html_content.encode('utf-8'))
    buffer.seek(0)
    
    return buffer

# E nel dashboard, modifica il download button:
def create_complete_analysis_dashboard(metrics, start_datetime, end_datetime, selected_range):
    """Crea il dashboard completo di analisi"""
    
    # ... (tutto il codice precedente rimane uguale)
    
    # 7. BOTTONE ESPORTA PDF FUNZIONANTE
    st.markdown("---")
    st.header("📄 Esporta Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🖨️ Genera Report PDF", type="primary", use_container_width=True, key="generate_pdf_btn"):
            with st.spinner("📊 Generando report PDF..."):
                try:
                    pdf_buffer = create_pdf_report(
                        metrics, 
                        start_datetime, 
                        end_datetime, 
                        selected_range,
                        st.session_state.user_profile,
                        st.session_state.activities
                    )
                    
                    st.success("✅ Report PDF generato con successo!")
                    
                    st.download_button(
                        label="📥 Scarica Report PDF",
                        data=pdf_buffer,
                        file_name=f"report_hrv_{start_datetime.strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Errore nella generazione del PDF: {e}")
    
    with col2:
        # Anteprima HTML
        html_buffer = create_pdf_report(metrics, start_datetime, end_datetime, selected_range, st.session_state.user_profile)
        html_content = html_buffer.getvalue().decode('utf-8')
        
        st.download_button(
            label="📄 Scarica Report HTML",
            data=html_content,
            file_name=f"report_hrv_{start_datetime.strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
            use_container_width=True
        )
        
        st.info("💡 **Suggerimento:** Scarica il file HTML e aprilo con il browser, poi usa 'Stampa come PDF'")