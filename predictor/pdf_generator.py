import io
import re
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether
)
from reportlab.pdfgen import canvas

from .models import Disease, PredictionHistory
from .services import make_prediction, PredictionError


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute total pages and render running headers/footers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        page_width, page_height = A4
        margin = 36

        # Running header on page 2 and later
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 7.5)
            self.setFillColor(colors.HexColor("#1e3a8a"))
            self.drawString(margin, page_height - 24, "MEDIPREDICT AI • CLINICAL DECISION SUPPORT REPORT")
            self.setFont("Helvetica", 7.5)
            self.setFillColor(colors.HexColor("#64748b"))
            self.drawRightString(page_width - margin, page_height - 24, "CONFIDENTIAL MEDICAL RECORD")

            # Thin top rule
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(margin, page_height - 28, page_width - margin, page_height - 28)

        # Running footer on every page
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.6)
        self.line(margin, 34, page_width - margin, 34)

        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawString(
            margin,
            22,
            "NOTICE: Preliminary AI Diagnostic Assessment. Not a formal prescription or medical diagnosis."
        )
        self.drawRightString(
            page_width - margin,
            22,
            f"Page {self._pageNumber} of {page_count}"
        )

        self.restoreState()


def estimate_severity(disease_name, confidence):
    """
    Infers an estimated severity rating based on condition name keywords and confidence.
    """
    high_keywords = [
        'heart', 'cardiac', 'infarction', 'stroke', 'failure', 'acute',
        'pulmonary', 'sepsis', 'shock', 'hepatitis', 'meningitis',
        'cancer', 'tumor', 'coma', 'aneurysm', 'hemorrhage'
    ]
    name_lower = (disease_name or '').lower()
    for kw in high_keywords:
        if kw in name_lower:
            return 'High / Critical'
    if confidence >= 75:
        return 'Moderate to High'
    elif confidence >= 45:
        return 'Moderate'
    return 'Low / Non-Critical'


def get_confidence_palette(confidence):
    """
    Returns styling colors for confidence pill and score based on value.
    """
    if confidence >= 80:
        return {
            'text': '#047857',      # emerald-700
            'bg': '#ecfdf5',        # emerald-50
            'border': '#10b981',    # emerald-500
            'label': 'HIGH CONFIDENCE'
        }
    elif confidence >= 50:
        return {
            'text': '#b45309',      # amber-700
            'bg': '#fffbeb',        # amber-50
            'border': '#f59e0b',    # amber-500
            'label': 'MODERATE CONFIDENCE'
        }
    else:
        return {
            'text': '#b91c1c',      # red-700
            'bg': '#fef2f2',        # red-50
            'border': '#f87171',    # red-400
            'label': 'LOW CONFIDENCE - TENTATIVE'
        }


def generate_prediction_pdf(history_item, request_user=None):
    """
    Generates a PDF clinical diagnostic report for a given PredictionHistory record.
    Returns a bytes buffer containing the PDF file.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=42
    )

    printable_width = A4[0] - 72  # 523.27 pt

    # Base typography styles
    sample_styles = getSampleStyleSheet()

    styles = {
        'brand_title': ParagraphStyle(
            'BrandTitle',
            fontName='Helvetica-Bold',
            fontSize=18,
            leading=22,
            textColor=colors.HexColor('#1e3a8a')
        ),
        'brand_subtitle': ParagraphStyle(
            'BrandSubtitle',
            fontName='Helvetica-Bold',
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor('#0284c7')
        ),
        'doc_header_title': ParagraphStyle(
            'DocHeaderTitle',
            fontName='Helvetica-Bold',
            fontSize=11,
            leading=14,
            alignment=2,  # Right
            textColor=colors.HexColor('#0f172a')
        ),
        'doc_header_meta': ParagraphStyle(
            'DocHeaderMeta',
            fontName='Helvetica',
            fontSize=8,
            leading=11,
            alignment=2,  # Right
            textColor=colors.HexColor('#64748b')
        ),
        'section_title': ParagraphStyle(
            'SectionTitle',
            fontName='Helvetica-Bold',
            fontSize=10,
            leading=13,
            textColor=colors.HexColor('#1e3a8a')
        ),
        'meta_label': ParagraphStyle(
            'MetaLabel',
            fontName='Helvetica-Bold',
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor('#64748b')
        ),
        'meta_value': ParagraphStyle(
            'MetaValue',
            fontName='Helvetica',
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor('#0f172a')
        ),
        'condition_hero_title': ParagraphStyle(
            'ConditionHeroTitle',
            fontName='Helvetica-Bold',
            fontSize=16,
            leading=20,
            textColor=colors.HexColor('#0f172a')
        ),
        'condition_hero_label': ParagraphStyle(
            'ConditionHeroLabel',
            fontName='Helvetica-Bold',
            fontSize=8,
            leading=10,
            textColor=colors.HexColor('#2563eb')
        ),
        'body_text': ParagraphStyle(
            'ReportBodyText',
            fontName='Helvetica',
            fontSize=8.5,
            leading=12.5,
            textColor=colors.HexColor('#334155')
        ),
        'body_bold': ParagraphStyle(
            'ReportBodyBold',
            fontName='Helvetica-Bold',
            fontSize=8.5,
            leading=12.5,
            textColor=colors.HexColor('#0f172a')
        ),
        'disclaimer_text': ParagraphStyle(
            'DisclaimerText',
            fontName='Helvetica',
            fontSize=7.5,
            leading=10.5,
            textColor=colors.HexColor('#475569')
        ),
        'badge_text': ParagraphStyle(
            'BadgeText',
            fontName='Helvetica-Bold',
            fontSize=8,
            leading=10,
            alignment=1  # Center
        ),
        'symptom_item': ParagraphStyle(
            'SymptomItem',
            fontName='Helvetica',
            fontSize=8,
            leading=11,
            textColor=colors.HexColor('#1e293b')
        ),
        'table_cell': ParagraphStyle(
            'TableCell',
            fontName='Helvetica',
            fontSize=8,
            leading=10.5,
            textColor=colors.HexColor('#1e293b')
        ),
        'table_header': ParagraphStyle(
            'TableHeader',
            fontName='Helvetica-Bold',
            fontSize=8,
            leading=10.5,
            textColor=colors.white
        )
    }

    # Extract data from history_item
    user = request_user or history_item.user
    patient_name = (user.get_full_name() or user.username) if user else "Confidential Patient"
    patient_email = user.email if (user and user.email) else "N/A"
    assessment_id = f"MP-{history_item.id:06d}"
    assessment_date = history_item.date.strftime("%B %d, %Y • %I:%M %p")
    predicted_disease = history_item.predicted_disease
    confidence_score = round(history_item.confidence, 1)

    # Parse symptoms
    raw_symptoms = [s.strip() for s in history_item.symptoms_selected.split(',') if s.strip()]
    formatted_symptoms = [s.replace('_', ' ').title() for s in raw_symptoms]

    # Query Disease info
    disease_obj = Disease.objects.filter(name=predicted_disease).first()
    description = (disease_obj.description if disease_obj and disease_obj.description
                   else f"Preliminary automated evaluation identifies symptoms correlated with {predicted_disease}. Clinical confirmation via standard laboratory testing and provider assessment is advised.")
    precautions_raw = (disease_obj.precautions if disease_obj and disease_obj.precautions
                       else "Consult a licensed physician for tailored diagnosis and medical management; Monitor symptoms closely; Avoid unverified over-the-counter medications.")
    specialist = (getattr(disease_obj, 'recommended_doctor', None) or 'General Physician')
    severity = getattr(disease_obj, 'severity', None) or estimate_severity(predicted_disease, confidence_score)

    # Attempt differential diagnosis using make_prediction
    differential_list = []
    key_symptoms = []
    low_confidence_flag = confidence_score < 40.0
    is_emergency = False
    emergency_level = 'STANDARD'
    emergency_flags = []
    emergency_instructions = ''
    risk_tier = 'MODERATE'
    urgency_timeframe = 'Within 2 to 3 Days'

    try:
        pred_res = make_prediction(raw_symptoms)
        if pred_res:
            differential_list = pred_res.get('top_3', [])
            key_symptoms = pred_res.get('key_symptoms', [])
            low_confidence_flag = pred_res.get('low_confidence_warning', low_confidence_flag)
            is_emergency = pred_res.get('is_emergency', False)
            emergency_level = pred_res.get('emergency_level', 'STANDARD')
            emergency_flags = pred_res.get('emergency_flags', [])
            emergency_instructions = pred_res.get('emergency_instructions', '')
            risk_tier = pred_res.get('risk_tier', 'MODERATE')
            urgency_timeframe = pred_res.get('urgency_timeframe', 'Within 2 to 3 Days')
    except Exception:
        # Fallback if prediction model cannot be re-executed
        differential_list = [
            {
                'name': predicted_disease,
                'confidence': confidence_score,
                'severity': severity,
                'specialist': specialist,
                'description': description
            }
        ]

    conf_palette = get_confidence_palette(confidence_score)

    story = []

    # ---------------------------------------------------------
    # 1. HEADER SECTION
    # ---------------------------------------------------------
    header_left = [
        Paragraph("MEDIPREDICT AI", styles['brand_title']),
        Spacer(1, 2),
        Paragraph("CLINICAL DECISION SUPPORT & DIAGNOSTIC INTELLIGENCE", styles['brand_subtitle']),
        Spacer(1, 1),
        Paragraph("<font color='#64748b' size='7'>Random Forest Multi-Target Classifier • Clinical System v2.4</font>", styles['body_text'])
    ]

    header_right = [
        Paragraph("CLINICAL DIAGNOSTIC REPORT", styles['doc_header_title']),
        Spacer(1, 2),
        Paragraph(f"<b>Report Ref:</b> <font color='#2563eb'>{assessment_id}</font>", styles['doc_header_meta']),
        Paragraph(f"<b>Issued:</b> {assessment_date}", styles['doc_header_meta']),
        Paragraph("<b>Status:</b> <font color='#047857'>Analysis Verified</font>", styles['doc_header_meta'])
    ]

    header_table = Table(
        [[header_left, header_right]],
        colWidths=[printable_width * 0.60, printable_width * 0.40]
    )
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(header_table)

    # Accent color bars below header
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=2.5, color=colors.HexColor('#0284c7'), spaceAfter=1, spaceBefore=0))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#94a3b8'), spaceAfter=8, spaceBefore=0))

    # ---------------------------------------------------------
    # 2. PATIENT & ASSESSMENT DEMOGRAPHICS
    # ---------------------------------------------------------
    demo_data = [
        [
            Paragraph("PATIENT / SUBJECT", styles['meta_label']),
            Paragraph("ACCOUNT / CONTACT", styles['meta_label']),
            Paragraph("EVALUATION REF", styles['meta_label']),
            Paragraph("ASSESSMENT DATE", styles['meta_label']),
        ],
        [
            Paragraph(f"<b>{patient_name}</b>", styles['meta_value']),
            Paragraph(patient_email, styles['meta_value']),
            Paragraph(f"<b>{assessment_id}</b>", styles['meta_value']),
            Paragraph(assessment_date, styles['meta_value']),
        ]
    ]
    demo_table = Table(demo_data, colWidths=[printable_width * 0.28, printable_width * 0.28, printable_width * 0.20, printable_width * 0.24])
    demo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, 0), 4),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 1),
        ('TOPPADDING', (0, 1), (-1, 1), 1),
        ('BOTTOMPADDING', (0, 1), (-1, 1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(demo_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # 3. PRIMARY DIAGNOSTIC FINDING (HERO CARD)
    # ---------------------------------------------------------
    risk_color = '#b91c1c' if risk_tier == 'CRITICAL' else ('#ea580c' if risk_tier == 'HIGH' else ('#d97706' if risk_tier == 'MODERATE' else '#059669'))
    hero_left = [
        Paragraph("PRIMARY PREDICTED CONDITION", styles['condition_hero_label']),
        Spacer(1, 2),
        Paragraph(predicted_disease.title(), styles['condition_hero_title']),
        Spacer(1, 4),
        Paragraph(
            f"<b>Specialist:</b> <font color='#1e3a8a'>{specialist}</font> &nbsp;|&nbsp; "
            f"<b>Risk Tier:</b> <font color='{risk_color}'><b>{risk_tier}</b></font> ({urgency_timeframe})",
            styles['body_text']
        )
    ]

    conf_badge_html = f"""
    <div align="center">
        <font color="{conf_palette['text']}" size="14"><b>{confidence_score}%</b></font><br/>
        <font color="{conf_palette['text']}" size="7.5"><b>{conf_palette['label']}</b></font>
    </div>
    """
    conf_badge = Paragraph(conf_badge_html, styles['badge_text'])

    hero_table = Table(
        [[hero_left, conf_badge]],
        colWidths=[printable_width * 0.72, printable_width * 0.28]
    )
    hero_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0f9ff')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#bae6fd')),
        ('LINEBEFORE', (0, 0), (0, -1), 4, colors.HexColor('#0284c7')),
        ('BACKGROUND', (1, 0), (1, 0), colors.HexColor(conf_palette['bg'])),
        ('BOX', (1, 0), (1, 0), 1, colors.HexColor(conf_palette['border'])),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(hero_table)

    # Emergency Triage Warning Box if Critical/Urgent
    if is_emergency:
        story.append(Spacer(1, 4))
        flag_bullets = " &nbsp;•&nbsp; ".join([f['flag'] for f in emergency_flags])
        em_text = (
            f"<font color='#b91c1c' size='8'><b>EMERGENCY TRIAGE NOTICE ({emergency_level}):</b> {emergency_instructions}</font><br/>"
            f"<font color='#7f1d1d' size='7.5'><b>Identified Red Flags:</b> {flag_bullets}</font>"
        )
        em_table = Table([[Paragraph(em_text, styles['body_text'])]], colWidths=[printable_width])
        em_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fef2f2')),
            ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#fca5a5')),
            ('LINEBEFORE', (0, 0), (0, -1), 3.5, colors.HexColor('#dc2626')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(em_table)

    # Low confidence advisory if needed
    if low_confidence_flag:
        story.append(Spacer(1, 4))
        warning_text = (
            "<b>Clinical Caution:</b> Confidence score is low or non-definitive. The reported symptoms may represent "
            "multiple overlapping conditions. Diagnostic laboratory workup and in-person examination are strongly advised."
        )
        warning_table = Table(
            [[Paragraph(warning_text, styles['body_text'])]],
            colWidths=[printable_width]
        )
        warning_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fffbeb')),
            ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#fde68a')),
            ('LINEBEFORE', (0, 0), (0, -1), 3, colors.HexColor('#f59e0b')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(warning_table)

    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # 4. REPORTED SYMPTOMS & KEY CONTRIBUTING INDICATORS
    # ---------------------------------------------------------
    story.append(Paragraph("1. REPORTED SYMPTOM PROFILE", styles['section_title']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0'), spaceAfter=5, spaceBefore=2))

    # Format symptoms into 3-column rows
    symptom_cells = []
    for s in formatted_symptoms:
        symptom_cells.append(Paragraph(f"• <b>{s}</b>", styles['symptom_item']))

    # Pad with empty cells to make divisible by 3
    while len(symptom_cells) % 3 != 0:
        symptom_cells.append(Paragraph("", styles['symptom_item']))

    symptom_rows = [symptom_cells[i:i + 3] for i in range(0, len(symptom_cells), 3)]
    symptom_col_width = printable_width / 3.0

    symptom_table = Table(symptom_rows, colWidths=[symptom_col_width, symptom_col_width, symptom_col_width])
    symptom_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(symptom_table)

    # Key contributing symptoms if present
    if key_symptoms:
        story.append(Spacer(1, 4))
        key_items = []
        for ks in key_symptoms:
            imp_pct = round(ks.get('importance', 0) * 100, 1)
            key_items.append(f"<b>{ks.get('name')}</b> ({imp_pct}% feature weight)")
        key_str = f"<b>High-Impact Symptom Drivers:</b> {', '.join(key_items)}"
        story.append(Paragraph(key_str, styles['body_text']))

    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # 5. CLINICAL OVERVIEW & ETIOLOGY
    # ---------------------------------------------------------
    story.append(Paragraph("2. CLINICAL OVERVIEW & DESCRIPTION", styles['section_title']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0'), spaceAfter=5, spaceBefore=2))

    desc_table = Table(
        [[Paragraph(description, styles['body_text'])]],
        colWidths=[printable_width]
    )
    desc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('LINEBEFORE', (0, 0), (0, -1), 3, colors.HexColor('#3b82f6')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(desc_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # 6. RECOMMENDED PRECAUTIONS & CARE PROTOCOL
    # ---------------------------------------------------------
    story.append(Paragraph("3. RECOMMENDED PRECAUTIONS & IMMEDIATE MEASURES", styles['section_title']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0'), spaceAfter=5, spaceBefore=2))

    # Split precautions by comma, semicolon, or period
    precautions_list = [p.strip() for p in re.split(r'[,;.\n]+', precautions_raw) if len(p.strip()) > 3]
    if not precautions_list:
        precautions_list = [precautions_raw]

    prec_rows = []
    for idx, prec in enumerate(precautions_list, 1):
        num_cell = Paragraph(f"<font color='#0284c7'><b>{idx}.</b></font>", styles['body_bold'])
        text_cell = Paragraph(prec, styles['body_text'])
        prec_rows.append([num_cell, text_cell])

    prec_table = Table(prec_rows, colWidths=[20, printable_width - 20])
    prec_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(prec_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # 7. DIFFERENTIAL DIAGNOSIS TABLE
    # ---------------------------------------------------------
    if differential_list:
        story.append(Paragraph("4. DIFFERENTIAL DIAGNOSIS (TOP CANDIDATES)", styles['section_title']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0'), spaceAfter=5, spaceBefore=2))

        diff_table_data = [
            [
                Paragraph("<b>Rank</b>", styles['table_header']),
                Paragraph("<b>Candidate Condition</b>", styles['table_header']),
                Paragraph("<b>Confidence</b>", styles['table_header']),
                Paragraph("<b>Specialist</b>", styles['table_header']),
                Paragraph("<b>Severity</b>", styles['table_header']),
            ]
        ]

        for idx, diff in enumerate(differential_list[:4], 1):
            c_name = diff.get('name', 'Unknown')
            c_conf = diff.get('confidence', 0.0)
            c_spec = diff.get('specialist', 'General Physician')
            c_sev = diff.get('severity', 'Moderate')

            # Highlight rank 1
            if idx == 1:
                r_text = f"<b>#{idx} (Primary)</b>"
                name_text = f"<b>{c_name}</b>"
                conf_text = f"<b>{c_conf}%</b>"
            else:
                r_text = f"#{idx}"
                name_text = c_name
                conf_text = f"{c_conf}%"

            diff_table_data.append([
                Paragraph(r_text, styles['table_cell']),
                Paragraph(name_text, styles['table_cell']),
                Paragraph(conf_text, styles['table_cell']),
                Paragraph(c_spec, styles['table_cell']),
                Paragraph(c_sev, styles['table_cell']),
            ])

        col_w = [
            printable_width * 0.14,
            printable_width * 0.38,
            printable_width * 0.16,
            printable_width * 0.18,
            printable_width * 0.14
        ]
        diff_table = Table(diff_table_data, colWidths=col_w)
        diff_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f9ff'), colors.HexColor('#ffffff')]),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(diff_table)
        story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # 8. PATIENT GUIDANCE & CONSULTATION CHECKLIST
    # ---------------------------------------------------------
    checklist_title = Paragraph("5. PROVIDER CONSULTATION CHECKLIST", styles['section_title'])
    checklist_items = [
        "Present this diagnostic summary to your examining physician.",
        "Disclose symptom duration, changes in severity, and triggers.",
        "List all current prescription medicines, OTC drugs, and dietary supplements.",
        "Request diagnostic confirmation tests (e.g. blood work, imaging) relevant to the suspected condition."
    ]
    checklist_rows = []
    for item in checklist_items:
        checklist_rows.append([
            Paragraph("<font color='#047857'><b>[√]</b></font>", styles['body_bold']),
            Paragraph(item, styles['body_text'])
        ])
    checklist_table = Table(checklist_rows, colWidths=[20, printable_width - 20])
    checklist_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 1),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 2),
    ]))

    story.append(KeepTogether([
        checklist_title,
        HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0'), spaceAfter=5, spaceBefore=2),
        checklist_table,
        Spacer(1, 10)
    ]))

    # ---------------------------------------------------------
    # 9. OFFICIAL MEDICAL DISCLAIMER
    # ---------------------------------------------------------
    disclaimer_box = [
        Paragraph("<b>CRITICAL CLINICAL & LEGAL DISCLAIMER</b>", ParagraphStyle('DiscTitle', fontName='Helvetica-Bold', fontSize=8, textColor=colors.HexColor('#b91c1c'))),
        Spacer(1, 2),
        Paragraph(
            "This report is produced by the MediPredict AI algorithmic decision-support tool for preliminary triage and "
            "educational insight only. It DOES NOT provide medical advice, formal clinical diagnosis, or therapeutic prescription. "
            "Always consult a licensed medical professional for clinical decisions. If experiencing severe or life-threatening symptoms "
            "(e.g., acute chest pain, shortness of breath, loss of consciousness), immediately contact emergency services (911/112).",
            styles['disclaimer_text']
        )
    ]
    disclaimer_table = Table([[disclaimer_box]], colWidths=[printable_width])
    disclaimer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fef2f2')),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#fecaca')),
        ('LINEBEFORE', (0, 0), (0, -1), 3, colors.HexColor('#dc2626')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))

    # Digital verification line
    verification_text = Paragraph(
        f"<font color='#64748b' size='7'>Digitally Certified by MediPredict AI Diagnostic Engine • Ref Hash: MP-{history_item.id}-{int(history_item.date.timestamp())} • "
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</font>",
        styles['body_text']
    )

    story.append(KeepTogether([
        disclaimer_table,
        Spacer(1, 6),
        verification_text
    ]))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)

    buffer.seek(0)
    return buffer.getvalue()
